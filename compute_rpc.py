import json
import metrics
import argparse
import numpy as np
import multiprocessing
from tqdm import trange
import signal, functools
from scipy.special import gamma
import re, os, sys, random, time
from scipy.stats import weibull_min
from scipy.optimize import minimize
from fraction import Fraction
from data_processing.answer_extraction import *
from eval.eval_script import *
from compute_perp import Evaluator, numberic_compare
from compute_sc import DSU
MAX_INT = sys.maxsize
INVALID_ANS = "[Invalid]"

#### Reasoning Pruning Module: Model probability with Weibull distribution ####

def weibull_pdf(x, k, lam):
    return (k / lam) * (x / lam) ** (k - 1) * np.exp(-((x / lam) ** k))

def weibull_mean(k, lam):
    return lam * gamma(1 + 1 / k)

def mixture_pdf(x, w1, k1, lam1, k2, lam2):
    return w1 * weibull_pdf(x, k1, lam1) + (1 - w1) * weibull_pdf(x, k2, lam2)

PDF_EPS = 1e-300  # avoid log(0) in neg_log_likelihood (Zero Init can yield ~0 pdf)


def neg_log_likelihood(params, data):
    w1, k1, lam1, k2, lam2 = params
    pdf_vals = mixture_pdf(data, w1, k1, lam1, k2, lam2)
    pdf_vals = np.clip(pdf_vals, PDF_EPS, None)
    return -np.sum(np.log(pdf_vals))

def calculate_membership_probabilities(data, w1, k1, lam1, k2, lam2):
    pdf1 = weibull_pdf(data, k1, lam1)
    pdf2 = weibull_pdf(data, k2, lam2)
    prob1 = w1 * pdf1 / (w1 * pdf1 + (1 - w1) * pdf2)
    prob2 = 1 - prob1
    return prob1, prob2

### Perplexity Consistency Module: Bridging the probability with self-consistency ####

def _get_initial_guess(init_method):
    """Fixed Init: [0.5, 1.0, 1.0, 1.5, 2.0]. Zero Init: small k, lam."""
    if init_method == "fixed":
        return [0.5, 1.0, 1.0, 1.5, 2.0]
    elif init_method == "zero":
        return [0.5, 0.01, 0.01, 0.01, 0.01]
    else:
        return [0.5, 1.0, 1.0, 1.5, 2.0]


def make_wpc_evaluator(init_method="fixed", w_lower=0.2, w_upper=0.8):
    """Return an evaluator that uses the given init and w bounds (for Table 9 / D.6)."""

    def wpc_evaluator(predicts, completions, perplexities, answer, equal_func, check_equal):
        m = len(predicts)
        dsu = DSU(m)
        probas = [np.exp(perplexities[i]) for i in range(m)]
        mean_proba = np.mean(probas)

        initial_guess = _get_initial_guess(init_method)
        result = minimize(
            neg_log_likelihood,
            initial_guess,
            args=(probas,),
            bounds=[(w_lower, w_upper), (0.01, None), (0.01, None), (0.01, None), (0.01, None)],
        )
        return _wpc_evaluator_core(predicts, completions, perplexities, answer, equal_func, check_equal, dsu, probas, mean_proba, result)

    return wpc_evaluator


# Default evaluator (Fixed Init, w in [0.2, 0.8]) for backward compatibility
wpc_evaluator = make_wpc_evaluator("fixed", 0.2, 0.8)


def _wpc_evaluator_core(predicts, completions, perplexities, answer, equal_func, check_equal, dsu, probas, mean_proba, result):
    """Shared core after Weibull fit (used by both wpc_evaluator and make_wpc_evaluator)."""
    m = len(predicts)
    w1, k1, lam1, k2, lam2 = result.x
    if weibull_mean(k1, lam1) < weibull_mean(k2, lam2):
        k1, lam1, k2, lam2 = k2, lam2, k1, lam1
        w1 = 1 - w1

    # Pruning reasoning paths with low probabilities
    remove = 0
    for i in range(m):
        completion_i = completions[i]
        logprob_i = perplexities[i]
        proba_i = np.exp(logprob_i)
        p1, p2 = calculate_membership_probabilities(proba_i, w1, k1, lam1, k2, lam2)
        if p1 < p2 and proba_i < mean_proba:
            proba_i = 0
            remove += 1
        else:
            dsu.attr[i][completion_i] = set([proba_i])
    
    # Combining internal probabilities and self-consistency
    for i in range(m):
        if dsu.get_father(i) != i:
            continue
        for j in range(i):
            ans_i = predicts[i]
            ans_j = predicts[j]
            completion_i = completions[i]
            completion_j = completions[j]
            if equal_func(ans_i, ans_j, completion_i, completion_j):
                dsu.merge(i, j)

    # Compute majority votes with probabilities
    max_prob, max_prob_count = 0, 0
    for i in range(m):
        if dsu.get_father(i) != i:
            continue
        prob_i = np.sum([np.sum(list(dsu.attr[i][k])) for k in dsu.attr[i].keys()])
        if prob_i > max_prob:
            max_prob = prob_i
            max_prob_count = 0
        if prob_i >= max_prob:
            max_prob_count += 1

    # Compute accuracy
    correct, answers = 0, []
    for i in range(m):
        if dsu.get_father(i) != i:
            continue
        ans_i = predicts[i]
        prob_i = np.sum([np.sum(list(dsu.attr[i][k])) for k in dsu.attr[i].keys()])
        answers.append([ans_i, prob_i, check_equal(ans_i, answer)])
        if prob_i < max_prob:
            continue
        if check_equal(ans_i, answer):
            correct += 1.0 / max_prob_count

    # Normalize probabilities
    sum_proba = np.sum([x[1] for x in answers])
    for i in range(len(answers)):
        answers[i][1] /= sum_proba

    return correct, answers

class RPCEvaluator(Evaluator):
    def __init__(self, init_method="fixed", w_bounds=(0.2, 0.8)):
        self.name = "RPC"
        self.init_method = init_method
        self.w_bounds = w_bounds
        # Do not store closure on self: multiprocessing cannot pickle it.
        # Create evaluator inside worker() instead.

    def worker(self, args):
        json_file, cache_file, K, seed = args
        evaluator_fn = make_wpc_evaluator(
            self.init_method, self.w_bounds[0], self.w_bounds[1]
        )
        acc, maximum, average, max_bins, avg_bins = self.process(
            json_file=json_file,
            cache_file=cache_file,
            equal_func=numberic_compare,
            evaluator=evaluator_fn,
            K=K,
            seed=seed
        )
        return acc, maximum, average, max_bins, avg_bins

