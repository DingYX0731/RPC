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

### Perplexity Consistency Module ###
def pc_evaluator(predicts, completions, perplexities, answer, equal_func, check_equal):
    m = len(predicts)
    probas = [np.exp(perplexities[i]) for i in range(m)]
    
    answer_probs = {}
    for i in range(m):
        ans_i = predicts[i]
        proba_i = probas[i]
        
        found = False
        for existing_ans in answer_probs:
            if equal_func(ans_i, existing_ans, completions[i], ""):
                answer_probs[existing_ans] += proba_i
                found = True
                break
        if not found:
            answer_probs[ans_i] = proba_i
    
    if not answer_probs:
        return 0.0, []
    
    max_prob = max(answer_probs.values())
    max_prob_answers = [ans for ans, prob in answer_probs.items() if prob == max_prob]
    max_prob_count = len(max_prob_answers)
    
    correct = 0.0
    answers = []
    sum_proba = sum(answer_probs.values())
    for ans, prob in answer_probs.items():
        norm_prob = prob / sum_proba if sum_proba != 0 else 0.0
        is_correct = check_equal(ans, answer)
        answers.append([ans, norm_prob, is_correct])
        if prob == max_prob and is_correct:
            correct += 1.0 / max_prob_count
    
    return correct, answers

class PCEvaluator(Evaluator):
    def __init__(self,):
        self.name = "PC"

    def worker(self, args):
        json_file, cache_file, K, seed = args
        acc, maximum, average, max_bins, avg_bins = self.process(
            json_file=json_file,
            cache_file=cache_file,
            equal_func=numberic_compare,
            evaluator=pc_evaluator,
            K=K,
            seed=seed
        )
        return acc, maximum, average, max_bins, avg_bins

