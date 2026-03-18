# for model in InternLM2-Math-Plus-7B Deepseek-Math-RL-7B InternLM2-Math-Plus-1.8B; do
#     for method in PPL SC RPC; do
#         python main.py --dataset MATH --model $model --method $method --K 64
#     done
#     for dataset in MathOdyssey AIME OlympiadBench; do
#         for method in PPL SC RPC; do
#             python main.py --dataset $dataset --model $model --method $method --K 128
#         done
#     done
# done
# python main.py --dataset MATH --model InternLM2-Math-Plus-7B --method PPL --K 1
# python main.py --dataset MATH --model InternLM2-Math-Plus-7B --method PPL --K 2
# python main.py --dataset MATH --model InternLM2-Math-Plus-7B --method PPL --K 4
# python main.py --dataset MATH --model InternLM2-Math-Plus-7B --method PPL --K 8
# python main.py --dataset MATH --model InternLM2-Math-Plus-7B --method PPL --K 16
# python main.py --dataset MATH --model InternLM2-Math-Plus-7B --method PPL --K 32
# python main.py --dataset MATH --model InternLM2-Math-Plus-7B --method PPL --K 40
# python main.py --dataset MATH --model InternLM2-Math-Plus-7B --method PPL --K 48
# python main.py --dataset MATH --model InternLM2-Math-Plus-7B --method PPL --K 64
# python main.py --dataset MATH --model InternLM2-Math-Plus-7B --method RPC --K 1
# python main.py --dataset MATH --model InternLM2-Math-Plus-7B --method RPC --K 2
# python main.py --dataset MATH --model InternLM2-Math-Plus-7B --method RPC --K 4
# python main.py --dataset MATH --model InternLM2-Math-Plus-7B --method RPC --K 8
# python main.py --dataset MATH --model InternLM2-Math-Plus-7B --method RPC --K 16
# python main.py --dataset MATH --model InternLM2-Math-Plus-7B --method RPC --K 32
# python main.py --dataset MATH --model InternLM2-Math-Plus-7B --method RPC --K 40
# python main.py --dataset MATH --model InternLM2-Math-Plus-7B --method RPC --K 48
# python main.py --dataset MATH --model InternLM2-Math-Plus-7B --method RPC --K 64
# python main.py --dataset MATH --model InternLM2-Math-Plus-7B --method SC --K 1
# python main.py --dataset MATH --model InternLM2-Math-Plus-7B --method SC --K 2
# python main.py --dataset MATH --model InternLM2-Msath-Plus-7B --method SC --K 4
# python main.py --dataset MATH --model InternLM2-Math-Plus-7B --method SC --K 8
# python main.py --dataset MATH --model InternLM2-Math-Plus-7B --method SC --K 16
# python main.py --dataset MATH --model InternLM2-Math-Plus-7B --method SC --K 32
# python main.py --dataset MATH --model InternLM2-Math-Plus-7B --method SC --K 40
# python main.py --dataset MATH --model InternLM2-Math-Plus-7B --method SC --K 48
# python main.py --dataset MATH --model InternLM2-Math-Plus-7B --method SC --K 64


# python main.py --dataset OlympiadBench --model InternLM2-Math-Plus-7B --method PPL --K 1
# python main.py --dataset OlympiadBench --model InternLM2-Math-Plus-7B --method PPL --K 2
# python main.py --dataset OlympiadBench --model InternLM2-Math-Plus-7B --method PPL --K 4
# python main.py --dataset OlympiadBench --model InternLM2-Math-Plus-7B --method PPL --K 8
# python main.py --dataset OlympiadBench --model InternLM2-Math-Plus-7B --method PPL --K 16
# python main.py --dataset OlympiadBench --model InternLM2-Math-Plus-7B --method PPL --K 32
# python main.py --dataset OlympiadBench --model InternLM2-Math-Plus-7B --method PPL --K 40
# python main.py --dataset OlympiadBench --model InternLM2-Math-Plus-7B --method PPL --K 48
# python main.py --dataset OlympiadBench --model InternLM2-Math-Plus-7B --method PPL --K 64
# python main.py --dataset OlympiadBench --model InternLM2-Math-Plus-7B --method PPL --K 80
# python main.py --dataset OlympiadBench --model InternLM2-Math-Plus-7B --method PPL --K 96
# python main.py --dataset OlympiadBench --model InternLM2-Math-Plus-7B --method PPL --K 112
# python main.py --dataset OlympiadBench --model InternLM2-Math-Plus-7B --method PPL --K 128

# python main.py --dataset OlympiadBench --model InternLM2-Math-Plus-7B --method SC --K 1
# python main.py --dataset OlympiadBench --model InternLM2-Math-Plus-7B --method SC --K 2
# python main.py --dataset OlympiadBench --model InternLM2-Math-Plus-7B --method SC --K 4
# python main.py --dataset OlympiadBench --model InternLM2-Math-Plus-7B --method SC --K 8
# python main.py --dataset OlympiadBench --model InternLM2-Math-Plus-7B --method SC --K 16
# python main.py --dataset OlympiadBench --model InternLM2-Math-Plus-7B --method SC --K 32
# python main.py --dataset OlympiadBench --model InternLM2-Math-Plus-7B --method SC --K 40
# python main.py --dataset OlympiadBench --model InternLM2-Math-Plus-7B --method SC --K 48
# python main.py --dataset OlympiadBench --model InternLM2-Math-Plus-7B --method SC --K 64
# python main.py --dataset OlympiadBench --model InternLM2-Math-Plus-7B --method SC --K 80
# python main.py --dataset OlympiadBench --model InternLM2-Math-Plus-7B --method SC --K 96
# python main.py --dataset OlympiadBench --model InternLM2-Math-Plus-7B --method SC --K 112
# python main.py --dataset OlympiadBench --model InternLM2-Math-Plus-7B --method SC --K 128

# python main.py --dataset OlympiadBench --model InternLM2-Math-Plus-7B --method RPC --K 1
# python main.py --dataset OlympiadBench --model InternLM2-Math-Plus-7B --method RPC --K 2
# python main.py --dataset OlympiadBench --model InternLM2-Math-Plus-7B --method RPC --K 4
# python main.py --dataset OlympiadBench --model InternLM2-Math-Plus-7B --method RPC --K 8
# python main.py --dataset OlympiadBench --model InternLM2-Math-Plus-7B --method RPC --K 16
# python main.py --dataset OlympiadBench --model InternLM2-Math-Plus-7B --method RPC --K 32
# python main.py --dataset OlympiadBench --model InternLM2-Math-Plus-7B --method RPC --K 40
# python main.py --dataset OlympiadBench --model InternLM2-Math-Plus-7B --method RPC --K 48
# python main.py --dataset OlympiadBench --model InternLM2-Math-Plus-7B --method RPC --K 64
# python main.py --dataset OlympiadBench --model InternLM2-Math-Plus-7B --method RPC --K 80
# python main.py --dataset OlympiadBench --model InternLM2-Math-Plus-7B --method RPC --K 96
# python main.py --dataset OlympiadBench --model InternLM2-Math-Plus-7B --method RPC --K 112
# python main.py --dataset OlympiadBench --model InternLM2-Math-Plus-7B --method RPC --K 128

# python main.py --dataset AIME --model InternLM2-Math-Plus-7B --method PPL --K 1
# python main.py --dataset AIME --model InternLM2-Math-Plus-7B --method PPL --K 2
# python main.py --dataset AIME --model InternLM2-Math-Plus-7B --method PPL --K 4
# python main.py --dataset AIME --model InternLM2-Math-Plus-7B --method PPL --K 8
# python main.py --dataset AIME --model InternLM2-Math-Plus-7B --method PPL --K 16
# python main.py --dataset AIME --model InternLM2-Math-Plus-7B --method PPL --K 32
# python main.py --dataset AIME --model InternLM2-Math-Plus-7B --method PPL --K 40
# python main.py --dataset AIME --model InternLM2-Math-Plus-7B --method PPL --K 48
# python main.py --dataset AIME --model InternLM2-Math-Plus-7B --method PPL --K 64
# python main.py --dataset AIME --model InternLM2-Math-Plus-7B --method PPL --K 80
# python main.py --dataset AIME --model InternLM2-Math-Plus-7B --method PPL --K 96
# python main.py --dataset AIME --model InternLM2-Math-Plus-7B --method PPL --K 112
# python main.py --dataset AIME --model InternLM2-Math-Plus-7B --method PPL --K 128

# python main.py --dataset AIME --model InternLM2-Math-Plus-7B --method RPC --K 1
# python main.py --dataset AIME --model InternLM2-Math-Plus-7B --method RPC --K 2
# python main.py --dataset AIME --model InternLM2-Math-Plus-7B --method RPC --K 4
# python main.py --dataset AIME --model InternLM2-Math-Plus-7B --method RPC --K 8
# python main.py --dataset AIME --model InternLM2-Math-Plus-7B --method RPC --K 16
# python main.py --dataset AIME --model InternLM2-Math-Plus-7B --method RPC --K 32
# python main.py --dataset AIME --model InternLM2-Math-Plus-7B --method RPC --K 40
# python main.py --dataset AIME --model InternLM2-Math-Plus-7B --method RPC --K 48
# python main.py --dataset AIME --model InternLM2-Math-Plus-7B --method RPC --K 64
# python main.py --dataset AIME --model InternLM2-Math-Plus-7B --method RPC --K 80
# python main.py --dataset AIME --model InternLM2-Math-Plus-7B --method RPC --K 96
# python main.py --dataset AIME --model InternLM2-Math-Plus-7B --method RPC --K 112
# python main.py --dataset AIME --model InternLM2-Math-Plus-7B --method RPC --K 128


python main.py --dataset AIME --model InternLM2-Math-Plus-7B --method SC --K 80
python main.py --dataset AIME --model InternLM2-Math-Plus-7B --method SC --K 96
python main.py --dataset AIME --model InternLM2-Math-Plus-7B --method SC --K 112
python main.py --dataset AIME --model InternLM2-Math-Plus-7B --method SC --K 128