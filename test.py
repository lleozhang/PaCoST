import torch
from scipy import stats
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--seed', type = int, default = 42)
    parser.add_argument('--model', type = str, default = 'llama_2_chat_7b', help = 'path to the model to be tested')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    model = args.model
    seed = args.seed
    
    
    ori_confi = torch.load(f'{model}_results/ori_mmlu_{seed}.pkl')
    par_confi = torch.load(f'{model}_results/par_mmlu_{seed}.pkl')

    r = stats.ttest_rel(ori_confi.numpy()[:], par_confi.numpy()[:], alternative = 'greater')
    print(r)
