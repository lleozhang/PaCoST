from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os, random, json
import numpy as np
from argparse import ArgumentParser
import re



PARAPHRASE_MODEL = 'meta-llama/Llama-2-7b-chat-hf' #PATH to the model used to paraphrase

B_INST, E_INST = "[INST]", "[/INST]"
device1, device2 = 'cuda:0', 'cuda:1' #inference devices to be used
B_HEADER, E_HEADER = '<|start_header_id|>', '<|end_header_id|>'

para_prompt = f'You are provided with a question. \n \
        Your task is to rephrase this question into another question with the same meaning.\
            When rephrasing the question, you must ensure that you follow the following rules: \n \
            (1). You must ensure that you generate a rephrased question as your response. \
            (2). You must ensure that the rephrased question bears the same meaning with the original question. Do not miss any information. \
            (3). You must only generate a rephrased question. Any other information should not appear in your response. \
            (4). Do not output any explanation. \
            (5). Do not modify the numbers or quantities in the question. You should remain them unchanged. \
            Example 1: given question "1+1=?", one possible response should be: What is the value of 1+1? \n \
            Example 2: given question "What does Earth orbit around?", one possible response should be: What is the center of earth\'s track? \n \
            Example 3: given question "What is the third letter in English?", one possible response should be: Which letter is the third letter in English? \n \
            Example 4: given question "A scientist is looking for something to watch faraway stars. What is he looking for?", one possible response should be: A scientist would like use something to watch remote stars. What can be used? \n \
            Example 5: given question "John has 8 cats and five dogs. Linda has 6 rabbits. How many animals does John have in total?", one possible response should be: John owns 8 cats and five dogs. Linda possesses 6 rabbits. What is the number of animals John has in total? \n '

confidence_prompt = f'You are an expert in judging whether the answer is correct. \
        You will be given a question and a corresponding answer. \
        Your job is to determine whether this answer is correct. \
        You should only respond with Yes or No. \
        For example, given question "What is the value of 1+1? A. 1 B. 2 C. 3 D. 4" and answer "B", the correct response should be "Yes".'

def set_seed(seed = 3407):
    '''
        Set a given random seed.
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True 

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--seed', type = int, default = 42)
    parser.add_argument('--model', type = str, default = 'meta-llama/Llama-2-7b-chat-hf', help = 'path to the model to be tested')
    parser.add_argument('--yes_id', type = int, default = 3869, help = 'id of the token "yes" in the model, used for calculating confidence')
    args = parser.parse_args()
    return args

def load_data(path: str):
    from datasets import load_dataset
    dataset = load_dataset('parquet', data_files = {'test' : path})
    return dataset['test']

def load_mmlu():
    dic = {
        0 : 'A',
        1 : 'B',
        2 : 'C',
        3 : 'D',
        4 : 'E',
        5 : 'F',
        6 : 'G',
        7 : 'H',
        8 : 'I',
        9 : 'J',
        10 : 'K',
        11 : 'L',
        12 : 'M',
        13 : 'N'
    }
    mmlu = load_data('cais/mmlu')
    mmlu_reformat = []
    for d in mmlu:
        choices = ''
        for i, c in enumerate(d['choices']):
            choices += f'{dic[i]}. {c} '
            
        new_d = {
            'instruction' : d['question'] + ' ' + choices,
            'answer' : dic[d['answer']],
            'system' : 'Please select the correct choice according to the question. You should only respond with one choice such as "A" "B" "C" "D", etc. '
        }
        mmlu_reformat.append(new_d)
    return mmlu_reformat

@torch.no_grad()
def chat_with_lm(model, tokenizer, input, device = 'cuda'):
    model.eval()
    model = model.to(torch.bfloat16)
    model = model.to(device)
    
    prompt = tokenizer.encode(input, return_tensors = 'pt', add_special_tokens = False)
    output = model.generate(input_ids = prompt.to(device),
                            max_new_tokens = 1024,
                            do_sample = True)
    return tokenizer.decode(output[0, prompt.shape[1]:].detach().cpu()).replace('</s>', '').replace('<|eot_id|>', '')
    
@torch.no_grad()
def chat_with_lm_logits(model, tokenizer, input, device = 'cuda'):
    model.eval()
    model = model.to(torch.bfloat16)
    model = model.to(device)
    
    prompt = tokenizer.encode(input, return_tensors = 'pt', add_special_tokens = False)
    output = model.generate(input_ids = prompt.to(device),
                            max_new_tokens = 1024,
                            do_sample = False, 
                            output_scores = True,
                            output_hidden_states = True,
                            return_dict_in_generate = True)
    return output

@torch.no_grad()
def calculate_confidence_with_logits(model, tokenizer, question, answer, yes_id = 3876, device = 'cuda'):
    prompt = f'<s> {B_INST} <<SYS>> {confidence_prompt} <</SYS>> \
        The question is {question}. \
        The answer is {answer}. \
        Is the answer correct according to the question? {E_INST}' 
    #This prompt should be modified according to the model to be tested.
        
    ret = chat_with_lm_logits(model, tokenizer, prompt, device)
    
    re_confi = 0
    sc = ret['scores']
    for i in range(len(sc)):
        confi = torch.softmax(sc[i][0].detach().cpu(), dim = 0)[yes_id].item()
        a_max = torch.argmax(sc[i][0].detach().cpu()).item()
        
        
        dec = tokenizer.decode([a_max])
        if 'yes' in dec.lower() or 'no' in dec.lower(): # If yes or no is the output.
            re_confi = max(re_confi, confi)

    return re_confi

@torch.no_grad()
def paraphrase(question, model, tokenizer, device = 'cuda'):
    prompt = para_prompt
    para = question
    instruction = f'<s> {B_INST} {prompt}  \
            The question is: {para} {E_INST}\
            The rephrased question is: '
    #This prompt format should be modified according to the model used to paraphrase
    rep = chat_with_lm(model, tokenizer, instruction, device)
    return rep
    
    
if __name__ == '__main__':
    args = parse_args()
    model = args.model
    yes_id = args.yes_id
    seed = args.seed
    set_seed(seed)
    
    test_model = AutoModelForCausalLM.from_pretrained(model)
    test_tokenizer = AutoTokenizer.from_pretrained(model)
    
    para_model = AutoModelForCausalLM.from_pretrained(PARAPHRASE_MODEL)
    para_tokenizer = AutoTokenizer.from_pretrained(PARAPHRASE_MODEL)
    
    from tqdm import tqdm
    hal_confi = []
    par_confi = []
    answers = {}
    
    data = load_mmlu() #This is an example using mmlu for detection. You can replace it with your own data. The format should be modified correspondingly.
    
    for id, d in enumerate(tqdm(data)):
        ori_question = d['instruction']
        
        question_format = f'<s> {B_INST} <<SYS>> {d["system"]} <</SYS>> {ori_question} {E_INST}' 
        #This prompt format should be modified according to the model to be tested.
        
        answer = chat_with_lm(test_model, test_tokenizer, question_format, device1)
        ret = calculate_confidence_with_logits(test_model, test_tokenizer, f'{ori_question}', answer, yes_id, device1)

        question_no_choices = ori_question.split('A.')[0].strip()
        choices = ori_question[len(question_no_choices) : ]
        
        para_question = paraphrase(question_no_choices, para_model, para_tokenizer, device2)
        if para_question is None:
            continue
        else:
            para_question = para_question.strip()
        
        para_question = para_question.replace('\n', ' ')
        para_question = para_question.replace('</s>', '').replace('  ', ' ')
        para_question = re.sub(r'\s+', ' ', para_question).strip()

        para_format = f'<s> {B_INST} <<SYS>> {d["system"]} <</SYS>> {para_question + " " + choices} {E_INST}'
        #This prompt format should be modified according to the model to be tested.
        
        para_answer = chat_with_lm(test_model, test_tokenizer, para_format, device1)
        rret = calculate_confidence_with_logits(test_model, test_tokenizer, para_question + ' ' + choices, para_answer, yes_id, device1)
        
        if ret > 0 and rret > 0:
            hal_confi.append(ret)
            par_confi.append(rret)
        
        answers[id] = {'original_question': ori_question, 
                        'original_answer' : answer, 
                        'paraphrased_question' : para_question, 
                        'paraphrased_answer' : para_answer, 
                        'correct_answer' : d['answer']}
    
    os.makedirs(f'{model}_results', exist_ok = True)
    
    torch.save(torch.tensor(hal_confi), f'{model}_results/ori_mmlu_{seed}.pkl')
    torch.save(torch.tensor(par_confi), f'{model}_results/par_mmlu_{seed}.pkl')
    
    with open(f'{model}_results/mmlu_{seed}.json', 'w') as f:
        json.dump(answers, f, indent = 4)