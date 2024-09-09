# from transformers import AutoModelForCausalLM, AutoTokenizer
import re

import ollama
def generate_res(prompt):
    response = ollama.chat(model='llama3:70b',messages=[
        {
            'role':'user',
            'content':prompt,
        },
    ])
    print('reasoning:',response['message']['content'])
    return response['message']['content']


def generate_score(prompt):
    response = ollama.chat(model='llama3:70b',messages=[
        {
            'role':'user',
            'content':prompt,
        },
    ])
    res = response['message']['content']
    # print(response['message']['content'])
    numbers = re.findall(r'\d+', res)
    numbers = [int(num) for num in numbers]
    filter_number = [num for num in numbers if 0< num <100]
    if not filter_number:
        return 0
    number = filter_number[-1]
    print('分数：',number)
    return number
