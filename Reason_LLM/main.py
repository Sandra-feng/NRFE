import json
import os.path
from utils import generate_res,generate_score
from tqdm import tqdm
import pandas as pd
import pickle
import re

def extract_quoted_text(text):
    """
    从文本中移除多个特定的句子。
    
    参数:
    text (str): 要处理的文本
    sentences_to_remove (list): 要移除的句子列表
    
    返回:
    str: 移除特定句子后的文本
    """
        # 要移除的句子
    # print('处理前：',text)
    if text is None:
        text = 'null'
    sentences_to_remove = [
        # "Here's a positive reasoning that can increase the credibility score of the real news:",
        # "Here's another negative reasoning that can decrease the credibility score of the real news:",
        "positive reasoning",
        "negative reasoning",
        "real news",
        "fake news",
        "fake News",
        "real News",
        "credibility score",
        "increase",
        "decrease"
    ]
    sentences_to_remove2 = [
        "Here's another that can the of the :",
        "Here's a that can the of the :",
        "Here is a that can the of the :",
        "Here's another attempt at generating a that can the of the :",
        "Here's a new that can the of the :",
        "Here's a revised that can the of the :",
        "Here's a rewritten that can the of the :",
        "Here's a possible that can the of the :"
    ]
    for sentence in sentences_to_remove:
        pattern = re.escape(sentence)+r'\s*'
        text = re.sub(pattern,'',text)
        # text = sentence.replace(sentence, "")
    for sentence2 in sentences_to_remove2:
        pattern = re.escape(sentence2)+r'\s*'
        print(text)
        text = re.sub(pattern,'',text)
    pa = r'"([^"]+)"'
    matches  = re.findall(pa,text)
    text = text[:300]
    quoted_text = ''.join(matches)
    print('处理后：',quoted_text)
    return quoted_text



def run_reason():
    prompt = 'Goal: rate the no reasoning-based credibility score of the {} News:{}. \n'
    prompt += 'Requirement 1: Give a concrete score value of type int from 0 to 100. \n'
    prompt += 'Requirement 2: Output format[int]'

    prompt12 = 'Goal: Generate an {} reasoning that can {} the credibility score of the {} News:{}\n.'
    prompt12 += 'Requirement 1: the no reasoning-based credibility score of the original news is:{}.\n'
    prompt12 += 'Requirement 2: The length of the reasoning is limit to 200 words.'

    prompt34 = 'Goal: rate the credibility of the {} reasoning-based {} News:{}.\n'
    prompt34 += 'Requirement 1: consider the no reasoning-based credibility score of the news {}.\n'
    prompt34 += 'Requirement 2: consider the negative reasoning {}.\n'
    prompt34 += 'Requirement 3: Output format[int].'

    prompt5 = "Goal: This {} reasoning can not {} the credibility of the news. "
    prompt5 += " Please re-generate another {} reasoning that can {} the credibility score of the {} news: {}.\n"
    prompt5 += 'Requirement 1: consider the no reasoning-based credibility score of the news {}.\n'
    prompt5 += 'Requirement 2: consider the {} reasoning {}.\n'
    prompt5 += 'Requirement 3:consider the {} reasoning-based credibility score of news: {}\n'
    prompt5 += 'Requirement 4: The length of the reasoning is limit to 200 words.'
    input_path = 'datasets/arg_gossipcop/test.json'
    # unshuff_data = pd.read_json(f'datasets/arg_gossipcop/train.json')
    # data = unshuff_data.sample(frac=1, random_state=42)
    save_path = '/mnt/C00C86F42C263BF6/fzl/DELL-main_707/datasets/arg_gossipcop/goosipcop_test724.json'
    try:
        with open(save_path,'r') as outfile:
            processed_lines = sum(1 for line in outfile)
    except FileNotFoundError:
        processed_lines = 0
    print(processed_lines)
    with open(input_path,'r') as infile, open(save_path,'a') as outfile:
        data = json.load(infile)
        for i,row in tqdm(enumerate(data),total=len(data)):
            if i < processed_lines:
                # print(i)
                continue;
            input_text = row['content']
            # print(input_text)
            label = row['label']
            # print(label)
            label = int(label)
            # print("新闻标签为",label)

            str1 = 'fake'
            str2 = 'real'
            str3 = 'positive'
            str4 = 'negative'
            str5 = 'increase'
            str6 = 'decrease'
            if label == 0:# the label 0 represents fake news假新闻
                original_score = prompt.format(str1, input_text)
                original_score = generate_score(original_score)

                #正向推理 降低
                positive_reason = prompt12.format(str3, str6,str1, input_text, original_score)
                positive_reason = generate_res(positive_reason)
                q1 = prompt34.format(str3,str1, input_text,original_score, positive_reason)
                pos_score = generate_score(q1)
                gap_pos = pos_score-original_score
                count=0
                while pos_score >= 50 and gap_pos > 0:  #当新闻置信度到50以下且生成的推理效果比单纯的新闻更能证实为假新闻时跳出循环
                    print("*************正向推理****************")
                    question_temp = prompt5.format(str3,str6,str3,str6,str1,input_text,original_score,str3,positive_reason,str3,pos_score)
                    positive_reason = generate_res(question_temp)
                    q1 = prompt34.format(str3,str1, input_text,original_score, positive_reason)
                    pos_score = generate_score(q1)
                    gap_pos = pos_score-original_score
                    count = count + 1
                    if count > 3:
                        # positive_reason = None
                        break
                positive_reason = extract_quoted_text(positive_reason)
                # data.at[index, 'forward_reason2'] = positive_reason
                row['forward_reason2'] = positive_reason
                
                #反向推理 reversed reasoning升
                negative_reason = prompt12.format(str4, str5,str1, input_text, original_score)
                negative_reason = generate_res(negative_reason)
                q_neg = prompt34.format(str4,str1, input_text,original_score, negative_reason)
                neg_score = generate_score(q_neg)
                gap_neg = original_score-neg_score
                count=0
                while gap_neg>0: #反向推理为了证实新闻为真新闻，当新闻置信度到50以上且生成的推理比单纯的新闻更能证实为真新闻时跳出循环
                    print("*************负向推理****************")
                    question_temp = prompt5.format(str4,str5,str4,str5,str1,input_text,original_score,str4,negative_reason,str4,neg_score)
                    negative_reason = generate_res(question_temp)
                    q2 = prompt34.format(str4,str1, input_text,original_score, negative_reason)
                    neg_score = generate_score(q2)
                    gap_neg = original_score-neg_score
                    count = count+1
                    if count > 3:
                        # negative_reason = None
                        break
                negative_reason = extract_quoted_text(negative_reason)
                # data.at[index, 'backward_reason2'] = negative_reason
                row['backward_reason2'] = negative_reason

            else: #label 1 denotes to real news 如果新闻为真新闻
                original_score = prompt.format(str2, input_text)
                original_score = generate_score(original_score)

                #正向推理 升高
                positive_reason = prompt12.format(str3,str5,str2, input_text, original_score)
                positive_reason = generate_res(positive_reason)
                q1 = prompt34.format(str3,str2, input_text,original_score, positive_reason)
                pos_score = generate_score(q1)
                gap_pos = pos_score-original_score
                count = 0
                while  pos_score <=50 and gap_pos<0:   #当新闻置信度50分以上为真新闻且生成的推理更能证实为真新闻时跳出循环，要让score2分数越来越高
                    print("*************正向推理****************")
                    question_temp = prompt5.format(str3,str5,str3,str5,str2,input_text,original_score,str3,positive_reason,str3,pos_score)
                    positive_reason = generate_res(question_temp)
                    q1 = prompt34.format(str3,str2, input_text,original_score, positive_reason)
                    pos_score = generate_score(q1)
                    gap_pos = pos_score-original_score
                    count = count+1
                    if count > 3:
                        # positive_reason = None
                        break
                positive_reason = extract_quoted_text(positive_reason)
                row['forward_reason2'] = positive_reason
                
                #反向推理 reversed reasoning 降低
                negative_reason = prompt12.format(str4, str6,str2, input_text, original_score)
                negative_reason = generate_res(negative_reason)
                q_neg = prompt34.format(str4,str2, input_text,original_score, negative_reason)
                neg_score = generate_score(q_neg)
                gap_neg = original_score-neg_score
                count = 0
                while  gap_neg<0:    #当新闻置信度50分以下且生成的推理更能证实为假新闻时跳出循环
                    print("*************反向推理****************")
                    question_temp = prompt5.format(str4,str6,str4,str6,str2,input_text,original_score,str4,negative_reason,str4,neg_score)
                    negative_reason = generate_res(question_temp)
                    q2 = prompt34.format(str4,str2, input_text,original_score, negative_reason)
                    neg_score = generate_score(q2)
                    gap_neg = original_score-neg_score
                    count = count+1
                    if count > 3:
                        # negative_reason = None
                        break
                negative_reason = extract_quoted_text(negative_reason)
                row['backward_reason2'] = negative_reason
            outfile.write(json.dumps(row)+'\n')


def main():
    run_reason()



if __name__ == '__main__':
    main()
