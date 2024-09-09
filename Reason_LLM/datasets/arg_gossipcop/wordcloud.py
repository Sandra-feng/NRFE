import json
import pandas as pd
from collections import Counter

def count_words(content_list):
    counter = Counter()
    for content in content_list:
        words = content.split()  # 简单分词，可以根据需要改进
        counter.update(words)
    return counter

def run_word_count():
    # 读取 JSON 数据集
    with open('train.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # 提取 content 列
    content_list = [item['content'] for item in data if 'content' in item]
    
    # 统计词频
    word_counter = count_words(content_list)
    
    # 将词频结果转换为 DataFrame
    word_freq_df = pd.DataFrame(word_counter.items(), columns=['Word', 'Frequency'])
    
    # 按频率降序排序
    word_freq_df = word_freq_df.sort_values(by='Frequency', ascending=False)
    
    # 输出为 XLSX 文件
    output_file = 'word_frequency.xlsx'
    word_freq_df.to_excel(output_file, index=False)
    print(f"词频统计结果已保存到 {output_file}")

def main():
    run_word_count()

if __name__ == '__main__':
    main()
