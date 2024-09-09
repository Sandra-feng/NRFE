import json

# 打开并读取JSON文件
with open('datasets/arg_gossipcop/train.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
# 如果数据是一个列表，遍历每个元素
if isinstance(data, list):
    for record in data:
        print("Content:", record["content"])
       