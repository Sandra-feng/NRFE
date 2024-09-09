import pandas as pd
import torch
from torchvision import models, transforms
from PIL import Image
from PIL import Image, UnidentifiedImageError
from transformers import BertTokenizer, BertModel
import numpy as np
from tqdm import tqdm 
# df_train = pd.read_csv("/mnt/C00C86F42C263BF6/CAFE-main/CAFE-main/train_gossipcop_emotion.csv")
df_test = pd.read_csv("/mnt/C00C86F42C263BF6/DELL-main/DELL-main/datasets/test-gossipcop2.csv",encoding='utf-8')
print(df_test.columns)
print(df_test.head())
# rootdir = 'images/'
class TextEncoder:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('/mnt/C00C86F42C263BF6/bert-base-uncased')
        self.model = BertModel.from_pretrained('/mnt/C00C86F42C263BF6/bert-base-uncased')

    def encode(self, text):
        # Encode the text, ensuring it's truncated to 512 tokens
        encoded_input = self.tokenizer.encode(
            text, 
            add_special_tokens=True, 
            return_tensors='pt', 
            max_length=512,  # Set the max length
            truncation=True  # Ensure truncation is active
        )
        with torch.no_grad():
            output = self.model(encoded_input)
        sentence_embedding = output[0][:, 0, :]
        return sentence_embedding


# 创建编码器实例
text_encoder = TextEncoder()

# 对DataFrame中的数据进行编码
encoded_content = []
emotion_positive = []
emotion_negative = []
propaganda_positive = []
propaganda_negative = []
labels = []
for idx, row in tqdm(df_test.iterrows(), total=df_test.shape[0], desc="Encoding"):
    content = row['content']
    label = row['label']
    # print(label)
    forward_reason1 = row['forward_reason1']
    backward_reason1 = row['backward_reason1']
    forward_reason2 = row['forward_reason2']
    backward_reason2 = row['backward_reason2']
    # print(forward_reason2)



    if pd.isna(content) or pd.isna(forward_reason1) or pd.isna(backward_reason1):
        continue  # 如果任何一个字段是NaN，则跳过这条记录
    if pd.isna(content) or pd.isna(forward_reason2) or pd.isna(backward_reason2):
        continue  # 如果任何一个字段是NaN，则跳过这条记录

    content_embedding = text_encoder.encode(content)
    forward_embedding1 = text_encoder.encode(forward_reason1)
    backward_embedding1 = text_encoder.encode(backward_reason1)
    forward_embedding2 = text_encoder.encode(forward_reason2)
    backward_embedding2 = text_encoder.encode(backward_reason2)

    if content_embedding is None or forward_embedding1 is None or backward_embedding1 is None:
        continue  # 如果编码失败（即返回None），则跳过这条记录
    if content_embedding is None or forward_embedding2 is None or backward_embedding2 is None:
        continue  # 如果编码失败（即返回None），则跳过这条记录

    # 将Tensor转换为numpy数组并存储
    encoded_content.append(content_embedding.detach().cpu().numpy())
    emotion_positive.append(forward_embedding1.detach().cpu().numpy())
    emotion_negative.append(backward_embedding1.detach().cpu().numpy())
    propaganda_positive.append(forward_embedding2.detach().cpu().numpy())
    propaganda_negative.append(backward_embedding2.detach().cpu().numpy())
    labels.append(label)
    # print(labels)

np.savez("test_newdata.npz",
         encoded_content=np.array(encoded_content),
         emotion_positive=np.array(emotion_positive),
         emotion_negative=np.array(emotion_negative),
         propaganda_positive = np.array(propaganda_positive),
         propaganda_negative = np.array(propaganda_negative),
         labels=np.array(labels))
    
    

