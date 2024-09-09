import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import copy
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm
# from dataset import FeatureDataset
from newmodel import Similarity, DetectionModule,Attention_Encoder,Reason_Similarity,Aggregator,BertEncoder
from sklearn import metrics
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import re
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.metrics import accuracy_score,f1_score

# 数据加载和预处理
# Configs
DEVICE = "cuda:0"
NUM_WORKER = 1
BATCH_SIZE = 6
LR = 1e-3
L2 = 0  # 1e-5
NUM_EPOCH = 100

torch.cuda.empty_cache()
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

print(torch.__version__)

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
# device = torch.device("cuda:0")
# 预处理
def text_preprocessing(text):
    """
    - 删除实体@符号(如。“@united”)
    — 纠正错误(如:'&amp;' '&')
    @参数 text (str):要处理的字符串
    @返回 text (Str):已处理的字符串
    """
    # 去除 '@name'
    text = re.sub(r'(@.*?)[\s]', ' ', text)

    #  替换'&amp;'成'&'
    text = re.sub(r'&amp;', '&', text)

    # 删除尾随空格
    text = re.sub(r'\s+', ' ', text).strip()
    # print(text)

    return text

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
        text = re.sub(pattern,'',text)
    pa = r'"([^"]+)"'
    matches  = re.findall(pa,text)
    text = text[:300]
    quoted_text = ''.join(matches)
    # print('处理后：',quoted_text)
    return quoted_text

def tokenize_and_numericalize_data(text, tokenizer):
    # print("Text to tokenize:", text)  # 打印传递给tokenizer的文本
    tokenized = tokenizer(text, truncation=True, padding='max_length', max_length=100)
    # print("Tokenized length:", len(tokenized['input_ids']))  # 打印tokenized长度
    return tokenized['input_ids']

class FakeNewsDataset(Dataset):

    def __init__(self, df, tokenizer, MAX_LEN):
        """
        参数:
            csv_file (string):包含文本和图像名称的csv文件的路径
            root_dir (string):目录
            transform(可选):图像变换
        """
        self.csv_data = df
        self.tokenizer = tokenizer
        self.MAX_LEN = MAX_LEN

    def __len__(self):
        return self.csv_data.shape[0]

    def __getitem__(self, idx):
        text = self.csv_data['statement'][idx]
        pos = self.csv_data['forward_reason2'][idx]
        # print(pos)
        neg = self.csv_data['backward_reason2'][idx]
        text = text_preprocessing(text)
        pos = text_preprocessing(extract_quoted_text(pos))
        neg = text_preprocessing(extract_quoted_text(neg))

        content_input_id = tokenize_and_numericalize_data(text,self.tokenizer)
        pos_input_id = tokenize_and_numericalize_data(pos,self.tokenizer)
        neg_input_id = tokenize_and_numericalize_data(neg,self.tokenizer)
        # print(content_input_id.shape)

        label = self.csv_data['target'][idx]
        label = int(label)
        label = torch.tensor(label)
        sample = {
            'content': torch.tensor(content_input_id),
            'pos_reason': torch.tensor(pos_input_id),
            'neg_reason': torch.tensor(neg_input_id),
            'label': label
        }

        return sample
    


df_train = pd.read_csv("/mnt/C00C86F42C263BF6/fzl/DELL-main_617/datasets/politifact_train2.csv")
df_test = pd.read_csv("/mnt/C00C86F42C263BF6/fzl/DELL-main_617/datasets/politifact_test2.csv")
# df_train = pd.read_csv("g:/A/politifact_test2.csv",encoding='utf-8')
# df_test = pd.read_csv("g:/A/politifact_test2.csv",encoding='utf-8')
# df_train = df_train.sample(frac=1,random_state=42)
# df_test = df_test.sample(frac=1,random_state=42)
print(df_test)
df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
# 假设标签列名为 'label'
labels = df_train['target']

# 计算标签的分布
label_counts = labels.value_counts()

# 打印标签分布
print(label_counts)

# # 可视化标签分布
# label_counts.plot(kind='bar')
# plt.xlabel('Labels')
# plt.ylabel('Counts')
# plt.title('train Label Distribution')
# plt.show()
MAX_LEN = 200
def collate_fn(batch):
    content = torch.stack([item['content'] for item in batch])
    pos_reason = torch.stack([item['pos_reason'] for item in batch])
    neg_reason = torch.stack([item['neg_reason'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    return {'content': content, 'pos_reason': pos_reason, 'neg_reason': neg_reason, 'label': labels}


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
# tokenizer = BertTokenizer.from_pretrained('G:/mulyimodalfakenews/bert-base-uncased', do_lower_case=True)
dataset_train = FakeNewsDataset(df_train, tokenizer, MAX_LEN)

dataset_val = FakeNewsDataset(df_test, tokenizer, MAX_LEN)

train_dataloader = DataLoader(dataset_train, batch_size=8,
                        shuffle=True, num_workers=4,collate_fn=collate_fn)

val_dataloader = DataLoader(dataset_val, batch_size=8,
                        shuffle=True, num_workers=4,collate_fn=collate_fn)


bert = BertEncoder(256,False)
bert.to(device)
bert2 = BertEncoder(256,False)
bert2.to(device)
# 定义整体的教师模型类
class TeacherModel(nn.Module):
    def __init__(self):
        super(TeacherModel, self).__init__()
        # 假设每个子模型已经定义并初始化

        self.attention = Attention_Encoder()
        self.R2T_usefulness = Similarity()  
        self.T2R_usefulness = Similarity()  
        self.Reason_usefulness = Reason_Similarity()
        self.aggregator = Aggregator()  

    def forward(self, content, positive, negative):
        # 教师模型的前向传播逻辑，可以根据实际情况调整
        pos_reason2text, pos_text2reason, positive = self.attention(content, positive)
        text_R2T_aligned, R2T_aligned, _ = self.R2T_usefulness(content, pos_reason2text)
        text_T2R_aligned, T2R_aligned, _ = self.T2R_usefulness(content, pos_text2reason)
        text_R_aligned, R_aligned, _ = self.Reason_usefulness(content, positive)
        output = self.aggregator(content, R2T_aligned, T2R_aligned, R_aligned)
        
        return output
# 找到最后一层的名字与参数


# 加载已经训练好的状态字典到教师模型中
def load_teacher_model(state_dicts):
    teacher_model = TeacherModel()
    teacher_model.attention.load_state_dict(state_dicts['attention'])
    teacher_model.R2T_usefulness.load_state_dict(state_dicts['R2T_usefulness'])
    teacher_model.T2R_usefulness.load_state_dict(state_dicts['T2R_usefulness'])
    # 加载到 Reason_usefulness 的倒数第二层为止
    teacher_model.Reason_usefulness.load_state_dict(state_dicts['Reason_usefulness'])
    teacher_model.aggregator.load_state_dict(state_dicts['aggregator'])

    return teacher_model

# 示例：加载已保存的状态字典并实例化教师模型
state_dicts = torch.load('/mnt/C00C86F42C263BF6/fzl/CAFE-main703/CAFE-main/best_teachermodel.pth')
teacher_model = load_teacher_model(state_dicts).to(device)
teacher_model.eval()  # 进入评估模式，用于推断

# 加载保存的模型参数
checkpoint = torch.load('/mnt/C00C86F42C263BF6/fzl/CAFE-main703/CAFE-main/best_teachermodel.pth')

# 从检查点中提取 BERT 参数
bert_state_dict = checkpoint['bert']
bert2_state_dic = checkpoint['bert2']
# 将保存的参数加载到当前 BERT 模型中
bert.load_state_dict(bert_state_dict)
bert2.load_state_dict(bert2_state_dic)




# 定义学生模型
class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim, output_dim=64, num_layers=1, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layers = nn.TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)  # [10, 500, 768] -> [10, 500, hidden_dim]
        x = x.permute(1, 0, 2)  # [500, 10, hidden_dim]，将 batch 维度移到第二维，以符合 TransformerEncoder 的输入要求
        transformer_output = self.transformer_encoder(x)  # [500, 10, hidden_dim]
        transformer_output = transformer_output.mean(dim=0)  # 取平均，得到 [10, hidden_dim]
        final_output = self.output_layer(transformer_output)  # [10, output_dim]
        return final_output

class StudentModel(nn.Module):
    def __init__(self, hidden_dim, num_heads, transformer_hidden_dim, output_dim=64):
        super(StudentModel, self).__init__()
        self.transformer = TransformerModel(hidden_dim, num_heads, transformer_hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        
        output = self.transformer(x)  # 添加 batch 维度进行处理
        
        return output

input_dim = 768  # 输入特征维度
num_heads = 8    # Transformer 的注意力头数
hidden_dim = 256  # Transformer 的隐藏层维度
output_dim = 64  # 最终输出维度
num_layers = 2   # Transformer 编码器的层数

student_model = TransformerModel(input_dim, num_heads, hidden_dim, output_dim, num_layers)

student_model = student_model.to(device)

# 定义知识蒸馏的损失函数
class DistillationLoss(nn.Module):
    def __init__(self, temperature, alpha):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, student_outputs, teacher_outputs):
        # 计算反向 KL 散度损失
        kd_loss = nn.KLDivLoss(reduction='batchmean')(
            torch.log_softmax(student_outputs / self.temperature, dim=1),
            torch.softmax(teacher_outputs / self.temperature, dim=1)
        ) * (self.temperature ** 2)
        
        return kd_loss



def evaluate():
        
    # 设置模型为评估模式
    student_model.eval()
    detection_module.eval()
    bert.eval()

    # 存储所有的预测标签和真实标签
    all_preds = []
    all_labels = []
    detection_pre_label_all = []
    detection_label_all = []
    # 禁用梯度计算
    with torch.no_grad():
        for batch in val_dataloader:
            news_content = batch['content'].to(device)
            pos =  batch["pos_reason"].to(device)
            neg = batch['neg_reason'].to(device)
            label = batch['label'].to(device)

            content = bert(news_content)

            # 获取学生模型的输出
            student_outputs = student_model(content)
            pre_detection = detection_module(student_outputs)
            
            # 假设 pre_detection 是 logit，需要转换为预测标签
            pre_label_detection = pre_detection.argmax(1)
            detection_label_all.append(label.detach().cpu().numpy())
            # print(pre_label_detection.type())
            
            detection_pre_label_all.append(pre_label_detection.detach().cpu().numpy())
            
        # 计算准确率和 F1 分数
        detection_pre_label_all = np.concatenate(detection_pre_label_all, 0)
        detection_label_all = np.concatenate(detection_label_all, 0)
        acc = accuracy_score(detection_pre_label_all, detection_label_all)
    # f1 = f1_score(all_labels, all_preds, average='weighted')

    print(f'Accuracy: {acc:.4f}')
    # print(f'F1 Score: {f1:.4f}')



# 训练学生模型
distillation_loss = DistillationLoss(temperature=2.0, alpha=0.7)  # 根据需要设置温度和alpha值

# 设置优化器
optimizer = optim.Adam(student_model.parameters(), lr=3e-5)

# detection = DetectionModule()
detection_module = DetectionModule()  
detection_module.to(device)
detection_module_state_dic = checkpoint['detection_module']
detection_module.load_state_dict(detection_module_state_dic)
loss_func_similarity = torch.nn.CosineEmbeddingLoss()
loss_func_detection = torch.nn.CrossEntropyLoss()
optim_task_detection = torch.optim.Adam(
    detection_module.parameters(), lr=1e-3, weight_decay=1e-5
)  
optim_bert = torch.optim.Adam(bert.parameters(),lr=1e-3,weight_decay=0)
optim_bert2 = torch.optim.Adam(bert2.parameters(),lr=1e-3,weight_decay=0)
student_model.train()
detection_module.train()
acc_best_train = 0.0
for epoch in range(10):
    total_loss = 0.0
    
    for batch in train_dataloader:
        news_content = batch['content'].to(device)
        pos =  batch["pos_reason"].to(device)
        neg = batch['neg_reason'].to(device)
        label = batch['label'].to(device)
            # c_input_ids, c_attn_mask = tuple(c.to(device) for c in content)
            # p_input_ids, p_attn_mask = tuple(p.to(device) for p in pos)
            # n_input_ids, n_attn_mask = tuple(n.to(device) for n in neg)

        content = bert(news_content)
        positive = bert2(pos)
        negative = bert2(neg)

        # 清除优化器的梯度
        optimizer.zero_grad()
        optim_task_detection.zero_grad()
        optim_bert.zero_grad()
        optim_bert2.zero_grad()

        # 教师模型的输出
        with torch.no_grad():
            teacher_outputs = teacher_model(content, positive, negative)
            # print(teacher_outputs.shape)
        
        # 学生模型的输出
        student_outputs = student_model(content)
        pre_detection = detection_module(student_outputs)
        
        # 计算损失
        loss_detection = loss_func_detection(pre_detection, label)
        loss_distillation = distillation_loss(student_outputs, teacher_outputs)
        total_loss_step = loss_detection + loss_distillation
        
        # 反向传播和优化
        total_loss_step.backward()
        optimizer.step()
        optim_bert.step()
        optim_bert2.step()
        optim_task_detection.step()
        
        total_loss += total_loss_step.item()
    
    # 打印每个epoch的平均损失
    print(f"Epoch {epoch+1}, Average Loss: {total_loss / len(val_dataloader)}")
    evaluate()

print("Training finished.")

print("学生模型训练完成")

