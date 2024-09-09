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
from sklearn.metrics import accuracy_score,f1_score,precision_recall_fscore_support,precision_score,recall_score
import warnings
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import ticker
warnings.filterwarnings("ignore")

###################################
#蒸馏模型
##################################




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
    tokenized = tokenizer(text, truncation=True, padding='max_length', max_length=200)
    # print("Tokenized length:", len(tokenized['input_ids']))  # 打印tokenized长度
    return tokenized['input_ids']

# 文本字符转为数字编码id
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
        try:

            text = self.csv_data['statement'][idx]
            pos = self.csv_data['forward_reason2'][idx]
            # print(pos)
            neg = self.csv_data['backward_reason2'][idx]
            # text = text_preprocessing(text)
            # pos = text_preprocessing(extract_quoted_text(pos))
            # neg = text_preprocessing(extract_quoted_text(neg))
            # print(pos)

            content_input_id = tokenize_and_numericalize_data(text,self.tokenizer)
            pos_input_id = tokenize_and_numericalize_data(pos,self.tokenizer)
            neg_input_id = tokenize_and_numericalize_data(neg,self.tokenizer)
        except (ValueError):
            return None
        # print(content_input_id.shape)

        label = self.csv_data['target'][idx]
        label = int(label)
        # print(label)
        label = torch.tensor(label)
        sample = {
            'content': torch.tensor(content_input_id),
            'pos_reason': torch.tensor(pos_input_id),
            'neg_reason': torch.tensor(neg_input_id),
            'label': label
        }

        return sample

import csv

#数据集处理
df_train = pd.read_csv("/mnt/C00C86F42C263BF6/fzl/DELL-main_617/datasets/politifact_train2.csv")
df_test = pd.read_csv("/mnt/C00C86F42C263BF6/fzl/DELL-main_617/datasets/politifact_test2.csv")
labels = df_train['target']
# 计算标签的分布
label_counts = labels.value_counts()

# 打印标签分布
print(label_counts)
csv_file = 'twitter15_student.csv'
with open(csv_file,mode='w',newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch','loss','Accuracy','macro_f1','mic_f1','fake_f1','fake_precision','fake_recall','true_f1','true_precision','true_recall'])

MAX_LEN = 200
def collate_fn(batch):
    content = torch.stack([item['content'] for item in batch if item is not None])
    pos_reason = torch.stack([item['pos_reason'] for item in batch if item is not None])
    neg_reason = torch.stack([item['neg_reason'] for item in batch if item is not None])
    labels = torch.stack([item['label'] for item in batch if item is not None])
    if len(batch)==0:
        return None
    return {'content': content, 'pos_reason': pos_reason, 'neg_reason': neg_reason, 'label': labels}


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
# tokenizer = BertTokenizer.from_pretrained('G:/mulyimodalfakenews/bert-base-uncased', do_lower_case=True)
dataset_train = FakeNewsDataset(df_train, tokenizer, MAX_LEN)

dataset_val = FakeNewsDataset(df_test, tokenizer, MAX_LEN)

train_dataloader = DataLoader(dataset_train, batch_size=16,
                        shuffle=True, num_workers=4,collate_fn=collate_fn)

val_dataloader = DataLoader(dataset_val, batch_size=16,
                        shuffle=True, num_workers=4,collate_fn=collate_fn)

##模型定义
bert = BertEncoder(256,False)
bert.to(device)
bert2 = BertEncoder(256,False)
bert2.to(device)
# 定义整体的教师模型类
class TeacherModel(nn.Module):
    def __init__(self):
        super(TeacherModel, self).__init__()
        # 假设每个子模型已经定义并初始化
        self.bert = BertEncoder(256,False)
        self.bert2 = BertEncoder(256,False)

        self.attention = Attention_Encoder()
        self.R2T_usefulness = Similarity()  
        self.T2R_usefulness = Similarity()  
        self.Reason_usefulness = Reason_Similarity()
        self.aggregator = Aggregator()
        self.detection_module = DetectionModule() 

    def forward(self, news_content, positive):
        # 教师模型的前向传播逻辑，可以根据实际情况调整
        content = self.bert(news_content)
        positive = self.bert2(positive)
        pos_reason2text, pos_text2reason, positive = self.attention(content, positive)
        text_R2T_aligned, R2T_aligned, _ = self.R2T_usefulness(content, pos_reason2text)
        text_T2R_aligned, T2R_aligned, _ = self.T2R_usefulness(content, pos_text2reason)
        text_R_aligned, R_aligned, _ = self.Reason_usefulness(content, positive)
        # output = self.aggregator(content,R_aligned)
        output = self.aggregator(content,R2T_aligned,T2R_aligned,R_aligned)
        # output = self.aggregator(content,R_aligned)
        teacher_pre = self.detection_module(output)
        
        return output , teacher_pre



# 加载已经训练好的状态字典模型参数到教师模型中
def load_teacher_model(state_dicts):
    teacher_model = TeacherModel()
    teacher_model.bert.load_state_dict(state_dicts['bert'])
    teacher_model.bert2.load_state_dict(state_dicts['bert2'])
    teacher_model.attention.load_state_dict(state_dicts['attention'])
    teacher_model.R2T_usefulness.load_state_dict(state_dicts['R2T_usefulness'])
    teacher_model.T2R_usefulness.load_state_dict(state_dicts['T2R_usefulness'])
    # 加载到 Reason_usefulness 的倒数第二层为止
    teacher_model.Reason_usefulness.load_state_dict(state_dicts['Reason_usefulness'])
    teacher_model.aggregator.load_state_dict(state_dicts['aggregator'])
    #teacher_model.detection_module.load_state_dict(state_dicts['detection_module'])

    return teacher_model

# 加载已保存的状态字典并实例化教师模型
state_dicts = torch.load('CAFE-main/poli_teachermodel.pth')
teacher_model = load_teacher_model(state_dicts).to(device)
teacher_model.eval()  # 进入评估模式，用于推断

# 加载保存的模型参数
checkpoint = torch.load('CAFE-main/poli_teachermodel.pth')

# 从检查点中提取 BERT 参数
bert_state_dict = checkpoint['bert']
bert.load_state_dict(bert_state_dict)


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x shape: (batch_size, seq_length, hidden_dim)
        query = self.query(x)   # (batch_size, seq_length, hidden_dim)
        key = self.key(x)       # (batch_size, seq_length, hidden_dim)
        value = self.value(x)   # (batch_size, seq_length, hidden_dim)
        # Calculate attention scores
        attention_scores = torch.bmm(query, key.transpose(1, 2))  
        attention_scores = self.softmax(attention_scores)        
        # Apply attention scores to value
        attention_output = torch.bmm(attention_scores, value)    
        summarized_output = attention_output.mean(dim=1)          
        return summarized_output



# 定义学生模型
class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim, output_dim=64, num_layers=12, dropout=0.3):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layers = nn.TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.attention = SelfAttention(256)

    def forward(self, x):
        x = self.embedding(x)  # [10, 500, 768] -> [10, 500, hidden_dim]
        x = x.permute(1, 0, 2)  # [500, 10, hidden_dim]，将 batch 维度移到第二维，以符合 TransformerEncoder 的输入要求
        transformer_output = self.transformer_encoder(x)  # [500, 10, hidden_dim]

        transformer_output = transformer_output.mean(dim=0)  # 取平均，得到 [10, hidden_dim]
        # final_output = self.attention(transformer_output)2
        final_output = self.output_layer(transformer_output)  # [10, output_dim]
        return final_output


input_dim = 768  # 输入特征维度
num_heads = 8    # Transformer 的注意力头数
hidden_dim = 256  # Transformer 的隐藏层维度
output_dim = 64  # 最终输出维度
num_layers = 12  # Transformer 编码器的层数

#学生模型实例化
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


lr = 3e-5
# 训练学生模型
distillation_loss = DistillationLoss(temperature=3, alpha=0.7)  # 根据需要设置温度和alpha值

# 设置优化器
optimizer = optim.Adam(student_model.parameters(), lr=lr,weight_decay=1e-5)

# detection = DetectionModule()
#检测模型的定义，损失函数的定义
detection_module = DetectionModule()  
detection_module.to(device)
detection_module.load_state_dict(checkpoint['detection_module'])
loss_func_similarity = torch.nn.CosineEmbeddingLoss()
loss_func_detection = torch.nn.CrossEntropyLoss()
square_loss = torch.nn.MSELoss()
optim_task_detection = torch.optim.Adam(
    detection_module.parameters(), lr=lr, weight_decay=1e-5
)  
optim_bert = torch.optim.Adam(bert.parameters(),lr=lr,weight_decay=1e-5)
optim_bert2 = torch.optim.Adam(bert2.parameters(),lr=lr,weight_decay=1e-5)
student_model.train()
# detection_module.train()
acc_best_train = 0.0

#测试函数
def evaluate(epoch,loss):
        
    # 设置模型为评估模式
    student_model.eval()
    detection_module.eval()
    bert.eval()
    all_student_outputs = []
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
                # c_input_ids, c_attn_mask = tuple(c.to(device) for c in content)
                # p_input_ids, p_attn_mask = tuple(p.to(device) for p in pos)
                # n_input_ids, n_attn_mask = tuple(n.to(device) for n in neg)

            content = bert(news_content)
            # 获取学生模型的输出T2R_aligned
            student_outputs = student_model(content)
            pre_detection = detection_module(student_outputs)
            # print(pre_detection)
            # 假设 pre_detection 是 logit，需要转换为预测标签
            pre_label_detection = pre_detection.argmax(1)
            detection_label_all.append(label.detach().cpu().numpy())
            # print(detection_label_all)
            
            detection_pre_label_all.append(pre_label_detection.detach().cpu().numpy())
            # print(detection_pre_label_all)
            all_student_outputs.append(student_outputs.detach().cpu().numpy())
            all_labels.append(label.detach().cpu().numpy())
            
        # 计算准确率和 F1 分数
        all_labels = np.concatenate(all_labels,axis=0)
        all_student_outputs = np.concatenate(all_student_outputs,axis=0)

        detection_pre_label_all = np.concatenate(detection_pre_label_all, 0)
        detection_label_all = np.concatenate(detection_label_all, 0)
        acc = accuracy_score(detection_pre_label_all, detection_label_all)
        prf = precision_recall_fscore_support(detection_pre_label_all, detection_label_all, average=None, labels=[0, 1])
        precision_0, recall_0, f1_0 = prf[0][0], prf[1][0], prf[2][0]
        precision_1, recall_1, f1_1 = prf[0][1], prf[1][1], prf[2][1]
        print(f'Accuracy: {acc:.4f}')

        # 计算宏观F1和微观F1分数
        macro_f1 = f1_score(detection_pre_label_all, detection_label_all, average='macro')
        micro_f1 = f1_score(detection_pre_label_all, detection_label_all, average='micro')
        print(f"宏观 F1 分数为: {macro_f1:.3f}")
        print(f"微观 F1 分数为: {micro_f1:.3f}")

        # 计算真假标签的精确度、召回率、F1分数
        precision_true = precision_score(detection_label_all, detection_pre_label_all, pos_label=1)
        recall_true = recall_score(detection_label_all, detection_pre_label_all, pos_label=1)
        f1_true = f1_score(detection_label_all, detection_pre_label_all, pos_label=1)

        precision_false = precision_score(detection_label_all, detection_pre_label_all, pos_label=0)
        recall_false = recall_score(detection_label_all, detection_pre_label_all, pos_label=0)
        f1_false = f1_score(detection_label_all, detection_pre_label_all, pos_label=0)


        print(f"Recall (False): {recall_false}")
        print(f"Precision (False): {precision_false}")
        print(f"F1 Score (False): {f1_false}")
        print(f"Recall (True): {recall_true}")
        print(f"Precision (True): {precision_true}")
        print(f"F1 Score (True): {f1_true}")

        with open(csv_file,mode='a',newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch,loss,acc,macro_f1,micro_f1,f1_false,precision_false,recall_false,f1_true,precision_true,recall_true])
        

        return acc,all_student_outputs, all_labels

#训练模型
def train():
    best_acc = 0 
    
    for epoch in range(30):
        total_loss = 0.0
        all_student_outputs = []
        all_labels = []
        for batch in train_dataloader:
            #获取数据
            news_content = batch['content'].to(device)
            pos =  batch["pos_reason"].to(device)
            neg = batch['neg_reason'].to(device)
            label = batch['label'].to(device)
            #数据编码
            content = bert(news_content)
            
            optimizer.zero_grad()
            optim_task_detection.zero_grad()
            optim_bert.zero_grad()

            # 教师模型的输出，得到教师模型的最后一层输出（分类层的前一层）
            with torch.no_grad():
                teacher_outputs,teacher_pre_logit = teacher_model(news_content, pos)

            #将content输入到学生模型，得到学生模型的分布
            student_outputs = student_model(content)
            #将学生模型的分布输入到检测器中，得到预测结果
            pre_detection = detection_module(student_outputs)

            all_student_outputs.append(student_outputs.detach().cpu().numpy())
            all_labels.append(label.detach().cpu().numpy())
            #计算蒸馏模型的预测损失，（硬标签）
            loss_detection = loss_func_detection(pre_detection, label)
            #计算蒸馏损失，也就是教师模型输出与学生模型输出的损失
            loss_distillation = distillation_loss(student_outputs, teacher_outputs)
            # loss_soft_labels = loss_func_detection(pre_detection, teacher_pre_logit)
            #计算总损失，蒸馏损失+预测损失
            total_loss_step =   loss_distillation + loss_detection
            # 反向传播和优化
            total_loss_step.backward()
            optimizer.step()
            optim_bert.step()
            optim_task_detection.step()
            
            total_loss += total_loss_step.item()
            torch.nn.utils.clip_grad_norm_(bert.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(bert2.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(detection_module.parameters(), max_norm=1.0)

        train_labels = np.concatenate(all_labels,axis=0)
        train_student_outputs = np.concatenate(all_student_outputs,axis=0)

        # 打印每个epoch的平均损失
        average_loss = total_loss / len(val_dataloader)
        print(f"Epoch {epoch+1}, Average Loss: {total_loss / len(val_dataloader)}")
        acc,all_student_outputs, all_labels= evaluate(epoch+1,average_loss)
        print('best acc:',best_acc)
        if acc >= best_acc:
            best_acc = acc
            
            #保存训练集
            np.savez('trainWOT2Routputs.npz',train_student_outputs = train_student_outputs,train_labels = train_labels)
            np.savez('testWOT2Roputputs.npz',all_student_outputs = all_student_outputs,all_labels=all_labels)
            tsne = TSNE(n_components=3, random_state=42)
            student_outputs_tsne_train = tsne.fit_transform(train_student_outputs)
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')
            for label in np.unique(train_labels):
                color = 'lightblue' if label == 1 else 'pink'
                ax.scatter(student_outputs_tsne_train[train_labels == label, 0], 
                        student_outputs_tsne_train[train_labels == label, 1], 
                        student_outputs_tsne_train[train_labels == label, 2],
                        label=f'Class {label}', alpha=0.5, c=color, s=15)
            ax.set_title('T-SNE of Train Student Outputs')
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            ax.set_zlabel('Component 3')

            ax.legend()
            plt.savefig('Polititrain.png')  # 保存训练集的图像
            plt.show()
            
            # 保存测试集的 T-SNE 可视化图
            tsne = TSNE(n_components=3, random_state=42)
            student_outputs_tsne_test = tsne.fit_transform(all_student_outputs)
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')
            for label in np.unique(all_labels):
                color = 'lightblue' if label == 1 else 'pink'
                ax.scatter(student_outputs_tsne_test[all_labels == label, 0], 
                        student_outputs_tsne_test[all_labels == label, 1], 
                        student_outputs_tsne_test[all_labels == label, 2],
                        label=f'Class {label}', alpha=0.5, c=color, s=15)
            ax.set_title('T-SNE of Test Student Outputs')
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            ax.set_zlabel('Component 3')

            ax.legend()
            plt.savefig('Polititest.png')  # 保存测试集的图像
            plt.show()


    print("Training finished.")

    print("学生模型训练完成")
train()


