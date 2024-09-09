import torch
import pandas as pd
import numpy as np
import transformers
import torchvision
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import torch.nn.functional as F
from transformers import BertModel
import random
import time
import os
from transformers import BeitModel, BeitConfig
import re
from PIL import UnidentifiedImageError
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import ticker
import copy
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix,precision_score,recall_score,f1_score,roc_auc_score,classification_report
from tqdm import tqdm
from dataset import FeatureDataset
from newmodel import Similarity, DetectionModule,Attention_Encoder,Reason_Similarity,Aggregator,BertEncoder
from sklearn import metrics
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import re
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# Configs
# DEVICE = "cuda:1"
NUM_WORKER = 1
BATCH_SIZE = 6
LR = 3e-5
L2 = 1e-4
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
##################################
# 数据预处理
##################################
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

    for sentence2 in sentences_to_remove2:
        pattern = re.escape(sentence2)+r'\s*'

    print('处理后：',text)
    return text

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
        try:

            text = self.csv_data['statement'][idx]
            pos = self.csv_data['forward_reason2'][idx]
            neg = self.csv_data['backward_reason2'][idx]
            text = text_preprocessing(text)
            pos = text_preprocessing(extract_quoted_text(pos))
            neg = text_preprocessing(extract_quoted_text(neg))

            content_input_id = tokenize_and_numericalize_data(text,self.tokenizer)
            pos_input_id = tokenize_and_numericalize_data(pos,self.tokenizer)
            neg_input_id = tokenize_and_numericalize_data(neg,self.tokenizer)
        except (ValueError):
            return None
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
    
###############################
# 读取数据集
###############################
df_train = pd.read_csv("/datasets/politifact_train2.csv")
df_test = pd.read_csv("/datasets/politifact_test2.csv")
print(df_test)
df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

MAX_LEN = 200
def collate_fn(batch):
    content = torch.stack([item['content'] for item in batch if item is not None])
    pos_reason = torch.stack([item['pos_reason'] for item in batch if item is not None])
    neg_reason = torch.stack([item['neg_reason'] for item in batch if item is not None])
    labels = torch.stack([item['label'] for item in batch if item is not None])
    if len(batch)==0:
        return None
    return {'content': content, 'pos_reason': pos_reason, 'neg_reason': neg_reason, 'label': labels}

# 数据集批次打包
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
dataset_train = FakeNewsDataset(df_train, tokenizer, MAX_LEN)
dataset_val = FakeNewsDataset(df_test, tokenizer, MAX_LEN)
train_dataloader = DataLoader(dataset_train, batch_size=10,
                        shuffle=True, num_workers=4,collate_fn=collate_fn)
val_dataloader = DataLoader(dataset_val, batch_size=10,
                        shuffle=True, num_workers=4,collate_fn=collate_fn)
############################
# 结果记录为csv文件的定义
############################
import csv
csv_file = 'teacher.csv'
with open(csv_file,mode='w',newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch','loss','Accuracy','macro_f1','mic_f1','fake_f1','fake_precision','fake_recall','true_f1','true_precision','true_recall'])


################################
#模型的训练与测试
################################
def train():
    # ---  Load Config  ---
    # device = torch.device(DEVICE)
    lr = LR #学习率
    l2 = L2  #L2正则化
    num_epoch = NUM_EPOCH  #批次
    # ---  Build Model & Trainer  ---
    bert = BertEncoder(256,False)
    bert.to(device)   #新闻编码器定义
    optim_bert = torch.optim.Adam(bert.parameters(),lr=lr,weight_decay=l2)  #优化器配置
    bert2 = BertEncoder(256,False)
    bert2.to(device)  #推理编码器定义
    optim_bert2 = torch.optim.Adam(bert2.parameters(),lr=lr,weight_decay=l2)
    attention = Attention_Encoder() #融合交互模型attention定义
    attention.to(device)
    optim_task_attention = torch.optim.Adam(
        attention.parameters(), lr=lr, weight_decay=l2
    )  
    #有用性判断模型的实例化
    R2T_usefulness = Similarity()  
    R2T_usefulness.to(device)
    optim_task_R2T = torch.optim.Adam(
        R2T_usefulness.parameters(), lr=lr, weight_decay=l2
    )
    T2R_usefulness = Similarity()  
    T2R_usefulness.to(device)
    optim_task_T2R = torch.optim.Adam(
        T2R_usefulness.parameters(), lr=lr, weight_decay=l2
    )  
    Reason_usefulness = Reason_Similarity()
    Reason_usefulness.to(device)
    optim_task_reason = torch.optim.Adam(
        Reason_usefulness.parameters(), lr=lr, weight_decay=l2
    )
    aggregator = Aggregator()
    aggregator.to(device)
    optim_task_aggregator = torch.optim.Adam(
        aggregator.parameters(), lr=lr, weight_decay=l2
    )
    detection_module = DetectionModule()  
    detection_module.to(device)
    loss_func_similarity = torch.nn.CosineEmbeddingLoss()
    loss_func_detection = torch.nn.CrossEntropyLoss()
    optim_task_detection = torch.optim.Adam(
        detection_module.parameters(), lr=lr, weight_decay=l2
    )  

    loss_R2T_list = []
    loss_T2R_list = []
    loss_reason_list = []
    loss_detection_list = []
    # ---  Model Training  ---
    loss_R2T_total = 0
    loss_detection_total = 0
    best_acc = 0
    best_loss = 1000
    acc_best_train = 0
    # 定义保存路径
    save_path = '/politifact731.pth'
    #epoch是训练的轮次
    for epoch in range(num_epoch):
        attention.train()
        R2T_usefulness.train()
        T2R_usefulness.train()
        detection_module.train()
        Reason_usefulness.train()

        corrects_pre_R2T = 0
        corrects_pre_T2R = 0
        corrects_pre_reason = 0
        corrects_pre_detection = 0
        loss_R2T_total = 0
        loss_T2R_total = 0
        loss_R_total = 0
        loss_detection_total = 0
        R2T_count = 0
        T2R_count = 0
        R_count = 0
        detection_count = 0

        epoch_loss_R2T = 0.0
        epoch_loss_T2R = 0.0
        epoch_loss_reason = 0.0
        epoch_loss_detection = 0.0
        num_batches = 0
        train_outputs = []
        train_labels = []
        #batch是数据集的批次，比如每批次8条数据，同时进行
        for batch in train_dataloader:
            #读取数据内容和推理、标签
            news_content = batch['content'].to(device)  #to.(device)是放到gpu上运行
            pos =  batch["pos_reason"].to(device)
            neg = batch['neg_reason'].to(device)
            label = batch['label'].to(device)
            #编码器对文本数据进行编码，也就是转为向量
            content = bert(news_content)
            positive = bert2(pos)
            negative = bert2(neg)
            #信息融合交互模块，建议去学一下cross attention了解一下pos_reason2text, pos_text2reason的区别
            # --- cross attention ---
            pos_reason2text, pos_text2reason, positive = attention(content, positive)
            neg_reason2text, neg_text2reason, negative = attention(content, negative)

            #有用性训练模块R2T_usefulness，分别输入正推理与负推理，将正推理的标签设置为1，表示有用，负推理的标签设置为0，表示无用，训练有用性检测网络。
            # --- task1 R2T  loss ---
            #损失是将标签拼接成了向量，使用的余弦相似度的损失函数
            text_aligned_match1, R2T_match, pred_R2T_match = R2T_usefulness(content, pos_reason2text)
            text_aligned_unmatch1, R2T_unmatch, pred_R2T_unmatch = R2T_usefulness(content, neg_reason2text)
            R2T_pred = torch.cat([pred_R2T_match.argmax(1), pred_R2T_unmatch.argmax(1)], dim=0)
            R2Tlabel_0 = torch.cat([torch.ones(pred_R2T_match.shape[0]), torch.zeros(pred_R2T_unmatch.shape[0])], dim=0).to(device)
            R2Tlabel_1 = torch.cat([torch.ones(pred_R2T_match.shape[0]), -1 * torch.ones(pred_R2T_unmatch.shape[0])], dim=0).to(device)
            text_aligned_4_task1 = torch.cat([text_aligned_match1, text_aligned_unmatch1], dim=0)
            R2T_aligned_4_task1 = torch.cat([R2T_match, R2T_unmatch], dim=0)
            loss_R2T = loss_func_similarity(text_aligned_4_task1, R2T_aligned_4_task1, R2Tlabel_1)

            # --- task2 T2R loss ---
            #task 1,2,3使用的方法都一样
            text_aligned_match2, T2R_match, pred_T2R_match = T2R_usefulness(content, pos_text2reason)
            text_aligned_unmatch2, T2R_unmatch, pred_T2R_unmatch = T2R_usefulness(content, neg_text2reason)
            T2R_pred = torch.cat([pred_T2R_match.argmax(1), pred_T2R_unmatch.argmax(1)], dim=0)
            T2R_label_0 = torch.cat([torch.ones(pred_T2R_match.shape[0]), torch.zeros(pred_T2R_unmatch.shape[0])], dim=0).to(device)
            T2R_label_1 = torch.cat([torch.ones(pred_T2R_match.shape[0]), -1 * torch.ones(pred_T2R_unmatch.shape[0])], dim=0).to(device)
            text_aligned_4_task2 = torch.cat([text_aligned_match2, text_aligned_unmatch2], dim=0)
            T2R_aligned_4_task2 = torch.cat([T2R_match, T2R_unmatch], dim=0)
            loss_T2R = loss_func_similarity(text_aligned_4_task2, T2R_aligned_4_task2, T2R_label_1)

            # --- task3 Reason loss ---
            text_aligned_match3, reason_match, pred_reason_match = Reason_usefulness(content, positive)
            text_aligned_unmatch3, reason_unmatch, pred_reason_unmatch = Reason_usefulness(content, negative)
            reason_pred = torch.cat([pred_reason_match.argmax(1), pred_reason_unmatch.argmax(1)], dim=0)
            reason_label_0 = torch.cat([torch.ones(pred_reason_match.shape[0]), torch.zeros(pred_reason_unmatch.shape[0])], dim=0).to(device)
            reason_label_1 = torch.cat([torch.ones(pred_reason_match.shape[0]), -1 * torch.ones(pred_reason_unmatch.shape[0])], dim=0).to(device)
            text_aligned_4_task3 = torch.cat([text_aligned_match3, text_aligned_unmatch3], dim=0)
            reason_aligned_4_task3 = torch.cat([reason_match, reason_unmatch], dim=0)
            loss_reason = loss_func_similarity(text_aligned_4_task3, reason_aligned_4_task3, reason_label_1)

            # --- TASK fake news Detection ---
            #经过有用性判断模块以后，得到经新闻内容加权过后的推理信息表示
            text_R2T_aligned, R2T_aligned, _ = R2T_usefulness(content, pos_reason2text)
            text_T2R_aligned, T2R_aligned, _ = T2R_usefulness(content, pos_text2reason)
            text_R_aligned, R_aligned, _ = Reason_usefulness(content, positive)
            # final_feature = aggregator(content, R2T_aligned, T2R_aligned, R_aligned)
            #将所有信息进行拼接融合
            final_feature = aggregator(content,R2T_aligned,T2R_aligned,R_aligned)
            #分类模块
            pre_detection = detection_module(final_feature)
            loss_detection = loss_func_detection(pre_detection, label)
            pre_label_detection = pre_detection.argmax(1)
            train_outputs.append(final_feature.detach().cpu().numpy())
            train_labels.append(label.detach().cpu().numpy())

            # 梯度清零
            optim_task_attention.zero_grad()
            optim_task_R2T.zero_grad()
            optim_task_T2R.zero_grad()
            optim_task_reason.zero_grad()
            optim_task_aggregator.zero_grad()
            optim_task_detection.zero_grad()
            optim_bert.zero_grad()
            optim_bert2.zero_grad()

            # 反向传播
            loss_R2T.backward(retain_graph=True)
            loss_T2R.backward(retain_graph=True)
            loss_reason.backward(retain_graph=True)
            loss_detection.backward()

            # 参数更新
            optim_task_attention.step()
            optim_task_R2T.step()
            optim_task_T2R.step()
            optim_task_reason.step()
            optim_task_detection.step()
            optim_task_aggregator.step()
            optim_bert2.step()
            optim_bert.step()

            corrects_pre_R2T += R2T_pred.eq(R2Tlabel_0).sum().item()
            corrects_pre_T2R += T2R_pred.eq(T2R_label_0).sum().item()
            corrects_pre_reason += reason_pred.eq(reason_label_0).sum().item()
            corrects_pre_detection += pre_detection.argmax(1).eq(label.view_as(pre_detection.argmax(1))).sum().item()

            # ---  Record  ---
            loss_R2T_total += loss_R2T.item() * (2 * content.shape[0])
            loss_T2R_total += loss_T2R.item() * (2 * content.shape[0])
            loss_R_total += loss_reason.item() * (2 * content.shape[0])
            loss_detection_total += loss_detection.item() * content.shape[0]
            R2T_count += (2 * content.shape[0] * 2)
            T2R_count += (2 * content.shape[0] * 2)
            R_count += (2 * content.shape[0] * 2)
            detection_count += content.shape[0]
            # 计算累积损失值
            epoch_loss_R2T += loss_R2T.item()
            epoch_loss_T2R += loss_T2R.item()
            epoch_loss_reason += loss_reason.item()
            epoch_loss_detection += loss_detection.item()
            num_batches += 1

        train_labels = np.concatenate(train_labels,axis=0)
        train_outputs = np.concatenate(train_outputs,axis=0)

        loss_R2T_train = loss_R2T_total / R2T_count
        loss_T2R_train = loss_T2R_total / T2R_count
        loss_R_train = loss_R_total / R_count
        loss_detection_train = loss_detection_total / detection_count
        acc_R2T_train = corrects_pre_R2T / R2T_count
        acc_T2R_train = corrects_pre_T2R / T2R_count
        acc_detection_train = corrects_pre_detection / detection_count
        # 计算平均损失值
        avg_loss_R2T = epoch_loss_R2T / num_batches
        avg_loss_T2R = epoch_loss_T2R / num_batches
        avg_loss_reason = epoch_loss_reason / num_batches
        avg_loss_detection = epoch_loss_detection / num_batches

        # 存储损失值
        loss_R2T_list.append(avg_loss_R2T)
        loss_T2R_list.append(avg_loss_T2R)
        loss_reason_list.append(avg_loss_reason)
        loss_detection_list.append(avg_loss_detection)

        # ---  Test  ---
        #############################
        #这里设置的是每训练完一个epoch就用测试集测试一次结果
        ################################
        acc_R2T_test, acc_detection_test, loss_R2T_test, loss_detection_test, cm_similarity, cm_detection,loss_T2R_test,acc_T2R_test,test_outputs,test_labels= test(bert,bert2,Reason_usefulness, attention,R2T_usefulness, detection_module,T2R_usefulness, aggregator,val_dataloader,epoch,avg_loss_detection)
        # best_acc = max(best_acc,acc_detection_test)
        
        # 在准确度达到最高时保存训练好的模型参数。
        if acc_detection_test >= best_acc and loss_detection_test <= best_loss:
            best_acc = acc_detection_test
            state_dicts = {
                'bert':bert.state_dict(),
                'bert2':bert2.state_dict(),
                'attention': attention.state_dict(),
                'R2T_usefulness': R2T_usefulness.state_dict(),
                'T2R_usefulness': T2R_usefulness.state_dict(),
                'Reason_usefulness': Reason_usefulness.state_dict(),
                'aggregator':aggregator.state_dict(),
                'detection_module': detection_module.state_dict(),
            }
            torch.save(state_dicts, save_path)

            print(f'Best model parameters saved to {save_path}')

        best_loss = loss_detection_test

        #用TSNE技术绘制散点图，查看数据分布情况
        np.savez('teacher.npz',train_student_outputs = train_outputs,train_labels = train_labels)
        np.savez('teacher.npz',all_student_outputs = test_outputs,all_labels=test_labels)
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        tsne = TSNE(n_components=3, random_state=42)
        student_outputs_tsne = tsne.fit_transform(train_outputs)
        # 设置背景颜色为深灰色
        fig.patch.set_facecolor('#B7B7B7')
        # ax.set_facecolor('#B7B7B7')
        # 绘制气泡图
        for i in range(len(train_labels)):
            color = '#58539f' if train_labels[i] == 0 else '#d86967'
            # edgecolor = 'red' if loaded_labels[i] == 0 else 'blue'
            ax.scatter(student_outputs_tsne[i, 0], student_outputs_tsne[i, 1], student_outputs_tsne[i, 2], 
                    c=color, s=10, alpha=0.5,  linewidth=1)

        # 保留网格线
        ax.grid(True)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        # 移除刻度线

        # 移除刻度线但保留网格线
        ax.xaxis.set_tick_params(width=0)
        ax.yaxis.set_tick_params(width=0)
        ax.zaxis.set_tick_params(width=0)

        # 移除刻度线实线
        ax.tick_params(axis='x', colors='none')
        ax.tick_params(axis='y', colors='none')
        ax.tick_params(axis='z', colors='none')

        plt.savefig('tsne_student_outputs_3d.png')
        plt.show()

        print('---  TASK1 R2T  ---')
        print(
            "EPOCH = %d \n acc_similarity_train = %.3f \n acc_similarity_test = %.3f \n loss_similarity_train = %.3f \n loss_similarity_test = %.3f \n" %
            (epoch + 1, acc_R2T_train, acc_R2T_test, loss_R2T_train, loss_R2T_test)
        )

        print('---  TASK2 T2R  ---')
        print(
            "EPOCH = %d \n acc_similarity_train = %.3f \n acc_similarity_test = %.3f \n loss_similarity_train = %.3f \n loss_similarity_test = %.3f \n" %
            (epoch + 1, acc_T2R_train, acc_T2R_test, loss_T2R_train, loss_T2R_test)
        )

        print('---  TASK3 Detection  ---')
        print(
            "EPOCH = %d \n acc_detection_train = %.3f \n acc_detection_test = %.3f \n  best_acc = %.3f \n loss_detection_train = %.3f \n loss_detection_test = %.3f \n" %
            (epoch + 1, acc_detection_train, acc_detection_test, best_acc, loss_detection_train, loss_detection_test)
        )

        print('---  TASK1 Similarity Confusion Matrix  ---')
        print('{}\n'.format(cm_similarity))

        print('---  TASK3 Detection Confusion Matrix  ---')
        print('{}\n'.format(cm_detection))

    # 绘制损失曲线图
    epochs = range(1, num_epoch + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss_R2T_list, label='R2T Loss')
    plt.plot(epochs, loss_T2R_list, label='T2R Loss')
    plt.plot(epochs, loss_reason_list, label='Reason Loss')
    plt.plot(epochs, loss_detection_list, label='Detection Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.show()


def test(bert,bert2,Reason_usefulness, attention,R2T_usefulness, detection_module,T2R_usefulness, aggregator,test_dataloader,epoch,avg_loss):
    bert.eval()
    bert2.eval()
    Reason_usefulness.eval()
    detection_module.eval()
    R2T_usefulness.eval()
    T2R_usefulness.eval()
    attention.eval()
    aggregator.eval()

    # device = torch.device(DEVICE)
    loss_func_detection = torch.nn.CrossEntropyLoss()
    loss_func_similarity = torch.nn.CosineEmbeddingLoss()

    R2T_count = 0
    T2R_count = 0
    detection_count = 0
    loss_R2T_total = 0
    loss_T2R_total = 0
    loss_detection_total = 0
    R2Tlabel_all = []
    T2R_label_all = []
    detection_label_all = []
    R2T_pre_label_all = []
    T2R_pre_label_all = []
    detection_pre_label_all = []

    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for batch in test_dataloader:
            news_content = batch['content'].to(device)
            pos =  batch["pos_reason"].to(device)
            neg = batch['neg_reason'].to(device)
            label = batch['label'].to(device)
            content = bert(news_content)
            positive = bert2(pos)
            negative = bert2(neg)

            # --- cross attention ---
            pos_reason2text, pos_text2reason, positive = attention(content, positive)
            neg_reason2text, neg_text2reason, negative = attention(content, negative)

            # --- task1 R2T  loss ---
            text_aligned_match1, R2T_match, pred_R2T_match = R2T_usefulness(content, pos_reason2text)
            text_aligned_unmatch1, R2T_unmatch, pred_R2T_unmatch = R2T_usefulness(content, neg_reason2text)
            R2T_pred = torch.cat([pred_R2T_match.argmax(1), pred_R2T_unmatch.argmax(1)], dim=0)
            R2Tlabel_0 = torch.cat([torch.ones(pred_R2T_match.shape[0]), torch.zeros(pred_R2T_unmatch.shape[0])], dim=0).to(device)
            R2Tlabel_1 = torch.cat([torch.ones(pred_R2T_match.shape[0]), -1 * torch.ones(pred_R2T_unmatch.shape[0])], dim=0).to(device)
            text_aligned_4_task1 = torch.cat([text_aligned_match1, text_aligned_unmatch1], dim=0)
            R2T_aligned_4_task1 = torch.cat([R2T_match, R2T_unmatch], dim=0)
            loss_R2T = loss_func_similarity(text_aligned_4_task1, R2T_aligned_4_task1, R2Tlabel_1)

            # --- task2 T2R loss ---
            text_aligned_match2, T2R_match, pred_T2R_match = T2R_usefulness(content, pos_text2reason)
            text_aligned_unmatch2, T2R_unmatch, pred_T2R_unmatch = T2R_usefulness(content, neg_text2reason)
            T2R_pred = torch.cat([pred_T2R_match.argmax(1), pred_T2R_unmatch.argmax(1)], dim=0)
            T2R_label_0 = torch.cat([torch.ones(pred_T2R_match.shape[0]), torch.zeros(pred_T2R_unmatch.shape[0])], dim=0).to(device)
            T2R_label_1 = torch.cat([torch.ones(pred_T2R_match.shape[0]), -1 * torch.ones(pred_T2R_unmatch.shape[0])], dim=0).to(device)
            text_aligned_4_task2 = torch.cat([text_aligned_match2, text_aligned_unmatch2], dim=0)
            T2R_aligned_4_task2 = torch.cat([T2R_match, T2R_unmatch], dim=0)
            loss_T2R = loss_func_similarity(text_aligned_4_task2, T2R_aligned_4_task2, T2R_label_1)

            # --- task3 Reason loss ---
            text_aligned_match3, reason_match, pred_reason_match = Reason_usefulness(content, positive)
            text_aligned_unmatch3, reason_unmatch, pred_reason_unmatch = Reason_usefulness(content, negative)
            reason_pred = torch.cat([pred_reason_match.argmax(1), pred_reason_unmatch.argmax(1)], dim=0)
            reason_label_0 = torch.cat([torch.ones(pred_reason_match.shape[0]), torch.zeros(pred_reason_unmatch.shape[0])], dim=0).to(device)
            reason_label_1 = torch.cat([torch.ones(pred_reason_match.shape[0]), -1 * torch.ones(pred_reason_unmatch.shape[0])], dim=0).to(device)
            text_aligned_4_task3 = torch.cat([text_aligned_match3, text_aligned_unmatch3], dim=0)
            reason_aligned_4_task3 = torch.cat([reason_match, reason_unmatch], dim=0)
            loss_reason = loss_func_similarity(text_aligned_4_task3, reason_aligned_4_task3, reason_label_1)

            # --- TASK fake news Detection ---
            text_R2T_aligned, R2T_aligned, _ = R2T_usefulness(content, pos_reason2text)
            text_T2R_aligned, T2R_aligned, _ = T2R_usefulness(content, pos_text2reason)
            text_R_aligned, R_aligned, _ = Reason_usefulness(content, positive)
            final_feature = aggregator(content,R2T_aligned,T2R_aligned,R_aligned)
            pre_detection = detection_module(final_feature)
            # print(pre_detection)
            loss_detection = loss_func_detection(pre_detection, label)
            pre_label_detection = pre_detection.argmax(1)
            # print(pre_label_detection)
            # ---  Record  ---
            all_outputs.append(final_feature.detach().cpu().numpy())
            all_labels.append(label.detach().cpu().numpy())

            loss_R2T_total += loss_R2T.item() * (2 * content.shape[0])
            loss_T2R_total += loss_T2R.item() * (2 * content.shape[0])
            loss_detection_total += loss_detection.item() * content.shape[0]
            R2T_count += (content.shape[0] * 2)
            T2R_count += (content.shape[0] * 2)
            detection_count += content.shape[0]

            R2T_pre_label_all.append(R2T_pred.detach().cpu().numpy())
            T2R_pre_label_all.append(T2R_pred.detach().cpu().numpy())
            detection_pre_label_all.append(pre_label_detection.detach().cpu().numpy())
            # print('-=======================-')
            # print(detection_pre_label_all)
            R2Tlabel_all.append(R2Tlabel_0.detach().cpu().numpy())
            T2R_label_all.append(T2R_label_0.detach().cpu().numpy())
            detection_label_all.append(label.detach().cpu().numpy())

        test_labels = np.concatenate(all_labels,axis=0)
        test_outputs = np.concatenate(all_outputs,axis=0)

        loss_R2T_test = loss_R2T_total / R2T_count
        loss_T2R_test = loss_T2R_total / T2R_count
        loss_detection_test = loss_detection_total / detection_count

        R2T_pre_label_all = np.concatenate(R2T_pre_label_all, 0)
        T2R_pre_label_all = np.concatenate(T2R_pre_label_all, 0)
        R2Tlabel_all = np.concatenate(R2Tlabel_all, 0)
        detection_pre_label_all = np.concatenate(detection_pre_label_all, 0)

        T2R_label_all = np.concatenate(T2R_label_all, 0)
        detection_label_all = np.concatenate(detection_label_all, 0)

        acc_R2T_test = accuracy_score(R2T_pre_label_all, R2Tlabel_all)
        acc_T2R_test = accuracy_score(T2R_pre_label_all, T2R_label_all)
        acc_detection_test = accuracy_score(detection_pre_label_all, detection_label_all)
        cm_similarity = confusion_matrix(R2T_pre_label_all, R2Tlabel_all)
        cm_detection = confusion_matrix(detection_pre_label_all, detection_label_all)
        acc_detection_test = accuracy_score(detection_pre_label_all, detection_label_all)
        print(f"Test Accuracy: {acc_detection_test}")

        # 计算真假标签的精确度、召回率、F1分数
        precision_true = precision_score(detection_label_all, detection_pre_label_all, pos_label=1)
        recall_true = recall_score(detection_label_all, detection_pre_label_all, pos_label=1)
        f1_true = f1_score(detection_label_all, detection_pre_label_all, pos_label=1)

        precision_false = precision_score(detection_label_all, detection_pre_label_all, pos_label=0)
        recall_false = recall_score(detection_label_all, detection_pre_label_all, pos_label=0)
        f1_false = f1_score(detection_label_all, detection_pre_label_all, pos_label=0)

        print(f"Precision (True): {precision_true}")
        print(f"Recall (True): {recall_true}")
        print(f"F1 Score (True): {f1_true}")

        print(f"Precision (False): {precision_false}")
        print(f"Recall (False): {recall_false}")
        print(f"F1 Score (False): {f1_false}")

        # 计算宏观F1与微观F1
        macro_f1 = f1_score(detection_label_all, detection_pre_label_all, average='macro')
        micro_f1 = f1_score(detection_label_all, detection_pre_label_all, average='micro')

        print(f"Macro F1 Score: {macro_f1}")
        print(f"Micro F1 Score: {micro_f1}")

        # 计算AUC
        auc_score = roc_auc_score(detection_label_all, detection_pre_label_all)
        print(f"AUC Score: {auc_score}")

        # 生成分类报告
        report = classification_report(detection_label_all, detection_pre_label_all)
        print(f"Classification Report:\n{report}")
        print(metrics.classification_report(detection_label_all, detection_pre_label_all, digits=4))

        with open(csv_file,mode='a',newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch,avg_loss,acc_detection_test,macro_f1,micro_f1,f1_false,precision_false,recall_false,f1_true,precision_true,recall_true])


    return acc_R2T_test, acc_detection_test, loss_R2T_test, loss_detection_test, cm_similarity, cm_detection,loss_T2R_test,acc_T2R_test,test_outputs,test_labels


if __name__ == "__main__":
    train()

