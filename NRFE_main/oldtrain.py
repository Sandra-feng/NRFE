import copy
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm
from dataset import FeatureDataset
from model import Similarity, DetectionModule,Attention_Encoder,Reason_Similarity,Aggregator
from sklearn import metrics
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import re
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
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

# 加载 .npz 文件
data = np.load("/twitter15.npz")
# 假设标签数组的键名为 'labels'
labels = data['labels']

# 计算标签的分布
unique, counts = np.unique(labels, return_counts=True)
label_counts = dict(zip(unique, counts))

# 打印标签分布
for label, count in label_counts.items():
    print(f'Label: {label}, Count: {count}')
# data2 = np.load("test_data.npz")
# 将numpy数组转换为torch Tensors
# encoded_content = torch.tensor(data['encoded_content']).float()  # 确保数据类型适用于PyTorch处理
content = torch.tensor(data['encoded_content']).float()
positive = torch.tensor(data['positive']).float()
negative = torch.tensor(data['negative']).float()
labels = torch.tensor(data['labels']).long()  # 对于分类任务，确保标签是长整型

# 创建 TensorDataset
dataset = TensorDataset(content, positive,negative,labels)

total_length = len(dataset)
train_size = int(0.9 * total_length)
test_size = total_length - train_size

# Randomly split dataset into training and testing dataset
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
# 创建 DataLoader
val_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=10, shuffle=False) 
# val_dataloader = DataLoader(dataset, batch_size=10, shuffle=True)  # shuffle=True 表示在每个epoch随机打乱数据

# 关闭 .npz 文件
data.close()

def train():
    # ---  Load Config  ---
    device = torch.device(DEVICE)

    lr = LR
    l2 = L2
    num_epoch = NUM_EPOCH
    

    # ---  Build Model & Trainer  ---
    attention = Attention_Encoder()
    attention.to(device)
    optim_task_attention = torch.optim.Adam(
        attention.parameters(), lr=lr, weight_decay=l2
    )  

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
    acc_best_train = 0
    # 定义保存路径
    save_path = '/best_teachermodel.pth'
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

        for content, positive, negative, label in val_dataloader:
            content = content.squeeze(1).to(device)
            positive = positive.squeeze(1).to(device)
            negative = negative.squeeze(1).to(device)
            label = label.to(device)
            # print('content',content.shape)

            # --- cross attention ---
            content, pos_reason2text, pos_text2reason, neg_reason2text, neg_text2reason = attention(content, positive, negative)

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
            final_feature = aggregator(content, R2T_aligned, T2R_aligned, R_aligned)
            pre_detection = detection_module(final_feature)
            loss_detection = loss_func_detection(pre_detection, label)

            # 梯度清零
            optim_task_attention.zero_grad()
            optim_task_R2T.zero_grad()
            optim_task_T2R.zero_grad()
            optim_task_reason.zero_grad()
            optim_task_aggregator.zero_grad()
            optim_task_detection.zero_grad()

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

        #训练结束，保存模型参数
        print(detection_module)
        if acc_detection_train > acc_best_train:
            acc_best_train = acc_detection_train
            # 保存模型和优化器的状态字典
            state_dicts = {
                'attention': attention.state_dict(),
                'R2T_usefulness': R2T_usefulness.state_dict(),
                'T2R_usefulness': T2R_usefulness.state_dict(),
                'Reason_usefulness': Reason_usefulness.state_dict(),
                'aggregator':aggregator.state_dict(),
                'detection_module': detection_module.state_dict(),
            }
            torch.save(state_dicts, save_path)

            print(f'Best model parameters saved to {save_path}')


        
        # ---  Test  ---

        acc_R2T_test, acc_detection_test, loss_R2T_test, loss_detection_test, cm_similarity, cm_detection,loss_T2R_test,acc_T2R_test = test(Reason_usefulness, attention,R2T_usefulness, detection_module,T2R_usefulness, aggregator,test_dataloader)
        best_acc = max(best_acc,acc_detection_test)
        # ---  Output  ---

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


def test(Reason_usefulness, attention,R2T_usefulness, detection_module,T2R_usefulness, aggregator,test_dataloader):
    Reason_usefulness.eval()
    detection_module.eval()
    R2T_usefulness.eval()
    T2R_usefulness.eval()
    attention.eval()
    aggregator.eval()

    device = torch.device(DEVICE)
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

    with torch.no_grad():
        for content, positive, negative, label in test_dataloader:

            content = content.squeeze(1).to(device)
            positive = positive.squeeze(1).to(device)
            negative = negative.squeeze(1).to(device)
            label = label.to(device)
            # --- cross attention ---
            content, pos_reason2text, pos_text2reason, neg_reason2text, neg_text2reason = attention(content, positive, negative)

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
            final_feature = aggregator(content, R2T_aligned, T2R_aligned, R_aligned)
            pre_detection = detection_module(final_feature)
            # print(pre_detection)
            loss_detection = loss_func_detection(pre_detection, label)
            pre_label_detection = pre_detection.argmax(1)
            print(pre_label_detection.type())
            # ---  Record  ---

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
            print(detection_pre_label_all)
            R2Tlabel_all.append(R2Tlabel_0.detach().cpu().numpy())
            T2R_label_all.append(T2R_label_0.detach().cpu().numpy())
            detection_label_all.append(label.detach().cpu().numpy())

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
        print(metrics.classification_report(detection_label_all, detection_pre_label_all, digits=4))

    return acc_R2T_test, acc_detection_test, loss_R2T_test, loss_detection_test, cm_similarity, cm_detection,loss_T2R_test,acc_T2R_test


if __name__ == "__main__":
    train()

