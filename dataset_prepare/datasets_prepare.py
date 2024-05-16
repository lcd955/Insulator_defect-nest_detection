import os
import shutil
import numpy as np
import re
import random

#设置随机种子，保证每次分割的数据集是确定的
random.seed(0)
np.random.seed(0)
os.environ['PYTHONHASHSEED'] = '0'

# 原始数据集的文件夹路径
img_dir ="/root/autodl-tmp/merged_insulator_data_new/img"    #"./data/images"
label_dir ='/root/autodl-tmp/merged_insulator_data_new/labels'       #"./data/labels"


# 目标文件夹路径
dataset_dir ="/root/autodl-tmp/datasets_0512"   #"./dataset"
img_train_dir = os.path.join(dataset_dir, "images/train")
img_val_dir = os.path.join(dataset_dir, "images/val")
label_train_dir = os.path.join(dataset_dir, "labels/train")
label_val_dir = os.path.join(dataset_dir, "labels/val")

# 创建需要的文件夹
os.makedirs(img_train_dir, exist_ok=True)
os.makedirs(img_val_dir, exist_ok=True)
os.makedirs(label_train_dir, exist_ok=True)
os.makedirs(label_val_dir, exist_ok=True)

# 获取所有图像文件
img_files = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]

# 随机洗牌
np.random.shuffle(img_files)

# 计算训练集和验证集的划分点
split_idx = int(len(img_files) * 0.7)

# 将文件分割为训练集和验证集
train_files = img_files[:split_idx]
val_files = img_files[split_idx:]

#匹配模块，确保数据集里如果有.jpg,.JPG,JPEG,jpeg也能有良好的表现
for f in train_files:
    shutil.copy(os.path.join(img_dir, f), os.path.join(img_train_dir, f))
    shutil.copy(os.path.join(label_dir, re.sub(r'.[jJ][pP][gG]|.[jJ][pP][eE][gG]$', '.txt', f)), os.path.join(label_train_dir, re.sub(r'.[jJ][pP][gG]|.[jJ][pP][eE][gG]$', '.txt', f)))

for f in val_files:
    shutil.copy(os.path.join(img_dir, f), os.path.join(img_val_dir, f))
    shutil.copy(os.path.join(label_dir, re.sub(r'.[jJ][pP][gG]|.[jJ][pP][eE][gG]$', '.txt', f)), os.path.join(label_val_dir, re.sub(r'.[jJ][pP][gG]|.[jJ][pP][eE][gG]$', '.txt', f)))
