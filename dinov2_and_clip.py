# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 00:28:46 2024

@author: s196222
"""

import random
import numpy as np

import torch
from transformers import AutoImageProcessor, AutoModel, AutoProcessor, CLIPModel
from PIL import Image
import torch.nn as nn
import os
import matplotlib.pyplot as plt

MAX_NUM = 200
MODEL_NAME = 'dinov2-base'
#MODEL_NAME = 'clip-vit-base-patch32'

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
if MODEL_NAME == 'dinov2-base':
    processor = AutoImageProcessor.from_pretrained('./dinov2-base')
    model = AutoModel.from_pretrained('./dinov2-base').to(device)
else:
    processor = AutoProcessor.from_pretrained("./clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("./clip-vit-base-patch32").to(device)


file_loaders = os.listdir('data')

plot_data_couple = []
plot_data_not = []
labels = []  # 存储每对图像的实际标签
for i in range(MAX_NUM):
    file_loader = random.choices(file_loaders, k=2)
    file_loader_path_1 = os.path.join('data', file_loader[0])
    file_loader_path_2 = os.path.join('data', file_loader[1])
    images_path = os.listdir(file_loader_path_1)
    images_path_2 = os.listdir(file_loader_path_2)
    image1 = Image.open(os.path.join(file_loader_path_1, images_path[0]))
    with torch.no_grad():
        inputs1 = processor(images=image1, return_tensors="pt").to(device)
        if MODEL_NAME == 'dinov2-base':
            outputs1 = model(**inputs1)
            image_features1 = outputs1.last_hidden_state
            image_features1 = image_features1.mean(dim=1)
        else:
            image_features1 = model.get_image_features(**inputs1)

    image2 = Image.open(os.path.join(file_loader_path_1, images_path[1]))
    with torch.no_grad():
        inputs2 = processor(images=image2, return_tensors="pt").to(device)
        if MODEL_NAME == 'dinov2-base':
            outputs2 = model(**inputs2)
            image_features2 = outputs2.last_hidden_state
            image_features2 = image_features2.mean(dim=1)
        else:
            image_features2 = model.get_image_features(**inputs2)

    image3 = Image.open(os.path.join(file_loader_path_2, images_path_2[0]))
    with torch.no_grad():
        inputs3 = processor(images=image3, return_tensors="pt").to(device)
        if MODEL_NAME == 'dinov2-base':
            outputs3 = model(**inputs3)
            image_features3 = outputs3.last_hidden_state
            image_features3 = image_features3.mean(dim=1)
        else:
            image_features3 = model.get_image_features(**inputs3)

    cos = nn.CosineSimilarity(dim=0)
    sim = cos(image_features1[0], image_features2[0]).item()
    sim = (sim + 1) / 2
    plot_data_couple.append(sim)
    labels.append(1)  # 一对图像的标签为1

    cos = nn.CosineSimilarity(dim=0)
    sim = cos(image_features1[0], image_features3[0]).item()
    sim = (sim + 1) / 2
    plot_data_not.append(sim)
    labels.append(0)  # 非一对图像的标签为0

fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')

# 绘制两个直方图
n1, bins1, patches1 = ax.hist(plot_data_couple, bins=50, density=True, alpha=0.5, color='blue')
n2, bins2, patches2 = ax.hist(plot_data_not, bins=50, density=True, alpha=0.5, color='green')

# 设置坐标轴范围和标签
ax.set_xlim([0, 1])
ax.set_xlabel('Value')
ax.set_ylabel('Frequency')

# 添加标题和图例
plt.title(MODEL_NAME)
ax.legend(['couple', 'not couple'])

# 设置横坐标刻度更详细
plt.xticks(np.arange(0, 1.1, 0.1))

# 显示图形
plt.show()

#dinov2
threshold = 0.85
correct_predictions = 0
total_predictions = MAX_NUM * 2  # 每次循环都会处理一对图像

for score_couple, score_not, label in zip(plot_data_couple, plot_data_not, labels):
    if score_couple >= threshold and label == 1:
        correct_predictions += 1
    if score_not < threshold and label == 0:
        correct_predictions += 1

accuracy = correct_predictions / total_predictions
print("模型的准确率为:", accuracy)

#clip
threshold = 0.9
correct_predictions = 0
total_predictions = MAX_NUM * 2  # 每次循环都会处理一对图像

for score_couple, score_not, label in zip(plot_data_couple, plot_data_not, labels):
    if score_couple >= threshold and label == 1:
        correct_predictions += 1
    if score_not < threshold and label == 0:
        correct_predictions += 1

accuracy = correct_predictions / total_predictions
print("模型的准确率为:", accuracy)

