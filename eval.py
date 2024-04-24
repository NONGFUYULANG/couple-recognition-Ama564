import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, ViTModel
import torch


import torch
import torch.nn as nn
import numpy as np


class dot_attention(nn.Module):
    """ 点积注意力机制"""

    def __init__(self, attention_dropout=0.0):
        super(dot_attention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        """
        前向传播
        :param q:
        :param k:
        :param v:
        :param scale:
        :param attn_mask:
        :return: 上下文张量和attention张量。
        """
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention = attention * scale        # 是否设置缩放
        if attn_mask:
            attention = attention.masked_fill(attn_mask, -np.inf)     # 给需要mask的地方设置一个负无穷。
        # 计算softmax
        attention = self.softmax(attention)
        # 添加dropout
        attention = self.dropout(attention)
        # 和v做点积。
        context = torch.bmm(attention, v)
        return context, attention
class MultiHeadAttention(nn.Module):
    """ 多头自注意力"""
    def __init__(self, model_dim=1024, num_heads=4, dropout=0.0):
        super(MultiHeadAttention, self).__init__()

        self.dim_per_head = model_dim//num_heads   # 每个头的维度
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.dot_product_attention = dot_attention(dropout)

        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)         # LayerNorm 归一化。

    def forward(self, key, value, query, attn_mask=None):
        residual = query
        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)

        # 线性映射。
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        # 按照头进行分割
        key = key.view(batch_size * num_heads, -1, dim_per_head)
        value = value.view(batch_size * num_heads, -1, dim_per_head)
        query = query.view(batch_size * num_heads, -1, dim_per_head)

        if attn_mask:
            attn_mask = attn_mask.repeat(num_heads, 1, 1)

        # 缩放点击注意力机制
        scale = (key.size(-1) // num_heads) ** -0.5
        context, attention = self.dot_product_attention(query, key, value, scale, attn_mask)

        # 进行头合并 concat heads
        context = context.view(batch_size, -1, dim_per_head * num_heads)

        # 进行线性映射
        output = self.linear_final(context)

        # dropout
        output = self.dropout(output)

        # 添加残差层和正则化层。
        output = self.layer_norm(residual + output)

        return output, attention


# q = torch.ones((1, 17, 400))
# k = torch.ones((1, 17, 400))
# v = k
# mutil_head_attention = MultiHeadAttention()
# output, attention = mutil_head_attention(q, k, v)
# print("context:", output.shape)
# print("attention:", attention.size(), attention)
import torch.nn as nn
import torch.nn.functional as F
from moe import SparseMoE
class VIT_Couple(nn.Module):
    def __init__(self, pretrained=False):
        super(VIT_Couple, self).__init__()
        self.model=ViTModel.from_pretrained("google/vit-large-patch16-224-in21k")
        for param in self.model.parameters():
            param.requires_grad = False
        self.fc2 = nn.Linear(512, 100)
        self.fc3 = nn.Linear(100, 2)
        self.fc1 = nn.Linear(1024, 512)
        self.sparse_moe = SparseMoE(512, 3, 2)
        self.dropout = nn.Dropout(p=0.2)
        self.multi_head_attention= MultiHeadAttention()

    def forward(self,x,y):
        x=x.view(x.shape[0],x.shape[2],x.shape[3],x.shape[4])#3,3,224,224
        y=y.view(y.shape[0],y.shape[2],y.shape[3],y.shape[4])
        x = self.model(x).pooler_output#1, 197, 768
        y = self.model(y).pooler_output#1, 197, 768
        x=x.view(x.shape[0],1,x.shape[1])
        y=y.view(y.shape[0],1,y.shape[1])
        pos=torch.concat((x,y),dim=1)# 1 2 768
        pos, attention = self.multi_head_attention(pos, pos, pos)# 1 2 768
        x=x.view(pos.shape[0],-1)
        pos=self.fc1(x)
        # pos=self.sparse_moe(pos)
        pos = self.dropout(pos)
        pos = F.relu(pos)
        pos=self.fc2(pos)
        pos = F.relu(pos)
        pos=self.fc3(pos)
        return pos
new_model = VIT_Couple().to('cpu') # we do not specify ``weights``, i.e. create untrained model
new_model.load_state_dict(torch.load('version_test4.pth'))
pic1='A _0_.jpg'
pic2='A _1_.jpg'
img = Image.open(pic1).convert("RGB")
img1 = Image.open(pic2).convert("RGB")
image_processor = AutoImageProcessor.from_pretrained("google/vit-large-patch16-224-in21k")
x =image_processor(img, return_tensors="pt")['pixel_values']
y =image_processor(img1, return_tensors="pt")['pixel_values']
print(x.shape)
with torch.no_grad():
    x=x.view(1,1,3,224,224)
    y = y.view(1, 1, 3, 224, 224)
    x=x.to('cpu')
    print(x.shape)
    y=y.to('cpu')
    new_model=new_model.to('cpu')
    outputs = new_model(x, y)
    print(outputs)
    _, predicted = torch.max(outputs.data, 1)
    print(predicted)


#1 代表是情侣头像， 0 代表不是