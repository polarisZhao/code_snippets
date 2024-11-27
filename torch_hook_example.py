import torch
import torch.nn as nn

from datasets import load_dataset
import functools
from collections import defaultdict

from functools import partial
import numpy as np
from tqdm import tqdm

def get_act_scales(model, tokenizer, dataset_path, num_samples=512, seq_len=512):
    """
    逐通道计算模型中每个线性层的激活最大值（绝对值）的尺度，用于后续的量化或归一化处理。

    参数:
        model (torch.nn.Module): 要分析的PyTorch模型。
        tokenizer (PreTrainedTokenizer): 用于将文本转换为模型输入的分词器。
        dataset_path (str): 数据集的路径，数据集应为JSON格式。
        num_samples (int): 要处理的样本数量，默认512。
        seq_len (int): 输入序列的最大长度，默认512。

    返回:
        act_scales (dict): 包含每个线性层激活最大值的字典，键为层名称，值为对应的最大值张量。
    """
    model.eval()  # 将模型设置为评估模式，禁用dropout等训练特有的操作
    device = next(model.parameters()).device  # 获取模型所在的设备（CPU或GPU）
    act_scales = {}  # 初始化一个字典，用于存储每个层的激活尺度

    def stat_tensor(name, tensor):
        """
        计算张量的绝对值最大值并更新act_scales字典。

        参数:
            name (str): 层的名称。
            tensor (torch.Tensor): 要统计的张量。
        """
        hidden_dim = tensor.shape[-1]  # 获取张量的最后一个维度大小（通常是隐藏维度）
        tensor = tensor.view(-1, hidden_dim).abs().detach()  # 将张量展平并取绝对值
        comming_max = torch.max(tensor, dim=0)[0].float().cpu()  # 计算每个隐藏维度上的最大值并转移到 CPU
        if name in act_scales:
            act_scales[name] = torch.max(act_scales[name], comming_max)  # 更新最大值
        else:
            act_scales[name] = comming_max  # 初始化最大值

    def stat_input_hook(m, x, y, name):
        """
        前向钩子函数，用于在每次前向传播时收集输入张量的统计信息。

        参数:
            m (torch.nn.Module): 当前层的模块。
            x (tuple): 输入到当前层的张量。
            y (torch.Tensor): 当前层的输出张量。
            name (str): 当前层的名称。
        """
        if isinstance(x, tuple):
            x = x[0]  # 如果输入是元组，则取第一个元素
        stat_tensor(name, x)  # 统计输入张量

    hooks = []  # 存储钩子函数，以便后续移除
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            # 对每个线性层注册一个前向钩子，用于统计激活
            hooks.append(
                m.register_forward_hook(functools.partial(stat_input_hook, name=name))
            )

    # 加载数据集，假设数据集为JSON格式，包含"text"字段
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = dataset.shuffle(seed=42)  # 随机打乱数据集

    # 遍历指定数量的样本，进行前向传播以收集激活信息
    for i in tqdm(range(num_samples), desc="Collecting activation scales"):
        input_ids = tokenizer(
            dataset[i]["text"],  # 获取第i个样本的文本
            return_tensors="pt",  # 返回PyTorch张量
            max_length=seq_len,  # 设置最大序列长度
            truncation=True  # 启用截断
        ).input_ids.to(device)  # 转移到模型所在的设备
        model(input_ids)  # 前向传播

    # 移除所有注册的钩子
    for h in hooks:
        h.remove()

    return act_scales  # 返回收集到的激活尺度
