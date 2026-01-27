---
title: AI 学习
description: AI 学习
tags: [AI]
category: AI
licenseName: "Unlicensed"
author: panxiao
sourceLink: "https://github.com/px6707/myblog"
draft: false
---

# AI 理论与开发完整学习路线

> 从零基础到 AI 工程师的系统化学习指南
> 
> 总时长：8-13 个月（全职）或 1.5-2 年（业余）

---

## 📋 目录

- [学习路线总览](#学习路线总览)
- [技能树](#技能树)
- [阶段 0：基础准备](#阶段-0基础准备1-2-个月)
- [阶段 1：AI 理论基础](#阶段-1ai-理论基础2-3-个月)
- [阶段 2：深度学习实践](#阶段-2深度学习实践2-3-个月)
- [阶段 3：大模型应用](#阶段-3大模型应用2-3-个月)
- [阶段 4：AI 工程化](#阶段-4ai-工程化1-2-个月)
- [阶段 5：高级专题](#阶段-5高级专题持续学习)
- [学习时间规划](#学习时间规划)
- [检验里程碑](#检验里程碑)
- [推荐资源汇总](#推荐资源汇总)

---

## 学习路线总览

```
阶段 0：基础准备（1-2 个月）
  ↓
阶段 1：AI 理论基础（2-3 个月）
  ↓
阶段 2：深度学习实践（2-3 个月）
  ↓
阶段 3：大模型应用（2-3 个月）
  ↓
阶段 4：AI 工程化（1-2 个月）
  ↓
阶段 5：高级专题（持续学习）
```

---

## 技能树

```
AI 开发者技能树
├── 🌳 基础技能（必修）
│   ├── Python 编程 ⭐⭐⭐⭐⭐
│   ├── 数学基础 ⭐⭐⭐⭐
│   ├── Linux 操作 ⭐⭐⭐
│   └── Git 版本控制 ⭐⭐⭐
│
├── 🧠 AI 理论（核心）
│   ├── 机器学习基础 ⭐⭐⭐⭐⭐
│   ├── 深度学习原理 ⭐⭐⭐⭐⭐
│   ├── 自然语言处理 ⭐⭐⭐⭐⭐
│   └── Transformer 架构 ⭐⭐⭐⭐⭐
│
├── 🛠️ 开发框架（实践）
│   ├── PyTorch ⭐⭐⭐⭐⭐
│   ├── Transformers ⭐⭐⭐⭐⭐
│   ├── LangChain ⭐⭐⭐⭐⭐
│   └── LlamaIndex ⭐⭐⭐⭐
│
├── 🚀 大模型技术（应用）
│   ├── Prompt Engineering ⭐⭐⭐⭐⭐
│   ├── RAG 检索增强 ⭐⭐⭐⭐⭐
│   ├── 模型微调 ⭐⭐⭐⭐
│   └── Agent 开发 ⭐⭐⭐⭐⭐
│
└── 🏗️ 工程能力（进阶）
    ├── 模型部署 ⭐⭐⭐⭐
    ├── 性能优化 ⭐⭐⭐
    ├── 系统设计 ⭐⭐⭐⭐
    └── MLOps ⭐⭐⭐
```

---

## 阶段 0：基础准备（1-2 个月）

### 技能点 1：Python 编程 ⭐⭐⭐⭐⭐

#### 必修内容

**基础语法**
- 数据类型（list, dict, set, tuple）
- 控制流（if, for, while）
- 函数和类
- 异常处理

**进阶特性**
- 装饰器
- 生成器
- 上下文管理器
- 异步编程（async/await）

**常用库**
- NumPy（数组计算）
- Pandas（数据处理）
- Matplotlib（可视化）

#### 学习资源

| 资源 | 类型 | 语言 | 质量 | 链接 |
|------|------|------|------|------|
| 廖雪峰 Python 教程 | 在线教程 | 中文 | ⭐⭐⭐⭐⭐ | https://www.liaoxuefeng.com/wiki/1016959663602400 |
| 菜鸟教程 Python | 在线教程 | 中文 | ⭐⭐⭐⭐ | https://www.runoob.com/python3/python3-tutorial.html |
| Python Crash Course | 书籍 | 英文 | ⭐⭐⭐⭐⭐ | https://nostarch.com/pythoncrashcourse2e |
| 莫烦 Python | 视频 | 中文 | ⭐⭐⭐⭐ | https://mofanpy.com/ |
| Real Python | 在线教程 | 英文 | ⭐⭐⭐⭐⭐ | https://realpython.com/ |

#### 实践平台
- LeetCode：https://leetcode.cn/
- 牛客网：https://www.nowcoder.com/

#### 检验标准
- ✅ 能独立写 500 行以上的项目
- ✅ 理解面向对象编程
- ✅ 熟练使用 NumPy 和 Pandas

---

### 技能点 2：数学基础 ⭐⭐⭐⭐

#### 必修内容

**线性代数**
- 向量和矩阵运算
- 矩阵乘法
- 特征值和特征向量
- **为什么重要**：神经网络就是矩阵运算

**微积分**
- 导数和梯度
- 链式法则
- 偏导数
- **为什么重要**：反向传播的数学基础

**概率统计**
- 概率分布
- 期望和方差
- 贝叶斯定理
- **为什么重要**：理解模型的不确定性

#### 学习资源

| 资源 | 类型 | 语言 | 质量 | 链接 |
|------|------|------|------|------|
| 3Blue1Brown 线性代数本质 | 视频 | 中英字幕 | ⭐⭐⭐⭐⭐ | https://www.bilibili.com/video/BV1ys411472E |
| 3Blue1Brown 微积分本质 | 视频 | 中英字幕 | ⭐⭐⭐⭐⭐ | https://www.bilibili.com/video/BV1qW411N7FU |
| 深度学习的数学 | 书籍 | 中文 | ⭐⭐⭐⭐ | - |
| Khan Academy | 在线课程 | 英文 | ⭐⭐⭐⭐⭐ | https://www.khanacademy.org/ |
| 麻省理工线性代数公开课 | 视频 | 中英字幕 | ⭐⭐⭐⭐⭐ | https://www.bilibili.com/video/BV1zx411g7gq |

#### 检验标准
- ✅ 理解矩阵乘法的几何意义
- ✅ 能手算简单的梯度
- ✅ 理解概率分布

---

### 技能点 3：Linux 和 Git ⭐⭐⭐

#### Linux 基础命令

```bash
# 文件操作
cd, ls, mkdir, rm, cp, mv

# 文本处理
cat, grep, awk, sed

# 进程管理
ps, top, kill

# 权限管理
chmod, chown
```

#### Git 基础命令

```bash
git init, clone, add, commit
git push, pull, branch, merge
git log, diff, reset
```

#### 学习资源

| 资源 | 类型 | 语言 | 质量 | 链接 |
|------|------|------|------|------|
| 鸟哥的 Linux 私房菜 | 书籍/网站 | 中文 | ⭐⭐⭐⭐⭐ | http://cn.linux.vbird.org/ |
| Git 官方中文教程 | 在线教程 | 中文 | ⭐⭐⭐⭐⭐ | https://git-scm.com/book/zh/v2 |
| 廖雪峰 Git 教程 | 在线教程 | 中文 | ⭐⭐⭐⭐ | https://www.liaoxuefeng.com/wiki/896043488029600 |
| Linux Journey | 在线教程 | 英文 | ⭐⭐⭐⭐ | https://linuxjourney.com/ |

#### 检验标准
- ✅ 能在 Linux 下流畅工作
- ✅ 能用 Git 管理代码

---

## 阶段 1：AI 理论基础（2-3 个月）

### 技能点 4：机器学习基础 ⭐⭐⭐⭐⭐

#### 核心概念

**监督学习**
- 线性回归
- 逻辑回归
- 决策树
- 随机森林

**无监督学习**
- K-means 聚类
- PCA 降维

**核心概念**
- 损失函数
- 梯度下降
- 过拟合和欠拟合
- 交叉验证

#### 学习资源

| 资源 | 类型 | 语言 | 质量 | 链接 |
|------|------|------|------|------|
| 吴恩达机器学习（网易云课堂） | 视频课程 | 中文字幕 | ⭐⭐⭐⭐⭐ | https://study.163.com/course/introduction/1004570029.htm |
| 李宏毅机器学习 | 视频课程 | 中文 | ⭐⭐⭐⭐⭐ | https://www.bilibili.com/video/BV1Wv411h7kN |
| 机器学习实战 | 书籍 | 中文 | ⭐⭐⭐⭐ | - |
| Scikit-learn 中文文档 | 文档 | 中文 | ⭐⭐⭐⭐⭐ | https://scikit-learn.org.cn/ |
| Andrew Ng - Machine Learning | 课程 | 英文 | ⭐⭐⭐⭐⭐ | https://www.coursera.org/learn/machine-learning |

#### 实践项目
1. 房价预测（线性回归）
2. 垃圾邮件分类（逻辑回归）
3. 手写数字识别（决策树）

#### 检验标准
- ✅ 理解训练/验证/测试集
- ✅ 能用 Scikit-learn 训练模型
- ✅ 理解过拟合和正则化

---

### 技能点 5：深度学习原理 ⭐⭐⭐⭐⭐

#### 核心内容

**神经网络基础**
- 感知机
- 多层感知机（MLP）
- 激活函数（ReLU, Sigmoid, Tanh）
- 前向传播和反向传播

**卷积神经网络（CNN）**
- 卷积层
- 池化层
- 经典架构（LeNet, VGG, ResNet）

**循环神经网络（RNN）**
- LSTM
- GRU
- 序列建模

**优化技巧**
- Batch Normalization
- Dropout
- 学习率调度
- 优化器（SGD, Adam）

#### 学习资源

| 资源 | 类型 | 语言 | 质量 | 链接 |
|------|------|------|------|------|
| 动手学深度学习（李沐） | 视频+书籍 | 中文 | ⭐⭐⭐⭐⭐ | https://zh.d2l.ai/ |
| 吴恩达深度学习专项课程 | 视频课程 | 中文字幕 | ⭐⭐⭐⭐⭐ | https://www.bilibili.com/video/BV1FT4y1E74V |
| 邱锡鹏《神经网络与深度学习》 | 书籍 | 中文 | ⭐⭐⭐⭐⭐ | https://nndl.github.io/ |
| 深度学习（花书） | 书籍 | 中文 | ⭐⭐⭐⭐⭐ | - |
| CS231n 卷积神经网络 | 课程 | 英文 | ⭐⭐⭐⭐⭐ | http://cs231n.stanford.edu/ |

#### 实践项目
1. MNIST 手写数字识别（MLP）
2. CIFAR-10 图像分类（CNN）
3. 文本情感分析（RNN）

#### 检验标准
- ✅ 能从零实现一个神经网络
- ✅ 理解反向传播的数学原理
- ✅ 能用 PyTorch 训练 CNN

---

### 技能点 6：自然语言处理 ⭐⭐⭐⭐⭐

#### 核心内容

**文本预处理**
- 分词（Tokenization）
- 词干提取
- 停用词过滤

**文本表示**
- Bag of Words
- TF-IDF
- Word2Vec
- GloVe

**序列模型**
- RNN for NLP
- Seq2Seq
- Attention 机制

#### 学习资源

| 资源 | 类型 | 语言 | 质量 | 链接 |
|------|------|------|------|------|
| 李宏毅 NLP 课程 | 视频 | 中文 | ⭐⭐⭐⭐⭐ | https://www.bilibili.com/video/BV1Wv411h7kN |
| 自然语言处理实战 | 书籍 | 中文 | ⭐⭐⭐⭐ | - |
| HanLP 自然语言处理 | 工具+教程 | 中文 | ⭐⭐⭐⭐ | https://github.com/hankcs/HanLP |
| CS224N - NLP with Deep Learning | 课程 | 英文 | ⭐⭐⭐⭐⭐ | http://web.stanford.edu/class/cs224n/ |
| Speech and Language Processing | 书籍 | 英文 | ⭐⭐⭐⭐⭐ | https://web.stanford.edu/~jurafsky/slp3/ |

#### 实践项目
1. 文本分类
2. 命名实体识别（NER）
3. 机器翻译

#### 检验标准
- ✅ 理解 Word Embedding
- ✅ 理解 Attention 机制
- ✅ 能训练文本分类模型

---

## 阶段 2：深度学习实践（2-3 个月）

### 技能点 7：PyTorch 深度掌握 ⭐⭐⭐⭐⭐

#### 核心内容

**基础操作**
- Tensor 操作
- 自动求导（autograd）
- 数据加载（DataLoader）

**模型构建**
- nn.Module
- 自定义层
- 模型保存和加载

**训练流程**
- 训练循环
- 验证循环
- 早停（Early Stopping）

**高级特性**
- 混合精度训练
- 分布式训练
- 模型量化

#### 学习资源

| 资源 | 类型 | 语言 | 质量 | 链接 |
|------|------|------|------|------|
| PyTorch 官方中文教程 | 文档 | 中文 | ⭐⭐⭐⭐⭐ | https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html |
| 深度学习框架 PyTorch 入门与实践 | 书籍 | 中文 | ⭐⭐⭐⭐ | - |
| 莫烦 PyTorch 教程 | 视频 | 中文 | ⭐⭐⭐⭐ | https://mofanpy.com/tutorials/machine-learning/torch/ |
| PyTorch 中文文档 | 文档 | 中文 | ⭐⭐⭐⭐⭐ | https://pytorch-cn.readthedocs.io/zh/latest/ |
| PyTorch Lightning | 框架 | 英文 | ⭐⭐⭐⭐ | https://lightning.ai/docs/pytorch/stable/ |

#### 实践项目
1. 从零实现 ResNet
2. 从零实现 LSTM
3. 从零实现 Transformer

#### 检验标准
- ✅ 能快速搭建各种网络结构
- ✅ 理解 PyTorch 的计算图
- ✅ 能调试训练过程

---

### 技能点 8：Transformer 架构 ⭐⭐⭐⭐⭐

#### 核心内容

**Attention 机制**
- Self-Attention
- Multi-Head Attention
- Scaled Dot-Product Attention

**Transformer 结构**
- Encoder
- Decoder
- Position Encoding

**BERT 系列**
- BERT 预训练
- 下游任务微调
- RoBERTa, ALBERT

**GPT 系列**
- GPT 架构
- 因果语言建模
- GPT-2, GPT-3

#### 学习资源

| 资源 | 类型 | 语言 | 质量 | 链接 |
|------|------|------|------|------|
| 李沐精读 Transformer 论文 | 视频 | 中文 | ⭐⭐⭐⭐⭐ | https://www.bilibili.com/video/BV1pu411o7BE |
| The Illustrated Transformer | 博客 | 中文翻译 | ⭐⭐⭐⭐⭐ | https://jalammar.github.io/illustrated-transformer/ |
| Transformer 从零详细实现 | 教程 | 中文 | ⭐⭐⭐⭐ | https://github.com/jadore801120/attention-is-all-you-need-pytorch |
| Attention Is All You Need | 论文 | 英文 | ⭐⭐⭐⭐⭐ | https://arxiv.org/abs/1706.03762 |
| The Annotated Transformer | 代码注释 | 英文 | ⭐⭐⭐⭐⭐ | http://nlp.seas.harvard.edu/annotated-transformer/ |

#### 实践项目
1. 从零实现 Transformer
2. BERT 文本分类
3. GPT 文本生成

#### 检验标准
- ✅ 能手写 Transformer
- ✅ 理解 Self-Attention 的数学原理
- ✅ 理解 BERT 和 GPT 的区别

---

## 阶段 3：大模型应用（2-3 个月）

### 技能点 9：Hugging Face Transformers ⭐⭐⭐⭐⭐

#### 核心内容

**模型使用**
- from_pretrained
- AutoModel, AutoTokenizer
- pipeline 快速推理

**模型微调**
- Trainer API
- 自定义训练循环
- 评估指标

**PEFT 技术**
- LoRA
- Prefix Tuning
- Adapter

#### 学习资源

| 资源 | 类型 | 语言 | 质量 | 链接 |
|------|------|------|------|------|
| Hugging Face 中文教程 | 文档 | 中文 | ⭐⭐⭐⭐⭐ | https://huggingface.co/docs/transformers/zh/index |
| Transformers 快速入门 | 教程 | 中文 | ⭐⭐⭐⭐ | https://transformers.run/ |
| PEFT 官方文档 | 文档 | 英文 | ⭐⭐⭐⭐⭐ | https://huggingface.co/docs/peft |
| Hugging Face Course | 课程 | 英文 | ⭐⭐⭐⭐⭐ | https://huggingface.co/learn/nlp-course |
| LoRA 论文解读 | 视频 | 中文 | ⭐⭐⭐⭐ | https://www.bilibili.com/video/BV1Qh4y1L7Wd |

#### 实践项目
1. 使用 BERT 做文本分类
2. 使用 GPT-2 做文本生成
3. LoRA 微调 LLaMA

#### 检验标准
- ✅ 能快速使用任何预训练模型
- ✅ 能微调模型到自己的任务
- ✅ 理解 LoRA 的原理

---

### 技能点 10：Prompt Engineering ⭐⭐⭐⭐⭐

#### 核心内容

**基础技巧**
- Zero-shot Prompting
- Few-shot Prompting
- Chain-of-Thought

**高级技巧**
- Self-Consistency
- Tree of Thoughts
- ReAct

**提示词设计**
- 角色设定
- 任务描述
- 输出格式
- 示例提供

#### 学习资源

| 资源 | 类型 | 语言 | 质量 | 链接 |
|------|------|------|------|------|
| 提示工程指南（中文） | 文档 | 中文 | ⭐⭐⭐⭐⭐ | https://www.promptingguide.ai/zh |
| LangGPT 结构化提示词 | 教程 | 中文 | ⭐⭐⭐⭐⭐ | https://github.com/yzfly/LangGPT |
| 吴恩达 Prompt Engineering 课程 | 视频 | 中文字幕 | ⭐⭐⭐⭐⭐ | https://www.bilibili.com/video/BV1Bo4y1A7FU |
| OpenAI Prompt Engineering Guide | 文档 | 英文 | ⭐⭐⭐⭐⭐ | https://platform.openai.com/docs/guides/prompt-engineering |
| Learn Prompting | 在线课程 | 英文 | ⭐⭐⭐⭐ | https://learnprompting.org/ |

#### 实践项目
1. 设计客服机器人的提示词
2. 设计代码生成的提示词
3. 设计数据分析的提示词

#### 检验标准
- ✅ 能设计高质量的提示词
- ✅ 理解不同提示技巧的适用场景
- ✅ 能优化提示词提升效果

---

### 技能点 11：RAG 检索增强 ⭐⭐⭐⭐⭐

#### 核心内容

**向量数据库**
- Embedding 原理
- 相似度计算
- Chroma, Qdrant, Pinecone

**检索策略**
- 语义检索
- 混合检索（BM25 + Vector）
- 分层检索

**RAG 优化**
- 文档切分策略
- 重排序（Reranking）
- 查询改写

#### 学习资源

| 资源 | 类型 | 语言 | 质量 | 链接 |
|------|------|------|------|------|
| RAG 从入门到精通 | 教程 | 中文 | ⭐⭐⭐⭐⭐ | https://github.com/NirDiamant/RAG_Techniques |
| LangChain 中文文档 | 文档 | 中文 | ⭐⭐⭐⭐ | https://python.langchain.com.cn/ |
| 向量数据库实战 | 教程 | 中文 | ⭐⭐⭐⭐ | https://zhuanlan.zhihu.com/p/647392838 |
| LlamaIndex 官方文档 | 文档 | 英文 | ⭐⭐⭐⭐⭐ | https://docs.llamaindex.ai/ |
| Advanced RAG Techniques | 博客 | 英文 | ⭐⭐⭐⭐⭐ | https://blog.llamaindex.ai/advanced-rag-techniques-an-illustrated-overview-04d193d8fec6 |

#### 实践项目
1. 构建企业知识库问答
2. 实现混合检索系统
3. 优化检索准确率

#### 检验标准
- ✅ 能构建完整的 RAG 系统
- ✅ 理解不同检索策略的优劣
- ✅ 能优化检索效果

---

### 技能点 12：LangChain 开发 ⭐⭐⭐⭐⭐

#### 核心内容

**基础组件**
- LLM 封装
- Prompt Template
- Output Parser

**链式调用**
- Sequential Chain
- Router Chain
- Transform Chain

**Agent 开发**
- Tool 定义
- ReAct Agent
- 自定义 Agent

**Memory 管理**
- Conversation Buffer
- Conversation Summary
- Vector Store Memory

#### 学习资源

| 资源 | 类型 | 语言 | 质量 | 链接 |
|------|------|------|------|------|
| LangChain 中文入门教程 | 教程 | 中文 | ⭐⭐⭐⭐⭐ | https://github.com/liaokongVFX/LangChain-Chinese-Getting-Started-Guide |
| LangChain 实战课 | 视频 | 中文 | ⭐⭐⭐⭐ | https://www.bilibili.com/video/BV1Qu4y1h7gx |
| LangChain 官方文档 | 文档 | 英文 | ⭐⭐⭐⭐⭐ | https://python.langchain.com/ |
| LangChain Cookbook | 示例 | 英文 | ⭐⭐⭐⭐⭐ | https://github.com/langchain-ai/langchain/tree/master/cookbook |
| LangGraph 文档 | 文档 | 英文 | ⭐⭐⭐⭐⭐ | https://langchain-ai.github.io/langgraph/ |

#### 实践项目
1. 构建多轮对话系统
2. 开发工具调用 Agent
3. 实现复杂的工作流

#### 检验标准
- ✅ 能快速构建 LLM 应用
- ✅ 能开发复杂的 Agent
- ✅ 理解 LangChain 的架构设计

---

## 阶段 4：AI 工程化（1-2 个月）

### 技能点 13：模型部署 ⭐⭐⭐⭐

#### 核心内容

**推理优化**
- 模型量化（int8, int4）
- 模型剪枝
- 知识蒸馏

**部署方案**
- FastAPI 服务
- vLLM 高性能推理
- TensorRT 加速

**容器化**
- Docker 打包
- Kubernetes 部署
- 负载均衡

#### 学习资源

| 资源 | 类型 | 语言 | 质量 | 链接 |
|------|------|------|------|------|
| 大模型部署实战 | 教程 | 中文 | ⭐⭐⭐⭐ | https://github.com/datawhalechina/llm-deploy |
| vLLM 中文文档 | 文档 | 中文 | ⭐⭐⭐⭐ | https://docs.vllm.ai/en/latest/ |
| FastAPI 中文教程 | 文档 | 中文 | ⭐⭐⭐⭐⭐ | https://fastapi.tiangolo.com/zh/ |
| Docker 从入门到实践 | 书籍 | 中文 | ⭐⭐⭐⭐⭐ | https://yeasy.gitbook.io/docker_practice/ |
| TensorRT 官方文档 | 文档 | 英文 | ⭐⭐⭐⭐ | https://docs.nvidia.com/deeplearning/tensorrt/ |

#### 实践项目
1. 部署 LLM 推理服务
2. 优化推理延迟
3. 实现自动扩缩容

#### 检验标准
- ✅ 能部署高性能推理服务
- ✅ 理解量化和优化技术
- ✅ 能处理高并发请求

---

### 技能点 14：模型微调实战 ⭐⭐⭐⭐

#### 核心内容

**微调框架**
- LLaMA-Factory
- Axolotl
- OpenAI Fine-tuning API

**数据准备**
- 数据收集
- 数据清洗
- 数据格式化

**训练技巧**
- 超参数调优
- 训练监控
- 模型评估

#### 学习资源

| 资源 | 类型 | 语言 | 质量 | 链接 |
|------|------|------|------|------|
| LLaMA-Factory 中文文档 | 文档 | 中文 | ⭐⭐⭐⭐⭐ | https://github.com/hiyouga/LLaMA-Factory/blob/main/README_zh.md |
| 大模型微调实战 | 教程 | 中文 | ⭐⭐⭐⭐⭐ | https://github.com/datawhalechina/self-llm |
| 动手学大模型微调 | 视频 | 中文 | ⭐⭐⭐⭐ | https://www.bilibili.com/video/BV1Qh4y1L7Wd |
| PEFT 实战指南 | 教程 | 中文 | ⭐⭐⭐⭐ | https://zhuanlan.zhihu.com/p/636038478 |
| Fine-tuning LLMs | 课程 | 英文 | ⭐⭐⭐⭐⭐ | https://www.deeplearning.ai/short-courses/finetuning-large-language-models/ |

#### 实践项目
1. 微调 Qwen 模型
2. 构建领域专用模型
3. 评估微调效果

#### 检验标准
- ✅ 能独立完成模型微调
- ✅ 理解不同微调方法的优劣
- ✅ 能评估和优化微调效果

---

### 技能点 15：系统设计 ⭐⭐⭐⭐

#### 核心内容

**架构设计**
- 微服务架构
- 消息队列
- 缓存策略

**数据管理**
- 数据版本控制
- 特征工程
- 数据增强

**监控和日志**
- Prometheus 监控
- ELK 日志
- 性能分析

#### 学习资源

| 资源 | 类型 | 语言 | 质量 | 链接 |
|------|------|------|------|------|
| 凤凰架构 | 书籍 | 中文 | ⭐⭐⭐⭐⭐ | https://icyfenix.cn/ |
| 系统设计入门 | 教程 | 中文 | ⭐⭐⭐⭐⭐ | https://github.com/donnemartin/system-design-primer/blob/master/README-zh-Hans.md |
| MLOps 实战 | 教程 | 中文 | ⭐⭐⭐⭐ | https://github.com/microsoft/MLOps |
| Designing Data-Intensive Applications | 书籍 | 英文 | ⭐⭐⭐⭐⭐ | - |

#### 实践项目
1. 设计 AI 应用架构
2. 实现完整的监控系统
3. 优化系统性能

#### 检验标准
- ✅ 能设计可扩展的系统
- ✅ 能处理生产环境问题
- ✅ 理解系统瓶颈和优化

---

## 阶段 5：高级专题（持续学习）

### 技能点 16：多模态 AI ⭐⭐⭐⭐

#### 核心内容
- 视觉-语言模型（CLIP, BLIP）
- 图像生成（Stable Diffusion）
- 视频理解
- 语音识别和合成

#### 学习资源

| 资源 | 类型 | 语言 | 质量 | 链接 |
|------|------|------|------|------|
| 多模态大模型综述 | 论文解读 | 中文 | ⭐⭐⭐⭐ | https://www.bilibili.com/video/BV1Qs4y1h7pz |
| CLIP 论文精读 | 视频 | 中文 | ⭐⭐⭐⭐⭐ | https://www.bilibili.com/video/BV1SL4y1s7LQ |
| Stable Diffusion 原理 | 教程 | 中文 | ⭐⭐⭐⭐ | https://zhuanlan.zhihu.com/p/617134893 |
| Multimodal Learning | 课程 | 英文 | ⭐⭐⭐⭐⭐ | https://cmu-multicomp-lab.github.io/mmml-course/fall2022/ |

---

### 技能点 17：强化学习 ⭐⭐⭐

#### 核心内容
- Q-Learning
- Policy Gradient
- PPO, RLHF
- 应用于 LLM 对齐

#### 学习资源

| 资源 | 类型 | 语言 | 质量 | 链接 |
|------|------|------|------|------|
| 强化学习纲要 | 书籍 | 中文 | ⭐⭐⭐⭐⭐ | https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow |
| 李宏毅强化学习 | 视频 | 中文 | ⭐⭐⭐⭐⭐ | https://www.bilibili.com/video/BV1MW411w79n |
| RLHF 详解 | 教程 | 中文 | ⭐⭐⭐⭐ | https://zhuanlan.zhihu.com/p/595579042 |
| CS285 Deep RL | 课程 | 英文 | ⭐⭐⭐⭐⭐ | https://rail.eecs.berkeley.edu/deeprlcourse/ |

---

### 技能点 18：AI 安全 ⭐⭐⭐

#### 核心内容
- 对抗样本
- 模型鲁棒性
- 隐私保护
- 内容审核

#### 学习资源

| 资源 | 类型 | 语言 | 质量 | 链接 |
|------|------|------|------|------|
| AI 安全综述 | 论文 | 中文 | ⭐⭐⭐⭐ | https://zhuanlan.zhihu.com/p/352581393 |
| 大模型安全 | 教程 | 中文 | ⭐⭐⭐⭐ | https://github.com/mo-xiaoxi/LLM_Security |
| Adversarial ML | 课程 | 英文 | ⭐⭐⭐⭐ | https://adversarial-ml-tutorial.org/ |

---

## 学习时间规划

### 全职学习（8-13 个月）

**每天 8 小时学习**

- 阶段 0：1 个月
- 阶段 1：2 个月
- 阶段 2：2 个月
- 阶段 3：2 个月
- 阶段 4：1 个月
- **总计**：8 个月

**额外时间**
- 项目实践：2-3 个月
- 求职准备：1-2 个月
- **总时长**：11-13 个月

---

### 业余学习（1.5-2 年）

**每天 2-3 小时学习**

- 阶段 0：2 个月
- 阶段 1：4 个月
- 阶段 2：4 个月
- 阶段 3：4 个月
- 阶段 4：2 个月
- **总计**：16 个月

**额外时间**
- 项目实践：4 个月
- 求职准备：2 个月
- **总时长**：22 个月（约 2 年）

---

## 检验里程碑

### 3 个月后

- ✅ 能用 Scikit-learn 训练机器学习模型
- ✅ 能用 PyTorch 实现简单神经网络
- ✅ 理解梯度下降和反向传播
- ✅ 完成 2-3 个小项目

---

### 6 个月后

- ✅ 能从零实现 Transformer
- ✅ 能微调 BERT 做文本分类
- ✅ 理解 Attention 机制
- ✅ 完成 1 个中型项目

---

### 12 个月后

- ✅ 能开发完整的 LLM 应用
- ✅ 能微调大模型（LoRA）
- ✅ 能构建 RAG 系统
- ✅ 能部署生产级服务
- ✅ 完成 1 个大型项目

---

## 推荐资源汇总

### 综合学习平台

| 平台 | 类型 | 语言 | 链接 |
|------|------|------|------|
| Datawhale | 开源学习社区 | 中文 | https://datawhale.club/ |
| 动手学深度学习 | 交互式教材 | 中文 | https://zh.d2l.ai/ |
| Coursera | 在线课程 | 中英文 | https://www.coursera.org/ |
| Kaggle | 数据竞赛+学习 | 英文 | https://www.kaggle.com/ |
| Papers with Code | 论文+代码 | 英文 | https://paperswithcode.com/ |

---

### 视频学习

| 资源 | 讲师 | 质量 | 链接 |
|------|------|------|------|
| 李沐动手学深度学习 | 李沐 | ⭐⭐⭐⭐⭐ | https://space.bilibili.com/1567748478 |
| 李宏毅机器学习 | 李宏毅 | ⭐⭐⭐⭐⭐ | https://speech.ee.ntu.edu.tw/~hylee/ml/2023-spring.php |
| 吴恩达深度学习 | Andrew Ng | ⭐⭐⭐⭐⭐ | https://www.coursera.org/specializations/deep-learning |
| 3Blue1Brown | Grant Sanderson | ⭐⭐⭐⭐⭐ | https://www.3blue1brown.com/ |

---

### 书籍推荐

| 书名 | 作者 | 难度 | 适合阶段 |
|------|------|------|---------|
| 《Python 编程：从入门到实践》 | Eric Matthes | 入门 | 阶段 0 |
| 《机器学习实战》 | Peter Harrington | 入门 | 阶段 1 |
| 《深度学习》（花书） | Ian Goodfellow | 进阶 | 阶段 1-2 |
| 《动手学深度学习》 | 李沐 | 中级 | 阶段 1-2 |
| 《神经网络与深度学习》 | 邱锡鹏 | 中级 | 阶段 1-2 |
| 《自然语言处理综论》 | Jurafsky | 进阶 | 阶段 2-3 |

---

### 实践平台

| 平台 | 用途 | 免费额度 | 链接 |
|------|------|---------|------|
| Google Colab | 免费 GPU | 有限制 | https://colab.research.google.com/ |
| Kaggle Notebooks | 数据竞赛 | 每周 30h GPU | https://www.kaggle.com/code |
| Hugging Face Spaces | 模型部署 | 免费 CPU | https://huggingface.co/spaces |
| Replicate | 模型推理 | 有限免费 | https://replicate.com/ |

---

### 社区和论坛

| 社区 | 特点 | 链接 |
|------|------|------|
| GitHub | 开源代码 | https://github.com/ |
| Stack Overflow | 技术问答 | https://stackoverflow.com/ |
| 知乎 | 中文技术讨论 | https://www.zhihu.com/ |
| Reddit r/MachineLearning | 英文社区 | https://www.reddit.com/r/MachineLearning/ |
| Hugging Face 论坛 | 模型讨论 | https://discuss.huggingface.co/ |

---

### 论文阅读

#### 必读经典论文（按重要性排序）

1. **Attention Is All You Need** (Transformer)
   - 链接：https://arxiv.org/abs/1706.03762
   - 解读：https://www.bilibili.com/video/BV1pu411o7BE

2. **BERT: Pre-training of Deep Bidirectional Transformers**
   - 链接：https://arxiv.org/abs/1810.04805
   - 解读：https://www.bilibili.com/video/BV1PL411M7eQ

3. **GPT-3: Language Models are Few-Shot Learners**
   - 链接：https://arxiv.org/abs/2005.14165
   - 解读：https://www.bilibili.com/video/BV1AF411b7xQ

4. **LoRA: Low-Rank Adaptation of Large Language Models**
   - 链接：https://arxiv.org/abs/2106.09685
   - 解读：https://www.bilibili.com/video/BV1Qh4y1L7Wd

5. **Chain-of-Thought Prompting**
   - 链接：https://arxiv.org/abs/2201.11903
   - 解读：https://zhuanlan.zhihu.com/p/589087074

---

## 学习建议

### 1. 理论与实践结合

**学习比例**
- 理论学习：30%
- 代码实践：50%
- 项目开发：20%

**每学一个概念，立即写代码验证**

---

### 2. 项目驱动学习

**建议的项目序列**
1. 手写数字识别（入门）
2. 文本情感分析（NLP 入门）
3. 聊天机器人（LLM 应用）
4. 企业知识库问答（RAG）
5. 完整的 AI 应用（综合）

**你的城投项目就是很好的实践！**

---

### 3. 阅读经典论文

**每周读 1-2 篇论文**

阅读方法：
1. 第一遍：快速浏览，了解大意
2. 第二遍：仔细阅读，理解细节
3. 第三遍：复现代码，深入理解

---

### 4. 参与开源社区

**推荐活动**
- 在 GitHub 上 star 优秀项目
- 提交 Issue 和 Pull Request
- 参加 Kaggle 竞赛
- 在 Hugging Face 分享模型
- 写技术博客

---

### 5. 持续学习

**AI 领域发展很快，需要持续学习**

- 关注顶会论文（NeurIPS, ICML, ACL）
- 订阅技术博客和 Newsletter
- 参加线上/线下技术分享
- 保持好奇心和探索精神

---

## 总结

### 核心路径

```
数学基础 → 机器学习 → 深度学习 → NLP → Transformer → 
大模型应用 → LangChain → RAG → 微调 → 部署
```

### 关键技能

- **PyTorch**（必须精通）
- **Transformers**（必须精通）
- **LangChain**（必须精通）
- **Prompt Engineering**（必须精通）
- **RAG**（必须精通）

### 学习心态

- **持续学习**（AI 发展太快）
- **动手实践**（理论必须验证）
- **参与社区**（交流很重要）
- **保持好奇**（探索新技术）

---

## 附录：常用工具和库

### Python 基础库
- NumPy：数组计算
- Pandas：数据处理
- Matplotlib：数据可视化
- Scikit-learn：机器学习

### 深度学习框架
- PyTorch：深度学习框架
- TensorFlow：深度学习框架
- JAX：高性能计算

### NLP 工具
- Transformers：预训练模型
- NLTK：自然语言处理
- spaCy：工业级 NLP
- jieba：中文分词

### LLM 应用框架
- LangChain：LLM 应用开发
- LlamaIndex：数据索引和检索
- Semantic Kernel：微软的 LLM 框架

### 向量数据库
- Chroma：轻量级向量数据库
- Qdrant：高性能向量数据库
- Pinecone：托管向量数据库
- Milvus：开源向量数据库

### 部署工具
- FastAPI：Web 框架
- vLLM：高性能推理
- TensorRT：推理加速
- Docker：容器化

---

**祝你学习顺利！🚀**

*最后更新：2026-01-27*
