---
title: 大模型微调与 RAG
published: 2026-01-27
description: 大模型微调与 RAG
tags: [AI]
category: 微调
licenseName: "Unlicensed"
author: panxiao
sourceLink: "https://github.com/px6707/myblog"
draft: false
---

> 本文深入探讨大模型应用中的两大核心技术：微调（Fine-tuning）和检索增强生成（RAG），帮助你在实际项目中做出正确的技术选型。

----

### 目录
* 核心概念
* 技术对比
* 应用场景
* 实现方法
* 工具生态
* 最佳实践

----

## 核心概念
### 什么是微调（Fine-tuning）？  
微调是在预训练模型的基础上，使用特定领域的数据进行进一步训练，使模型更适应特定任务的过程。 
```
预训练模型（通用能力） + 领域数据微调 = 专用模型
``` 
### 微调核心特点
  * 改变模型内在能力：调整模型参数，让模型"学会"领域知识
  * 一次性投入：训练完成后，知识固化在模型中
  * 推理时无需额外资源：直接使用，响应快速  
形象比喻：考前突击复习，把知识记在脑子里


### 什么是 RAG（检索增强生成）？
RAG 是在生成回答时，先从外部知识库检索相关信息，再将检索结果作为上下文传给模型的技术。
```
用户查询 → 向量检索相关文档 → 将文档作为上下文 → LLM 生成答案
```
### RAG 核心特点
  * 提供外部知识：不改变模型，只是给模型提供参考资料
  * 实时更新：知识库可以随时更新
  * 推理时需要检索：每次查询都要搜索相关文档
形象比喻：开卷考试，考试时可以查资料

## 技术对比
深度对比表. 
|维度|微调（Fine-tuning）|RAG|RAG + 微调|
|----|----|----|----|
|知识来源|模型参数内部|外部知识库|两者结合|
|知识更新|❌ 需要重新训练|✅ 实时更新|✅ 常用知识微调，新知识 RAG|
|响应速度|✅ 快（无检索）|⚠️ 较慢（需检索）|⭐ 常见问题快，新问题查|
|准确性|⭐⭐⭐⭐|⭐⭐⭐|⭐⭐⭐⭐⭐
|成本|💰💰 训练成本高|💰 检索成本低|💰💰 一次性投入
数据需求|数百到数万条|无需训练数据|少量训练数据
实施难度|⭐⭐⭐⭐ 高|⭐⭐ 低|⭐⭐⭐⭐ 高
可解释性|❌ 黑盒|✅ 可追溯来源|✅ 可追溯
领域适应|✅ 深度理解|⚠️ 依赖检索质量|✅ 最佳

## 工作原理对比
### 微调的工作流程
``` python
# 训练阶段（一次性）
训练数据 = [
    {"input": "什么是纳统？", "output": "纳统是指纳入统计局统计的投资金额"},
    {"input": "查询在建项目", "output": "应该调用 get_dashboard_info 工具"},
    # ... 数千条
]
微调后的模型 = 基础模型.train(训练数据)
# 推理阶段（每次查询）
用户："查询纳统金额"
模型：直接理解"纳统" → 调用工具 → 返回结果

```

### RAG 的工作流程
``` python
# 准备阶段（一次性）
文档库 = [
    "纳统是指纳入统计局统计的投资金额",
    "在建项目查询使用 get_dashboard_info 工具",
    # ... 数万份文档
]
向量数据库.index(文档库)
# 推理阶段（每次查询）
用户："查询纳统金额"
相关文档 = 向量数据库.search("纳统金额")
模型：基于检索到的文档 → 理解"纳统" → 调用工具 → 返回结果
```

## 应用场景
### 只用微调的场景
✅ 适用情况
1. 大量专业术语

``` python
# 场景：医疗、法律、金融领域
问题：通用模型不理解专业术语
解决：微调让模型"记住"术语含义
# 示例：医疗领域
用户："患者的 HbA1c 指标是多少？"
通用模型：不知道 HbA1c 是什么
微调模型：知道这是糖化血红蛋白，直接查询数据库
```
2. 固定行为模式

``` python
# 场景：需要按特定格式输出
需求：所有回答必须包含"数据来源"、"置信度"
解决：微调固化输出格式
# 示例
微调前：随意回答
微调后：
  "答案：XXX
   数据来源：XX报表
   置信度：95%"
```

3. 离线部署

```python
# 场景：无法访问外部 API
需求：完全离线运行
解决：微调小模型（如 7B），部署在本地
```
4. 极致性能要求

```python
# 场景：毫秒级响应
问题：RAG 检索需要时间
解决：微调后直接推理，无检索延迟
```

### 只用 RAG 的场景
✅ 适用情况
1. 知识频繁更新

```python
# 场景：新闻、政策、产品文档
问题：每天都有新内容，微调跟不上
解决：RAG 实时更新知识库
# 示例：企业知识库
今天：新增 10 份文档 → 直接索引到向量库
明天：用户就能查询到最新信息
```
2. 知识量巨大

```python
# 场景：百万级文档
问题：无法全部用于微调（成本和时间）
解决：RAG 索引所有文档，按需检索
# 示例：法律法规库
文档：100 万份法律文件
RAG：索引所有文档，查询时检索相关的 5-10 份
```

3. 需要可追溯性

```python
# 场景：合规、审计要求
需求：必须说明答案来源
解决：RAG 返回引用的文档
# 示例
回答："根据《XX规定》第3条..."
来源：[文档链接]
```
4. 快速验证需求

```python
# 场景：MVP（最小可行产品）
目标：快速上线，验证需求
解决：RAG 实施快，无需训练
```

### 结合使用的场景
⭐ 最佳实践
1. 企业级 AI 助手

```python
# 场景：城投项目管理系统（你的项目）
微调部分：
- 理解业务术语（"纳统"、"形象进度"）
- 工具选择能力（优先用 get_dashboard_info）
- 专业表达风格
RAG 部分：
- 项目文档检索（环评报告、施工方案）
- 最新政策查询
- 历史项目案例
# 工作流程
用户："东站项目的环保措施有哪些？"
微调模型：理解"环保措施" = 环评相关内容
RAG：检索东站项目的环评报告
模型：基于检索结果，生成专业回答
```
2. 客服机器人

```python
微调部分：
- 对话风格（礼貌、专业）
- 常见问题的标准回答
- 意图识别能力
RAG 部分：
- 产品手册检索
- 最新促销活动
- 用户历史记录
# 示例
用户："这个产品怎么用？"
微调：识别为"产品使用咨询"
RAG：检索产品手册相关章节
模型：生成友好的使用指南
```
3. 代码助手

```python
微调部分：
- 代码语法理解
- 编程最佳实践
- Bug 模式识别
RAG 部分：
- 项目代码库检索
- API 文档查询
- Stack Overflow 案例
# 示例
用户："如何实现用户认证？"
微调：理解这是安全相关问题
RAG：检索项目中的认证代码示例
模型：生成符合项目规范的代码
```
----

## 实现方法
### 微调方法详解
1. 全量微调（Full Fine-tuning）
原理：调整模型的所有参数

```python
# 优势
- 效果最好
- 模型完全适应新任务
# 劣势
- 需要大量数据（数万到数十万条）
- 计算资源消耗大（需要多张 GPU）
- 训练时间长（数小时到数天）
- 存储成本高（需要保存完整模型）
# 适用场景
- 预算充足
- 数据量大
- 对效果要求极高
```
2. LoRA（Low-Rank Adaptation）⭐ 推荐
原理：只训练少量额外参数

```python
# 原理示意
原始模型参数：235B（冻结不动）
LoRA 参数：~100M（只训练这部分）
# 优势
- 数据需求少（数百到数千条）
- 训练快（几分钟到几小时）
- 存储小（只需保存适配器，几百 MB）
- 可以多任务切换
# 代码示例
from peft import LoraConfig, get_peft_model
lora_config = LoraConfig(
    r=8,                    # LoRA 秩
    lora_alpha=16,          # 缩放因子
    target_modules=["q_proj", "v_proj"],  # 目标层
    lora_dropout=0.05,
)
model = get_peft_model(base_model, lora_config)
model.train()
```
3. Prompt Tuning
原理：只训练提示词的嵌入

```python
# 优势
- 更轻量级
- 只需要几十到几百条数据
- 训练极快
# 劣势
- 效果相对较弱
# 适用场景
- 资源极度受限
- 快速实验
```
### RAG 方法详解
1. 基础 RAG
流程：检索 → 拼接 → 生成

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
# 1. 构建向量库
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=OpenAIEmbeddings()
)
# 2. 检索
def rag_query(question):
    # 检索相关文档
    docs = vectorstore.similarity_search(question, k=5)
    
    # 拼接上下文
    context = "\n".join([doc.page_content for doc in docs])
    
    # 生成回答
    prompt = f"基于以下内容回答问题：\n{context}\n\n问题：{question}"
    response = llm.invoke(prompt)
    
    return response
```
2. 混合检索（Hybrid Search）⭐ 推荐
原理：结合关键词搜索（BM25）和向量搜索

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
# BM25：关键词精确匹配
bm25 = BM25Retriever.from_documents(documents)
# 向量：语义相似度
vector_retriever = vectorstore.as_retriever()
# 融合检索器
ensemble = EnsembleRetriever(
    retrievers=[bm25, vector_retriever],
    weights=[0.4, 0.6]  # BM25 占 40%，向量占 60%
)
# 使用
results = ensemble.get_relevant_documents(query)
```
优势：

- BM25 擅长精确关键词匹配
- 向量搜索擅长语义理解
- 两者结合，鲁棒性强

3. 分层检索（Hierarchical Retrieval）
原理：两级索引，先粗筛再精筛

```python
class HierarchicalRAG:
    def __init__(self):
        # 第一层：文档级别（摘要）
        self.doc_index = Chroma(collection_name="documents")
        
        # 第二层：段落级别（详细内容）
        self.chunk_index = Chroma(collection_name="chunks")
    
    def search(self, query):
        # 阶段 1：从百万文档中筛选出 20 个相关文档
        relevant_docs = self.doc_index.similarity_search(query, k=20)
        doc_ids = [d.metadata['doc_id'] for d in relevant_docs]
        
        # 阶段 2：在这 20 个文档的段落中精确搜索
        results = self.chunk_index.similarity_search(
            query,
            k=5,
            filter={"doc_id": {"$in": doc_ids}}
        )
        return results
```
适用场景：超大规模文档库（百万级）

4. 智能路由 + 专用索引
原理：根据问题类型路由到专门的索引

```python
class RoutedRAG:
    def __init__(self):
        # 为不同业务建立专用索引
        self.indexes = {
            "技术文档": Chroma(collection_name="tech"),
            "业务流程": Chroma(collection_name="business"),
            "财务报表": Chroma(collection_name="finance"),
        }
        
        self.router = ChatOpenAI(model="gpt-4")
    
    def search(self, query):
        # 1. 路由到正确的索引
        category = self.router.invoke(f"这个问题属于哪个类别？{query}")
        
        # 2. 只在相关索引中搜索
        if category in self.indexes:
            return self.indexes[category].similarity_search(query)
```
优势：性能最优，只搜索相关索引

------
## 工具生态
### 微调工具
1. LLaMA-Factory ⭐ 推荐
特点：

- 支持 100+ 模型
- 内置 LoRA、QLoRA 等方法
- Web UI 界面友好
- 中文文档完善

```bash
# 安装
pip install llama-factory
# 使用 Web UI
llamafactory-cli webui
# 命令行微调
llamafactory-cli train \
    --model_name_or_path Qwen/Qwen-7B \
    --dataset your_data \
    --finetuning_type lora \
    --output_dir ./output
```
2. Hugging Face PEFT
特点：

- 官方库，稳定可靠
- 支持 LoRA、Prefix Tuning 等
- 与 Transformers 深度集成
``` python
from peft import LoraConfig, get_peft_model, TaskType
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
)
model = get_peft_model(base_model, config)
```
3. OpenAI Fine-tuning API
特点：

- 托管服务，无需管理基础设施
- 支持 GPT-3.5、GPT-4
- 按使用付费
```python
import openai
# 上传训练数据
openai.File.create(file=open("training_data.jsonl"), purpose='fine-tune')
# 创建微调任务
openai.FineTuningJob.create(training_file="file-xxx", model="gpt-3.5-turbo")
```
4. Axolotl
特点：

-  配置驱动，灵活性高
- 支持多 GPU 训练
- 适合研究和实验

### RAG 工具

1. LangChain ⭐ 推荐
特点：

- 生态最完善
- 支持多种向量数据库
- 内置各种检索策略
```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
# 构建 RAG 链
vectorstore = Chroma.from_documents(docs, OpenAIEmbeddings())
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(),
    retriever=vectorstore.as_retriever()
)
# 查询
answer = qa_chain.invoke("你的问题")
```
2. LlamaIndex
特点：

专注于数`据索引
支持多种数据源
高级查询能力
python
from llama_index import VectorStoreIndex, SimpleDirectoryReader
# 加载文档
documents = SimpleDirectoryReader('data').load_data()
# 构建索引
index = VectorStoreIndex.from_documents(documents)
# 查询
query_engine = index.as_query_engine()
response = query_engine.query("你的问题")
```
3. Haystack
特点：

- 企业级解决方案
- 支持混合检索
- 可视化 Pipeline
### 向量数据库
轻量级
Chroma ⭐ 入门推荐

```python
# 优势：简单易用，适合原型开发
# 劣势：单机部署，性能有限
from langchain_chroma import Chroma
vectorstore = Chroma(persist_directory="./db")
```
FAISS

```python
# 优势：性能极佳，支持 GPU
# 劣势：只有内存存储，需手动持久化
from langchain_community.vectorstores import FAISS
vectorstore = FAISS.from_documents(docs, embeddings)
```
生产级
Qdrant ⭐ 生产推荐

```python
# 优势：性能优秀，支持分布式，过滤能力强
# 部署：Docker 一键部署
from langchain_qdrant import Qdrant
vectorstore = Qdrant(
    url="http://localhost:6333",
    collection_name="my_docs"
)
```
Milvus

```python
# 优势：支持十亿级向量，分布式架构
# 劣势：部署复杂，资源占用大
```
Pinecone

```python
# 优势：完全托管，零运维
# 劣势：付费服务，数据在云端
```