---
title: 企业 AI 开发学习
published: 2025-01-27
description: AI 学习
tags: [AI]
category: AI
licenseName: "Unlicensed"
author: panxiao
sourceLink: "https://github.com/px6707/myblog"
draft: false
---

# 企业级 AI 开发快速上手指南

> 面向实战的 AI 应用开发路径
> 
> 总时长：2-3 个月即可上手，6 个月达到熟练

---

## 📋 目录

- [学习路径总览](#学习路径总览)
- [核心技能树](#核心技能树)
- [第 1 阶段：LLM 应用基础](#第-1-阶段llm-应用基础2-3-周)
- [第 2 阶段：LangChain 实战](#第-2-阶段langchain-实战2-3-周)
- [第 3 阶段：Prompt Engineering](#第-3-阶段prompt-engineering1-2-周)
- [第 4 阶段：RAG 检索增强](#第-4-阶段rag-检索增强2-3-周)
- [第 5 阶段：模型微调](#第-5-阶段模型微调可选2-3-周)
- [第 6 阶段：部署与优化](#第-6-阶段部署与优化1-2-周)
- [实战项目](#实战项目)
- [学习时间规划](#学习时间规划)
- [常见问题](#常见问题)

---

## 学习路径总览

```
第 1 阶段：LLM 应用基础（2-3 周）
  ↓
第 2 阶段：LangChain 实战（2-3 周）
  ↓
第 3 阶段：Prompt Engineering（1-2 周）
  ↓
第 4 阶段：RAG 检索增强（2-3 周）
  ↓
第 5 阶段：模型微调（可选，2-3 周）
  ↓
第 6 阶段：部署与优化（1-2 周）
```

**核心理念**：边学边做，快速迭代，按需深入

---

## 核心技能树

```
企业级 AI 开发技能树
├── 🎯 核心技能（必修）
│   ├── LLM API 调用 ⭐⭐⭐⭐⭐
│   ├── LangChain 框架 ⭐⭐⭐⭐⭐
│   ├── Prompt Engineering ⭐⭐⭐⭐⭐
│   └── Agent 开发 ⭐⭐⭐⭐⭐
│
├── 🔧 进阶技能（推荐）
│   ├── RAG 检索增强 ⭐⭐⭐⭐
│   ├── 向量数据库 ⭐⭐⭐⭐
│   ├── 模型微调（LoRA）⭐⭐⭐
│   └── FastAPI 部署 ⭐⭐⭐⭐
│
└── 🚀 优化技能（可选）
    ├── 性能优化 ⭐⭐⭐
    ├── 成本优化 ⭐⭐⭐
    └── 监控和日志 ⭐⭐⭐
```

---

## 第 1 阶段：LLM 应用基础（2-3 周）

### 学习目标

- 理解大语言模型的基本概念
- 能够调用各种 LLM API
- 理解 Token、Temperature、Top-p 等参数
- 能够处理 LLM 的输入输出

### 核心知识点

#### 1. LLM 基础概念

**必须理解的概念**
- Token 和 Tokenization
- Context Window（上下文窗口）
- Temperature（温度参数）
- Top-p / Top-k 采样
- System Prompt vs User Prompt

**学习资源**

| 资源 | 类型 | 语言 | 质量 | 链接 |
|------|------|------|------|------|
| OpenAI API 官方文档 | 文档 | 英文 | ⭐⭐⭐⭐⭐ | https://platform.openai.com/docs/introduction |
| 大模型基础概念入门 | 视频 | 中文 | ⭐⭐⭐⭐⭐ | https://www.bilibili.com/video/BV1bx4y1Q7rX |
| 大语言模型原理与应用 | 文章 | 中文 | ⭐⭐⭐⭐ | https://zhuanlan.zhihu.com/p/597586623 |

#### 2. API 调用实战

**OpenAI API**

```python
from openai import OpenAI

client = OpenAI(api_key="your-api-key")

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "你是一个有帮助的助手"},
        {"role": "user", "content": "什么是机器学习？"}
    ],
    temperature=0.7,
    max_tokens=500
)

print(response.choices[0].message.content)
```

**国产大模型 API**

```python
# Qwen API（兼容 OpenAI 格式）
from openai import OpenAI

client = OpenAI(
    api_key="your-qwen-api-key",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 使用方式与 OpenAI 相同
```

**学习资源**

| 资源 | 类型 | 语言 | 质量 | 链接 |
|------|------|------|------|------|
| OpenAI Python SDK | 文档 | 英文 | ⭐⭐⭐⭐⭐ | https://github.com/openai/openai-python |
| 通义千问 API 文档 | 文档 | 中文 | ⭐⭐⭐⭐⭐ | https://help.aliyun.com/zh/dashscope/ |
| 智谱 AI API 文档 | 文档 | 中文 | ⭐⭐⭐⭐ | https://open.bigmodel.cn/dev/api |
| 百度文心一言 API | 文档 | 中文 | ⭐⭐⭐⭐ | https://cloud.baidu.com/doc/WENXINWORKSHOP/index.html |

#### 3. 流式输出处理

```python
# 流式输出（打字机效果）
stream = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "讲一个故事"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### 实战练习

1. **基础对话机器人**：实现一个简单的命令行聊天程序
2. **参数实验**：测试不同 Temperature 对输出的影响
3. **多轮对话**：实现带历史记录的对话系统
4. **流式输出**：实现打字机效果的输出

### 检验标准

- ✅ 能够调用至少 2 种 LLM API
- ✅ 理解并能调整模型参数
- ✅ 能够处理流式输出
- ✅ 能够实现多轮对话

---

## 第 2 阶段：LangChain 实战（2-3 周）

### 学习目标

- 掌握 LangChain 核心组件
- 能够构建 Chain 和 Agent
- 能够定义和使用 Tool
- 理解 Memory 管理

### 核心知识点

#### 1. LangChain 基础组件

**核心组件**
- LLM Wrapper（模型封装）
- Prompt Template（提示词模板）
- Output Parser（输出解析）
- Chain（链式调用）

**学习资源**

| 资源 | 类型 | 语言 | 质量 | 链接 |
|------|------|------|------|------|
| LangChain 中文入门教程 | 教程 | 中文 | ⭐⭐⭐⭐⭐ | https://github.com/liaokongVFX/LangChain-Chinese-Getting-Started-Guide |
| LangChain 官方文档 | 文档 | 英文 | ⭐⭐⭐⭐⭐ | https://python.langchain.com/docs/get_started/introduction |
| LangChain 中文文档 | 文档 | 中文 | ⭐⭐⭐⭐ | https://python.langchain.com.cn/ |
| LangChain 实战教程 | 视频 | 中文 | ⭐⭐⭐⭐ | https://www.bilibili.com/video/BV1sN4y1J7bP |

#### 2. Prompt Template

```python
from langchain_core.prompts import ChatPromptTemplate

# 创建提示词模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个{role}"),
    ("user", "{input}")
])

# 使用模板
messages = prompt.format_messages(
    role="Python 专家",
    input="如何读取 CSV 文件？"
)
```

#### 3. Chain 链式调用

```python
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# 构建链
llm = ChatOpenAI(model="gpt-3.5-turbo")
chain = prompt | llm | StrOutputParser()

# 执行
result = chain.invoke({
    "role": "Python 专家",
    "input": "如何读取 CSV 文件？"
})
```

#### 4. Tool 工具定义

```python
from langchain.tools import tool

@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息"""
    # 实际调用天气 API
    return f"{city}的天气是晴天，温度 25°C"

@tool
def search_database(query: str) -> str:
    """在数据库中搜索信息"""
    # 实际查询数据库
    return f"查询结果：{query}"
```

#### 5. Agent 开发

```python
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

# 定义工具列表
tools = [get_weather, search_database]

# 创建提示词
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有帮助的助手"),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# 创建 Agent
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

# 执行
result = agent_executor.invoke({
    "input": "北京的天气怎么样？"
})
```

**学习资源**

| 资源 | 类型 | 语言 | 质量 | 链接 |
|------|------|------|------|------|
| LangChain Agent 教程 | 教程 | 中文 | ⭐⭐⭐⭐⭐ | https://python.langchain.com.cn/docs/modules/agents/ |
| LangChain Agent 开发 | 视频 | 中文 | ⭐⭐⭐⭐ | https://www.bilibili.com/video/BV1TM4y1W7TT |
| LangChain Cookbook | 示例 | 英文 | ⭐⭐⭐⭐⭐ | https://github.com/langchain-ai/langchain/tree/master/cookbook |

#### 6. Memory 管理

```python
from langchain.memory import ConversationBufferMemory

# 创建记忆
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# 使用记忆
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True
)
```

### 实战练习

1. **简单 Chain**：实现一个翻译 Chain
2. **工具调用**：实现 3-5 个自定义工具
3. **Agent 开发**：构建一个能调用工具的 Agent
4. **多轮对话**：实现带记忆的对话系统

### 检验标准

- ✅ 能够使用 Prompt Template
- ✅ 能够构建 Chain
- ✅ 能够定义自定义 Tool
- ✅ 能够开发 Agent
- ✅ 能够管理对话历史

---

## 第 3 阶段：Prompt Engineering（1-2 周）

### 学习目标

- 掌握提示词设计原则
- 能够优化提示词提升效果
- 理解各种提示技巧
- 能够设计复杂任务的提示词

### 核心知识点

#### 1. 提示词设计原则

**基本原则**
- 清晰明确（Be Clear and Specific）
- 提供上下文（Provide Context）
- 给出示例（Use Examples）
- 分步骤（Step by Step）
- 设定角色（Role Playing）

**学习资源**

| 资源 | 类型 | 语言 | 质量 | 链接 |
|------|------|------|------|------|
| 提示工程指南（中文） | 文档 | 中文 | ⭐⭐⭐⭐⭐ | https://www.promptingguide.ai/zh |
| LangGPT 结构化提示词 | 教程 | 中文 | ⭐⭐⭐⭐⭐ | https://github.com/yzfly/LangGPT |
| 吴恩达 Prompt Engineering 课程 | 视频 | 中文字幕 | ⭐⭐⭐⭐⭐ | https://www.bilibili.com/video/BV1Bo4y1A7FU |
| OpenAI Prompt 最佳实践 | 文档 | 英文 | ⭐⭐⭐⭐⭐ | https://platform.openai.com/docs/guides/prompt-engineering |
| Prompt 提示词技巧 | 文章 | 中文 | ⭐⭐⭐⭐ | https://zhuanlan.zhihu.com/p/632369186 |

#### 2. 基础提示技巧

**Zero-shot Prompting**

```python
prompt = """
请分析以下文本的情感（积极/消极/中性）：

文本：今天天气真好，心情很愉快。

情感：
"""
```

**Few-shot Prompting**

```python
prompt = """
请分析以下文本的情感：

示例 1：
文本：今天天气真好，心情很愉快。
情感：积极

示例 2：
文本：下雨了，心情有点低落。
情感：消极

示例 3：
文本：今天是周一。
情感：中性

现在分析：
文本：这个产品质量很不错，值得推荐。
情感：
"""
```

**Chain-of-Thought（思维链）**

```python
prompt = """
问题：一个班级有 30 个学生，其中 60% 是女生。如果新来了 5 个男生，现在女生占比是多少？

请一步步思考：
1. 首先计算原来有多少女生
2. 然后计算原来有多少男生
3. 计算新来男生后的总人数
4. 最后计算新的女生占比

让我们开始：
"""
```

#### 3. 高级提示技巧

**角色设定**

```python
system_prompt = """
你是一位资深的 Python 工程师，拥有 10 年的开发经验。
你擅长：
- 编写高质量、可维护的代码
- 性能优化
- 最佳实践

你的回答风格：
- 简洁明了
- 提供代码示例
- 解释关键概念
"""
```

**结构化输出**

```python
prompt = """
请分析以下产品评论，并以 JSON 格式输出：

评论：这个手机拍照效果很好，但是电池续航一般。

输出格式：
{
    "sentiment": "积极/消极/中性",
    "positive_aspects": ["优点1", "优点2"],
    "negative_aspects": ["缺点1", "缺点2"],
    "rating": 1-5
}
"""
```

**ReAct（推理+行动）**

```python
prompt = """
你需要回答用户的问题，可以使用以下工具：
- search_database: 搜索数据库
- calculate: 执行计算
- get_weather: 获取天气

请按照以下格式思考和行动：

Thought: 我需要做什么？
Action: 使用哪个工具
Action Input: 工具的输入
Observation: 工具的输出
... (重复 Thought/Action/Observation)
Thought: 我现在知道答案了
Final Answer: 最终答案

问题：北京今天的天气怎么样？
"""
```

**学习资源**

| 资源 | 类型 | 语言 | 质量 | 链接 |
|------|------|------|------|------|
| Chain-of-Thought 论文解读 | 文章 | 中文 | ⭐⭐⭐⭐ | https://zhuanlan.zhihu.com/p/589087074 |
| ReAct 提示技巧 | 教程 | 中文 | ⭐⭐⭐⭐ | https://www.promptingguide.ai/zh/techniques/react |
| Prompt Engineering 实战 | 视频 | 中文 | ⭐⭐⭐⭐ | https://www.bilibili.com/video/BV1No4y1t7Zn |

### 实战练习

1. **情感分析**：设计提示词进行文本情感分析
2. **数据提取**：从非结构化文本中提取结构化信息
3. **代码生成**：设计提示词生成高质量代码
4. **复杂推理**：使用 CoT 解决数学或逻辑问题

### 检验标准

- ✅ 能够设计清晰有效的提示词
- ✅ 能够使用 Few-shot 提升效果
- ✅ 能够使用 CoT 处理复杂任务
- ✅ 能够设计结构化输出
- ✅ 能够优化提示词降低成本

---

## 第 4 阶段：RAG 检索增强（2-3 周）

### 学习目标

- 理解 RAG 的原理和应用场景
- 掌握向量数据库的使用
- 能够构建文档问答系统
- 能够优化检索效果

### 核心知识点

#### 1. RAG 基础概念

**什么是 RAG？**
- Retrieval（检索）：从知识库中找到相关文档
- Augmented（增强）：将文档作为上下文
- Generation（生成）：LLM 基于上下文生成答案

**为什么需要 RAG？**
- 解决 LLM 知识过时问题
- 提供可追溯的信息来源
- 降低幻觉（Hallucination）
- 支持私有知识库

**学习资源**

| 资源 | 类型 | 语言 | 质量 | 链接 |
|------|------|------|------|------|
| RAG 从入门到精通 | 教程 | 中文 | ⭐⭐⭐⭐⭐ | https://github.com/NirDiamant/RAG_Techniques |
| RAG 原理详解 | 文章 | 中文 | ⭐⭐⭐⭐⭐ | https://zhuanlan.zhihu.com/p/647392838 |
| LangChain RAG 教程 | 教程 | 中文 | ⭐⭐⭐⭐ | https://python.langchain.com.cn/docs/use_cases/question_answering/ |
| RAG 检索增强生成实战 | 视频 | 中文 | ⭐⭐⭐⭐ | https://www.bilibili.com/video/BV1w8411B7jK |

#### 2. 向量数据库

**Embedding（向量化）**

```python
from langchain_openai import OpenAIEmbeddings

# 创建 Embedding 模型
embeddings = OpenAIEmbeddings()

# 向量化文本
text = "今天天气很好"
vector = embeddings.embed_query(text)

print(f"向量维度: {len(vector)}")  # 1536
print(f"向量前 5 个值: {vector[:5]}")
```

**Chroma 向量数据库**

```python
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. 加载文档
loader = TextLoader("document.txt")
documents = loader.load()

# 2. 切分文档
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(documents)

# 3. 创建向量数据库
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# 4. 检索
results = vectorstore.similarity_search("查询问题", k=3)
```

**学习资源**

| 资源 | 类型 | 语言 | 质量 | 链接 |
|------|------|------|------|------|
| Chroma 官方文档 | 文档 | 英文 | ⭐⭐⭐⭐⭐ | https://docs.trychroma.com/ |
| 向量数据库对比 | 文章 | 中文 | ⭐⭐⭐⭐ | https://zhuanlan.zhihu.com/p/647392838 |
| 向量数据库与 Embedding | 视频 | 中文 | ⭐⭐⭐⭐ | https://www.bilibili.com/video/BV1Qu4y1h7gx |
| Qdrant 官方文档 | 文档 | 英文 | ⭐⭐⭐⭐ | https://qdrant.tech/documentation/ |

#### 3. 构建 RAG 系统

**基础 RAG**

```python
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# 创建检索器
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 创建 QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# 提问
result = qa_chain.invoke({"query": "什么是机器学习？"})
print(result["result"])
print(result["source_documents"])
```

**自定义 RAG**

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 自定义提示词
template = """
基于以下上下文回答问题。如果上下文中没有相关信息，请说"我不知道"。

上下文：
{context}

问题：{question}

答案：
"""

prompt = ChatPromptTemplate.from_template(template)

# 构建 RAG Chain
def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 使用
answer = rag_chain.invoke("什么是机器学习？")
```

#### 4. RAG 优化技巧

**文档切分策略**

```python
# 按字符切分
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # 块大小
    chunk_overlap=50,    # 重叠部分
    separators=["\n\n", "\n", "。", "！", "？", " ", ""]
)

# 按语义切分
from langchain_experimental.text_splitter import SemanticChunker

splitter = SemanticChunker(embeddings)
```

**混合检索（BM25 + Vector）**

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# BM25 检索器（关键词）
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 3

# 向量检索器（语义）
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 混合检索器
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6]  # BM25 占 40%，向量占 60%
)
```

**重排序（Reranking）**

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank

# 创建重排序器
compressor = CohereRerank()

# 创建压缩检索器
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)
```

**学习资源**

| 资源 | 类型 | 语言 | 质量 | 链接 |
|------|------|------|------|------|
| RAG 优化技巧 | 文章 | 中文 | ⭐⭐⭐⭐⭐ | https://zhuanlan.zhihu.com/p/670925591 |
| 混合检索实战 | 教程 | 中文 | ⭐⭐⭐⭐ | https://python.langchain.com.cn/docs/modules/data_connection/retrievers/ensemble |
| Advanced RAG | 博客 | 英文 | ⭐⭐⭐⭐⭐ | https://blog.llamaindex.ai/advanced-rag-techniques-an-illustrated-overview-04d193d8fec6 |

### 实战练习

1. **文档问答**：构建一个企业文档问答系统
2. **检索优化**：实现混合检索并对比效果
3. **多文档源**：整合多个文档源的 RAG 系统
4. **引用来源**：实现带引用来源的回答

### 检验标准

- ✅ 能够使用向量数据库
- ✅ 能够构建基础 RAG 系统
- ✅ 能够优化文档切分策略
- ✅ 能够实现混合检索
- ✅ 能够评估检索效果

---

## 第 5 阶段：模型微调（可选，2-3 周）

### 学习目标

- 理解何时需要微调
- 掌握 LoRA 微调方法
- 能够使用 LLaMA-Factory 微调
- 能够评估微调效果

### 核心知识点

#### 1. 何时需要微调？

**需要微调的场景**
- 通用模型效果不够好
- 有大量领域专业术语
- 需要特定的输出格式
- 需要固化某些行为模式

**不需要微调的场景**
- Prompt Engineering 就能解决
- 数据量太少（< 500 条）
- 预算和时间有限
- 知识频繁更新

**学习资源**

| 资源 | 类型 | 语言 | 质量 | 链接 |
|------|------|------|------|------|
| 大模型微调实战 | 教程 | 中文 | ⭐⭐⭐⭐⭐ | https://github.com/datawhalechina/self-llm |
| LoRA 原理详解 | 文章 | 中文 | ⭐⭐⭐⭐⭐ | https://zhuanlan.zhihu.com/p/636038478 |
| 微调 vs RAG vs Prompt | 对比 | 中文 | ⭐⭐⭐⭐ | https://zhuanlan.zhihu.com/p/647392838 |

#### 2. LLaMA-Factory 微调

**安装和启动**

```bash
# 安装
pip install llama-factory

# 启动 Web UI
llamafactory-cli webui
```

**准备训练数据**

```json
[
  {
    "messages": [
      {"role": "system", "content": "你是一个有帮助的助手"},
      {"role": "user", "content": "什么是纳统？"},
      {"role": "assistant", "content": "纳统是指纳入统计局统计的投资金额"}
    ]
  },
  {
    "messages": [
      {"role": "system", "content": "你是一个有帮助的助手"},
      {"role": "user", "content": "查询在建项目"},
      {"role": "assistant", "content": "应该调用 get_dashboard_info 工具"}
    ]
  }
]
```

**Web UI 微调步骤**
1. 选择模型（如 Qwen/Qwen-7B）
2. 上传训练数据
3. 选择微调方法（LoRA）
4. 设置参数（r=8, lora_alpha=16）
5. 开始训练
6. 导出模型

**学习资源**

| 资源 | 类型 | 语言 | 质量 | 链接 |
|------|------|------|------|------|
| LLaMA-Factory 中文文档 | 文档 | 中文 | ⭐⭐⭐⭐⭐ | https://github.com/hiyouga/LLaMA-Factory/blob/main/README_zh.md |
| LLaMA-Factory 视频教程 | 视频 | 中文 | ⭐⭐⭐⭐⭐ | https://www.bilibili.com/video/BV1Qh4y1L7Wd |
| LoRA 微调实战 | 教程 | 中文 | ⭐⭐⭐⭐ | https://zhuanlan.zhihu.com/p/636038478 |

#### 3. 使用微调后的模型

**部署为 API 服务**

```bash
# 使用 vLLM 部署
pip install vllm

python -m vllm.entrypoints.openai.api_server \
    --model ./my_finetuned_model \
    --port 8000
```

**在 LangChain 中使用**

```python
from langchain_openai import ChatOpenAI

# 连接到微调模型
model = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy",
    model="my_finetuned_model"
)

# 使用（和之前一样）
response = model.invoke("查询在建项目")
```

### 实战练习

1. **数据准备**：从对话日志生成训练数据
2. **LoRA 微调**：使用 LLaMA-Factory 微调模型
3. **效果评估**：对比微调前后的效果
4. **模型部署**：部署微调后的模型

### 检验标准

- ✅ 能够准备训练数据
- ✅ 能够使用 LLaMA-Factory 微调
- ✅ 能够评估微调效果
- ✅ 能够部署微调后的模型

---

## 第 6 阶段：部署与优化（1-2 周）

### 学习目标

- 能够部署 AI 应用
- 掌握性能优化技巧
- 了解成本优化方法
- 能够监控和调试

### 核心知识点

#### 1. FastAPI 部署

**创建 API 服务**

```python
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import ChatOpenAI

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

llm = ChatOpenAI(model="gpt-3.5-turbo")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    response = llm.invoke(request.message)
    return ChatResponse(response=response.content)

# 运行：uvicorn main:app --host 0.0.0.0 --port 8000
```

**学习资源**

| 资源 | 类型 | 语言 | 质量 | 链接 |
|------|------|------|------|------|
| FastAPI 中文教程 | 文档 | 中文 | ⭐⭐⭐⭐⭐ | https://fastapi.tiangolo.com/zh/ |
| FastAPI 快速入门 | 视频 | 中文 | ⭐⭐⭐⭐ | https://www.bilibili.com/video/BV1NL411N7vQ |

#### 2. Docker 容器化

**Dockerfile**

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**docker-compose.yml**

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./data:/app/data
```

**学习资源**

| 资源 | 类型 | 语言 | 质量 | 链接 |
|------|------|------|------|------|
| Docker 从入门到实践 | 书籍 | 中文 | ⭐⭐⭐⭐⭐ | https://yeasy.gitbook.io/docker_practice/ |
| Docker 入门教程 | 视频 | 中文 | ⭐⭐⭐⭐ | https://www.bilibili.com/video/BV1s54y1n7Ev |

#### 3. 性能优化

**缓存策略**

```python
from functools import lru_cache
import hashlib

# 内存缓存
@lru_cache(maxsize=100)
def get_embedding(text: str):
    return embeddings.embed_query(text)

# Redis 缓存
import redis
r = redis.Redis(host='localhost', port=6379)

def cached_llm_call(prompt: str):
    # 生成缓存键
    cache_key = hashlib.md5(prompt.encode()).hexdigest()
    
    # 检查缓存
    cached = r.get(cache_key)
    if cached:
        return cached.decode()
    
    # 调用 LLM
    response = llm.invoke(prompt)
    
    # 存入缓存
    r.setex(cache_key, 3600, response.content)
    
    return response.content
```

**并发处理**

```python
import asyncio
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo")

async def process_batch(questions):
    tasks = [llm.ainvoke(q) for q in questions]
    results = await asyncio.gather(*tasks)
    return results

# 使用
questions = ["问题1", "问题2", "问题3"]
results = asyncio.run(process_batch(questions))
```

**学习资源**

| 资源 | 类型 | 语言 | 质量 | 链接 |
|------|------|------|------|------|
| LLM 应用性能优化 | 文章 | 中文 | ⭐⭐⭐⭐ | https://zhuanlan.zhihu.com/p/647392838 |
| 缓存策略详解 | 教程 | 中文 | ⭐⭐⭐⭐ | https://python.langchain.com.cn/docs/modules/model_io/llms/llm_caching |

#### 4. 成本优化

**Prompt 压缩**

```python
# 减少不必要的上下文
def compress_context(context, max_length=1000):
    if len(context) > max_length:
        return context[:max_length] + "..."
    return context

# 使用更便宜的模型
cheap_llm = ChatOpenAI(model="gpt-3.5-turbo")  # 便宜
expensive_llm = ChatOpenAI(model="gpt-4")      # 贵

# 简单任务用便宜模型
if is_simple_task(question):
    response = cheap_llm.invoke(question)
else:
    response = expensive_llm.invoke(question)
```

**流式输出**

```python
# 流式输出可以更快显示结果，提升用户体验
for chunk in llm.stream("讲一个故事"):
    print(chunk.content, end="", flush=True)
```

#### 5. 监控和日志

```python
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# 记录请求
@app.post("/chat")
async def chat(request: ChatRequest):
    start_time = datetime.now()
    
    try:
        response = llm.invoke(request.message)
        
        # 记录成功
        logger.info(f"Request: {request.message[:50]}... | "
                   f"Response: {response.content[:50]}... | "
                   f"Time: {(datetime.now() - start_time).total_seconds()}s")
        
        return ChatResponse(response=response.content)
    
    except Exception as e:
        # 记录错误
        logger.error(f"Error: {str(e)} | Request: {request.message}")
        raise
```

### 实战练习

1. **API 服务**：部署一个 FastAPI 服务
2. **Docker 部署**：容器化你的应用
3. **性能优化**：实现缓存和并发
4. **监控日志**：添加完整的日志系统

### 检验标准

- ✅ 能够部署 FastAPI 服务
- ✅ 能够使用 Docker 容器化
- ✅ 能够实现缓存优化
- ✅ 能够添加监控和日志

---

## 实战项目

### 项目 1：智能客服机器人（入门）

**功能需求**
- 多轮对话
- 常见问题解答
- 情感识别
- 工单创建

**技术栈**
- LangChain
- Prompt Engineering
- Memory 管理

**时间**：1 周

---

### 项目 2：企业知识库问答（进阶）

**功能需求**
- 文档上传和索引
- 智能检索
- 引用来源
- 多文档源整合

**技术栈**
- RAG
- 向量数据库（Chroma）
- 混合检索

**时间**：2 周

---

### 项目 3：代码助手（进阶）

**功能需求**
- 代码生成
- 代码解释
- Bug 修复
- 代码审查

**技术栈**
- LangChain Agent
- Tool 调用
- Prompt Engineering

**时间**：2 周

---

### 项目 4：数据分析助手（综合）

**功能需求**
- 自然语言查询数据库
- 数据可视化
- 报告生成
- 趋势分析

**技术栈**
- LangChain Agent
- SQL Tool
- 图表生成
- RAG

**时间**：3 周

---

## 学习时间规划

### 全职学习（2-3 个月）

**每天 6-8 小时**

| 阶段 | 时间 | 重点 |
|------|------|------|
| 第 1 阶段 | 2-3 周 | LLM API 调用 |
| 第 2 阶段 | 2-3 周 | LangChain 框架 |
| 第 3 阶段 | 1-2 周 | Prompt Engineering |
| 第 4 阶段 | 2-3 周 | RAG 系统 |
| 第 5 阶段 | 2-3 周（可选） | 模型微调 |
| 第 6 阶段 | 1-2 周 | 部署优化 |

**总计**：10-16 周（2.5-4 个月）

---

### 业余学习（4-6 个月）

**每天 2-3 小时**

| 阶段 | 时间 | 重点 |
|------|------|------|
| 第 1 阶段 | 3-4 周 | LLM API 调用 |
| 第 2 阶段 | 4-5 周 | LangChain 框架 |
| 第 3 阶段 | 2-3 周 | Prompt Engineering |
| 第 4 阶段 | 4-5 周 | RAG 系统 |
| 第 5 阶段 | 4-5 周（可选） | 模型微调 |
| 第 6 阶段 | 2-3 周 | 部署优化 |

**总计**：19-25 周（4.5-6 个月）

---

## 常见问题

### Q1：需要深度学习基础吗？

**不需要！** 企业级 AI 开发主要是应用层面，不需要深入理解深度学习原理。

**需要了解的**：
- LLM 的基本概念（Token、Temperature 等）
- Embedding 的作用
- 基本的概率概念

**不需要了解的**：
- 反向传播算法
- 梯度下降数学推导
- Transformer 内部实现

---

### Q2：Python 需要多熟练？

**中级水平即可**

**必须掌握**：
- 基础语法（变量、函数、类）
- 数据结构（list、dict）
- 文件操作
- 异常处理

**推荐掌握**：
- 装饰器
- 异步编程（async/await）
- 常用库（requests、json）

**不需要**：
- 元编程
- C 扩展
- 底层优化

---

### Q3：需要购买 GPU 吗？

**不需要！**

**原因**：
- 使用 API 调用模型（OpenAI、Qwen 等）
- 微调可以用云服务或 Google Colab
- RAG 不需要 GPU

**什么时候需要 GPU**：
- 本地部署大模型
- 大规模微调
- 实时推理要求极高

---

### Q4：成本大概多少？

**学习阶段**：
- API 调用：$10-50/月
- 云服务器（可选）：$20-50/月
- 总计：$30-100/月

**生产环境**：
- API 调用：$100-1000/月（取决于流量）
- 云服务器：$50-200/月
- 向量数据库：$0-100/月
- 总计：$150-1300/月

---

### Q5：如何选择学习路径？

**如果你的目标是**：

**快速上手（1-2 个月）**
→ 只学第 1-3 阶段
→ 重点：LangChain + Prompt

**完整掌握（2-3 个月）**
→ 学完第 1-4 阶段
→ 重点：LangChain + RAG

**深入精通（3-4 个月）**
→ 学完所有阶段
→ 重点：全栈能力

---

### Q6：遇到问题怎么办？

**推荐资源**：

1. **官方文档**
   - LangChain：https://python.langchain.com/
   - OpenAI：https://platform.openai.com/docs/

2. **社区**
   - GitHub Issues
   - Stack Overflow
   - 知乎
   - Discord/Slack 社区

3. **实践**
   - 多写代码
   - 多做项目
   - 多看别人的代码

---

## 学习建议

### 1. 边学边做

```
理论学习：30%
代码实践：50%
项目开发：20%
```

**每学一个概念，立即写代码验证！**

---

### 2. 从简单开始

```
第 1 周：调用 API，实现简单对话
第 2 周：使用 LangChain，构建 Chain
第 3 周：优化 Prompt，提升效果
第 4 周：构建 RAG，实现文档问答
...
```

**不要一开始就做复杂项目！**

---

### 3. 重视 Prompt

```
好的 Prompt > 复杂的代码
```

**80% 的问题可以通过优化 Prompt 解决！**

---

### 4. 保持更新

**AI 领域发展很快**

- 关注 LangChain 更新
- 关注新模型发布
- 关注最佳实践
- 参与社区讨论

---

### 5. 实战为王

**最好的学习方式是做项目**

- 从小项目开始
- 逐步增加复杂度
- 解决实际问题
- 积累经验

---

## 总结

### 核心路径

```
LLM API → LangChain → Prompt → RAG → 微调 → 部署
```

### 最小学习集

**2-3 个月即可上手**

- LLM API 调用
- LangChain 基础
- Prompt Engineering
- RAG 基础

### 关键技能

- **LangChain**（必须精通）
- **Prompt Engineering**（必须精通）
- **RAG**（推荐掌握）
- **微调**（按需学习）

### 学习心态

- **快速上手**（不要追求完美）
- **边做边学**（实践最重要）
- **持续迭代**（不断优化）
- **保持好奇**（探索新技术）

---

**祝你学习顺利！开始你的 AI 开发之旅吧！🚀**

*最后更新：2026-01-27*
