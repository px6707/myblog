---
title: LangChain 核心机制教程
published: 2026-06-15
description: 以 Python 版 LangChain / LangGraph v1 为语境，梳理 Runnable、并发语义、Agent 中间件生命周期、LangGraph 人工审批与工具创建方式
tags: [LangChain, LangGraph, Runnable, Agent, HITL]
category: AI
licenseName: "Unlicensed"
author: panxiao
sourceLink: "https://github.com/px6707/myblog"
draft: false
---

# LangChain 核心机制教程：Runnable、并发、生命周期与人工审批

本文以 **Python 版 LangChain / LangGraph v1 系列**为语境；安装后仍应以所用版本的 API 文档为准。文中审批示例只修改图状态，不会真的发送邮件。

## 先建立一张地图

- **LangChain Core / LCEL**：`Runnable`、提示词、模型、解析器以及 `|` 组合，适合构建可组合的数据处理链。
- **LangChain Agent**：模型决定何时调用工具；`create_agent` 的中间件负责在模型或工具调用周围插入策略。
- **LangGraph**：显式状态、节点、检查点和中断，适合需要暂停、恢复、人工审批的工作流。
- **前端**：展示消息与审批表单；不能仅凭前端按钮保证权限或阻止重复执行。

这些概念有关联，但不是同一个抽象：**Runnable 的 `batch` 并发、Agent 的循环、中间件钩子、LangGraph 的中断**分别解决不同问题。

## 一、Runnable：统一执行协议

`Runnable` 把输入转换成输出，常用方法如下：

| 方法 | 含义 |
| --- | --- |
| `invoke` / `ainvoke` | 单次同步 / 异步调用 |
| `batch` / `abatch` | 批量处理；默认实现适合 I/O 密集型任务 |
| `stream` / `astream` | 以迭代方式取得输出 |

注意：**有 `stream()` 方法不代表每一步都会实时吐出 token**。如果链中某一步不能对上游的输出块做流式转换，后续输出要等那一步完成才开始。`RunnableLambda` 默认就是这种可能阻断逐块传递的步骤。参见 [Runnable](https://reference.langchain.com/python/langchain-core/runnables/base/Runnable) 与 [RunnableSequence](https://reference.langchain.com/python/langchain-core/runnables/base/RunnableSequence)。

### 1.1 `RunnableLambda` 与顺序组合

以下示例不需要模型或 API Key：

```python
from langchain_core.runnables import RunnableLambda

add_one = RunnableLambda(lambda x: x + 1)
square = RunnableLambda(lambda x: x * x)

chain = add_one | square
assert chain.invoke(4) == 25
assert chain.batch([1, 2, 3]) == [4, 9, 16]
```

`|` 创建的是顺序执行链：上一段的输出成为下一段的输入。`RunnableLambda` 很适合普通函数适配；若要**边接收上游块、边输出下游块**，使用 `RunnableGenerator` 或实现 `transform`。

### 1.2 `RunnableParallel`：同一输入分发给多个分支

```python
from langchain_core.runnables import RunnableLambda, RunnableParallel

branches = RunnableParallel(
    double=RunnableLambda(lambda x: x * 2),
    square=RunnableLambda(lambda x: x * x),
)
assert branches.invoke(3) == {"double": 6, "square": 9}

# 在顺序链中，字典也会被转换为并行分支。
chain = RunnableLambda(lambda x: x + 1) | {
    "double": RunnableLambda(lambda x: x * 2),
    "square": RunnableLambda(lambda x: x * x),
}
assert chain.invoke(3) == {"double": 8, "square": 16}
```

这里的“并行”是分支可并发执行，**不保证完成顺序**。最终字典按键汇总结果；它不是让单个模型调用内部变快的魔法。两个分支若都调用付费模型，成本通常也会增加。

### 1.3 `RunnableGenerator`：逐块处理

```python
from typing import Iterator
from langchain_core.runnables import RunnableGenerator

def label_chunks(chunks: Iterator[str]) -> Iterator[str]:
    for chunk in chunks:
        yield f"[{chunk}]"

labeler = RunnableGenerator(label_chunks)
assert list(labeler.transform(iter(["你", "好"]))) == ["[你]", "[好]"]

# 单次完整输入仍可调用 stream；这里只会得到一个输出块。
assert list(labeler.stream("你好")) == ["[你好]"]
```

`RunnableGenerator` 的重要能力是接收上游迭代器；若上游本身只产出一个完整字符串，它不会凭空创造逐 token 的流。接模型时通常可组合为 `prompt | model | StrOutputParser() | labeler`。参见 [RunnableGenerator 参考](https://reference.langchain.com/python/langchain-core/runnables/base)。

### 1.4 自定义 Runnable：何时才需要

简单逻辑优先用 `RunnableLambda`。确需自定义类、序列化配置或优化批量接口时，再考虑继承 `RunnableSerializable`。通常只实现 `invoke` 即可使用默认 `batch`；**不要仅为展示并发而重写 `batch`**，否则还需自己处理 `RunnableConfig`、异常策略、最大并发量和异步语义。默认 `batch` 使用线程池；可以通过调用时的配置限制并发，例如 `chain.batch([1, 2, 3], config={"max_concurrency": 2})`。

如果底层服务有真正的批处理 API，覆盖 `batch` 才可能带来显著收益。参见 [batch 参考](https://reference.langchain.com/python/langchain-core/runnables/base/Runnable/batch)。

## 二、并发、完成顺序与结果顺序

Python 标准库的 `as_completed` 会按任务**完成顺序**取回 Future；LangChain 的 `batch` 通常按**输入顺序**返回结果。这是两种不同需求。

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import sleep

def work(seconds: float) -> float:
    sleep(seconds)
    return seconds

inputs = [0.3, 0.1, 0.2]
ordered: list[float | None] = [None] * len(inputs)

with ThreadPoolExecutor(max_workers=3) as pool:
    futures = {pool.submit(work, value): index for index, value in enumerate(inputs)}
    for future in as_completed(futures):
        ordered[futures[future]] = future.result()

assert ordered == inputs
```

这个例子只是解释索引映射，不表示每个 LangChain 项目都应该手写线程池。实际项目中还要考虑：并发上限、模型服务限流、超时、部分失败、取消和成本。模型调用属于外部 I/O，**同时请求越多不一定越快或越便宜**。

## 三、Agent 中间件生命周期

以下钩子属于 **LangChain `create_agent` 的中间件**，不是所有 Runnable 链都会自动触发。根据当前[自定义中间件文档](https://docs.langchain.com/oss/python/langchain/middleware/custom)：

| 钩子 | 触发时机 | 常见用途 |
| --- | --- | --- |
| `before_agent` | 一次 Agent 调用开始前 | 初始化本次调用的状态、审计上下文 |
| `before_model` | 每次模型调用前 | 修剪上下文、预算检查、输入校验 |
| `after_model` | 每次模型响应后 | 检查完整响应、记录用量 |
| `after_agent` | 一次 Agent 正常完成后 | 最终结果处理 |
| `wrap_model_call` | 包住每次模型调用 | 重试、降级、缓存、动态模型选择 |
| `wrap_tool_call` | 包住每次工具调用 | 工具权限门控、超时、结果处理 |

几点容易混淆：

1. `before_agent` 的“一次”指**一次 Agent invocation**，不是应用启动时一次，也不是自然语言意义上的“用户 Query”——一次请求可以包含多次模型调用。
2. `after_agent` 不应被当成无条件执行的 `finally`。异常、取消或中断场景若必须清理资源，应在外层请求生命周期或 `try/finally` 中保证。
3. `wrap_model_call` 并非“不能检查输出”；它拿得到模型响应。但如果规则依赖**完整响应**，`after_model` 往往更直观。无论选哪个钩子，都不能声称它能自动撤回已经流给用户的内容；需要严格防泄露时，应在对外发送前设计独立的审核边界。
4. 对有副作用的工具，不要盲目自动重试。重试前要弄清上一次是否已经执行成功，并使用幂等键或状态查询。

一个简化的模型前检查：

```python
from langchain.agents.middleware import AgentState, before_model
from langgraph.runtime import Runtime

@before_model
def log_message_count(state: AgentState, runtime: Runtime) -> None:
    print("本次模型调用前的消息数：", len(state["messages"]))
    return None

# 在 create_agent(..., middleware=[log_message_count]) 中使用。
```

真实系统的日志应避免直接打印提示词、用户输入、密钥及敏感业务数据。

## 四、人工审批：LangGraph 的中断与恢复

`interrupt()` 会让图暂停，并把指定的 JSON 可序列化数据交给调用方；同一个 `thread_id` 与检查点用于恢复。恢复时，`Command(resume=...)` 的值会成为 `interrupt()` 的返回值。**恢复会从发生中断的节点开头重新运行**，因此中断前的副作用可能重复；把写操作放在审批后，并让外部写操作本身具备幂等保护。参见 [Interrupts 官方文档](https://docs.langchain.com/oss/python/langgraph/interrupts)。

### 4.1 最小可运行示例：审批一封“模拟邮件”

```python
from typing import TypedDict
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

class MailState(TypedDict):
    to: str
    content: str
    status: str

def review(state: MailState) -> dict[str, str]:
    decision = interrupt({
        "action": "send_email",
        "to": state["to"],
        "content": state["content"],
        "allowed_decisions": ["approve", "reject"],
    })
    if decision not in ("approve", "reject"):
        raise ValueError("审批结果只能是 approve 或 reject")
    return {"status": decision}

def route_after_review(state: MailState) -> str:
    if state["status"] == "approve":
        return "execute"
    if state["status"] == "reject":
        return "rejected"
    raise ValueError("未经批准，禁止执行")

def execute(state: MailState) -> dict[str, str]:
    # 教学示例：这里不调用真实邮件服务。
    # 实际发送前还必须验证收件人、审批权限，并使用幂等键防重。
    return {"status": f"would_send_to:{state['to']}"}

def rejected(state: MailState) -> dict[str, str]:
    return {"status": "rejected"}

builder = StateGraph(MailState)
builder.add_node("review", review)
builder.add_node("execute", execute)
builder.add_node("rejected", rejected)
builder.add_edge(START, "review")
builder.add_conditional_edges("review", route_after_review)
builder.add_edge("execute", END)
builder.add_edge("rejected", END)

graph = builder.compile(checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": str(uuid4())}}

paused = graph.invoke(
    {"to": "team@example.com", "content": "会议已改期", "status": "pending"},
    config=config,
)
assert "__interrupt__" in paused  # 默认 invoke() 的返回形式
print("待审批：", paused["__interrupt__"][0].value)

# 同一 thread_id 恢复；把下面的 approve 改为 reject 可测试拒绝分支。
finished = graph.invoke(Command(resume="approve"), config=config)
print("结果：", finished["status"])
```

安装依赖：`pip install -U langchain-core langgraph`。本例使用 `InMemorySaver`，**仅适合演示**：进程重启后状态会丢失，生产环境需持久化 checkpointer。若要观察拒绝分支，需使用**新的 `thread_id`** 重新调用初始输入，再以 `Command(resume="reject")` 恢复；不要在已完成的同一轮上重复提交。

### 4.2 前后端如何对接

上例展示的是**通用 `interrupt()`**。其中的 `action`、`allowed_decisions` 是本教程自定义的业务数据，**不是** LangChain 内置 `HumanInTheLoopMiddleware` 的固定响应格式。不要把两套格式混用。

接口可以设计为：

1. `POST /runs`：后端从认证上下文确定用户，生成 `run_id` / `thread_id`，调用图并返回 `completed` 或 `waiting_for_approval`、待审批内容。
2. 前端展示**具体动作、参数和影响范围**，让用户批准或拒绝；不能仅显示“是否继续”。
3. `POST /runs/{run_id}/decision`：后端验证当前用户是否有权审批该 run，检查其仍处于待审批状态，然后用保存的 `thread_id` 执行 `Command(resume=...)`。
4. 若网络超时，前端通过 `GET /runs/{run_id}` 查询真实状态；**不要直接再次提交批准**。恢复接口应有幂等或去重策略；检查点只负责保存图状态，不保证外部邮件、数据库写入等副作用恰好执行一次。

不要直接信任前端传来的 `user_id` 或任意 `thread_id`；它们必须与服务端认证身份和授权关系绑定。原教程的 FastAPI 示例写死了邮件内容、忽略用户输入、用内存字典按用户保存唯一线程，还通过匹配异常字符串识别中断，均不宜照搬。常规 `graph.invoke()` 会在返回值的 `__interrupt__` 暴露中断；若使用 `stream_events(..., version="v3")`，则通过 `stream.interrupted` / `stream.interrupts` 读取，不需要用异常字符串判断。

### 4.3 原生 HITL 中间件与手写中断

- 已经使用 LangChain `create_agent`，希望对特定工具调用做批准、编辑或拒绝：优先评估其 **Human-in-the-Loop middleware**。
- 自己编排多节点审批、表单补充或跨系统工作流：直接使用 **LangGraph `interrupt()`** 更灵活。

两者都不能替代后端鉴权、业务校验与幂等处理。参见 [LangChain HITL 文档](https://docs.langchain.com/oss/python/langchain/human-in-the-loop)。

## 五、创建工具的三种方式

| 方式 | 适合情况 | 提醒 |
| --- | --- | --- |
| `@tool` | 普通函数快速暴露为工具 | 给模型看的名称、说明和参数描述要清晰 |
| `StructuredTool.from_function()` | 已有函数不便改动，或要组合同步/异步实现、显式参数 schema | 不是 `@tool` 的“高级替代品”，只是组装形式不同 |
| 继承 `BaseTool` | 需要更复杂的类封装或重写执行细节 | 注入数据库等资源时注意生命周期与序列化 |

### 5.1 `@tool`

```python
from langchain_core.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    """计算两个整数的乘积。"""
    return a * b

assert multiply.invoke({"a": 5, "b": 3}) == 15
```

### 5.2 `StructuredTool.from_function()`

```python
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

class MultiplyInput(BaseModel):
    a: int = Field(description="第一个乘数")
    b: int = Field(description="第二个乘数")

def multiply_impl(a: int, b: int) -> int:
    return a * b

multiply_tool = StructuredTool.from_function(
    func=multiply_impl,
    name="multiply",
    description="计算两个整数的乘积",
    args_schema=MultiplyInput,
)
assert multiply_tool.invoke({"a": 4, "b": 7}) == 28
```

如果需要异步实现，可以给 `coroutine=` 传入 `async def` 函数；`await multiply_tool.ainvoke(...)` 必须放在异步函数或支持顶层 `await` 的环境中。

### 5.3 继承 `BaseTool`

```python
from typing import Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

class MultiplyInput(BaseModel):
    a: int = Field(description="第一个乘数")
    b: int = Field(description="第二个乘数")

class MultiplyTool(BaseTool):
    name: str = "multiply"
    description: str = "计算两个整数的乘积"
    args_schema: Type[BaseModel] = MultiplyInput

    def _run(self, a: int, b: int) -> int:
        return a * b

tool_instance = MultiplyTool()
assert tool_instance.invoke({"a": 6, "b": 8}) == 48
```

对有副作用的工具（邮件、转账、删除、改库）还应在**工具执行层**检查权限、校验参数、记录审计并防止重复执行；模型生成了合法 schema，不等于可以执行。参见 [工具文档](https://docs.langchain.com/oss/python/langchain/tools)。

## 六、从教学代码到生产系统

1. **安全**：后端认证、按 run 授权；模型提示词和前端按钮都不能代替权限控制。
2. **持久化**：生产使用持久化检查点，保存运行状态；同一用户可以有多个 run，不要用“一个用户一个线程”的内存字典。
3. **幂等**：审批重试、工具超时重试、进程恢复都可能导致重复请求。把审批 ID 与业务操作 ID 传到真正执行外部写入的服务。
4. **数据边界**：结构化校验、业务规则校验、人工审批是三道不同防线。
5. **可观测性**：记录 run_id、节点、工具结果、耗时、错误与审批决策；敏感输入、密钥和用户数据应按策略脱敏，不直接写日志。
6. **测试**：至少覆盖批准、拒绝、非法决策、同一审批重复提交、恢复时节点重跑、工具执行失败和进程重启后的恢复。

## 参考资料

- [Runnable API 参考](https://reference.langchain.com/python/langchain-core/runnables/base/Runnable)
- [RunnableSequence 与流式语义](https://reference.langchain.com/python/langchain-core/runnables/base/RunnableSequence)
- [LangChain 自定义中间件](https://docs.langchain.com/oss/python/langchain/middleware/custom)
- [LangGraph 中断与恢复](https://docs.langchain.com/oss/python/langgraph/interrupts)
- [LangChain 工具](https://docs.langchain.com/oss/python/langchain/tools)
- [LangChain 人工审批](https://docs.langchain.com/oss/python/langchain/human-in-the-loop)
