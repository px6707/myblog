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

**`Runnable`** 是 LangChain Core 里的统一执行抽象：任意「接收输入 → 产生输出」的单元（提示词、模型、解析器、普通函数包装等）都实现同一套接口，因而可以用 `|` 串成链，并用同一组方法调用。

常用方法如下：

| 方法 | 含义 |
| --- | --- |
| `invoke` / `ainvoke` | 单次同步 / 异步调用 |
| `batch` / `abatch` | 批量处理；默认实现适合 I/O 密集型任务 |
| `stream` / `astream` | 以迭代方式取得输出 |

注意：**有 `stream()` 方法不代表每一步都会实时吐出 token**。如果链中某一步不能对上游的输出块做流式转换，后续输出要等那一步完成才开始。`RunnableLambda` 默认就是这种可能阻断逐块传递的步骤。参见 [Runnable](https://reference.langchain.com/python/langchain-core/runnables/base/Runnable) 与 [RunnableSequence](https://reference.langchain.com/python/langchain-core/runnables/base/RunnableSequence)。

### 1.1 `RunnableLambda` 与顺序组合

**`RunnableLambda`** 把普通 Python 函数（或 `lambda`）包成一个 `Runnable`，使其也能 `invoke` / `batch` / `|` 组合。它本身不调用模型，只做你写的那段同步变换；默认按「整段输入 → 整段输出」工作，一般不会把上游的 token 流逐块往下传。

以下示例不需要模型或 API Key：

```python
from langchain_core.runnables import RunnableLambda

add_one = RunnableLambda(lambda x: x + 1)
square = RunnableLambda(lambda x: x * x)

chain = add_one | square
assert chain.invoke(4) == 25
assert chain.batch([1, 2, 3]) == [4, 9, 16]
```

`|` 创建的是顺序执行链：上一段的输出成为下一段的输入。`RunnableLambda` 很适合把已有函数接进 LCEL；若要**边接收上游块、边输出下游块**，使用 `RunnableGenerator` 或实现 `transform`。

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

以下钩子属于 **LangChain `create_agent` 的中间件**，不是所有 Runnable 链都会自动触发。根据当前[自定义中间件文档](https://docs.langchain.com/oss/python/langchain/middleware/custom)与官方对中间件场景的说明：

| 钩子 | 触发时机 | 典型场景 |
| --- | --- | --- |
| `before_agent` | 一次 Agent 调用开始前（整次 invocation 只一次） | 加载长期记忆 / 用户画像；校验首包输入；初始化本次会话资源（如官方 Shell 中间件在进循环前打开 shell）；写入「run 开始」审计 |
| `before_model` | 每次模型调用前 | 消息过长时摘要或裁剪（如 SummarizationMiddleware）；去 PII；改写 / 注入 system prompt；检查 token 预算；拦截违规输入 |
| `after_model` | 每次模型响应后、工具执行前 | 输出护栏 / 内容审核；再扫一遍 PII；记录用量；人工审批「是否执行接下来的 tool_calls」（HITL 常落在这里） |
| `after_agent` | 一次 Agent **正常完成**后（整次只一次） | 落库最终答案；发送完成通知；释放 `before_agent` 中打开的资源 |
| `wrap_model_call` | 包住每次模型调用 | 超时重试；主模型失败切备用；响应缓存；按用户等级动态换模型或换工具列表；用小模型先筛工具再绑定（如 LLMToolSelector） |
| `wrap_tool_call` | 包住每次工具调用 | 权限门控（谁能发邮件）；参数校验；超时与限流；幂等 / 防重；截断或改写超长工具结果；记审计日志 |

粗分：`before_*` / `after_*` 偏「前后钩子改状态或做检查」；`wrap_*` 偏「包一层，决定要不要真正调用、失败怎么兜底」。

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

人工审批要拆成「后端跑图 / 存状态」和「前端展示并回传决策」两段。一次典型往返如下：

```text
前端 POST /runs
  → 后端鉴权后生成 run_id、thread_id，invoke 图
  → 若命中 interrupt：返回 waiting_for_approval + 待审批内容
  → 若跑完：返回 completed + 结果

前端展示待审批详情，用户选择批准 / 拒绝
  → POST /runs/{run_id}/decision
  → 后端校验权限与「仍待审批」后，用保存的 thread_id 执行 Command(resume=...)
  → 返回最终结果或下一轮等待

网络超时或页面刷新时：GET /runs/{run_id} 查真实状态，不要直接再点一次「批准」
```

#### 后端：用代码区分「要审批」还是「已完成」

```python
# 伪代码：API 层把 LangGraph 结果映射成对外 status
# waiting_for_approval / completed 是你定义的业务字段，不是图内置返回值
#
# thread_id 能跨请求恢复，前提是 compile 时挂了 checkpointer：
#   演示可用 InMemorySaver；生产常用 PostgresSaver 等持久化实现。
# checkpointer 存图的 checkpoint；save_run 仍要存业务 run 索引（见下）。

from uuid import uuid4
from langgraph.checkpoint.memory import InMemorySaver  # 或 PostgresSaver
from langgraph.types import Command

# 模块加载时编译一次；无 checkpointer 则 interrupt 后无法靠 thread_id 恢复
checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)


def create_run(user, payload):
    run_id = str(uuid4())
    thread_id = str(uuid4())
    # thread_id 是 checkpointer 的检索键：本次 invoke 写入的 checkpoint 挂在这个键下
    config = {"configurable": {"thread_id": thread_id}}

    # 用户身份只来自服务端认证上下文，不要用前端传来的 user_id
    result = graph.invoke(payload, config=config)

    interrupts = result.get("__interrupt__") or []
    if interrupts:
        approval = interrupts[0].value  # 即 interrupt({...}) 传入的 JSON
        # save_run：业务表；图状态已由 checkpointer 按 thread_id 保存
        save_run(run_id, user_id=user.id, thread_id=thread_id,
                 status="waiting_for_approval", approval=approval)
        return {"run_id": run_id, "status": "waiting_for_approval", "approval": approval}

    # 没有 __interrupt__：图已跑完（checkpoint 通常仍保留，不会自动删 thread_id）
    save_run(run_id, user_id=user.id, thread_id=thread_id,
             status="completed", state=result)
    return {"run_id": run_id, "status": "completed", "state": result}


def decide_run(user, run_id, decision):
    run = load_run(run_id)
    assert run.user_id == user.id          # 鉴权：可否审批该 run
    assert run.status == "waiting_for_approval"
    assert decision in ("approve", "reject")
    # 幂等：已处理过的 decision 直接返回上次结果，不要再 resume

    # 同一 thread_id → checkpointer 取出暂停时的 checkpoint，再 resume
    config = {"configurable": {"thread_id": run.thread_id}}
    result = graph.invoke(Command(resume=decision), config=config)

    interrupts = result.get("__interrupt__") or []
    if interrupts:
        approval = interrupts[0].value
        update_run(run_id, status="waiting_for_approval", approval=approval)
        return {"run_id": run_id, "status": "waiting_for_approval", "approval": approval}

    update_run(run_id, status="completed", state=result)
    return {"run_id": run_id, "status": "completed", "state": result}


def get_run(user, run_id):
    run = load_run(run_id)
    assert run.user_id == user.id
    # 超时/刷新后前端只调这个对齐状态，不要盲目再 POST decision
    return {
        "run_id": run_id,
        "status": run.status,       # waiting_for_approval | completed | ...
        "approval": run.approval,   # 待审批时才有
        "state": run.state,         # 完成时才有
    }
```

流式时同样靠「有没有中断」分支，不要解析异常字符串：

```python
# 使用 stream_events(..., version="v3") 时，中断在返回的 stream 对象上：
#   stream.interrupted  → bool
#   stream.interrupts   → 本次 pause 的 Interrupt 列表（.value 即 interrupt({...}) 的载荷）
# 这两个是 LangGraph 事件流 API 的字段，不是下面这种自造函数。

def create_run_streaming(user, payload):
    run_id = str(uuid4())
    thread_id = str(uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # 同样要求 graph = builder.compile(checkpointer=...)
    stream = graph.stream_events(payload, config=config, version="v3")

    # messages 会边产生边迭代；循环结束 = 本次运行已停（跑完或 interrupt 暂停）
    # 不是「先攒齐所有 token 再发给前端」，而是消费完本轮投影后再看 interrupted
    for message in stream.messages:
        for token in message.text:
            yield {"type": "token", "text": token}

    # 到这里 stream.messages 已耗尽：要么 completed，要么因 interrupt() 暂停
    if stream.interrupted:
        approval = stream.interrupts[0].value
        save_run(run_id, user_id=user.id, thread_id=thread_id,
                 status="waiting_for_approval", approval=approval)
        yield {"type": "status", "status": "waiting_for_approval", "approval": approval}
    else:
        save_run(run_id, user_id=user.id, thread_id=thread_id,
                 status="completed", state=stream.output)
        yield {"type": "status", "status": "completed", "state": stream.output}
```

#### 前端：用代码决定渲染流式区还是审批组件

```ts
// 伪代码：只认后端显式 status / 事件类型，不要猜「是不是还要调工具」

type RunResponse = {
  run_id: string
  status: "waiting_for_approval" | "completed"
  approval?: { action: string; to: string; content: string; allowed_decisions: string[] }
  state?: unknown
}

function renderByStatus(res: RunResponse) {
  if (res.status === "waiting_for_approval") {
    hideStreamingCursor()
    // 审批 UI 的数据必须来自 res.approval，不要前端自己编一份
    showApprovalPanel(res.approval!)
    return
  }
  if (res.status === "completed") {
    hideApprovalPanel()
    showFinalResult(res.state)
  }
}

// 非流式：一次 POST 到底
async function startRun(payload: unknown) {
  const res = await postJSON<RunResponse>("/runs", payload)
  renderByStatus(res)
}

// 先流式、后审批：同一条事件流里切换 UI
// msg.type 不是框架自带字段，而是后端 SSE 按约定推送的信封，例如前面 create_run_streaming 里：
//   yield {"type": "token", "text": ...}
//   yield {"type": "status", "status": "waiting_for_approval" | "completed", ...}
async function startRunStreaming(payload: unknown) {
  for await (const msg of openEventStream("/runs/stream", payload)) {
    if (msg.type === "token") {
      appendStreamingText(msg.text)          // 渲染流式输出
      continue
    }
    if (msg.type === "status" && msg.status === "waiting_for_approval") {
      hideStreamingCursor()
      showApprovalPanel(msg.approval)        // 切换审批组件
      continue
    }
    if (msg.type === "status" && msg.status === "completed") {
      hideApprovalPanel()
      showFinalResult(msg.state)
    }
  }
}

async function submitDecision(runId: string, decision: "approve" | "reject") {
  const res = await postJSON<RunResponse>(`/runs/${runId}/decision`, { decision })
  renderByStatus(res)
}

async function recoverAfterTimeout(runId: string) {
  // 不确定时只 GET，禁止直接再点批准
  const res = await getJSON<RunResponse>(`/runs/${runId}`)
  renderByStatus(res)
}
```

对接时还要注意：

- 不要信任前端传来的 `user_id` 或任意 `thread_id`，必须与服务端身份和授权绑定。
- 上例 `interrupt({...})` 里的 `action`、`allowed_decisions` 是**本教程自定义的业务字段**，不是 `HumanInTheLoopMiddleware` 的固定协议；两套格式不要混用。
- 用 `graph.invoke()` 时，中断出现在返回值的 `__interrupt__`；用 `stream_events(..., version="v3")` 时看 `stream.interrupted` / `stream.interrupts`，不要靠匹配异常字符串判断是否中断。

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
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

class MultiplyInput(BaseModel):
    a: int = Field(description="第一个乘数")
    b: int = Field(description="第二个乘数")

class MultiplyTool(BaseTool):
    name: str = "multiply"
    description: str = "计算两个整数的乘积"
    args_schema: Type[BaseModel] = MultiplyInput

    # --- 必须：同步实现。tool.invoke(...) / tool.run(...) 最终进这里 ---
    def _run(
        self,
        a: int,
        b: int,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> int:
        # run_manager 可选：需要时向上抛进度/日志；多数业务工具可忽略
        return a * b

    # --- 可选：原生异步。tool.ainvoke(...) / await tool.arun(...) 优先进这里 ---
    # 不重写时，框架默认：await run_in_executor(None, self._run, ...)
    # 即把上面的 _run 丢进线程池，事件循环不堵，但并不是真异步 I/O
    async def _arun(
        self,
        a: int,
        b: int,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> int:
        # 有 AsyncClient / 异步 DB 时在这里 await；本例无 I/O，直接算即可
        return a * b

    # --- 一般不要为了写业务去重写这些对外入口 ---
    # 它们负责：参数校验 / args_schema、回调与 tracing、错误处理策略、再转到 _run/_arun。
    # 业务写进 invoke/run 会绕开上述封装，Agent、中间件、可观测性也容易对不上。
    # 工具逻辑只放 _run / _arun；需要改调用行为时用属性（如 handle_tool_error）或中间件。
    # def invoke(self, input, config=None, **kwargs): ...
    # async def ainvoke(self, input, config=None, **kwargs): ...
    # def run(self, tool_input, ...): ...
    # async def arun(self, tool_input, ...): ...

tool_instance = MultiplyTool()
assert tool_instance.invoke({"a": 6, "b": 8}) == 48
# assert await tool_instance.ainvoke({"a": 6, "b": 8}) == 48  # 在 async 函数里调用
```

对有副作用的工具（邮件、转账、删除、改库）还应在**工具执行层**（`_run` / `_arun` 内）检查权限、校验参数、记录审计并防止重复执行；模型生成了合法 schema，不等于可以执行。参见 [工具文档](https://docs.langchain.com/oss/python/langchain/tools)。

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
