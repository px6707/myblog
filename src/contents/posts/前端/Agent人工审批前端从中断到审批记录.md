---
title: Agent 人工审批前端：从中断到可保留的审批卡片
published: 2026-06-21
description: 用 Vue 3 理解 Deep Agents 的人工审批、多个待审动作、自定义中断表单，以及审批后保留卡片的实现。
tags: [Vue, LangChain, Deep Agents, Agent, HITL]
category: 前端
licenseName: "Unlicensed"
author: panxiao
sourceLink: "https://docs.langchain.com/oss/python/langchain/frontend/human-in-the-loop"
draft: false
---

一个 Agent 准备发送邮件时，前端不能只显示“正在调用工具”。用户需要看到收件人、主题和正文，决定批准还是拒绝；作出决定后，还应该能回看自己批准了什么。这就是 Agent 的 **Human-in-the-Loop（HITL，人工介入）** 界面。

本文接着《给 Deep Agents 做一个看得懂的 Vue 前端》的技术路线：Python Deep Agent 运行在 Agent Server 上，Vue 3 通过 `@langchain/vue` v1 的 `useStream` 查看运行状态。重点是审批的数据流和前端交互，不涉及完整的服务部署。

## 一次审批实际发生了什么

```text
Agent 准备调用受保护的工具
        ↓
后端保存检查点并发出 interrupt
        ↓
Vue 展示动作与参数，等待用户决定
        ↓
前端提交决定，同一个 thread 恢复运行
        ↓
工具执行、被修改后执行，或被拒绝
```

`interrupt` 是一个暂停点。它把待审批信息交给前端，同时保存可恢复的运行状态；用户的决定会在恢复时返回给 Agent。后端需要检查点，前端也要保留同一 thread 的身份，否则无法从正确的位置继续。[LangGraph 中断说明](https://docs.langchain.com/oss/python/langgraph/interrupts)

## 路线一：审批普通工具调用

Deep Agents 可以用 `interrupt_on` 指定哪些工具在执行前等待审批。例如邮件工具只允许批准或拒绝：

```python
from deepagents import create_deep_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import MemorySaver

@tool
def notify_email(to: str, subject: str, body: str) -> str:
    """发送邮件。"""
    # 示例中省略实际邮件服务调用
    return f"已向 {to} 发送邮件"

agent = create_deep_agent(
    model="openai:gpt-5.5",
    tools=[notify_email],
    interrupt_on={
        "notify_email": {"allowed_decisions": ["approve", "reject"]},
    },
    checkpointer=MemorySaver(),
)
```

`MemorySaver` 便于本地理解流程；生产环境应使用持久化检查点。审批保护的是**工具真正执行之前**的边界，前端按钮本身不是权限校验。后端仍要检查用户身份、资源权限与最终参数。[Deep Agents 人工审批](https://docs.langchain.com/oss/python/deepagents/human-in-the-loop)

内置 HITL 的中断值含两组按顺序对应的数据：

| 字段 | 含义 |
| --- | --- |
| `actionRequests` | 待执行工具的名称、参数与说明 |
| `reviewConfigs` | 每个动作允许哪些决定 |

前端先展示动作，再按对应的 `allowedDecisions` 绘制按钮。不要只根据工具名猜测用户是否能编辑参数。[前端 HITL 模式](https://docs.langchain.com/oss/python/langchain/frontend/human-in-the-loop)

```ts
import { computed } from "vue";
import { useStream } from "@langchain/vue";

type Decision =
  | { type: "approve" }
  | { type: "reject"; message: string };

type HITLRequest = {
  actionRequests: Array<{
    name: string;
    args: Record<string, unknown>;
    description?: string;
  }>;
  reviewConfigs: Array<{
    allowedDecisions: Array<"approve" | "reject" | "edit" | "respond">;
  }>;
};

const stream = useStream({
  apiUrl: import.meta.env.VITE_AGENT_URL ?? "http://localhost:2024",
  assistantId: "agent",
});

const pending = computed(
  () => stream.interrupt.value?.value as HITLRequest | undefined,
);

async function submitDecisions(decisions: Decision[]) {
  const request = pending.value;
  if (!request || decisions.length !== request.actionRequests.length) return;
  await stream.respond({ decisions });
}
```

这个 `Decision` 类型只覆盖本例允许的批准与拒绝。若后端也允许 `edit`，要提交完整的 `editedAction`；`respond` 则是**由人代替工具提供结果**，不能拿它冒充拒绝。一次中断若有三个 `actionRequests`，就要按原顺序提交三个决定；不要点第一张卡就单独恢复整次运行。[前端 HITL 模式](https://docs.langchain.com/oss/python/langchain/frontend/human-in-the-loop)

界面可以让用户先为每个动作选择决定，在收齐全部决定后统一调用 `submitDecisions`。提交期间禁用按钮；失败时保留待审卡片和错误提示，避免用户以为已经批准成功。

## 路线二：工具自己定义审批表单

有些动作需要的信息超出“批准 / 拒绝”。例如订机票时，还要选择舱位和保险。此时可以在工具内部调用 `interrupt()`，把表单的 `formType`、标题、上下文和字段传给前端。恢复时，`interrupt()` 的返回值就是用户提交的决定。[自定义中断表单](https://docs.langchain.com/oss/python/langchain/frontend/human-in-the-loop)

```ts
// 后端工具内部的关键片段（TypeScript）
const decision = interrupt<InterruptCard, ReviewDecision>({
  formType: "flight-booking",
  title: "确认订票",
  context: { origin, destination, date },
  fields: [
    { name: "seatClass", label: "舱位", type: "select" },
  ],
});

if (!decision.approved) return "订票已取消";
// 只有批准后才执行真正的订票操作
```

这个示例展示的是中断的数据形状；`InterruptCard`、`ReviewDecision`、工具定义和检查点仍需在后端项目中实现。Python 后端也能用 `langgraph.types.interrupt()` 实现同样的暂停与恢复流程，前后端只需对齐 JSON 协议。**自定义表单的返回值由你定义，不要直接套用上一节内置 HITL 的 `actionRequests / decisions` 格式。**[LangGraph 中断说明](https://docs.langchain.com/oss/python/langgraph/interrupts)

## 为什么点完批准，卡片会消失

待审卡片通常这样渲染：

```vue
<ApprovalCard v-if="stream.interrupt.value" :card="stream.interrupt.value.value" />
```

用户提交决定后，运行恢复，`stream.interrupt` 清空。上面的 `v-if` 随之变为假，卡片就从页面消失；如果工具接下来还要运行几秒，用户会暂时看不到刚才批准的内容。这是**界面状态来源变化**，不是审批记录一定被删除。

解决办法是让卡片从“当前中断”转成“消息历史里的一条已处理记录”：

```text
提交前：stream.interrupt → 可编辑的审批卡片
提交后：stream.messages  → 只读的审批卡片
```

官方自定义表单示例把已处理卡片放进一条 `AIMessage` 的 `response_metadata`，然后同时恢复中断并更新消息状态：[将卡片保持在屏幕上](https://docs.langchain.com/oss/python/langchain/frontend/human-in-the-loop)

```ts
import { AIMessage } from "langchain";

type ReviewDecision = { approved: boolean; values?: Record<string, unknown> };
type InterruptCard = {
  title: string;
  context: Record<string, unknown>;
  resolved?: boolean;
  decision?: ReviewDecision;
};

async function resolveCard(card: InterruptCard, decision: ReviewDecision) {
  const resolvedCard = { ...card, resolved: true, decision };
  const cardMessage = new AIMessage({
    content: decision.approved ? "审批已通过" : "审批已拒绝",
    response_metadata: { cards: resolvedCard },
  });

  await stream.respond(decision, {
    update: { messages: [cardMessage] },
  });
}
```

这里的两个参数各有职责：`decision` 回到暂停的工具，让它继续或取消；`update.messages` 将审批回执写进线程状态。SDK 会先乐观显示这条消息，再与恢复后的状态对齐。界面从 `response_metadata.cards` 识别它，并用只读卡片渲染：

```ts
function cardFromMessage(message: AIMessage) {
  return (message.response_metadata as { cards?: InterruptCard }).cards;
}
```

只有后端状态确实保存了这条消息、线程检查点可恢复时，刷新后才能继续看到卡片。这套 `respond(decision, { update })` 写法针对**自定义中断表单**；普通内置 HITL 审批可以先按上一节完成闭环，再决定是否需要持久化的专用卡片。[官方前端示例](https://docs.langchain.com/oss/python/langchain/frontend/human-in-the-loop)

## 上线前检查这五件事

1. **动作先于决定**：完整展示工具名、目标资源和参数，让用户知道自己在批准什么。
2. **决定与动作一一对应**：多个动作按顺序收齐决定；按钮只显示后端允许的类型。
3. **拒绝有明确语义**：拒绝时说明原因，避免 Agent 把失败误认为可以再次尝试同一操作。
4. **恢复可追踪**：同一个 thread、可用检查点、提交中的状态，以及失败后重试的界面都要明确。
5. **回执可回看**：如果产品需要审计，保存决定、操作者与时间；审批卡片持久化只是可视化的一部分。

人工审批最重要的边界是：**后端决定何时暂停和是否执行工具，前端负责让人看清动作、作出决定，并准确展示结果。**
