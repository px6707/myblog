---
title: 给 Deep Agents 做一个看得懂的 Vue 前端
published: 2026-06-20
description: 用 Vue 3 展示 Deep Agents 的流式 Markdown、待办、子 Agent、工具调用与浏览器无头工具。
tags: [Vue, LangChain, Deep Agents, Agent]
category: 前端
licenseName: "Unlicensed"
author: panxiao
sourceLink: "https://docs.langchain.com/oss/python/deepagents/frontend/overview"
draft: false
---

Agent 跑一个复杂任务时，最后一句“已完成”往往不够。用户还想知道它正在做哪一步、把工作交给了谁、为什么停下来等待确认，以及交付的文件改了什么。Deep Agents 前端的核心，就是把这些运行信息变成可以阅读和操作的界面。

本文以 **Python `create_deep_agent` + LangGraph Agent Server + Vue 3 + `@langchain/vue` v1** 为例，搭建一条从后端状态到前端视图的路线。代码是关键接线示例；模型、服务地址和鉴权需要按自己的项目配置。

## 先画清楚数据边界

```text
Vue 页面 ── useStream ── Agent Server ── Python Deep Agent
   │                              │
   └── 文件 API（按需） ─────────── 沙箱
```

`useStream` 提供五类最值得展示的信息：

| 数据 | 界面用途 |
| --- | --- |
| `stream.messages` | 协调 Agent 的对话和最终总结 |
| `stream.subagents` | 子 Agent 的发现信息、状态与命名空间 |
| `stream.toolCalls` | 根流中已组装的工具调用和执行状态 |
| `stream.values` | 待办等自定义状态 |
| `stream.interrupt` | 等待用户处理的中断 |

根流的消息不等于所有子 Agent 的消息合集。子 Agent 的详细消息和工具调用，要用其发现快照单独订阅。这一点决定了页面适合做成“协调对话 + 计划 + 专家卡片”，而不是把所有输出塞进同一条聊天记录。[官方前端总览](https://docs.langchain.com/oss/python/deepagents/frontend/overview)

## 后端：让计划和委派有数据可读

下面注册一个名为 `agent` 的图。`TodoListMiddleware` 让 Agent 可以写入 `todos`；`researcher` 是可被委派的子 Agent。

```python
# agent.py
from deepagents import create_deep_agent
from langchain.agents.middleware import TodoListMiddleware

agent = create_deep_agent(
    model="openai:gpt-5.5",
    system_prompt="把复杂任务拆成步骤执行，并总结最终结果。",
    middleware=[TodoListMiddleware()],
    subagents=[
        {
            "name": "researcher",
            "description": "检索资料并核查事实",
            "system_prompt": "报告证据来源和不确定性。",
        }
    ],
)
```

```json
{
  "dependencies": ["."],
  "graphs": { "agent": "./agent.py:agent" }
}
```

上面第二段保存为 `langgraph.json`。项目还需安装 `deepagents`、`langchain-openai` 和本地开发用的 `langgraph-cli[inmem]`，并在**后端**配置模型密钥。运行 `langgraph dev --no-browser` 后，本地服务通常位于 `http://localhost:2024`。`agent` 是 Agent Server 的 assistant ID，浏览器没有直接导入 Python 对象。[Deep Agents 快速开始](https://docs.langchain.com/oss/python/deepagents/quickstart)

## Vue：先把根流显示出来

先安装 SDK、消息类型及 Markdown 渲染依赖，然后在页面里创建流：

```bash
npm install @langchain/vue @langchain/core langchain marked dompurify
```

```vue
<script setup lang="ts">
import { computed, ref } from "vue";
import { useStream } from "@langchain/vue";
import type { BaseMessage } from "@langchain/core/messages";
import { AIMessage, HumanMessage } from "langchain";
import MarkdownMessage from "./MarkdownMessage.vue";
import ToolCallCard from "./ToolCallCard.vue";

type Todo = {
  content: string;
  status: "pending" | "in_progress" | "completed";
};

interface AgentState {
  messages: BaseMessage[];
  todos?: Todo[];
}

const input = ref("");
const stream = useStream<AgentState>({
  apiUrl: import.meta.env.VITE_AGENT_URL ?? "http://localhost:2024",
  assistantId: "agent",
});

const todos = computed(() => stream.values.value?.todos ?? []);
const subagents = computed(() => [...stream.subagents.value.values()]);

function toolCallsFor(message: BaseMessage) {
  if (!AIMessage.isInstance(message)) return [];
  const ids = new Set((message.tool_calls ?? []).map((call) => call.id));
  return stream.toolCalls.value.filter((call) => ids.has(call.callId));
}

async function send() {
  const text = input.value.trim();
  if (!text || stream.isLoading.value) return;
  input.value = "";
  await stream.submit({ messages: [{ type: "human", content: text }] });
}
</script>

<template>
  <main>
    <aside v-if="todos.length" aria-label="任务计划">
      <p v-for="(todo, index) in todos" :key="index">
        {{ todo.status }} · {{ todo.content }}
      </p>
    </aside>

    <section aria-label="协调 Agent 消息">
      <template v-for="message in stream.messages.value" :key="message.id">
        <template v-if="AIMessage.isInstance(message)">
          <MarkdownMessage :content="message.text" />
          <ToolCallCard
            v-for="call in toolCallsFor(message)"
            :key="call.callId"
            :call="call"
          />
        </template>
        <p v-else-if="HumanMessage.isInstance(message)">{{ message.text }}</p>
      </template>
    </section>

    <aside aria-label="子 Agent">
      <p v-for="subagent in subagents" :key="subagent.id">
        {{ subagent.name }} · {{ subagent.status }}
      </p>
    </aside>

    <form @submit.prevent="send">
      <input v-model="input" aria-label="输入任务" />
      <button type="submit" :disabled="stream.isLoading.value">发送</button>
    </form>
  </main>
</template>
```

这里有两个容易踩的点。第一，`stream` 是对象，模板里访问它的内部 Ref 要写 `.value`；顶层的 `todos`、`subagents` 则由 Vue 模板自动解包。第二，`stream.values` 是状态快照，不能拿 `stream.values.messages` 替代流式消息。若后端没有写待办，`todos` 为空是正常情况。[待办模式](https://docs.langchain.com/oss/python/deepagents/frontend/todo-list)

## 让流式回复保留 Markdown 格式

`useStream` 会持续更新 AI 消息的 `text`。把它直接放进 `<p>`，标题、列表、代码块和表格都只会显示成原始字符。Vue 可以用 `marked` 解析，再用 `DOMPurify` 净化 HTML，最后交给 `v-html`。顺序必须是**解析 → 净化 → 渲染**。[Markdown 消息模式](https://docs.langchain.com/oss/python/langchain/frontend/markdown-messages)

```vue
<!-- MarkdownMessage.vue -->
<script setup lang="ts">
import { computed } from "vue";
import { marked } from "marked";
import DOMPurify from "dompurify";

const props = defineProps<{ content: string }>();
marked.setOptions({ gfm: true, breaks: true });

const html = computed(() =>
  props.content
    ? DOMPurify.sanitize(marked.parse(props.content) as string)
    : "",
);
</script>

<template>
  <div v-if="content" class="markdown-content" v-html="html" />
</template>
```

AI 消息和子 Agent 的 AI 消息都可以复用这个组件；用户消息继续用文本插值。普通聊天长度直接随 `content` 更新重新解析即可。若长回复出现滚动卡顿，再用 `requestAnimationFrame` 合并更新，并检查长代码行、宽表格是否溢出。

## 展开子 Agent 时再订阅详情

`stream.subagents` 里的记录适合做卡片标题和状态提示；它不包含完整消息。用户展开卡片时，再通过 `useMessages(stream, subagent)` 和 `useToolCalls(stream, subagent)` 获取该命名空间的内容。卡片可以挂在触发委派的协调 Agent 工具调用下方，也可以放进固定侧栏。子 Agent 完成状态是 `complete`，待办完成状态则是 `completed`。[子 Agent 流式展示](https://docs.langchain.com/oss/python/deepagents/frontend/subagent-streaming)

```ts
import { useMessages, useToolCalls } from "@langchain/vue";

const messages = useMessages(stream, subagent);
const toolCalls = useToolCalls(stream, subagent);
```

计划面板同样只读结构化状态。可以用已完成项数量计算一个步骤完成比例，但 Agent 随时可能新增或修改步骤，所以这个数字不等于剩余时间。

## 工具调用：把状态放回对应的消息

AI 消息中的 `tool_calls` 记录了工具调用 ID，`stream.toolCalls.value` 则把参数、结果和执行状态组装成卡片数据。上面 `toolCallsFor(message)` 用 `callId` 匹配 `message.tool_calls[].id`，让卡片出现在触发调用的消息下面。子 Agent 内的工具调用继续通过 `useToolCalls(stream, subagent)` 读取，不要把所有调用混进根流。[工具调用模式](https://docs.langchain.com/oss/python/langchain/frontend/tool-calling)

```vue
<!-- ToolCallCard.vue：通用兜底卡片 -->
<script setup lang="ts">
type Call = {
  callId: string;
  name: string;
  input: unknown;
  output: unknown | null;
  status: "running" | "finished" | "error";
  error?: string;
};
defineProps<{ call: Call }>();
</script>

<template>
  <article :aria-busy="call.status === 'running'">
    <strong>{{ call.name }}</strong> · {{ call.status }}
    <p v-if="call.status === 'error'" role="alert">
      {{ call.error ?? "工具执行失败" }}
    </p>
    <details>
      <summary>参数与结果</summary>
      <pre>{{ JSON.stringify(call.input, null, 2) }}</pre>
      <pre v-if="call.status === 'finished'">{{ JSON.stringify(call.output, null, 2) }}</pre>
    </details>
  </article>
</template>
```

同一个 `callId` 会从 `running` 更新到 `finished` 或 `error`；多个调用也可以同时运行。已知工具可按名称换成专用卡片，但读取 `output` 字段前应先校验其结构。工具返回的文本不要直接送进 `v-html`。

## 无头工具：在浏览器执行，再恢复 Agent

有些能力只能在用户浏览器中运行，例如定位、剪贴板、文件选择器或 IndexedDB。无头工具让后端保留正常的工具 schema，在调用时通过 `interrupt()` 暂停；前端提供同名工具的 `.implement(...)`，并在 `useStream({ tools: [...] })` 中注册。匹配后，SDK 执行浏览器代码并把结果送回 Agent，原来的工具卡片仍能显示执行状态。[无头工具模式](https://docs.langchain.com/oss/python/langchain/frontend/headless-tools)

下面以定位为例。在 Python 后端增加工具，再把 `tools=[geolocation_get]` 加进前文的 `create_deep_agent(...)`：

```python
from langchain.tools import ToolRuntime, tool
from langgraph.types import interrupt

@tool("geolocation_get", description="读取当前用户浏览器的位置")
def geolocation_get(runtime: ToolRuntime) -> dict:
    return interrupt({
        "type": "tool",
        "tool_call": {
            "id": runtime.tool_call_id,
            "name": "geolocation_get",
            "args": {},
        },
    })
```

前端工具的名称和参数必须与后端一致。安装 `zod` 后定义工具并附加浏览器实现：

```bash
npm install zod
```

```ts
// browser-tools.ts
import * as z from "zod";
import { tool } from "langchain";

const definition = tool({
  name: "geolocation_get",
  description: "读取当前用户浏览器的位置",
  schema: z.object({}),
});

export const geolocationGet = definition.implement(async () => {
  const position = await new Promise<GeolocationPosition>((resolve, reject) =>
    navigator.geolocation.getCurrentPosition(resolve, reject),
  );
  return {
    latitude: position.coords.latitude,
    longitude: position.coords.longitude,
    accuracy: position.coords.accuracy,
  };
});
```

最后在前文创建 `useStream<AgentState>` 的位置增加 `tools: [geolocationGet]`，并导入该实现：

```ts
import { geolocationGet } from "./browser-tools";

const stream = useStream<AgentState>({
  apiUrl: import.meta.env.VITE_AGENT_URL ?? "http://localhost:2024",
  assistantId: "agent",
  tools: [geolocationGet],
});
```

定位需要浏览器授权；拒绝授权时应显示工具错误。结果只能返回可序列化数据，不能返回 DOM 节点或文件句柄。浏览器执行也不意味着数据永远留在本机：**返回的位置会发送给后端，让 Agent 继续运行**。敏感能力应让用户明确知道会发送什么，并在需要时增加审批。后端运行还必须具备可恢复的 thread 与检查点。

## 审批和文件视图要各自接上后端

当后端为有副作用的工具设置人工审批，运行会进入中断。前端应展示待执行动作、参数和允许的决策，再把决定交回 Agent。一次中断可能含多个 `actionRequests`，提交的 `decisions` 必须逐项对应。`reject` 用来拒绝操作；`respond` 是替需要人工输入的工具提供结果，不能用作拒绝。[人工审批模式](https://docs.langchain.com/oss/python/langchain/frontend/human-in-the-loop)

```ts
// 仅示意：中断中只有一个动作，且允许 approve
await stream.respond({ decisions: [{ type: "approve" }] });
```

审批 API 在部分官方模式页仍有旧式 `stream.submit(null, { command: { resume } })` 示例。实际接入时，应与项目安装的 `@langchain/vue` 版本核对。

如果 Agent 会修改文件，文件树、文件内容和 Diff 需要后端另设受控 API。`useStream` 不会自动成为文件管理器。文件接口应按用户和 thread 校验访问权，把路径限制在对应沙箱内；前端可以在写文件工具完成后刷新，再用一次最终状态复核。这样用户才能看见“改了哪里”，而不只是听 Agent 说“改好了”。[沙箱前端模式](https://docs.langchain.com/oss/python/deepagents/frontend/sandbox)

## 按这个顺序实现

1. 接通 `useStream`，展示协调 Agent 的消息。
2. 启用待办，显示 `stream.values.todos`。
3. 展示子 Agent 快照，再按需订阅卡片详情。
4. 渲染流式 Markdown 与工具卡片，处理运行、完成和失败。
5. 按需加入无头工具，让浏览器执行并恢复 Agent。
6. 为有副作用的工具接入审批。
7. 最后增加沙箱文件树与 Diff。

这套顺序每一步都能独立验证。最终页面呈现的不只是 Agent 的回答，还包括它如何计划、委派、等待用户决策，以及交付了哪些具体结果。
