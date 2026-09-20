# LangChain 前端模式示例

这些项目是从 LangChain 文档演示区的 **Download project** 按钮下载的 Vue 前端 + Python Agent 示例，彼此独立，也不依赖博客项目的 `package.json`。

| 项目 | 官方文档 |
| --- | --- |
| [markdown-messages](./markdown-messages/) | [Markdown messages](https://docs.langchain.com/oss/python/langchain/frontend/markdown-messages) |
| [tool-calling](./tool-calling/) | [Tool calling](https://docs.langchain.com/oss/python/langchain/frontend/tool-calling) |
| [headless-tools](./headless-tools/) | [Headless tools](https://docs.langchain.com/oss/python/langchain/frontend/headless-tools) |
| [human-in-the-loop](./human-in-the-loop/) | [Human-in-the-Loop](https://docs.langchain.com/oss/python/langchain/frontend/human-in-the-loop) |
| [hitl-interrupt-forms](./hitl-interrupt-forms/) | [Human-in-the-Loop 的表单演示](https://docs.langchain.com/oss/python/langchain/frontend/human-in-the-loop) |

## 独立运行

任选一个项目，进入其目录后执行：

```bash
cp .env.example packages/agent/.env
# 在 packages/agent/.env 中填写真实的 ANTHROPIC_API_KEY
make install
make dev
```

需要 Node.js 22+、pnpm、Python 3.11+ 和 GNU Make。请在所选项目目录运行 `node --version`，确认当前 shell 实际使用的是 Node 22+。前端地址是 `http://localhost:4100`，Agent API 地址是 `http://localhost:2024`。五个项目使用相同端口，因此一次运行一个。各项目的 README 有详细说明。

## 已验证的范围

2026-09-18 在 macOS、Node.js 23.11、pnpm 9.15、Python 3.12 下：五个 ZIP 完整且无不安全解压路径；五个 Python 图均能加载，Agent 服务逐个启动且 `/ok` 返回 200；五个 Vue 前端均成功构建，开发服务器逐个返回页面。`tool-calling` 的 `make dev` 同时启动了两端，前端、Agent 和前端代理接口均返回 200。当前环境没有 Anthropic API Key，因此没有验证真实模型对话或完整工具调用流程。

这些官方下载包仍有一个类型检查缺口：除 `markdown-messages` 外，四个项目的 TypeScript 源码引用了 ZIP 中未附带的 `@langchain/playground-agents` 类型包。Vite 构建和页面启动成功，因为相关引用仅用于类型；`vue-tsc --noEmit` 会报找不到该模块。全部项目还会因 TypeScript 6 对 `baseUrl` 的弃用提示而在默认类型检查中报错。这里保留下载源码原样，并在此记录限制。
