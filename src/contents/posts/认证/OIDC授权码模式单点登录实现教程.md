---
title: OIDC 授权码模式单点登录实现教程
published: 2026-05-18
description: 基于 OIDC Authorization Code 的业务系统单点登录实现说明：术语、登录/退出主流程、关键参数、接口契约、异常路径、配置清单与验收要点
tags: [SSO, OIDC, 统一认证, IdP, 单点登录]
category: 认证
licenseName: "Unlicensed"
author: panxiao
sourceLink: "https://github.com/px6707/myblog"
draft: false
---

# OIDC 授权码模式单点登录实现教程

本文说明业务系统如何对接**统一认证平台（IdP）**，用 OIDC **授权码模式（Authorization Code）** 实现单点登录（SSO）。示例地址、路径与密钥均为占位符，可按项目替换。

文中约定：

- **IdP（Identity Provider）**：身份提供方，负责认证用户  
- **本系统 / Client（RP）**：依赖方，即接入 SSO 的业务应用  
- 协议参数沿用 OIDC 标准名：`client_id`、`redirect_uri`、`code`、`state` 等  

---

## 一、参与方与配置项

| 参与方或参数 | 示例 | 作用 |
| --- | --- | --- |
| 本系统 | `https://app.example.com` | 浏览器访问的业务站点；前端与 `/api/` 可同源 |
| 统一认证（IdP） | `https://idp.example.com` | 企业统一登录服务 |
| Discovery | `https://idp.example.com/.../.well-known/openid-configuration` | 返回授权、换令牌、用户信息等真实端点 |
| `SSO_CLIENT_ID` | `<client_id>` | 本系统在 IdP 登记的 OIDC 客户端标识，**不是**用户账号 |
| `SSO_REDIRECT_URI` | `https://app.example.com/api/auth/sso/callback` | 认证完成后浏览器必须回到的**固定回调** |
| `SSO_CLIENT_SECRET` | `<服务端密钥>` | 仅后端在授权码换令牌时使用，禁止进前端或 URL |

两个容易混淆的地址：

- **`redirect_uri`**：事先在 IdP 登记的固定回调接口  
- **`redirect`**：登录成功后要回到的**站内业务路径**（如 `/settings/users`），由本系统校验，只允许站内相对路径  

实际授权、换票地址以 Discovery 响应为准，不要在代码里写死某厂商路径。

---

## 二、登录主流程（授权码模式）

```text
浏览器 → 本系统 /api/auth/sso/login → IdP 授权页
IdP → 浏览器（302 到本系统回调）
浏览器 → 本系统 /api/auth/sso/callback
本系统后端 → IdP 换令牌、取用户信息
浏览器 → 本系统 /login?ticket=... → /api/auth/sso/complete → 原业务页
```

### 2.1 如何进入授权：两种常见入口

未登录访问**系统内部页面**时（下文以 `/settings/users` 为例，任意需登录的站内路径均可），并不只有「先到本系统登录页再点按钮」这一种做法。常见至少两种：

| 入口方式 | 未登录访问内部页时发生什么 | 适用 |
| --- | --- | --- |
| **A. 经本系统登录页** | 守卫先到 `/login?redirect=...`，用户再点「统一登录」 | 同时保留本地账号密码、多种登录方式时 |
| **B. 守卫直接发起 SSO** | 守卫不经过登录页 UI，直接让浏览器访问本系统 SSO 入口（或前端按 PKCE 自行跳 IdP） | 仅统一认证、希望少一步点击时 |

两种方式在协议上同属授权码模式；差别主要在**谁触发授权、是否展示本系统登录页**。下文 2.2 按 **A** 逐步写；采用 **B** 时，把「点统一登录」换成「路由守卫自动跳转」即可，后续换票、建会话步骤相同。

采用 **B** 时也有两种落地：

1. **推荐（与本文其余章节一致）**：守卫跳到本系统后端入口  
   `window.location = /api/auth/sso/login?redirect=<原内部页>`  
   仍由后端生成 `state`、写 Cookie、再 302 到 IdP。用户体感是「打开内部页 → 立刻出现 IdP」，中间没有本系统登录页。
2. **前端直拼 IdP 授权 URL**：守卫直接 `location` 到 IdP 的 `authorization_endpoint`。此时必须由前端自己生成并保存 `state`（以及无法安全持有 `client_secret` 时的 **PKCE**），回调页也要能完成校验；**不能**在仍使用「仅后端持有 `client_secret`、仅后端签发 `state`」的实现里，简单地删掉后端登录入口改由前端硬编码 IdP 地址。

**PKCE（Proof Key for Code Exchange）**是什么：授权码模式的扩展，用来防止授权码被拦截后被第三方换走令牌。流程要点是：客户端先生成随机 `code_verifier`，再把它的摘要 `code_challenge` 放进授权请求；换令牌时必须出示原始 `code_verifier`，IdP 校验通过才发 token。浏览器 / 移动端等**公开客户端**放不住 `client_secret`，因此用 PKCE 代替「靠密钥证明换票方就是当初发起授权的那一方」。本文主路径是后端机密客户端 + `client_secret`，一般不强制 PKCE；只有「前端自己直跳 IdP 换票」时才需要它。

### 2.2 逐步说明（以入口 A 为例）

1. **保存原内部页**  
   未登录访问内部页（例：`/settings/users`，仅为示例路径）时，前端守卫先到 `/login?redirect=%2Fsettings%2Fusers`。登录页可用 `GET /api/auth/sso/status` 判断统一登录是否启用。用户直接打开 `/login` 并点「统一登录」、没有原内部页时，可用 `/` 作为 `redirect`。  
   若用入口 **B**，守卫把当前内部路径编码进 `redirect` 后，直接进入下一步，不必先渲染登录页。

2. **进入本系统登录入口（后端 302，不是页面）**  
   浏览器访问：  
   `GET /api/auth/sso/login?redirect=%2Fsettings%2Fusers`  
   用户通常只看到浏览器被带到 IdP。

3. **绑定本次授权请求**  
   后端依次做：拉取 Discovery → 取出授权端点 → 生成 `state` → 记入短期待完成状态 → 设置短期 `HttpOnly` 的 state Cookie → `302` 到 IdP。

   **Discovery 是什么**  
   OIDC 约定 IdP 提供一份公开的元数据文档（OpenID Provider Configuration），通常地址形如：

   ```text
   https://idp.example.com/.../.well-known/openid-configuration
   ```

   本系统后端用 HTTP GET 拉取该 JSON。文档里会给出本 IdP 实际使用的各接口地址，至少包括：

   | 字段 | 含义 |
   | --- | --- |
   | `authorization_endpoint` | 浏览器应前往的**授权页/授权接口** |
   | `token_endpoint` | 后端用授权码换令牌的地址 |
   | `userinfo_endpoint` | 后端用令牌取用户信息的地址 |

   因此下面示例里的 `https://idp.example.com/oidc/authorize` **不是业务代码写死的路径**，而是 Discovery 响应里 `authorization_endpoint` 的值（各厂商路径不同，以实际返回为准）。后端把查询参数拼到该地址上，再对浏览器返回 `302 Location`。

   **授权跳转形态示例**（主机与路径来自 Discovery）：

   ```text
   https://idp.example.com/oidc/authorize
     ?response_type=code
     &client_id=<client_id>
     &redirect_uri=https%3A%2F%2Fapp.example.com%2Fapi%2Fauth%2Fsso%2Fcallback
     &scope=openid%20profile
     &state=<签名状态>
   ```

   建议显式带上 `scope`（至少 `openid`；按 IdP 要求追加 `profile` 等）。

   **`state` 是什么、做什么用**  
   `state` 由**本系统后端**在发起授权前生成，经授权 URL 交给 IdP，IdP 在回调时**原样带回**（与 `code` 一起出现在回调查询参数里）。它不是用户身份，也不是访问令牌。

   作用主要有两点：

   1. **把「这次浏览器发起的授权」和「稍后的回调」绑在一起**：回调里带回来的 `state` 必须能对上本次请求（常见再配合 HttpOnly Cookie、服务端一次性待完成记录），防止别人拿截获/伪造的回调串到你的会话上（CSRF / 登录串用）。  
   2. **在授权往返中安全携带本系统自己的上下文**：例如用户原先要去的站内 `redirect` 路径、发起时间等，可编进签名后的 `state`（或只放服务端、用 `state` 当键），回调校验通过后再取出，避免把未校验的返回地址直接信回调参数。

   示例实现里，`state` 可以是「站内返回路径 + 时间戳」的 HMAC 签名值；Cookie 存其摘要；服务端再存一份短期待完成记录。回调时三者一起校验，并设 TTL（如 10 分钟）。**不能**简化成「只比对 URL 上出现过一个同名字符串」。

4. **IdP 认证用户**  
   无有效 IdP 会话则展示登录页；同浏览器已有会话则可免密发码。登录页与会话 Cookie 属于 IdP，不属于本系统。

5. **浏览器带授权码回来**  
   典型：`GET /api/auth/sso/callback?code=<一次性授权码>&state=<原state>`  
   这是**浏览器跳转**，不是 IdP 服务器主动调你的后端。正常回调只带 `code`、`state`（或失败时的 `error`），不带用户密码。

6. **校验 state → 换令牌 → 取用户信息**  
   核对 URL `state`、Cookie、服务端待完成记录与时效；通过后用 `code` 调 `token_endpoint`（`client_id` + `client_secret` 做客户端认证），拿到 IdP `access_token`，再调 `userinfo_endpoint`。IdP 令牌可放进 `HttpOnly` Cookie，供退出时识别「本次是否统一登录会话」；**它不是本系统 API 的登录令牌**。

7. **建立本系统用户与会话**  
   按 IdP 用户唯一标识查找或创建本地账号，更新资料；首登可给「未分配角色」等受限态，由本系统再赋权。签发**本系统自己的**登录令牌，但不要直接塞进跳转 URL：改发短期、一次性 `ticket`，再跳到  
   `/login?ticket=<票据>&redirect=%2Fsettings%2Fusers`。

8. **前端兑换票据并回到业务页**  
   `POST /api/auth/sso/complete`，body：`{"ticket":"..."}` → 得到本系统 `access_token` → 本地存储 → `/api/auth/me` → 进入原页面。未赋权则进等待页，**IdP 登录成功 ≠ 本系统业务权限全部放开**。

### 2.3 跨系统「免输密码」在发生什么

系统 A、B 各自对接**同一 IdP**。用户在 A 登录后，浏览器里已有 **IdP 会话 Cookie**。再打开 B 时，B 仍走自己的授权请求；浏览器访问 IdP 时带上该会话，IdP 常直接给 B 发**专属**授权码。A、B **不共享**本地令牌、角色或各自的 `state` Cookie。

---

## 三、关键参数各管什么

| 名称 | 谁生成、谁用 | 目的 |
| --- | --- | --- |
| `client_id` | IdP 分配；授权请求携带 | 标明「哪个业务系统在要认证」 |
| `redirect_uri` | 本系统配置且在 IdP 登记 | 限定授权码只能回到登记回调；协议/主机/端口/路径必须一致 |
| `redirect` | 本系统前端记录，后端校验 | 登录后回站内业务页；**不是** IdP 回调 |
| `state` | 本系统生成；IdP 原样带回 | 绑定「这次浏览器发起的授权」与回调，防串用/篡改返回页 |
| `code` | IdP → 回调 | 短时一次性授权码，后端拿去换 IdP 令牌 |
| IdP `access_token` | IdP → 本系统后端 | 拉用户信息；可进 HttpOnly Cookie；≠ 本系统 API 令牌 |
| `ticket` | 本系统后端 → 登录页兑换 | 把后端已完成的登录安全交给前端，避免长期令牌进 URL |
| 本系统 `access_token` | 本系统签发；前端保存 | 调本系统 API、识别本地用户与角色 |

推荐校验方式（示例实现）：`state` =「站内路径 + 时间戳」的 HMAC；Cookie 存 `state` 的摘要；服务端再存一份短期待完成记录。回调时三者一起验，TTL 如 10 分钟。**不能**简化成「只比对 URL 上的 `state` 字符串」。

---

## 四、为什么这样设计

- **为什么常见实现先走本系统 `/api/auth/sso/login`，而不是前端自己拼 IdP 地址？**  
  在「后端持有 `client_secret`、后端签发 `state`」的方案里，需要 Discovery、签 `state`、写待完成状态并设 HttpOnly Cookie，再 302 到 IdP。未登录访问内部页时可以**跳过本系统登录页**（入口 B），但仍建议经过该后端入口，这样用户体感已是「直达 IdP」。若要让前端完全自行跳 IdP，需另做 PKCE 与前端状态保存，不能只删掉后端入口。

- **为什么 IdP 回调后还要进本系统逻辑？**  
  IdP 只证明「此人在统一认证侧已认证」。本系统还要映射本地账号、停用检查、角色与自己的会话令牌。

- **为什么用授权码而不是把令牌直接扔给浏览器？**  
  `client_secret` 与换票留在后端，密钥不出浏览器。

- **为什么还要 `ticket`？**  
  避免把可长期使用的本系统令牌写进地址栏、历史记录和 Referer。

---

## 五、退出流程

```text
前端清除本地 access_token
  → POST /api/auth/logout
  → 若本次为统一登录会话：返回 IdP 退出 URL
  → 浏览器访问 IdP 退出（清 IdP 会话）
  → 回到本系统登录页或约定回落点
```

要点：

1. 先清本系统令牌，再调后端退出接口（退出宜幂等：本系统 JWT 已过期也应能清 Cookie）。  
2. 「是否跳转 IdP 退出」应看**本次会话如何登录**（如 `login_method=sso`），不要只看「账号是否来自统一认证同步」。  
3. **只清本系统令牌、保留 IdP 会话** → 下次点统一登录仍可能免密。要真正「全站登出」，必须清 IdP 会话。  
4. 密码本地登录与统一登录并存时，本地登录成功后应清掉旧的 IdP 令牌 Cookie，避免误判退出路径。

---

## 六、接口契约（示例）

下列路径名为示例，可按项目调整。

| 方法 | 路径 | 作用 | 成功时要点 |
| --- | --- | --- | --- |
| `GET` | `/api/auth/sso/status` | 是否启用统一登录 | `{ "enabled": true }` |
| `GET` | `/api/auth/sso/login?redirect=` | 准备授权并 302 到 IdP | 设置 state Cookie；`Location` = 授权 URL |
| `GET` | `/api/auth/sso/callback` | 收 `code`/`state` 或 `error` | 校验 → 换票 → 建用户 → 302 到 `/login?ticket=...` |
| `POST` | `/api/auth/sso/complete` | 兑换 `ticket` | `{ "access_token", "user" }`；ticket 一次性 |
| `POST` | `/api/auth/logout` | 退出 | `{ "redirect_url", "sso_logout_redirected" }`；清相关 Cookie |
| `GET` | `/api/auth/me` | 当前用户 | 需本系统 `Authorization` |

换令牌（后端 → IdP）常见形态：

- `POST token_endpoint`  
- `grant_type=authorization_code`  
- `code`、`redirect_uri`（须与授权时一致）  
- 客户端认证：HTTP Basic（`client_id:client_secret`）或 IdP 要求的其它方式  

---

## 七、异常与分支（实现时必须处理）

| 场景 | 建议行为 |
| --- | --- |
| 回调带 `error`（用户取消等） | 302 回登录页并提示「已取消或失败」 |
| 缺 `code` / `state` | 400 或回登录页提示从统一入口重新登录 |
| state / Cookie / 服务端记录不一致或过期 | 「状态已失效，请重新登录」 |
| Discovery / 换票 / userinfo 网络或 4xx/5xx | 503/401 + 明确文案；勿把原始令牌打进对用户可见错误 |
| `ticket` 过期、重放、乱填 | 401「凭证已失效」 |
| 本地用户停用或不存在 | 兑换阶段拒绝，不发可用会话 |
| 首登未分配业务角色 | 进等待/无权限页，不开放业务菜单 |
| 授权途中后端重启或多实例未共享 state/ticket | 表现为状态/凭证失效 → 需共享存储（见第九节） |

失败时优先 **302 回登录页带错误参数**，避免在回调 URL 上长时间停留暴露 `code`。

---

## 八、统一认证侧与本系统配置清单

### 8.1 在 IdP 控制台登记

1. 创建 OIDC 客户端，拿到 `client_id` / `client_secret`  
2. 登记精确的 `redirect_uri`（协议、域名、端口、路径一字不差）  
3. 确认授权类型含 Authorization Code  
4. 确认可用 `scope`、userinfo 字段（至少要有稳定用户唯一标识，如 `sub`）  
5. 确认登出 URL 与回落参数约定（各厂商差异大）  
6. 确认 Discovery 地址可被**本系统后端**访问  

### 8.2 本系统环境变量（示例名）

| 变量 | 说明 |
| --- | --- |
| `SSO_ENABLED` | 开关；仅当其它项齐全时真正启用 |
| `SSO_DOMAIN` / `SSO_DISCOVERY_URL` | IdP 根或完整 Discovery |
| `SSO_CLIENT_ID` / `SSO_CLIENT_SECRET` | 客户端凭证 |
| `SSO_REDIRECT_URI` | 与 IdP 登记完全一致的回调 |
| 本系统令牌密钥（如 `APP_AUTH_SECRET`） | 签 `state`、签发本系统 JWT 等 |

### 8.3 网络与部署边界

1. `redirect_uri` 必须是浏览器可达地址，不要填容器内网名、不要填前端 `/login`。  
2. **两条链路都要通**：浏览器 ↔ 本系统、浏览器 ↔ IdP；以及 **后端 ↔ IdP**（Discovery / token / userinfo）。  
3. 生产建议全站 HTTPS；Cookie 设 `Secure`；改 HTTPS 时同步改 IdP 登记与本系统配置。  
4. Cookie 建议：`HttpOnly`、`SameSite=Lax`（或按跨站场景评估）、路径尽量收窄。  
5. 日志不要原样长期保留含令牌的 IdP 响应；工单勿贴 `client_secret`、`code`、`state`、`ticket`、真实令牌。

---

## 九、会话与多实例

| 项 | 建议 |
| --- | --- |
| 本系统 `access_token` 过期 | 走本系统刷新或重新统一登录；与 IdP token 生命周期解耦 |
| `state` / `ticket` 存哪 | 单机可用进程内存；**多实例必须** Redis 等共享存储，否则负载均衡会随机「状态已失效」 |
| 多标签页 | `ticket` 一次性；重复打开带同一 ticket 的 URL 应失败 |
| 时钟偏差 | `state` 校验可允许很小负时差（如 60s），仍要硬上限 TTL |

---

## 十、安全检查清单（上线前）

- [ ] `redirect` 只允许站内相对路径，拒绝 `//evil`、反斜杠、外域  
- [ ] `redirect_uri` 与 IdP 登记完全一致，不可被请求参数改写为任意 URL  
- [ ] `state` + Cookie + 服务端一次性消费，防 CSRF / 回调串用  
- [ ] `client_secret` 仅服务端；前端无密钥  
- [ ] `ticket` 短 TTL、单次兑换  
- [ ] 生产 HTTPS + `Secure` Cookie  
- [ ] 日志脱敏；密钥可轮换  
- [ ] 统一登录成功后仍做本系统授权（角色 / 停用）  
- [ ] 退出能清 IdP 会话（若产品要求真正全局登出）

---

## 十一、验收用例（最小集）

1. 未登录访问深链 → 统一登录 → 回到原深链  
2. 已有 IdP 会话时再登录本系统 → 免密或少一步密码  
3. 另一业务系统（同 IdP）打开 → 通常免密拿到**该系统自己的**会话  
4. 人为改坏 `state` / 去掉 Cookie → 失败且需重新登录  
5. 重复提交同一 `ticket` → 第二次失败  
6. 首登无角色 → 进等待页，业务接口仍拒绝  
7. 统一登录后退出 → 再进统一登录应重新认证（若已调 IdP 退出）  
8. 仅清本系统 token、不调 IdP 退出 → 可复现「仍免密」（用于理解会话边界）  
9. Discovery 或 IdP 不可达 → 有明确错误，不白屏  
10. 多实例（若已上）→ 登录过程中切换实例仍成功（共享 state/ticket）

---

## 十二、实现模块怎么拆（对照用）

| 模块 | 职责 |
| --- | --- |
| 前端登录页 | 统一登录按钮、`ticket` 兑换、错误提示、回业务页 |
| 前端路由守卫 | 未登录跳转、本地角色限制 |
| 后端认证路由 | login / callback / complete / logout / status |
| OIDC 客户端封装 | Discovery、`state`、换令牌、userinfo、ticket |
| 本地用户服务 | IdP 用户标识 ↔ 本地账号、角色、停用 |

---

## 十三、本教程覆盖范围与未覆盖项

**已覆盖：** OIDC 授权码 + 后端持有 `client_secret`、登录/退出主路径、参数辨析、异常、配置、安全与验收。

**未展开（需要时另开专题）：** 前端 PKCE 纯 SPA 方案、SAML、静默授权 `prompt=none`、IdP refresh token 续期策略、账号合并冲突策略、各厂商专有登出参数细节。
