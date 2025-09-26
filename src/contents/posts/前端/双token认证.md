---
title: 双token认证
published: 2025-09-26
description: 双token认证
tags: [js, token]
category: 前端
licenseName: "Unlicensed"
author: panxiao
sourceLink: "https://github.com/px6707/myblog"
draft: false
---

### 什么是Token
Token是服务器颁发给客户端的凭证，客户端在每次请求时都需要携带这个凭证，服务器通过这个凭证来验证客户端的身份。

### Token认证流程

1. 用户登陆：用户提供用户名和密码
2. 服务器验证：服务器验证用户名和密码是否正确
3. 生成Token：如果验证成功，服务器会生成Token
4. 返回Token：服务器将Token返回给客户端
5. 客户端存储Token：客户端将Token存储在本地，通常是localStorage或sessionStorage、cookie等
6. 携带Token：客户端在每次请求时都需要携带Token
7. 服务器验证：服务器通过Token来验证客户端的身份
8. 返回数据：服务器返回数据给客户端

如此一来服务端就能知道客户端的身份，就能返回对应的数据，并保证数据安全。
> 问题：token为了防止被盗用，需要设置过期时间，过期后需要重新登陆。且一般有效期较短。这就导致了用户需要频繁登陆

### 双Token认证
为了解决这个问题，可以使用双Token认证，即使用两个Token，一个用于短期认证，一个用于长期认证。即Access Token有效期短，Refresh Token有效期长。

- 登陆阶段
1. 用户登陆：用户提供用户名和密码
2. 服务器验证：服务器验证用户名和密码是否正确
3. 生成Token：如果验证成功，服务器会生成Access Token和Refresh Token
4. 返回Token：服务器将Access Token和Refresh Token返回给客户端
5. 客户端存储Token：客户端将Access Token和Refresh Token存储在本地，通常是localStorage或sessionStorage、cookie等
- 正常请求阶段
6. 携带Token：客户端在每次请求时都需要携带Access Token
7. 服务器验证：服务器通过Access Token来验证客户端的身份
8. 返回数据：服务器返回数据给客户端
- 刷新Token阶段
9. Access Token 过期，服务器返回401,也可以通过过期时间由前端检测是否过期。
10. 客户端检测到401或者Access Token过期，使用Refresh Token请求新的Access Token
11. 服务器验证Refresh Token，如果验证成功，返回新的Access Token
12. 客户端使用新Token重新发起原请求


### 两种Token
1. Access Token 有效期短，用于短期认证
2. Refresh Token 有效期长（可以设置为几天），用于长期认证,用来刷新AccessToken。最好存储在httpOnly Cookie，防止js访问，抵御XSS攻击

### 实现方式

```javascript
  import axios from 'axios'

  // 登录
  async login(username, password) {
    const response = await fetch('/api/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password })
    });
    
    const data = await response.json();
    setTokens(data.access_token, data.refresh_token, data.access_token_expires_at, data.refresh_token_expires_at);
  }

  // 设置Token
  setTokens(accessToken, refreshToken, accessTokenExpiresAt, refreshTokenExpiresAt) {
    localStorage.setItem('access_token', accessToken);
    localStorage.setItem('refresh_token', refreshToken);
    localStorage.setItem('access_token_expires_at', accessTokenExpiresAt);
    localStorage.setItem('refresh_token_expires_at', refreshTokenExpiresAt);
  }

  // 刷新Token
  async refreshAccessToken(refreshToken) {
    const response = await fetch('/api/refresh', {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${refreshToken}` }
    });
    
    const data = await response.json();
    setTokens(data.access_token, data.refresh_token, data.access_token_expires_at, data.refresh_token_expires_at);
    return data.access_token;
  }


// 请求队列，用于存储待处理的请求
let queue = []

// 标识是否正在刷新 token
let isRefreshing = false

// 请求拦截器
axios.interceptors.request.use(
  (req) => {
    const accessToken = localStorage.getItem('access_token');
    const refreshToken = localStorage.getItem('refresh_token');
    const accessTokenExpiresAt = localStorage.getItem('access_token_expires_at');
    const refreshTokenExpiresAt = localStorage.getItem('refresh_token_expires_at');
    // 验证 token
    if (accessToken) {
      // 设置请求头中的 Authorization
      if (req.headers['Authorization'] !== null) {
        req.headers['Authorization'] = accessToken
      }

      // 判断 token 是否过期
      if (Date.now() > accessTokenExpiresAt - 2000) {
        // 判断 refreshToken 是否过期
        if (Date.now() > refreshTokenExpiresAt - 2000) {
          alert('登录状态已失效，请重新登录')
          logout()
        } else {
          // 如果不在刷新中，则刷新 token
          if (!isRefreshing) {
            isRefreshing = true

            refreshAccessToken(refreshToken)
              .then((token) => {
                queue.forEach((cb) => cb(token)) // 处理队列中的请求
                queue = []
                isRefreshing = false
              })
              .catch(() => {
                logout()
              })
          }

          // 返回一个新的 Promise，等待 token 刷新完成
          return new Promise((resolve) => {
            queue.push((token) => {
              if (req.headers) {
                req.headers['Authorization'] = token // 重新设置 token
              }
              resolve(req)
            })
          })
        }
      }
    }

    return req
  },
  (error) => {
    return Promise.reject(error) // 请求错误处理
  },
)

```

