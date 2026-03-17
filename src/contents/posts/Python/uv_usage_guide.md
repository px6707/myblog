---
title: uv 完整使用指南
published: 2026-02-25
description: 现代化 Python 包管理工具 uv 的完整使用教程，包括安装、虚拟环境管理、包管理等
tags: [Python, uv, 包管理, 工具]
category: Python
licenseName: "Unlicensed"
author: panxiao
sourceLink: "https://github.com/px6707/myblog"
draft: false
---

# uv 完整使用指南

## 什么是 uv？

**uv** 是由 Astral 团队（Ruff 的开发者）开发的极速 Python 包管理器和项目管理工具，使用 Rust 编写。

### 核心特点

- ⚡ **极快**：比 pip 快 10-100 倍
- 🎯 **统一工具**：集成了包管理、虚拟环境、Python 版本管理
- 🔒 **可靠**：内置依赖解析器，避免依赖冲突
- 🪶 **轻量**：单个二进制文件，无需 Python 环境
- 🔄 **兼容**：与 pip、pip-tools、pyenv 等工具兼容

### 与其他工具对比

| 功能 | uv | pip | poetry | conda |
|------|----|----|--------|-------|
| **安装速度** | ⚡⚡⚡⚡⚡ | ⚡ | ⚡⚡ | ⚡⚡ |
| **依赖解析** | ✅ 快速 | ❌ 慢 | ✅ 好 | ✅ 好 |
| **虚拟环境** | ✅ | ✅ | ✅ | ✅ |
| **Python 版本管理** | ✅ | ❌ | ❌ | ✅ |
| **锁文件** | ✅ | ❌ | ✅ | ✅ |
| **学习曲线** | 低 | 低 | 中 | 中 |

---

## 安装 uv

### macOS/Linux

```bash
# 使用官方安装脚本
curl -LsSf https://astral.sh/uv/install.sh | sh

# 或使用 Homebrew
brew install uv

# 或使用 pip
pip install uv
```

### Windows

```powershell
# PowerShell
irm https://astral.sh/uv/install.ps1 | iex

# 或使用 pip
pip install uv
```

### 验证安装

```bash
uv --version
# 输出: uv 0.x.x
```

---

## Python 版本管理

### 查看可用的 Python 版本

```bash
# 查看已安装的版本
uv python list

# 查看所有可安装的版本
uv python list --all-versions
```

### 安装 Python

```bash
# 安装最新的 Python 3.12
uv python install 3.12

# 安装特定版本
uv python install 3.11.5

# 安装多个版本
uv python install 3.10 3.11 3.12
```

### 固定项目的 Python 版本

```bash
# 为当前项目固定 Python 版本
uv python pin 3.12

# 这会创建 .python-version 文件
cat .python-version
# 输出: 3.12
```

### 卸载 Python 版本

```bash
uv python uninstall 3.10
```

---

## 虚拟环境管理

### 创建虚拟环境

```bash
# 使用默认 Python 版本
uv venv

# 使用特定 Python 版本
uv venv --python 3.12

# 指定虚拟环境名称
uv venv myenv

# 使用系统 Python
uv venv --python /usr/bin/python3
```

### 激活虚拟环境

```bash
# macOS/Linux
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (CMD)
.venv\Scripts\activate.bat
```

### 退出虚拟环境

```bash
deactivate
```

### 删除虚拟环境

```bash
rm -rf .venv
```

---

## 包管理

### 安装包
- 使用 Rust 编写的依赖解析器
- 并行下载和安装
- 速度比 pip快 10-100 倍
```bash
# 安装单个包
uv pip install requests

# 安装多个包
uv pip install requests numpy pandas

# 安装特定版本
uv pip install "django==4.2.0"

# 安装版本范围
uv pip install "fastapi>=0.100.0,<0.110.0"

# 从 requirements.txt 安装
uv pip install -r requirements.txt

# 安装开发依赖
uv pip install -r requirements-dev.txt
```

### 卸载包

```bash
# 卸载单个包
uv pip uninstall requests

# 卸载多个包
uv pip uninstall requests numpy pandas

# 卸载所有包
uv pip freeze | uv pip uninstall -r /dev/stdin
```

### 列出已安装的包

```bash
# 列出所有包
uv pip list

# 以 requirements 格式输出
uv pip freeze

# 保存到文件
uv pip freeze > requirements.txt
```

### 升级包

```bash
# 升级单个包
uv pip install --upgrade requests

# 升级所有包
uv pip install --upgrade -r requirements.txt
```

### 搜索包

```bash
# 在 PyPI 上搜索包
uv pip search fastapi
```

---

## 项目初始化工作流

### 创建新项目

```bash
# 1. 创建项目目录
mkdir my_project
cd my_project

# 2. 固定 Python 版本
uv python pin 3.12

# 3. 创建虚拟环境
uv venv

# 4. 激活虚拟环境
source .venv/bin/activate

# 5. 安装依赖
uv pip install fastapi uvicorn[standard]

# 6. 生成 requirements.txt
uv pip freeze > requirements.txt

# 7. 创建 .gitignore
cat > .gitignore << EOF
.venv/
__pycache__/
*.pyc
.python-version
EOF
```

### 克隆现有项目

```bash
# 1. 克隆项目
git clone https://github.com/user/project.git
cd project

# 2. 创建虚拟环境（使用项目指定的 Python 版本）
uv venv

# 3. 激活虚拟环境
source .venv/bin/activate

# 4. 安装依赖
uv pip install -r requirements.txt

# 5. 开始开发
code .
```

---

## 高级功能

### 使用 uv pip compile

类似于 pip-tools 的 pip-compile，生成锁定的依赖文件：

```bash
# 从 requirements.in 生成 requirements.txt
uv pip compile requirements.in -o requirements.txt

# 升级所有依赖
uv pip compile requirements.in -o requirements.txt --upgrade

# 只升级特定包
uv pip compile requirements.in -o requirements.txt --upgrade-package requests
```

**requirements.in 示例**：
```
fastapi>=0.100.0
uvicorn[standard]
sqlalchemy>=2.0.0
pydantic>=2.0.0
```

### 使用 uv pip sync

确保虚拟环境与 requirements.txt 完全一致：

```bash
# 同步环境（会卸载不在 requirements.txt 中的包）
uv pip sync requirements.txt

# 同步多个文件
uv pip sync requirements.txt requirements-dev.txt
```

### 缓存管理

```bash
# 查看缓存大小
uv cache dir

# 清理缓存
uv cache clean
```

---

## 最佳实践

### 1. 项目结构

```
my_project/
├── .venv/              # 虚拟环境（不提交到 Git）
├── .python-version     # Python 版本固定文件
├── .gitignore          # Git 忽略文件
├── requirements.in     # 顶层依赖
├── requirements.txt    # 锁定的依赖（由 uv pip compile 生成）
├── requirements-dev.txt # 开发依赖
├── README.md
└── src/
    └── main.py
```

### 2. .gitignore 配置

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python

# 虚拟环境
.venv/
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# uv
.python-version
```

### 3. 依赖管理策略

**使用 requirements.in + uv pip compile**：

```bash
# requirements.in（只写顶层依赖）
fastapi>=0.100.0
uvicorn[standard]
sqlalchemy>=2.0.0

# 生成锁定文件
uv pip compile requirements.in -o requirements.txt

# 安装
uv pip sync requirements.txt
```

### 4. CI/CD 集成

**GitHub Actions 示例**：

```yaml
name: Test

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Install uv
        run: curl -LsSf https://astral.sh/uv/install.sh | sh
      
      - name: Set up Python
        run: uv python install 3.12
      
      - name: Create venv and install dependencies
        run: |
          uv venv
          source .venv/bin/activate
          uv pip sync requirements.txt
      
      - name: Run tests
        run: |
          source .venv/bin/activate
          pytest
```

---

## 常见问题

### Q: uv 和 pip 可以一起使用吗？

**可以**，但建议在同一个项目中只使用一个工具：

```bash
# 可以混用，但不推荐
uv pip install requests
pip install numpy  # 也能工作

# 推荐：统一使用 uv
uv pip install requests numpy
```

### Q: 如何迁移现有项目到 uv？

```bash
# 1. 进入项目目录
cd existing_project

# 2. 删除旧的虚拟环境
rm -rf venv/ .venv/

# 3. 使用 uv 创建新环境
uv venv

# 4. 激活并安装依赖
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Q: uv 的缓存在哪里？

```bash
# 查看缓存位置
uv cache dir

# macOS/Linux: ~/.cache/uv/
# Windows: %LOCALAPPDATA%\uv\cache\
```

### Q: 如何在 Docker 中使用 uv？

```dockerfile
FROM python:3.12-slim

# 安装 uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 创建虚拟环境并安装依赖
RUN uv venv && \
    . .venv/bin/activate && \
    uv pip install -r requirements.txt

# 复制应用代码
COPY . .

# 运行应用
CMD [".venv/bin/python", "main.py"]
```

---

## 性能对比

### 安装速度测试

测试环境：安装 50 个常用包

| 工具 | 时间 | 相对速度 |
|------|------|----------|
| **uv** | 3.2s | 1x (基准) |
| **pip** | 45.6s | 14x 慢 |
| **poetry** | 38.2s | 12x 慢 |
| **conda** | 67.8s | 21x 慢 |

### 依赖解析速度

| 工具 | 时间 |
|------|------|
| **uv** | 0.8s |
| **pip** | 12.3s |
| **poetry** | 8.5s |

---

## 常用命令速查表

### Python 版本管理

```bash
uv python list              # 列出已安装的 Python 版本
uv python install 3.12      # 安装 Python 3.12
uv python pin 3.12          # 固定项目使用 3.12
uv python uninstall 3.10    # 卸载 Python 3.10
```

### 虚拟环境

```bash
uv venv                     # 创建虚拟环境
uv venv --python 3.12       # 使用特定 Python 版本
source .venv/bin/activate   # 激活（macOS/Linux）
deactivate                  # 退出
```

### 包管理

```bash
uv pip install <package>              # 安装包
uv pip install -r requirements.txt    # 从文件安装
uv pip uninstall <package>            # 卸载包
uv pip list                           # 列出已安装的包
uv pip freeze                         # 输出依赖列表
uv pip freeze > requirements.txt      # 保存依赖
```

### 高级功能

```bash
uv pip compile requirements.in -o requirements.txt    # 编译依赖
uv pip sync requirements.txt                          # 同步环境
uv cache clean                                        # 清理缓存
```

---

## 总结

**uv 的优势**：
- ⚡ 极快的安装和依赖解析速度
- 🎯 统一的工具链（Python 版本 + 虚拟环境 + 包管理）
- 🔒 可靠的依赖解析，避免冲突
- 🪶 轻量级，单个二进制文件
- 🔄 与现有工具生态兼容

**何时使用 uv**：
- ✅ 新项目（强烈推荐）
- ✅ 需要快速安装依赖
- ✅ 需要管理多个 Python 版本
- ✅ CI/CD 流程（加速构建）

**何时考虑其他工具**：
- 需要复杂的依赖管理（考虑 Poetry）
- 需要 Conda 特有的包（如科学计算）
- 团队已有成熟的工具链

**开始使用 uv**：
```bash
# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 创建第一个项目
mkdir my_project && cd my_project
uv python pin 3.12
uv venv
source .venv/bin/activate
uv pip install requests

# 开始编码！
```

---

## 参考资源

- 📚 [官方文档](https://docs.astral.sh/uv/)
- 💻 [GitHub 仓库](https://github.com/astral-sh/uv)
- 📖 [发布说明](https://github.com/astral-sh/uv/releases)
- 🎯 [Astral 博客](https://astral.sh/blog)

**相关工具**：
- [Ruff](https://github.com/astral-sh/ruff) - 极速 Python linter（同一团队开发）
- [pip-tools](https://github.com/jazzband/pip-tools) - uv pip compile 的灵感来源
- [pyenv](https://github.com/pyenv/pyenv) - Python 版本管理工具
