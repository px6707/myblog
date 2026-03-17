---
title: Linux 基础命令完整教程
published: 2026-02-24
description: 详细介绍 Linux 常用命令及其使用方法和实战示例
tags: [Linux, Shell, 命令行, 运维]
category: Linux
licenseName: "Unlicensed"
author: panxiao
sourceLink: "https://github.com/px6707/myblog"
draft: false
---

# Linux 基础命令完整教程

## 目录
- [文件和目录操作](#文件和目录操作)
- [文件查看和编辑](#文件查看和编辑)
- [文件搜索](#文件搜索)
- [文件权限管理](#文件权限管理)
- [进程管理](#进程管理)
- [系统信息](#系统信息)
- [网络命令](#网络命令)
- [压缩和解压](#压缩和解压)
- [用户和组管理](#用户和组管理)
- [磁盘管理](#磁盘管理)
- [文本处理](#文本处理)
- [其他常用命令](#其他常用命令)

---

## 文件和目录操作

### `ls` - 列出目录内容

列出当前目录或指定目录的文件和子目录。

```bash
# 列出当前目录
ls

# 列出详细信息（权限、所有者、大小、修改时间等）
ls -l

# 列出所有文件（包括隐藏文件，以 . 开头的文件）
ls -a

# 组合使用：列出所有文件的详细信息
ls -la

# 按时间排序，最新的在前
ls -lt

# 按大小排序
ls -lS

# 以人类可读的格式显示文件大小（K, M, G）
ls -lh

# 递归列出所有子目录
ls -R

# 列出指定目录
ls /home/user/documents
```

### `cd` - 切换目录

改变当前工作目录。

```bash
# 切换到指定目录
cd /home/user/documents

# 切换到用户主目录
cd ~
# 或直接
cd

# 切换到上一级目录
cd ..

# 切换到上两级目录
cd ../..

# 切换到上一次所在的目录
cd -

# 切换到根目录
cd /
```

### `pwd` - 显示当前目录

打印当前工作目录的完整路径。

```bash
# 显示当前目录的绝对路径
pwd

# 输出示例: /home/user/documents
```

### `mkdir` - 创建目录

创建新目录。

```bash
# 创建单个目录
mkdir mydir

# 创建多个目录
mkdir dir1 dir2 dir3

# 递归创建多级目录（如果父目录不存在，一并创建）
mkdir -p parent/child/grandchild

# 创建目录并设置权限
mkdir -m 755 mydir
```

### `rmdir` - 删除空目录

删除空目录（目录必须为空）。

```bash
# 删除空目录
rmdir emptydir

# 递归删除空目录
rmdir -p parent/child/grandchild
```

### `rm` - 删除文件或目录

删除文件或目录。

```bash
# 删除文件
rm file.txt

# 删除多个文件
rm file1.txt file2.txt file3.txt

# 删除目录及其内容（递归删除）
rm -r mydir

# 强制删除，不提示确认
rm -f file.txt

# 递归强制删除目录（危险操作，谨慎使用）
rm -rf mydir

# 删除前提示确认
rm -i file.txt

# 删除所有 .txt 文件
rm *.txt
```

### `cp` - 复制文件或目录

复制文件或目录到指定位置。

```bash
# 复制文件
cp source.txt destination.txt

# 复制文件到指定目录
cp file.txt /home/user/backup/

# 复制多个文件到目录
cp file1.txt file2.txt /home/user/backup/

# 递归复制目录
cp -r sourcedir/ destdir/

# 复制时保留文件属性（权限、时间戳等）
cp -p file.txt backup.txt

# 交互式复制（覆盖前提示）
cp -i source.txt destination.txt

# 复制时显示详细信息
cp -v file.txt /backup/
```

### `mv` - 移动或重命名文件

移动文件/目录到新位置，或重命名文件/目录。

```bash
# 重命名文件
mv oldname.txt newname.txt

# 移动文件到目录
mv file.txt /home/user/documents/

# 移动多个文件到目录
mv file1.txt file2.txt /home/user/documents/

# 移动目录
mv olddir/ newdir/

# 交互式移动（覆盖前提示）
mv -i source.txt destination.txt

# 强制移动（不提示）
mv -f source.txt destination.txt
```

### `touch` - 创建空文件或更新时间戳

创建新的空文件，或更新已存在文件的访问和修改时间。

```bash
# 创建空文件
touch newfile.txt

# 创建多个空文件
touch file1.txt file2.txt file3.txt

# 更新文件的时间戳为当前时间
touch existingfile.txt

# 只更新访问时间
touch -a file.txt

# 只更新修改时间
touch -m file.txt
```

---

## 文件查看和编辑

### `cat` - 查看文件内容

连接文件并打印到标准输出。

```bash
# 查看文件内容
cat file.txt

# 查看多个文件
cat file1.txt file2.txt

# 显示行号
cat -n file.txt

# 合并多个文件到新文件
cat file1.txt file2.txt > merged.txt

# 追加内容到文件
cat file1.txt >> file2.txt

# 显示非打印字符
cat -v file.txt
```

### `less` - 分页查看文件

以分页方式查看文件内容，支持向前和向后翻页。

```bash
# 分页查看文件
less file.txt

# 常用快捷键：
# 空格键 - 向下翻一页
# b - 向上翻一页
# j 或 ↓ - 向下移动一行
# k 或 ↑ - 向上移动一行
# g - 跳到文件开头
# G - 跳到文件末尾
# /pattern - 向下搜索 pattern
# ?pattern - 向上搜索 pattern
# n - 重复上次搜索
# q - 退出
```

### `more` - 分页查看文件

以分页方式查看文件（只能向下翻页）。

```bash
# 分页查看文件
more file.txt

# 常用快捷键：
# 空格键 - 向下翻一页
# Enter - 向下移动一行
# q - 退出
```

### `head` - 查看文件开头

显示文件的前几行。

```bash
# 显示前 10 行（默认）
head file.txt

# 显示前 20 行
head -n 20 file.txt
# 或简写
head -20 file.txt

# 显示前 100 个字节
head -c 100 file.txt

# 查看多个文件的开头
head file1.txt file2.txt
```

### `tail` - 查看文件末尾

显示文件的后几行。

```bash
# 显示后 10 行（默认）
tail file.txt

# 显示后 20 行
tail -n 20 file.txt
# 或简写
tail -20 file.txt

# 实时监控文件变化（常用于查看日志）
tail -f /var/log/syslog

# 从第 100 行开始显示到文件末尾
tail -n +100 file.txt

# 实时监控，并显示最后 50 行
tail -n 50 -f logfile.log
```

### `nano` - 文本编辑器

简单易用的命令行文本编辑器。

```bash
# 打开或创建文件
nano file.txt

# 常用快捷键：
# Ctrl + O - 保存文件
# Ctrl + X - 退出编辑器
# Ctrl + K - 剪切当前行
# Ctrl + U - 粘贴
# Ctrl + W - 搜索
# Ctrl + \ - 替换
# Ctrl + G - 显示帮助
```

### `vi` / `vim` - 强大的文本编辑器

功能强大的文本编辑器。

```bash
# 打开文件
vi file.txt
# 或
vim file.txt

# 基本操作：
# i - 进入插入模式（开始编辑）
# Esc - 退出插入模式，回到命令模式
# :w - 保存文件
# :q - 退出
# :wq 或 :x - 保存并退出
# :q! - 不保存强制退出
# dd - 删除当前行
# yy - 复制当前行
# p - 粘贴
# u - 撤销
# /pattern - 搜索
# :set number - 显示行号
```

---

## 文件搜索

### `find` - 查找文件

在目录树中搜索文件。

```bash
# 在当前目录查找文件
find . -name "file.txt"

# 在指定目录查找
find /home/user -name "*.txt"

# 忽略大小写查找
find . -iname "file.txt"

# 查找目录
find . -type d -name "mydir"

# 查找文件
find . -type f -name "*.log"

# 查找大于 100MB 的文件
find . -type f -size +100M

# 查找小于 10KB 的文件
find . -type f -size -10k

# 查找 7 天内修改过的文件
find . -type f -mtime -7

# 查找 30 天前修改的文件
find . -type f -mtime +30

# 查找并删除（危险操作）
find . -name "*.tmp" -delete

# 查找并执行命令
find . -name "*.txt" -exec cat {} \;

# 查找空文件
find . -type f -empty

# 查找空目录
find . -type d -empty

# 按权限查找
find . -type f -perm 644
```

### `locate` - 快速查找文件

使用数据库快速查找文件（需要先更新数据库）。

```bash
# 查找文件
locate file.txt

# 忽略大小写
locate -i file.txt

# 只显示存在的文件
locate -e file.txt

# 限制结果数量
locate -n 10 file.txt

# 更新数据库（需要 root 权限）
sudo updatedb
```

### `which` - 查找命令位置

显示命令的完整路径。

```bash
# 查找命令位置
which python

# 输出示例: /usr/bin/python

# 查找多个命令
which python java node
```

### `whereis` - 查找命令、源码和手册

查找二进制文件、源代码和手册页的位置。

```bash
# 查找命令相关文件
whereis python

# 输出示例: python: /usr/bin/python /usr/lib/python2.7 /usr/share/man/man1/python.1.gz

# 只查找二进制文件
whereis -b python

# 只查找手册
whereis -m python
```

---

## 文件权限管理

### `chmod` - 修改文件权限

改变文件或目录的访问权限。

```bash
# 使用数字模式设置权限
# 权限说明：
# r(读)=4, w(写)=2, x(执行)=1
# 第一位：所有者权限
# 第二位：组权限
# 第三位：其他用户权限

# 设置文件权限为 rwxr-xr-x (755)
chmod 755 file.txt

# 设置文件权限为 rw-r--r-- (644)
chmod 644 file.txt

# 递归修改目录权限
chmod -R 755 mydir/

# 使用符号模式
# u=用户(user), g=组(group), o=其他(others), a=所有(all)
# +=添加权限, -=移除权限, ==设置权限

# 给所有者添加执行权限
chmod u+x file.sh

# 给所有人添加读权限
chmod a+r file.txt

# 移除组的写权限
chmod g-w file.txt

# 给所有者设置读写权限，其他人只读
chmod u=rw,go=r file.txt

# 使脚本可执行
chmod +x script.sh
```

### `chown` - 修改文件所有者

改变文件或目录的所有者和所属组。

```bash
# 修改文件所有者
sudo chown username file.txt

# 修改文件所有者和组
sudo chown username:groupname file.txt

# 只修改组
sudo chown :groupname file.txt

# 递归修改目录
sudo chown -R username:groupname mydir/

# 修改符号链接本身（不是目标文件）
sudo chown -h username symlink
```

### `chgrp` - 修改文件所属组

改变文件或目录的所属组。

```bash
# 修改文件所属组
sudo chgrp groupname file.txt

# 递归修改目录
sudo chgrp -R groupname mydir/
```

### `umask` - 设置默认权限

设置创建新文件和目录时的默认权限掩码。

```bash
# 查看当前 umask 值
umask

# 设置 umask（新文件默认权限为 666-022=644）
umask 022

# 设置 umask（新文件默认权限为 666-077=600）
umask 077
```

---

## 进程管理

### `ps` - 显示进程状态

显示当前运行的进程。

```bash
# 显示当前用户的进程
ps

# 显示所有进程
ps aux

# 显示详细的进程树
ps auxf

# 显示指定用户的进程
ps -u username

# 显示进程的完整命令
ps aux --width 200

# 按内存使用排序
ps aux --sort=-%mem | head

# 按 CPU 使用排序
ps aux --sort=-%cpu | head

# 查找特定进程
ps aux | grep python
```

### `top` - 实时显示进程

动态显示系统中进程的资源占用情况。

```bash
# 启动 top
top

# 常用快捷键：
# q - 退出
# h - 帮助
# k - 终止进程（输入 PID）
# M - 按内存使用排序
# P - 按 CPU 使用排序
# 1 - 显示每个 CPU 核心的使用情况
# u - 显示指定用户的进程

# 指定刷新间隔（秒）
top -d 5

# 只显示指定用户的进程
top -u username
```

### `htop` - 增强版 top

更友好的交互式进程查看器（需要安装）。

```bash
# 启动 htop
htop

# 常用快捷键：
# F1 - 帮助
# F2 - 设置
# F3 - 搜索进程
# F4 - 过滤
# F5 - 树形显示
# F6 - 排序
# F9 - 终止进程
# F10 - 退出
```

### `kill` - 终止进程

发送信号给进程，通常用于终止进程。

```bash
# 终止进程（默认发送 SIGTERM 信号）
kill PID

# 强制终止进程（发送 SIGKILL 信号）
kill -9 PID

# 发送指定信号
kill -SIGTERM PID

# 终止多个进程
kill PID1 PID2 PID3

# 列出所有可用信号
kill -l
```

### `killall` - 按名称终止进程

通过进程名终止所有匹配的进程。

```bash
# 终止所有名为 firefox 的进程
killall firefox

# 强制终止
killall -9 firefox

# 交互式终止（确认后终止）
killall -i firefox
```

### `pkill` - 按模式终止进程

通过模式匹配终止进程。

```bash
# 终止所有包含 python 的进程
pkill python

# 终止指定用户的进程
pkill -u username

# 强制终止
pkill -9 python
```

### `bg` / `fg` - 后台/前台运行

管理后台和前台任务。

```bash
# 将当前任务放到后台（先按 Ctrl+Z 暂停任务）
bg

# 将后台任务调到前台
fg

# 将指定任务调到前台
fg %1

# 列出后台任务
jobs

# 在后台运行命令
command &

# 运行命令并忽略挂断信号
nohup command &
```

### `nice` / `renice` - 调整进程优先级

设置或修改进程的优先级。

```bash
# 以指定优先级运行命令（-20 到 19，数值越小优先级越高）
nice -n 10 command

# 修改运行中进程的优先级
renice -n 5 -p PID

# 修改用户所有进程的优先级
sudo renice -n 5 -u username
```

---

## 系统信息

### `uname` - 显示系统信息

显示系统的基本信息。

```bash
# 显示内核名称
uname

# 显示所有信息
uname -a

# 显示内核版本
uname -r

# 显示机器硬件名称
uname -m

# 显示操作系统名称
uname -o
```

### `hostname` - 显示或设置主机名

显示或修改系统的主机名。

```bash
# 显示主机名
hostname

# 显示 FQDN（完全限定域名）
hostname -f

# 设置主机名（需要 root 权限）
sudo hostname newhostname
```

### `uptime` - 显示系统运行时间

显示系统已运行时间和负载。

```bash
# 显示系统运行时间、用户数和负载
uptime

# 输出示例: 10:30:15 up 5 days, 2:15, 3 users, load average: 0.15, 0.20, 0.18
```

### `date` - 显示或设置日期时间

显示或设置系统日期和时间。

```bash
# 显示当前日期和时间
date

# 以指定格式显示
date "+%Y-%m-%d %H:%M:%S"

# 显示 UTC 时间
date -u

# 设置日期时间（需要 root 权限）
sudo date -s "2024-01-01 12:00:00"

# 显示指定日期
date -d "2024-01-01"

# 显示昨天的日期
date -d "yesterday"

# 显示 3 天后的日期
date -d "+3 days"
```

### `cal` - 显示日历

显示日历。

```bash
# 显示当前月份日历
cal

# 显示整年日历
cal 2024

# 显示指定月份
cal 12 2024

# 显示前后三个月
cal -3
```

### `whoami` - 显示当前用户

显示当前登录的用户名。

```bash
# 显示当前用户名
whoami
```

### `who` - 显示登录用户

显示当前登录系统的用户。

```bash
# 显示所有登录用户
who

# 显示详细信息
who -a

# 显示启动时间
who -b
```

### `w` - 显示登录用户及其活动

显示当前登录用户及其正在执行的操作。

```bash
# 显示登录用户和活动
w

# 不显示头部信息
w -h
```

### `last` - 显示登录历史

显示用户登录历史记录。

```bash
# 显示最近的登录记录
last

# 显示指定用户的登录记录
last username

# 显示最近 10 条记录
last -n 10
```

---

## 网络命令

### `ping` - 测试网络连通性

测试与目标主机的网络连接。

```bash
# ping 主机
ping google.com

# ping 指定次数后停止
ping -c 4 google.com

# 设置 ping 间隔（秒）
ping -i 2 google.com

# ping IPv6 地址
ping6 ipv6.google.com
```

### `ifconfig` - 配置网络接口

显示或配置网络接口（较老的命令）。

```bash
# 显示所有网络接口
ifconfig

# 显示指定接口
ifconfig eth0

# 启用网络接口
sudo ifconfig eth0 up

# 禁用网络接口
sudo ifconfig eth0 down

# 设置 IP 地址
sudo ifconfig eth0 192.168.1.100
```

### `ip` - 网络配置工具

现代的网络配置工具（推荐使用）。

```bash
# 显示所有网络接口
ip addr show
# 或简写
ip a

# 显示路由表
ip route show
# 或简写
ip r

# 显示指定接口
ip addr show eth0

# 添加 IP 地址
sudo ip addr add 192.168.1.100/24 dev eth0

# 删除 IP 地址
sudo ip addr del 192.168.1.100/24 dev eth0

# 启用接口
sudo ip link set eth0 up

# 禁用接口
sudo ip link set eth0 down
```

### `netstat` - 网络统计

显示网络连接、路由表、接口统计等。

```bash
# 显示所有连接
netstat -a

# 显示监听端口
netstat -l

# 显示 TCP 连接
netstat -t

# 显示 UDP 连接
netstat -u

# 显示进程信息
sudo netstat -p

# 显示数字地址（不解析主机名）
netstat -n

# 组合使用：显示所有 TCP 监听端口及进程
sudo netstat -tlnp

# 显示路由表
netstat -r
```

### `ss` - Socket 统计

现代的 socket 统计工具（比 netstat 更快）。

```bash
# 显示所有 socket
ss -a

# 显示监听的 socket
ss -l

# 显示 TCP socket
ss -t

# 显示 UDP socket
ss -u

# 显示进程信息
ss -p

# 组合使用：显示所有监听的 TCP 端口
ss -tln

# 显示指定端口的连接
ss -tn sport = :80
```

### `curl` - 传输数据

从服务器传输数据或向服务器发送数据。

```bash
# 获取网页内容
curl https://example.com

# 下载文件
curl -O https://example.com/file.zip

# 下载并指定文件名
curl -o myfile.zip https://example.com/file.zip

# 显示响应头
curl -I https://example.com

# 发送 POST 请求
curl -X POST -d "param1=value1&param2=value2" https://example.com/api

# 发送 JSON 数据
curl -X POST -H "Content-Type: application/json" -d '{"key":"value"}' https://example.com/api

# 跟随重定向
curl -L https://example.com

# 显示详细信息
curl -v https://example.com

# 使用代理
curl -x proxy.example.com:8080 https://example.com
```

### `wget` - 下载文件

从网络下载文件。

```bash
# 下载文件
wget https://example.com/file.zip

# 下载并指定文件名
wget -O myfile.zip https://example.com/file.zip

# 断点续传
wget -c https://example.com/largefile.zip

# 后台下载
wget -b https://example.com/file.zip

# 限速下载（KB/s）
wget --limit-rate=200k https://example.com/file.zip

# 下载整个网站
wget -r -p -k https://example.com

# 下载 FTP 文件
wget ftp://ftp.example.com/file.zip
```

### `ssh` - 远程登录

通过 SSH 协议连接远程主机。

```bash
# 连接远程主机
ssh username@hostname

# 指定端口
ssh -p 2222 username@hostname

# 使用密钥文件
ssh -i ~/.ssh/id_rsa username@hostname

# 执行远程命令
ssh username@hostname 'ls -la'

# 端口转发
ssh -L 8080:localhost:80 username@hostname
```

### `scp` - 远程复制文件

通过 SSH 在主机间复制文件。

```bash
# 复制本地文件到远程主机
scp file.txt username@hostname:/path/to/destination/

# 复制远程文件到本地
scp username@hostname:/path/to/file.txt /local/path/

# 递归复制目录
scp -r mydir/ username@hostname:/path/to/destination/

# 指定端口
scp -P 2222 file.txt username@hostname:/path/

# 显示进度
scp -v file.txt username@hostname:/path/
```

---

## 压缩和解压

### `tar` - 打包和解包

创建或提取 tar 归档文件。

```bash
# 创建 tar 归档
tar -cvf archive.tar files/

# 创建 gzip 压缩的 tar 归档
tar -czvf archive.tar.gz files/

# 创建 bzip2 压缩的 tar 归档
tar -cjvf archive.tar.bz2 files/

# 解压 tar 归档
tar -xvf archive.tar

# 解压 gzip 压缩的 tar 归档
tar -xzvf archive.tar.gz

# 解压 bzip2 压缩的 tar 归档
tar -xjvf archive.tar.bz2

# 解压到指定目录
tar -xzvf archive.tar.gz -C /path/to/directory/

# 查看归档内容（不解压）
tar -tvf archive.tar

# 只解压指定文件
tar -xzvf archive.tar.gz file.txt

# 参数说明：
# c - 创建归档
# x - 解压归档
# v - 显示详细信息
# f - 指定文件名
# z - 使用 gzip 压缩/解压
# j - 使用 bzip2 压缩/解压
# t - 列出内容
# C - 指定目录
```

### `gzip` / `gunzip` - 压缩和解压

使用 gzip 算法压缩文件。

```bash
# 压缩文件（原文件会被删除）
gzip file.txt

# 解压文件
gunzip file.txt.gz

# 保留原文件
gzip -k file.txt

# 递归压缩目录中的文件
gzip -r mydir/

# 查看压缩文件内容
zcat file.txt.gz

# 指定压缩级别（1-9，9 最高）
gzip -9 file.txt
```

### `bzip2` / `bunzip2` - 压缩和解压

使用 bzip2 算法压缩文件（压缩率更高）。

```bash
# 压缩文件
bzip2 file.txt

# 解压文件
bunzip2 file.txt.bz2

# 保留原文件
bzip2 -k file.txt

# 查看压缩文件内容
bzcat file.txt.bz2
```

### `zip` / `unzip` - ZIP 压缩

创建和解压 ZIP 文件。

```bash
# 创建 zip 文件
zip archive.zip file1.txt file2.txt

# 递归压缩目录
zip -r archive.zip mydir/

# 解压 zip 文件
unzip archive.zip

# 解压到指定目录
unzip archive.zip -d /path/to/directory/

# 查看 zip 内容（不解压）
unzip -l archive.zip

# 解压时覆盖已存在文件
unzip -o archive.zip

# 解压时不覆盖已存在文件
unzip -n archive.zip

# 添加文件到已存在的 zip
zip -u archive.zip newfile.txt

# 从 zip 中删除文件
zip -d archive.zip file.txt
```

---

## 用户和组管理

### `useradd` - 添加用户

创建新用户账户。

```bash
# 创建用户
sudo useradd username

# 创建用户并指定主目录
sudo useradd -m -d /home/username username

# 创建用户并指定 shell
sudo useradd -s /bin/bash username

# 创建用户并指定用户组
sudo useradd -g groupname username

# 创建用户并添加到多个组
sudo useradd -G group1,group2 username

# 创建系统用户
sudo useradd -r username
```

### `usermod` - 修改用户

修改用户账户信息。

```bash
# 修改用户主目录
sudo usermod -d /new/home username

# 修改用户 shell
sudo usermod -s /bin/zsh username

# 添加用户到组
sudo usermod -aG groupname username

# 修改用户名
sudo usermod -l newname oldname

# 锁定用户
sudo usermod -L username

# 解锁用户
sudo usermod -U username
```

### `userdel` - 删除用户

删除用户账户。

```bash
# 删除用户
sudo userdel username

# 删除用户及其主目录
sudo userdel -r username
```

### `passwd` - 修改密码

修改用户密码。

```bash
# 修改当前用户密码
passwd

# 修改指定用户密码（需要 root 权限）
sudo passwd username

# 删除用户密码
sudo passwd -d username

# 锁定用户密码
sudo passwd -l username

# 解锁用户密码
sudo passwd -u username
```

### `groupadd` - 添加组

创建新用户组。

```bash
# 创建组
sudo groupadd groupname

# 创建组并指定 GID
sudo groupadd -g 1001 groupname
```

### `groupdel` - 删除组

删除用户组。

```bash
# 删除组
sudo groupdel groupname
```

### `groups` - 显示用户所属组

显示用户所属的所有组。

```bash
# 显示当前用户所属组
groups

# 显示指定用户所属组
groups username
```

### `id` - 显示用户和组信息

显示用户的 UID、GID 和所属组。

```bash
# 显示当前用户信息
id

# 显示指定用户信息
id username

# 只显示 UID
id -u

# 只显示 GID
id -g

# 显示所有组 ID
id -G
```

### `su` - 切换用户

切换到其他用户。

```bash
# 切换到 root 用户
su

# 切换到指定用户
su username

# 切换用户并加载其环境
su - username

# 以其他用户身份执行命令
su -c "command" username
```

### `sudo` - 以超级用户权限执行

以 root 权限执行命令。

```bash
# 以 root 权限执行命令
sudo command

# 切换到 root shell
sudo -i

# 以指定用户身份执行命令
sudo -u username command

# 编辑 sudoers 文件
sudo visudo

# 列出当前用户的 sudo 权限
sudo -l
```

---

## 磁盘管理

### `df` - 显示磁盘空间

显示文件系统的磁盘空间使用情况。

```bash
# 显示所有文件系统
df

# 以人类可读格式显示
df -h

# 显示 inode 使用情况
df -i

# 只显示指定类型的文件系统
df -t ext4

# 显示指定目录所在的文件系统
df /home
```

### `du` - 显示目录空间

显示目录或文件的磁盘使用情况。

```bash
# 显示当前目录大小
du

# 以人类可读格式显示
du -h

# 显示目录总大小
du -sh

# 显示所有文件和目录的大小
du -ah

# 显示指定深度
du -h --max-depth=1

# 按大小排序
du -h | sort -h

# 显示最大的 10 个目录
du -h --max-depth=1 | sort -hr | head -10
```

### `mount` - 挂载文件系统

挂载文件系统到目录。

```bash
# 显示所有挂载点
mount

# 挂载设备到目录
sudo mount /dev/sdb1 /mnt/usb

# 挂载 ISO 文件
sudo mount -o loop image.iso /mnt/iso

# 以只读方式挂载
sudo mount -o ro /dev/sdb1 /mnt/usb

# 重新挂载为读写
sudo mount -o remount,rw /mnt/usb
```

### `umount` - 卸载文件系统

卸载已挂载的文件系统。

```bash
# 卸载文件系统
sudo umount /mnt/usb

# 强制卸载
sudo umount -f /mnt/usb

# 延迟卸载（等待使用完毕）
sudo umount -l /mnt/usb
```

### `fdisk` - 磁盘分区工具

管理磁盘分区。

```bash
# 列出所有磁盘和分区
sudo fdisk -l

# 编辑指定磁盘的分区
sudo fdisk /dev/sdb

# 在 fdisk 交互模式中：
# m - 显示帮助
# p - 显示分区表
# n - 创建新分区
# d - 删除分区
# w - 保存并退出
# q - 不保存退出
```

### `mkfs` - 创建文件系统

在分区上创建文件系统。

```bash
# 创建 ext4 文件系统
sudo mkfs.ext4 /dev/sdb1

# 创建 ext3 文件系统
sudo mkfs.ext3 /dev/sdb1

# 创建 FAT32 文件系统
sudo mkfs.vfat /dev/sdb1

# 创建 NTFS 文件系统
sudo mkfs.ntfs /dev/sdb1
```

### `fsck` - 检查和修复文件系统

检查和修复文件系统错误。

```bash
# 检查文件系统（需要先卸载）
sudo fsck /dev/sdb1

# 自动修复错误
sudo fsck -y /dev/sdb1

# 检查所有文件系统
sudo fsck -A
```

---

## 文本处理

### `grep` - 搜索文本

在文件中搜索匹配的文本模式。

```bash
# 在文件中搜索
grep "pattern" file.txt

# 忽略大小写
grep -i "pattern" file.txt

# 递归搜索目录
grep -r "pattern" /path/to/dir/

# 显示行号
grep -n "pattern" file.txt

# 反向匹配（显示不匹配的行）
grep -v "pattern" file.txt

# 只显示匹配的文件名
grep -l "pattern" *.txt

# 显示匹配行的前后各 3 行
grep -C 3 "pattern" file.txt

# 显示匹配行后的 3 行
grep -A 3 "pattern" file.txt

# 显示匹配行前的 3 行
grep -B 3 "pattern" file.txt

# 使用正则表达式
grep -E "pattern1|pattern2" file.txt

# 统计匹配行数
grep -c "pattern" file.txt

# 只显示匹配的部分
grep -o "pattern" file.txt
```

### `sed` - 流编辑器

对文本进行转换和处理。

```bash
# 替换文本（只显示，不修改文件）
sed 's/old/new/' file.txt

# 替换所有匹配项
sed 's/old/new/g' file.txt

# 直接修改文件
sed -i 's/old/new/g' file.txt

# 删除空行
sed '/^$/d' file.txt

# 删除指定行
sed '3d' file.txt

# 删除第 2 到第 5 行
sed '2,5d' file.txt

# 显示指定行
sed -n '10,20p' file.txt

# 在行首添加文本
sed 's/^/prefix/' file.txt

# 在行尾添加文本
sed 's/$/suffix/' file.txt

# 多个替换
sed -e 's/old1/new1/g' -e 's/old2/new2/g' file.txt
```

### `awk` - 文本处理工具

强大的文本分析和处理工具。

```bash
# 打印第一列
awk '{print $1}' file.txt

# 打印第一列和第三列
awk '{print $1, $3}' file.txt

# 打印最后一列
awk '{print $NF}' file.txt

# 使用指定分隔符
awk -F: '{print $1}' /etc/passwd

# 打印行号和内容
awk '{print NR, $0}' file.txt

# 条件过滤
awk '$3 > 100' file.txt

# 计算总和
awk '{sum += $1} END {print sum}' file.txt

# 计算平均值
awk '{sum += $1; count++} END {print sum/count}' file.txt

# 格式化输出
awk '{printf "%-10s %s\n", $1, $2}' file.txt
```

### `cut` - 提取列

从文件中提取指定列。

```bash
# 提取第 1 列（以制表符分隔）
cut -f1 file.txt

# 提取第 1 和第 3 列
cut -f1,3 file.txt

# 提取第 1 到第 3 列
cut -f1-3 file.txt

# 使用指定分隔符
cut -d: -f1 /etc/passwd

# 提取字符位置 1-10
cut -c1-10 file.txt
```

### `sort` - 排序

对文本行进行排序。

```bash
# 排序文件
sort file.txt

# 逆序排序
sort -r file.txt

# 数字排序
sort -n file.txt

# 按第 2 列排序
sort -k2 file.txt

# 去重排序
sort -u file.txt

# 忽略大小写
sort -f file.txt

# 按月份排序
sort -M file.txt
```

### `uniq` - 去重

删除重复的相邻行。

```bash
# 去除重复行（需要先排序）
sort file.txt | uniq

# 只显示重复行
sort file.txt | uniq -d

# 只显示不重复的行
sort file.txt | uniq -u

# 统计每行出现次数
sort file.txt | uniq -c

# 忽略大小写
sort file.txt | uniq -i
```

### `wc` - 统计

统计文件的行数、单词数和字节数。

```bash
# 统计行数、单词数、字节数
wc file.txt

# 只统计行数
wc -l file.txt

# 只统计单词数
wc -w file.txt

# 只统计字节数
wc -c file.txt

# 只统计字符数
wc -m file.txt

# 统计多个文件
wc file1.txt file2.txt
```

### `tr` - 字符转换

转换或删除字符。

```bash
# 转换为大写
echo "hello" | tr 'a-z' 'A-Z'

# 转换为小写
echo "HELLO" | tr 'A-Z' 'a-z'

# 删除字符
echo "hello123" | tr -d '0-9'

# 压缩重复字符
echo "hello    world" | tr -s ' '

# 替换字符
echo "hello" | tr 'l' 'L'
```

---

## 其他常用命令

### `echo` - 输出文本

显示文本或变量值。

```bash
# 输出文本
echo "Hello World"

# 输出变量
echo $HOME

# 不换行输出
echo -n "Hello"

# 输出到文件（覆盖）
echo "text" > file.txt

# 输出到文件（追加）
echo "text" >> file.txt

# 输出转义字符
echo -e "Line1\nLine2"
```

### `printf` - 格式化输出

按指定格式输出文本。

```bash
# 格式化输出
printf "Name: %s, Age: %d\n" "Alice" 25

# 输出到文件
printf "text\n" > file.txt

# 格式化数字
printf "%.2f\n" 3.14159
```

### `history` - 命令历史

显示命令历史记录。

```bash
# 显示命令历史
history

# 显示最近 10 条命令
history 10

# 执行历史命令（编号为 100）
!100

# 执行上一条命令
!!

# 执行最近的以 git 开头的命令
!git

# 清除历史记录
history -c
```

### `alias` - 命令别名

创建命令别名。

```bash
# 创建别名
alias ll='ls -la'

# 查看所有别名
alias

# 删除别名
unalias ll

# 临时忽略别名
\ls
```

### `man` - 查看手册

显示命令的手册页。

```bash
# 查看命令手册
man ls

# 搜索手册页
man -k keyword

# 查看指定章节
man 5 passwd
```

### `help` - 命令帮助

显示 shell 内置命令的帮助。

```bash
# 查看内置命令帮助
help cd

# 查看所有内置命令
help
```

### `clear` - 清屏

清除终端屏幕。

```bash
# 清屏
clear

# 或使用快捷键 Ctrl+L
```

### `exit` - 退出

退出当前 shell 或终端。

```bash
# 退出
exit

# 带退出码退出
exit 0
```

### `ln` - 创建链接

创建硬链接或符号链接。

```bash
# 创建符号链接（软链接）
ln -s /path/to/file linkname

# 创建硬链接
ln /path/to/file linkname

# 强制创建（覆盖已存在的链接）
ln -sf /path/to/file linkname

# 查看链接指向
ls -l linkname
```

### `xargs` - 构建命令

从标准输入构建和执行命令。

```bash
# 删除查找到的文件
find . -name "*.tmp" | xargs rm

# 指定每次传递的参数数量
echo "1 2 3 4 5" | xargs -n 2 echo

# 使用占位符
find . -name "*.txt" | xargs -I {} cp {} /backup/

# 并行执行
find . -name "*.jpg" | xargs -P 4 -I {} convert {} {}.png
```

### `tee` - 双向输出

同时输出到标准输出和文件。

```bash
# 输出到屏幕和文件
echo "text" | tee file.txt

# 追加到文件
echo "text" | tee -a file.txt

# 输出到多个文件
echo "text" | tee file1.txt file2.txt

# 配合 sudo 写入需要权限的文件
echo "text" | sudo tee /etc/config.txt
```

### `watch` - 定期执行命令

定期执行命令并显示结果。

```bash
# 每 2 秒执行一次（默认）
watch df -h

# 指定间隔时间（秒）
watch -n 5 ls -l

# 高亮显示变化
watch -d free -m

# 执行复杂命令
watch 'ps aux | grep python'
```

---

## 总结

这份教程涵盖了 Linux 最常用的命令，包括：

1. **文件和目录操作**：ls, cd, mkdir, rm, cp, mv, touch
2. **文件查看和编辑**：cat, less, head, tail, nano, vim
3. **文件搜索**：find, locate, which, whereis
4. **文件权限**：chmod, chown, chgrp, umask
5. **进程管理**：ps, top, kill, bg, fg, nice
6. **系统信息**：uname, hostname, uptime, date, whoami
7. **网络命令**：ping, ifconfig, ip, netstat, curl, wget, ssh, scp
8. **压缩解压**：tar, gzip, bzip2, zip
9. **用户管理**：useradd, usermod, passwd, su, sudo
10. **磁盘管理**：df, du, mount, fdisk, mkfs
11. **文本处理**：grep, sed, awk, cut, sort, uniq, wc
12. **其他工具**：echo, history, alias, man, ln, xargs, tee, watch

掌握这些命令，你就能高效地在 Linux 系统中工作了！建议多加练习，熟能生巧。
