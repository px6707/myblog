---
title: NumPy 完整教程 - Python 科学计算基础
published: 2026-02-09
description: 详细介绍 NumPy 的核心概念、API 使用方法和实战示例
tags: [Python, NumPy, 数据分析, 科学计算]
category: Python
licenseName: "Unlicensed"
author: panxiao
sourceLink: "https://github.com/px6707/myblog"
draft: false
---

> NumPy 是 Python 科学计算的基础库，提供高性能的多维数组对象和丰富的数学函数。
----

## 目录
* NumPy 简介
* 安装与导入
* 数组创建
* 数组属性
* 数组索引与切片
* 数组运算
* 数学函数
* 统计函数
* 线性代数
* 数组操作
* 广播机制
* 实战案例

----

## NumPy 简介

### 什么是 NumPy？

NumPy（Numerical Python）是 Python 科学计算的核心库，提供：
- **高性能多维数组对象**（ndarray）
- **丰富的数学函数库**
- **线性代数、傅里叶变换、随机数生成**等工具
- **与 C/C++ 集成**的能力

### 为什么使用 NumPy？

```python
# 传统 Python 列表
python_list = [1, 2, 3, 4, 5]
result = [x * 2 for x in python_list]  # 需要循环

# NumPy 数组（向量化操作）
import numpy as np
numpy_array = np.array([1, 2, 3, 4, 5])
result = numpy_array * 2  # 直接运算，速度快 10-100 倍
```

**核心优势：**
- ✅ **性能优异**：底层用 C 实现，比 Python 列表快 10-100 倍
- ✅ **内存高效**：连续内存存储，占用空间小
- ✅ **向量化操作**：无需显式循环，代码简洁
- ✅ **广播机制**：不同形状数组可以直接运算
- ✅ **生态完善**：Pandas、SciPy、Scikit-learn 等都基于 NumPy

----

## 安装与导入

### 安装 NumPy

```bash
# 使用 pip 安装
pip install numpy

# 使用 conda 安装
conda install numpy

# 安装特定版本
pip install numpy==1.24.0
```

### 导入 NumPy

```python
import numpy as np

# 查看版本
print(np.__version__)  # 输出：1.24.3

# 查看配置信息
np.show_config()
```

----

## 数组创建

### 1. 从 Python 列表创建

```python
import numpy as np

# 一维数组
arr1d = np.array([1, 2, 3, 4, 5])
print(arr1d)  # [1 2 3 4 5]

# 二维数组
arr2d = np.array([[1, 2, 3], [4, 5, 6]])
print(arr2d)
# [[1 2 3]
#  [4 5 6]]

# 指定数据类型
arr_float = np.array([1, 2, 3], dtype=np.float64)
print(arr_float)  # [1. 2. 3.]

# 从元组创建
arr_tuple = np.array((1, 2, 3))
```

**参数说明：**
- `object`：Python 列表、元组或其他序列
- `dtype`：数据类型（int32, float64, complex 等）
- `ndmin`：最小维度数

### 2. 使用内置函数创建

#### `np.zeros()` - 创建全零数组

```python
# 一维全零数组
zeros_1d = np.zeros(5)
print(zeros_1d)  # [0. 0. 0. 0. 0.]

# 二维全零数组
zeros_2d = np.zeros((3, 4))
print(zeros_2d)
# [[0. 0. 0. 0.]
#  [0. 0. 0. 0.]
#  [0. 0. 0. 0.]]

# 指定数据类型
zeros_int = np.zeros((2, 3), dtype=int)
```

**参数说明：**
- `shape`：数组形状，整数或元组
- `dtype`：数据类型，默认 float64

#### `np.ones()` - 创建全一数组

```python
# 一维全一数组
ones_1d = np.ones(4)
print(ones_1d)  # [1. 1. 1. 1.]

# 三维全一数组
ones_3d = np.ones((2, 3, 4))
print(ones_3d.shape)  # (2, 3, 4)
```

#### `np.full()` - 创建指定值数组

```python
# 创建全为 7 的数组
full_arr = np.full((3, 3), 7)
print(full_arr)
# [[7 7 7]
#  [7 7 7]
#  [7 7 7]]

# 创建全为 3.14 的数组
pi_arr = np.full((2, 4), 3.14)
```

**参数说明：**
- `shape`：数组形状
- `fill_value`：填充值

#### `np.eye()` - 创建单位矩阵

```python
# 3x3 单位矩阵
identity = np.eye(3)
print(identity)
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]

# 4x4 单位矩阵，数据类型为整数
identity_int = np.eye(4, dtype=int)

# 非方阵单位矩阵
eye_rect = np.eye(3, 5)
```

**参数说明：**
- `N`：行数
- `M`：列数（默认等于 N）
- `k`：对角线偏移（默认 0）

#### `np.arange()` - 创建等差数组

```python
# 类似 Python 的 range()
arr1 = np.arange(10)
print(arr1)  # [0 1 2 3 4 5 6 7 8 9]

# 指定起始、结束、步长
arr2 = np.arange(1, 10, 2)
print(arr2)  # [1 3 5 7 9]

# 浮点数步长
arr3 = np.arange(0, 1, 0.1)
print(arr3)  # [0.  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9]

# 倒序
arr4 = np.arange(10, 0, -1)
print(arr4)  # [10  9  8  7  6  5  4  3  2  1]
```

**参数说明：**
- `start`：起始值（包含）
- `stop`：结束值（不包含）
- `step`：步长（默认 1）

#### `np.linspace()` - 创建线性等分数组

```python
# 0 到 10 之间等分 5 个数
arr1 = np.linspace(0, 10, 5)
print(arr1)  # [ 0.   2.5  5.   7.5 10. ]

# 0 到 1 之间等分 11 个数
arr2 = np.linspace(0, 1, 11)
print(arr2)  # [0.  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1. ]

# 不包含结束点
arr3 = np.linspace(0, 10, 5, endpoint=False)
print(arr3)  # [0. 2. 4. 6. 8.]

# 返回步长
arr4, step = np.linspace(0, 10, 5, retstep=True)
print(f"数组: {arr4}, 步长: {step}")
```

**参数说明：**
- `start`：起始值
- `stop`：结束值
- `num`：生成的样本数（默认 50）
- `endpoint`：是否包含结束值（默认 True）
- `retstep`：是否返回步长（默认 False）

**`arange` vs `linspace`：**
```python
# arange：指定步长
np.arange(0, 10, 2)    # [0 2 4 6 8]

# linspace：指定数量
np.linspace(0, 10, 5)  # [ 0.   2.5  5.   7.5 10. ]
```

#### `np.logspace()` - 创建对数等分数组

```python
# 10^0 到 10^3 之间对数等分 4 个数
arr1 = np.logspace(0, 3, 4)
print(arr1)  # [   1.   10.  100. 1000.]

# 2^0 到 2^10 之间对数等分 11 个数（以 2 为底）
arr2 = np.logspace(0, 10, 11, base=2)
print(arr2)  # [   1.    2.    4.    8.   16.   32.   64.  128.  256.  512. 1024.]
```

### 3. 随机数组创建

#### `np.random.rand()` - 均匀分布 [0, 1)

```python
# 一维随机数组
rand1 = np.random.rand(5)
print(rand1)  # [0.5488135  0.71518937 0.60276338 0.54488318 0.4236548 ]

# 二维随机数组
rand2 = np.random.rand(3, 4)
print(rand2)
```

#### `np.random.randn()` - 标准正态分布

```python
# 标准正态分布（均值 0，标准差 1）
randn1 = np.random.randn(5)
print(randn1)

# 二维标准正态分布
randn2 = np.random.randn(3, 3)
```

#### `np.random.randint()` - 随机整数

```python
# 0 到 10 之间的随机整数
randint1 = np.random.randint(0, 10, size=5)
print(randint1)  # [6 3 7 4 6]

# 1 到 100 之间的 3x4 随机整数数组
randint2 = np.random.randint(1, 100, size=(3, 4))
print(randint2)
```

**参数说明：**
- `low`：最小值（包含）
- `high`：最大值（不包含）
- `size`：输出形状

#### `np.random.choice()` - 随机抽样

```python
# 从数组中随机选择
arr = np.array([10, 20, 30, 40, 50])
choice1 = np.random.choice(arr, size=3)
print(choice1)  # [30 10 40]

# 不重复抽样
choice2 = np.random.choice(arr, size=3, replace=False)

# 带权重抽样
choice3 = np.random.choice(arr, size=5, p=[0.1, 0.1, 0.2, 0.3, 0.3])
```

#### 设置随机种子

```python
# 设置随机种子，保证结果可复现
np.random.seed(42)
print(np.random.rand(3))  # 每次运行结果相同

# 新的随机数生成器（推荐）
rng = np.random.default_rng(42)
print(rng.random(3))
```

----

## 数组属性

```python
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

# 形状（维度大小）
print(arr.shape)  # (3, 4) - 3 行 4 列

# 维度数
print(arr.ndim)   # 2 - 二维数组

# 元素总数
print(arr.size)   # 12 - 共 12 个元素

# 数据类型
print(arr.dtype)  # int64

# 每个元素的字节大小
print(arr.itemsize)  # 8 字节

# 数组总字节数
print(arr.nbytes)    # 96 字节 (12 * 8)

# 数组的步长（每个维度跨越的字节数）
print(arr.strides)   # (32, 8)
```

**常用属性总结：**

| 属性 | 说明 | 示例 |
|------|------|------|
| `shape` | 数组形状 | `(3, 4)` |
| `ndim` | 维度数 | `2` |
| `size` | 元素总数 | `12` |
| `dtype` | 数据类型 | `int64` |
| `itemsize` | 每个元素字节数 | `8` |
| `nbytes` | 总字节数 | `96` |

----

## 数组索引与切片

### 1. 一维数组索引

```python
arr = np.array([10, 20, 30, 40, 50])

# 基本索引
print(arr[0])      # 10 - 第一个元素
print(arr[-1])     # 50 - 最后一个元素
print(arr[2])      # 30 - 第三个元素

# 切片
print(arr[1:4])    # [20 30 40] - 索引 1 到 3
print(arr[:3])     # [10 20 30] - 前 3 个
print(arr[2:])     # [30 40 50] - 从索引 2 到末尾
print(arr[::2])    # [10 30 50] - 每隔一个取一个
print(arr[::-1])   # [50 40 30 20 10] - 反转数组
```

### 2. 二维数组索引

```python
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# 单个元素访问
print(arr2d[0, 1])     # 2 - 第 0 行第 1 列
print(arr2d[1, 2])     # 6 - 第 1 行第 2 列
print(arr2d[-1, -1])   # 9 - 最后一行最后一列

# 行切片
print(arr2d[0, :])     # [1 2 3] - 第 0 行所有列
print(arr2d[1, :])     # [4 5 6] - 第 1 行所有列

# 列切片
print(arr2d[:, 0])     # [1 4 7] - 所有行的第 0 列
print(arr2d[:, 2])     # [3 6 9] - 所有行的第 2 列

# 子数组切片
print(arr2d[0:2, 1:3])
# [[2 3]
#  [5 6]]

print(arr2d[1:, :2])
# [[4 5]
#  [7 8]]
```

### 3. 布尔索引

```python
arr = np.array([1, 2, 3, 4, 5, 6])

# 创建布尔掩码
mask = arr > 3
print(mask)  # [False False False  True  True  True]

# 使用布尔索引
print(arr[mask])       # [4 5 6]

# 直接使用条件
print(arr[arr > 3])    # [4 5 6]
print(arr[arr % 2 == 0])  # [2 4 6] - 偶数

# 多条件组合
data = np.array([10, -5, 15, -3, 20, 8])
print(data[(data > 0) & (data < 15)])  # [10  8]
print(data[(data < 0) | (data > 15)])  # [-5 -3 20]

# 修改元素
arr[arr > 3] = 0
print(arr)  # [1 2 3 0 0 0]
```

**布尔运算符：**
- `&`：与（AND）
- `|`：或（OR）
- `~`：非（NOT）

### 4. 花式索引（整数数组索引）

```python
arr = np.array([10, 20, 30, 40, 50])

# 使用整数数组索引
indices = np.array([0, 2, 4])
print(arr[indices])    # [10 30 50]

# 二维数组花式索引
arr2d = np.array([[1, 2], [3, 4], [5, 6]])
row_indices = np.array([0, 2])
col_indices = np.array([1, 0])
print(arr2d[row_indices, col_indices])  # [2 5]

# 组合使用
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(arr2d[[0, 2], :])  # 选择第 0 行和第 2 行
# [[1 2 3]
#  [7 8 9]]
```

----

## 数组运算

### 1. 算术运算

```python
arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([10, 20, 30, 40])

# 加法
print(arr1 + arr2)     # [11 22 33 44]
print(np.add(arr1, arr2))

# 减法
print(arr2 - arr1)     # [ 9 18 27 36]
print(np.subtract(arr2, arr1))

# 乘法（逐元素）
print(arr1 * arr2)     # [ 10  40  90 160]
print(np.multiply(arr1, arr2))

# 除法
print(arr2 / arr1)     # [10. 10. 10. 10.]
print(np.divide(arr2, arr1))

# 整除
print(arr2 // arr1)    # [10 10 10 10]

# 取余
print(arr2 % arr1)     # [0 0 0 0]

# 幂运算
print(arr1 ** 2)       # [ 1  4  9 16]
print(np.power(arr1, 2))
```

### 2. 标量运算

```python
arr = np.array([1, 2, 3, 4, 5])

# 所有元素加 10
print(arr + 10)        # [11 12 13 14 15]

# 所有元素乘以 2
print(arr * 2)         # [ 2  4  6  8 10]

# 所有元素平方
print(arr ** 2)        # [ 1  4  9 16 25]

# 所有元素除以 2
print(arr / 2)         # [0.5 1.  1.5 2.  2.5]
```

### 3. 比较运算

```python
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([5, 4, 3, 2, 1])

# 逐元素比较
print(arr1 == arr2)    # [False False  True False False]
print(arr1 > arr2)     # [False False False  True  True]
print(arr1 < arr2)     # [ True  True False False False]
print(arr1 >= 3)       # [False False  True  True  True]

# 比较函数
print(np.equal(arr1, arr2))
print(np.greater(arr1, arr2))
print(np.less_equal(arr1, 3))
```

### 4. 逻辑运算

```python
arr1 = np.array([True, True, False, False])
arr2 = np.array([True, False, True, False])

# 逻辑与
print(np.logical_and(arr1, arr2))  # [ True False False False]

# 逻辑或
print(np.logical_or(arr1, arr2))   # [ True  True  True False]

# 逻辑非
print(np.logical_not(arr1))        # [False False  True  True]

# 逻辑异或
print(np.logical_xor(arr1, arr2))  # [False  True  True False]
```

----

## 数学函数

### 1. 三角函数

```python
angles = np.array([0, 30, 45, 60, 90])
radians = np.deg2rad(angles)  # 角度转弧度

# 正弦
print(np.sin(radians))
# [0.         0.5        0.70710678 0.8660254  1.        ]

# 余弦
print(np.cos(radians))
# [1.00000000e+00 8.66025404e-01 7.07106781e-01 5.00000000e-01 6.12323400e-17]

# 正切
print(np.tan(radians))

# 反三角函数
print(np.arcsin([0, 0.5, 1]))
print(np.arccos([1, 0.5, 0]))
print(np.arctan([0, 1, np.inf]))

# 弧度转角度
print(np.rad2deg(radians))  # [ 0. 30. 45. 60. 90.]
```

### 2. 指数和对数函数

```python
arr = np.array([1, 2, 3, 4, 5])

# 自然指数 e^x
print(np.exp(arr))
# [  2.71828183   7.3890561   20.08553692  54.59815003 148.4131591 ]

# 2^x
print(np.exp2(arr))
# [ 2.  4.  8. 16. 32.]

# 自然对数 ln(x)
print(np.log(arr))
# [0.         0.69314718 1.09861229 1.38629436 1.60943791]

# 以 10 为底的对数
print(np.log10(arr))
# [0.         0.30103    0.47712125 0.60205999 0.69897   ]

# 以 2 为底的对数
print(np.log2(arr))
# [0.        1.        1.5849625 2.        2.321928 ]

# log(1 + x)，对小数更精确
print(np.log1p([0.001, 0.01, 0.1]))
```

### 3. 舍入函数

```python
arr = np.array([1.2, 2.5, 3.7, 4.5, -1.5, -2.7])

# 四舍五入
print(np.round(arr))
# [ 1.  2.  4.  4. -2. -3.]

# 向下取整
print(np.floor(arr))
# [ 1.  2.  3.  4. -2. -3.]

# 向上取整
print(np.ceil(arr))
# [ 2.  3.  4.  5. -1. -2.]

# 截断（取整数部分）
print(np.trunc(arr))
# [ 1.  2.  3.  4. -1. -2.]

# 保留小数位数
print(np.around(arr, decimals=1))
# [ 1.2  2.5  3.7  4.5 -1.5 -2.7]
```

### 4. 其他数学函数

```python
arr = np.array([1, 4, 9, 16, 25])

# 平方根
print(np.sqrt(arr))
# [1. 2. 3. 4. 5.]

# 立方根
print(np.cbrt(arr))

# 平方
print(np.square(arr))
# [  1  16  81 256 625]

# 绝对值
arr_neg = np.array([-1, -2, 3, -4])
print(np.abs(arr_neg))
# [1 2 3 4]

# 符号函数
print(np.sign(arr_neg))
# [-1 -1  1 -1]

# 最大值和最小值
arr1 = np.array([1, 5, 3])
arr2 = np.array([4, 2, 6])
print(np.maximum(arr1, arr2))  # [4 5 6] - 逐元素最大值
print(np.minimum(arr1, arr2))  # [1 2 3] - 逐元素最小值
```

----

## 统计函数

### 1. 基本统计

```python
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# 均值
print(np.mean(data))        # 5.5
print(data.mean())          # 5.5

# 中位数
print(np.median(data))      # 5.5

# 标准差
print(np.std(data))         # 2.8722813232690143

# 方差
print(np.var(data))         # 8.25

# 最小值和最大值
print(np.min(data))         # 1
print(np.max(data))         # 10

# 求和
print(np.sum(data))         # 55

# 累积和
print(np.cumsum(data))
# [ 1  3  6 10 15 21 28 36 45 55]

# 累积积
print(np.cumprod(data))
# [      1       2       6      24     120     720    5040   40320  362880 3628800]
```

### 2. 沿轴统计

```python
data_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# 沿列统计（axis=0）
print(np.mean(data_2d, axis=0))  # [4. 5. 6.]
print(np.sum(data_2d, axis=0))   # [12 15 18]

# 沿行统计（axis=1）
print(np.mean(data_2d, axis=1))  # [2. 5. 8.]
print(np.sum(data_2d, axis=1))   # [ 6 15 24]

# 全局统计
print(np.mean(data_2d))          # 5.0
print(np.sum(data_2d))           # 45
```

### 3. 百分位数和分位数

```python
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# 百分位数
print(np.percentile(data, 25))   # 3.25 - 25% 分位数
print(np.percentile(data, 50))   # 5.5  - 50% 分位数（中位数）
print(np.percentile(data, 75))   # 7.75 - 75% 分位数

# 多个百分位数
print(np.percentile(data, [25, 50, 75]))
# [3.25 5.5  7.75]

# 分位数（0-1 之间）
print(np.quantile(data, [0.25, 0.5, 0.75]))
# [3.25 5.5  7.75]
```

### 4. 查找索引

```python
arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])

# 最小值索引
print(np.argmin(arr))       # 1

# 最大值索引
print(np.argmax(arr))       # 5

# 二维数组
arr_2d = np.array([[1, 5, 3], [8, 2, 9]])
print(np.argmin(arr_2d, axis=0))  # [0 1 0] - 每列最小值的行索引
print(np.argmax(arr_2d, axis=1))  # [1 2]   - 每行最大值的列索引
```

### 5. 相关性和协方差

```python
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

# 相关系数矩阵
correlation = np.corrcoef(x, y)
print(correlation)
# [[1.         0.83205029]
#  [0.83205029 1.        ]]

# 协方差矩阵
covariance = np.cov(x, y)
print(covariance)
# [[2.5 2.5]
#  [2.5 2.5]]
```

### 6. NaN 处理函数

```python
data_with_nan = np.array([1, 2, np.nan, 4, 5])

# 忽略 NaN 的统计
print(np.nanmean(data_with_nan))   # 3.0
print(np.nanstd(data_with_nan))    # 1.5811388300841898
print(np.nansum(data_with_nan))    # 12.0
print(np.nanmin(data_with_nan))    # 1.0
print(np.nanmax(data_with_nan))    # 5.0

# 检查 NaN
print(np.isnan(data_with_nan))
# [False False  True False False]
```

----

## 线性代数

### 1. 矩阵乘法

```python
# 点积（内积）
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(np.dot(a, b))     # 32 (1*4 + 2*5 + 3*6)

# 矩阵乘法
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(np.dot(A, B))
# 1*5 + 2*7  1*6 + 2*8
# 3*5+4*7  3*6+4*8
# [[19 22]
#  [43 50]]

# 使用 @ 运算符（Python 3.5+）
print(A @ B)
# [[19 22]
#  [43 50]]

# matmul 函数
print(np.matmul(A, B))
```

### 2. 矩阵转置

```python
A = np.array([[1, 2, 3], [4, 5, 6]])

# 转置
print(A.T)
# [[1 4]
#  [2 5]
#  [3 6]]

# transpose 函数
print(np.transpose(A))
```

### 3. 行列式和逆矩阵

```python
A = np.array([[1, 2], [3, 4]])

# 行列式
det = np.linalg.det(A)
print(det)  # -2.0

# 逆矩阵
inv = np.linalg.inv(A)
print(inv)
# [[-2.   1. ]
#  [ 1.5 -0.5]]

# 验证：A * A^-1 = I
print(A @ inv)
# [[1. 0.]
#  [0. 1.]]
```

### 4. 特征值和特征向量

```python
A = np.array([[1, 2], [2, 1]])

# 计算特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(A)

print("特征值:", eigenvalues)
# [ 3. -1.]

print("特征向量:\n", eigenvectors)
# [[ 0.70710678 -0.70710678]
#  [ 0.70710678  0.70710678]]
```

### 5. 矩阵分解

```python
A = np.array([[1, 2], [3, 4], [5, 6]])

# SVD 分解（奇异值分解）
U, s, Vt = np.linalg.svd(A)
print("U:\n", U)
print("奇异值:", s)
print("V^T:\n", Vt)

# QR 分解
A = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
Q, R = np.linalg.qr(A)
print("Q:\n", Q)
print("R:\n", R)

# Cholesky 分解（正定矩阵）
A = np.array([[4, 2], [2, 3]], dtype=float)
L = np.linalg.cholesky(A)
print("L:\n", L)
```

### 6. 求解线性方程组

```python
# 求解 Ax = b
A = np.array([[3, 1], [1, 2]])
b = np.array([9, 8])

x = np.linalg.solve(A, b)
print(x)  # [2. 3.]

# 验证
print(A @ x)  # [9. 8.]
```

### 7. 矩阵的秩和迹

```python
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# 矩阵的秩
rank = np.linalg.matrix_rank(A)
print(rank)  # 2

# 矩阵的迹（对角线元素之和）
trace = np.trace(A)
print(trace)  # 15 (1 + 5 + 9)
```

----

## 数组操作

### 1. 改变形状

```python
arr = np.arange(12)
print(arr)  # [ 0  1  2  3  4  5  6  7  8  9 10 11]

# reshape - 改变形状（返回新数组）
arr_2d = arr.reshape(3, 4)
print(arr_2d)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]

# reshape 为三维
arr_3d = arr.reshape(2, 3, 2)
print(arr_3d.shape)  # (2, 3, 2)

# -1 自动计算维度
arr_2d = arr.reshape(3, -1)  # 自动计算列数
print(arr_2d.shape)  # (3, 4)

# resize - 原地修改
arr.resize(3, 4)
print(arr)

# flatten - 展平为一维（返回副本）
arr_flat = arr_2d.flatten()
print(arr_flat)  # [ 0  1  2  3  4  5  6  7  8  9 10 11]

# ravel - 展平为一维（返回视图）
arr_ravel = arr_2d.ravel()
print(arr_ravel)
```

### 2. 数组拼接

```python
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])

# 垂直拼接（沿行）
v_concat = np.vstack((arr1, arr2))
print(v_concat)
# [[1 2]
#  [3 4]
#  [5 6]
#  [7 8]]

# 水平拼接（沿列）
h_concat = np.hstack((arr1, arr2))
print(h_concat)
# [[1 2 5 6]
#  [3 4 7 8]]

# concatenate - 通用拼接
concat_0 = np.concatenate((arr1, arr2), axis=0)  # 等同于 vstack
concat_1 = np.concatenate((arr1, arr2), axis=1)  # 等同于 hstack

# 一维数组拼接
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(np.concatenate((a, b)))  # [1 2 3 4 5 6]
```

### 3. 数组分割

```python
arr = np.arange(12).reshape(4, 3)

# 垂直分割（按行）
v_split = np.vsplit(arr, 2)  # 分成 2 部分
print(v_split[0])
# [[0 1 2]
#  [3 4 5]]

# 水平分割（按列）
h_split = np.hsplit(arr, 3)  # 分成 3 部分
print(h_split[0])
# [[ 0]
#  [ 3]
#  [ 6]
#  [ 9]]

# split - 通用分割
split_0 = np.split(arr, 2, axis=0)  # 沿行分割
split_1 = np.split(arr, 3, axis=1)  # 沿列分割

# 不等分分割
arr = np.arange(10)
parts = np.split(arr, [3, 7])  # 在索引 3 和 7 处分割
print(parts)  # [array([0, 1, 2]), array([3, 4, 5, 6]), array([7, 8, 9])]
```

### 4. 数组重复

```python
arr = np.array([1, 2, 3])

# repeat - 重复元素
print(np.repeat(arr, 3))
# [1 1 1 2 2 2 3 3 3]

# 指定每个元素重复次数
print(np.repeat(arr, [2, 3, 1]))
# [1 1 2 2 2 3]

# tile - 重复整个数组
print(np.tile(arr, 2))
# [1 2 3 1 2 3]

# 二维重复
print(np.tile(arr, (2, 3)))
# [[1 2 3 1 2 3 1 2 3]
#  [1 2 3 1 2 3 1 2 3]]
```

### 5. 数组排序

```python
arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])

# sort - 返回排序后的副本
sorted_arr = np.sort(arr)
print(sorted_arr)  # [1 1 2 3 4 5 6 9]

# 原地排序
arr.sort()
print(arr)  # [1 1 2 3 4 5 6 9]

# 降序排序
arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])
print(np.sort(arr)[::-1])  # [9 6 5 4 3 2 1 1]

# 二维数组排序
arr_2d = np.array([[3, 1, 4], [1, 5, 9]])
print(np.sort(arr_2d, axis=0))  # 沿列排序
print(np.sort(arr_2d, axis=1))  # 沿行排序

# argsort - 返回排序后的索引
arr = np.array([3, 1, 4, 1, 5])
indices = np.argsort(arr)
print(indices)  # [1 3 0 2 4]
print(arr[indices])  # [1 1 3 4 5]
```

### 6. 唯一值和去重

```python
arr = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])

# 唯一值
unique = np.unique(arr)
print(unique)  # [1 2 3 4]

# 返回索引和计数
unique, indices, counts = np.unique(arr, return_index=True, return_counts=True)
print("唯一值:", unique)      # [1 2 3 4]
print("首次出现索引:", indices)  # [0 1 3 6]
print("出现次数:", counts)    # [1 2 3 4]
```

----

## 广播机制

广播（Broadcasting）是 NumPy 中强大的机制，允许不同形状的数组进行运算。

### 广播规则

1. 如果两个数组维度数不同，较小维度的数组会在前面补 1
2. 如果两个数组在某个维度上大小不同，且其中一个为 1，则该维度会被"拉伸"
3. 如果两个数组在某个维度上大小不同，且都不为 1，则报错

### 示例

```python
# 示例 1：标量与数组
arr = np.array([1, 2, 3, 4])
print(arr + 10)  # [11 12 13 14]

# 示例 2：一维与二维
arr_1d = np.array([1, 2, 3])
arr_2d = np.array([[10], [20], [30]])

result = arr_1d + arr_2d
print(result)
# arr_1d:  (3,)    →  (1, 3)   # 在前面补 1
# arr_2d:  (3, 1)  →  (3, 1)   # 保持不变
# arr_1d: (1, 3) → (3, 3)   # 沿第 0 维复制 3 次
#         [[1, 2, 3]]  →  [[1, 2, 3],
#                          [1, 2, 3],
#                          [1, 2, 3]]
# arr_2d: (3, 1) → (3, 3)   # 沿第 1 维复制 3 次
#         [[10],           [[10, 10, 10],
#          [20],     →      [20, 20, 20],
#          [30]]            [30, 30, 30]]
# [[11 12 13]
#  [21 22 23]
#  [31 32 33]]

# 示例 3：形状 (3, 1) 和 (1, 4)
a = np.array([[1], [2], [3]])      # (3, 1)
b = np.array([[10, 20, 30, 40]])   # (1, 4)

result = a + b
print(result)
# [[11 21 31 41]
#  [12 22 32 42]
#  [13 23 33 43]]

# 示例 4：实际应用 - 标准化
data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
mean = data.mean(axis=0)  # [4. 5. 6.]
std = data.std(axis=0)    # [2.44948974 2.44948974 2.44948974]

normalized = (data - mean) / std
print(normalized)
```

----

## 实战案例

### 案例 1：图像处理

```python
# 模拟一个 RGB 图像（100x100x3）
image = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)

# 转换为灰度图
gray = np.mean(image, axis=2).astype(np.uint8)
print(gray.shape)  # (100, 100)

# 图像翻转
flipped_h = image[:, ::-1, :]  # 水平翻转
flipped_v = image[::-1, :, :]  # 垂直翻转

# 图像裁剪
cropped = image[25:75, 25:75, :]  # 裁剪中心 50x50 区域

# 亮度调整
brightened = np.clip(image * 1.5, 0, 255).astype(np.uint8)
# np.clip() 用于将数组中的值限制在指定范围内，超出范围的值会被裁剪到边界值。
# .astype(np.uint8) 转换类型到 int 型
```

### 案例 2：数据分析

```python
# 模拟学生成绩数据
np.random.seed(42)
scores = np.random.randint(60, 100, size=(100, 5))  # 100 个学生，5 门课

# 每个学生的平均分
student_avg = scores.mean(axis=1)
print(f"平均分最高的学生: {np.argmax(student_avg)}")

# 每门课的平均分
course_avg = scores.mean(axis=0)
print(f"各科平均分: {course_avg}")

# 及格率（>= 60 分）
pass_rate = (scores >= 60).sum(axis=0) / len(scores) * 100
print(f"各科及格率: {pass_rate}%")

# 找出总分前 10 名
total_scores = scores.sum(axis=1)
top_10_indices = np.argsort(total_scores)[-10:][::-1]
print(f"前 10 名学生索引: {top_10_indices}")
```

### 案例 3：时间序列分析

```python
# 模拟股票价格数据
np.random.seed(42)
days = 100
prices = 100 + np.cumsum(np.random.randn(days) * 2)

# 计算移动平均（5 日均线）
window = 5
moving_avg = np.convolve(prices, np.ones(window)/window, mode='valid')

# 计算收益率
returns = np.diff(prices) / prices[:-1] * 100

# 找出最大涨幅和跌幅
max_gain_idx = np.argmax(returns)
max_loss_idx = np.argmin(returns)

print(f"最大涨幅: {returns[max_gain_idx]:.2f}% (第 {max_gain_idx+1} 天)")
print(f"最大跌幅: {returns[max_loss_idx]:.2f}% (第 {max_loss_idx+1} 天)")

# 波动率（标准差）
volatility = np.std(returns)
print(f"波动率: {volatility:.2f}%")
```

### 案例 4：矩阵运算 - 线性回归

```python
# 简单线性回归：y = ax + b
np.random.seed(42)
x = np.linspace(0, 10, 50)
y = 2 * x + 1 + np.random.randn(50) * 2  # y = 2x + 1 + 噪声

# 构造设计矩阵
X = np.vstack([x, np.ones(len(x))]).T  # [[x1, 1], [x2, 1], ...]

# 最小二乘法求解：(X^T X)^-1 X^T y
coeffs = np.linalg.inv(X.T @ X) @ X.T @ y
a, b = coeffs

print(f"拟合结果: y = {a:.2f}x + {b:.2f}")

# 预测
y_pred = a * x + b

# 计算 R²
ss_res = np.sum((y - y_pred) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r_squared = 1 - (ss_res / ss_tot)
print(f"R² = {r_squared:.4f}")
```

### 案例 5：数据清洗

```python
# 模拟带缺失值的数据
data = np.array([[1, 2, np.nan], 
                 [4, np.nan, 6], 
                 [7, 8, 9],
                 [np.nan, 11, 12]])

# 检测缺失值
print("缺失值位置:")
print(np.isnan(data))

# 删除含缺失值的行
clean_data = data[~np.isnan(data).any(axis=1)]
print("删除缺失行后:")
print(clean_data)

# 用均值填充缺失值
col_means = np.nanmean(data, axis=0)
indices = np.where(np.isnan(data))
data_filled = data.copy()
for i, j in zip(*indices):
    data_filled[i, j] = col_means[j]

print("填充后:")
print(data_filled)
```

----

## 总结

### NumPy 核心优势

1. **性能卓越**：底层 C 实现，比纯 Python 快 10-100 倍
2. **内存高效**：连续内存存储，占用空间小
3. **语法简洁**：向量化操作，无需显式循环
4. **功能丰富**：涵盖数学、统计、线性代数等
5. **生态完善**：Pandas、SciPy、Scikit-learn 等都基于 NumPy

### 常用函数速查表

| 类别 | 函数 | 说明 |
|------|------|------|
| **创建** | `array`, `zeros`, `ones`, `arange`, `linspace` | 数组创建 |
| **形状** | `reshape`, `flatten`, `ravel`, `transpose` | 形状操作 |
| **索引** | `[]`, `boolean indexing`, `fancy indexing` | 数组索引 |
| **运算** | `+`, `-`, `*`, `/`, `@`, `**` | 算术运算 |
| **统计** | `mean`, `median`, `std`, `var`, `sum` | 统计函数 |
| **数学** | `sin`, `cos`, `exp`, `log`, `sqrt` | 数学函数 |
| **线性代数** | `dot`, `inv`, `det`, `eig`, `svd` | 矩阵运算 |
| **拼接** | `vstack`, `hstack`, `concatenate` | 数组拼接 |
| **排序** | `sort`, `argsort`, `unique` | 排序去重 |

### 学习建议

1. **掌握基础**：数组创建、索引、切片
2. **理解广播**：不同形状数组的运算规则
3. **熟悉向量化**：避免使用 Python 循环
4. **实践应用**：在实际项目中使用 NumPy
5. **深入学习**：线性代数、统计分析等高级功能

### 相关资源

- [NumPy 官方文档](https://numpy.org/doc/)
- [NumPy 用户指南](https://numpy.org/doc/stable/user/index.html)
- [NumPy API 参考](https://numpy.org/doc/stable/reference/index.html)

----

**Happy Coding with NumPy! 🚀**
