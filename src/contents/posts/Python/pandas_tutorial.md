---
title: Pandas 完整教程 - Python 科学计算基础
published: 2026-02-24
description: 详细介绍 Pandas 的核心概念、API 使用方法和实战示例
tags: [Python, NumPy, 数据分析, 科学计算]
category: Python
licenseName: "Unlicensed"
author: panxiao
sourceLink: "https://github.com/px6707/myblog"
draft: false
---

# Pandas 完整教程

## 目录
- [简介](#简介)
- [安装](#安装)
- [数据结构](#数据结构)
  - [Series](#series)
  - [DataFrame](#dataframe)
- [数据读取与写入](#数据读取与写入)
- [数据查看与选择](#数据查看与选择)
- [数据清洗](#数据清洗)
- [数据转换](#数据转换)
- [数据聚合与分组](#数据聚合与分组)
- [数据合并](#数据合并)
- [时间序列](#时间序列)
- [数据可视化](#数据可视化)

---

## 简介

Pandas 是 Python 中最强大的数据分析库之一，提供了高效的数据结构和数据分析工具。它特别适合处理表格数据、时间序列数据等。

---

## 安装

```bash
pip install pandas
```

---

## 数据结构

### Series

**Series** 是一维标签数组，可以存储任何数据类型。

#### 创建 Series

##### `pd.Series()`
创建一个 Series 对象。

```python
import pandas as pd

# 从列表创建 Series
s = pd.Series([1, 3, 5, 7, 9])
print(s)
# 输出:
# 0    1
# 1    3
# 2    5
# 3    7
# 4    9
# dtype: int64

# 从字典创建 Series，字典的键会成为索引
s = pd.Series({'a': 1, 'b': 2, 'c': 3})
print(s)
# 输出:
# a    1
# b    2
# c    3
# dtype: int64

# 指定索引
s = pd.Series([10, 20, 30], index=['x', 'y', 'z'])
print(s)
# 输出:
# x    10
# y    20
# z    30
# dtype: int64
```

#### Series 属性

##### `.index`
获取或设置 Series 的索引。

```python
s = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
print(s.index)  # 输出: Index(['a', 'b', 'c'], dtype='object')

# 修改索引
s.index = ['x', 'y', 'z']
print(s)
# 输出:
# x    1
# y    2
# z    3
# dtype: int64
```

##### `.values`
获取 Series 的值（NumPy 数组）。

```python
s = pd.Series([1, 2, 3])
print(s.values)  # 输出: [1 2 3]
print(type(s.values))  # 输出: <class 'numpy.ndarray'>
```

##### `.dtype`
获取 Series 的数据类型。

```python
s = pd.Series([1, 2, 3])
print(s.dtype)  # 输出: int64

s = pd.Series(['a', 'b', 'c'])
print(s.dtype)  # 输出: object
```

##### `.shape`
获取 Series 的形状。

```python
s = pd.Series([1, 2, 3, 4, 5])
print(s.shape)  # 输出: (5,)
```

##### `.size`
获取 Series 中元素的个数。

```python
s = pd.Series([1, 2, 3, 4, 5])
print(s.size)  # 输出: 5
```

---

### DataFrame

**DataFrame** 是二维标签数据结构，类似于电子表格或 SQL 表。

#### 创建 DataFrame

##### `pd.DataFrame()`
创建一个 DataFrame 对象。

```python
import pandas as pd

# 从字典创建 DataFrame
data = {
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'city': ['New York', 'Paris', 'London']
}
df = pd.DataFrame(data)
print(df)
# 输出:
#       name  age      city
# 0    Alice   25  New York
# 1      Bob   30     Paris
# 2  Charlie   35    London

# 从列表的列表创建 DataFrame
data = [
    ['Alice', 25, 'New York'],
    ['Bob', 30, 'Paris'],
    ['Charlie', 35, 'London']
]
df = pd.DataFrame(data, columns=['name', 'age', 'city'])
print(df)

# 指定索引
df = pd.DataFrame(data, columns=['name', 'age', 'city'], index=['a', 'b', 'c'])
print(df)
# 输出:
#       name  age      city
# a    Alice   25  New York
# b      Bob   30     Paris
# c  Charlie   35    London
```

#### DataFrame 属性

##### `.index`
获取或设置 DataFrame 的行索引。

```python
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
print(df.index)  # 输出: RangeIndex(start=0, stop=3, step=1)

# 设置索引
df.index = ['x', 'y', 'z']
print(df)
# 输出:
#    A  B
# x  1  4
# y  2  5
# z  3  6
```

##### `.columns`
获取或设置 DataFrame 的列名。

```python
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
print(df.columns)  # 输出: Index(['A', 'B'], dtype='object')

# 重命名列
df.columns = ['col1', 'col2']
print(df)
# 输出:
#    col1  col2
# 0     1     4
# 1     2     5
# 2     3     6
```

##### `.shape`
获取 DataFrame 的形状（行数，列数）。

```python
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
print(df.shape)  # 输出: (3, 2) - 3行2列
```

##### `.size`
获取 DataFrame 中元素的总数。

```python
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
print(df.size)  # 输出: 6 (3行 × 2列)
```

##### `.dtypes`
获取每列的数据类型。

```python
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [1.5, 2.5, 3.5],
    'C': ['a', 'b', 'c']
})
print(df.dtypes)
# 输出:
# A      int64
# B    float64
# C     object
# dtype: object
```

---

## 数据读取与写入

### 读取数据

##### `pd.read_csv()`
从 CSV 文件读取数据。

```python
# 读取 CSV 文件
df = pd.read_csv('data.csv')

# 指定分隔符
df = pd.read_csv('data.txt', sep='\t')

# 指定编码
df = pd.read_csv('data.csv', encoding='utf-8')

# 跳过前几行
df = pd.read_csv('data.csv', skiprows=2)

# 只读取指定列
df = pd.read_csv('data.csv', usecols=['name', 'age'])

# 指定索引列
df = pd.read_csv('data.csv', index_col='id')
```

##### `pd.read_excel()`
从 Excel 文件读取数据。

```python
# 读取 Excel 文件
df = pd.read_excel('data.xlsx')

# 读取指定的工作表
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')

# 读取多个工作表
dfs = pd.read_excel('data.xlsx', sheet_name=['Sheet1', 'Sheet2'])
# dfs 是一个字典，键是工作表名，值是 DataFrame
```

##### `pd.read_json()`
从 JSON 文件读取数据。

```python
# 读取 JSON 文件
df = pd.read_json('data.json')

# 指定 JSON 的方向
df = pd.read_json('data.json', orient='records')
# orient 可以是: 'split', 'records', 'index', 'columns', 'values'
```

##### `pd.read_sql()`
从 SQL 数据库读取数据。

```python
import sqlite3

# 创建数据库连接
conn = sqlite3.connect('database.db')

# 读取整个表
df = pd.read_sql('SELECT * FROM users', conn)

# 使用表名
df = pd.read_sql_table('users', conn)

# 执行查询
df = pd.read_sql_query('SELECT name, age FROM users WHERE age > 25', conn)

# 关闭连接
conn.close()
```

### 写入数据

##### `.to_csv()`
将 DataFrame 写入 CSV 文件。

```python
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

# 写入 CSV 文件
df.to_csv('output.csv')

# 不写入索引
df.to_csv('output.csv', index=False)

# 指定分隔符
df.to_csv('output.txt', sep='\t')

# 指定编码
df.to_csv('output.csv', encoding='utf-8')

# 只写入指定列
df.to_csv('output.csv', columns=['A'])
```

##### `.to_excel()`
将 DataFrame 写入 Excel 文件。

```python
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

# 写入 Excel 文件
df.to_excel('output.xlsx', index=False)

# 指定工作表名
df.to_excel('output.xlsx', sheet_name='MySheet')

# 写入多个工作表
with pd.ExcelWriter('output.xlsx') as writer:
    df1.to_excel(writer, sheet_name='Sheet1')
    df2.to_excel(writer, sheet_name='Sheet2')
```

##### `.to_json()`
将 DataFrame 写入 JSON 文件。

```python
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

# 写入 JSON 文件
df.to_json('output.json')

# 指定方向
df.to_json('output.json', orient='records')

# 格式化输出
df.to_json('output.json', orient='records', indent=4)
```

##### `.to_sql()`
将 DataFrame 写入 SQL 数据库。

```python
import sqlite3

df = pd.DataFrame({'name': ['Alice', 'Bob'], 'age': [25, 30]})

# 创建数据库连接
conn = sqlite3.connect('database.db')

# 写入数据库
df.to_sql('users', conn, if_exists='replace', index=False)
# if_exists 可以是: 'fail', 'replace', 'append'

conn.close()
```

---

## 数据查看与选择

### 查看数据

##### `.head()`
查看前 n 行数据（默认 5 行）。

```python
df = pd.DataFrame({'A': range(10), 'B': range(10, 20)})

print(df.head())  # 前5行
print(df.head(3))  # 前3行
# 输出:
#    A   B
# 0  0  10
# 1  1  11
# 2  2  12
```

##### `.tail()`
查看后 n 行数据（默认 5 行）。

```python
df = pd.DataFrame({'A': range(10), 'B': range(10, 20)})

print(df.tail())  # 后5行
print(df.tail(3))  # 后3行
# 输出:
#    A   B
# 7  7  17
# 8  8  18
# 9  9  19
```

##### `.info()`
查看 DataFrame 的基本信息。

```python
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [1.5, 2.5, 3.5],
    'C': ['a', 'b', 'c']
})

df.info()
# 输出:
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 3 entries, 0 to 2
# Data columns (total 3 columns):
#  #   Column  Non-Null Count  Dtype  
# ---  ------  --------------  -----  
#  0   A       3 non-null      int64  
#  1   B       3 non-null      float64
#  2   C       3 non-null      object 
# dtypes: float64(1), int64(1), object(1)
# memory usage: 200.0+ bytes
```

##### `.describe()`
查看数值列的统计摘要。

```python
df = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': [10, 20, 30, 40, 50]
})

print(df.describe())
# 输出:
#              A          B
# count  5.000000   5.000000
# mean   3.000000  30.000000
# std    1.581139  15.811388
# min    1.000000  10.000000
# 25%    2.000000  20.000000
# 50%    3.000000  30.000000
# 75%    4.000000  40.000000
# max    5.000000  50.000000
```

##### `.value_counts()`
统计每个值出现的次数。

```python
s = pd.Series(['a', 'b', 'a', 'c', 'b', 'a'])

print(s.value_counts())
# 输出:
# a    3
# b    2
# c    1
# dtype: int64

# 显示百分比
print(s.value_counts(normalize=True))
# 输出:
# a    0.500000
# b    0.333333
# c    0.166667
# dtype: float64
```

### 选择数据

##### 列选择
通过列名选择列。

```python
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'city': ['New York', 'Paris', 'London']
})

# 选择单列（返回 Series）
print(df['name'])
# 输出:
# 0      Alice
# 1        Bob
# 2    Charlie
# Name: name, dtype: object

# 选择多列（返回 DataFrame）
print(df[['name', 'age']])
# 输出:
#       name  age
# 0    Alice   25
# 1      Bob   30
# 2  Charlie   35
```

##### `.loc[]`
通过标签选择数据。

```python
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6],
    'C': [7, 8, 9]
}, index=['x', 'y', 'z'])

# 选择单行
print(df.loc['x'])
# 输出:
# A    1
# B    4
# C    7
# Name: x, dtype: int64

# 选择多行
print(df.loc[['x', 'z']])
# 输出:
#    A  B  C
# x  1  4  7
# z  3  6  9

# 选择行和列
print(df.loc['x', 'A'])  # 输出: 1
print(df.loc[['x', 'y'], ['A', 'C']])
# 输出:
#    A  C
# x  1  7
# y  2  8

# 使用切片
print(df.loc['x':'y', 'A':'B'])
# 输出:
#    A  B
# x  1  4
# y  2  5
```

##### `.iloc[]`
通过位置索引选择数据。

```python
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6],
    'C': [7, 8, 9]
})

# 选择单行
print(df.iloc[0])
# 输出:
# A    1
# B    4
# C    7
# Name: 0, dtype: int64

# 选择多行
print(df.iloc[[0, 2]])
# 输出:
#    A  B  C
# 0  1  4  7
# 2  3  6  9

# 选择行和列
print(df.iloc[0, 1])  # 输出: 4
print(df.iloc[[0, 1], [0, 2]])
# 输出:
#    A  C
# 0  1  7
# 1  2  8

# 使用切片
print(df.iloc[0:2, 0:2])
# 输出:
#    A  B
# 0  1  4
# 1  2  5
```

##### `.at[]` 和 `.iat[]`
快速访问单个值。

```python
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]}, index=['x', 'y', 'z'])

# 使用标签访问
print(df.at['x', 'A'])  # 输出: 1

# 使用位置访问
print(df.iat[0, 0])  # 输出: 1
```

##### 布尔索引
使用条件选择数据。

```python
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'David'],
    'age': [25, 30, 35, 28],
    'city': ['New York', 'Paris', 'London', 'Tokyo']
})

# 单个条件
print(df[df['age'] > 28])
# 输出:
#       name  age    city
# 1      Bob   30   Paris
# 2  Charlie   35  London

# 多个条件（使用 & 和 |）
print(df[(df['age'] > 25) & (df['city'] == 'Paris')])
# 输出:
#   name  age   city
# 1  Bob   30  Paris

# 使用 isin()
print(df[df['city'].isin(['Paris', 'Tokyo'])])
# 输出:
#     name  age   city
# 1    Bob   30  Paris
# 3  David   28  Tokyo
```

---

## 数据清洗

### 处理缺失值

##### `.isna()` / `.isnull()`
检测缺失值。

```python
import numpy as np

df = pd.DataFrame({
    'A': [1, 2, np.nan, 4],
    'B': [5, np.nan, np.nan, 8],
    'C': [9, 10, 11, 12]
})

# 检测缺失值
print(df.isna())
# 输出:
#        A      B      C
# 0  False  False  False
# 1  False   True  False
# 2   True   True  False
# 3  False  False  False

# 统计每列的缺失值数量
print(df.isna().sum())
# 输出:
# A    1
# B    2
# C    0
# dtype: int64
```

##### `.notna()` / `.notnull()`
检测非缺失值。

```python
df = pd.DataFrame({
    'A': [1, 2, np.nan, 4],
    'B': [5, np.nan, np.nan, 8]
})

print(df.notna())
# 输出:
#        A      B
# 0   True   True
# 1   True  False
# 2  False  False
# 3   True   True
```

##### `.dropna()`
删除包含缺失值的行或列。

```python
df = pd.DataFrame({
    'A': [1, 2, np.nan, 4],
    'B': [5, np.nan, np.nan, 8],
    'C': [9, 10, 11, 12]
})

# 删除包含任何缺失值的行
print(df.dropna())
# 输出:
#      A    B   C
# 0  1.0  5.0   9
# 3  4.0  8.0  12

# 删除所有值都是缺失的行
print(df.dropna(how='all'))

# 删除包含缺失值的列
print(df.dropna(axis=1))
# 输出:
#     C
# 0   9
# 1  10
# 2  11
# 3  12

# 保留至少有 2 个非缺失值的行
print(df.dropna(thresh=2))
# 输出:
#      A    B   C
# 0  1.0  5.0   9
# 1  2.0  NaN  10
# 3  4.0  8.0  12
```

##### `.fillna()`
填充缺失值。

```python
df = pd.DataFrame({
    'A': [1, 2, np.nan, 4],
    'B': [5, np.nan, np.nan, 8]
})

# 用指定值填充
print(df.fillna(0))
# 输出:
#      A    B
# 0  1.0  5.0
# 1  2.0  0.0
# 2  0.0  0.0
# 3  4.0  8.0

# 用前一个值填充（前向填充）
print(df.fillna(method='ffill'))
# 输出:
#      A    B
# 0  1.0  5.0
# 1  2.0  5.0
# 2  2.0  5.0
# 3  4.0  8.0

# 用后一个值填充（后向填充）
print(df.fillna(method='bfill'))

# 用均值填充
print(df.fillna(df.mean()))
# 输出:
#           A         B
# 0  1.000000  5.000000
# 1  2.000000  6.500000
# 2  2.333333  6.500000
# 3  4.000000  8.000000
```

### 处理重复值

##### `.duplicated()`
检测重复行。

```python
df = pd.DataFrame({
    'A': [1, 2, 2, 3],
    'B': [4, 5, 5, 6]
})

# 检测重复行
print(df.duplicated())
# 输出:
# 0    False
# 1    False
# 2     True  # 第2行是重复的
# 3    False
# dtype: bool

# 基于指定列检测重复
print(df.duplicated(subset=['A']))
# 输出:
# 0    False
# 1    False
# 2     True
# 3    False
# dtype: bool
```

##### `.drop_duplicates()`
删除重复行。

```python
df = pd.DataFrame({
    'A': [1, 2, 2, 3],
    'B': [4, 5, 5, 6]
})

# 删除重复行
print(df.drop_duplicates())
# 输出:
#    A  B
# 0  1  4
# 1  2  5
# 3  3  6

# 基于指定列删除重复
print(df.drop_duplicates(subset=['A']))

# 保留最后一个重复项
print(df.drop_duplicates(keep='last'))
# 输出:
#    A  B
# 0  1  4
# 2  2  5
# 3  3  6
```

### 数据类型转换

##### `.astype()`
转换数据类型。

```python
df = pd.DataFrame({
    'A': ['1', '2', '3'],
    'B': ['4.5', '5.5', '6.5']
})

# 转换为整数
df['A'] = df['A'].astype(int)
print(df['A'].dtype)  # 输出: int64

# 转换为浮点数
df['B'] = df['B'].astype(float)
print(df['B'].dtype)  # 输出: float64

# 一次转换多列
df = df.astype({'A': int, 'B': float})
```

##### `pd.to_numeric()`
将数据转换为数值类型。

```python
s = pd.Series(['1', '2', '3', 'four'])

# 转换为数值，无法转换的设为 NaN
result = pd.to_numeric(s, errors='coerce')
print(result)
# 输出:
# 0    1.0
# 1    2.0
# 2    3.0
# 3    NaN
# dtype: float64

# 忽略无法转换的值
result = pd.to_numeric(s, errors='ignore')
print(result)
# 输出: 原始 Series 不变
```

##### `pd.to_datetime()`
将数据转换为日期时间类型。

```python
s = pd.Series(['2023-01-01', '2023-02-15', '2023-03-20'])

# 转换为日期时间
result = pd.to_datetime(s)
print(result)
# 输出:
# 0   2023-01-01
# 1   2023-02-15
# 2   2023-03-20
# dtype: datetime64[ns]

# 指定日期格式
s = pd.Series(['01/01/2023', '02/15/2023'])
result = pd.to_datetime(s, format='%m/%d/%Y')
```

---

## 数据转换

### 排序

##### `.sort_values()`
按值排序。

```python
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 20],
    'score': [85, 90, 95]
})

# 按单列排序
print(df.sort_values('age'))
# 输出:
#       name  age  score
# 2  Charlie   20     95
# 0    Alice   25     85
# 1      Bob   30     90

# 降序排序
print(df.sort_values('age', ascending=False))

# 按多列排序
df2 = pd.DataFrame({
    'A': [1, 2, 1, 2],
    'B': [4, 3, 2, 1]
})
print(df2.sort_values(['A', 'B']))
# 输出:
#    A  B
# 2  1  2
# 0  1  4
# 3  2  1
# 1  2  3
```

##### `.sort_index()`
按索引排序。

```python
df = pd.DataFrame({
    'A': [1, 2, 3]
}, index=['c', 'a', 'b'])

print(df.sort_index())
# 输出:
#    A
# a  2
# b  3
# c  1

# 降序排序
print(df.sort_index(ascending=False))
```

### 重命名

##### `.rename()`
重命名行或列。

```python
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6]
})

# 重命名列
print(df.rename(columns={'A': 'col1', 'B': 'col2'}))
# 输出:
#    col1  col2
# 0     1     4
# 1     2     5
# 2     3     6

# 重命名索引
df.index = ['x', 'y', 'z']
print(df.rename(index={'x': 'row1', 'y': 'row2'}))
# 输出:
#       A  B
# row1  1  4
# row2  2  5
# z     3  6

# 使用函数重命名
print(df.rename(columns=str.lower))
```

### 应用函数

##### `.apply()`
对行或列应用函数。

```python
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6]
})

# 对列应用函数
print(df.apply(lambda x: x * 2))
# 输出:
#    A   B
# 0  2   8
# 1  4  10
# 2  6  12

# 对行应用函数
print(df.apply(lambda x: x.sum(), axis=1))
# 输出:
# 0    5
# 1    7
# 2    9
# dtype: int64

# 应用自定义函数
def custom_func(x):
    return x.max() - x.min()

print(df.apply(custom_func))
# 输出:
# A    2
# B    2
# dtype: int64
```

##### `.applymap()` / `.map()`
对每个元素应用函数。

```python
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6]
})

# applymap 对 DataFrame 的每个元素应用函数
print(df.applymap(lambda x: x ** 2))
# 输出:
#    A   B
# 0  1  16
# 1  4  25
# 2  9  36

# map 对 Series 的每个元素应用函数
s = pd.Series([1, 2, 3])
print(s.map(lambda x: x * 10))
# 输出:
# 0    10
# 1    20
# 2    30
# dtype: int64

# 使用字典映射
s = pd.Series(['a', 'b', 'c'])
mapping = {'a': 1, 'b': 2, 'c': 3}
print(s.map(mapping))
# 输出:
# 0    1
# 1    2
# 2    3
# dtype: int64
```

##### `.replace()`
替换值。

```python
df = pd.DataFrame({
    'A': [1, 2, 3, 1],
    'B': [4, 5, 6, 4]
})

# 替换单个值
print(df.replace(1, 100))
# 输出:
#      A  B
# 0  100  4
# 1    2  5
# 2    3  6
# 3  100  4

# 替换多个值
print(df.replace([1, 2], [100, 200]))
# 输出:
#      A  B
# 0  100  4
# 1  200  5
# 2    3  6
# 3  100  4

# 使用字典替换
print(df.replace({'A': {1: 100, 2: 200}}))
# 输出:
#      A  B
# 0  100  4
# 1  200  5
# 2    3  6
# 3  100  4
```

### 字符串操作

##### `.str` 访问器
对字符串列进行操作。

```python
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie']
})

# 转换为小写
print(df['name'].str.lower())
# 输出:
# 0      alice
# 1        bob
# 2    charlie
# Name: name, dtype: object

# 转换为大写
print(df['name'].str.upper())

# 获取字符串长度
print(df['name'].str.len())
# 输出:
# 0    5
# 1    3
# 2    7
# Name: name, dtype: int64

# 检查是否包含子串
print(df['name'].str.contains('li'))
# 输出:
# 0     True
# 1    False
# 2     True
# Name: name, dtype: bool

# 替换子串
print(df['name'].str.replace('li', 'LI'))
# 输出:
# 0      ALIce
# 1        Bob
# 2    CharLIe
# Name: name, dtype: object

# 分割字符串
s = pd.Series(['a-b-c', 'd-e-f'])
print(s.str.split('-'))
# 输出:
# 0    [a, b, c]
# 1    [d, e, f]
# dtype: object

# 提取子串
print(df['name'].str[0:3])
# 输出:
# 0    Ali
# 1    Bob
# 2    Cha
# Name: name, dtype: object
```

---

## 数据聚合与分组

### 基本统计

##### `.sum()`
求和。

```python
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6]
})

# 列求和
print(df.sum())
# 输出:
# A     6
# B    15
# dtype: int64

# 行求和
print(df.sum(axis=1))
# 输出:
# 0    5
# 1    7
# 2    9
# dtype: int64
```

##### `.mean()`
求平均值。

```python
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6]
})

print(df.mean())
# 输出:
# A    2.0
# B    5.0
# dtype: float64
```

##### `.median()`
求中位数。

```python
df = pd.DataFrame({
    'A': [1, 2, 3, 4, 5]
})

print(df.median())
# 输出:
# A    3.0
# dtype: float64
```

##### `.std()` 和 `.var()`
求标准差和方差。

```python
df = pd.DataFrame({
    'A': [1, 2, 3, 4, 5]
})

print(df.std())   # 标准差
print(df.var())   # 方差
```

##### `.min()` 和 `.max()`
求最小值和最大值。

```python
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6]
})

print(df.min())
# 输出:
# A    1
# B    4
# dtype: int64

print(df.max())
# 输出:
# A    3
# B    6
# dtype: int64
```

##### `.count()`
统计非缺失值的数量。

```python
df = pd.DataFrame({
    'A': [1, 2, np.nan],
    'B': [4, 5, 6]
})

print(df.count())
# 输出:
# A    2
# B    3
# dtype: int64
```

### 分组操作

##### `.groupby()`
按指定列分组。

```python
df = pd.DataFrame({
    'category': ['A', 'B', 'A', 'B', 'A'],
    'value': [10, 20, 30, 40, 50]
})

# 按 category 分组并求和
print(df.groupby('category').sum())
# 输出:
#           value
# category       
# A            90
# B            60

# 按 category 分组并求平均值
print(df.groupby('category').mean())
# 输出:
#           value
# category       
# A          30.0
# B          30.0

# 多列分组
df = pd.DataFrame({
    'category': ['A', 'B', 'A', 'B'],
    'type': ['X', 'X', 'Y', 'Y'],
    'value': [10, 20, 30, 40]
})
print(df.groupby(['category', 'type']).sum())
# 输出:
#                value
# category type       
# A        X       10
#          Y       30
# B        X       20
#          Y       40
```

##### `.agg()` / `.aggregate()`
对分组应用多个聚合函数。

```python
df = pd.DataFrame({
    'category': ['A', 'B', 'A', 'B', 'A'],
    'value': [10, 20, 30, 40, 50]
})

# 应用多个聚合函数
print(df.groupby('category').agg(['sum', 'mean', 'count']))
# 输出:
#          value              
#            sum  mean count
# category                   
# A           90  30.0     3
# B           60  30.0     2

# 对不同列应用不同的聚合函数
df = pd.DataFrame({
    'category': ['A', 'B', 'A', 'B'],
    'value1': [10, 20, 30, 40],
    'value2': [1, 2, 3, 4]
})
print(df.groupby('category').agg({
    'value1': 'sum',
    'value2': 'mean'
}))
# 输出:
#           value1  value2
# category                
# A             40     2.0
# B             60     3.0
```

##### `.transform()`
对分组应用函数并返回与原数据相同形状的结果。

```python
df = pd.DataFrame({
    'category': ['A', 'B', 'A', 'B', 'A'],
    'value': [10, 20, 30, 40, 50]
})

# 计算每个分组的平均值，并将结果广播到原数据
df['group_mean'] = df.groupby('category')['value'].transform('mean')
print(df)
# 输出:
#   category  value  group_mean
# 0        A     10        30.0
# 1        B     20        30.0
# 2        A     30        30.0
# 3        B     40        30.0
# 4        A     50        30.0
```

##### `.filter()`
根据分组条件过滤数据。

```python
df = pd.DataFrame({
    'category': ['A', 'B', 'A', 'B', 'A'],
    'value': [10, 20, 30, 40, 50]
})

# 只保留分组总和大于 70 的组
result = df.groupby('category').filter(lambda x: x['value'].sum() > 70)
print(result)
# 输出:
#   category  value
# 0        A     10
# 2        A     30
# 4        A     50
```

### 透视表

##### `.pivot_table()`
创建透视表。

```python
df = pd.DataFrame({
    'date': ['2023-01', '2023-01', '2023-02', '2023-02'],
    'category': ['A', 'B', 'A', 'B'],
    'value': [10, 20, 30, 40]
})

# 创建透视表
pivot = df.pivot_table(
    values='value',
    index='date',
    columns='category',
    aggfunc='sum'
)
print(pivot)
# 输出:
# category     A   B
# date              
# 2023-01   10.0  20.0
# 2023-02   30.0  40.0
```

##### `.pivot()`
重塑数据（不进行聚合）。

```python
df = pd.DataFrame({
    'date': ['2023-01', '2023-01', '2023-02', '2023-02'],
    'category': ['A', 'B', 'A', 'B'],
    'value': [10, 20, 30, 40]
})

# 重塑数据
pivot = df.pivot(index='date', columns='category', values='value')
print(pivot)
# 输出:
# category   A   B
# date            
# 2023-01   10  20
# 2023-02   30  40
```

---

## 数据合并

### 连接

##### `pd.concat()`
沿轴连接多个 DataFrame。

```python
df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
df2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})

# 垂直连接（默认）
result = pd.concat([df1, df2])
print(result)
# 输出:
#    A  B
# 0  1  3
# 1  2  4
# 0  5  7
# 1  6  8

# 重置索引
result = pd.concat([df1, df2], ignore_index=True)
print(result)
# 输出:
#    A  B
# 0  1  3
# 1  2  4
# 2  5  7
# 3  6  8

# 水平连接
result = pd.concat([df1, df2], axis=1)
print(result)
# 输出:
#    A  B  A  B
# 0  1  3  5  7
# 1  2  4  6  8
```

### 合并

##### `pd.merge()`
类似 SQL 的合并操作。

```python
df1 = pd.DataFrame({
    'key': ['A', 'B', 'C'],
    'value1': [1, 2, 3]
})
df2 = pd.DataFrame({
    'key': ['A', 'B', 'D'],
    'value2': [4, 5, 6]
})

# 内连接（默认）
result = pd.merge(df1, df2, on='key')
print(result)
# 输出:
#   key  value1  value2
# 0   A       1       4
# 1   B       2       5

# 左连接
result = pd.merge(df1, df2, on='key', how='left')
print(result)
# 输出:
#   key  value1  value2
# 0   A       1     4.0
# 1   B       2     5.0
# 2   C       3     NaN

# 右连接
result = pd.merge(df1, df2, on='key', how='right')
print(result)
# 输出:
#   key  value1  value2
# 0   A     1.0       4
# 1   B     2.0       5
# 2   D     NaN       6

# 外连接
result = pd.merge(df1, df2, on='key', how='outer')
print(result)
# 输出:
#   key  value1  value2
# 0   A     1.0     4.0
# 1   B     2.0     5.0
# 2   C     3.0     NaN
# 3   D     NaN     6.0

# 基于多列合并
df1 = pd.DataFrame({
    'key1': ['A', 'B'],
    'key2': ['X', 'Y'],
    'value1': [1, 2]
})
df2 = pd.DataFrame({
    'key1': ['A', 'B'],
    'key2': ['X', 'Y'],
    'value2': [3, 4]
})
result = pd.merge(df1, df2, on=['key1', 'key2'])
print(result)
# 输出:
#   key1 key2  value1  value2
# 0    A    X       1       3
# 1    B    Y       2       4
```

##### `.join()`
基于索引合并。

```python
df1 = pd.DataFrame({
    'A': [1, 2, 3]
}, index=['a', 'b', 'c'])

df2 = pd.DataFrame({
    'B': [4, 5, 6]
}, index=['a', 'b', 'd'])

# 左连接（默认）
result = df1.join(df2)
print(result)
# 输出:
#    A    B
# a  1  4.0
# b  2  5.0
# c  3  NaN

# 内连接
result = df1.join(df2, how='inner')
print(result)
# 输出:
#    A  B
# a  1  4
# b  2  5
```

---

## 时间序列

### 创建日期时间

##### `pd.date_range()`
创建日期范围。

```python
# 创建日期范围
dates = pd.date_range('2023-01-01', periods=5)
print(dates)
# 输出:
# DatetimeIndex(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04',
#                '2023-01-05'],
#               dtype='datetime64[ns]', freq='D')

# 指定频率
dates = pd.date_range('2023-01-01', periods=5, freq='M')  # 月
print(dates)

dates = pd.date_range('2023-01-01', periods=5, freq='H')  # 小时
print(dates)

# 指定起止日期
dates = pd.date_range('2023-01-01', '2023-01-10')
print(dates)
```

##### `pd.to_datetime()`
转换为日期时间。

```python
# 从字符串转换
dates = pd.to_datetime(['2023-01-01', '2023-02-15', '2023-03-20'])
print(dates)

# 从多列创建日期时间
df = pd.DataFrame({
    'year': [2023, 2023, 2023],
    'month': [1, 2, 3],
    'day': [1, 15, 20]
})
df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
print(df)
# 输出:
#    year  month  day       date
# 0  2023      1    1 2023-01-01
# 1  2023      2   15 2023-02-15
# 2  2023      3   20 2023-03-20
```

### 日期时间属性

##### `.dt` 访问器
访问日期时间属性。

```python
dates = pd.date_range('2023-01-01', periods=5)
df = pd.DataFrame({'date': dates})

# 提取年份
print(df['date'].dt.year)
# 输出:
# 0    2023
# 1    2023
# 2    2023
# 3    2023
# 4    2023
# Name: date, dtype: int32

# 提取月份
print(df['date'].dt.month)

# 提取日
print(df['date'].dt.day)

# 提取星期几（0=周一，6=周日）
print(df['date'].dt.dayofweek)

# 提取星期几的名称
print(df['date'].dt.day_name())
# 输出:
# 0       Sunday
# 1       Monday
# 2      Tuesday
# 3    Wednesday
# 4     Thursday
# Name: date, dtype: object

# 提取季度
print(df['date'].dt.quarter)
```

### 时间序列操作

##### 重采样 `.resample()`
对时间序列数据进行重采样。

```python
# 创建时间序列数据
dates = pd.date_range('2023-01-01', periods=10, freq='D')
df = pd.DataFrame({
    'date': dates,
    'value': range(10)
})
df.set_index('date', inplace=True)

# 按周重采样并求和
weekly = df.resample('W').sum()
print(weekly)
# 输出:
#             value
# date             
# 2023-01-01      0
# 2023-01-08     28
# 2023-01-15     17

# 按月重采样并求平均值
monthly = df.resample('M').mean()
print(monthly)
```

##### 时间偏移
对日期进行偏移。

```python
from pandas.tseries.offsets import Day, MonthEnd

dates = pd.date_range('2023-01-01', periods=3)
df = pd.DataFrame({'date': dates})

# 加上 5 天
df['date_plus_5'] = df['date'] + Day(5)
print(df)
# 输出:
#         date date_plus_5
# 0 2023-01-01  2023-01-06
# 1 2023-01-02  2023-01-07
# 2 2023-01-03  2023-01-08

# 移动到月末
df['month_end'] = df['date'] + MonthEnd(0)
print(df)
```

##### 滚动窗口 `.rolling()`
计算滚动统计。想上查找数 window 个数据点进行计算

```python
df = pd.DataFrame({
    'value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
})

# 计算 3 天滚动平均
df['rolling_mean'] = df['value'].rolling(window=3).mean()
print(df)
# 输出:
#    value  rolling_mean
# 0      1           NaN
# 1      2           NaN
# 2      3           2.0
# 3      4           3.0
# 4      5           4.0
# 5      6           5.0
# 6      7           6.0
# 7      8           7.0
# 8      9           8.0
# 9     10           9.0

# 计算滚动总和
df['rolling_sum'] = df['value'].rolling(window=3).sum()
print(df)
```

---

## 数据可视化

Pandas 内置了基于 Matplotlib 的绘图功能。

##### `.plot()`
绘制基本图表。

```python
import matplotlib.pyplot as plt

df = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': [2, 4, 6, 8, 10]
})

# 折线图（默认）
df.plot()
plt.show()

# 柱状图
df.plot(kind='bar')
plt.show()

# 水平柱状图
df.plot(kind='barh')
plt.show()

# 散点图
df.plot(kind='scatter', x='A', y='B')
plt.show()

# 直方图
df['A'].plot(kind='hist')
plt.show()

# 箱线图
df.plot(kind='box')
plt.show()

# 饼图
df['A'].plot(kind='pie')
plt.show()
```

##### 自定义图表

```python
df = pd.DataFrame({
    'date': pd.date_range('2023-01-01', periods=10),
    'value': [1, 3, 2, 5, 4, 7, 6, 9, 8, 10]
})
df.set_index('date', inplace=True)

# 自定义标题和标签
df.plot(
    title='时间序列图',
    xlabel='日期',
    ylabel='数值',
    figsize=(10, 6),
    color='red',
    linestyle='--',
    marker='o'
)
plt.show()
```

---

