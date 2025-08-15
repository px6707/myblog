---
title: 单片机存储技术FlashE²PROM MaskROM EPROM的区别
published: 2025-08-15
description: 单片机存储技术FlashE²PROM MaskROM EPROM的区别.
tags: [单片机]
category: 单片机
licenseName: "Unlicensed"
author: panxiao
sourceLink: "https://github.com/px6707/myblog"
draft: false
---

## FlashE²PROM MaskROM EPROM的区别
### FlashE²PROM
- 擦除方式：电擦除，通过电信号擦除数据
- 编程方式：电编程，可在电路中进行
- 擦除粒度： 通常按块（sector）擦除
- 擦写次数： 10,000-100,000次
- 特点：
  - 非易失性（断电不丢失数据）
  - 可在线编程
  - 无需从电路中取出即可更新程序
  - 擦写速度快
- 应用： 现代单片机如STC89C52RC等，适合需要频繁更新程序的应用场景
### MaskROM

- 擦除方式：不可擦除
- 编程方式：再制造过程中通过掩模（mask）一次性编程
- 擦写次数： 0次，不可重写
- 特点：
  - 非易失性（断电不丢失数据）
  - 一旦制造完成不可更改
  - 成本低
  - 可靠性高
- 应用：80C51等早期单片机适合大批量生产且程序固定不变的场景

### EPROM
- 擦除方式：紫外线擦除（需要特殊的紫外线擦除器）
- 编程方式：需要专用编程器
- 擦写次数： 约1000次
- 特点：
  - 非易失性（断电不丢失数据）
  - 需要从电路中取出才能擦除
  - 芯片上有石英窗口用于紫外线照射
  - 擦除过程需要20-30分钟
- 应用：87C51等单片机，适合开发阶段和小批量生产

##### 技术演进关系
MaskROM → EPROM → FlashE²PROM 代表了单片机存储技术的演进过程，每一代技术都提高了灵活性和便利性，使得程序更新越来越方便。在现代单片机中，FlashE²PROM已经成为主流选择。