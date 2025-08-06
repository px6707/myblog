---
title: pnpm如何解决幽灵依赖和磁盘浪费
published: 2025-08-06
description: pnpm如何解决幽灵依赖和磁盘浪费.
tags: [工程化, pnpm]
category: 工程化
licenseName: "Unlicensed"
author: panxiao
sourceLink: "https://github.com/px6707/2025_WEB_STUDY/blob/main/node/pnpm.md"
draft: false
---
## pnpm

pnpm：使用硬链接和符号链接，每个包都有独立的依赖空间
严格的依赖隔离
避免幽灵依赖
节省磁盘空间
安装速度快

### 硬链接 Hard Link

- 本质上是文件系统中指向统一物理数据（inode）的多个目录条目（文件名）。它直接指向文件内容
- 关键特性
    1. 文件内容共享：所有硬连接都平等的指向同一份物理数据。修改任何一个链接，其他所有链接看到的都是修改后的内容
    2. 独立性：删除任何一个硬连接（文件名），包括删除原始文件，只要还有一个硬连接存在，文件数据就不会被删除
    3. 无大小开销：创建硬链接只增加一个目录项，不消耗额外的磁盘空间
    4. 局限性：只能链接文件，不能链接目录。硬连接通常不能跨越不同的分区或卷

### 符号连接
- 本质上是一个特殊的文件，其内容存储的是另一个文件或目录的路径，它执行另一个路径名
- 关键特性
    1. 间接性：他不直接指向文件内容，而是指向另一个路径名
    2. 依赖目标：如果目标文件/目录被移动、重命名、删除，符号连接将失效，成为断链接（悬空链接）
    3. 有大小开销： 符号连接本身是一个小文件，占用少量磁盘空间来存储目标路径字符串
    4. 灵活性：可以链接文件和目录，可以跨域不同的分区/卷


### pnpm工作原理

1. 幽灵依赖问题
    - 问题： 项目只安装了A包（npm install A）。但是A包自己依赖了B包。因为是扁平化结构，B包也会被提升到node_modules的根目录。结果就是，你在你的代码里，明明没有在package.json里声明过B，但你却可以import B from 'B'。万一有一天，A包升级了，不再依赖B了，你的项目就会在某个意想不到的地方突然崩溃，而你甚至都不知道B是从哪来的。
    - 解决：pnpm 的 node_modules 里面只会看到 package.json 中明确声明的依赖。你项目里依赖的A包，它自己所依赖的B包，会被存放在node_modules/.pnpm/这个特殊的目录里，然后通过 符号链接（Symbolic Link） 的方式，链接到A包的node_modules里。这意味着，在项目中不能 import B
2. 磁盘浪费
    - 问题：如果你电脑上有10个项目，这10个项目都依赖了lodash，那么在npm/yarn的模式下，你的磁盘上就会实实在在地存着10份一模一样的lodash代码。
    - 解决：pnpm会在你的电脑上创建一个“全局内容可寻址存储区”（content-addressable store），通常在用户主目录下的.pnpm-store里。所有项目的所有依赖，都只会在这个全局仓库里，实实在在地只存一份。项目需要lodash时，pnpm不会去复制一份lodash到你的node_modules里，而是通过 硬链接（Hard Link） 的方式，从全局仓库链接一份过来。硬链接几乎不占用磁盘空间。


3. 安装速度的瓶颈
    - 问题：虽然npm和yarn都有缓存机制，但在安装依赖时，它们仍然需要做大量的I/O操作，去复制、移动那些文件。当项目越来越大，node_modules越来越大，安装速度就会越来越慢。
    - 解决：大部分依赖都是通过“链接”的方式实现的，而不是“复制”，所以pnpm在安装依赖时，大大减少了磁盘I/O操作。

### 为什么解决幽灵依赖使用软连接，解决磁盘浪费使用硬连接？
#### 解决磁盘浪费使用硬连接，目标是在多个项目中共享完全的包文婧内容，避免重复存储
1. 使用硬连接的原因
    - 硬连接高效共享：所有安装位置（项目的.pnpm）的包文件都是内容存储区.npn-store的硬连接,修改文件内容都会反映到所有地方。
    - 硬连接节省空间：创建硬连接几乎不占用磁盘空间
    - 硬连接稳定： 即使原始存储区的文件被清理（PNPM 通常不会主动清理正在使用的），只要项目中的硬链接还存在，文件内容就依然可访问（直到所有硬链接被删除）。这对于项目运行稳定性很重要。
 2. 不使用软链接的原因
    - 无法节省空间：符号链接虽然很小，但依然有开销
    - 无法保证稳定性：符号链接依赖于原始文件，如果原始文件被删除，符号链接将失效

#### 解决幽灵依赖使用软连接，目标是防止未声明的包（依赖的依赖）意外地出现在顶级 node_modules 下被直接引用。

1. 使用符号链接的原因
    - 创建虚拟视图：PNPM 在项目的 node_modules 目录下，为每个直接依赖创建一个符号链接。这个符号链接指向 .pnpm 目录内该依赖实际安装的位置（该位置包含依赖自己的 node_modules，里面有它自己的依赖，这些依赖也是硬链接到存储区的）。
    - 隔离性：依赖的依赖被严格嵌套在 .pnpm/<依赖>@版本/node_modules 下面。项目代码在顶级 node_modules 下只能看到直接依赖（符号链接）和少量必要的平铺包（如 pnpm 自身）。
    - 路径控制：符号链接本质上是指向另一个路径的指针。PNPM 利用这一点精确控制哪些包在哪些路径下“可见”。通过将直接依赖符号链接到顶级，而让间接依赖隐藏在嵌套结构中，完美实现了依赖树的隔离。
    - 灵活性：符号链接可以轻松指向 .pnpm 目录内复杂的嵌套结构，创建这种“虚拟”的目录视图非常方便。
2. 不使用硬链接的原因
    - 无法创建结构/视图：硬链接只能链接文件，不能链接目录（这是文件系统限制）。而 node_modules 结构的核心是目录组织。硬链接无法用来在 node_modules 根目录下“创建”一个指向 .pnpm 内某个子目录的“入口”。
    - 无法实现隔离
    - 破坏嵌套依赖解析 顶层的 A 包如果是硬连接，则 A 包里面不能再有硬连接，即 A 包node_moduleszhong 的依赖项也无法使用硬连接创建。
 3. PNPM 实际结构
 project/
  node_modules/
    react -> .pnpm/react@18.2.0/node_modules/react (符号链接)
    next -> .pnpm/next@14.1.0/node_modules/next (符号链接)
  .pnpm/
    react@18.2.0/
      node_modules/
        react/ (硬链接文件集合)
    next@14.1.0/
      node_modules/
        next/ (硬链接文件集合)
        react -> ../../react@18.3.0/node_modules/react (符号链接) # 关键隔离层
    react@18.3.0/
      node_modules/
        react/ (独立的硬链接集合)

### pnpm 硬连接原理
您指出了一个非常关键的技术限制！**硬链接确实无法链接目录**（在POSIX和Windows文件系统中都是如此），这正是PNPM实现中需要巧妙设计的核心挑战。PNPM并没有直接硬链接整个目录，而是通过一套精密的组合策略实现目录级共享：

#### PNPM 解决目录硬链接限制的秘诀：**文件级硬链接 + 虚拟目录结构**

##### 1.  **内容可寻址存储（Content-Addressable Store）**
  *   PNPM 在全局存储目录 (如 `~/.pnpm-store/v3/files`) 中存储包文件。
  *   **关键点：存储的是 *文件*，而不是 *目录*。** 每个文件根据其内容生成唯一的哈希值作为文件名（或目录名的一部分）。例如：
      ```
      ~/.pnpm-store/v3/files/00/xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
      ~/.pnpm-store/v3/files/f1/yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy
      ```
  *   这样，**相同的文件内容无论来自哪个包，在存储区只会保存一份**（因为哈希值相同）。

##### 2.  **文件级硬链接（File-level Hard Links）**
  *   当安装一个包到项目时，PNPM 不会复制包目录下的所有文件。
  *   **核心操作：** 对于包目录下的 **每一个文件**，PNPM 在项目的 `.pnpm` 目录中创建该文件的 **硬链接**，指向存储区中对应的唯一物理文件。
  *   **目标位置：** 这些硬链接被组织在 `.pnpm/<package-name>@<version>/node_modules/<package-name>/` 目录下。
      *   例如，`lodash@4.17.21` 包中的 `index.js` 文件在项目中的硬链接路径可能是：
          ```
          project/.pnpm/lodash@4.17.21/node_modules/lodash/index.js
          ```
      *   这个 `index.js` 文件就是一个硬链接，指向存储区中哈希值对应的那个唯一物理文件。

##### 3.  **重建虚拟目录结构（Recreating Virtual Directory Structures）**
  *   上一步只是在 `.pnpm` 下创建了一个包含硬链接文件的目录树，它**模拟**了原始包的目录结构。
  *   这个目录结构 (`project/.pnpm/lodash@4.17.21/node_modules/lodash/`) 包含了包的所有文件（硬链接形式）和必要的子目录结构。**它本质上就是一个普通的目录，里面的文件是共享的硬链接。**
  *   **为什么不是硬链接目录？** 文件系统不允许。PNPM 通过“在目标位置重新创建目录结构，并在这个结构内为每个文件创建硬链接”来模拟。

##### 4.  **符号链接暴露依赖（Soft Links for Dependency Exposure）**
  *   为了在项目的 `node_modules` 根目录下让包**可见**（解决幽灵依赖），PNPM 在项目的顶级 `node_modules` 目录中创建**符号链接**。
  *   **操作：**
      1.  对于项目的**直接依赖**（在 `package.json` 中声明的），在 `project/node_modules/<package-name>` 处创建一个**符号链接**。
      2.  这个符号链接指向 `.pnpm` 目录内对应的虚拟包目录：
          ```
          project/node_modules/lodash -> ./.pnpm/lodash@4.17.21/node_modules/lodash
          ```
  *   这样，Node.js 在 `project/node_modules` 下查找 `lodash` 时，会找到这个符号链接，并跟随它跳转到 `.pnpm/lodash@4.17.21/node_modules/lodash`，然后访问那里的文件（硬链接）。

##### 5.  **处理嵌套依赖（Nested Dependencies - Virtual Store）**
  *   包自己的依赖（比如 `lodash` 依赖了 `some-helper`）需要放在它自己的 `node_modules` 下，以确保隔离性，避免幽灵依赖。
  *   **问题：** `.pnpm/lodash@4.17.21/node_modules/lodash/` 本身只是一个模拟的包目录（包含硬链接文件），它没有能力直接包含一个真实的 `node_modules` 子目录。
  *   **解决方案 - 虚拟存储目录 (Virtual Store Directory):**
      1.  在 `.pnpm/lodash@4.17.21/node_modules/` 下，除了符号链接指向的 `lodash` 目录外，PNPM 还会创建一个特殊的 `.pnpm` 子目录（有时称为“虚拟存储目录”，虽然命名可能不同，但作用一致）。
      2.  **关键：** 在这个 `.pnpm/lodash@4.17.21/node_modules/.pnpm/` 目录下，PNPM 会为 `lodash` 的依赖包（如 `some-helper@1.0.0`）创建**符号链接**，指向它们在 `.pnpm` 主目录中的虚拟包位置：
          ```
          project/.pnpm/lodash@4.17.21/node_modules/.pnpm/some-helper@1.0.0/node_modules/some-helper
          ```
      3.  PNPM 在 `.pnpm/lodash@4.17.21/node_modules/lodash/` 的同级目录下创建一个指向 `some-helper` 的**符号链接**：
          ```
          project/.pnpm/lodash@4.17.21/node_modules/some-helper -> ./.pnpm/some-helper@1.0.0/node_modules/some-helper
          ```
  *   **结果：** 当 `lodash` 内部的代码执行 `require('some-helper')` 时，Node.js 的解析过程如下：
      1.  从 `lodash` 文件的位置 (`project/.pnpm/lodash@4.17.21/node_modules/lodash/index.js`) 开始查找。
      2.  向上找到 `project/.pnpm/lodash@4.17.21/node_modules/lodash/node_modules` (不存在或为空)。
      3.  继续向上找到 `project/.pnpm/lodash@4.17.21/node_modules`。
      4.  在这个目录下找到了符号链接 `some-helper` -> `./.pnpm/some-helper@1.0.0/node_modules/some-helper`。
      5.  跟随符号链接，最终找到 `some-helper` 包的代码（同样是文件硬链接）。
  *   **隔离性达成：** `some-helper` **只**在 `lodash` 的“作用域”（即 `project/.pnpm/lodash@4.17.21/node_modules/`）内可见。项目代码直接在 `project/node_modules` 下 `require('some-helper')` 会找不到它，从而避免了幽灵依赖。

#### 总结：PNPM 如何“硬链接整个包”

1.  **分解包：** 将包视为一组文件。
2.  **文件级硬链接：** 在全局存储区和项目的 `.pnpm/<package>@<version>/node_modules/<package>/` 目录之间，为包内的**每一个文件**创建硬链接。这实现了**文件内容**的跨项目共享，节省磁盘空间。
3.  **重建目录：** 在 `.pnpm` 下为每个包版本重建其原始的目录结构（包含硬链接文件）。
4.  **符号链接暴露：** 在项目的顶级 `node_modules` 中使用符号链接，将直接依赖“映射”到 `.pnpm` 下重建的包目录。
5.  **嵌套符号链接隔离：** 在 `.pnpm/<package>@<version>/node_modules/` 下使用符号链接指向该包的依赖，将这些依赖严格限制在该包的“作用域”内，解决幽灵依赖问题。

**简单来说：PNPM 通过“为包内每个文件创建硬链接” + “在`.pnpm`下重建包目录结构” + “使用符号链接灵活组织依赖树视图” 的组合拳，巧妙地绕过了硬链接不能链接目录的限制，同时实现了磁盘空间节省和依赖隔离两大核心目标。** 硬链接负责解决物理存储问题，符号链接负责解决逻辑结构和访问路径问题。