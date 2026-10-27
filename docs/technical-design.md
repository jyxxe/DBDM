# DBDM：动态矩阵块扩散推理模型技术设计与实验计划

> 版本：v0.1（研究提案）  
> 日期：2026-10-27  
> 状态：待验证架构；所有规模和性能均为设计目标，不代表已实现或经过实验确认。

## 摘要

本文提出 **Dynamic Block Diffusion Model（DBDM）**：一种以可变长度连续矩阵作为内部工作空间的条件生成与推理架构。系统包含两个编码器、一个动态块扩散网络、一个操作控制器和一个自回归文本解码器。内部状态由若干个 64 槽连续矩阵块构成。推理核心能够选择 **APPEND（追加 64 槽）**、**EDIT（重写已有 64 槽区间）**或 **STOP（停止）**。每个块的生成或修订由条件扩散或流匹配执行。输出解码器每次仅通过交叉注意力读取 64 槽局部窗口，同时通过因果自注意力读取已有输出 Token；其窗口滑动步长与回答详细程度由条件控制。

计划构建五个尺度的模型家族，依次预训练编解码系统、训练连续空间生成器和动态编辑策略，再进行有限的端到端微调。主要研究假设是：在相近的训练数据、参数与推理计算预算下，**可重写的动态连续状态**能够改善多步推理、错误纠正与多粒度文本输出，而不必将内部每步推理离散化为自然语言。

**范围说明：**本方案不是对现有 Transformer 已经使用矩阵计算这一事实的简单重述；其待验证创新在于：连续工作矩阵的动态长度、随机位置重写、块级扩散更新，以及仅观察局部潜在窗口的自回归解码。

---

## 1. 目标、非目标与研究问题

### 1.1 目标

1. 建立能被自回归 Decoder 稳定读取的连续潜在矩阵空间。
2. 允许内部矩阵长度随任务复杂度动态增长，基本添加单位为 64 个槽。
3. 允许修改任意已有的 64 槽连续区间，而不仅是最新块。
4. 将每次局部块的生成/修订建模为多步扩散或 Flow Matching。
5. Decoder 的矩阵可见范围限制为 64 槽，利用可调滑动步长和显式详细度指令实现简略/详细回答。
6. 在相同任务上与常规 AR、块级扩散、不可修改潜变量模型做计算预算匹配的比较。

### 1.2 非目标

- 不保证固定容量矩阵能够无损存储任意长度文本。
- 不假设每个潜在槽具有固定、可人类解释的含义。
- 不假设连续浮点运算天然提供精确数学推理；需显式验证或借助符号工具。
- 不假设减少中间文本 Token 就一定提高速度：块操作次数 × 每块去噪步数可能带来很高的前向成本。

### 1.3 核心可证伪假设

- **H1（可解码）**：可变数量的 64 槽块能表示信息，并支持 64 槽局部窗口 AR 解码。
- **H2（可生成）**：条件扩散可在不访问真实答案的情况下生成可解码的目标潜在块。
- **H3（可编辑）**：EDIT 对原有块的条件修订能提升最终答案准确率，而非只是还原加噪的原状态。
- **H4（资源自适应）**：操作控制器能够在准确率与计算预算之间学习有效的 APPEND/EDIT/STOP 决策。
- **H5（详略控制）**：在相同固定潜在矩阵上，步长加显式详细度标签可稳定控制答案覆盖率与长度，且结论保持一致。

---

## 2. 数学对象与记号

| 符号 | 含义 | 形状/范围 |
|---|---|---|
| $x$ | 原始输入 | Token 序列或其他模态 |
| $y$ | 目标输出文本、推理轨迹或结构化答案 | Token 序列 |
| $C=E_{in}(x)$ | 输入条件记忆 | $L\times d_c$ |
| $Z^*=E_{lat}(y)$ | 训练用目标潜在矩阵 | $n\times d_z$ |
| $M_k$ | 第 $k$ 次操作后的工作矩阵 | $n_k\times d_z$ |
| $b$ | 基础块长度 | **64** |
| $K_k$ | 当前基础块数 | $n_k=64K_k$ |
| $a_k$ | 外层操作类型 | APPEND / EDIT / STOP |
| $s_k$ | EDIT 选中区间的起点 | $0\le s_k\le n_k-64$ |
| $W$ | 局部读写窗口 | $64\times d_z$ |
| $T$ | 块内扩散采样步数 | 例如 4、8、16、32 |
| $r$ | Decoder 窗口滑动步长 | 1–64 |
| $v$ | 输出详细度控制变量 | 简略/标准/详细 |

**矩阵方向约定：**实现代码默认采用 `[batch, slots, channels]`，即 $M\in\mathbb R^{n\times d_z}$；如以 $d_z\times n$ 表示，只是转置，含义不变。

### 2.1 状态和输入分离

工作矩阵不等于全部事实库。输入条件 $C$ 在推理期间可以保留，并为局部更新器提供交叉注意力的外部记忆。否则长问题压缩到有限槽数会损失必要信息。

### 2.2 外层操作

$$
(a_k,s_k)=\operatorname{Controller}_{\psi}(M_k,C,\mathrm{budget}_k).
$$

$$
M_{k+1}=\begin{cases}
\operatorname{Concat}(M_k,\operatorname{SampleBlock}_{\theta}(C,M_k)), &a_k=\mathrm{APPEND},\\
\operatorname{Replace}(M_k,s_k,\operatorname{RepairBlock}_{\theta}(M_k,C,s_k)),&a_k=\mathrm{EDIT},\\
M_k,&a_k=\mathrm{STOP}.
\end{cases}
$$

APPEND 初期只在末尾插入新块；EDIT 第一版限定 $s_k$ 为 64 的整数倍，后续扩展到任意连续区间（可跨原始块边界）。

---

## 3. 端到端架构

```text
用户输入 x
    |
    v
Input Encoder E_in ------------------------------+
    |                                            |
    v                                            |
条件记忆 C                                        |
                                                 |
当前工作矩阵 M_k -- Block Summary / Retrieval ---+
    |                                            |
    +--> Action Controller --> APPEND / EDIT / STOP
                   |                             |
                   +--> 选择新块或 64 槽窗口 -----+
                                                 |
                                Conditional Block Diffusion / Flow
                                         (T 次局部去噪)
                                                 |
                                       替换 / 追加到 M_k
                                                 |
                                         继续或 STOP
                                                 |
                                     固定最终矩阵 M_final
                                                 |
                            64 槽 Sliding Window Reader
                                                 |
                       AR Decoder + token history + detail tag
                                                 |
                                             文本输出
```

### 3.1 两个 Encoder

**Input Encoder $E_{in}$**：负责表示问题、约束、事实和上下文，是扩散模型的条件源。  
**Latent Encoder $E_{lat}$**：在训练阶段把目标文本/推理轨迹映射到可解码的连续块矩阵，是定义目标分布的工具。两个 Encoder 在同一档位内参数规模相同，**但默认不共享权重**；可以在消融实验中尝试共享 token embedding 或底层 Transformer。

注意：推理时不知道目标文本 $y$，因此不运行 $E_{lat}(y)$，不得发生目标泄漏。

### 3.2 潜在矩阵构造

将目标长度映射到 $K$ 个 64 槽块，采用可学习槽查询与跨注意力池化，或先进行可逆性更好的局部分块编码后使用潜在池化。不要假设 64 个潜在槽等于 64 个 Token。编码器应提供显式块数 $K$、长度/掩码信息，以便训练可变长度和终止决策。

### 3.3 Diffusion 主干

使用双向 Transformer，接收：待更新 64 槽矩阵、噪声时间步嵌入、输入条件、其他块的概览向量、位置编码、当前操作类型。

必要时将 $d_z$ 投影到更大的 $d_h$：

$$
H=\operatorname{Linear}_{d_z\to d_h}(W),\qquad
\widehat W=\operatorname{Linear}_{d_h\to d_z}(F_\theta(H,t,C,\mathrm{context})).
$$

这样不必让 256 维矩阵承载一个过深的 100M Transformer。

### 3.4 分层记忆

- 每个 64 槽块提取概要向量 $h_i=\operatorname{Pool}(B_i)$。
- 局部扩散优先读取概要 $H=[h_1,\ldots,h_K]$，按需检索细粒度块。
- EDIT 后刷新被修改块及其依赖缓存；不得继续使用旧摘要。
- 比较全量 cross-attention、块概要注意力、top-k 检索的准确率和延迟。

### 3.5 自回归 Decoder（明确保留）

Decoder 以因果 Self-Attention 读取先前输出 Token，Cross-Attention 每次仅读取 $64\times d_z$ 的当前潜在窗口，额外接收详细度 $v$ 和步长 $r$：

$$
p(y_t\mid y_{<t},W_{j(t)},v,r).
$$

**关键策略**：为 Decoder 增设 READ/ADVANCE（移动窗口）或 WRITE（输出 Token）的门控；移动窗口不输出 Token。若只机械地每输出 $q$ 个 Token 就滑动，容易产生信息覆盖不均、重复或遗漏。

$$
p_{j+1}=\min(p_j+r,n-64).
$$

训练或推理时应保证末窗口可覆盖最后 64 槽，并提供窗口结束与 EOS 处理。若 $n<64$，需 padding 和可见性 mask。

**重要**：步长不是答案长度的充分决定因素。回答详略应通过 `(stride, verbosity, max_output_tokens)` 三者联合控制；详细程度需要配对监督。

---

## 4. 扩散与 Flow Matching 的两条实现路径

### 4.1 DDPM 风格潜在扩散

训练目标潜在窗口 $W_0$：

$$
W_t=\sqrt{\bar\alpha_t}W_0+\sqrt{1-\bar\alpha_t}\epsilon,
\quad\epsilon\sim\mathcal N(0,I).
$$

$$
\mathcal L_{\epsilon}=\mathbb E\|\epsilon-\epsilon_\theta(W_t,t,C,\mathrm{context},a,s)\|_2^2.
$$

采样根据噪声调度多步更新。**修改操作不能只以“旧块去噪后恢复旧块”为训练目标**，否则不会学会错误修复。

### 4.2 Flow Matching（推荐并行比较）

在噪声 $W_N$ 与目标 $W^*$ 之间构造路径：

$$
W_\tau=(1-\tau)W_N+\tau W^*,\quad \tau\in[0,1].
$$

$$
\mathcal L_{FM}=\mathbb E\big\|v_\theta(W_\tau,\tau,C)-\left(W^*-W_N\right)\big\|_2^2.
$$

采样时积分向量场；需在不同步数下比较数值稳定性和答案质量。Flow Matching 不是天然更准，需实验确定。

### 4.3 EDIT 专用的目标构造

从 $E_{lat}$ 编码的正确矩阵 $M^*$ 创建损坏版 $\widetilde M$：局部替换、删除事实对应信息、扰乱关系、错误中间步骤、用模型自身早期采样生成的错误块等。构造 $(\widetilde M,s,M^*)$ 的修复监督，再从修复后矩阵重新解码，检查答案是否提升。

对真实推理的 EDIT 不能只依赖随机加噪；需要可验证环境、轨迹或奖励来区分“内容变化”和“有效修正”。

---

## 5. 五档参数规模（用户确定的目标）

| 档位 | $d_z$ | $E_{in}$ | $E_{lat}$ | AR Decoder | Diffusion | 完整总量 |
|---|---:|---:|---:|---:|---:|---:|
| S1 | 256 | 20M | 20M | 50M | 100M | **190M** |
| S2 | 512 | 50M | 50M | 100M | 200M | **400M** |
| S3 | 768 | 100M | 100M | 200M | 500M | **900M** |
| S4 | 1024 | 200M | 200M | 500M | 1B | **1.9B** |
| S5 | 1536 | 500M | 500M | 1B | 2B | **4B** |

**第一阶段仅训练两个 Encoder 与 Decoder：**90M / 200M / 400M / 900M / 2B 参数。**第二阶段新增的 Diffusion：**100M / 200M / 500M / 1B / 2B 参数。控制器、投影头等小组件须纳入实际参数盘点；以上是名义规模，不是精准 `model.num_parameters()`。

### 5.1 初步结构搜索范围

| 档位 | Encoder 主干参考 | Decoder 主干参考 | Diffusion 主干参考 |
|---|---|---|---|
| S1 | 256 维，约 12–16 层 | 512 维，约 7–9 层 | 768 维，约 9–11 层 |
| S2 | 512 维，约 9–12 层 | 768 维，约 7–9 层 | 1024 维，约 10–12 层 |
| S3 | 768 维，约 9–12 层 | 1024 维，约 9–11 层 | 1536 维，约 12–14 层 |
| S4 | 1024 维，约 11–14 层 | 1536 维，约 11–13 层 | 2048 维，约 14–16 层 |
| S5 | 1536 维，约 14–17 层 | 2048 维，约 13–15 层 | 2560 维，约 18–20 层 |

以上 **不是精确的网络实例化表**。准确规模依赖 vocab size、词嵌入共享、是否使用 GQA、Self/Cross-Attention 数量、FFN 扩展倍数、Norm、投影和控制头。必须编写参数计数脚本精确调整 `num_layers`、`ffn_dim` 与共享策略。尤其 S1 的两个 20M Encoder 如果采用大型词表，嵌入表会占据较大比例。推荐实验阶段共享 tokenizer（不要求共享各模块的 embedding 权重），再评估 8K/16K/32K 词表。

### 5.2 推荐第一批真实实验

- **S1 首先落地**：尽快验证编码重构、窗口读取和最小 APPEND/EDIT 管线。
- **S2 作为首个完整系统**：验证控制器、潜在重写、可调回答详略。
- **S3 作为研究基线**：开展具有统计意义的消融与对照实验。
- **S4/S5 暂不直接全面训练**：待 S2/S3 对照结果证明收益后再扩大规模。

---

## 6. 分阶段训练路线

### 阶段 0：数据与基线

- 先构建标准 AR Encoder–Decoder、无 EDIT 的潜在块生成器与同参数量纯 AR baseline。
- 数据划分避免模板泄漏，按组合结构/操作深度/数字长度等方式制作 OOD 测试集。
- 训练集保留三类监督：(x,y)、(x,reasoning,y)、(x,structured_state,y)。

### 阶段 1A：Latent Encoder + AR Decoder

输入 $y$，编码为可变长度 $Z^*$，通过仅可见 64 槽的窗口式 AR Decoder 重构 $y$。

$$
\mathcal L_{recon}=-\sum_t\log p_\theta(y_t\mid y_{<t},W_{j(t)},v,r).
$$

建议增加 latent 归一化、少量噪声鲁棒性、块遮蔽、操作位置编码，以及窗口访问覆盖约束。

**必须检查**：关闭矩阵读取时 Decoder 性能是否明显下降；对矩阵置零、打乱、替换后，输出是否表现出相应差异。不能只凭低交叉熵证明矩阵被利用。

### 阶段 1B：Input Encoder

使用输入理解和条件预测目标训练 $E_{in}$；如采用成对数据，可用与目标有关的辅助预测/对比任务。避免让模型在训练或验证时接触目标答案的潜在编码作为输入。

### 阶段 1C：对齐与联合微调

在已稳定的矩阵空间中，联合训练输入编码、目标编码、输出解码的接口；按需锁定部分预训练模块，防止潜在空间无约束漂移。将矩阵窗口、详细度及输出长度训练目标共同纳入。

### 阶段 2A：冻结潜在空间，训练条件 Block Diffusion

$$
Z^*=\operatorname{StopGrad}(E_{lat}(y)).
$$

采样一块、一个噪声时刻训练网络，而不是每个样本都完整展开所有去噪步骤。先学一块的条件生成，再学多块 APPEND。

### 阶段 2B：EDIT 修复

构造目标已知的部分破坏、逻辑冲突和中间步骤修正监督，再用模型自身生成的错误状态做短轨迹训练。验证修改前后**最终答案**而非仅 latent MSE。

### 阶段 2C：操作控制器

训练三类操作的策略损失与位置选择；早期可用专家轨迹/启发式标签，后期引入可验证任务奖励及计算预算惩罚。给 STOP 明确的学习信号。

### 阶段 3：有限端到端调整

在固定的阶段 2 模型之上，解冻部分 Encoder、Decoder 与 Diffusion 参数，利用终端答案指标优化。使用较小学习率和潜在空间稳定正则项，防止先前训练的潜在分布漂移。

---

## 7. 推荐训练损失

$$
\mathcal L =
\lambda_r\mathcal L_{recon}+
\lambda_d\mathcal L_{diff/FM}+
\lambda_a\mathcal L_{action}+
\lambda_s\mathcal L_{span}+
\lambda_e\mathcal L_{edit}+
\lambda_c\mathcal L_{consistency}+
\lambda_b\mathcal L_{budget}.
$$

| 损失 | 监督对象 | 关键风险 |
|---|---|---|
| $L_{recon}$ | 文本重构/AR next token | Decoder 忽略矩阵 |
| $L_{diff/FM}$ | 目标潜在块生成 | 潜在空间难采样 |
| $L_{action}$ | APPEND/EDIT/STOP | 操作标签依赖启发式 |
| $L_{span}$ | 64 槽修改起点 | 任意窗口定位困难 |
| $L_{edit}$ | 修复后目标或任务奖励 | 仅学习加噪还原，不会纠错 |
| $L_{consistency}$ | 结构约束/跨块关系 | 一致性判别器可能失准 |
| $L_{budget}$ | 操作次数和计算预算 | 过度惩罚会使模型过早 STOP |

不同阶段只启用相关损失；不要在训练初期同时优化所有模块和操作目标。

---

## 8. Decoder 局部窗口与回答详略

窗口宽度固定 64。以 $n=256$ 为例，保证覆盖末端时，不同步长对应的窗口数：

| 步长 $r$ | 窗口数 | 相邻窗口重叠槽数 |
|---:|---:|---:|
| 1 | 193 | 63 |
| 8 | 25 | 56 |
| 16 | 13 | 48 |
| 32 | 7 | 32 |
| 64 | 4 | 0 |

计算公式（当 $n\ge64$ 且允许最后一步被截短以覆盖终点）：

$$
N_{windows}=\left\lceil\frac{n-64}{r}\right\rceil+1.
$$

**注：**窗口数量不是输出 Token 数；慢速滑动不保证更多有意义细节，快速滑动不保证简洁但正确。训练时同时提供 `(r,v,max_tokens)`，引入成对详略答案和事实一致性评测。

### 8.1 两种推进方案

1. **固定周期推进**：每输出 $q$ 个 Token 移动一次，适合作为简单 baseline。
2. **可学习推进**：模型预测 WRITE 或 ADVANCE，或由对齐模块预测窗口移动；推荐作为正式方案。

### 8.2 停止与缓存

- Decoder 已输出 Token 的自注意力采用 KV cache。
- 矩阵窗口改变时需刷新对应 Cross-Attention K/V；不能错误复用旧窗口缓存。
- 为避免 Decoder 在后续解码中看到被修改的潜在矩阵，推荐 **内部推理完毕后冻结最终矩阵，再开始文本解码**。
- 允许回答中途回到先前窗口的机制可以作为后续拓展，不宜第一版引入。

---

## 9. 数据集与任务设计

### 9.1 早期可验证任务

- 多位整数运算与进位传播（采用严格程序校验）。
- 图可达性、最短路径、拓扑依赖和简单规划。
- 逻辑约束满足、小型 SAT、状态跟踪。
- 表格条件筛选与聚合计算。
- 需要多次局部修正的程序执行模拟。

### 9.2 文本任务

- 长度可控的摘要/问答：同问题提供 3 个不同详细级别。
- 给定错误解释，要求保留正确部分并修正局部错误。
- 将结构化事实解码为自然语言。
- 检查事实覆盖、幻觉、重复率、篇幅及结构符合程度。

### 9.3 防止数据泄漏

- 组合任务按图大小、路径长度、模板组合与随机种子划分训练/测试。
- 数学任务不要仅在测试时变大数字，也需要新的操作组合与更长依赖深度。
- 对真实问答提供独立外部评测，防止训练数据重合。

---

## 10. 实验矩阵与消融设计

| 编号 | 改动 | 想回答的问题 |
|---|---|---|
| A0 | 纯 AR，同计算预算 | 新架构相比经典生成方式是否有效 |
| A1 | Latent Encoder + AR Decoder，无 Diffusion | 矩阵是否有用且可解码 |
| A2 | 固定长度潜在 Diffusion | 动态增长是否必要 |
| A3 | 只允许 APPEND | EDIT 是否带来增益 |
| A4 | APPEND + EDIT，无全局概要 | 全局一致性模块是否有用 |
| A5 | APPEND + EDIT + STOP | 学习停止是否优于固定预算 |
| A6 | 整矩阵 Decoder vs 64 槽窗口 | 局部读取的质量和效率代价 |
| A7 | r=8/16/32/64 | 步长如何影响答案详略与准确性 |
| A8 | 自适应 ADVANCE vs 固定推进 | 学习推进是否改善覆盖 |
| A9 | DDPM vs Flow Matching | 连续生成范式比较 |
| A10 | 目标编码仅答案 vs 推理轨迹 vs 混合 | 潜在空间应该表达什么 |
| A11 | 冻结 vs 部分共享 vs 端到端 | 编解码器训练策略比较 |

### 10.1 必须报告的指标

- **正确性**：任务准确率、程序校验率、复杂度 OOD 准确率。
- **效率**：端到端延迟、FLOPs、吞吐、显存峰值、每答案去噪前向次数。
- **矩阵行为**：实际块数、APPEND/EDIT/STOP 分布、平均修订次数、修改后答案改善率。
- **Decoder 行为**：输出 Token 数、重要事实覆盖、重复率、窗口移动次数、跨窗口一致性。
- **稳健性**：不同随机种子、不同最大块数、扰动后的修复率、长输入失败模式。

### 10.2 统计要求

对 S1/S2 至少采用多个训练随机种子，报告均值与置信区间；更大档位在预算有限时可先进行单次探索，再对关键结论复制实验。所有基线应尽可能匹配训练数据、训练 FLOPs 和推理 FLOPs，而不仅仅匹配参数量。

---

## 11. 显存与算力预算

下表仅估算常见 BF16 + AdamW 全参数训练时的**参数、梯度、优化器状态**量级，使用每参数约 16 bytes 的保守经验值，**不含激活、注意力工作空间、通信、缓存、冻结网络驻留成本**。

| 档位 | 阶段 1 训练参数 | 阶段 1 基础状态显存 | 阶段 2 Diffusion 参数 | Diffusion 基础状态显存 |
|---|---:|---:|---:|---:|
| S1 | 90M | 约 1.44 GB | 100M | 约 1.6 GB |
| S2 | 200M | 约 3.2 GB | 200M | 约 3.2 GB |
| S3 | 400M | 约 6.4 GB | 500M | 约 8.0 GB |
| S4 | 900M | 约 14.4 GB | 1B | 约 16 GB |
| S5 | 2B | 约 32 GB | 2B | 约 32 GB |

阶段 2 可能还需要在 GPU 上运行冻结的两个 Encoder 和 Decoder；如果离线缓存目标 latent，可以减少 Latent Encoder 的在线需求，但在线验证答案时仍需要 Decoder。

**关键成本公式（粗略）：**

$$
N_{denoiser\_forwards}\approx N_{outer\_ops}\times T.
$$

例如 8 次外层操作、每次 16 步去噪，意味着 128 次去噪网络前向，不包含控制器和 AR 解码。首次实验优先尝试少步采样与蒸馏，测量真实 wall-clock 成本。

### 11.1 工程策略

- BF16，FlashAttention/SDPA，梯度检查点，长度 bucket，动态 padding。
- S4/S5 采用 FSDP/ZeRO/张量并行等分片策略。
- 仅在有限短轨迹上反向传播操作序列，避免整个动态工作流全展开导致显存爆炸。
- 记录计算预算：训练总 Tokens/样本、更新步数、FLOPs、GPU 小时和推理延迟。

---

## 12. 仓库结构建议

```text
dbdm/
  configs/
    s1.yaml
    s2.yaml
    s3.yaml
    s4.yaml
    s5.yaml
  models/
    input_encoder.py
    latent_encoder.py
    ar_decoder.py
    block_denoiser.py
    action_controller.py
    block_memory.py
    window_reader.py
  training/
    stage1_autoencoder.py
    stage1_input_encoder.py
    stage2_diffusion.py
    stage2_edit.py
    stage2_controller.py
    stage3_joint.py
  data/
    synthetic_reasoning.py
    text_multigranularity.py
    corrupted_states.py
  eval/
    reconstruction.py
    reasoning_accuracy.py
    edit_success.py
    window_coverage.py
    compute_profile.py
  tests/
    test_shape_invariants.py
    test_append_edit.py
    test_window_mask.py
    test_no_label_leakage.py
    test_kv_cache.py
  scripts/
    count_params.py
    profile_memory.py
    run_ablation.py
```

### 12.1 最小算法草图

```python
# 演示逻辑，不是可直接训练的完整代码。
def reason_and_decode(x, detail, stride, max_ops, steps):
    C = input_encoder(x)
    M = initial_memory(C)  # [batch, 64*K, d_z]
    for k in range(max_ops):
        action, start = controller(M, C, budget=max_ops-k)
        if action == 'STOP':
            break
        if action == 'APPEND':
            block = sample_block_diffusion(C, M, steps=steps)
            M = append_block(M, block)
        else:
            old = select_window(M, start, width=64)
            new = repair_block_diffusion(old, C, M, steps=steps)
            M = replace_window(M, start, new)
        M = refresh_block_summaries(M)

    M = M.detach()  # inference stage: freeze latent matrix
    return ar_decode_sliding(M, detail=detail, stride=stride)
```

---

## 13. 关键技术风险与解决策略

| 风险 | 可见症状 | 应对方案 |
|---|---|---|
| 潜在表示坍塌 | 不同文本有近似 latent | 方差与信息量监测、重构/对比监督 |
| Decoder 忽略矩阵 | 矩阵扰动不影响输出 | 限制直接输入旁路、反事实矩阵测试 |
| Diffusion 只会模仿训练 latent | 样本可解码但任务答案错误 | 条件目标、答案校验、OOD 评测 |
| EDIT 无真实修正能力 | 修改后答案不改善 | 修复对、错误状态训练、结果奖励 |
| 全局一致性破坏 | 修改前块后，后块陈旧 | 依赖摘要刷新、二次检查、局部重算 |
| 操作次数爆炸 | 无限 APPEND 或重复 EDIT | 最大预算、计算成本损失、STOP 校准 |
| 局部读窗口遗漏信息 | 答案漏条件或跳步 | 全局排序/摘要、推进监督、覆盖率约束 |
| 步长仅影响重复率 | 小步长更长但不更好 | 详略监督、同事实一致性目标 |
| 训练推理分布不一致 | 教师 latent 好、采样 latent 差 | 噪声鲁棒 Decoder、采样轨迹微调 |
| 计算成本过高 | 性能提升抵不过成本 | 减少采样步数、缓存、蒸馏、检索优化 |

---

## 14. 项目里程碑与验收条件

### M0：训练基础设施

- 参数计数、数据管道、断点恢复、形状测试、profiling 可用。
- 5 档配置文件可实例化，且报告真实参数量和显存。

### M1：连续矩阵自编码

- S1 可以将不同长度目标文本编码为 64 槽块序列并完成重构。
- Matrix ablation 显示 Decoder 真正依赖潜在信息。
- 对不同步长和详细度有可观测、可度量的输出差异。

### M2：条件生成

- S1/S2 Diffusion 不访问目标文本仍能生成可解码矩阵。
- 对比 DDPM/Flow 的生成质量与采样耗时。

### M3：动态操作

- APPEND、EDIT、STOP 的边界和缓存语义正确。
- EDIT 相比无 EDIT 基线在预设错误修复任务上具有统计显著增益。

### M4：规模实验

- 先完成 S1/S2 多随机种子消融；验证有效后进入 S3。
- S4/S5 的启动以相同计算预算下是否取得真实增益为门槛。

---

## 15. 需要尽早确定的关键决策

1. **潜在矩阵表达什么？** 答案、推理轨迹、结构化状态，还是其混合物？建议 S1 对比三种方案，而不是事先认定一种最优。
2. **是否必须严格 64 槽可见？** 初期坚持，以避免 Decoder 偷看整个矩阵；然后与全矩阵对照检验代价。
3. **EDIT 起点是否可以任意？** 先对齐块边界，再升级滑动区间。
4. **用 DDPM 还是 Flow？** 均实现轻量版本，依据实际采样步数、质量与成本决定。
5. **是否保留 Input Encoder 条件供 Decoder 直接读取？** 第一版建议不提供，以检查矩阵信息瓶颈；后续作为能力与可靠性对照单独加入。
6. **是否复用预训练 Decoder？** 若要开放领域中文问答，建议增设预训练 Decoder 对照实验；从零训练适合机制验证，但不保证语言能力。

---

## 16. 相关工作与差异定位

- **Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models（ICLR 2025）**：块级扩散、可变长度序列生成；与本方案区别在于其研究重点是离散语言变量的块级生成，而本方案提出可重写的连续内部工作矩阵和局部 AR 读取。  
  https://proceedings.iclr.cc/paper_files/paper/2025/hash/7ede97c3e082c6df10a8d6103a2eebd2-Abstract-Conference.html
- **Latent Diffusion for Language Generation（NeurIPS 2023）**：训练语言 Autoencoder 后在连续潜空间进行扩散，并由预训练 Decoder 输出文本，为分阶段编解码与潜在扩散提供基础。  
  https://proceedings.neurips.cc/paper_files/paper/2023/hash/b2a2bd5d5051ff6af52e1ef60aefd255-Abstract-Conference.html
- **LaDiR: Latent Diffusion Enhances LLMs for Text Reasoning（ICLR 2026）**：将推理步骤编码为潜在块并进行扩散细化，是本研究非常接近的先例。本项目需要通过随机访问 EDIT、动态矩阵管理和局部可控输出证明独立增益。  
  https://proceedings.iclr.cc/paper_files/paper/2026/hash/843e1d8ab4819c8c8a4221fce2c8d455-Abstract-Conference.html

**新颖性声明的边界：**“块扩散”“潜在扩散”“动态长度”或“编辑已有结果”单独都不是新概念。论文贡献应落到具体的联合机制、训练算法、可验证的修订策略以及与现有基线相比的实证收益。

---

## 17. 建议的最小可行研究结论标准

以下条件同时成立时，才可初步认为 DBDM 提供了独立研究价值：

1. 在共同的样本与推理计算预算约束下，带 EDIT 的动态潜在状态模型优于不允许 EDIT 的匹配基线。
2. 回答详略可控，并且不同详略答案的核心事实与结论保持一致。
3. Decoder 确实使用 64 槽窗口，窗口设定与矩阵内容改变会产生可以解释的影响。
4. 额外块数与修订轮数的收益在跨任务、多随机种子条件下可重复。
5. 模型能报告完整的延迟、FLOPs、显存和失败案例，而不是只报告答案准确率。

**最终研究问题：**动态增长且可随机位置重写的连续潜在工作空间，能否在严格的计算预算下，提供比一次性连续生成或纯自回归推理更好的组合推理与可控语言输出？
