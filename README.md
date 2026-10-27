# DBDM

**Dynamic Block Diffusion Model（动态矩阵块扩散推理模型）**：研究在动态增长、可随机位置重写的连续潜在矩阵中完成内部推理，再通过 64 槽局部窗口自回归解码为文本。

> 当前状态：研究设计与实验计划，尚无模型实现或验证结果。参数规模与性能均为设计目标。

## 文档

从 [docs 文档索引](docs/README.md)开始阅读。

- [完整技术设计与实验计划](docs/technical-design.md)
- [架构与状态语义](docs/architecture.md)
- [分阶段训练](docs/training.md)
- [实验与评测规范](docs/experiments.md)
- [工程实施指南](docs/engineering.md)
- [里程碑与验收](docs/roadmap.md)
- [设计决策记录](docs/decisions.md)
- [实验记录模板](docs/templates/experiment-record.md)

## 研究重点

1. 通过 APPEND 追加 64 槽，通过 EDIT 修复已有 64 槽区间，通过 STOP 结束内部推理。
2. 比较条件块扩散与 Flow Matching，并验证目标答案不可见时的生成能力。
3. 用严格局部读取的 AR Decoder 输出文本，通过步长、详细度和长度上限联合控制详略。
4. 在训练与推理计算预算匹配的条件下，检查动态状态和 EDIT 的实际收益。

计划先验证 S1/S2，再依据可重复的实验结果推进更大规模。详见[里程碑](docs/roadmap.md)。
