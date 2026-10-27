# 工程实施指南

> 状态：建议接口与开发清单，建档时仓库为空。以下文件路径是规划，不能视为已实现模块；来源为[完整技术设计](technical-design.md)第 5、8、11–13 节。

## 建议模块布局

| 目录 | 规划文件 | 职责 |
| --- | --- | --- |
| `configs/` | `s1.yaml`–`s5.yaml` | 模型尺寸、采样、预算、输出与训练配置 |
| `models/` | `input_encoder.py`、`latent_encoder.py` | 输入条件与训练目标编码 |
| `models/` | `block_memory.py`、`action_controller.py` | 块状态、摘要、动作与位置选择 |
| `models/` | `block_denoiser.py` | DDPM／Flow 的条件块生成与修复 |
| `models/` | `window_reader.py`、`ar_decoder.py` | 64 槽读取、WRITE/ADVANCE 与文本输出 |
| `training/` | `stage1_autoencoder.py`、`stage1_input_encoder.py` | 阶段 1A／1B，后续补充接口对齐入口 |
| `training/` | `stage2_diffusion.py`、`stage2_edit.py`、`stage2_controller.py` | 阶段 2 的生成、修复与控制器 |
| `training/` | `stage3_joint.py` | 有限端到端调整 |
| `data/` | `synthetic_reasoning.py`、`text_multigranularity.py`、`corrupted_states.py` | 可校验任务、详略配对与错误状态 |
| `eval/` | `reconstruction.py`、`reasoning_accuracy.py`、`edit_success.py` | 重构、条件任务与修复评测 |
| `eval/` | `window_coverage.py`、`compute_profile.py` | 窗口行为、质量与成本关联 |
| `scripts/` | `count_params.py`、`profile_memory.py`、`run_ablation.py` | 真实规模核算、显存测量与实验调度 |
| `tests/` | 形状、状态操作、窗口 mask、目标隔离、缓存测试 | 验证关键不变量 |

## 建议接口契约

| 组件 | 输入 | 输出与约束 |
| --- | --- | --- |
| Input Encoder | 输入 Token 与 mask | `C`、条件 mask；不需要目标答案 |
| Latent Encoder | 训练目标、目标类型、长度规则 | `Z*`、块数、有效 mask；标记为监督路径 |
| Block Memory | 工作矩阵、样本有效长度、摘要和版本 | 受约束的 append/edit；维护摘要失效范围 |
| Controller | `C`、当前状态、剩余预算 | 动作、编辑起点、必要的概率或置信记录 |
| Block Updater | 噪声窗口、时间、`C`、状态摘要、操作、位置；EDIT 加旧窗口 | 严格 64 槽更新窗口；区间外不能被隐式改写 |
| Window Reader | 最终矩阵、有效 mask、当前位置、步长 | 64 槽窗口与 mask、下一位置、是否到末端 |
| AR Decoder | 当前窗口、Token 历史、详细度、长度上限 | WRITE／ADVANCE／EOS 与 Token；记录实际访问轨迹 |

内存状态建议携带样本级 `valid_slots`、`matrix_version` 和摘要有效标记。跨 batch 操作按样本应用；某样本 STOP 不应让仍在运行的其他样本停止。

## 配置契约

| 配置组 | 需要显式记录的字段 |
| --- | --- |
| 潜在空间 | `block_size=64`、`d_z`、目标类型、初始块策略、最大块数、归一化 |
| 模型 | tokenizer／词表、各组件层数、隐藏维、FFN、头数、GQA、共享策略 |
| 块更新 | `ddpm` 或 `flow`、噪声／路径调度、求解器、训练时刻采样、推理步数 |
| 控制器 | 最大动作数、成本预算、EDIT 对齐方式、非法动作 mask、STOP 与强制终止规则 |
| 读取与输出 | 64 槽宽度、步长、详细度、最大输出 Token、固定周期或学习推进、读取动作上限 |
| 训练 | 阶段、冻结模块、损失权重、优化器、学习率、精度、batch、梯度累积与随机种子 |
| 数据与评测 | 数据版本、划分规则、校验器、指标版本、硬件和延迟测量边界 |

字段名为建议，不代表现有配置解析器支持。示例中的规模和采样步数不能绕过参数计数、显存 profiling 或阶段验收。运行命令应在真实脚本完成并验证后补充，本文不提供未经验证的安装或训练命令。

## 五档名义参数规模

| 档位 | `d_z` | `E_in` | `E_lat` | AR Decoder | Diffusion | 完整总量 | 阶段 1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S1 | 256 | 20M | 20M | 50M | 100M | 190M | 90M |
| S2 | 512 | 50M | 50M | 100M | 200M | 400M | 200M |
| S3 | 768 | 100M | 100M | 200M | 500M | 900M | 400M |
| S4 | 1024 | 200M | 200M | 500M | 1B | 1.9B | 900M |
| S5 | 1536 | 500M | 500M | 1B | 2B | 4B | 2B |

完整总量包含训练用 `E_lat`，不是推理时每一步的活跃参数量。表内尚未精确计入控制器、投影和其他小组件；实际报告需分别给出各组件、全模型去重总量和本阶段可训练参数量。共享权重只能在去重总量中计一次。

原稿层数与宽度表仅为结构搜索范围。词表、embedding、Cross-Attention 和 FFN 都会显著影响计数，特别是 S1；不得把粗略配置直接宣称为精确的 20M／100M 模型。

## 显存核算

按原稿的 BF16 + AdamW 全参数训练经验估算，每个训练参数约 16 bytes，仅计算参数、梯度和优化器基础状态：

| 档位 | 阶段 1 基础状态 | 阶段 2 Diffusion 基础状态 |
| --- | ---: | ---: |
| S1 | 约 1.44 GB | 约 1.6 GB |
| S2 | 约 3.2 GB | 约 3.2 GB |
| S3 | 约 6.4 GB | 约 8.0 GB |
| S4 | 约 14.4 GB | 约 16 GB |
| S5 | 约 32 GB | 约 32 GB |

GB 采用十进制口径。这不是最低 GPU 容量要求：未计激活、注意力空间、通信、缓存、冻结 Encoder／Decoder 驻留和额外训练组件。主权重、优化器与分片实现不同，实际 bytes／参数也不同，须用目标配置测量。

可按需采用 BF16、SDPA／FlashAttention、梯度检查点、长度分桶与动态 padding；大规模再评估 FSDP／ZeRO 等分片。离线目标 latent 能减少 `E_lat` 的在线成本，但需绑定 checkpoint 与数据版本。短轨迹反传与少步采样先实测，不预先宣称加速收益。

## 实现验证清单

以下为未来代码实现的检查要求，本次文档整理没有新增或运行模型测试。

- [ ] APPEND 恰好增加 64 槽，旧矩阵内容不变；EDIT 保持长度且仅改写目标区间；STOP 保持状态。
- [ ] 拒绝越界／负起点和非法步长；首版拒绝非 64 对齐 EDIT；空状态动作有明确规则。
- [ ] `n=64`、`n<64`、不同 batch 长度和不能整除步长的末窗口都正确处理。
- [ ] 64 槽 Cross-Attention mask 严格，padding 与窗口外槽位不可见。
- [ ] 无效末端 ADVANCE 和反复 EDIT 受预算限制，所有终止均有原因记录。
- [ ] EDIT 后受影响摘要与依赖缓存刷新；相同窗口起点但不同矩阵版本不复用旧 K/V。
- [ ] 缓存与无缓存输出在预定数值容差内一致；Decoder Token 历史保持因果。
- [ ] 推理入口不读取标签、目标 latent 或真实目标长度；教师条件模式单独命名。
- [ ] 参数计数区分冻结、训练、共享与总量；五档配置可实例化或明确说明资源限制。
- [ ] 固定种子和数据版本可复现，断点恢复不会重复或跳过未记录的数据。

实际项目接入前需先阅读仓库内 `AGENTS.md`、现有模块和测试规范，再决定路径映射，避免以本建议覆盖既有工程约定。
