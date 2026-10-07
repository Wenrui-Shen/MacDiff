# No-share 与原版归一化 B 日志分析（2026-10-07）

本次两组均已完成 400 epoch 预训练和 100 epoch LP。No-share 的 LP best 为 **86.1960%**，最后 20 轮均值 **86.0007%**；原版 B 的 best 为 **85.1832%**，最后 20 轮均值 **85.1110%**。No-share 相对当前 shared512 的用户口述 best 85.79% 提高 **0.4060 个百分点**，是值得保留并复验的正向结果。原版 B 这一整套设置尚未显示收益，当前不建议直接把 B 加到 no-share。

## 来源、实验定义与完整性

| 日志 | 用户附件 |
|---|---|
| No-share 预训练 | [02d00a0e](C:/Users/97537/.codex/attachments/02d00a0e-de94-42cb-98f4-6ad1cc73af47/已粘贴的文本.txt) |
| No-share LP | [cc1abd5f](C:/Users/97537/.codex/attachments/cc1abd5f-c4d1-481b-8670-badf284c8140/已粘贴的文本.txt) |
| 原版 B 预训练 | [826649bf](C:/Users/97537/.codex/attachments/826649bf-4dae-4569-b341-3e1581c0cf92/已粘贴的文本.txt) |
| 原版 B LP | [4cb61fd1](C:/Users/97537/.codex/attachments/4cb61fd1-2294-42bf-82dd-a231581f3661/已粘贴的文本.txt) |
| 历史 shared512 预训练 | [64857d73](C:/Users/97537/.codex/attachments/64857d73-03e2-4bf9-97ed-e6ca4fba0355/已粘贴的文本.txt) |

五份文件均为逐行 JSON；各 PT 恰有 400 条 epoch 0–399，各 LP 恰有 100 条 epoch 0–99。没有解析错误、重复、缺失 epoch 或数值 NaN/Inf。有限的 epoch 均值不能确认逐 step 没有 AMP 跳步。所有 epoch 编号从 0 开始。

实验身份与 batch 依据用户说明、此前提供命令和本地配置；这些 JSON 不含服务器 Git SHA、checkpoint args、share 开关或归一化参数，不能独立确认所有实际运行设置。

| 设置 | No-share 文本版 | 原版 MacDiff B |
|---|---|---|
| 骨架归一化 | A | B（原配置注释的 new） |
| 文本任务 | T→S=1，S→T=0.1，512/无末端输出 LN | 原生骨架预训练 |
| PT 每卡 batch / 累积 / 双卡有效 batch | 32 / 2 / 128 | 128 / 1 / 256 |
| LP 每卡 batch / 累积 / 双卡有效 batch | 64 / 1 / 128 | 128 / 1 / 256 |
| PT lr / min_lr | 0.001 / 0.00001 | 0.001 / 0.0005 |
| 预训练 / LP | 400 / 100 epoch；LP lr=0.1 | 400 / 100 epoch；LP lr=0.1 |

A：mean=[-0.0058,-0.1333,-0.0246]，var=[0.0206,0.0805,0.0218]。
B：mean=[-0.0024,-0.2132,-0.0446]，var=[0.0525,0.1527,0.0513]。

## LP：No-share 的提升不只出现在单轮峰值

| 指标 | No-share | 原版 B |
|---|---:|---:|
| Best Top-1 | **86.196021% @75** | **85.183164% @92** |
| 最后 Top-1 | 85.892771% | 85.037603% |
| 最后 10 轮 Top-1 均值 ± SD | 85.989811 ± 0.064983% | 85.108564 ± 0.042546% |
| 最后 20 轮 Top-1 均值 ± SD | 86.000728 ± 0.061392% | 85.110990 ± 0.043181% |
| Best Top-5（独立取最大值） | 97.640708% @95 | 97.440563% @91 |
| 最后 train CE | 0.528918 | 0.210226 |
| 最后 test CE | 4.493405 | 2.329487 |

表中 SD 是同一次运行中各 epoch 的总体标准差，单位为百分点，不是独立训练重复的标准误或置信区间。两组 LP 的 n_parameters 均为 384060，与 6400×60+60 的 Linear 头一致。

No-share best 比 shared512 用户口述 best 85.79% 高 0.406021 个百分点；最后 20 轮集中在 86.00% 附近，说明它自身后期没有跌回很低的水平。不过旧 shared512 没有完整 LP 日志，不能比较它们的末 20 轮均值或置信度，也不能据单个 seed 宣称可重复、统计显著的提升。

No-share 与本次原版 B 的 best 相差 1.012858 个百分点，末 20 轮均值相差 0.889738 个百分点。这是两套实验的观测差距，不能作为文本任务的纯增益：模型、归一化、batch、优化器更新次数与 LR 日程均有差别。

## No-share 预训练：S→T 有改善，后期回升仍在

全 400 轮总 loss 满足：

```text
native + 0.02*skel_uniformity + 0.02*text_uniformity + 1*T→S + 0.1*S→T
```

最大核算残差 6.918e-9，确认两条任务权重 1/0.1 生效。所有 epoch 有效 local 数均为 6；目标库 epoch0 覆盖 40064 个样本，此后为 40091。No-share 与 shared512 的逐 epoch LR 完全一致。

| 最后 50 轮均值 | Shared512 | No-share512 | No-share 相对变化 |
|---|---:|---:|---:|
| 原生噪声 MSE | 0.01435490 | 0.01444116 | +0.60% |
| T→S MSE | 0.01989607 | 0.01986733 | −0.14% |
| S→T MSE | 0.02710436 | 0.02045860 | **−24.52%** |
| 总 loss | 0.04063979 | 0.04002396 | −1.52% |

原生 MSE没有降低，T→S 几乎相同，因此 LP 增益不能简单解释为骨架重建更准。S→T 后期拟合有明显改善，与更好的 LP 同时出现，但不是对因果机制的证明。No-share 增加独立 decoder 参数，并仍保留 T→S→remap→目标库→S→T→encoder 的间接路径；结果支持这套拆分设计，尚不能单独证明共享梯度冲突。

| S→T 阶段均值 | Shared512 | No-share512 |
|---|---:|---:|
| 100–149 | 0.010192 | 0.009197 |
| 150–199 | 0.010623 | 0.009272 |
| 200–249 | 0.012662 | 0.010555 |
| 250–299 | 0.017566 | 0.013209 |
| 300–349 | 0.023785 | 0.017453 |
| 350–399 | 0.027104 | 0.020459 |

No-share 的最低 S→T=0.008858924 在 epoch142；末轮=0.021208357，是最低的 2.394 倍。Shared512 最低=0.009960991 @142，末轮=0.027889738，是最低的 2.800 倍。拆分减轻了回升，但没有消除这一现象，不能称动态目标后期问题已解决。

No-share 最后 50 轮 text energy=0.999738，global+local variance=0.925459；shared 分别为 0.999535/0.936355。Variance 稍低但 LP 更高，再次说明合并 variance 不能直接用来排序表征收益。

本次 no-share 累计 epoch 时间 22.986h，shared512 为 22.468h；中位数 206.58s/202.02s，allocated peak 17725.14/17615.01 MiB。差异约 0.518h、110.13 MiB，服务器负载和运行环境未控制，不能当严格 benchmark。原版 B 日志不含 epoch_seconds，无法从这些 JSON 比较其耗时。

## 原版 B：训练正常收敛，目前没有采纳依据

PT 总 loss 从 0.977344 下降到 0.010089，最低 0.009821 @386。最后 50 轮均值=0.010056，300–349 均值=0.010142，已经接近平台。没有持续发散、NaN 或训练中断证据。

原版 total=原生噪声 MSE + 0.02×骨架 uniformity；这份 JSON 没有记录分项。不能直接用它的 0.0101 与文本版总 loss 0.0400 比较：文本版包含额外任务，且 B 会改变骨架信号尺度及扩散 SNR。更低预训练 loss 不能代替 LP。

LP 在 60–79 轮均值约 85.048%，80–99 轮约 85.111%，后期稳定在 85.1%。相对历史原版口述约 85.86%，本次 best 低约 0.677 个百分点；这套 B+当前 batch 协议没有超过历史结果。但是没有同协议原版 A 对照，不能把下降单独归因于 B。

## CE 与 LP 后期波动的解释边界

No-share 的 test CE 虽高，Top-1 更好：峰值 CE=6.390759 @17，最低=4.417592 @84，末轮=4.493405。原版 B 末轮 CE=2.329487。CE 看真实类别的概率，Top-1 只看最大 logit 的类别；错误预测的高置信度可能使 CE 很高，故二者可以共存。没有逐样本 logits，不能据此确认校准、BN 或增强分布哪一项是主因。

当前本地 evaluate 使用标准 CrossEntropyLoss，并按样本数加权、跨 rank 汇总；没有看到求和或漏除 batch 的静态证据。不能据本地代码保证服务器 SHA 完全相同。训练使用随机旋转和 [0.5,1] crop，验证使用无旋转、[0.95] crop；train/test CE 并非相同输入分布。

两组 LP 的 epoch90–99 均 lr=0，符合最后 10 轮保持 min_lr=0 的日程。Head 中 BatchNorm 在每轮 train 模式下仍更新 running statistics，因此即使 FC 权重停止梯度更新，测试结果仍可小幅变化。原版 B 的 epoch92 best 不能解释为该轮有新的梯度优化收益。

## 之后的实验选择

1. 保留 A/T→S=1/S→T=0.1/512/no-share 作为当前工作基线；先对它与 shared 的收益做有限重复确认，不用一次 best 宣称最终突破。
2. B 暂不叠加到 no-share。若要判断 B 本身的效果，最直接缺失的对照是原版 A、相同每卡128/accum1、相同 PT lr/min_lr 和 LP 协议；这轮只变 mean/var。
3. 若只复跑 LP seed，可检查分类头的随机性，不能替代额外预训练 seed。固定 CLIP、shuffle/geometry、150 LP 等旧方案不在本轮重新安排。

计算脚本与全部精确统计：
- [四面板对照曲线 PNG](D:/program/MacDiff/handoff_artifacts/noshare_normb_20261007/comparison.png) / [SVG](D:/program/MacDiff/handoff_artifacts/noshare_normb_20261007/comparison.svg)
- [analyze_logs.py](D:/program/MacDiff/handoff_artifacts/noshare_normb_20261007/analyze_logs.py)
- [summary.json](D:/program/MacDiff/handoff_artifacts/noshare_normb_20261007/summary.json)

未修改训练模型、实验配置、缓存或服务器进程。
