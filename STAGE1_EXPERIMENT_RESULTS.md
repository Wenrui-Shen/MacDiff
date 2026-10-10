# Stage1 实验汇总

更新日期：2026-10-10；10月7日汇总保留为历史快照。数据集为NTU60 XSub，评价为只输入骨架的linear probe。汇总所有本地已保存的Stage1数值成绩、定性反馈、未完成评价和计划；保留 [总审计](D:/program/MacDiff/EXPERIMENT_RESULTS.md) 的实验ID。

2026-10-07快照有13组数值成绩，其中4组有完整PT400+LP100附件，1组有完整PT但LP只有口述，8组主要保存历史成绩摘要。另有9项Stage1状态记录，不能统一算作已完成的独立训练。当前最高LP best为T12的86.196021%；S→T=0.5尚无回传结果。

`T→S`为文本到骨架任务权重，`S→T`为骨架到文本任务权重。`H/norm`是S→T decoder的隐藏维度/末端归一化，与CLIP缓存维度不同。`share=True`共享原生骨架重建与T→S骨架decoder主体；S→T文本decoder独立。

## 2026-10-10：decoder3结果与下一组（当前状态）

新增 **T15**，当前共15组原版/文本Stage1数值记录；下文10月7日的13组统计及10月9日的14组统计均为历史快照。

| ID | 实验身份 | LP成绩 | 证据与缺项 |
|---|---|---:|---|
| T15 | 当前七句v3/sample_target_blend/no-share/A；native/T→S各3层，S→T仍5层/512/none；权重1/0.1，PT不旋转（结构身份按上一会话上下文登记） | **85.84%，用户口述** | 2026-10-10用户反馈；完整PT/LP日志、分数统计口径（best/末轮）、对应epoch、last/末20、最终实际batch/accum、checkpoint/args及服务器SHA未核验 |

T15是文本no-share实验；T14是原版A/decoder3/PT500、checkpoint-499 LP的85.704755%，两组分别保留。此前目标库FileExistsError属于历史尝试，不能再据此说当前decoder3没有训练成绩；其处理方式和最终运行参数尚未核验。

**下一组：用户自己仅将text_decoder_depth 5→3，骨架decoder_depth保持3，以T15为控制。**S→T hidden512/output_norm=none及其他实际参数沿用控制组；PT随机旋转延后，下次再做，不叠加到本组。新组目前无回传成绩；本次助手只更新文档，没有改配置/模型或启动训练。详情见 [handoff最新状态](D:/program/MacDiff/handoff.md:5) 和 [当前实验计划](D:/program/MacDiff/tools/vlm_pilot/STAGE1_TEXT_EXPERIMENT_PLAN.md:3)。

## 1. 全部已保存的数值成绩

分数单位为%。除约数及口述值外，近期分数由原始LP日志重新统计。早期分数沿用历史摘要，未用当前配置补填缺失的实际参数。

| ID | 实验与目标机制 | H/norm | share | T→S | S→T | LP best | 证据完整性 |
|---|---|---|---|---:|---:|---:|---|
| T01 | 原版MacDiff历史基线，checkpoint-399 | — | — | — | — | 约85.86 | 历史摘要；实际batch/LR等不齐 |
| T02 | 旧BPE，第一版learned remap；256维在线动态目标 | 256/LN | 未核准 | 未核准 | 1 | 83.03 | 历史摘要；旧实际配置不齐 |
| T03 | 旧BPE，512维fixed CLIP，原L2尺度 | 256/LN | — | 0 | 1 | 84.64 | 历史摘要，无完整PT/LP日志 |
| T04 | 旧BPE，512维fixed CLIP，原L2尺度 | 256/LN | — | 0 | 0.1 | 85.95 | 历史摘要，无完整PT/LP日志 |
| T05 | 旧BPE，512维fixed CLIP，RMS尺度 | 256/LN | — | 0 | 1 | 约83.7 | 历史摘要，精确分数也缺 |
| T06 | 旧BPE，512维fixed CLIP，RMS尺度 | 256/LN | — | 0 | 0.1 | 85.82 | 历史摘要；LP有效batch64 |
| T07 | 旧BPE，512维RMS目标，参数EMA remap，momentum0.999 | 256/LN | False | 0.1 | 0.1 | 85.77 | 历史摘要及checkpoint诊断；LP有效batch128 |
| T08 | 旧BPE，512维RMS目标，参数EMA remap，momentum0.999 | 256/LN | True | 0.1 | 0.1 | 85.78 | 历史摘要及checkpoint诊断；LP有效batch128 |
| T09 | 新七句v3，sample_target_blend，骨架归一化A | 256/LN | True | 0.1 | 0.1 | 85.801795 | 完整PT400+LP100，best epoch87 |
| T10 | 新七句v3，sample_target_blend，骨架归一化A | 256/LN | True | 1 | 0.1 | 85.892771 | 完整PT400+LP100，best epoch88 |
| T11 | 新七句v3，sample_target_blend，骨架归一化A | 512/none | True | 1 | 0.1 | 85.79，口述 | 完整PT400，完整LP日志缺失 |
| T12 | 新七句v3，sample_target_blend，骨架归一化A | 512/none | False | 1 | 0.1 | **86.196021** | 完整PT400+LP100，best epoch75 |
| T13 | 原版MacDiff，骨架归一化B；PT/LP每卡batch128 | — | — | — | — | 85.183164 | 完整PT400+LP100，best epoch92 |

T03/T04、T05/T06记录的区别为S→T权重；原始args缺失，不能保证其余实际运行设置完全一致。T01–T08的归一化、seed、每卡batch/累积/卡数、LR日程等未知项不按当前YAML补齐。

T07/T08对remap网络参数做EMA。T09–T12则保存每个样本的目标，在optimizer step后更新为 `0.9*旧样本目标 + 0.1*当前在线remap输出`，不属于同一目标机制。T02的在线目标本身是256维，不能套用后续隐藏256预测512维噪声的结构分析。

T11曾记录85.75，用户已更正为85.79；这里只使用85.79。T09/T10当时的H256/LN不能从现已改为H512/none的sentence YAML反推。

## 2. 完整LP日志的统一统计

best epoch使用日志中的零基编号，末轮为epoch99，末20为epoch80–99的算术均值。

| ID | LP best | best epoch | 末轮 | 末20均值 | 末20总体标准差 |
|---|---:|---:|---:|---:|---:|
| T09 | 85.801795 | 87 | 85.547065 | 85.639556 | 0.070856 |
| T10 | 85.892771 | 88 | 85.619845 | 85.705361 | 0.073005 |
| T11 | 85.79，口述 | 未提供 | 未提供 | 未提供 | 未提供 |
| T12 | **86.196021** | 75 | 85.892771 | **86.000728** | 0.061392 |
| T13 | 85.183164 | 92 | 85.037603 | 85.110990 | 0.043181 |

这些标准差描述一次运行内的轮间波动，不是多seed结果。四组完整LP日志的epoch90–99学习率为0；末20均值仍包含不同后期阶段。

## 3. 实际运行口径与比较边界

| ID/组别 | 已保存的运行口径 | 尚未核准的部分 |
|---|---|---|
| T01–T08 | 历史成绩及目标/结构摘要；T06 LP有效batch64，T07/T08 LP有效batch128 | 原始PT/LP日志与完整实参；不能按当前默认值补齐 |
| T09 | PT400、LP100；LP日程与有效batch128一致 | 首组实际每卡batch命令缺失 |
| T10–T12 | 保存的文本组命令：双卡，PT每卡32/累积2；LP每卡64/累积1；有效batch128；PT400、LP100、LP lr0.1、seed0 | checkpoint actual args及服务器代码SHA；T11完整LP日志 |
| T13 | 双卡，PT/LP每卡128/累积1，有效batch256；PT400、LP100、LP lr0.1、seed0；PT lr0.001、min_lr0.0005 | checkpoint actual args及服务器代码SHA；同口径原版A对照结果 |

T10–T12结构/权重身份按用户说明和保存命令登记，数值日志没有直接打印share、H/norm或输入归一化。文本组PT min_lr为0.00001，与T13不同。

- T09→T10：记录的变化为T→S从0.1到1；LP best增加0.0910个百分点，末20均值增加0.0658个百分点。
- T10→T11：同时扩宽文本decoder并移除末端LN；best由85.892771到口述85.79，不能拆分两个结构变化的影响。
- T11→T12：记录的变化为share=True到False；best提高约0.4060个百分点。T11缺完整LP日志，当前是一次运行的初步证据。
- T10→T12还涉及H/norm与share同时变化，不能把全部0.3033个百分点归于no-share。
- T13与文本组在模型、归一化、batch和PT LR日程上都有区别；85.183164不能单独证明B相对A的收益或损失。

## 4. 没有完整最终数值的Stage1记录

| ID | 实验/评价 | 现有记录与状态 | 仍缺什么 |
|---|---|---|---|
| P01 | 新七句shared512/A，T→S=1、S→T=1 | 用户已反馈LP明显下降 | 精确best、PT/LP日志 |
| P02 | 新七句noshare512/A，T→S=1、S→T=0.5 | 当前选择；PT/LP命令已保存 | 尚无回传结果或可核实进度 |
| P03 | 新七句noshare512，S→T=0.2 | st02配置与建议保存；用户选择0.5替代 | 未确认实际跑过，不计完成实验 |
| P04 | T11的checkpoint-150 LP | 首轮“72多”，用户决定不继续；checkpoint-399首轮74.65也仅口述 | 中止评价，没有最终best |
| P05 | 旧BPE sample_target_blend shared/H256，T→S/S→T=0.1/0.1 | 原始PT0–19；后来的ckpt130恢复命令指向shared，首次恢复失败 | 最终PT/LP、修正后恢复是否成功 |
| P06 | 旧BPE sample_target_blend noshare/H256 | 配置和建议命令 | 无实际运行/结果证据 |
| P07 | 新七句fixed CLIP RMS/H512/none，S→T=0.1 | 配置和smoke/PT/LP示例 | 无实际运行/结果证据，不能套用旧BPE的85.82 |
| P09 | Stage1 OSE peer/cross-instance diffusion | 设计、配置和README用法 | 无实际训练/LP结果证据 |
| P10 | 原版A，匹配T13的有效batch256和LR日程 | 已指出需要该对照 | 未确认运行，无结果 |

P02沿用 `pretrain_madiff_text_sentence_sample_target_blend_noshare_st02.yaml`，但CLI明确覆盖 `lambda_skeleton_to_text=0.5`；实际实验登记为0.5，输出目录后缀为 `_s2t05`，不能按文件名登记为0.2。准确命令见 [实验计划](D:/program/MacDiff/tools/vlm_pilot/STAGE1_TEXT_EXPERIMENT_PLAN.md:3)。

NTU60 XView、NTU120、PKU配置及v/区域bias/1024/部分共享等讨论，没有已跑证据时不填入成绩表。已有Stage1 checkpoint的几何、条件打乱和one-shot/readout诊断也不增加预训练实验次数。

## 5. 日志与历史来源

| ID | 原始PT | 原始LP |
|---|---|---|
| T09 | [22235368](C:/Users/97537/.codex/attachments/22235368-fb22-4b3a-b4bb-9a3e2de686df/已粘贴的文本.txt) | [d2e7f9e1](C:/Users/97537/.codex/attachments/d2e7f9e1-4468-49bf-8cc1-7d9a30f822c4/已粘贴的文本.txt) |
| T10 | [81d248b4](C:/Users/97537/.codex/attachments/81d248b4-c708-4b84-ac44-3cf027e8f732/已粘贴的文本.txt) | [6cc77fa4](C:/Users/97537/.codex/attachments/6cc77fa4-58fa-4466-9fa3-8afb35f169f9/已粘贴的文本.txt) |
| T11 | [64857d73](C:/Users/97537/.codex/attachments/64857d73-03e2-4bf9-97ed-e6ca4fba0355/已粘贴的文本.txt) | 缺完整日志，口述85.79 |
| T12 | [02d00a0e](C:/Users/97537/.codex/attachments/02d00a0e-de94-42cb-98f4-6ad1cc73af47/已粘贴的文本.txt) | [cc1abd5f](C:/Users/97537/.codex/attachments/cc1abd5f-c4d1-481b-8670-badf284c8140/已粘贴的文本.txt) |
| T13 | [826649bf](C:/Users/97537/.codex/attachments/826649bf-4dae-4569-b341-3e1581c0cf92/已粘贴的文本.txt) | [4cb61fd1](C:/Users/97537/.codex/attachments/4cb61fd1-2294-42bf-82dd-a231581f3661/已粘贴的文本.txt) |

附件仍在本机Codex attachments目录，未复制进仓库。PT/LP附件覆盖、SHA256及完整性检查见 [审计JSON](D:/program/MacDiff/handoff_artifacts/experiment_results_audit_20261007.json)。

- T01–T08：[早期成绩表](D:/program/MacDiff/handoff_artifacts/handoff_before_20261001_refresh.md:40)；T07/T08的参数EMA机制与诊断在同文第3节。
- T09/T10：[2026-10-04归档对照](D:/program/MacDiff/handoff_artifacts/handoff_before_20261004_session_close.md:26)。
- T11：[512训练分析](D:/program/MacDiff/tools/vlm_pilot/ST512_TRAINING_LOG_ANALYSIS.md)。
- T12/T13：[noshare与NormB分析](D:/program/MacDiff/tools/vlm_pilot/NOSHARE_NORMB_LOG_ANALYSIS.md)及 [精确统计](D:/program/MacDiff/handoff_artifacts/noshare_normb_20261007/summary.json)。
- P05：[原始前20轮日志](C:/Users/97537/.codex/attachments/d0c2985f-c3a5-4814-a2e2-5ba0fc9ef962/已粘贴的文本.txt)。
- 旧EMA诊断：[d027fa4a JSON](C:/Users/97537/.codex/attachments/d027fa4a-f322-41be-9472-8ebf4c422cfa/已粘贴的文本.txt)，协议 `ema_bidirectional_targeted_diagnostic_v1`；诊断batch8/seed20260927不当作预训练或LP参数。
- [Stage1文本几何统计](D:/program/MacDiff/handoff_artifacts/stage1_text_geometry_summary.json)和 [读出比较协议](D:/program/MacDiff/STAGE1_READOUT_COMPARISON.md)另存checkpoint诊断，不混入LP成绩。

本次只汇总记录，没有启动训练或改变实验参数。


## 2026-10-09：最新Stage1实验与下一组

新增 **T14**，目前共14组原版/文本Stage1数值记录（此前13组结论为10月7日快照）。本组是native MacDiff，A与decoder3按用户说明，PT500；用户已更正实际LP加载checkpoint-499，LP100。

| ID | PT / LP实际命令 | best | best epoch | last | 末20 mean ± population SD |
|---|---|---:|---:|---:|---:|
| T14 | 双卡PT64×accum1=128；LP128×accum1=256；seed0；PT min_lr由日志确认5e-4 | 85.704755 | 94 | 85.595585 | 85.635917 ± 0.038372 |

不能把本组登记成“仅decoder5→3”的纯消融，或误写成LP399；epoch/LP组织等也与历史对照不同。PT/LP同目录，部分早期PT checkpoint被LP覆盖，但399/499不在覆盖范围。完整依据见 [T14分析](D:/program/MacDiff/handoff_artifacts/decoder3_20261009/README.md) 与 [命令](D:/program/MacDiff/handoff_artifacts/decoder3_20261009/commands.json)。

10月9日已准备以T12为控制的骨架decoder3配置/命令，S→T仍5层；10月10日用户已回传该组口述85.84%，登记为T15，当前下一组见顶部最新状态。此前1/.5计划仍没有回传结果，不将其叠加到decoder3本组。原始日志来源是 [PT500+LP100合并附件](C:/Users/97537/.codex/attachments/f6643024-0636-4677-ab2b-60b92d4ec967/已粘贴的文本.txt)。


### 2026-10-09追加待实验：PT随机旋转

用户已选择在当前no-share decoder3无旋转控制之后，单独开启train_feeder_args.random_rot=True，其余实际PT/LP参数固定；不改source_rot、flip、cache、权重或深度。此项只是计划，无成绩、不增加已完成实验数；详情见 [最新实验计划](D:/program/MacDiff/tools/vlm_pilot/STAGE1_TEXT_EXPERIMENT_PLAN.md:3)。10月9日decoder3曾因已有目标库描述文件报FileExistsError；10月10日已回传T15口述成绩，不能再把该失败当当前阻塞；尝试命令为PT64/accum1，原准备命令32/accum2，后续消融须匹配最终成功控制的实际组织。
