# MacDiff 实验结果总表与保存完整性审计

审计日期：2026-10-07。范围为本地 `handoff.md`、归档交接、其他 Markdown、配置、已有 JSON 统计以及文档明确引用的附件。没有访问服务器；不能确认尚未回传的运行进度，也不能恢复从未写入这些材料的聊天内容。

结论：**主要已报告成绩有保存，但不同实验的配置、完整日志与结果没有全部保存齐。** 目前查到13组原版/文本 Stage1 数值成绩、6组 Stage2 数值摘要。Stage1中4组有仍可读取的完整PT400+LP100附件，1组有完整PT但LP仅口述，另外8组主要只有历史成绩摘要。Stage2六组均只有交接里的结果摘要，没有本地完整PT/LP日志或sweep CSV。

35份 YAML 是当前配置库存，不代表35次实际训练。同一个 YAML 的 CLI 权重、batch、代码版本或模型结构不同，仍需分开登记；重复上传日志也不增加实验次数。

原版与文本Stage1另有 [独立汇总](D:/program/MacDiff/STAGE1_EXPERIMENT_RESULTS.md)，包含13组成绩、完整LP统一统计及Stage1待结果/计划项。

## 1. 原版与文本 Stage1：已有数值结果

`H/norm` 是S→T decoder的隐藏维度/末端输出归一化，不是CLIP缓存维度。除注明外，以下数值在来源中按LP best记录。结构来自历史文档/配置，日志未包含实际model_args时不视为独立核实。

| ID | 区分实验的关键配置 | LP结果 | 已保存证据 | 仍缺什么 |
|---|---|---:|---|---|
| T01 | 原版MacDiff历史基线；checkpoint-399 | 约85.86% | Stage2归档及两份Stage1归档都有摘要 | 精确best、PT/LP原始日志、完整命令/实际batch与LR、seed、服务器SHA |
| T02 | 第一版双向learned remap；旧BPE、256维在线动态目标、H256/LN；S→T=1 | 83.03% | [早期成绩表](D:/program/MacDiff/handoff_artifacts/handoff_before_20261001_refresh.md:44)、旧文本协议 | 原始日志、准确旧配置快照；T→S权重/share/LP batch未核准 |
| T03 | fixed CLIP原L2尺度；旧BPE、512维冻结目标、H256/LN；T→S=0/S→T=1 | 84.64% | [早期成绩表](D:/program/MacDiff/handoff_artifacts/handoff_before_20261001_refresh.md:45) | 原始PT/LP日志、best epoch/末轮、实际运行参数 |
| T04 | 同T03，仅S→T=0.1 | 85.95% | [早期成绩表](D:/program/MacDiff/handoff_artifacts/handoff_before_20261001_refresh.md:46) | 同上，尤其实际LP batch |
| T05 | fixed CLIP RMS；旧BPE、512维冻结目标、H256/LN；T→S=0/S→T=1 | 约83.7% | [早期成绩表](D:/program/MacDiff/handoff_artifacts/handoff_before_20261001_refresh.md:47) | 更精确分数、原始日志、实际运行参数 |
| T06 | 同T05，仅S→T=0.1 | 85.82% | [早期成绩表](D:/program/MacDiff/handoff_artifacts/handoff_before_20261001_refresh.md:48)；历史LP有效batch64 | 原始PT/LP日志、完整实际命令；不能用当前默认补齐 |
| T07 | 旧参数EMA remap，momentum0.999；旧BPE、512 RMS目标、H256/LN；no-share；T→S/S→T=0.1/0.1 | 85.77% | [早期成绩表](D:/program/MacDiff/handoff_artifacts/handoff_before_20261001_refresh.md:49)及checkpoint-399诊断；历史LP有效batch128 | 原始训练/LP日志、best epoch/末轮与完整args；诊断参数不是LP参数 |
| T08 | 同T07，share=True | 85.78% | [早期成绩表](D:/program/MacDiff/handoff_artifacts/handoff_before_20261001_refresh.md:50)及checkpoint-399诊断；历史LP有效batch128 | 同上；文档保存训练摘要，不等于完整训练日志附件 |
| T09 | 新七句v3、sample_target_blend、A、share=True、H256/LN；T→S/S→T=0.1/0.1 | **85.801795% @LP87** | 完整PT400/LP100附件；[归档分析](D:/program/MacDiff/handoff_artifacts/handoff_before_20261004_session_close.md:26) | checkpoint实际args/SHA；首组每卡batch缺原始命令，LP日程与有效batch128一致 |
| T10 | 同T09，仅T→S=1，S→T=0.1 | **85.892771% @LP88** | 完整PT400/LP100附件；[归档对照](D:/program/MacDiff/handoff_artifacts/handoff_before_20261004_session_close.md:28) | checkpoint实际args/SHA；注意当前sentence YAML已经改成H512/none |
| T11 | 新七句v3、sample_target_blend、A、share=True、**H512/none**；T→S=1/S→T=0.1 | **85.79%，口述** | 完整PT400附件及[512分析](D:/program/MacDiff/tools/vlm_pilot/ST512_TRAINING_LOG_ANALYSIS.md) | **完整LP日志**、best epoch、末轮/末20与CE、actual args/SHA |
| T12 | 同T11，share=False；T→S=1/S→T=0.1；PT32×accum2、LP64×accum1、双卡有效batch128 | **86.196021% @LP75** | 完整PT400/LP100、[分析报告](D:/program/MacDiff/tools/vlm_pilot/NOSHARE_NORMB_LOG_ANALYSIS.md)、[精确统计](D:/program/MacDiff/handoff_artifacts/noshare_normb_20261007/summary.json) | actual args/SHA及独立重复；数值日志不直接打印share/归一化 |
| T13 | 原版MacDiff/B；PT/LP每卡128×accum1、双卡有效batch256；PT min_lr=5e-4 | **85.183164% @LP92** | 完整PT400/LP100及同一[分析报告](D:/program/MacDiff/tools/vlm_pilot/NOSHARE_NORMB_LOG_ANALYSIS.md)/统计 | actual args/SHA；同协议原版A控制组尚无结果 |

T09–T12的目标更新为 `0.9*旧样本目标 + 0.1*当前step后在线remap`。T07/T08是网络参数EMA，两者不能混为同一目标机制。旧“512 CLIP”也不等于新H512 decoder。

T12末轮85.892771%、末20均值86.000728%；T13末轮85.037603%、末20均值85.110990%。这些均值来自一次运行的epoch，不是重复训练的置信区间。T11的85.75已被用户更正为85.79，当前只使用85.79。

## 2. 可以重新读取的完整训练/LP日志

本轮逐项确认文件仍存在、epoch唯一连续、数值有限；附件路径、SHA256与重新统计的覆盖范围保存在 [审计明细](D:/program/MacDiff/handoff_artifacts/experiment_results_audit_20261007.json)。

| 对应ID | PT附件 | LP附件 | 实际覆盖 |
|---|---|---|---|
| T09 | [22235368](C:/Users/97537/.codex/attachments/22235368-fb22-4b3a-b4bb-9a3e2de686df/已粘贴的文本.txt) | [d2e7f9e1](C:/Users/97537/.codex/attachments/d2e7f9e1-4468-49bf-8cc1-7d9a30f822c4/已粘贴的文本.txt) | PT0–399，LP0–99 |
| T10 | [81d248b4](C:/Users/97537/.codex/attachments/81d248b4-c708-4b84-ac44-3cf027e8f732/已粘贴的文本.txt) | [6cc77fa4](C:/Users/97537/.codex/attachments/6cc77fa4-58fa-4466-9fa3-8afb35f169f9/已粘贴的文本.txt) | PT0–399，LP0–99 |
| T11 | [64857d73](C:/Users/97537/.codex/attachments/64857d73-03e2-4bf9-97ed-e6ca4fba0355/已粘贴的文本.txt) | 未找到；目前只有口述85.79 | PT0–399 |
| T12 | [02d00a0e](C:/Users/97537/.codex/attachments/02d00a0e-de94-42cb-98f4-6ad1cc73af47/已粘贴的文本.txt) | [cc1abd5f](C:/Users/97537/.codex/attachments/cc1abd5f-c4d1-481b-8670-badf284c8140/已粘贴的文本.txt) | PT0–399，LP0–99 |
| T13 | [826649bf](C:/Users/97537/.codex/attachments/826649bf-4dae-4569-b341-3e1581c0cf92/已粘贴的文本.txt) | [4cb61fd1](C:/Users/97537/.codex/attachments/4cb61fd1-2294-42bf-82dd-a231581f3661/已粘贴的文本.txt) | PT0–399，LP0–99 |

附件位于本机Codex attachments目录，不是仓库内的原始log副本；Markdown保存了结果和路径。新T12/T13另有仓库内精确JSON统计，T09/T10则有归档摘要。若需要把原始日志和代码一起归档，应另保存日志副本；本次没有声称这些外部附件已经纳入Git。

T10的fe23bb66/a64cb5b7是重复上传，不另计实验。T07/T08的[d027fa4a](C:/Users/97537/.codex/attachments/d027fa4a-f322-41be-9472-8ebf4c422cfa/已粘贴的文本.txt)是checkpoint-399几何/条件诊断，不是PT或LP日志。诊断batch8、seed20260927不能当训练参数。

## 3. Stage2：成绩已在归档保存，运行记录不完整

来源均为 [Stage2交接的结果摘要](D:/program/MacDiff/handoff_stage2_legacy_20260905.md:29)。低LR版本的历史协议为Joint-only K=2、SyncBN、无额外增强、mask0.9、6400维joint-aware特征、SGD backbone/head LR=0.001/0.25；最终实际args仍需服务器记录确认。

| ID | 不同实验配置 | 保存的LP数字 | 原文位置 | 缺项 |
|---|---|---:|---|---|
| S01 | 高LR ReSA+OSE，backbone/head LR均0.25 | Stage2 ckpt080最高77.49%；ckpt100为77.27% | [归档](D:/program/MacDiff/handoff_stage2_legacy_20260905.md:32) | 原始LP/sweep CSV；当时JMB/Joint-only、BN、增强版本未严格绑定 |
| S02 | 低LR ReSA+OSE，global-random mask；backbone/head LR=0.001/0.25 | 最终Stage2 checkpoint LP约85.22% | [归档](D:/program/MacDiff/handoff_stage2_legacy_20260905.md:34) | 全部checkpoint sweep结果、原始日志、LP best/末轮口径 |
| S03 | 同S02，关闭ReSA梯度，OSE及两个mixed权重仍1 | 85.02% | [归档](D:/program/MacDiff/handoff_stage2_legacy_20260905.md:407) | 原始PT/LP日志、实际args、LP epoch/末轮 |
| S04 | 同S02，仅teacher tau_t=0.04→0.06 | 85.12% | [归档](D:/program/MacDiff/handoff_stage2_legacy_20260905.md:431) | 同上 |
| S05 | ReSA-only；per-joint每关节保留3/30 token，总75；OSE三个权重为0 | 85.35% | [归档](D:/program/MacDiff/handoff_stage2_legacy_20260905.md:37) | 具体Stage2 checkpoint、原始LP日志、best epoch/末轮 |
| S06 | 同S05，启用OSE三个权重为1；tau_t=0.04 | 85.17% | [归档](D:/program/MacDiff/handoff_stage2_legacy_20260905.md:37) | 同上 |

公共YAML为 `pretrain_madiff_stage2.yaml`；低LR与消融由 `script_pretrain_stage2*.sh`/环境变量覆盖。当前YAML默认LR0.25不能反填低LR实验。上述“最终checkpoint”指Stage2 checkpoint，不等于LP epoch99；没有原始LP曲线，无法重新核实每个摘要数字的best/末轮口径。原始sweep脚本设计提取Max accuracy，只能作为设计证据。

## 4. 有状态但没有完整最终结果：不能统一叫“记录丢失”

| ID | 实验/评价 | 已保存到哪里、目前能确认什么 | 缺失或正确状态 |
|---|---|---|---|
| P01 | 新shared512/A，T→S=1/S→T=1 | handoff记载用户反馈LP明显下降 | **最明确的数值缺项：精确best和PT/LP日志均未提供** |
| P02 | 新no-share512/A，T→S=1/S→T=0.5 | [当前命令](D:/program/MacDiff/tools/vlm_pilot/STAGE1_TEXT_EXPERIMENT_PLAN.md:3)已保存；沿用st02 YAML，CLI覆盖0.2为0.5，输出全部_s2t05 | 当前选择；尚无回传结果或可核实进度，不推断已完成 |
| P03 | No-share S→T=0.2 | st02配置和助手建议有保存 | 用户已选择0.5替代；未确认跑过，不算漏记一个完成实验 |
| P04 | Shared512/0.1的checkpoint-150 LP | handoff及512分析保存首轮“72多”；用户决定不继续 | **未完成的评价**；没有最终best，不应编造或强制补跑 |
| P05 | 旧BPE/sample_target_blend/shared/H256，T→S/S→T=0.1/0.1 | [归档](D:/program/MacDiff/handoff_artifacts/handoff_before_20261001_refresh.md:21)保存PT0–19摘要，[原始20轮附件](C:/Users/97537/.codex/attachments/d0c2985f-c3a5-4814-a2e2-5ba0fc9ef962/已粘贴的文本.txt)仍可读取；ckpt130恢复命令指向shared，首次恢复失败 | 最终PT/LP缺失，修正后恢复是否成功无证据；不能当T08参数EMA或T09新七句的结果 |
| P06 | 旧BPE/sample_target_blend/no-share/H256 | 配置与建议命令有保存 | 未找到实际运行或结果证据 |
| P07 | 新七句fixed CLIP RMS/H512/none/S→T=0.1 | sentence_fixed_rms配置及smoke/PT/LP示例有保存 | 未找到实际运行或结果证据；不能套用T06旧BPE的85.82 |
| P08 | Dense/full-token Stage2 OSE | [归档](D:/program/MacDiff/handoff_stage2_legacy_20260905.md:124)明确当时无服务器结果/中间LP，配置和命令已保存 | 未找到后续训练/LP结果，不能填成已完成 |
| P09 | Stage1 OSE peer/cross-instance diffusion | 设计、YAML、README使用方法有保存 | 未找到实际训练/LP结果 |
| P10 | 与T13同协议的原版A、PT/LP有效batch256 | 作为缺失控制组有记录 | 未确认运行；这是尚未取得的对照，不是已做结果遗失 |

其他NTU60 XView、NTU120、PKU配置、v/区域bias/1024、部分共享、Stage2 K=1/SyncBN关闭等讨论或支持项，没有本地已跑证据时不登记为完成实验。

## 5. 诊断与失败尝试单独保存

- 冻结Stage1的full/masked几何、单exemplar结果和masked Stage2 teacher诊断保存在Stage2归档，属于checkpoint分析，不另计预训练实验。
- 旧EMA条件打乱与remap几何有d027fa4a原始JSON；不是LP结果。
- [stage1_text_geometry_summary.json](D:/program/MacDiff/handoff_artifacts/stage1_text_geometry_summary.json)保存了文本/原骨架几何比较；它是train诊断，不是官方测试LP。
- [STAGE1_READOUT_COMPARISON.md](D:/program/MacDiff/STAGE1_READOUT_COMPARISON.md)主要保存读出比较协议，未找到正式comparison.json/.md数值；one-shot结果也不能冒充全标签LP。
- 旧Stage2全token/JMB试跑OOM、旧全局256读出ReSA近均匀异常，以及目标库恢复命令解析失败有文字记录，都是失败/诊断事件，不应虚构完成训练成绩。

## 6. 本轮发现的汇总遗漏与过时表述

1. **T02的83.03%未进入当前handoff早期简表，但两份归档都保存了。** 属于汇总遗漏，不是分数消失。本总表及handoff现已补上。
2. **Stage2六组分数主要只在Stage2归档。** 本总表已集中，仍保留其口径与参数缺项。
3. `INPUT_NORMALIZATION_AUDIT.md`的31 YAML、B生效0、NTU60 XSub全A是2026-10-04快照；当前35份，多出no-share、st02和B PT/LP四份，B已生效并测试。本轮为旧快照添加说明，不把旧统计冒充当前扫描。
4. 当前handoff曾把shared YAML标为“当前预训练YAML”；实际当前是no-share st02加CLI0.5，本轮修正表格标签。
5. 2026-10-04归档顶部已有T11=85.79，底部仍有“待运行”旧残留；旧512分析里的等待S→T1/B未跑状态也已过时。保留历史原文，通过醒目说明指向当前handoff与本总表。
6. 当前实验计划已经明确分界：顶部为0.5，后面是2026-10-04历史规划；其中旧的待测状态不能另算当前缺项。

## 7. 优先补齐的记录

优先是P01的精确LP分数与日志、T11的完整LP日志；其次是T01–T08和S01–S06的原始日志、实际命令/checkpoint args与服务器SHA。T05还缺更精确成绩，S05/S06缺评价的checkpoint身份，S01/S02缺完整sweep表。P02等当前/计划项等用户提供实际结果后登记，P03/P04不要伪造完成结果。

今后每次至少保存：实验ID、文本/cache与target版本、模型H/norm/share、归一化、两条任务权重、PT/LP每卡batch与累积/卡数、LR日程、seed、代码SHA、checkpoint、PT/LP日志路径、LP best及对应epoch、末轮和末20均值。参数按实际CLI/args登记，不能只写一个YAML文件名。

本轮只整理文档和证据索引，没有修改训练模型、实验配置、缓存、checkpoint或启动训练。


## 2026-10-09：最新实验与当前下一组

新增 **T14**，目前共14组原版/文本Stage1数值记录（此前13组结论为10月7日快照）。本组是native MacDiff，A与decoder3按用户说明，PT500；用户已更正实际LP加载checkpoint-499，LP100。

| ID | PT / LP实际命令 | best | best epoch | last | 末20 mean ± population SD |
|---|---|---:|---:|---:|---:|
| T14 | 双卡PT64×accum1=128；LP128×accum1=256；seed0；PT min_lr由日志确认5e-4 | 85.704755 | 94 | 85.595585 | 85.635917 ± 0.038372 |

不能把本组登记成“仅decoder5→3”的纯消融，或误写成LP399；epoch/LP组织等也与历史对照不同。PT/LP同目录，部分早期PT checkpoint被LP覆盖，但399/499不在覆盖范围。完整依据见 [T14分析](D:/program/MacDiff/handoff_artifacts/decoder3_20261009/README.md) 与 [命令](D:/program/MacDiff/handoff_artifacts/decoder3_20261009/commands.json)。

用户已确认下一组为T12控制（1/.1、A、PT400），只改骨架decoder_depth为3，S→T仍5层；配置/命令已准备，未运行，无新成绩。此前1/.5计划仍没有回传结果，不将其叠加到decoder3本组。原始日志来源是 [PT500+LP100合并附件](C:/Users/97537/.codex/attachments/f6643024-0636-4677-ab2b-60b92d4ec967/已粘贴的文本.txt)。
