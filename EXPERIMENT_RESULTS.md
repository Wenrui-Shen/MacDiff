# MacDiff 实验结果总表与保存完整性审计

审计日期：2026-10-07；最新结果追加于2026-10-18。范围为本地 `handoff.md`、归档交接、其他 Markdown、配置、已有 JSON 统计以及文档明确引用的附件。没有访问服务器；不能确认尚未回传的运行进度，也不能恢复从未写入这些材料的聊天内容。

结论：**主要已报告成绩有保存，但不同实验的配置、完整日志与结果没有全部保存齐。** 10月7日审计快照查到13组原版/文本Stage1数值成绩、6组Stage2数值摘要；当时Stage1中4组有完整PT400+LP100附件，1组有完整PT但LP仅口述，另外8组主要只有历史成绩摘要。当前Stage1已追加到T24，共24条配置/成绩记录，最新结果见下文。Stage2六组均只有交接里的结果摘要，没有本地完整PT/LP日志或sweep CSV。

35份 YAML 是当前配置库存，不代表35次实际训练。同一个 YAML 的 CLI 权重、batch、代码版本或模型结构不同，仍需分开登记；重复上传日志也不增加实验次数。

原版与文本Stage1另有 [独立汇总](D:/program/MacDiff/STAGE1_EXPERIMENT_RESULTS.md)，包含最新24条记录、完整LP统计及Stage1待结果/计划项；下文10月7日审计正文保留历史快照。

## 2026-10-18：no-share 5/5/3，T24（最新已完成）

用户明确确认对应10月16日5/5/3组，登记T24：相对T23仅native/T→S各3→5层，S→T仍3，其余保留。PT400/LP100连续有限，两阶段逐epoch LR与T23完全相同，text uni0、权重1/.1组合、目标库40091和LP参数384060由日志核验；完整服务器Namespace/代码、数据、checkpoint SHA及加载身份仍缺。

LP **best85.522804492%@89、last85.383309101、末20均值85.414240689±.054243770**；相对T23 best−.758127134pp、last−.697476953pp、末20−.675642897pp，末20每轮均低。PT末20 native+.02*skel_uni=.016053768166（−4.739631%）、T→S=.021204194818（−3.808544%）、S→T=.007920559358（−26.559421%）：拟合PT目标更好但LP更弱，不能用loss降低判断表示改善。

加深native可能降低对encoder条件的依赖、加深T→S可能改变remap/动态目标，这是待验证机制，不是已证实shortcut/过拟合。两套decoder同时变化，不能拆分贡献；batch/floor/remap/text uni保持，不能当此次的新改动。T23仍最高best，T21仍最高后期均值，不自动更改训练设置。详见[Stage1汇总](D:/program/MacDiff/STAGE1_EXPERIMENT_RESULTS.md:9)和[no-share分析及原始附件SHA](D:/program/MacDiff/tools/vlm_pilot/NOSHARE_NORMB_LOG_ANALYSIS.md:3)。仅更新现有记录，不复制日志/新建目录/改训练代码或启动训练。

## 2026-10-16：全decoder3/remap2048/text uni0 no-share，T23（历史已完成）

用户回传此前待结果no-share的完整PT400/LP100，登记T23；轮数连续、数值有限。运行身份按既有命令/用户上下文，日志直接核验PT floor5e-4、T→S1/S→T.1总loss组合、text uni0、LP384060参数；完整Namespace/实际源码与checkpoint身份仍缺。

LP best **86.280931625%@73**，last **86.080786054%**，末20 **86.089883586±0.065868988%**。相对T22原版3层：best **+0.818777313pp**、last **+0.745997098pp**、末20 **+0.698083473pp**；相对T21旧no-share5层：best **+0.060650170pp**、末20 **-0.044577872pp**，峰值略高但后期整体略低，不能称全面超过旧组。

T23相对T22有93/100个LP epoch较高、末20全部较高，PT和LP LR逐轮相同；native+.02*skel_uni末20仅比T22高.301527%，LP改善没有伴随更低原生重建loss。S→T后期比旧T21明显降低但仍从最低.006286279954@175回升到末轮.010948793664，不能直接归因text uni关闭或单独扩宽remap。三套深度、PT floor、remap和text uni同时改变；不同epoch不作跨seed显著性证据。

原始附件：[T23 PT](C:/Users/97537/.codex/attachments/1531a82d-ffc0-41e3-8a13-f08563cb2845/已粘贴的文本.txt)、[T23 LP](C:/Users/97537/.codex/attachments/80c62e64-514d-4dbb-96ef-feda33238344/已粘贴的文本.txt)。SHA256：PT=`988c79bc59c0ffedc07a767e3409e39df0a951c9039b25891435cbe59df75990`；LP=`4acf5fee64fe294c6f549d624719a2273d8260b484d54e0f3e39167e27b1d66f`。

完整比较见[Stage1汇总](D:/program/MacDiff/STAGE1_EXPERIMENT_RESULTS.md)及[既有no-share分析](D:/program/MacDiff/tools/vlm_pilot/NOSHARE_NORMB_LOG_ANALYSIS.md)。T22/T23配对已完成；本次仅更新现有记录，不复制日志/新增归档目录，不改训练配置或启动训练。



## 2026-10-15：原版decoder3/min_lr5e-4，T22（已完成，配对no-share结果见T23）

用户说明为既有新3层原版命令的结果，登记T22，当前22条Stage1配置/成绩记录。PT400/LP100完整连续且有限；PT末20 LR5e-4，LP逐epoch LR与T20完全相同，可训练参数384060。按用户与命令身份为官方隔离入口/native/A/decoder3、PT2×32/accum1、LP2×64/accum1、seed0、checkpoint399；完整Namespace与源码/数据/checkpoint SHA仍未独立核验。

LP **best85.462154313%@90、last85.334788956、末20均值85.391800113±.045036580**。相对T20原版5层/min_lr1e-5：best−.539786500pp、last−.448811259pp、末20−.458515284pp；全部100轮LP准确率均低于T20对应轮。PT末20loss=.016801853122，对比T20=.014573484324高15.290570%；LP末20train loss从.402430325676降到.334522595087，test loss从4.493198760020升到4.684069968878，显示相同LP协议下测试泛化较弱，不能单据此确诊PT过拟合。

**深度5→3与PT floor1e-5→5e-4同时改变，不能认定下降全由减层造成。**保留用户PT5e-4/LP0偏好，不自动改训练配置。新no-share全3层/decoder512/remap2048/text uni0尚待结果，后续应先与T22配对。完整统计、原始附件路径及SHA见[Stage1最新结果](D:/program/MacDiff/STAGE1_EXPERIMENT_RESULTS.md:9)。未复制原日志/新建目录/修改训练代码或启动实验。

## 2026-10-14：5层no-share双卡公共配置，T21（历史已完成）

新增T21，当前21条Stage1配置/成绩记录；它是10月13日5层no-share的独立PT+LP结果，T20的同seed重复LP仍不另编号。用户确认native/T→S/S→T各5层、text decoder512/none、remap512、A、seed0、PT400/双卡每卡32/accum1、LP100/每卡64/accum1、权重1/.1、checkpoint399、七句v3/sample_target_blend。**本组PT min_lr仍为1e-5，不是新3层/5e-4/remap2048计划。**

| 指标 | T20 原版5层 | T21 no-share5层 | T21−T20 (pp) |
|---|---:|---:|---:|
| LP best (%) | 86.001940813@95 | **86.220281456@88** | **+.218340643** |
| last (%) | 85.783600214 | **86.147501239** | **+.363901024** |
| 末20 mean (%) | 85.850315396 | **86.134461457** | **+.284146061** |
| 末20 population std (pp) | .072069144 | .052393987 | 不作显著性检验 |

PT400/LP100连续且有限，PT与LP的逐epoch LR都与T20完全相同。末20每轮LP均高于T20对应轮，但这是一次训练的轮间结果，不是独立重复。相比T12，best仅约+.024260pp、末20+.133734pp；T12 PT accum2/global128与T21 accum1/global64不同。

末20 native MSE+.02 skel_uni=.014517515377，T20=.014573484324，仅低.384%；T21 S→T最低.009536486467@116→末轮.029400900783（3.083倍），回升比T12更大。S→T loss更高而LP较好，不把文本MSE当LP代理，不据训练loss回升确认过拟合或直接增加dropout。

原始附件：[PT](C:/Users/97537/.codex/attachments/d5aac71b-95b1-4329-99ec-ce12e28ac411/已粘贴的文本.txt)、[LP](C:/Users/97537/.codex/attachments/e064c6fe-694a-4bd4-a546-77cadd05ba2f/已粘贴的文本.txt)；详细统计、SHA与分析更新在[既有no-share分析文档](D:/program/MacDiff/tools/vlm_pilot/NOSHARE_NORMB_LOG_ANALYSIS.md)。配置身份由用户确认；日志不含完整Namespace/代码、数据、checkpoint SHA或软件版本，未作全部独立核验。10月15日新原版3层已回传T22，新no-share仍待；本节保留T21原身份。

## 2026-10-13：官方双卡默认PT+LP，T20（历史已完成）

新增T20，当前20条Stage1配置/成绩记录。同一PT的两次LP作为T20重复执行保存，不另编号T21。用户提供PT400+两份LP100完整epoch日志，明确确认LP seed均0，输出目录分别lp/lp1，其他算法参数相同。

| 执行 | best LP acc | best epoch | last | 末20均值 | 末20总体std |
|---|---:|---:|---:|---:|---:|
| T20 LP1（lp，seed0） | 86.001940813 | 95 | 85.783600214 | 85.850315396 | 0.072069144 |
| T20 LP2（lp1，seed0） | 86.001940813 | 95 | 85.783600214 | 85.850315396 | 0.072069144 |

两份LP附件**逐字节、100轮全部字段完全一致**，SHA256相同；用户已确认是同seed不同目录的两次执行。本次显示固定seed LP可精确复现，不能据同seed重复估计跨seed或PT seed方差。

PT0–399完整、数值有限，末loss0.014521730；epoch380–399学习率约1e-5，确认末端LR正确。LP0–99完整；epoch90–99 LR0但head BN仍更新，best95是该阶段结果。按已给默认命令为官方隔离源码/native/A/decoder5/PT400/双卡PT32有效64/checkpoint399/LP100双卡64有效128；全部actual args与服务器源码/数据/checkpoint SHA及环境仍缺，不能仅由epoch日志补填。

距第三方口述86.2约**0.198059个百分点**；不把残差归为硬件、某个单独配置或已确认实现bug。完整日志、统计及身份已归档：[分析报告](D:/program/MacDiff/handoff_artifacts/official_dualgpu_results_20261013/README.md)、[精确JSON](D:/program/MacDiff/handoff_artifacts/official_dualgpu_results_20261013/summary.json)。

## 2026-10-12：旋转encoder复评LP64，85.89%（T20之前）

新增**T19**：用户明确报告复用T18旋转checkpoint499，仅将LP每卡batch128→64，其余沿用保存配置，LP成绩**85.89%（口述）**。这是一条新的LP配置成绩记录，不是新增预训练；当前共19条Stage1数值记录。best/末轮口径、对应epoch、last/末20及完整日志、实际seed/Namespace未提供。

与同encoder、LP每卡128的T18 best85.789665239%粗略比较，约**+0.10个百分点**；口述85.89只有两位小数且统计口径未核实，不将0.100334761作为精确效应值。距第三方口述86.2约0.31个百分点。

batch改变会同时改变局部BN、global梯度batch（256→128）、每epoch更新次数及增强/RNG组织，因此这不是同配置的随机重复，不能从这两分数估计seed方差。固定encoder、batch与全部LP协议后，多个LP seed才用于估计head/增强波动；仍不能替代多个PT seed。干净官方双卡默认PT&&LP随后于10月13日回传T20完整日志，见上文最新结果。[口述记录JSON](D:/program/MacDiff/handoff_artifacts/native_rotation_results_20261012/lp_bs64_user_report.json)。

## 2026-10-12：原版随机旋转对照结果（T19之前）

新增T17/T18，当前共**18组**原版/文本Stage1数值记录。用户粘贴runner JSON，保存全部精度和服务器日志路径；尚未提供完整PT/LP日志、run_manifest、actual args/seed、源码SHA、数据SHA或checkpoint，因此不称为独立重算完整日志。

| ID | PT random_rot | best LP acc | best epoch | last | 末20均值 | 末20总体std |
|---|---|---:|---:|---:|---:|---:|
| T17 | False | 85.638039799 | 64 | 85.474284367 | 85.557678338 | 0.042406356 |
| T18 | True | 85.789665239 | 88 | 85.619844774 | 85.684133938 | 0.044831925 |

按准备脚本协议，两组均native/A、decoder3、PT500、lr1e-3/min_lr1e-5、双卡PT64/accum1有效128；checkpoint499 LP100、lr.1、双卡128/accum1有效256。协议字段不能代替服务器实参核验；summary不含seed，不将默认0写成实测值。

旋转−不旋转：best **+0.151625439个百分点**、last **+0.145560408个百分点**、末20均值 **+0.126455600个百分点**。小幅正收益仍不足以补上复现差距；单次比较与epoch std不能证明多seed稳定性。T18距第三方口述86.2仍为0.410334761个百分点。

建议先复用T18 checkpoint499，仅将LP每卡batch128→64（accum1保持，原run保存JSON中的seed/LR/其他参数继承），检查LP协议贡献。第三方双卡3090有效PT64已确认、LP128仍估计；本次128/256不同。用户随后要求命令并回传LP64为85.89%，登记T19；见顶部。[原始JSON及分析](D:/program/MacDiff/handoff_artifacts/native_rotation_results_20261012/README.md)、[推荐LP命令](D:/program/MacDiff/handoff_artifacts/native_rotation_results_20261012/recommended_lp_bs64.sh)。

## 2026-10-11：第三方双卡3090复现参考

用户进一步确认：原MacDiff默认配置、双卡3090、每卡PT batch32；按默认accum1，有效PT batch64。用户报告NTU60 XSub LP为**86.2%**。LP每卡64、双卡有效128仍是用户估计，未独立确认；未提供实际命令、日志、checkpoint args、seed或链接。此记录属于第三方参考，不编号为T17，不改变本项目16组Stage1成绩计数。

当前原版旋转脚本的PT有效128、LP有效256与此参考不同。batch尚未调整；decoder3、PT500和正确LR仍按用户既定要求固定。实现方面，已将PT累积尾窗口恢复为官方丢弃策略，DDP开关保持现有设置，详见[最新交接](D:/program/MacDiff/handoff.md:5)。

## 2026-10-11：text decoder depth3结果（历史状态）

新增 **T16**，当前共16组原版/文本Stage1数值记录。T15与T16是两次文本实验，T01原版历史约85.86%与T16本次85.86%不是同一实验。

| ID | 实验身份 | LP成绩 | 证据与缺项 |
|---|---|---:|---|
| T16 | 相对T15仅S→T text_decoder_depth 5→3；native/T→S仍各3层，S→T hidden512/output_norm=none；其余按T15控制计划（最终实参未核验） | **85.86%，用户口述** | 2026-10-11用户反馈；完整PT/LP日志、best/末轮口径及对应epoch、last/末20、实际batch/accum、checkpoint、seed/args及服务器SHA未提供 |

与T15口述85.84%相差 **+0.02个百分点**；这是单次两位小数的结果，尚不能确认降低文本decoder深度改善LP。该组已有成绩，不再当作待结果；本次只登记，没有改训练配置/模型或启动训练。原版MacDiff旋转对照已于10月12日回传T17/T18，见上文最新结果。最新状态见 [handoff](D:/program/MacDiff/handoff.md:5)。

## 2026-10-10：decoder3结果与当时下一组（历史状态，以上文为准）

新增 **T15**，当前共15组原版/文本Stage1数值记录；下文10月7日的13组统计及10月9日的14组统计均为历史快照。

| ID | 实验身份 | LP成绩 | 证据与缺项 |
|---|---|---:|---|
| T15 | 当前七句v3/sample_target_blend/no-share/A；native/T→S各3层，S→T仍5层/512/none；权重1/0.1，PT不旋转（结构身份按上一会话上下文登记） | **85.84%，用户口述** | 2026-10-10用户反馈；完整PT/LP日志、分数统计口径（best/末轮）、对应epoch、last/末20、最终实际batch/accum、checkpoint/args及服务器SHA未核验 |

T15是文本no-share实验；T14是原版A/decoder3/PT500、checkpoint-499 LP的85.704755%，两组分别保留。此前目标库FileExistsError属于历史尝试，不能再据此说当前decoder3没有训练成绩；其处理方式和最终运行参数尚未核验。

**10月10日选择的text_decoder_depth 5→3、骨架3层实验已于10月11日回传85.86%，登记为T16。**S→T hidden512/output_norm=none及其余参数原计划沿用T15控制，最终实参未核验；随机旋转方向已另选原版MacDiff对照。详情见 [handoff最新状态](D:/program/MacDiff/handoff.md:5) 和 [当前实验计划](D:/program/MacDiff/tools/vlm_pilot/STAGE1_TEXT_EXPERIMENT_PLAN.md:3)。

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

10月9日已准备以T12为控制的骨架decoder3配置/命令，S→T仍5层；10月10日用户已回传该组口述85.84%，登记为T15，当前下一组见顶部最新状态。此前1/.5计划仍没有回传结果，不将其叠加到decoder3本组。原始日志来源是 [PT500+LP100合并附件](C:/Users/97537/.codex/attachments/f6643024-0636-4677-ab2b-60b92d4ec967/已粘贴的文本.txt)。
