# Stage1 实验汇总

更新日期：2026-10-18；10月7日汇总保留为历史快照。数据集为NTU60 XSub，评价为只输入骨架的linear probe。汇总所有本地已保存的Stage1数值成绩、定性反馈、未完成评价和计划；保留 [总审计](D:/program/MacDiff/EXPERIMENT_RESULTS.md) 的实验ID。

2026-10-07快照有13组数值成绩，其中4组有完整PT400+LP100附件，1组有完整PT但LP只有口述，8组主要保存历史成绩摘要。另有9项Stage1状态记录，不能统一算作已完成的独立训练。当前24条配置/成绩记录，最高LP best为T23的86.280932%；T22/T23三层配对及T24 no-share 5/5/3均已回传，T21旧5层仍有最高的后期LP均值。S→T=0.5仅为历史候选。

`T→S`为文本到骨架任务权重，`S→T`为骨架到文本任务权重。`H/norm`是S→T decoder的隐藏维度/末端归一化，与CLIP缓存维度不同。`share=True`共享原生骨架重建与T→S骨架decoder主体；S→T文本decoder独立。

## 2026-10-18：no-share 5/5/3已完成T24，加深后LP下降

用户明确确认本次为10月16日5/5/3组，登记T24。相对T23仅native/T→S decoder_depth从3改5，S→T仍3，其余保留remap2048/text decoder512/none/text uni0/权重1/.1/PT min_lr5e-4/PT400总64/LP100总128/seed0/checkpoint399。配置差异已本地核对；日志不含完整Namespace，实际服务器代码/数据/checkpoint SHA及加载信息仍未独立核验。

| 指标 | T23 3/3/3 | T24 5/5/3 | T24−T23 |
|---|---:|---:|---:|
| LP best (%) | 86.280931625 @73 | 85.522804492 @89 | −.758127134 pp |
| LP last (%) | 86.080786054 | 85.383309101 | −.697476953 pp |
| LP末20均值 (%) | 86.089883586 | 85.414240689 | −.675642897 pp |
| LP末20 population std (pp) | .065868988 | .054243770 | 非跨seed方差 |
| native+.02*skel_uni末20 | .016852515289 | .016053768166 | −4.739631% |
| T→S末20 | .022043740370 | .021204194818 | −3.808544% |
| S→T末20 | .010784990420 | .007920559358 | −26.559421% |

PT400/LP100完整连续且有限；两阶段LR均与T23逐epoch完全相同，text uni全程0，总loss符合1/.1组合（最大残差1.776237379e-9），目标库epoch1起40091。末20所有LP轮均低于T23，不只是峰值下降。LP末20train CE由.358558020升至.401845749，不是简单“训练更好、测试更差”的LP过拟合模式。

当前直接证据是：两套骨架decoder加深后，PT目标拟合更好、相同LP协议下分类表现更弱。更强native decoder可能降低对encoder条件的依赖，是待验证假设；T→S不直接反传骨架encoder，而是改变remap/动态目标后间接影响S→T。两套decoder同时改变，不能确定哪一条分支造成下降，不把较低S→T MSE等同于更强语义。T24与T23的batch/floor/remap/text uni没有作为新变量改变，不能用这些旧因素代替解释此次下降。

相对T21旧5/5/5，T24 best−.697476964pp、末20−.720220769pp；协议含floor/remap/text uni等差异，不能合并成纯深度对照。相对T22原版3层虽best+.060650179pp，但native深度不同，不能算T24文本收益；同协议native5/floor5e-4基线尚无结果。T23仍最高best、T21仍最高后期均值，不自动改训练设置或新增消融。

完整分支统计、资源、静态LP入口差异及原始附件SHA见[no-share分析顶部](D:/program/MacDiff/tools/vlm_pilot/NOSHARE_NORMB_LOG_ANALYSIS.md:3)。[PT原始附件](C:/Users/97537/.codex/attachments/c50df5b1-ebf8-41fb-bb9c-9ed9ddeaf01e/已粘贴的文本.txt)、[LP原始附件](C:/Users/97537/.codex/attachments/f9c98910-3ae0-4a78-820d-a301f57dca5c/已粘贴的文本.txt)。本次只更新5份既有文档，不改训练代码/配置、不复制日志/新建目录或启动训练。

## 2026-10-16：LP best变化的因素对照与归因边界（10月18日补T24）

以下单位均为百分点pp，是已观察到的单次运行差值，不是超参的固定收益或跨seed效应估计。结合近期T14–T23与已有T09–T12参考对照，不用跨配置回归强行拆分无法识别的因素。

| 改动 | 对照 | best变化 | 可支持的解释 |
|---|---|---:|---|
| 加入整套no-share文本辅助，5层协议 | T20→T21 | +.218341 | 公共参数更匹配，末20+.284146；PT实现/额外任务/RNG仍不同 |
| 加入整套no-share文本辅助，3层/高floor协议 | T22→T23 | +.818777 | 末20+.698083，PT/LP LR逐轮相同；整套方案收益，不是no-share开关单独收益 |
| PT增加随机旋转 | T17→T18 | +.151625 | 保存runner仅改PT旋转；末20+.126456，限原版3层/PT500协议 |
| LP总batch256→128，同encoder | T18→T19 | 约+.10 | T19只有85.89口述，完整LP日志/actual seed未核验 |
| native/T→S共享→不共享，七句512/none | T11→T12 | 约+.406021 | T11基线为85.79口述；该七句协议初步支持no-share，不代表所有目标模式有效 |
| T→S权重.1→1，S→T保持.1 | T09→T10 | +.090975 | 完整LP日志，末20+.065805；首组实际PT batch等未完全核验 |
| PT有效batch128→64（accum2→1） | T12→T21 | +.024260 | 主要已知设置相同，末20+.133734；best收益小，非严格服务器身份审计后的唯一因素证明 |
| S→T decoder5→3，骨架侧保持3/3 | T15→T16 | +.02 | 两者均口述两位小数，尚无明确深度收益证据 |
| native/T→S各3→5，S→T保持3 | T23→T24 | −.758127 | 用户确认5/5/3，PT/LP LR逐轮相同，末20−.675643；两套decoder同时变化，不拆分各自贡献 |

### 不能按单因素使用的差值

| 实验变化 | 对照 | best变化 | 原因 |
|---|---|---:|---|
| 原版decoder5→3 + PT floor1e-5→5e-4 | T20→T22 | −.539787 | 两个因素同时改变，不能给深度或floor单独定量 |
| no-share骨架两套decoder5→3 | T12→T15 | 约−.356021 | T15完整实参/日志缺失，实际batch/累积身份不完整，仅是减层可能不利的弱线索 |
| 全decoder5→3 + floor提高 + remap512→2048 + text uni关闭 | T21→T23 | +.060650 | 整个组合best略高，末20−.044578；不能分别给remap或text uni记功 |
| S→T hidden256→512 + 去最终LN | T10→T11 | 约−.102771 | 同时改变宽度和输出归一化，且T11为口述 |

旧参数EMA的shared→no-share是85.78→85.77（−.01pp），未显示收益。因此+.406只属于新七句/512/none参考对照，不能视为通用no-share收益。旧固定CLIP/BPE中S→T权重1→.1曾观察到84.64→85.95（+1.31）及约83.7→85.82（约+2.12）；旧实际batch/args不全，不能直接迁移为当前七句目标的预期收益。

当前不能独立估计remap2048、关闭text uniformity、PT min_lr5e-4、PT400→500或归一化A/B各自能提高多少LP best。0.818777是相对T22的85.462154基线，不能直接叠加到T20的86.001941上。

5/5/3已于10月18日回传T24：best−.758127pp、末20−.675643pp，新增容量未带来预期LP收益；仍不能分别拆开native和T→S贡献。当前配置和训练命令不因结果自动调整，不重跑该已完成组。

## 2026-10-16：近期结果统计（10月18日补T24）

| ID | 实验 | LP best (%) | 末20均值 (%) |
|---|---|---:|---:|
| T14 | 原版3层，PT500，PT floor5e-4 | 85.7048 | 85.6359 |
| T15 | no-share 3/3/5，口述 | 85.84 | 未提供 |
| T16 | no-share 3/3/3，口述 | 85.86 | 未提供 |
| T17 | 原版3层，PT500，不旋转，floor1e-5 | 85.6380 | 85.5577 |
| T18 | 原版3层，PT500，旋转，floor1e-5 | 85.7897 | 85.6841 |
| T19 | 复用T18，LP总batch改128，口述 | 85.89 | 未提供 |
| T20 | 原版5层，PT400，floor1e-5 | 86.0019 | 85.8503 |
| T21 | no-share 5/5/5，floor1e-5 | 86.2203 | 86.1345 |
| T22 | 原版3层，PT400，floor5e-4 | 85.4622 | 85.3918 |
| T23 | no-share 3/3/3，floor5e-4 | 86.2809 | 86.0899 |
| T24 | no-share 5/5/3，floor5e-4 | 85.5228 | 85.4142 |

no-share深度按native/T→S/S→T排序；T15/T16/T19为用户口述，无完整末20统计。T17/T18有summary，T14/T20–T23有完整日志；不同PT轮数、batch、floor、remap和text uni使这些行不构成统一协议消融。T21 remap512/text uni.02，T23 remap2048/text uni0。当前最高best为T23，最高已知末20均值为T21。

**5/5/3已完成T24**，保持T23的remap2048/text uni0/权重1/.1/PT floor5e-4等设置；独立配置 [pretrain_madiff_text_sentence_sample_target_blend_noshare_dec5_st3.yaml](D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_text_sentence_sample_target_blend_noshare_dec5_st3.yaml)保留运行身份，原T23配置保留。历史命令见[当前计划](D:/program/MacDiff/tools/vlm_pilot/STAGE1_TEXT_EXPERIMENT_PLAN.md)，不重复启动。

## 2026-10-16：全decoder3/remap2048/text uni0，T23（最新已完成）

按用户本次“noshare跑出来了”与此前唯一待回传no-share组登记T23：native/T→S/S→T各3层、text decoder512/none、remap2048、A/person0、七句v3/sample_target_blend、T→S=1/S→T=.1、text uni0、骨架uni=.02、PT400/总64/seed0/min_lr5e-4，LP100/总128/checkpoint399。JSON直接验证轮数、LR、分支loss组合、text uni关闭及LP参数384060；深度/宽度、实际batch/seed、checkpoint/入口、代码/数据身份仍依据此前命令和用户上下文，缺完整Namespace及服务器SHA，不能称全部独立核验。用户10月15日要求以后新命令使用根目录入口；本次日志没有记录LP实际入口，不反推是否改用了主入口。

LP best **86.280931625%@73**，last **86.080786054%**，末20 **86.089883586±0.065868988%**。相对T22原版3层：best **+0.818777313pp**、last **+0.745997098pp**、末20 **+0.698083473pp**；相对T21旧no-share5层：best **+0.060650170pp**、末20 **-0.044577872pp**，峰值略高但后期整体略低，不能称全面超过旧组。

| 组 | LP best (%) | epoch | last (%) | 末20 mean ± population std (%) |
|---|---:|---:|---:|---:|
| T20 原版5层/min_lr1e-5 | 86.001940813 | 95 | 85.783600214 | 85.850315396 ± 0.072069144 |
| T21 no-share5层/remap512/text uni.02 | 86.220281456 | 88 | 86.147501239 | 86.134461457 ± 0.052393987 |
| T22 原版3层/min_lr5e-4 | 85.462154313 | 90 | 85.334788956 | 85.391800113 ± 0.045036580 |
| T23 no-share全3层/remap2048/text uni0 | 86.280931625 | 73 | 86.080786054 | 86.089883586 ± 0.065868988 |

PT400/LP100连续且有限，PT/LP逐epoch LR与T22相同。text uni全程0，分支组合权重核算残差最大1.772e-9；目标库epoch1起覆盖40091。native+.02*skel_uni末20=0.016852515289，与T22总loss相比仅高.301527%，重建loss没有同步下降。S→T最低.006286279954@175→末轮.010948793664（1.741697倍），末20.010784990420，比T21低62.851858%，但目标移动、深度/floor/remap/text uni同时改变，不能作单因素归因。epoch时间累计18.111070h、allocated峰11241.912MiB；详细分支曲线和比较边界见[既有no-share分析](D:/program/MacDiff/tools/vlm_pilot/NOSHARE_NORMB_LOG_ANALYSIS.md)。

原始附件：[T23 PT](C:/Users/97537/.codex/attachments/1531a82d-ffc0-41e3-8a13-f08563cb2845/已粘贴的文本.txt)、[T23 LP](C:/Users/97537/.codex/attachments/80c62e64-514d-4dbb-96ef-feda33238344/已粘贴的文本.txt)。SHA256：PT=`988c79bc59c0ffedc07a767e3409e39df0a951c9039b25891435cbe59df75990`；LP=`4acf5fee64fe294c6f549d624719a2273d8260b484d54e0f3e39167e27b1d66f`。

当前T22/T23配对均已完成，旧“新no-share待回传”只保留为历史状态。不修改训练配置或启动实验，不跑seed1；decoder蒸馏实验的服务器状态与结果没有据此推断。



## 2026-10-15：原版decoder3、PT min_lr5e-4，T22（已完成，配对no-share结果见T23）

用户明确说明本次是前述原版新3层组，登记T22。运行身份对应10月14日原版命令：隔离官方入口、A、one_person=True、PT不旋转、decoder3、PT400/每卡32/accum1/lr.001/min_lr.0005、checkpoint399、LP100/每卡64/accum1/lr.1/min_lr0、双卡seed0。完整epoch日志直接确认400/100轮、PT floor5e-4、LP LR与T20逐轮完全相同及可训练参数384060；完整Namespace/服务器代码、数据及checkpoint SHA未提供，其余身份按用户说明与既有命令记录。

| 指标 | T20 原版5层/min_lr1e-5 | T22 原版3层/min_lr5e-4 | T22−T20 |
|---|---:|---:|---:|
| LP best (%) | 86.001940813 @95 | 85.462154313 @90 | −.539786500 pp |
| last (%) | 85.783600214 | 85.334788956 | −.448811259 pp |
| 末20 mean (%) | 85.850315396 | 85.391800113 | −.458515284 pp |
| 末20 population std (pp) | .072069144 | .045036580 | 非跨seed方差 |
| PT末20总loss | .014573484324 | .016801853122 | +15.290570% |
| LP末20 train loss | .402430325676 | .334522595087 | −.067907730589 |
| LP末20 test loss | 4.493198760020 | 4.684069968878 | +.190871208858 |

PT0–399、LP0–99连续且所有数值有限，无缺轮/NaN/Inf。T22 PT末轮loss=.016918268116836707，最低=.01659404159190294@394；末20std=.00010965354433240797，末50均值=.016832726012979346；后期在平台附近波动，没有日志可见的发散。原版总loss包含骨架uniformity，未单独记录native MSE与uniformity，不能归因其中某项。LP best top5=97.319262509%@95、末轮top5=97.246482305%。

**本次下降贯穿全部100个LP epoch，每轮均低于T20对应轮，末20也全低；不是只少了一个峰值。**较低LP训练loss、较高测试loss与较低测试准确率，说明当前冻结特征在相同LP协议下的测试泛化较弱；不能仅据此确认PT encoder过拟合。LP90–99 lr=0，head BN仍更新，所以best@90正常。轮间std不能替代跨PT seed重复。

**不能把−.54pp单独归因decoder5→3：本次同时把PT min_lr1e-5→5e-4。**最后20轮LR升50倍；epoch299约从.000127203升到.000559194；400轮记录的epoch平均LR之和比例为1.510414649（仅描述日程幅度，不等于参数更新范数）。较浅decoder改变重建容量及传给encoder的梯度；高floor让后期保持较大步长，这两者及其交互均待控制验证。PT loss高15.3%也不能单独证明encoder表示差来自decoder容量。用户偏好仍为PT5e-4、LP0，不据此自动改回或启动补实验。若后续专门隔离深度，固定5e-4补5层可与T22比较，尚未选定/执行。

T14/T17/T18也是3层，但PT500及PT/LP有效batch等不同，不能拼成此次深度/min_lr单因素对照。当前只待新no-share全3层/decoder512/remap2048/text uni0结果；收到后应先与T22配对，不能直接拿旧T20当它的同协议原版控制。

**10月15日用户补充强调：此前已为对齐论文，把PT设为双卡每卡64、总128，成绩仍低。**不能只拿T22的PT400/global64解释整个原版复现差距。T14是decoder3/PT500/global128/min_lr5e-4，best85.704755%；随后T17/T18按保存协议已是decoder3/PT500/global128/min_lr1e-5，分别不旋转85.638040%、旋转85.789665%；T19复用T18 encoder把LP总batch256→128，口述85.89%。因此batch、500轮、较低floor及旋转均已有对应尝试，尚不足以恢复论文3层86.4。T17/T18是summary证据，完整实际运行身份仍缺，但不能忽略已有结果、把这些因素重新列成未尝试的确定解法。论文3层优势尚未在本项目复现，原因未定位；T20/T22不是纯深度对照这一边界仍成立。

原始附件：[PT](C:/Users/97537/.codex/attachments/6ced52ec-59e2-4d95-bbd7-ca61f15a4006/已粘贴的文本.txt)、[LP](C:/Users/97537/.codex/attachments/1a8e81d8-eef0-438b-8ab6-5a649fbddbb4/已粘贴的文本.txt)。SHA256：PT=`d86df98120f15ece762ee9238f21b433471779008f73a7ca445c17373a12bf82`；LP=`af3d7134f7e09590c46f73e3ed89328509b41183322b7af3ad7b7cc0adeb941a`。仅更新既有记录，不复制日志/新建归档目录，不修改训练代码/配置或启动训练。

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
