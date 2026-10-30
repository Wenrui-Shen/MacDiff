# Stage1 实验汇总

更新日期：2026-10-30；10月7日汇总保留为历史快照。数据集为NTU60 XSub，评价为只输入骨架的linear probe。汇总所有本地已保存的Stage1数值成绩、定性反馈、未完成评价和计划；保留 [总审计](D:/program/MacDiff/EXPERIMENT_RESULTS.md) 的实验ID。

## 2026-10-30：随机学生普通蒸馏checkpoint-140，阶段性LP完整回传

用户明确本次为此前checkpoint-140的LP。对应旧400轮日程（match200/ramp40），checkpoint epoch从0计数，因此完成141轮且全程lambda=1；**本条单列为普通蒸馏学生的阶段读出，尚未登记新实验ID，不能当作新100轮/snapshot10/lambda1.25外推试验的结果。** 学生随机初始化，教师为纯原生T20 checkpoint399，encoder8/dim256逐层表示MSE；无文本或原生diffusion decoder损失。预期PT每卡128/双卡global256、warmup20、lr.001/min_lr.0005、noAMP、A/person0；LP100/每卡128/global256、lr.1/min_lr0、seed0。上述运行身份依据用户说明和已给命令，JSON无Namespace/权重加载路径/服务器源码或checkpoint SHA。

完整epoch0–99连续唯一、全部数值有限，可训练参数始终384060，epoch90–99 LR为0。直接重读本次原日志与T20保存的lp1.jsonl计算：

| 指标 | T20原生教师历史LP | 普通蒸馏学生checkpoint-140 | 学生减教师 |
|---|---:|---:|---:|
| best (%) | 86.001940813 @95 | 86.074721030 @73 | +.072780217 pp |
| last (%) | 85.783600214 | 86.001940826 | +.218340611 pp |
| epoch80–99 mean (%) | 85.850315396 | 86.009218846 | +.158903450 pp |
| epoch80–99 population std (pp) | .072069144 | .032298707 | — |
| epoch80–99 train CE | .402430325676 | .176710370336 | — |
| epoch80–99 test CE | 4.493198760020 | 2.232693104827 | — |

按相同LP epoch序号参照，100轮80高/20低，末20为19高/1低，因此末段表现并非只有一个偶然best。**这仍不是同协议净收益：T20历史LP每卡64/global128，本次按给定命令每卡128/global256，局部BN、每轮优化更新数等不同。** 没有独立重复，末20std只是同一次LP的轮间波动，不能证明跨seed稳定超越；CE更低也不足以单独确认校准提升。

当前支持的结论：随机学生仅用普通表示蒸馏，其完整骨架输入下的LP已达到教师历史水平；高训练cos没有伴随明显下游性能损失。该结果不检验外推，也不能证明snapshot10或teacher-full候选有效。新100轮直接外推仍待实际结果，继续保留单教师当前方案，不自动实施完整教师或SCD异构分支。

原附件：[学生checkpoint-140 LP](/C:/Users/ruiz1/.codex/attachments/2a89c35c-c350-489e-9ab5-19b8e810a48a/已粘贴的文本.txt)，20117 bytes，SHA256=`f8a0cea6425886b672706d410a354f6a5e025b209839d4c8a056a3ad3004f897`。教师参照：[T20保存日志](/D:/Program/NEW/MacDiff/handoff_artifacts/official_dualgpu_results_20261013/lp1.jsonl)。

## 2026-10-29：Inverse Cosine SNR采样OTFM完整回传，T34

用户明确本次两份附件为新OTFM，登记T34，当前共**34条配置/成绩记录**。预期对应10月27日的 `inverse_cosine_snr_v1` 单因素时间采样对照：原OT条件路径/速度目标、sigma_min=1e-4、encoder8/native decoder3、A/person0、PT400/global64/floor5e-4/noAMP/seed0→原第8层LP100/global128/floor0/checkpoint399。实际Namespace、服务器源码/数据/checkpoint SHA及LP加载输出未提供；不从epoch JSON补齐所有实参。此组单列FM，不并入原生/文本四组均值。

### 完整性、采样与目标核验

直接解析原附件：PT0–399/LP0–99各400/100条，epoch唯一连续、所有数值有限，LP可训练参数全程384060。PT400轮与LP100轮LR分别与T29、T26、T22逐项完全相同。总loss满足 `FM+.02*uniformity`，最大绝对残差2.757428754e-9。

新日志全部400轮的 `train_time_mean` 平均0.501685468，`train_snr_middle_fraction` 平均0.901815158（**90.1815%**），后者各epoch范围0.895142772–0.906874002；与新采样表90.2%预期一致，明显不同于旧uniform的约51.95%理论比例。支持新采样已生效，不是证明服务器全部定义已核验。

### LP比较

| 指标 | T29旧uniform OTFM | T34新SNR采样OTFM | T26原生diffusion dec3 |
|---|---:|---:|---:|
| best (%) | 84.631246997 @90 | **85.231683676 @71** | 85.710819994 @73 |
| last (%) | 84.588791878 | 85.031538116 | 85.419699192 |
| epoch80–99 mean (%) | 84.512372650 | **85.049126665** | 85.445172265 |
| epoch80–99 population std (pp) | 0.055134978 | 0.075286871 | 0.054504224 |
| epoch80–99 train CE | 0.315357905005 | 0.180525480303 | 0.352072116944 |
| epoch80–99 test CE | 4.322367658325 | 4.674421139989 | 4.627313369447 |

相对T29：best **+0.600436679pp**、last +0.442746238pp、末20均值 **+0.536754015pp**；100个LP epoch全部更高，末20也全部更高。这是整条LP曲线的改善，不只是一次峰值。按给定协议是只改时间采样的单次正向观察，尚无独立重复或完整服务器身份核验，不能宣称稳定因果收益。

相对T26：best **-0.479136318pp**、末20均值 **-0.396045600pp**；100轮29高/71低、末20全部低。相对T22 best -0.230470636pp、末20 -0.342673447pp，末20同样全部低。匹配系数SNR后追回部分差距，仍未追平历史原生diffusion。T22/T26为历史AMP协议；已安排的同精度原生dec3 noAMP组仍无结果，不把剩余差距全部归为FM方法。

LP末20的train CE从T29的.31536降至.18053，test CE则从4.32237升至4.67442；top1同时改善，说明CE与准确率并不单调对应。不能仅凭训练CE下降认定表示改善，也不能凭测试CE升高确诊encoder过拟合或坍塌。没有logits/梯度/特征方差来定位该现象。末20std是同一次LP的轮间波动，不是跨seed方差。

### PT与结论边界

末20 total=0.055657468372、velocity=0.055065175240、uniformity=0.029614657312；末轮total=0.055259707083，也是本次最低epoch均值。velocity较T29末20低37.397945%，但采样分布改变了期望目标，不能把此差值当相同评价集上的去噪改善幅度，也不与原生epsilon loss直接比较。日志未见发散；没有microstep细节，不补推每步行为。峰allocated7025.704102MiB/reserved7688MiB，与T29记录基本相同；无耗时字段，不给吞吐结论。

当前最有支持的解释是：时间/噪声难度分布是此前掉点的一个有效调整方向；它没有消除FM与原diffusion的全部差异。即使系数SNR相同，OT输入幅度与原扩散不同；在同一OT路径上 `epsilon_hat=x_t-t*v_hat`、`epsilon_error²=t²*velocity_error²`，无权velocity目标仍有不同时间权重。以上是残余差距的候选机制，不能由汇总日志定位唯一原因或认定decoder旁路。优先使用已安排的同精度原生对照；如继续诊断，利用现有checkpoint固定batch/t/noise/mask分桶比较条件置零/打乱与encoder梯度。没有自动改采样、损失、AMP或启动新训练。

原附件：[T34 PT](/C:/Users/97537/.codex/attachments/e9cf75ea-0c11-462b-80eb-fd9a0da5c22c/已粘贴的文本.txt)、[T34 LP](/C:/Users/97537/.codex/attachments/4232e20e-e9f1-4629-be21-202535cc0920/已粘贴的文本.txt)。SHA256：PT=`fb3c960a7adeaac55b4fd49fa3edf09f8797355584e829818a6b578a144cc30f`；LP=`c9c6ec796a36b1c69235e39bb5fdd4763ceba3a9420804eccead3676d11fe924`。本次重读T29/T26/T22的原附件共同计算；不复制日志或新建归档目录。用户另索要最佳no-share全3层＋新读取命令，已用既有YAML给出，见[当前计划](/D:/program/MacDiff/tools/vlm_pilot/STAGE1_TEXT_EXPERIMENT_PLAN.md:3)，该文本组未回传结果，不预编号。


## 2026-10-28：后三层diffusion口述LP成绩85.71%，T33

用户回报“后三层diffusion的跑出来85.71”，新增T33，当前共**33条配置/成绩记录**，不等于33次完整独立PT。此前“多层尚无成绩/不预登记T33”的文字为对应日期历史状态，本节覆盖其当前状态。原始PT/LP日志未提供，best/last口径、epoch、末20均值、重复次数、实际Namespace/seed/服务器版本/checkpoint加载均不补猜。

| 项目 | 已知/预期定义 |
|---|---|
| 用户口述LP分数 | **85.71%**，两位小数；best/last待确认 |
| 方法身份 | 预期last3_shared_norm_decoder_final_layer_lp_v3，依据此前下发命令，未由新日志验证 |
| 预期预训练 | encoder8/256，第6/7/8层共享原self.norm及同一原生MacDiff decoder3，分别前向 |
| 预期目标 | L8+.1*(L6+L7)/2+.02*uni8；同t/epsilon/mask/condition dropout |
| 预期评价 | 原main_linprobe.py，仅第8层linprobe2/6400维，不融合 |
| 预期协议 | PT400/总batch64/floor5e-4/noAMP/seed0→LP100/总128/floor0；实际完整轮数未核验 |
| 原始日志/末20统计 | 未提供 |

历史原生dec3 T26的LP best为85.710819994%，按两位小数也为85.71。只能先记录本次口述成绩与该参考近似；不能把两位小数推成精确持平，也不能在best/last口径、AMP/noAMP差异及单次重复未控制时证明多层监督无效。当前没有新增可见提升证据，不据此自动改辅助权重、融合LP或重跑。T33为独立多层辅助方法，**不并入此前原生/文本四组均值**。代码/命令及版本定义见[多层方法文档](/D:/program/MacDiff/tools/MACDIFF_MULTILAYER_DIFFUSION.md)。

同日KL对比学习核对：用户回传主线A完整key v2的10325/10326命令，明确lambda_contrastive=.0001、lambda_skeleton_to_text=0。前者是InfoNCE，后者只禁用旧S→T diffusion。v2覆盖基类的zero辅助方法，计算并返回加权CL，再由基类forward加到total；日志字段为train_loss_contrastive/train_loss_contrastive_weighted。用户随后确认是理解误会，未报告运行异常或KL新分数；不改变训练配置，不由启动命令反推服务器源码或运行进度。

## 2026-10-27：teacher AA0、CL=1e-3回传T32，PT400/LP附件仅0–89

用户明确本次两份附件为此前从CL=1e-4提高至1e-3的新teacher AA0 PT→LP，登记T32；当前共**32条配置/成绩记录**。两次旧AA0 T30/T31为CL=1e-4的从头PT→LP口述best86.03%/85.87%，保持原身份，不用本次日志替代。新多层diffusion仅准备实现与命令，未预登记T33。

### 日志完整性与LP结果

PT epoch0–399共400行，LP epoch0–89共90行，均连续、无重复且所有数值有限。LP90–99未提供，因此不登记最终LP100的best/last/epoch80–99均值，不根据附件结束推断服务器进程状态。LP可训练参数全90轮384060。

| 指标 | T32 AA0 / CL1e-3 |
|---|---:|
| 已收到0–89的LP best | **85.286268831%@72** |
| 最新已收到epoch89 | 85.219553633% |
| epoch70–89 mean | **85.122816606%** |
| epoch70–89 population std | .080791743pp |
| epoch70–89 train CE / test CE | .350636187915 / 4.754225385513 |

同一LP范围0–89的best及共同70–89窗口比较（不把其他组90–99混进来）：

| 对照 | 对照0–89 best (%) | T32−对照best (pp) | 对照70–89均值 (%) | T32−对照均值 (pp) |
|---|---:|---:|---:|---:|
| T22 原生dec3 | 85.456089296 | −.169820464 | 85.381186333 | −.258369727 |
| T26 原生dec3 repeat | 85.710819994 | −.424551163 | 85.452753533 | −.329936927 |
| T23 文本no-share dec3 | 86.280931625 | −.994662794 | 86.076540547 | −.953723942 |

已有90轮中相对T26高26轮/低64轮，相对T22高29轮/低61轮；70–89全部20轮均低于T22/T26/T23，不只是best少一个峰值。T22完整LP100 best85.462154313%@90不用于上表0–89配对。T32当前best对旧CL1e-4两次口述best平均85.95%暂低.663731169pp；旧两次无逐epoch日志，不能当同窗口重复或纯权重因果估计。

### PT目标确为AA0、CL1e-3

400轮均满足total=native+.001*contrastive，最大绝对残差**6.784075159e-10**；AA虽记录原始数值，未加入总loss。PT末20（380–399）total=.017406733829、native=.016965798015、raw CL=.440935797751、raw AA=1.969242851922。加权CL=.000440935798，逐epoch CL占total比例平均约2.533223%；这只是loss数值占比，不是encoder梯度占比。native日志本身已包含骨架uniformity，不能从日志进一步拆分。

PT末轮total=.017552777118/native=.017111399853/raw CL=.441377260053。总loss与CL下降并进入平台，未见日志可见的发散。PT末20 native比T26同期原生total=.016885736994高.474134%，重建项接近但LP更弱；CL优化更充分不等于动作线性可分性提高。T32与T26已有同epoch PT400/LP90的LR逐项完全相同，不能用少跑PT或不同LR解释；也不能由epoch汇总证明所有microstep状态。

T32的70–89 train CE比T26的.366092858179低，而test CE高于T26的4.636991416509，说明该LP窗口测试泛化较弱；不足以确诊encoder过拟合、表示坍塌或梯度冲突。日志没有分支梯度/特征方差，也没有旧AA0两次的原始日志，本次仅说明提高CL权重没有在已收到LP中带来收益。新组按此前给定AA0命令预期native dec3/A/person0/noAMP；实际Namespace、服务器源码/数据/checkpoint SHA及LP加载输出仍缺，历史T22/T26为AMP协议，保留精度边界。

日志峰allocated7142.120117MiB（约6.975GiB）/reserved7736MiB（约7.555GiB）。无耗时字段，不推测吞吐或teacher独立开销。

### 证据与后续已授权实现

原始附件：[T32 PT](/C:/Users/97537/.codex/attachments/06ef643d-6580-4184-aa55-a4fcac4c7f9b/已粘贴的文本.txt)、[T32 LP0–89](/C:/Users/97537/.codex/attachments/df64ec6a-93b4-4901-b433-aebaf605c500/已粘贴的文本.txt)。SHA256：PT=e40e5a9e36da1fd18f3a78c41ae0161171c89d8f7b756c432e72d47ae8b6c7e5；LP=6654f381008f8587b257053a85d2cf324c4d852ff478f95c04d343a67c11ea80。本次直接解析原附件，不复制日志或新建归档目录。

用户已要求完成“后三层监督→第8层LP”：[方法/运行命令](/D:/program/MacDiff/tools/MACDIFF_MULTILAYER_DIFFUSION.md)，encoder6/7/8共享原self.norm及同一原生decoder3、分别前向监督，不新增参数，共享噪声/t/mask/condition dropout，L=L8+.1*(L6+L7)/2+.02*uni8，LP维持现有第8层读出。当前仅代码/配置/验证与命令就绪，未启动训练或登记该方法成绩。固定预训练教师蒸馏等待用户参考文献，未接入。

## 2026-10-26：按原生MacDiff decoder深度重分四组，新增teacher AA0两次从头结果T30/T31

用户明确：dec3/dec5均指**原生MacDiff骨架重建decoder的decoder_depth**，不按S→T的text_decoder_depth、T→S读取器或EMA teacher层数分类。三元深度仍按native/T→S/S→T书写：T15的3/3/5归文本dec3，T24的5/5/3归文本dec5。T25为文本一致性变体，归文本dec3并保留方法标签；teacher和OTFM另列，不混进纯原生/文本扩散四组。

### 四组全部历史成绩与平均

每个既有配置/成绩ID等权取一次，平均=该组已报LP分数之和/记录数。**这是不同配置的历史描述统计，不是同配置独立PT重复的均值，也不能直接用两组均值相减归因文本或decoder深度。** T19与T18共用PT，保留为不同LP配置记录；T20同一PT的两次逐字节相同LP仅计一次。早期文本结构按归档实现/配置归类，实际服务器args仍不完整。

| 组别（按native decoder） | 记录数 | 已记录LP分数平均 (%) | 有末20记录数 | 末20均值的组内平均 (%) |
|---|---:|---:|---:|---:|
| 原生MacDiff dec3 | 6 | 85.6992 | 5 | 85.5429 |
| 原生MacDiff dec5 | 2 | 85.5926 | 2 | 85.4807 |
| 文本 dec3 | 4 | 85.8552 | 1 | 86.0899 |
| 文本 dec5 | 13 | 85.3934 | 5 | 85.7789 |

主均值保留全部历史已报数值：T05只有约83.7%，因此文本dec5平均也只能视为近似；T15/T16/T19旧口述尚未确认best/last，表头使用“已记录LP分数”而非全部承诺为best。若排除这三条口径未确认记录，原生dec3为85.6611%（5条），文本dec3为85.8605%（2条，T23/T25）。T13为归一化B及大batch变体，仍按native深度归原生dec5；其中原生A的明确dec5成绩是T20=86.0019%。

“末20”统一为LP epoch80–99：只对该组有该字段的记录求均值，缺失不补零，样本数见表。它与跨记录LP分数平均是不同指标，单run轮间std也不是独立训练方差。

### 原生MacDiff dec3：逐条记录

| ID | 配置/证据备注 | 已报LP (%) | epoch80–99均值 (%) |
|---|---|---:|---:|
| T14 | PT500，总batch128，LP总256，floor5e-4 | 85.7048 | 85.6359 |
| T17 | PT500，不旋转，PT总128/LP总256，floor1e-5 | 85.6380 | 85.5577 |
| T18 | PT500，旋转，PT总128/LP总256，floor1e-5 | 85.7897 | 85.6841 |
| T19 | 复用T18的PT，只把LP总batch改128；口述，best/last未确认 | 85.8900 | 未提供 |
| T22 | PT400，总64，floor5e-4；历史官方隔离入口 | 85.4622 | 85.3918 |
| T26 | PT400，总64，floor5e-4；seed0从头repeat，根入口 | 85.7108 | 85.4452 |

### 原生MacDiff dec5：逐条记录

| ID | 配置/证据备注 | 已报LP (%) | epoch80–99均值 (%) |
|---|---|---:|---:|
| T13 | 归一化B；PT400，总batch256，floor5e-4 | 85.1832 | 85.1110 |
| T20 | 归一化A；PT400，总64，floor1e-5；同PT重复LP去重 | 86.0019 | 85.8503 |

### 文本 dec3：逐条记录

| ID | 配置/证据备注 | 已报LP (%) | epoch80–99均值 (%) |
|---|---|---:|---:|
| T15 | 七句no-share 3/3/5；口述，best/last未确认 | 85.8400 | 未提供 |
| T16 | 七句no-share 3/3/3；口述，best/last未确认 | 85.8600 | 未提供 |
| T23 | 七句no-share 3/3/3；PT总64，floor5e-4，remap2048/text uni0 | 86.2809 | 86.0899 |
| T25 | 文本nscale双向骨架一致性，native/T→S=3/3；口述LP best | 85.4400 | 未提供 |

### 文本 dec5：逐条记录

| ID | 配置/证据备注 | 已报LP (%) | epoch80–99均值 (%) |
|---|---|---:|---:|
| T02 | 旧BPE；首版双向learned remap，S→T=1 | 83.0300 | 未提供 |
| T03 | 旧BPE；fixed CLIP原尺度，S→T=1 | 84.6400 | 未提供 |
| T04 | 旧BPE；fixed CLIP原尺度，S→T=.1 | 85.9500 | 未提供 |
| T05 | 旧BPE；fixed CLIP RMS，S→T=1；约数 | 83.7000 | 未提供 |
| T06 | 旧BPE；fixed CLIP RMS，S→T=.1；LP总64 | 85.8200 | 未提供 |
| T07 | 旧BPE；参数EMA remap，no-share，.1/.1 | 85.7700 | 未提供 |
| T08 | 旧BPE；参数EMA remap，shared，.1/.1 | 85.7800 | 未提供 |
| T09 | 七句v3；sample target/shared/H256/LN，.1/.1 | 85.8018 | 85.6396 |
| T10 | 七句v3；sample target/shared/H256/LN，1/.1 | 85.8928 | 85.7054 |
| T11 | 七句v3；shared/H512/none，1/.1；LP口述 | 85.7900 | 未提供 |
| T12 | 七句v3；no-share 5/5/5，PT总128，floor1e-5 | 86.1960 | 86.0007 |
| T21 | 七句no-share 5/5/5；PT总64，floor1e-5，remap512/text uni.02 | 86.2203 | 86.1345 |
| T24 | 七句no-share 5/5/3；PT总64，floor5e-4，remap2048/text uni0 | 85.5228 | 85.4142 |

### 近期七句no-share子集，供当前方案判断

这是按事先定义的“七句v3、sample_target_blend、no-share、非一致性”筛选，不按成绩高低选择。文本dec3保留T15/T16/T23，均值**85.9936%**（3条；前两条口述best/last待核）；文本dec5保留T12/T21/T24，均值**85.9797%**（3条；T24虽然S→T=3，仍必须归native dec5）。两组均存在协议变化，不能当作同配置重复，也不以约.014pp差值确认深度优劣。T09–T12/T21/T24全部七句文本（含shared）的dec5均值约85.9039%。

当前teacher的参照应更聚焦：原生dec3/PT400/总64/floor5e-4的T22/T26 best分别为85.462154313%和85.710819994%，均值**85.586487154%**；它们为同seed历史入口/代码条件下的两次PT记录，不是跨seed方差估计。原生dec5/A的T20为86.001940813%，但floor1e-5且native深度不同，不作teacher单因素基线。历史匹配观察仍为T21−T20 best+.218340643pp；T23−T22 best+.818777312pp；T23−T26 best+.570111631pp。

### teacher AA0：T30/T31，用户确认两次均为从头PT→LP的LP best

| ID | 方法 | LP best (%) | 来源 | last/末20/原始日志 |
|---|---|---:|---|---|
| T30 | teacher AA0，native dec3 | 86.03 | 2026-10-26用户口述，明确从头PT→LP | 未提供 |
| T31 | teacher AA0，native dec3 | 85.87 | 2026-10-26用户口述，明确另一次从头PT→LP | 未提供 |
| 平均 | 两次已确认best的算术平均 | **85.95** | 不等同末20均值 | — |

当前共**31条配置/成绩记录**，不等于31次独立完整PT。两次AA0不是同一PT的重复LP；此前“AA0尚无成绩/仍29条”均为历史状态。已给AA0命令是--lambda_alignment 0、--lambda_contrastive .0001、--disable_amp，配置native decoder_depth=3；实际两次Namespace/seed/服务器源码/checkpoint身份和LP日志未提供，不把预期命令当独立核验的全部实参。

两次best相差.16pp；均值对T22/T26原生dec3均值**+.363512847pp**，对T26单次分别+.319180006/+.159180006pp，均值对T26+.239180006pp。对原生dec5/A T20为−.051940813pp，仅作跨深度数值参考。AA0关闭对齐项、保留EMA全局InfoNCE，当前显示小幅正向观察，不能将其归为AA收益；历史原生T22/T26采用AMP、AA0给定协议关闭AMP，同精度原生dec3仍缺结果，尚不能确认稳定的方法净收益。

### 四组之外及证据索引

- T01：原生历史约85.86%，未保存可确认的decoder深度，暂不硬归dec5；若之后确认dec5，包含T01/T13/T20的分数均值约85.6817%。
- T27/T28：teacher v2 AA.1、v3 AA1，分别仅收到LP0–95/0–89，不与新AA0混算均值。
- T29：OT条件FM/native dec3，best84.631246997%，作为不同目标的方法另列。
- T30/T31：teacher AA0，不并入“原生无辅助”的四组平均。
- 第三方86.2%不是本项目独立实验，不并入；仅有命令、未收到成绩的新5/5/5或新读取配置也不并入。

早期结构与深度来源：[历史配置审计](/D:/program/MacDiff/handoff_artifacts/noshare_config_audit_20261008/README.md:25)、[旧交接结构](/D:/program/MacDiff/handoff_artifacts/handoff_before_20261001_refresh.md:101)；T09/T10/T12/T13数值优先用[已有精确审计JSON](/D:/program/MacDiff/handoff_artifacts/experiment_results_audit_20261007.json)，T14用[精确summary](/D:/program/MacDiff/handoff_artifacts/decoder3_20261009/summary.json)，T17/T18用[用户runner摘要](/D:/program/MacDiff/handoff_artifacts/native_rotation_results_20261012/user_summary.json)。其他各ID完整历史身份及附件入口保留于下文。本次只补统计及口述记录，未修改训练代码/配置或启动训练。

## 2026-10-23：原生OT条件Flow Matching回传T29，PT400/LP100

用户明确本次附件为OTFM，登记T29，当前共29条配置/成绩记录。预期身份为此前已选`conditional_ot_noise_to_data_v2`、sigma_min=1e-4、encoder8/decoder3、A/person0、PT关闭AMP、PT400/global64/min_lr5e-4、LP100/global128、seed0和checkpoint399；实际Namespace、服务器源码/checkpoint SHA及LP加载输出未提供，不能仅由epoch JSON确认全部实参。不是mini-batch OT、文本或teacher组。

直接解析：PT0–399/LP0–99连续、无重复、所有数值有限；PT/LP逐epoch LR分别与T26完全相同，LP可训练参数始终384060。PT满足total=FM+.02*uniformity，最大绝对残差1.315e-9。比较同架构历史T22/T26，不拿native5 T20或no-share T23作为单因素基线：

| 指标 | T22原生dec3 | T26原生dec3 repeat | T29 OT-FM dec3 |
|---|---:|---:|---:|
| LP best (%) | 85.462154313 @90 | 85.710819994 @73 | **84.631246997 @90** |
| LP last (%) | 85.334788956 | 85.419699192 | 84.588791878 |
| LP80–99 mean (%) | 85.391800113 | 85.445172265 | **84.512372650** |
| LP80–99 population std (pp) | .045036580 | .054504224 | .055134978 |
| LP80–99 train CE | .334522595087 | .352072116944 | .315357905005 |
| LP80–99 test CE | 4.684069968878 | 4.627313369447 | 4.322367658325 |

相对T26 best−1.079572997pp、末20−.932799616pp，100个共同LP epoch全部较低；相对T22分别−.830907316/−.879427463pp，不能只归为某一轮峰值波动。T29训练/测试CE反而较低，准确率与CE衡量不同，不能单凭掉点诊断过拟合。T29预期全程noAMP，历史T22/T26为此前AMP协议；已安排的同精度native3结果仍未回传，不能把方法/精度/RNG差异全部归为FM。

PT末20：total=.088636326182、velocity MSE=.087960651531、uniformity=.033783732637；末轮分别.089201045443/.088526128904/.033745826091。训练明显下降并进入平台，未显示发散；速度目标与epsilon目标不同，不将其与T26 total=.016885736994直接相除解释表示差距。峰allocated7025.699219MiB/reserved7688MiB，无耗时数据。

**代码审查与优先诊断：**路径和导数对应[原FM论文公式20–23](https://arxiv.org/html/2210.02747v2#S4.SS1)：`x_t=s(t)*epsilon+t*x`、`u=x-(1-sigma_min)*epsilon`，连续时间只在embedding乘999。暂未发现方向、速度倍率或loss配方的明显错误；不等于验证服务器版本/真实autograd。OT轨迹/生成质量的优点不保证冻结encoder的LP提高。

从生产`get_named_beta_schedule`/`betas_for_alpha_bar`用NumPy重算inverse_cosine(1)的1000个alpha_bar，与当前uniform t的OT系数比较，定义系数SNR为data系数²/noise系数²（不含实际骨架方差）：

| 系数SNR区间 | 原生inverse_cosine | OT-FM uniform t |
|---|---:|---:|
| <.1 | 4.9% | 24.0259% |
| .1–10 | **90.2%** | **51.9546%** |
| >10 | 4.9% | 24.0195% |

OT比例用100万均匀t中点数值积分、sigma_min=1e-4；区间是解释分布差异的示例阈值，不是已验证的最佳学习窗口。均匀采样时间不保留相同噪声难度分布。此外，同一OT路径令`epsilon_hat=x_t-t*v_hat`，有`||epsilon_hat-epsilon||²=t²*||v_hat-u||²`：无权velocity MSE与无权epsilon MSE的时间权重不同，不能当成只换目标名称。以上是优先可检验因素，不是掉点因果证明；需固定batch/t/noise/mask按时间分桶，比较正确条件与置零/打乱条件的损失增量、encoder梯度，才可判断decoder是否绕过encoder。当前汇总不能确认旁路或坍塌。

建议先看已安排的同精度native3，再考虑保持OT路径/速度目标、仅匹配噪声难度采样的单因素对照；未实现新采样器、改权重/归一化、接文本或启动新训练。本地FM/累积回归21项为16通过/5缺Torch跳过；完整训练/LP回传不等于那5项服务器单测已提供。

原始附件：[T29 PT](C:/Users/97537/.codex/attachments/cd86ea7e-aa04-4ca4-bfbe-4155fa9f2f0a/已粘贴的文本.txt)、[T29 LP](C:/Users/97537/.codex/attachments/0555937b-0803-4272-bbc8-5aa4461554a5/已粘贴的文本.txt)。SHA256：PT=`24ec6782257bdd33ff400d66566282a79dfacd76595a4a3b77ccd3af88d11c34`；LP=`e0ce269c4002d2d1249ffff5d2f48df3ecc19b33599f126e84e5bc96e95faf95`。未复制日志或新建归档目录；以下28条等表述为对应日期历史状态，以本节29条为准。

## 2026-10-22补充：输入归一化实测是诊断，Stage1成绩仍28条

用户回传六数据协议训练集统计控制台；当前NTU60 XSub PT person0/model_input/model_positions的mean=`[-.003405887,-.220585,-.04651074]`、var=`[.04780319,.159242,.0464151]`，旧A后normalized mean=`[.01668058,-.3076392,-.1483983]`、std=`[1.523333,1.406471,1.459155]`；方差/A约2.3205/1.9782/2.1291。40091条训练样本完整，one-person含空位置与排空结果接近，all两槽约40.9593%位置为空人物帧。该附件仅显示model_input/当前self_shift，seed0每样本一次变换、约7位有效数字；未上传完整JSON/CSV或数据，不称本地独立重算。

这不证明改归一化会提高LP，也不能解释同样使用A的原版/文本组差异或T28掉点；T13/B还同时改batch和LR，不是本次新统计的验证。PKUv2 PT/LP常数不一致、PKU非前缀padding和六协议完整统计已写入[归一化审查](D:/program/MacDiff/tools/vlm_pilot/INPUT_NORMALIZATION_AUDIT.md:3)。[控制台原附件](C:/Users/97537/.codex/attachments/d7a88672-8604-41db-aa66-e513fd8c1e1e/已粘贴的文本.txt)，SHA256=`75627e10b77ea1a2ab2fb20adfb7b728e8395556acc7577e6e34e5173668327f`。不新增实验编号，不改训练配置或旧checkpoint的LP归一化，未启动新训练。

## 2026-10-21最新回传T28：teacher v3、AA=1，checkpoint-80后关闭AMP续训

用户已确认：teacher v3、AA=1，从checkpoint-80恢复后关闭AMP，登记为T28。PT第81–399轮共319行、LP第0–89轮共90行，均连续无重复、所有上传数值有限；PT0–80和LP90–99未提供。附件只有JSON，仍缺实际Namespace、服务器源码/数值补丁版本、checkpoint SHA及LP实际加载输出。**T28是原v3运行的精度变更续训，不是新增独立初始化、AA0组、从头FP32或完整PT400/LP100附件，也不据附件结束判断服务器状态。**当前共28条配置/成绩记录。

319轮均满足`total=native+1e-4*CL+1*AA`，最大绝对残差2.585072e-9。AA为0时即便日志保留原始AA值，total也应只含native+1e-4*CL；这组total实际包含完整AA，不能当新给的AA0命令结果。PT已有319轮LR和LP已有90轮LR分别与T26同epoch逐项完全相同，LP可训练参数始终384060。

比较固定共同LP0–89范围、后期窗口70–89，不把它与其他组80–99混比：

| 指标 | T28 teacher v3 AA=1 | T26原生dec3 repeat | T27 teacher v2 |
|---|---:|---:|---:|
| 截至89轮best (%) | **69.802280459 @87** | 85.710819994 @73 | 85.346918988 @74 |
| epoch89 acc (%) | 69.505094626 | 85.365114041 | 85.140708422 |
| 70–89 mean (%) | **69.109958775** | 85.452753533 | 85.235019429 |
| 70–89 population std (pp) | .526551330 | .089002443 | .067277280 |
| 70–89 train CE | 2.932905349 | .366092858 | .344577169 |
| 70–89 test CE | 5.287712235 | 4.636991417 | 4.605997929 |

相对T26，同范围best−15.908539535pp、70–89均值−16.342794758pp；相对T27分别−15.544638529/−16.125060655pp。T27完整已有0–95的best85.377244084不用于本次0–89配对。新LP从epoch0的44.019893264提高到69.802280459，但已有90轮整体明显弱于两组历史结果；缺最后10轮，不登记最终LP100分数。epoch内std不是跨训练方差，分差也不能单独证明某个loss梯度冲突。

PT末20固定380–399：

| 分支 | 未加权均值 | 加权均值 | 占total数值 |
|---|---:|---:|---:|
| native（含.02骨架uniformity） | .022601757304 | .022601757304 | 2.615312% |
| InfoNCE | 3.677871770895 | .000367787177 | .042558% |
| AA | .841239377104 | .841239377104 | **97.342131%** |
| total | .864208921464 | .864208921464 | 100% |

AA约为native的37.2201倍，**均为loss标量占比，不是encoder梯度占比**。相对epoch81，末20 AA下降40.8602%、CL上升40.4925%（2.617841290→3.677871771）；total下降主要来自AA，而native维持约.022x。末20 native较T26高33.8512%；v2的AA含教师侧FC，当前v3使用直接教师目标，不能把两者绝对AA值差直接当成相同目标的收敛优劣。新native仍未拆分epsilon与uniformity。

这些现象提示当前AA配置/联合目标需要优先排查，但缺分支梯度范数/夹角、W/adapter与特征尺度/方差及匹配的AA0对照，不能确认AA梯度主导、冲突、特征坍塌或把全部掉点归因AA/关闭AMP。用户已确认resume80，前81个epoch已受AA=1/原AMP训练，后来关闭AMP并不会撤销早期更新；后续AA0须沿已给命令从头训练，比较时保留精度历史差异。

本地LP源码仍按同名键加载学生，EMA键有独立前缀，encoder冻结、head训练；缺实际启动输出/checkpoint身份，不能证明服务器确实载入了预期权重。train/test CE均高于历史组，不仅凭test曲线诊断过拟合。下一步优先看已选AA0结果和实际加载信息，不自动调权重、重训或扩大矩阵。本次只分析/记录，未改训练代码。

日志峰allocated7142.120117MiB（6.974727GiB）、reserved7736MiB（7.554688GiB）；不同run整体峰差不能隔离AMP或teacher独立开销，日志无耗时字段。原始附件：[新PT81–399](C:/Users/97537/.codex/attachments/69c66ad7-e69b-4f32-aa0c-944c9384a6ea/已粘贴的文本.txt)、[新LP0–89](C:/Users/97537/.codex/attachments/69f299ee-0550-4a25-ac55-2a4622db49c9/已粘贴的文本.txt)。SHA256：PT=`ebf703d1f56ca1b892284db1683010478ebe4216056540d22de208ff84c91193`；LP=`230f728f205d7153ca79ba2eb51a6902f739c580c229994b62d2f25489819375`。未复制日志或新建目录。

## 以下为此前登记与历史快照

**此前未完成状态：**v3（无教师adapter、AA=1）在PT epoch89约600/626后报native=NaN退出，失败batch CL=2.761038065/AA=1.763285041有限，LP未执行。随后提供FP32防护、关闭AMP及resume80命令；10月21日已确认顶部T28为v3/AA1/resume80后关闭AMP，上传PT81–399/LP0–89，实际服务器补丁及LP加载输出仍未提供。不得继续把“尚无任何续跑日志”当作当前状态，或把新附件当从头AA0。详见[方法文档](D:/program/MacDiff/tools/MACDIFF_SKELETON_TEACHER.md)与handoff。

2026-10-07快照有13组数值成绩，其中4组有完整PT400+LP100附件，1组有完整PT但LP只有口述，8组主要保存历史成绩摘要。另有9项Stage1状态记录，不能统一算作已完成的独立训练。当前28条配置/成绩记录（含T25口述、T26原版dec3完整重跑、T27教师PT400/已收到LP96轮及T28 v3续训PT81–399/LP90轮），不代表28次独立从头训练或完整PT→LP100。最高LP best为T23的86.280932%；T21旧5层仍有最高的已知后期LP均值。S→T=0.5仅为历史候选。

`T→S`为文本到骨架任务权重，`S→T`为骨架到文本任务权重。`H/norm`是S→T decoder的隐藏维度/末端归一化，与CLIP缓存维度不同。`share=True`共享原生骨架重建与T→S骨架decoder主体；S→T文本decoder独立。

**2026-10-20三组复核进度：**原版decoder3完整PT→LP已回传，登记T26；原版decoder5/最佳no-share复跑尚未收到成绩，不推断服务器是否已启动。三组仍按用户选择保持seed0、PT min_lr统一5e-4/LP0；原版5层是新增统一floor基线，不是历史T20的1e-5重复。共同PT400/总64、LP100/总128、各新目录/目标库、根入口/配置，命令见[当前计划顶部](D:/program/MacDiff/tools/vlm_pilot/STAGE1_TEXT_EXPERIMENT_PLAN.md:3)。同seed重复不估计独立PT seed方差，历史入口/源码变化仍需核对。独立teacher v2现已回传T27（PT400/已收到LP96轮），不混入本组三项对照。

## 2026-10-20：骨架teacher v2回传T27，PT400/LP附件截至95，未显示收益

用户确认使用上次给出的`paper_aa_v2`命令和代码。按该方法记录T27：native dec3 + raw full-context EMA .999 + InfoNCE（温度.07/队列65536/权重1e-4）+ AA .1、无约束W、双侧可训练Linear、无额外LN。PT0–399完整连续，LP附件只有0–95共96轮、无缺口，96–99未提供；**不把本附件当完整LP100，不据此判断服务器停止或仍在运行。**所有上传数值有限，PT400个与LP已有96个epoch的LR分别与T26逐项相同，LP参数384060；实际Namespace/数据/源码/checkpoint SHA及加载路径仍缺。

比较限定为共同LP0–95范围，后期窗口统一76–95：

| 指标 | T26 原生dec3 repeat | T27 teacher v2 | T27−T26 |
|---|---:|---:|---:|
| 截至95的LP best (%) | 85.710819994 @73 | **85.377244084 @90** | **−.333575910 pp** |
| epoch95 acc (%) | 85.522804481 | 85.249878727 | −.272925753 pp |
| epoch76–95 mean (%) | 85.445778765 | **85.220766640** | **−.225012124 pp** |
| epoch76–95 population std (pp) | .060166068 | .066510872 | 非跨seed方差 |
| PT末20 native项 | .016885736994 | .016861288887 | −.144785546% |

共同96轮中teacher的acc高33轮/低63轮，76–95全部20轮均低于T26；因此此次已收到的后期表现也较弱，不能只解释成峰值差异。T27截至95的best相对T22低.084910228pp、相对T23低.903687541pp。T27在76–95的train/test CE分别.335875270446/4.597443325036，低于T26的.358396644044/4.631767558061，但acc仍低，不能仅由CE或PT重建loss判断分类效果，也不能据此直接诊断过拟合。

分支日志符合`total=native+1e-4*InfoNCE+.1*AA`，400轮最大组合残差3.708e-10。native已含.02骨架uniformity，没有拆分epsilon与uniformity。PT末20：

| 分支 | 外层加权前均值 | 加权后均值 | 相对native项数值 |
|---|---:|---:|---:|
| native | .016861288887 | .016861288887 | 100% |
| InfoNCE | 1.152050936098 | .000115205094 | .683252% |
| AA | .000031790329 | .000003179033 | .018854% |

AA epoch0=.587048086281，epoch1=.045770959389，epoch10=.000263573867，epoch20=.000053618220，末20=.000031790329（较epoch0低约18466倍）。InfoNCE前期峰值10.334863195785，末20为1.152050936098，说明该对比目标在被优化；不支持把两个分支统称为未工作，也不等价于动作类别LP改善。loss数值占比不是encoder梯度占比，不据此直接调大权重。

**本次代码复查确认AA存在可绕过encoder的退化解，但尚未证明本次checkpoint发生了该退化。**当前损失比较`A_j(sum_i W_ij S_i)`和`A'_j(detach(T_j))`，两侧adapter一起由优化器更新。两侧Linear的weight/bias都为0时，无论encoder表征是什么，AA都能为0；共同缩小投影也能降低未归一化SmoothL1。这是目标设计上的缺口，不是“没有梯度”或“教师encoder被反传”。教师侧adapter的直接训练是原文未说明细节处的实现选择，不能称作者明确规定；此前只验证梯度路径，未充分识别这个目标风险。

为验证该数学通路，用NumPy后端执行实际`alignment_loss`函数，固定学生/教师特征和W：两侧projection共同scale=1/.1/.01/0时，loss分别=.572135329/.006927092/.000069271/0；仅学生projection置0、教师projection固定时loss=.444690168，保持非零。此为代码目标反例，不是服务器特征坍塌实测。现有日志缺投影输出RMS/样本间方差、adapter/W范数及encoder辅助梯度，因此低AA可能来自更好对齐，也可能来自投影缩小，不能仅凭小loss断言坍塌。

后续应先利用现有checkpoint/固定小批数据检查投影前后RMS、样本方差、W及两侧adapter范数、各分支到encoder的梯度；优先考虑固定教师侧投影以堵住“两边一起缩小”的直接通路，保留双侧FC结构、同位置匹配及.1权重。冻结教师投影也不是整体防坍塌/LP增益保证。本次仅分析，不自动改训练代码、权重、论文未给的更新规则或新增完整训练矩阵。

日志PT峰allocated=6233.940918MiB（6.087833GiB）、reserved=7154MiB（6.986328GiB）；相对T26分别增加205.381836/210MiB。两个run整体峰值之差不能当成教师分支独立显存占用；当前数据不显示显存不足导致此次LP下降。日志没有耗时字段，不能据此比较训练速度。

原始附件：[T27 PT](C:/Users/97537/.codex/attachments/e9ea3649-d741-44b9-ade0-11ae23e6230a/已粘贴的文本.txt)、[T27 LP截至95](C:/Users/97537/.codex/attachments/d1ac388f-d8af-44c8-a6a2-2e792c301db4/已粘贴的文本.txt)。SHA256：PT=`a765a156948ea9b8e2d8c2b5db304b5c4dc29168cf1d9b453abaf8c77dd011c6`；LP=`f90649d2fbddb219e7c12b579693a6335ae6284018859058d9c3d9d6a0cae981`。不复制日志或新建归档目录。

## 2026-10-20：原版decoder3 seed0完整重跑，T26

用户明确本次两份日志是已选`dec3 repeat`，按独立完整PT重跑登记T26；不是T20同一PT的重复LP。身份依据用户说明及已保存命令：根`main_pretrain.py`/`pretrain_madiff_decoder3.yaml`，encoder8/decoder3、A/person0、无PT旋转、PT400/双卡32/accum1/总64、lr.001/min_lr5e-4、seed0；LP100/双卡64/总128/lr.1/min_lr0、加载本组checkpoint399。**日志没有Namespace、服务器源码/数据/checkpoint SHA或实际权重加载路径，以上命令身份不能称全部已独立核验。**

直接逐行解析：PT0–399和LP0–99各完整连续、无重复epoch，所有日志数值有限。PT400个epoch及LP100个epoch的学习率分别与T22逐项完全相同，也分别与T23完全相同；PT380–399为5e-4、LP90–99为0。LP可训练参数全程384060。没有辅助分支拆分loss，PT总loss不能拆成epsilon MSE/骨架uniformity；epoch汇总也不能证明microstep没有AMP跳步。

| 指标 | T22 历史原版dec3 | T26 dec3 repeat | T26−T22 |
|---|---:|---:|---:|
| LP best (%) | 85.462154313 @90 | **85.710819994 @73** | **+.248665681 pp** |
| LP last (%) | 85.334788956 | **85.419699192** | +.084910236 pp |
| LP末20均值 (%) | 85.391800113 | **85.445172265** | **+.053372153 pp** |
| LP末20 population std (pp) | .045036580 | .054504224 | 非跨PT/seed方差 |
| PT末20总loss | .016801853122 | .016885736994 | +.499253692% |
| LP末20train CE | .334522595087 | .352072116944 | +.017549521857 |
| LP末20test CE | 4.684069968878 | 4.627313369447 | −.056756599430 |

T26 PT末轮loss=.017004353167，最低=.016668007081@394；PT loss从epoch0即与T22不同，400轮均非逐项相等，390轮高于T22、10轮低于T22。LP也是100轮均非逐项相等，60轮高于/40轮低于T22，末20轮15轮较高。best提升约.249pp，但末20只提升约.053pp，不能把峰值差直接当成整体表示提升同样多。最佳epoch73之后未维持该峰值；末20仍有小幅正向差异。

日志报告PT CUDA peak allocated=6028.559082MiB（约5.887GiB）、reserved=6944MiB（6.781GiB），400轮记录一致。它是该原生dec3运行的资源记录，不是teacher v2显存实测。

与历史T23 no-share比较，T23仍高于T26：best+.570111631pp、末20均值+.644711320pp。T23相对原配对T22的+.818777pp历史差值保留；不能用新T26覆盖T22或把两种参照混作同一配对。no-share复跑尚无新成绩，现阶段不确认其稳定增益。此次T26是同seed且历史入口由官方隔离入口转根入口的复核，不是独立seed试验，不能由这两次估计PT随机方差、把差异全归随机性或确认某个代码改动为因。

原始附件：[T26 PT](C:/Users/97537/.codex/attachments/eb43956c-dde5-4142-a6a8-fd0c3b8db0bb/已粘贴的文本.txt)、[T26 LP](C:/Users/97537/.codex/attachments/7311faea-8fa7-44fe-bdf2-34538cb40c70/已粘贴的文本.txt)。SHA256：PT=`d6dd57876460850322d00d31a033312cc1a395ef120977a9511cfce90eaad7a1`；LP=`c5a1a8616ceacaaa434fb51a7a0adbeb12530786d71630ed9aa8c14ab5c0a6ff`。本次仅更新现有结果/计划/handoff，不复制日志或新增归档目录，不改训练代码、不启动训练。

## 2026-10-19：nscale双向骨架一致性T25，口述LP best85.44%

用户明确确认85.44为噪声尺度归一化后的新一致性组LP最高值，native/T→S各3层、native/T→S/consistency权重1/1/1。登记T25，来源为聊天口述，无新附件；不填best epoch、last、末20均值或完整PT/LP轮数，不称实际服务器模型/优化参数、checkpoint加载和数值稳定性已独立核验。期望运行身份及命令见[现有计划顶部](D:/program/MacDiff/tools/vlm_pilot/STAGE1_TEXT_EXPERIMENT_PLAN.md:3)，不能用本地YAML替代完整日志。

| 记录 | LP best (%) | 与T22原版3层之差 (pp) | 证据 |
|---|---:|---:|---|
| T22 原版3层 | 85.462154313 | — | 完整epoch日志 |
| T23 旧no-share 3/3/3 | 86.280931625 | +.818777312 | 完整epoch日志 |
| T25 新nscale一致性 | 85.44 | 约−.02 | 用户确认的口述最高值 |

T25比T23约低.84pp，本次没有显示LP收益。修复高噪声x0还原放大只解决显式loss尺度问题，不构成表示质量或语义传递有效性的证据；完整日志未提供，不宣称NaN问题已彻底消失或训练稳定已确认。

用户质疑全部收益可能为原生波动。当前缺相同协议的独立PT重复，不能估计原生训练方差，也不能把T23+.82pp直接确认成稳定方法收益或直接抹为噪声。T20同PT/同seed两次LP逐epoch一致，说明该组LP可复现，不能估计PT随机性；单次LP末20轮std不是跨训练方差。不安排用户已拒绝的seed1，不重复已做的固定CLIP/条件打乱诊断。当前保留T23为候选、建议暂不扩展一致性；未修改代码/配置、未启动新实验。

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

## 近期结果统计（2026-10-21补T28）

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
| T25 | nscale一致性，口述 | 85.44 | 未提供 |
| T26 | 原版3层seed0 repeat，PT400，floor5e-4 | 85.7108 | 85.4452 |
| T27 | teacher v2，PT400，LP附件截至95 | 85.3772（截至95） | 未齐；76–95为85.2208 |
| T28 | teacher v3 AA1，从80关闭AMP续训 | 69.8023（截至89） | 未齐；70–89为69.1100 |

no-share深度按native/T→S/S→T排序；T15/T16/T19为用户口述，无完整末20统计。完整LP末20指80–99；T27仅收到0–95、T28仅0–89，各自76–95/70–89值不能与其他行80–99直接配对，同窗口比较见顶部。T17/T18有summary，T14/T20–T24/T26有完整日志；不同PT轮数、batch、floor、remap和text uni使这些行不构成统一协议消融。T21 remap512/text uni.02，T23 remap2048/text uni0。当前最高best为T23，最高已知末20均值为T21。

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
