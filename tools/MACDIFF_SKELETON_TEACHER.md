# Native MacDiff：完整骨架EMA教师、全局对比和位置对应的跨层对齐

2026-10-19：按用户要求实现骨架侧方法，不使用文本或预训练教师。LP只评估学生encoder。它借鉴Gong等的EMA全局InfoNCE和ACA²Net的跨层混合；不是ACA²Net原样复现。

## 2026-10-27：AA0 / CL=1e-3新结果T32，LP附件截至89

用户确认提高CL权重的新PT→LP。本次PT400完整，LP只收到0–89共90轮，连续有限；已收到best**85.286268831%@72**、epoch89=85.219553633%、70–89 mean**85.122816606±.080791743%**。缺最后10轮，不写最终LP100或epoch80–99均值。旧CL1e-4的T30/T31仍为86.03%/85.87%、平均85.95%，不被新日志覆盖。

total=native+.001*CL，400轮最大残差6.784075159e-10，确认AA系数0/CL1e-3的损失组合；raw AA可非零，但不贡献总loss。末20 raw CL=.440935797751，乘权重后.000440935798，占total平均2.533223%；不把标量loss占比当梯度占比。

相对T26原生dec3同范围0–89 best−.424551163pp、70–89均值−.329936927pp；70–89每轮都低。相对旧AA0口述best均值暂低.663731169pp，但覆盖范围和旧完整日志不足。PT native与T26接近（+.474134%）而LP较弱，本次没有显示加大CL权重的收益；梯度冲突/坍塌未证实。当前32条成绩记录，完整来源/SHA与比较见[Stage1顶部](/D:/program/MacDiff/STAGE1_EXPERIMENT_RESULTS.md:5)。

用户选择转向单独测试[后三层完全共享norm/decoder监督→第8层LP](/D:/program/MacDiff/tools/MACDIFF_MULTILAYER_DIFFUSION.md)，该新方法不接EMA/CL/AA。本次没有改变teacher默认配置或启动训练。

## 2026-10-26：AA0两次从头PT→LP已回报，T30/T31

用户确认86.03%/85.87%分别为两次从头PT→LP的LP best，登记T30/T31，平均**85.95%**、两次相差.16pp。不是同一PT重复LP；无新日志，last/末20/实际seed/Namespace/checkpoint身份未提供。以下旧“AA0未回传”均为对应日期状态。

已给AA0命令关闭对齐项、保留EMA全局InfoNCE（lambda_contrastive=1e-4），native decoder_depth=3，预期从头noAMP；不能拿alignment_teacher_layers=5或别的decoder深度把它归成dec5。主要数值参照为同native dec3/PT400/总64/floor5e-4的T22/T26：best85.462154313%/85.710819994%，均值85.586487154%，因此本次AA0平均**+.363512847pp**；相对T26分别+.319180006/+.159180006pp，均值+.239180006pp。

AA0与原生dec5/A T20=86.001940813%相比为−.051940813pp，仅作跨深度参考。历史原生使用AMP，AA0给定协议关闭AMP，同精度原生dec3尚缺结果；当前是小幅正向观察，不能确认稳定方法净收益、归为AA对齐收益，或把两次LP best均值当epoch末20均值。完整四组分类及31条记录状态见[Stage1结果顶部](/D:/program/MacDiff/STAGE1_EXPERIMENT_RESULTS.md:5)。本次未改模型/配置或启动训练。

## 2026-10-21回传T28：v3 AA1，从80关闭AMP续训，LP截至89 best69.8023

用户已确认teacher v3、AA=1、从checkpoint-80恢复后关闭AMP，登记T28。PT81–399共319行、LP0–89共90行，连续且所有上传数值有限；不是完整0–399/0–99日志或独立从头FP32。total符合native+1e-4*CL+AA，不能当作下方AA0结果。服务器源码/数值补丁版本、checkpoint SHA及实际LP加载输出仍缺。LP截至89 best69.802280459%@87、70–89均值69.109958775；对T26同窗口分别−15.908539535/−16.342794758pp，已有LR相同、参数384060。

PT末20native=.022601757304、AA=.841239377104、CL=3.677871770895；AA占total数值97.3421%，CL较81轮上升40.4925%。当前联合训练表现明显较差，但loss值占比不是梯度占比；未测分支梯度夹角/特征尺度与方差，不能确诊冲突或坍塌。已确认resume80，前期AA1/AMP历史不会因关闭AMP而消失；AA0按下面新目录从头训练，比较时保留精度历史差异。详情、原附件/SHA见[Stage1最新记录](D:/program/MacDiff/STAGE1_EXPERIMENT_RESULTS.md)。本次只分析记录，没有修改模型或启动新实验。

## 2026-10-21已选：独立AA=0对照；repeat只指原生3/5层与no-share

用户明确repeat指原生decoder3、decoder5和最佳no-share，不是teacher；这三组关闭AMP的独立命令在[Stage1计划](D:/program/MacDiff/tools/vlm_pilot/STAGE1_TEXT_EXPERIMENT_PLAN.md)顶部。用户随后回传并确认的teacher AA1续训结果已登记顶部T28；独立AA0尚未收到结果，当前仍不能确认梯度冲突。

本次独立teacher命令显式`--lambda_alignment 0 --lambda_contrastive 0.0001 --disable_amp`，从头seed0/PT400→LP100，新目录、无resume；不把已受AA=1训练的encoder恢复成所谓纯AA=0对照。默认YAML和模型AA仍1，仅CLI覆盖为0，保存的model_args将记录0。当前实现为保持DDP图仍计算AA，日志loss_alignment可非零，但外层贡献和有效AA梯度为0；不承诺因此省去计算/显存。其他项保留native epsilon+.02骨架uniformity、全局InfoNCE、完整EMA教师。AA0结果可以检验AA在匹配协议下的净影响，不能单凭LP差异证明梯度冲突。

公共main_pretrain已新增`--disable_amp`，PT scaler统一跟随该开关；默认native/no-share仍AMP=True，teacher仍False。LP协议保持原样。助手未启动训练，AA0未预登记成绩ID。

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10299 main_pretrain_skeleton_teacher.py --config config/ntu60_xsub_joint/pretrain_madiff_skeleton_teacher.yaml --disable_amp --lambda_contrastive 0.0001 --lambda_alignment 0 --batch_size 32 --accum_iter 1 --epochs 400 --warmup_epochs 20 --min_lr_epochs 20 --lr 0.001 --min_lr 0.0005 --mask_ratio 0.9 --motion_aware_tau -1 --seed 0 --output_dir output_dir/ntu60_xsub_skeleton_teacher_v3_aa0_dec3_noamp_seed0_20261021/pt --log_dir output_dir/ntu60_xsub_skeleton_teacher_v3_aa0_dec3_noamp_seed0_20261021/pt/tensorboard && OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10300 main_linprobe.py --config config/ntu60_xsub_joint/linprobe_madiff.yaml --finetune output_dir/ntu60_xsub_skeleton_teacher_v3_aa0_dec3_noamp_seed0_20261021/pt/checkpoint-399.pth --batch_size 64 --accum_iter 1 --epochs 100 --warmup_epochs 0 --min_lr_epochs 10 --lr 0.1 --min_lr 0 --seed 0 --dist_eval --output_dir output_dir/ntu60_xsub_skeleton_teacher_v3_aa0_dec3_noamp_seed0_20261021/lp --log_dir output_dir/ntu60_xsub_skeleton_teacher_v3_aa0_dec3_noamp_seed0_20261021/lp/tensorboard
```

## 2026-10-20此前：用户选择关闭整个teacher PT的AMP

teacher入口和专用YAML默认`enable_amp=False`，新增`--disable_amp`显式命令行开关；该开关可覆盖配置中的True，不能写成旧`--enable_amp False`。当前学生encoder、完整EMA teacher、native decoder及所有loss均不使用FP16 autocast；保留上一节数值补丁的decoder/loss FP32包装和错误诊断。AA=1、InfoNCE1e-4、EMA/队列、batch、LR和400轮日程均按既定设置。

公共`NativeScalerWithGradNormCount`增加默认True的可选`enabled`参数；PT主入口仅对`supports_skeleton_teacher`模型令它跟随AMP开关，其他方法保持原scaler选择。当前teacher PT为`GradScaler(enabled=False)`，不放大loss/梯度；完整优化窗口前检查梯度范数，非有限即在optimizer.step之前抛错，因此失败时不提交EMA/Queue。不自动clip或跳batch。LP仍用原本默认关闭autocast的既有入口，其head/scaler协议未改。

旧v3完整checkpoint仍可恢复；关闭的GradScaler加载旧AMP scaler状态时按[PyTorch1.8.1官方实现](https://raw.githubusercontent.com/pytorch/pytorch/v1.8.1/torch/cuda/amp/grad_scaler.py)不启用缩放。保存args记录`full_fp32_no_amp_v1`，新checkpoint的scaler状态为空字典，模型/optimizer/EMA/Queue继续保存。新默认输出目录后缀`_fp32_noamp`，下方两个命令均显式`--disable_amp`；恢复仍从原始v3 checkpoint-80起（须实际存在且有效），不是从上个未确认执行过的`_nativefp32`目录取权重。精度变化属于continuation，不是原AMP逐位重现。

本地教师11通过/11无Torch跳过，累积5通过，共**16通过/11跳过**。实际parser的配置/CLI覆盖、实际main的teacher/native scaler选择、实际scaler的非有限梯度范数在step前中止控制流已通过；真实Torch旧AMP状态加载/未缩放梯度用例仍缺Torch跳过。当时无关闭AMP后的服务器日志；10月21日已回传顶部T28，记录续训81–399有限和整体显存峰值，仍无耗时或独立显存剖析。关闭AMP不保证排除数据或优化本身的非有限值。

## 2026-10-20此前：v3首次PT在epoch89中断，增加native decoder FP32防护

服务器回传v3命令和异常：epoch89的600/626附近，最后一次打印均值native=.0230、CL=2.6478、AA=1.3380；失败batch报native=NaN、CL=2.761038064957、AA=1.763285040855。该次PT没有完成，不产生LP成绩；`PT && LP`会在PT退出后停止。附带的`target`/`t=36`是原生随机调试输出，不能当成失败batch的精确身份。日志峰值max mem=6227MiB，不是OOM错误。两个辅助loss有限只定位当次故障首先在native合成项显现，不排除AA此前更新改变了encoder/decoder条件，也不证明encoder全部元素有限。

检查原生decoder发现FP16 `q @ k`先于缩放/softmax，FeatureModulation也有多重乘法；有限输入可在FP16溢出。依据[PyTorch1.8.1 AMP算子表](https://raw.githubusercontent.com/pytorch/pytorch/v1.8.1/docs/source/amp.rst)及[局部关闭autocast的官方示例](https://raw.githubusercontent.com/pytorch/pytorch/v1.8.1/torch/cuda/amp/autocast_mode.py)，当时仅在teacher子类覆盖native decoder和diffusion loss精度，调用原生方法前禁用autocast并将输入转FP32；学生encoder和EMA teacher当时仍沿用外层AMP，随后用户选择整体关闭AMP，见顶部。未改原生基类、公共engine、AA=1、归一化、LR、clip或EMA规则。此为针对已知易溢出路径的数值防护，尚无失败batch/checkpoint复现，不能宣称服务器NaN的具体算子已查实或以后不会复发；计算与显存开销待实测。

新增失败诊断区分`decoder_prediction`/`diffusion_loss`/`total_loss`，打印rank、原始sample IDs、输入/输出/条件/t/mask等可用张量的dtype、非有限元素数、有限元素absmax/RMS；native total异常时另列diffusion标量。错误仍抛出，不置零loss、不跳batch、不偷偷重置EMA/Queue。临时诊断引用在finally清理，不增加checkpoint张量或模型参数。

v3模型身份和model_args保持兼容；保存args增加`skeleton_teacher_numerics_version=native_decoder_loss_fp32_v1`记录当前精度策略。**旧v3完整checkpoint可恢复到此数值策略，属于改变精度的continuation，不是原AMP逐位重现；旧v1/v2仍拒绝。**原生每10轮保存一次，epoch89失败时最近预期为旧目录`checkpoint-80.pth`，须以服务器实际存在且内容有效的文件为准；恢复从81开始，`epochs`仍400以保留原LR日程。默认及命令新输出目录后缀`_nativefp32`，避免把续跑日志混进旧日志或覆盖旧权重。恢复命令见下方。

本地教师9通过/10因无Torch跳过，累积回归5通过，共**14通过/10跳过**。新增NumPy执行实际精度包装方法，验证外层AMP状态恢复、输入转FP32及`128×128×4=65536`在FP16溢出而FP32缩放后32768有限；这不是CUDA复现。另新增真实CUDA decoder QK溢出回归和错误诊断测试，因本地缺Torch尚未执行。当前没有服务器补丁后结果，未把这次失败登记为已完成PT/LP或新增成绩ID。

## 2026-10-20：v3方法结构，去掉教师侧adapter，AA权重改为1

按用户明确要求修改：删除教师侧的5个256→256 Linear模块及其参数，直接以`detach(T_j[ids_keep])`为目标；学生侧5个FC、无约束8×5跨层W、同位置token匹配、FP32 SmoothL1及其均值归约保留。模型构造器、入口回退值、YAML和下方PT命令的`lambda_alignment`统一为1.0；InfoNCE仍为1e-4。EMA教师encoder仍eval/no_grad，只在成功完整优化步后更新。

方法身份为`raw_full_position_student_adapter_aa1_v3`，初始运行目录为`ntu60_xsub_skeleton_teacher_v3_student_adapter_aa1_dec3_seed0_20261020`；数值补丁后默认新目录加`_nativefp32`后缀。旧v1/v2配置和checkpoint不作为当前完整resume。下方参数表、数据流、公式和运行命令均对应v3；T27继续保留v2身份，不把旧结果改名为v3。首次v3 PT在89中断，尚无LP；助手未启动训练。

数值补丁前教师检查8通过/8缺Torch跳过，累积回归5通过，共13通过/8跳过；当前计数见顶部。实际`alignment_loss`的NumPy检查使用固定特征/带负权重W：学生混合3/0、原始教师目标.5/1.5，SmoothL1平均为1.5；学生FC置0后仍为.5625，验证教师目标不受可训练目标投影缩小影响。Torch测试已更新为原始教师目标detach、学生梯度、无教师adapter参数和总loss权重检查，尚未实际执行。此改动堵住v2双侧FC共同缩小的直接路径，不保证整个EMA系统不会退化或LP必然提升。

## 2026-10-20历史：v2首组T27回传，AA目标退化风险待查

用户确认运行v2。PT400轮完整，上传LP仅0–95共96轮；截至95 best85.377244084%@90，epoch76–95均值85.220766640。相对原生dec3 repeat T26同范围best−.333575910pp、76–95均值−.225012124pp，后20个共同epoch全部较低。没有96–99日志，不称完整LP100，不推断服务器是否终止。两阶段LR与T26共同范围逐项一致、上传指标有限、分支组合匹配1e-4/.1；实际命令及版本由用户确认，缺完整Namespace和checkpoint身份，完整证据见[Stage1最新记录](D:/program/MacDiff/STAGE1_EXPERIMENT_RESULTS.md)。

PT末20native=.016861288887，InfoNCE=1.152050936098，AA=.000031790329；加权后CL=.000115205094、AA=.000003179033，分别为native项数值的.683252%/.018854%，不是encoder梯度占比。AA从epoch0的.587048086281到末20降低约18466倍，不应把小AA当成学到了更好的表示。

**当时v2允许一个明确的退化解：学生和教师两侧可训练Linear一起输出0，则AA对任意encoder特征都为0。**原文未给教师侧adapter更新规则，而v2采用其跟随AA梯度直接训练，未约束目标投影幅度；这允许绕过encoder，不能仅由“梯度路径正常”认为目标设计可靠。数值验证执行当时实际`alignment_loss`，固定特征和W，仅共同缩小两侧projection：scale=1/.1/.01/0时loss=.572135329/.006927092/.000069271/0，证实该通路。此反例不是本次checkpoint已经退化的实测证据；需读取投影前后RMS/样本方差、W及adapter范数与encoder辅助梯度才能区分投影缩小和真正对齐。InfoNCE在下降，也不能据AA小就声称整个encoder/teacher坍塌。

当时建议先诊断现有checkpoint、考虑固定教师侧projection，分析阶段未修改代码。随后用户明确选择去掉教师侧adapter、AA权重1，当前实现以顶部v3及下方数据流为准。

日志PT peak allocated6233.940918MiB（6.087833GiB）、reserved7154MiB（6.986328GiB），比T26整体峰值多205.381836/210MiB。整体峰值差不等于教师独立开销，本次未见显存不足记录；日志没有耗时字段，不能据此判断训练速度。服务器实际跑出PT/LP指标不代替真实梯度、Queue跨rank一致和AMP跳步测试。

## 2026-10-19历史：按用户要求修订AA为v2

当时用户明确要求：AA除按`ids_keep`匹配同位置token和外层权重`.1`以外，回到原文的结构。当时模型、入口、YAML和测试作了以下修订；当前已由v3替代：

- `alignment_weights`直接参与跨层线性组合，允许正负值，不再softmax；有效`8×5`矩阵对应原文完整S×T矩阵的最后5个目标列。
- 每个目标层同时有学生`A_j`和教师侧`A'_j`两个256→256 Linear adapter，移除原先只在学生侧使用恒等adapter的设计。
- 移除AA额外添加的双侧无affine LN，直接在两侧投影结果间计算FP32 SmoothL1（阈值1），保留batch/token/channel和目标层取均值。
- EMA encoder仍冻结、eval/no_grad。仅其输出特征detach，教师侧adapter在no_grad区域之外执行；两个adapter集合与W都由现有AdamW训练，教师侧adapter不归EMA更新。**原文未明确教师侧adapter的更新规则，这里采用普通可训练投影，不冒称已经验证作者代码行为。**
- 原文未说明W/adapter初始化。当前W采用零均值正态std=.02，不施加同层偏置；两侧Linear沿用native的Xavier uniform/零bias。这是明确记录的默认实现选择，不是论文给出的数值。
- 方法身份改为`raw_full_position_paper_aa_v2`。旧v1配置及checkpoint不能作为v2完整resume；默认输出与命令改为`ntu60_xsub_skeleton_teacher_paper_aa_dec3_seed0_20261019`，防止与v1混用。没有启动训练，也未加入之前仅讨论过的教师batch分块。

10月19日实现阶段教师测试7通过/8缺Torch跳过，累积回归5通过，共**12通过/8跳过**。新增NumPy用例执行实际`alignment_loss`函数，验证带负权重、两侧adapter且无额外LN的手算结果`.875`；它不验证autograd或AMP。真实Torch用例补充两侧adapter梯度、教师encoder无grad、EMA不覆盖AA参数，以及旧v1恢复拒绝。Python3.8语法检查通过；当时无GPU实测，10月20日已有顶部T27的PT/部分LP日志，真实单元测试/状态验证仍未回传。

当时直接核对[PyTorch v1.8.1官方求导定义](https://raw.githubusercontent.com/pytorch/pytorch/v1.8.1/tools/autograd/derivatives.yaml)：`smooth_l1_loss`同时定义input和target的梯度，因此v2可训练教师侧adapter不需要升级Torch来获得target梯度。v3已移除该adapter；源码核对不代替上述真实运行测试。

## 两篇论文参数对照（当前实现列已同步v3）

先前审计重新读取实际源码、两个本地PDF全文，并渲染核对ACA²Net公式、设置、表VI和图4；随后按用户要求迭代AA。**v3保留跨层线性组合与SmoothL1，按用户新选择改为仅学生侧投影，AA名义系数为1。教师原始骨架、同位置token和训练任务仍是MacDiff适配，不称整个ACA²Net等价复现。真实Torch/AMP/DDP验证与恢复校验缺口仍保留如下。**

证据来源：

- Gong等，[本地论文](D:/DL/08371-GongT.pdf)，印刷页3151–3153（PDF第3–5页）；[AAAI官方全文](https://ojs.aaai.org/index.php/AAAI/article/view/32324/34479)。
- ACA²Net，[本地论文](D:/program/paper/Asymmetric_Context-Guided_Adaptive_Alignment_Network_for_Skeleton-Based_Action_Recognition.pdf)，印刷页5941–5945、5947–5949（PDF第3–7、9–11页），式9–11/20、IV-B设置、表VI/IX和图4。
- Gong正文只对部分超参数明示数值，其余称沿用MAMP。以下标为“沿用推定”的内容，依据[MAMP原论文4.2节](https://arxiv.org/pdf/2308.07092)，不是已经拿到Gong运行配置。MAMP正文decoder默认5层，而[官方NTU60 XSub配置](https://github.com/maoyunyao/MAMP/blob/main/config/ntu60_xsub_joint/pretrain_mamp_t120_layer8%2B3_mask90.yaml)是3层，故不能由一句“follow MAMP”断言Gong必为5层或3层。

### 实现正确性与验证缺口

| 检查点 | 当前源码证据与判断 |
|---|---|
| 教师输入是否被归一化两次/污染 | `forward`先`_prepare_target(source)`；该路径clone后归一化，再调用native forward。教师目标不会因native后续输入视图写入而重复归一化；学生/教师读取同crop、同person0 |
| token是否位置错配 | 学生保留实际`ids_keep`；教师完全绕过random_masking，保持temporal-major/joint-minor顺序，再gather。当前默认4帧同网格下位置对应成立 |
| 梯度是否误入教师/decoder | EMA encoder参数requires_grad=False，前向no_grad、始终eval；全局对比只回学生encoder；AA还训练W及学生侧5个adapter，不经过decoder。教师目标直接detach且没有教师侧adapter；native分支仍训练encoder和decoder，这是静态依赖结论 |
| EMA初始化/恢复顺序 | DDP同步后复制当前学生，随后加载resume；已保存EMA不会被初始化钩子覆盖。EMA只复制encoder组件，不复制decoder/AA |
| 累积/AMP跳步 | 完整累积边界且scaler未降scale才commit；尾窗口和AMP跳步不推进EMA/Queue。队列修改在backward之后，没有发现前向后、反向前覆盖负样本tensor的问题 |
| 队列跨rank | 成功step后按相同rank顺序all_gather keys与IDs再入队；默认drop_last及DistributedSampler使各rank批大小一致。队列/指针/ID/count均为可保存buffer；未运行真实多进程通信 |
| 队列冷启动/负样本 | 空队列首窗口CE=0有意如此；同原始sample ID的历史视图被屏蔽。FIFO重复采样本身不是错误，其他同动作样本仍视为负样本；未使用标签 |
| LP是否误读教师 | 学生encoder沿用原键名；EMA权重有`ema_teacher.`前缀，下游strict=False把其列为额外键，不会替换同名学生参数；未实际执行服务器LP加载 |

**恢复校验缺口（不影响当前无resume默认命令）：**`main_pretrain_skeleton_teacher.py`的`_validate_resume_definition`只比较`model`和`model_args`，不核对当前数据身份、`train_feeder_args`、`mask_ratio`、`motion_aware_tau`或训练日程。队列里的ID却是数据集行号；如果换到另一份同形状数据后resume，旧keys/IDs仍会加载，正负样本排除可能对应到不同序列。当前校验API根本没有收到这些当前协议参数，这一缺口可直接由源码确认。若后续修复，应至少绑定数据/行序身份和mask协议；有意更改日程应单独说明，不能称原协议精确续跑。本次只记录，未更改恢复行为。

另需区分：完整state恢复不等于逐位精确续跑。当前公共checkpoint与原生一样没有保存/恢复Python、NumPy、Torch及CUDA RNG状态，这是继承的限制。

修改前v1复跑为11通过/6跳过，v2为12通过/8跳过；当前v3无AMP版以顶部的**16通过/11跳过**为准。tensor用例包括位置、梯度、无AMP/零权重native一致性、EMA/Queue、模拟gather、LP加载、CUDA溢出/诊断及关闭scaler的恢复/梯度检查，均尚未执行；其中gather用例本身也是mock，不是真实DDP。不能把标准库/NumPy检查当GPU验证，改变精度后也不宣称与原AMP native逐位一致。

### 结构、目标和辅助分支

| 项目 | 当前teacher版本 | Gong论文 | ACA²Net论文 |
|---|---|---|---|
| 主要重建任务 | MacDiff diffusion epsilon MSE，另加.02骨架uniformity | masked joint reconstruction + NAT | masked temporal motion reconstruction |
| 教师输入 | 同crop完整原始骨架，A归一化，无人为小扰动 | 完整骨架序列 | 时间运动差分序列 |
| 教师mask | 0，保留750token | 0，完整输入 | .7，保留75token |
| 学生mask | .9，随机，保留75token，tau=-1 | 大比例mask；.9按MAMP沿用推定 | .9，Motion-Aware |
| 教师mask策略 | 无 | 无 | Motion Compass-Aware，运动强度+方向，融合lambda=1 |
| 输入帧数/时间patch | 120；学生4、教师4 | 正文未单列；120/4按MAMP沿用推定 | 90；学生3、教师9 |
| 学生/教师encoder | 8/8层，256维，8heads，FFN1024 | 教师与学生同架构；具体维度沿用MAMP | 8/8层，256维，8heads，FFN1024 |
| decoder | native diffusion D3，256维 | 普通重建decoder；具体深度存在MAMP正文/官方配置歧义 | 普通重建D3，256维 |
| 教师更新 | 学生初始化复制，成功optimizer step后EMA .999 | 明确EMA，momentum=.999 | 正文没有清楚交代EMA/冻结/优化器更新规则，不能据此认定与当前相同或不同 |
| 全局特征 | 最终norm→GAP→L2，无额外projection MLP | GAP后cosine，无需假设另有MLP | 没有本文这一路队列InfoNCE |
| InfoNCE温度/队列/权重 | .07 / 65536 / 1e-4 | .07 / 65536 / 1e-4，三项数值一致 | 不适用 |
| 队列额外规则 | 空队列起步；排除同ID历史视图；成功step入队 | 描述FIFO；未明确空队列初始化、同ID排除或AMP跳步规则 | 不适用 |
| 对齐目标层 | 教师最后5层，即4–8层 | 没有AA分支 | Top K=5，表VI优于1/3/7；该消融使用教师mask .8，不能误作最终默认 |
| 跨层W | 有效8×5，直接线性组合，允许正负且无和为1约束 | 无 | 式9为学生层线性组合；图4所标W含正负值，结构已对齐 |
| W初始化 | 零均值正态std=.02，无同层偏置；学生Linear为native Xavier/零bias | 无 | 未明确初始化；不能把图中后续epoch的数值当初始化 |
| adapter | 仅5个学生侧256→256 Linear，由优化器训练；教师侧无adapter，直接使用detach特征 | 无 | 式9学生A_j、式10教师A'_j，两侧均有adapter；比较Linear/MLP后选择Linear；未明确教师侧adapter更新规则；v3此项为用户选定差异 |
| AA归一化/归约 | 移除额外LN；FP32 SmoothL1，batch/token/channel和目标层均值 | 无 | SmoothL1（阈值1）；没有规定额外双侧LN，完整均值归约未明确 |
| AA外层系数 | 1 | 无 | 式20 L=LAlign+mu*LRecon，mu=1；AA和Recon名义系数均1 |
| token对应关系 | 教师按学生物理位置ids_keep取对应token | 无token AA | 未明确当前这种同网格位置gather；W解决的是跨层聚合 |
| 人物/小扰动/AMP细节 | PT固定person0；学生joint_noise=[1,.005]；当前teacher PT autocast和GradScaler均关闭，所有前向FP32 | 未明确这些与当前完全相同，不能从论文补猜 | 未明确这些与当前完全相同，不能从论文补猜 |

**这里有三项不可误读：**

1. A归一化是数据集固定的通道均值/方差，Gong的NAT是每个时间片段、每个关节、各坐标轴沿时间独立标准化重建目标；两者不是同一种处理。当前只借用Gong的对比机制，未实现其完整NAT重建方法。
2. ACA²Net的分段和mask是成对设计：学生90/3×25×.1=75token，教师90/9×25×.3=75token；当前学生120/4×25×.1=75、教师120/4×25=750。教师更多token不是自动bug，但相同batch下其attention矩阵元素数为ACA²Net教师的约100倍，token线性部分约10倍；这是形状推导，不是整网训练时间/显存实测。不能只把当前教师mask改成.7：这样保留225token，而且同位置gather还必须处理教师被mask的位置。
3. 当前v3沿用无约束跨层W、无额外LN，并按用户要求去掉教师侧adapter、将AA系数改为1；学生adapter仍为Xavier初始化FC，未恢复v1恒等初始化/softmax。名义系数相同不代表encoder梯度或训练目标等价，因为目标模态、归约和重建loss不同。把motion patch9教师直接放入现有EMA也有embedding形状不一致的问题，需要重新设计。

### 训练与LP协议

| 项目 | 当前保存命令 | Gong论文 | ACA²Net论文 |
|---|---|---|---|
| PT epochs | 400 | 400（MAMP沿用推定） | 400 |
| PT总batch | 2卡×32×accum1=64 | 128（MAMP沿用推定） | 96，原文4张4090 |
| PT优化器 | AdamW，betas(.9,.95)，WD .05；bias/norm按timm免WD | MAMP沿用推定：AdamW，betas(.9,.95)，WD .05；细部分组未单列 | AdamW，betas(.9,.95)，WD .05；细部分组未单列 |
| PT LR峰值/最低值 | .001 / .0005，显式lr不随batch自动缩放 | .001 / .0005（MAMP沿用推定） | .001 / .0005 |
| PT warmup/cosine/floor | warmup20；约epoch380到floor，末20轮保持 | warmup20+cosine（MAMP沿用推定）；论文未明确末20轮floor | warmup20+cosine；未明确末20轮floor |
| 训练crop | [.5,1] | [.5,1]（MAMP沿用推定） | [.5,1] |
| PT random rotation | False | 正文未单列；MAMP官方NTU60配置True，不能直接断言Gong实际值 | 正文未明确，不能补猜 |
| LP epochs/LR | 100；SGD .1→0 | 100；.1→0 | 100；SGD .1→0 |
| LP总batch | 2卡×64×accum1=128 | 256，正文明确 | 256，正文明确 |
| LP LR末端 | 末10轮固定0；即后10轮优化器LR为0 | cosine到0，未写末10轮固定0 | cosine到0，未写末10轮固定0 |
| LP测试crop | [.95] | .9（MAMP正文沿用推定） | .9，正文明确 |
| LP读出 | MacDiff linprobe2：平均人物/时间，保留关节，6400维，dropout.3+BN+FC | 正文仅说固定骨干+linear classifier，无法据此确认6400维读出、dropout或BN | 正文仅说固定学生+linear classifier，未明确当前head细节 |
| LP encoder | 只评学生，不评教师 | 保留online encoder | 保留student encoder |

这些是**当前教师组的协议差异**，不能重新解释成历史原生差距的唯一原因。之前已经做过PT500/global128等原生实验，且当前batch64、PT floor5e-4、LP floor0和seed0是既定选择；本次未改动、未新增实验矩阵。

后续优先完成已有tensor测试和实际双卡短验证，确认loss/梯度有限、教师无grad、成功step计数及Queue跨rank相同、AMP跳步保护和实际LP加载。运行验证通过只证明实现可运行，不证明论文等价或LP收益。

## 文件与默认设置

- 入口：`main_pretrain_skeleton_teacher.py`，复用原生数据/优化器/训练循环。
- 模型：`model/transformer_macdiff_skeleton_teacher.py`，继承原生MacDiff，实际调用其`forward_macdiff`和masked epsilon loss。
- 配置：`config/ntu60_xsub_joint/pretrain_madiff_skeleton_teacher.yaml`。
- 测试：`tests/test_skeleton_teacher.py`。
- 公共入口只增加DDP初始化后、resume加载前的可选教师初始化；公共engine只为带`supports_skeleton_teacher`标记的模型传sample index、记录loss和提交辅助状态。普通native/text/OSE模型不会进入新钩子。

默认学生/教师encoder均8层、256维、8 heads、FFN1024；native decoder3层、256维、FFN1024。teacher只是encoder，没有decoder。数据为NTU60 XSub、120帧、25关节、PT仅person0、时间patch4帧/关节patch1，mask0.9。PT400轮、双卡每卡32、accum1、LR.001、warmup20、min_lr5e-4、末20轮floor、seed0；LP100轮、每卡64、LR.1、min_lr0、seed0，复用既有linprobe配置。

| 新参数 | 默认值 | 定义 |
|---|---:|---|
| `ema_momentum` | .999 | 成功optimizer step后更新教师 |
| `contrast_temperature` | .07 | InfoNCE温度 |
| `contrast_queue_size` | 65536 | 全rank共享语义的FIFO负样本队列容量 |
| `alignment_teacher_layers` | 5 | 教师最后5个block输出，即第4–8层 |
| `lambda_contrastive` | 1e-4 | 全局InfoNCE外层权重 |
| `lambda_alignment` | 1 | 位置对应跨层SmoothL1外层权重 |
| `teacher_method_version` | `raw_full_position_student_adapter_aa1_v3` | 方法身份，写入checkpoint args，拒绝旧v1/v2续训 |

两个辅助权重可分别为0做消融。为保持`find_unused_parameters=False`的DDP图一致，0权重时仍计算对应分支，乘0后梯度为0。两个辅助分支从epoch0参与，无t筛选、GT warmup或权重ramp。

## 完整训练数据流

1. Feeder从一个训练序列随机裁剪[.5,1]并插值到120帧，输出同一crop的`source`和加小关节高斯扰动的`source_aug`。两侧均取person0。原始坐标先按原生A均值/方差归一化；默认不做self_shift。这里“干净”是指没有人为添加小扰动/扩散噪声，不表示数据没有测量噪声。
2. 学生读取`source_aug`：4帧×1关节×3坐标映射为256维，叠加可学习时间和空间位置embedding，得到`[B,750,256]`。沿用原生随机mask，保留75个，记录原始位置索引`ids_keep:[B,75]`。8层encoder输出`S1...S8:[B,75,256]`；最终norm得到`latent`，均值池化得到`[B,256]`。
3. **原生diffusion分支**：从归一化原始`source`独立采样完整0–999的`t`和真实高斯噪声epsilon，得到`x_t`。将75个学生latent和675个全局均值特征按`ids_restore`还原为`[B,750,256]`条件。原生.1条件dropout保留。3层native decoder读取完整`x_t`、t和条件，输出`[B,750,12]`的epsilon预测。只在675个encoder-masked patch计算原有epsilon MSE，并保留.02倍骨架token uniformity。数值补丁在teacher子类内将decoder和去噪loss置于FP32子区域；方程/参数键不变，AMP下的计算轨迹及资源占用会改变。
4. **教师前向**：同一crop的归一化原始`source`进入EMA教师。没有小高斯扰动、diffusion噪声或mask。教师750个token按原始时空顺序直接进入8层encoder，不调用mask_ratio=0仍会打乱顺序的随机mask函数。教师始终eval/no_grad，只复制encoder的embedding、位置参数、blocks和最终norm。
5. **全局InfoNCE**：学生最终norm后的75个token做GAP再L2归一化，得到query `q:[B,256]`；教师最终norm后的全部750个token同样池化/归一化，得到key `k:[B,256]`。同一样本q/k为正对，已提交FIFO队列中的其他训练样本key为负对，softmax温度.07。依据原始sample index屏蔽队列里该样本自己的历史视图，不读取动作标签。队列从空开始，第一完整优化窗口的InfoNCE为0，而非使用随机负样本。全局对比梯度只回学生encoder。
6. **位置对应跨层对齐**：在教师第4–8个block结束时，按学生`ids_keep`取出同一时间patch、同一关节的75个教师token。这些token已读取完整750-token上下文。学生8层特征用`8×5`可学习实数矩阵W直接线性组合，无softmax约束。每个目标层仅配一个学生侧256→256 Linear `A_j`；教师侧没有adapter。对`A_j(sum_i W_ij S_i)`和`detach(T_j[ids_keep])`直接计算FP32 SmoothL1，不添加额外LN，在batch/token/channel维和5个目标层取平均。梯度回学生8层、W和5个学生adapter；教师目标detach，不回EMA encoder或decoder。
7. **反向与更新**：总loss见下式。普通native dropout即使把某个样本的decoder条件置零，该样本仍参与两个encoder辅助损失。辅助损失不依赖t，不经过noise2point/x0还原。按原有累积策略执行反向；仅完整累积窗口且GradScaler没有跳步时，先完成optimizer step，再用更新后的学生参数做教师EMA，并将该窗口前向时的旧教师keys跨rank all_gather后写队列。各rank使用相同keys/IDs顺序；队列满后覆盖最早项。AMP失败和不完整尾窗口不推进EMA或队列。

```text
同一crop原始骨架 ──A归一化──┬── q_sample(t, epsilon) ── x_t ─────┐
                           │                                   │
同一crop小扰动骨架 ──归一化─┼─ patch/位置embedding ─ mask75/750 ─ 学生E8
                           │                                   ├─ latent/均值补全条件 ─ native D3(x_t,t) ─ epsilon MSE
                           │                                   ├─ GAP/L2 ─────────── InfoNCE(q,k,Queue)
                           │                                   └─ S1...S8 ─ W/A_j ───── SmoothL1
                           └─ 教师E8(完整750，EMA/no_grad/eval) ─┬─ GAP/L2 ─ k
                                                              └─ 第4–8层按ids_keep取75 ─ detach ─ target
```

公式：

\[
L_{native}=L_\epsilon+0.02L_{skel\_uniformity},\qquad
L=L_{native}+10^{-4}L_{InfoNCE}+L_{AA}.
\]

W为无softmax约束的可学习实数矩阵，目标层j为教师第4–8层：

\[
\widehat S_j=A_j\left(\sum_{i=1}^{8}W_{ij}S_i\right),\quad
\widetilde T_j=\operatorname{sg}[T_j[ids\_keep]],\qquad
L_{AA}=\frac{1}{5}\sum_{j=4}^{8}\operatorname{SmoothL1}
\left(\widehat S_j,\widetilde T_j\right).
\]

此处按用户最新选择保留token物理位置对应、仅学生侧Linear及AA权重1。跨层W/学生投影/SmoothL1借鉴原式，去掉教师投影是明确的方法差异。初始化和均值归约采用上文记录的默认选择。教师encoder无需预训练checkpoint；初始化复制当前随机学生，后续`theta_teacher=.999*theta_teacher+.001*theta_student`，不包括学生adapter及W。

## 保存、恢复与LP

完整PT checkpoint含学生encoder/native decoder、EMA教师、W和学生侧adapter、负样本keys/IDs/指针/数量、EMA更新计数、optimizer/scaler和args。专用入口将CLI辅助参数合入保存的`model_args`；resume要求同方法、相同完整model_args和完整优化器状态，拒绝原生/text/OSE checkpoint、旧v1/v2及变更权重/深度/队列定义的“继续训练”。DDP后先同步教师初始化，再加载完整resume，不会把已保存教师重置为学生。

native FP32补丁及关闭AMP均未增加权重或改变model_args，因此允许旧v3完整resume；`args.skeleton_teacher_numerics_version`记录新计算策略，不参加方法结构相等检查。旧AMP scaler在当前disabled状态下不参与更新；optimizer/EMA/Queue继续恢复。精度改变须保留原checkpoint/旧日志，并在新目录记录continuation；公共保存仍不含RNG，恢复不保证精确复现原失败batch。

LP继续用`main_linprobe.py`和`config/ntu60_xsub_joint/linprobe_madiff.yaml`，只加载同名的学生encoder参数。教师、decoder、AA和队列是下游模型的额外键，不参与LP。当前既有`linprobe2`协议对两个人物分别编码，输入完整骨架、不mask，按人物和时间平均、保留25个关节，将25×256展平为6400维，经原有head dropout/BN/线性分类器得到60类输出；encoder冻结，只训练head。不评估教师encoder。

新增epoch JSON为`train_loss_native`（已包含.02骨架uniformity）、`train_loss_contrastive`、`train_loss_alignment`，均为未乘辅助外层权重的分支值；`train_loss`是组合后的总loss。没有text energy/variance/valid token统计。

## 运行命令

服务器先跑新增测试（有Torch时会执行实际前向、梯度、队列、EMA和LP加载用例）：

```bash
python -m unittest discover -s tests -p test_skeleton_teacher.py -v
```

双卡单行PT→LP，teacher PT关闭autocast/GradScaler、FP32，原生学生decoder3，从头训练，PT min_lr5e-4/LP min_lr0：

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10295 main_pretrain_skeleton_teacher.py --config config/ntu60_xsub_joint/pretrain_madiff_skeleton_teacher.yaml --disable_amp --lambda_contrastive 0.0001 --lambda_alignment 1 --batch_size 32 --accum_iter 1 --epochs 400 --warmup_epochs 20 --min_lr_epochs 20 --lr 0.001 --min_lr 0.0005 --mask_ratio 0.9 --motion_aware_tau -1 --seed 0 --output_dir output_dir/ntu60_xsub_skeleton_teacher_v3_student_adapter_aa1_dec3_seed0_20261020_fp32_noamp/pt --log_dir output_dir/ntu60_xsub_skeleton_teacher_v3_student_adapter_aa1_dec3_seed0_20261020_fp32_noamp/pt/tensorboard && OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10296 main_linprobe.py --config config/ntu60_xsub_joint/linprobe_madiff.yaml --finetune output_dir/ntu60_xsub_skeleton_teacher_v3_student_adapter_aa1_dec3_seed0_20261020_fp32_noamp/pt/checkpoint-399.pth --batch_size 64 --accum_iter 1 --epochs 100 --warmup_epochs 0 --min_lr_epochs 10 --lr 0.1 --min_lr 0 --seed 0 --dist_eval --output_dir output_dir/ntu60_xsub_skeleton_teacher_v3_student_adapter_aa1_dec3_seed0_20261020_fp32_noamp/lp --log_dir output_dir/ntu60_xsub_skeleton_teacher_v3_student_adapter_aa1_dec3_seed0_20261020_fp32_noamp/lp/tensorboard
```

本次epoch89失败后，从旧v3 `checkpoint-80.pth`恢复的单行PT→LP；先同步补丁模型/入口/YAML/测试并运行上方测试。此命令要求服务器实际有该有效完整checkpoint；若没有，应先定位真实最新完整文件，不伪造恢复点。

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10295 main_pretrain_skeleton_teacher.py --config config/ntu60_xsub_joint/pretrain_madiff_skeleton_teacher.yaml --disable_amp --resume output_dir/ntu60_xsub_skeleton_teacher_v3_student_adapter_aa1_dec3_seed0_20261020/pt/checkpoint-80.pth --lambda_contrastive 0.0001 --lambda_alignment 1 --batch_size 32 --accum_iter 1 --epochs 400 --warmup_epochs 20 --min_lr_epochs 20 --lr 0.001 --min_lr 0.0005 --mask_ratio 0.9 --motion_aware_tau -1 --seed 0 --output_dir output_dir/ntu60_xsub_skeleton_teacher_v3_student_adapter_aa1_dec3_seed0_20261020_fp32_noamp/pt --log_dir output_dir/ntu60_xsub_skeleton_teacher_v3_student_adapter_aa1_dec3_seed0_20261020_fp32_noamp/pt/tensorboard && OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10296 main_linprobe.py --config config/ntu60_xsub_joint/linprobe_madiff.yaml --finetune output_dir/ntu60_xsub_skeleton_teacher_v3_student_adapter_aa1_dec3_seed0_20261020_fp32_noamp/pt/checkpoint-399.pth --batch_size 64 --accum_iter 1 --epochs 100 --warmup_epochs 0 --min_lr_epochs 10 --lr 0.1 --min_lr 0 --seed 0 --dist_eval --output_dir output_dir/ntu60_xsub_skeleton_teacher_v3_student_adapter_aa1_dec3_seed0_20261020_fp32_noamp/lp --log_dir output_dir/ntu60_xsub_skeleton_teacher_v3_student_adapter_aa1_dec3_seed0_20261020_fp32_noamp/lp/tensorboard
```

仅InfoNCE对照可令`--lambda_alignment 0`，并自行使用不同输出目录避免覆盖。没有启动任何训练。

## 当前验证边界

助手本地缺Torch/PyYAML/timm，没有安装依赖、执行模型或启动GPU任务。v3数值补丁的定义、默认/CLI、旧v1/v2恢复拒绝、实际engine控制流、实际AA和FP32包装函数NumPy检查11项通过；11项真实Torch/梯度/EMA/队列/LP加载/诊断/CUDA溢出/scaler恢复检查缺Torch跳过，累积策略5项通过，共16通过/11跳过。Python3.8语法检查通过。T27为旧v2 PT400/LP96；T28为用户确认的v3 AA1从80关AMP续训，上传PT81–399/LP0–89及整体显存峰值，具体服务器补丁版本和真实测试输出仍缺。续训日志不能代替上述跳过的状态/梯度/CUDA验证。

完整750-token教师没有encoder反向图，队列keys约64MiB、IDs约.5MiB。T27按用户确认的v2命令运行，日志峰值见顶部；尚无独立教师分支显存剖析或训练耗时记录。
