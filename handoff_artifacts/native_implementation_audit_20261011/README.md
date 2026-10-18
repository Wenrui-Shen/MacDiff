# 当前实现与官方 MacDiff 对照（2026-10-11）

比较官方提交 `692888f1e2a4227216511aac6733dff31b587213` 与本地实际工作区；本地 HEAD 为 `e854ff11cc60e0c02838331cfbf59586e3b6e15c`。官方原始文件已在本地 Git 对象中，因此本次直接读取完整源码，不依赖旧交接结论。范围以 NTU60 XSub、native PT→LP 为主，并核查文本/OSE扩展的隔离和预处理改动。

**结论：当前仓库保留了官方的原生核心数学计算，但原生训练入口仍有实现差异，不能称为完全未经修改的官方基线。PT DDP 设置仍不同。用户随后要求恢复官方不完整累积窗口处理，现已恢复；当前旋转脚本使用 accum_iter=1，更新次数不受影响。静态检查没有发现足以直接解释 LP 成绩差距的原生 loss/网络层改动；尚未进行前向、梯度和 DDP 数值等价验证。**

## 原生路径中的实现差异

| 位置 | 官方实现 | 当前实现 | 当前 native 运行的含义 |
|---|---|---|---|
| PT DDP | `find_unused_parameters=True`，未覆写 `broadcast_buffers`（默认True） | `find_unused_parameters=False, broadcast_buffers=False` | 对所有 PT 都生效。关闭扩展后的native模型只有LayerNorm，未见注册buffer；OSE字段仅赋None，未创建Queue/teacher。未发现由此直接改变native公式的路径，但通信/梯度归约行为未做GPU验证。LP自己的DDP配置未改。 |
| PT 梯度累积尾窗口 | 每批loss除以accum_iter，只在整窗口末更新；不足一个窗口的尾部梯度不step，并在下轮zero_grad丢弃 | **已按用户要求恢复相同策略**，native/text及OSE两条PT路径一致 | 恢复前按实际window_size完成尾部step；本次删除该行为。尾部不更新text target/EMA、OSE teacher/Queue，AMP成功step检查保留。当前accum1和标准NTU60双卡32/2的626步不受影响。 |
| 模型公开forward接口 | 默认mask_ratio=.80、motion_aware_tau=.75；第三个位置参数是mask_ratio；接受任意kwargs | 默认.90/-1；第三个位置参数为peer，另加OSE参数 | 不显式传mask/tau或使用原位置参数的外部调用会改变行为。当前PT engine显式传入mask_ratio与motion_aware_tau，所选配置.9/-1，不触发默认值差异。内部forward_macdiff默认仍与官方相同。 |
| 条件组装 | 在forward_decoder内cat global/local条件；gather的整数index使用repeat | 移至build_global_local_condition；整数index用expand视图 | 数学上仍是可见位置local、其余位置pooled global、按ids_restore恢复位置。expand作用于整数索引，未改变condition或其梯度设计。encoder宽度与dim_feat一致时值/形状相同；未做浮点/梯度逐位验证。 |
| encoder blocks | 直接逐层执行 | 增加checkpoint_blocks开关，默认False | 原生forward未启用；选择False分支后AST与官方一致。不能把OSE专用checkpoint当作原版默认启用。 |
| native Feeder | __getitem__里完成crop→flip→公共rotation→source augmentation | 前三步抽成_load_processed，再执行source augmentation | 内联helper后AST完全一致；没有额外增强、额外随机调用或返回顺序变化。新FeederOSE只在明确选它时使用。 |
| PT engine辅助逻辑 | 捕获sampler/mask/hyperparam更新异常；常规日志 | 直接调用sampler更新；移除两个不存在的hook；新增显存/时间统计和文本target更新guard | 两版native sampler更新函数均只打印，native没有另外两个hook。无文本时target逻辑跳过；max_train_steps默认0。新增显存统计与epoch末统计归约，未见进入梯度目标的路径。 |
| LP兼容与指标 | 假定model.module存在；评估CE按batch均值平均 | 支持非DDP模型；CE按样本数加权 | DDP训练选择的head与官方相同。CE日志口径变了；Top1/Top5原本就按样本数加权，该修改不改变best LP acc。 |
| util/misc | 从torch._six导入inf；普通resume | 从math导入inf；新增文本resume一致性检查 | inf数值相同；NativeScaler、梯度范数和普通保存恢复算法未改。text_cache为空时文本resume检查跳过。 |
| 包导入 | 只导入native/downstream与原feeder模块 | 额外导入Stage2、text与Stage2 feeder | 对新增模块的顶层代码扫描只发现torch.no_grad装饰器，未见模块导入时实例化模型、采样随机数或修改全局精度/seed。 |

当前 PT 模型选择检查要求 `model.transformer_macdiff_text.Transformer` 与非空text_cache配套；原版旋转脚本明确选native模型、普通Feeder、enable_ose=False、text_cache空。此时不加载cache，不创建target bank，不排除exemplar，不建立OSE momentum encoder，不调用文本辅助loss。

## 已确认保持一致的核心

以下以相同参数为前提，不把用户明确更改的decoder深度当作实现bug：

- MLP、Attention、Block、FeatureModulation、DecoderBlock、SkeleEmbed、权重初始化、random/motion-aware masking、patchify、masked diffusion loss、token uniformity公式均保持官方实现。独立核心函数/类AST一致；构造器仅增加10个赋None的OSE字段。
- native forward的标准化、person0截取、diffusion timestep采样、epsilon采样、可见token encoder、global/local条件、unconditional drop和总loss计算与官方对应。去除局部变量改名、条件helper搬移、诊断print格式及非法prediction报错差异后AST一致；诊断if中的NumPy随机判断仍保留参加比较。
- diffusion目录全部15个文件、model/transformer_downstream.py、model/util.py、model/drop.py、feeder/tools.py、util/lr_sched.py、util/pos_embed.py、util/class_acc.py、requirements.txt均与官方文件一致（只归一化CRLF/LF）。
- PT AdamW仍为lr显式值、betas(.9,.95)、同一timm add_weight_decay分组；LP仍为冻结encoder、linprobe2、BN+FC、SGD lr.1/momentum.9/wd0。LP权重加载、位置插值和missing_keys断言未改。
- AMP仍PT默认True、LP默认False，NativeScaler仍使用GradScaler；cudnn.benchmark=True、seed+rank、worker种子、DistributedSampler及set_epoch协议均未改。两份入口原有argparse选项的default/type/action等设置全部一致。

## 数据预处理与配置

NTU60三份预处理脚本改的是输入/输出/statistics路径、mkdir方式和np.savez输出文件路径。去噪的10个函数、raw读取的2个函数完全一致；seq_transformation中除split_dataset的保存路径外，其余函数一致。所有14份NTU60/NTU120统计名单文件与官方一致。因此没有发现本地NTU60主体排序、去噪阈值、关节变换或XSub划分算法改变。**这不证明服务器现有NPZ来源相同；本次没有实际数据可核验。**

NTU120同样改路径，包含setup≥18的extension目录。PKUv1额外将cross-view/cross-subject文件名改为cross_view/cross_subject；PKUv2改输出目录。这些不进入NTU60训练。

两份原生NTU60 PT/LP YAML相对官方只改data_path，mean/var、one_person默认、噪声和旋转等数值设置未改变。其他18份原始配置的路径改动见之前配置审计，本次原始文件清单也逐项保存。当前新旋转runner另外建立明确配置，不能与“官方默认脚本”混称：

| 项目 | 官方公开脚本+YAML | 当前普通PT脚本 / 新旋转对照 |
|---|---|---|
| PT GPU×每卡batch×accum | 4×32×1，有效128 | 普通脚本默认2×32×1，有效64；旋转runner 2×64×1，有效128 |
| LP GPU×每卡batch×accum | 4×64×1，有效256 | 旋转runner 2×128×1，有效256，但每卡BN batch不同；原script_linprobe_madiff.sh本身未改，仍4卡模板 |
| PT深度/轮数 | decoder5、400epoch | 普通脚本仍5/400；旋转runner 3/500，依用户指令 |
| PT旋转 | YAML的random_rot=True被注释，默认False | runner顺序比较False/True；LP训练True、测试False保持官方设置 |
| PT LR | 脚本lr1e-3/min_lr1e-5，覆写YAML min_lr5e-4 | 普通脚本及旋转runner相同；T14实际日志min_lr5e-4属于运行参数差异 |
| normalization | A mean/var | 原生配置及runner均A |

所以之前“两个batch配置global都256”的描述只说明梯度batch相同，不能保证LP BN轨迹相同。用户后续确认第三方双卡3090/86.2使用原版默认配置、每卡PT32（默认accum1→有效64）；LP每卡64/有效128仍为用户估计。尚无实际命令/日志；当前旋转runner有效PT128/LP256与该参考不同。

## 文本组与原版的额外差异

T15/T16按用户描述使用text no-share扩展，属于主动改变预训练目标的实验：

- 新增文本remap、T→S decoder及逐层cross-attention reader、独立S→T noise decoder和样本target bank；额外模块初始化会消耗随机数。
- native分支之外有T→S和S→T loss；当前no-share T→S不直接连接skeleton encoder，S→T通过encoder的pooled/visible token条件传梯度。
- 文本forward会过滤空保留人物的loss行，原生forward没有该过滤；多做text timestep/noise采样，而且诊断随机调用布局与原生不同。相同seed不表示跨epoch使用相同mask/noise/augmentation序列。
- sample target在成功optimizer step后更新，AMP跳步时不更新；native没有这套状态或更新。

即使把两种文本方向权重置0，仍有额外初始化、cache/bank、模型参数与DDP未使用参数的问题，不应直接把“text模型权重0”用作干净native控制。旋转runner已经直接选native模型。

## DDP开关解释与本次恢复

`find_unused_parameters=True`会从forward返回的tensor遍历autograd图，识别未得到梯度的可训练参数并将其标记为ready，避免归约一直等待；不会给未用参数生成梯度。False省掉这次遍历，要求相关参数确实参加loss/backward，否则可能出现reduction未完成的错误。若参数全部用到，两者理想平均梯度目标相同，不能据此保证浮点训练轨迹完全一致。

`broadcast_buffers=True`会在forward前将rank0的注册buffer同步到其他rank，例如BN running_mean/running_var。False不做这次广播，buffer各自维护；**模型参数的初始同步和梯度all-reduce仍执行**。该开关不等于SyncBatchNorm，不会把当前各卡batch均值/方差合成global BN。当前native PT使用LayerNorm、没有注册buffer，故此项没有待广播状态；LP的DDP设置本次未动。

已查[PyTorch v1.8.1 DDP源码](https://github.com/pytorch/pytorch/blob/v1.8.1/torch/nn/parallel/distributed.py)，与项目requirements固定版本一致；本次仅恢复尾窗口，两个DDP开关仍False。

## 验证与边界

- 读取官方92个原始文件：55个文件一致，37个文本内容改变；不是37处算法修改，包含README、18份路径配置及8份预处理路径文件。新增扩展文件另由Git diff统计。
- 22项原生路径AST检查通过；官方原有PT/LP parser选项无变化；保留每文件sha256、完整检查说明及累积窗口计算。
- 本机Python缺Torch、NumPy、PyYAML；没有运行模型、数据、GPU/DDP或500epoch训练。静态重构匹配不能替代forward/gradient/optimizer step等价验证，也不能证明服务器checkout与本地一致。
- 初次审计仅生成产物。用户后续明确要求恢复尾窗口，本次修改engine_pretrain.py的两条PT路径，并新增5项标准库控制流测试；连同8项旋转runner测试共13/13通过。测试执行实际引擎的更新语句、验证完整窗口/尾部丢弃及AMP跳步状态更新，不代表Torch autograd或DDP数值测试。DDP、模型、配置与启动脚本未改。

若正确LR旋转对照仍无收益，源码层面有价值的排除项是：在同一服务器、同一数据、相同PT/LP参数下使用干净官方提交；或仅在隔离native配置中恢复官方DDP选项作对照。不要全局改DDP影响文本/OSE，也不要把静态审查结论当作硬件问题的证据。

复查资料：[自动审计JSON](audit.json)、[审计器](audit.py)、[原生路径完整差异](native_path.patch)、[预处理差异](data_paths.patch)、[启动及NTU60配置差异](launch_configs.patch)。官方基线：[官方固定提交](https://github.com/LehongWu/MacDiff/tree/692888f1e2a4227216511aac6733dff31b587213)。

## 2026-10-14：MAMP与MacDiff对齐分析

本次直接读取MAMP官方提交09e6252a5c7d061c226395d25e6aa47a69e7e58d（2023-09-22）、对应NTU60 XSub YAML、PT/LP脚本、模型和scheduler，并核对MAMP/ MacDiff论文。仅追加现有文档，没有复制新仓库、增加配置/日志文件、改训练代码或启动实验。

| 项目 | MAMP官方NTU60 XSub | MacDiff官方公开NTU60脚本/配置 | T20及当前noshare命令 |
|---|---|---|---|
| encoder | 8层，dim256，heads8，MLP1024；120帧，patch4×1 | 同样公共骨干参数 | 相同 |
| 骨架decoder | 3层、dim256 | 5层、dim256 | 5层；noshare T→S也5层 |
| PT epoch/warmup | 400/20 | 400/20（论文PT500） | 400/20 |
| PT lr/min_lr | 1e-3/5e-4 | YAML仍5e-4，但启动脚本显式覆写1e-5；论文也是1e-5 | 1e-3/1e-5；T20日志确认末端1e-5 |
| PT scheduler | warmup后cosine一直到400，无末轮保持 | cosine到380，最后20轮保持min_lr | MacDiff scheduler |
| PT batch | 脚本4卡×32，accum1，有效128 | 官方脚本同4×32=128 | 2×32=64，按用户选的双卡默认参考 |
| LP batch | 脚本2卡×128，accum1，有效256 | 官方4卡×64，有效256 | 2×64=128 |
| LP读出 | protocol=linprobe；平均人物/时间/关节→256维；BN+FC，无head dropout | protocol=linprobe2；平均人物/时间、保留关节→25×256=6400维；head dropout0.3，BN+FC | MacDiff读出；日志FC可训练参数384060 |
| LP schedule | 100轮，lr.1，warmup0；cosine到100趋向0 | 100轮，lr.1，warmup0；到90到0并保持末10轮 | MacDiff scheduler |
| PT AMP | 默认False，脚本未启用 | 默认True | True |
| PT旋转/采样 | rotationTrue，motion-aware tau.80，mask.9 | PT YAML不旋转；random masking tau-1、mask.9。论文写PT旋转 | PT不旋转/random mask.9 |
| 人物与归一化 | 保留输入两人；运动目标按patch标准化（norm_skes_loss=True） | PT默认person0，坐标按mean/var A标准化以做diffusion | person0/A；现sample target bank要求one_personTrue |

MAMP来源：[NTU60 PT配置](https://github.com/maoyunyao/MAMP/blob/09e6252a5c7d061c226395d25e6aa47a69e7e58d/config/ntu60_xsub_joint/pretrain_mamp_t120_layer8%2B3_mask90.yaml)、[PT脚本](https://github.com/maoyunyao/MAMP/blob/09e6252a5c7d061c226395d25e6aa47a69e7e58d/script_pretrain_mamp.sh)、[LP配置](https://github.com/maoyunyao/MAMP/blob/09e6252a5c7d061c226395d25e6aa47a69e7e58d/config/ntu60_xsub_joint/linprobe_t120_layer8.yaml)、[LP脚本](https://github.com/maoyunyao/MAMP/blob/09e6252a5c7d061c226395d25e6aa47a69e7e58d/script_linprobe.sh)、[LR调度](https://github.com/maoyunyao/MAMP/blob/09e6252a5c7d061c226395d25e6aa47a69e7e58d/util/lr_sched.py)、[LP读出](https://github.com/maoyunyao/MAMP/blob/09e6252a5c7d061c226395d25e6aa47a69e7e58d/model/transformer.py#L65)、[PT目标](https://github.com/maoyunyao/MAMP/blob/09e6252a5c7d061c226395d25e6aa47a69e7e58d/model_mamp/transformer.py#L437)、[AMP默认](https://github.com/maoyunyao/MAMP/blob/09e6252a5c7d061c226395d25e6aa47a69e7e58d/main_pretrain.py#L77)。MacDiff来源：[PT脚本](https://github.com/LehongWu/MacDiff/blob/692888f1e2a4227216511aac6733dff31b587213/script_pretrain_madiff.sh)、[默认PT配置](https://github.com/LehongWu/MacDiff/blob/692888f1e2a4227216511aac6733dff31b587213/config/ntu60_xsub_joint/pretrain_madiff.yaml)、[LP读出](https://github.com/LehongWu/MacDiff/blob/692888f1e2a4227216511aac6733dff31b587213/model/transformer_downstream.py#L65)、[LR调度](https://github.com/LehongWu/MacDiff/blob/692888f1e2a4227216511aac6733dff31b587213/util/lr_sched.py)。

### decoder：数据集配置、论文默认与不同decoder结构要分开

MAMP的NTU60配置确实decoder3，但[MAMP论文](https://arxiv.org/pdf/2308.07092#page=8)表10的NTU60 decoder3/5都是84.9，正文整体默认为5层；NTU120最好为5层。不能把“MAMP一律3层”当统一规则。

[MacDiff论文](https://lehongwu.github.io/ECCV24MacDiff/macdiff-paper.pdf#page=14)表10：depth2/3/4/5为84.8/86.4/86.0/85.9；原文明确5层默认是考虑生成能力。因此做LP表征学习时，3层是有依据的候选；T20 decoder5的86.00194已接近论文5层消融85.9，但不能仅据表10断定主表86.4运行实参全部属于3层。

MAMP decoder是latent+learnable mask token→普通Transformer→归一化运动预测；MacDiff是带噪骨架+t+encoder条件→AdaLN调制→epsilon预测。相同层数/宽度不保证参数量或decoder任务难度相同。骨架decoder对应decoder_depth；noshare该参数同时改变native/T→S，text_decoder_depth另管S→T，应作为不同因素。

### min LR：已有方法明确不同，复制floor不等于复制日程

MAMP论文及代码都明确训练1e-3→5e-4；MacDiff论文及公开启动脚本都明确1e-3→1e-5。因此1e-5是复现MacDiff时有依据的值，不能因为MAMP是基线就认定它错误；5e-4可做独立优化消融，不保证提高LP。MacDiff YAML残留5e-4与MAMP数值相同，但仅相同值不足以证明作者意图。

floor5e-4是floor1e-5的50倍，改变的是整个cosine区间；使用MacDiff400轮调度时，epoch200的lr分别为7.5e-4与5.05e-4。若直接运行MAMP full cosine，epoch200约7.706448e-4，epoch399约5.000085e-4。只改min_lr仍保留MacDiff末20轮保持，不算完整MAMP日程；若单独测试完整MAMP调度，另需min_lr_epochs=0，LP也对应0。该改变本次未实施。

### 公平对照应先明确比较目标

1. 当前noshare是在T20 MacDiff双卡profile上测试文本增益，公共PT/LP参数已对齐；它是native vs text配对，不自动等同于MAMP完整协议。当前用户已选5层/400/PT32/LP64/seed0，不改正在使用的命令，PTseed1仍明确不跑。
2. 若要与MAMP做统一评价，LP读出与head dropout、global/local batch和LR日程比只看decoder/min_lr更直接。MAMP256维FC有15420参数，MacDiff6400维FC有384060；虽都是线性层，保留的空间信息不同，引用两篇原始成绩不能当成读出完全相同的控制。共同LP协议需要所有比较方法一致执行，不能只改其中一个方法再与旧分数直接作因果结论。
3. 已有MacDiff checkpoint可做protocol=linprobe的256维LP读出对照，无需重训encoder；固定batch/LR/seed等公共参数，单独量化该head协议（同时含dropout变化），不保证升分。做完整MAMP LP协议则另需batch总256及full cosine，分开命名。
4. 若主要关注表征decoder容量，可先配对测试native/noshare的骨架decoder3，保留当前min_lr1e-5及S→T5层/512不变；再独立测试min_lr5e-4。T14旧3层/500/高floor与新T20在batch、epoch和实现路径上不同，不能当干净min_lr对照。

以上是分析与候选，用户尚未选定新的MAMP对齐实验；当前noshare尚无回传结果。本次未改YAML、代码或实验编号，不创建额外artifact文件。

