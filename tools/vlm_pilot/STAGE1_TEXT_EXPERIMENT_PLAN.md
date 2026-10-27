# Stage1 文本实验待测变量与顺序（更新2026-10-27）

**2026-10-27实现边界更正：**用户追问后重新明确：下方B已实现的是细patch embedding均值合并近似，不是原论文独立大patch投影。该近似是助手为保持EMA同形作的工程选择，不能作为用户要求的原论文式投影已经完成。真正12帧投影的初始化/学习/更新尚需定义；默认A继续完整EMA key＋InfoNCE＋T→S＋global关系KL，代码/默认配置未因解释更改。

## 2026-10-27最新消融层级：默认full key，一级粗patch随机70%，二级再考察运动/MCAM

用户再次澄清：“额外消融只改大patch＋随机mask70%，运动差分和motion compass mask作为额外消融的额外消融”。主方案A仍完整骨架key v2，不mask；已实现可选一级B的[配置](/D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_text_relation_teacher_coarse70_decoder3.yaml)、[测试](/D:/program/MacDiff/tests/test_text_relation_teacher_coarse.py)，方法和完整命令见[文档顶部](/D:/program/MacDiff/tools/MACDIFF_TEXT_RELATION.md:3)。B在同关节方向合并3个相邻4帧EMA patch embedding（含平均位置）为12帧尺度粗token，250个随机保留75个后进teacher Transformer，保留EMA参数同形；这是受细投影权重约束的适配，不是论文独立大patch MotionEmbed复现。75粗token覆盖225细patch，信息质量未验证。

二级候选分别B+运动差分、B+MCAM及B+两者，均未实施/未启动，不混入一级B。B其余与A一致：query75个细token GAP256，共享CL/KL骨架queue/文本queue65536，EMA.999、CL1e-4、KL.1、T→S1、uni.02、原PT400→LP100/noAMP。B版本coarse3_mean_random70_ema_gap_shared_queue_kl_v3、teacher_view=coarse_mask/group3/mask.7；A默认full保持，跨视图resume拒绝。B端口10327/10328，输出output_dir/ntu60_xsub_textrel_coarse3mean_m70_q65536_cl1e4_kl01_gap256_dec3_noamp_seed0_20261027；A仍原10325/10326。

B新增5通过/2缺Torch跳过；B+v2+v1+累积28通过/9跳过，整合回归79通过/73跳过；真实Torch/GPU/DDP待服务器验证，未启动训练，无新成绩编号。同步更新v2模型/入口/测试及新增B配置/测试。新消融不替代默认主线，也不接local KL或AA对齐。

## 2026-10-27最新澄清：完整EMA key的骨架queue由CL和全局KL共用（v2）

用户已明确：原生MacDiff+key encoder teacher对比学习，骨架侧直接使用其完整输入key队列，文本侧建立对应队列；query为10%可见token的全局池化，KL仍global，不加六部位local。新[模型](/D:/program/MacDiff/model/transformer_macdiff_text_relation_teacher.py)、[入口](/D:/program/MacDiff/main_pretrain_text_relation_teacher.py)、[配置](/D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_text_relation_teacher_decoder3.yaml)、[方法/完整命令](/D:/program/MacDiff/tools/MACDIFF_TEXT_RELATION.md:3)。masked GAP256有梯度、完整EMA key GAP256 no_grad、文本平滑global512 detach；同一个relation_skeleton_keys用于InfoNCE负样本与KL，配对文本同IDs/slot，入库完整key不是query。K65536、EMA.999、CL.07/1e-4，KL两侧.1/权重.1。保留T→S/七句/remap2048、原生diffusion与uni.02，无AA/S→T decoder/投影头/local KL。

用户保留256/6400讨论空间；当前按明确GAP定义用256，6400训练模式尚未定义/实现，LP仍原第8层6400。成功完整步提交配对queue并更新EMA，跳步/尾窗不提交；checkpoint带EMA/queue/计数，独立身份拒绝v1等方法resume。PT400/总64/floor5e-4/noAMP/seed0→LP100/总128/floor0，端口10325/10326，输出output_dir/ntu60_xsub_textrel_fullkey_q65536_cl1e4_kl01_gap256_dec3_noamp_seed0_20261027。旧v1 4096 masked-query入库对照保留。

新v2 9通过/3缺Torch跳过；v2+v1+累积23通过/7跳过；相关回归74通过/71跳过。真实Torch/GPU/DDP待服务器验证，未启动训练、无新成绩、不预分配编号；仍32条配置/成绩记录。同步新4文件及修改的基类关系模型/入口/测试，沿用已有公共支持文件。此前六部位local建议本次不实施。

## 2026-10-27最新选择：保留T→S，S→T改为文本关系分布蒸馏

用户已明确选择并授权实现。新[方法与完整PT&&LP命令](/D:/program/MacDiff/tools/MACDIFF_TEXT_RELATION.md)、[模型](/D:/program/MacDiff/model/transformer_macdiff_text_relation.py)、[入口](/D:/program/MacDiff/main_pretrain_text_relation.py)、[配置](/D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_text_relation_decoder3.yaml)。保留S→S、T→S decoder3/七句/remap2048/逐样本目标平滑，删除S→T decoder；用global骨架GAP256和global平滑文本512各自对4096对FIFO求相似度分布，KL(stopgrad(P_T)||P_S)，直接训练encoder，不增加投影头。共享slot/IDs，排除self，仅成功完整步提交配对条目，DDP gather、checkpoint带库；不是全样本索引库，不是对称KL，不接local关系/OPD/OTFM/多层监督。

初始lambda_relation=.1，tau_T=tau_S=.1；原uni.02、T→S权重1、A/person0、encoder8/native decoder3保持。PT400/总64/floor5e-4/noAMP/seed0→第8层LP100/总128/floor0。端口10323/10324，输出output_dir/ntu60_xsub_textrel_kl01_k4096_t01_dec3_noamp_seed0_20261027。同步当前engine_pretrain.py的新状态提交开关。新分支9通过/4缺Torch跳过，相关整合回归65通过/68跳过；无真实GPU/DDP验证、未启动训练，新组无成绩、不预编号。下方旧S→T候选仍为历史讨论，新关系分支已独立落实。

## 2026-10-27新增选择：OTFM的Inverse Cosine信噪比采样对照

用户已要求实现，现用原OTFM入口/模型及[新配置](/D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_flow_matching_inverse_cosine.yaml)，方法与完整PT&&LP命令见[FM文档顶部](/D:/program/MacDiff/tools/MACDIFF_FLOW_MATCHING.md:3)。只将uniform时间采样替换为inverse_cosine_snr_v1：均匀抽原生1000步inverse_cosine(1)索引，按t=a/[b+(1-sigma_min)*a]映射到OT物理时间，系数SNR逐点对应；OT路径、速度目标、encoder8/decoder3、A/person0、uni.02均保留，不接文本/teacher/后三层监督。PT400/noAMP/总64/floor5e-4→原第8层LP100/总128/floor0/seed0。新目录output_dir/ntu60_xsub_otfm_invcos_snr_dec3_noamp_seed0_20261027，端口10321/10322。旧uniform/T29和下方多层diffusion保持各自身份；新组无成绩、不预编号，助手未启动训练。FM+累积27项，21通过/6缺Torch跳过，真实Torch/CUDA/DDP待服务器验证。

## 2026-10-27当前选择：后三层完全共享norm和decoder的diffusion监督，第8层LP

用户已要求实现，专用[方法与完整PT&&LP命令](/D:/program/MacDiff/tools/MACDIFF_MULTILAYER_DIFFUSION.md)。encoder8/256，第6/7/8层共享原self.norm和同一原生decoder3、分别前向监督，不新增参数；同一t/noise/mask/条件dropout，L=L8+.1*(L6+L7)/2+.02*uni8。默认辅助总权重.1，可由--lambda_aux_diffusion覆盖。纯骨架、新目录、PT400/总64/seed0/min_lr5e-4/noAMP；LP100/总128/min_lr0，原入口只读第8层，不融合。端口10319/10320，输出output_dir/ntu60_xsub_multilayer678_sharednormdec3_aux01_lp8_noamp_seed0_20261027，加载pt/checkpoint-399.pth。

本次同时回传teacher AA0/CL1e-3，登记T32：PT400完整、LP附件0–89，best85.286268831%@72、70–89 mean85.122816606%，目前未见提高CL权重收益。原始附件/SHA和同窗口比较见[结果记录](/D:/program/MacDiff/STAGE1_EXPERIMENT_RESULTS.md:5)。旧AA0 CL1e-4的T30/T31平均85.95%保留；新方法无成绩、不预登记T33。蒸馏等待用户参考文献，本次未接入。

多层+累积测试23项，16通过、7因无Torch跳过；实际GPU/AMP/DDP和LP运行尚未验证，助手未启动训练。以下旧文本/teacher/FM计划保留其历史身份。

## 2026-10-23讨论存档：S→T监督改进的统一入口

用户明确要求“我们之前讨论的S到T的改进也记录起来”。本节统一整理本轮讨论，便于后续会话继续；**当前没有选定实施哪项，没有给新loss定权重/训练时机，没有接入模型或启动对应训练，也不新增成绩ID，仍29条。**讨论目标是改善原生骨架encoder的LP，文本重建/生成质量本身不能代替LP验证。

### 现状与已实现改动

当前S→T是骨架可见token及其GAP作为条件，对加噪的文本向量预测epsilon；目标是detached的global＋六local句向量（512维），不是词表token，也不是逐词生成。`sample_target_blend`沿用逐样本目标值递推。它提供对encoder的直接辅助梯度；T→S本身训练文本remap/文本条件骨架decoder，不读取骨架encoder。

新`global_mod_local_attention_v1`是已实现的**T→S文本读取方式**改动（global/time调制、六local注意力），不是下面三类S→T监督的实现。新读取全5层YAML和命令保持现有S→T diffusion、目标库和1/.1权重；不能将其后续成绩归因于尚未加入的冻结交互、特征回归或自回归。

### 三个候选与目前讨论程度

| 方向 | 监督定义及梯度 | 当前状态 |
|---|---|---|
| A. 冻结T→S decoder的交互去噪 | 骨架特征经条件适配器送入冻结文本条件骨架decoder，用GT骨架epsilon计算masked MSE；更新骨架encoder/适配器，交互loss不更新decoder | 已讨论机制与限制；冻结时机/适配接口/权重未选，未实现 |
| B. 回归配对文本特征 | 骨架GAP经轻量256→512投影，与固定CLIP句向量做归一化cosine回归；更新骨架encoder/骨架投影，文本目标停止梯度 | 已讨论显式配对对齐；先global仅为建议，未实现/选权重 |
| C. 自回归描述生成 | 按“骨架作为条件，逐词/子词生成对应描述”作候选理解；若采用，使用next-token交叉熵反传骨架encoder | 用户已问“那那种自回归的方式呢”，随后切换查看结果，尚未完成具体讨论、没有选择模型或实现 |

A的公式为`L_cross=E[MSE_masked(D_T_frozen(x_t,t;P_S(E_S(masked_aug_x))),epsilon)]`。必须区分“交互时冻结、文本本路仍更新”的交替训练和“同run先训练后固定快照”；后者只是助手建议，用户未选择。不能冻结随机初始化网络就称已有语义；冻结参数须保留输入autograd。完整冻结范围包含reader/调制/输出head，且不同于旧native decoder-body冻结、head可训练的distill。它只约束条件对去噪有用，不能保证等于配对文本语义；已有条件敏感性诊断不作为全新未做方向重复包装。

B的公式为`L_sem=mean(1-cos(P(GAP(E_S(masked_aug_x))),stopgrad(CLIP(caption))))`；归一化平方L2与之相差常数系数2。固定文本语义锚点，避免把两侧无约束可训练投影共同回归的常量解当作成功对齐。保持轻量骨架头与native任务，但这不保证encoder原表示的LP受益；描述区分度、错误描述、crop与person0配对仍限制监督。若与A结合，对齐原CLIP的骨架预测还须经与冻结D_T配套的冻结remap/归一化/位置处理进入decoder，保留输入梯度；原CLIP空间与remap后的decoder条件空间不能混用。

对C目前只记录待讨论框架：若选择生成描述，可定义`L_AR=-mean_j log p(y_j | y_<j, E_S(masked_aug_x))`。后续需明确因果mask/teacher forcing、padding归约、小型decoder共同训练还是冻结预训练语言模型，以及global/六local如何组织。现有v3缓存保存七个句向量，不是next-token监督；应从现存`captions.json`原始global/local字符串按所选tokenizer生成token ID，并保持sample/person0配对，而不是要求重新生成VLM描述或把六个身体区域当时间序列。上述是候选解释与待定项，不能写成用户已选定的训练方案。文本历史可能使语言模型少依赖骨架，caption loss降低也不证明语义信息已进入encoder或LP提高。

### 共同边界与下一步讨论项

- 对比学习、文本关系矩阵蒸馏和配对文本条件输出蒸馏只是其他候选，不自动与A/B/C全部叠加；骨架teacher AA0的InfoNCE对照是另一项已给命令，不能混称为文本语义对齐实验。
- 保留native骨架重建的建议，是为维持文本没有覆盖的姿态/轨迹信息；是否替换现有S→T、是否保留文本本路T→S及各权重，须在具体方案中明确，尚未选定。
- 后续按用户选择继续：先厘清要验证“固定decoder的重建约束”“配对特征对齐”还是“描述生成监督”，再定义单独配置与对照。不把已经做过的fixed-CLIP S→T去噪，冒充本轮直接cosine回归已经验证；不从语义目标、梯度通路或loss下降承诺LP增益。

本轮原始详细讨论继续保留在下方“骨架条件经过冻结T→S decoder获得去噪监督”及“直接语义对齐与decoder条件空间”小节，本节作为最新汇总入口。

## 2026-10-23新增配置：no-share全5层＋新global/local读取

用户要求再给新读取配置，沿用刚才全5层设置，新建`config/ntu60_xsub_joint/pretrain_madiff_text_global_local_decoder5.yaml`。对旧全5层YAML的运行字段差异严格只有`text_conditioning_mode: global_mod_local_attention_v1`及output/log路径；encoder8、三套decoder5、remap2048、S→T512/none、七句v3/sample_target_blend、text uni0/骨架uni.02、1/.1、A/person0、noAMP、PT400/global64/LP100/global128/seed0保持。旧三层和旧全5层配置均保留；冻结decoder交互、直接回归、自回归仍仅讨论，未加入本配置。

复用七句缓存，使用独立输出/新目标库，从头PT后接对应checkpoint399 LP。服务器需同步新YAML和当前`model/transformer_macdiff_text.py`、`main_pretrain.py`、`util/misc.py`、`engine_pretrain.py`。端口10313/10314为本次选择；若服务器已使用这个输出目录，另换新路径，不复用或删除旧目标库。没有启动训练或预登记T30，不推断服务器进度。

完整服务器仓库根单行命令：

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10313 main_pretrain.py --config config/ntu60_xsub_joint/pretrain_madiff_text_global_local_decoder5.yaml --disable_amp --lambda_text_to_skeleton 1 --lambda_skeleton_to_text 0.1 --batch_size 32 --accum_iter 1 --epochs 400 --warmup_epochs 20 --min_lr_epochs 20 --lr 0.001 --min_lr 0.0005 --mask_ratio 0.9 --motion_aware_tau -1 --seed 0 --output_dir output_dir/ntu60_xsub_noshare_globalmod_localattn_all_dec5_h512_remap2048_textuni0_noamp_seed0_20261023/pt --log_dir output_dir/ntu60_xsub_noshare_globalmod_localattn_all_dec5_h512_remap2048_textuni0_noamp_seed0_20261023/pt/tensorboard && OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10314 main_linprobe.py --config config/ntu60_xsub_joint/linprobe_madiff.yaml --finetune output_dir/ntu60_xsub_noshare_globalmod_localattn_all_dec5_h512_remap2048_textuni0_noamp_seed0_20261023/pt/checkpoint-399.pth --batch_size 64 --accum_iter 1 --epochs 100 --warmup_epochs 0 --min_lr_epochs 10 --lr 0.1 --min_lr 0 --seed 0 --dist_eval --output_dir output_dir/ntu60_xsub_noshare_globalmod_localattn_all_dec5_h512_remap2048_textuni0_noamp_seed0_20261023/lp --log_dir output_dir/ntu60_xsub_noshare_globalmod_localattn_all_dec5_h512_remap2048_textuni0_noamp_seed0_20261023/lp/tensorboard
```

已用两入口真实get_args_parser的AST提取结果解析CLI，并静态读取当前简单YAML字段，核对AMP/权重/层数/轮数/两卡batch/accum/LR/末端floor/seed/PT-LP路径与checkpoint399；对旧全5层配置仅三项字段不同。此检查不使用PyYAML/Torch，不等于真实YAML加载、模型构造或GPU/DDP训练验证。

## 2026-10-23讨论：骨架条件经过冻结T→S decoder获得去噪监督

用户提出替换当前S→T文本diffusion监督：两路各自训练diffusion，交互时将骨架特征作为冻结T→S decoder的条件、预测骨架噪声，反传优化骨架encoder。本节按两路分别为`E_S→D_S`骨架条件骨架去噪、`R_T(CLIP)→D_T`文本条件骨架去噪理解；用户尚未要求实现，未选冻结时机/权重/global-local适配器，不新增配置、训练命令或成绩ID。

设`x_t=a_t*x+b_t*epsilon`，`z_S=E_S(masked_aug_x)`、`c_S=P_S(z_S)`。交互loss为`L_cross=MSE_masked(D_T_frozen(x_t,t; c_S), epsilon)`，更新E_S/P_S，交互loss不更新D_T。native及文本本路各自GT epsilon监督保留；首轮不自动加入反向交互或特征/输出一致性loss。与现有S→T区别是回归骨架噪声而非文本噪声，语义来源为D_T经过文本条件训练后的参数；不从epsilon反算x0。冻结网络、优化其条件输入在梯度上成立，[Textual Inversion作者说明](https://textual-inversion.github.io/)提供相关先例，但不证明本方案的骨架LP收益。

必须区分：①交互分支冻结D_T，文本本路仍更新D_T，是跨step变化的交替训练；②同一run内先训练文本路、再固定一份D_T快照，是固定teacher。后者是便于隔离变量的候选建议，不是用户已选择的协议，更不要求外部预训练checkpoint；不能从随机初始化就冻结后宣称已有语义。冻结参数时仍须保留对c_S的autograd，不能把交互forward放进torch.no_grad或detach(c_S)。完整冻结范围包括输入/位置/时间处理、text reader/调制、decoder主体、norm与输出head；它与旧native decoder-body冻结、head可训练的decoder-distill并非同一方法。

现有骨架token256维、文本条件512维且global+六local，需要P_S和统一token/区域/mask接口；不能直接把75个可见骨架token当作七句文本。只用global须让文本训练与交互共用该接口，或明确如何用骨架生成所有local条件；交互仍喂真实local文本会留下直接文本通路，不能据loss低就证明encoder学习到语义。适配器大小也影响监督是否主要由P_S吸收。

该loss给的是“固定文本条件decoder可使用的重建条件”约束，没有显式要求c_S等于配对文本c_T；不同/偏离文本分布的条件可能产生相似噪声预测。因此不能把成功去噪等同于严格文本语义对齐、防坍塌或LP收益。应针对所用冻结快照复用已有条件敏感性诊断、比较跨噪声等级的条件影响/encoder梯度；不将此前已做的条件打乱包装成全新未做实验。若另做配对文本条件输出蒸馏，那是额外监督，尚未选定。

### 继续讨论：直接语义对齐与decoder条件空间

用户追问“怎么让骨架跟文本语义对齐，用骨架特征回归文本特征？”最直接的显式候选为固定CLIP配对目标、骨架侧轻量Linear256→512：`u_S=normalize(P(GAP(E_S(masked_aug_x))))`、`u_T=stopgrad(normalize(CLIP(caption)))`，`L_sem=mean(1-dot(u_S,u_T))`。单位向量的平方L2与该loss相差常数系数2。更新E_S/P，不由该loss更新CLIP或文本remap；不把两侧无约束可训练投影一起回归当成固定语义锚点。先global可避免同时引入六部位/缺失区域/crop对齐变量；native骨架重建保留。此为此前直接特征监督候选的细化，未实现/选权重/启动，不能把旧fixed-CLIP S→T去噪成绩当成本项直接cosine回归结果。

这是投影后表示的配对特征对齐，不能保证encoder全部维度/可线性读出的语义改善；轻量头降低高容量适配器独自拟合任务的空间，但不是保证。字幕近似、错误或与crop/person0不符会限制监督；固定目标不等于必防退化，骨架坐标/细节信息仍需native任务。跨模态对比损失（[CLIP原论文](https://arxiv.org/pdf/2103.00020)，归一化投影/配对交叉熵）和此前文本关系矩阵蒸馏是其他显式约束，均不自动加入；相近动作文本不宜全部当严格负例。

若与冻结T→S方案结合，须区分原CLIP语义空间与decoder已训练的remap条件空间：对齐原CLIP的骨架预测，再经过与冻结D_T配套的冻结R_T/归一化/位置处理进入D_T，保留对输入梯度；不能直接把原CLIP空间的预测喂给只在R_T(CLIP)上训练的decoder而假设接口一致。也可对齐冻结R_T后的目标，但那是另一种锚点，其语义几何未必等于原CLIP；两种不能混称同一loss。不自动把cosine、对比、关系蒸馏和decoder交互全部叠加。

FM状态更正：已收到T29完整PT400/LP100、best84.631246997%，详见[Stage1结果](D:/program/MacDiff/STAGE1_EXPERIMENT_RESULTS.md:5)；下一节“无结果”为10月22日历史状态。

## 2026-10-22：已新增独立原生FM基线，文本监督仍待讨论

按用户最新补充“要做Optimal flow matching”，当前以原论文OT条件高斯路径替换单路原生MacDiff的diffusion，方法`conditional_ot_noise_to_data_v2`、sigma_min=1e-4；用户已明确选择噪声→对应骨架、保持原生配对，不接mini-batch最优配对。encoder8/decoder3、global/local调制、masked损失和uni.02保留，独立入口/模型/YAML，不接文本或teacher。配置默认关闭AMP，PT400→原LP100、seed0、新`_otfm_`目录、10309/10310端口；完整命令及验证范围见[FM方法文档](D:/program/MacDiff/tools/MACDIFF_FLOW_MATCHING.md)。本地16通过/5缺Torch跳过，无训练或LP结果，不新增成绩编号。以下七句缓存和文本读取实现继续保留，候选新监督尚未实现。

## 2026-10-21：全局调制＋局部注意力已实现，监督方案仍只讨论

用户本轮授权先修改文本读取方式，并调研2025/2026论文后讨论监督；没有授权接入新监督或启动训练。当前纯骨架teacher、T28及已给三组repeat的身份继续保留。以下新结构有独立配置，不把它称为T23 repeat。

新配置：[pretrain_madiff_text_global_local_decoder3.yaml](D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_text_global_local_decoder3.yaml)。模型仍为`model.transformer_macdiff_text.Transformer`，新增`model_args.text_conditioning_mode: global_mod_local_attention_v1`；默认`concat_memory`保留所有旧配置的读取方式与旧state_dict结构。旧decoder3 YAML和三组repeat命令未改。

新配置沿用T23的七句v3缓存、person0、A归一化、encoder8/三套decoder3、remap2048、S→T512/none、sample_target_blend目标值递推、loss权重1/.1和骨架uni.02/text uni0。按用户最新精度选择显式`enable_amp: False`；输出为新目录`output_dir/ntu60_xsub_globalmod_localattn_dec3_seed0_20261021/pt`。缓存无需重编码；新目标库必须在新输出目录创建，不复用/删除历史目标库。不把当前配置当作已经运行。

### 本次实际数据流

1. global512和六个local512继续经过现有RMS residual remap。local保留人物/部位embedding和有效mask。
2. global经独立Linear512→256；时间步仍先生成64维正弦embedding，再经64→256→256的SiLU MLP。二者相加为`c:[A,256]`。
3. local单独经Linear512→256形成`[A,6,256]` memory；global不进入K/V，padding在投影前后均清零。
4. 每层独立SiLU+Linear256→2304从c产生三组shift/scale/gate，分别用于local cross-attention、骨架self-attention和FFN。每个子层先做`(1+scale)*LN(x)+shift`，再把门控输出残差加回x。调制末层权重Normal(std=.001)、shift/scale bias0、gate bias.1，使初始两路文本均有有效路径。这里不是TDSM联合更新local隐状态的完整复现。
5. 输入/输出仍为带噪骨架750位置→epsilon12维/patch，masked MSE与native共用原t/epsilon/mask。新T→S仍不读取encoder输出，不新增encoder梯度路径。S→T、干净文本目标detach、目标库递推与loss组合均未改；干净global不注入S→T。
6. no-share只复制T→S实际使用的self-attention/MLP及输入/输出模块，不复制已不用的native FeatureModulation，避免DDP未用参数。代码也支持共享attention/MLP主体；global/local调制器和reader仍独立，新配置使用no-share。
7. `main_pretrain.py`保存实际`text_conditioning_mode`，`util.misc.load_model`在加载权重前比较模式；旧checkpoint缺字段按concat_memory处理，新旧模式双向拒绝完整resume。encoder名称仍兼容原LP，旧权重不能作为新结构完整续训。

本地验证：`tests.test_macdiff_text tests.test_text_decoder_width tests.test_text_consistency`共59项，17通过/42缺Torch跳过。新增3项实际forward方法的NumPy执行检查通过（global不进local K/V但改变预测、有效local影响预测/NaN padding无影响/全空拒绝、双向resume模式检查）；新增5项真实Torch检查覆盖多层两路影响、padding、共享/no-share梯度和未用参数、loss/目标detach、resume，因无Torch未执行。Python3.8语法和diff空白检查通过；没有真实Torch/CUDA/DDP/显存/LP结果，没有启动训练。

### 2025–2026论文调研与待讨论监督

检索截至2026-10-21；以下使用论文/作者公开稿和出版社页面。论文结果的训练协议与本项目不同，不把其数值作为本项目预期收益。候选适配是本项目设计建议，不声称论文已验证这些组合或学术首创。

| 论文 | 核实的相关机制 | 对当前问题的启发 |
|---|---|---|
| [GFP: Towards Efficient General Feature Prediction in Masked Skeleton Modeling，ICCV2025](https://arxiv.org/html/2509.03609v1) | 轻量运动输入目标网络，预测从局部到全局的层级特征；联合学习使用batch方差/协方差约束防止常数解 | 辅助目标可转向多时间尺度运动摘要；不能只接两个可训练投影再MSE。已阅读方法3.1–3.3 |
| [MLS³L，Knowledge-Based Systems 2025](https://www.sciencedirect.com/science/article/abs/pii/S0950705125007063) | LMM描述引导语义蒸馏，Weighted Soft Alignment逐渐建立多对多关系并估计配对质量 | 文本监督可以传递样本关系和可信度；不必默认每个不同样本都是负例。本项核对出版社摘要和引言，未获得完整WSA公式 |
| [AMR: Exploring Adaptive Masked Reconstruction，CVPR2026](https://arxiv.org/html/2606.11450v1) | 解耦decoder支持更大patch；按多尺度局部运动能量给坐标重建加权 | 可先只借鉴运动加权，把native去噪梯度分配给更有信息的区域；大patch重建结果不直接证明epsilon加权有效。已阅读方法3.1–3.3 |
| [TranCLR: Beyond Binary Contrast，2026公开稿](https://arxiv.org/html/2604.17914v1) | 全局插值/局部时空替换构建过渡锚点，用多层几何约束描述动作连续性 | 表示关系应保留相似程度；可借鉴其连续几何动机，不能把我们的关系蒸馏说成原文算法。已阅读方法3.2–3.3；此处保守标公开稿，不依第三方列表认定会议身份 |

优先讨论三个相互独立的候选，均未实现：

**A. 七句文本提供软关系，直接监督encoder的关系矩阵。** 对固定CLIP global和各部位local分别计算batch内相似度分布（排除自身）；encoder全局/部位池化特征计算对应分布，以KL匹配两侧关系。文本端固定且detach；可加入有界的配对可信度。这样允许多个语义接近样本共同接近，不要求256维骨架逐坐标拟合512维文本或复制文本epsilon。与上轮提出的逐样本cosine回归不同，优化的是几何关系。必须先检查缓存关系是否有区分度，防止近乎均匀或错误的文本关系成为目标；部位缺失、crop与整段描述不匹配需处理。首选讨论项。

**B. 运动信息加权的native去噪。** 从同一crop的干净骨架计算多窗口关节运动能量，经有界、带正下限的映射产生detached权重；按masked位置归一化计算`sum(mask*w*MSE_epsilon)/sum(mask*w)`。直接经native decoder训练encoder；不靠S→T或EMA教师。先使用纯骨架权重，六句文本是否再用于部位权重留作第二步。静态姿态和微小动作仍可能有辨识力，不能把低运动区域权重置0。该式是对扩散的本项目适配，不是AMR原文的epsilon目标。

**C. 部位×多时间尺度的运动摘要预测。** 从同一干净crop计算固定的速度能量、净位移、关节距离/角度变化等摘要，学生masked encoder经轻量query预测各部位、多个时间尺度的摘要，native diffusion保留。固定几何目标避免双方投影共同缩小的直接退化通路；需按训练集统计标准化各类量、明确旋转/视角与crop规则。它借鉴GFP的层级目标思想，但不是复现其可学习TGN，也不把几何摘要直接称为已验证的高级语义。可后续讨论学习式目标网络，届时必须单独设计目标稳定性/防退化约束。

T25双向x0一致性已口述85.44、无明确收益；T28原始跨层AA出现严重LP下降。因此不把无额外约束的跨层MSE、简单加大AA权重或旧重建一致性重复包装成新方向。以上仅讨论，不接入loss、不调整已有训练或启动实验。

## 2026-10-21最新：三组关闭AMP的repeat，另有独立teacher AA=0对照

本日回传并由用户确认T28：teacher v3、AA=1，从checkpoint-80恢复后关闭AMP；上传PT81–399/LP0–89，并非三组repeat或teacher AA0。LP截至89 best69.802280459，70–89均值69.109958775，相对T26同范围低15.908540/16.342795pp。实际服务器源码/数值补丁版本、checkpoint SHA及LP加载输出仍缺，详情见[Stage1顶部](D:/program/MacDiff/STAGE1_EXPERIMENT_RESULTS.md)。三组关闭AMP repeat及AA0仍未收到可确认身份的新结果，不据此推断服务器进度。

用户纠正repeat只指原生decoder3、原生decoder5和最佳no-share（T23全3层），不是teacher。10月21日上一条误给的teacher repeat命令不作为这三组身份，也不据此推断其已启动。当前提供三组完整seed0/PT400→LP100、关闭AMP、全新目录的命令；旧T26等AMP结果继续保留，不覆盖。用户另要求独立AA=0组；后续AA1续训日志已登记T28，但不能由此确认梯度冲突。

`--disable_amp`已移到公共main_pretrain解析器，PT的GradScaler统一跟随args.enable_amp；默认native/no-share仍开AMP，只有显式关闭或配置False时关。teacher入口继承同一开关，默认仍关。不改三组模型/目标、batch、LR、归一化、文本缓存或LP协议，未启动任务/预登记新成绩。使用此命令前同步最新main_pretrain.py、main_pretrain_skeleton_teacher.py、util/misc.py及当前模型/配置。四条使用同两张卡，按用户自己的资源安排分别运行。

三组均从头训练，无resume；no-share以新输出路径建立新目标库，继续复用原文本特征缓存。PT总batch64、LP总128、PT min_lr5e-4/LP0、各400/100轮、seed0保持既定选择；原版5层为统一高floor基线，仍不同于旧T20的1e-5。

原生decoder3：

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10301 main_pretrain.py --config config/ntu60_xsub_joint/pretrain_madiff_decoder3.yaml --disable_amp --batch_size 32 --accum_iter 1 --epochs 400 --warmup_epochs 20 --min_lr_epochs 20 --lr 0.001 --min_lr 0.0005 --mask_ratio 0.9 --motion_aware_tau -1 --seed 0 --output_dir output_dir/ntu60_xsub_native_dec3_repeat_noamp_seed0_20261021/pt --log_dir output_dir/ntu60_xsub_native_dec3_repeat_noamp_seed0_20261021/pt/tensorboard && OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10302 main_linprobe.py --config config/ntu60_xsub_joint/linprobe_madiff.yaml --finetune output_dir/ntu60_xsub_native_dec3_repeat_noamp_seed0_20261021/pt/checkpoint-399.pth --batch_size 64 --accum_iter 1 --epochs 100 --warmup_epochs 0 --min_lr_epochs 10 --lr 0.1 --min_lr 0 --seed 0 --dist_eval --output_dir output_dir/ntu60_xsub_native_dec3_repeat_noamp_seed0_20261021/lp --log_dir output_dir/ntu60_xsub_native_dec3_repeat_noamp_seed0_20261021/lp/tensorboard
```

原生decoder5，统一min_lr=5e-4：

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10303 main_pretrain.py --config config/ntu60_xsub_joint/pretrain_madiff.yaml --disable_amp --batch_size 32 --accum_iter 1 --epochs 400 --warmup_epochs 20 --min_lr_epochs 20 --lr 0.001 --min_lr 0.0005 --mask_ratio 0.9 --motion_aware_tau -1 --seed 0 --output_dir output_dir/ntu60_xsub_native_dec5_repeat_noamp_minlr5e4_seed0_20261021/pt --log_dir output_dir/ntu60_xsub_native_dec5_repeat_noamp_minlr5e4_seed0_20261021/pt/tensorboard && OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10304 main_linprobe.py --config config/ntu60_xsub_joint/linprobe_madiff.yaml --finetune output_dir/ntu60_xsub_native_dec5_repeat_noamp_minlr5e4_seed0_20261021/pt/checkpoint-399.pth --batch_size 64 --accum_iter 1 --epochs 100 --warmup_epochs 0 --min_lr_epochs 10 --lr 0.1 --min_lr 0 --seed 0 --dist_eval --output_dir output_dir/ntu60_xsub_native_dec5_repeat_noamp_minlr5e4_seed0_20261021/lp --log_dir output_dir/ntu60_xsub_native_dec5_repeat_noamp_minlr5e4_seed0_20261021/lp/tensorboard
```

最佳no-share（T23，全3层）：

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10305 main_pretrain.py --config config/ntu60_xsub_joint/pretrain_madiff_text_sentence_sample_target_blend_noshare_decoder3.yaml --disable_amp --lambda_text_to_skeleton 1 --lambda_skeleton_to_text 0.1 --batch_size 32 --accum_iter 1 --epochs 400 --warmup_epochs 20 --min_lr_epochs 20 --lr 0.001 --min_lr 0.0005 --mask_ratio 0.9 --motion_aware_tau -1 --seed 0 --output_dir output_dir/ntu60_xsub_noshare_all_dec3_t23_repeat_noamp_seed0_20261021/pt --log_dir output_dir/ntu60_xsub_noshare_all_dec3_t23_repeat_noamp_seed0_20261021/pt/tensorboard && OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10306 main_linprobe.py --config config/ntu60_xsub_joint/linprobe_madiff.yaml --finetune output_dir/ntu60_xsub_noshare_all_dec3_t23_repeat_noamp_seed0_20261021/pt/checkpoint-399.pth --batch_size 64 --accum_iter 1 --epochs 100 --warmup_epochs 0 --min_lr_epochs 10 --lr 0.1 --min_lr 0 --seed 0 --dist_eval --output_dir output_dir/ntu60_xsub_noshare_all_dec3_t23_repeat_noamp_seed0_20261021/lp --log_dir output_dir/ntu60_xsub_noshare_all_dec3_t23_repeat_noamp_seed0_20261021/lp/tensorboard
```

独立teacher AA=0完整命令见[骨架教师方法文档](D:/program/MacDiff/tools/MACDIFF_SKELETON_TEACHER.md)顶部；该组不纳入三组repeat。AA=0仍计算未加权AA值以保持DDP参数参与图，但其损失贡献/有效梯度为0；InfoNCE保留1e-4，不能叫作原生MacDiff对照。


## 2026-10-19已选三组seed0完整PT→LP（10月20日：dec3已完成T26）

用户要求原版decoder3、原版decoder5、no-share最好的一次再做PT→LP，随后明确选择保持seed0完整重复检查同设置复现性；进一步明确三组PT min_lr统一5e-4，原版5层作为新基线，不按T20历史1e-5重复。原版3层对应T22、最佳no-share对应T23（最高best86.280931625，不是T21最高后期均值或T25 nscale）。三组encoder仍8层；native3是decoder3，no-share为native/T→S/S→T各3层。10月20日原版3层完整日志已回传T26；其余两组未收到新成绩，不推断服务器是否已经启动。

T26 best85.710819994%@73、last85.419699192、末20均值85.445172265±.054504224%；相对T22 best+.248665681pp、末20+.053372153pp。PT/LP学习率与T22逐epoch完全相同，PT末20loss高.499253692%；两阶段从epoch0就非逐项复现。同seed且历史隔离入口→当前根入口，不估计独立PT seed方差或把分差全归随机性。完整比较、原始附件及SHA见[Stage1最新结果](D:/program/MacDiff/STAGE1_EXPERIMENT_RESULTS.md)。

独立骨架teacher v2已回传T27（用户确认版本），PT400完整、LP附件仅0–95；截至95 best85.377244084，76–95均值85.220766640，同窗口比T26分别低.333576/.225012pp。发现当前双侧可训练投影允许一起趋0使AA低而encoder不改善；日志AA四数量级下降与该风险相符，尚缺checkpoint投影/方差证据。优先做现有checkpoint诊断，不自动改权重、冻结投影或新增完整训练。此组不是三组复核中的no-share，完整记录见Stage1结果与handoff。

| 组 | native/T→S/S→T深度 | PT min_lr | 对照身份 |
|---|---|---|---|
| 原版3层 | 3/—/— | 5e-4 | 已完成T26 best85.710819994，T22历史best85.462154313 |
| 原版5层 | 5/—/— | 5e-4 | 未回传新成绩；新增统一floor基线，T20的1e-5/86.001940813仅作历史参照 |
| 最佳no-share | 3/3/3 | 5e-4 | 未回传复跑成绩；T23历史best86.280931625 |

共同PT400/双卡每卡32/accum1/总64/lr.001/warmup20/末20 floor，A归一化/person0/mask.9/无PT旋转/小joint_noise保持；LP100/每卡64/总128/lr.1/warmup0/min_lr0/末10 floor/dist_eval/checkpoint399/seed0。no-share仍为七句v3/sample_target_blend、remap2048、S→T decoder512/none、权重1/.1、text uni0、骨架uni.02；不是新一致性方法。每组用新的PT/LP目录，no-share重新建目标库，不加resume、不读旧目标库；只在LP加载本组新PT checkpoint399。

新增根配置 `config/ntu60_xsub_joint/pretrain_madiff_decoder3.yaml`；5层及no-share复用现有根配置。不修改主入口/模型/engine以“强制确定性”，避免这轮同时改变训练实现；未承诺同seed逐bit一致。T22历史原版使用隔离入口，当前按用户约定统一根入口，因此相对历史的差异仍可能含源码/环境变化，不能一律归因随机采样。相同seed重跑用于检查执行可复现性，不估计独立seed方差；比较best、last、末20均值及完整曲线，保留Namespace/代码版本/新checkpoint加载路径。dec3已登记T26，另两组返回前不预登记完成或推断服务器进度。

以下保留各组Linux单行PT&&LP供核对身份，从仓库根运行，无需cd。原版3层已完成T26，不重复启动。三条都使用GPU0,1；另外两组只按用户自己的实际运行进度使用，助手不自动启动或重跑。

原版decoder3：已完成T26，以下保留命令身份：

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10287 main_pretrain.py --config config/ntu60_xsub_joint/pretrain_madiff_decoder3.yaml --batch_size 32 --accum_iter 1 --epochs 400 --warmup_epochs 20 --min_lr_epochs 20 --lr 0.001 --min_lr 0.0005 --mask_ratio 0.9 --motion_aware_tau -1 --seed 0 --output_dir output_dir/ntu60_xsub_native_dec3_repeat_seed0_20261019/pt --log_dir output_dir/ntu60_xsub_native_dec3_repeat_seed0_20261019/pt/tensorboard && OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10288 main_linprobe.py --config config/ntu60_xsub_joint/linprobe_madiff.yaml --finetune output_dir/ntu60_xsub_native_dec3_repeat_seed0_20261019/pt/checkpoint-399.pth --batch_size 64 --accum_iter 1 --epochs 100 --warmup_epochs 0 --min_lr_epochs 10 --lr 0.1 --min_lr 0 --seed 0 --dist_eval --output_dir output_dir/ntu60_xsub_native_dec3_repeat_seed0_20261019/lp --log_dir output_dir/ntu60_xsub_native_dec3_repeat_seed0_20261019/lp/tensorboard
```

原版decoder5：新增min_lr=5e-4基线：

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10289 main_pretrain.py --config config/ntu60_xsub_joint/pretrain_madiff.yaml --batch_size 32 --accum_iter 1 --epochs 400 --warmup_epochs 20 --min_lr_epochs 20 --lr 0.001 --min_lr 0.0005 --mask_ratio 0.9 --motion_aware_tau -1 --seed 0 --output_dir output_dir/ntu60_xsub_native_dec5_minlr5e4_seed0_20261019/pt --log_dir output_dir/ntu60_xsub_native_dec5_minlr5e4_seed0_20261019/pt/tensorboard && OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10290 main_linprobe.py --config config/ntu60_xsub_joint/linprobe_madiff.yaml --finetune output_dir/ntu60_xsub_native_dec5_minlr5e4_seed0_20261019/pt/checkpoint-399.pth --batch_size 64 --accum_iter 1 --epochs 100 --warmup_epochs 0 --min_lr_epochs 10 --lr 0.1 --min_lr 0 --seed 0 --dist_eval --output_dir output_dir/ntu60_xsub_native_dec5_minlr5e4_seed0_20261019/lp --log_dir output_dir/ntu60_xsub_native_dec5_minlr5e4_seed0_20261019/lp/tensorboard
```

最佳no-share全3层：按T23配置重跑seed0：

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10291 main_pretrain.py --config config/ntu60_xsub_joint/pretrain_madiff_text_sentence_sample_target_blend_noshare_decoder3.yaml --lambda_text_to_skeleton 1 --lambda_skeleton_to_text 0.1 --batch_size 32 --accum_iter 1 --epochs 400 --warmup_epochs 20 --min_lr_epochs 20 --lr 0.001 --min_lr 0.0005 --mask_ratio 0.9 --motion_aware_tau -1 --seed 0 --output_dir output_dir/ntu60_xsub_noshare_all_dec3_t23_repeat_seed0_20261019/pt --log_dir output_dir/ntu60_xsub_noshare_all_dec3_t23_repeat_seed0_20261019/pt/tensorboard && OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10292 main_linprobe.py --config config/ntu60_xsub_joint/linprobe_madiff.yaml --finetune output_dir/ntu60_xsub_noshare_all_dec3_t23_repeat_seed0_20261019/pt/checkpoint-399.pth --batch_size 64 --accum_iter 1 --epochs 100 --warmup_epochs 0 --min_lr_epochs 10 --lr 0.1 --min_lr 0 --seed 0 --dist_eval --output_dir output_dir/ntu60_xsub_noshare_all_dec3_t23_repeat_seed0_20261019/lp --log_dir output_dir/ntu60_xsub_noshare_all_dec3_t23_repeat_seed0_20261019/lp/tensorboard
```

## 2026-10-19历史教师讨论（现已实现独立v2，见handoff）

用户明确不采用已预训练模型作教师，转向可见骨架更多、decoder层对齐等思路；这覆盖下节冻结T23双教师的历史候选。三组既定复跑命令保持，不在其中加入教师机制。本轮仅讨论，不新增/修改教师训练实现、参数或默认权重。

优先候选是输入不对称的在线教师：学生骨架encoder仍只见10% patch；EMA骨架教师见同一crop的50%～100%，可采用嵌套mask使学生可见集合为教师的子集，输入更干净，教师自身从学生初始化并随成功优化步EMA更新，不冻结随机网络。教师target不反传，额外信息来自更完整骨架上下文，EMA只做时间平滑。原native/T→S GT epsilon损失继续训练学生；先仅GT warmup，再渐增蒸馏权重，避免一开始强拟合随机教师。先预测归一化的全局/对应区域encoder特征；若预测masked patch特征需位置查询predictor补齐学生看不到的token，不能直接将75个可见token与750个教师token逐项MSE。完整教师的注意力计算明显更贵，50%可见可作为计算量折中；不承诺目标更准或LP提高。

要保留文本→encoder作用，可考虑从头学习的双EMA链：更多可见骨架→E_bar教师→监督在线R_S（辅助512→256头）；R_S通过原T→S GT denoising和该项学习，R_bar为在线remap的EMA→监督masked E_S（辅助256→512头）。教师均停止梯度、学生各自GT监督保留，原CLIP缓存照常使用，不额外加载已训练MacDiff checkpoint。此为两个阶段上互相提供目标的候选，不是随机初始化后冻结；EMA并不保证不坍塌，权重和warmup尚未选择。不使用text uniformity/对比学习/跨样本关系损失。若仅两个学生分别回归同一个骨架教师，则native梯度不含文本，此边界保持。

decoder层对齐本身没有新增输入信息：两个随机学生直接对齐仍有移动目标问题。两路独立t/噪声时，隐藏状态含各自噪声/时间步成分，逐层坐标MSE可能要求匹配不可推断的干扰；可先对齐不含扩散噪声的条件投影/语义汇聚特征。若要做逐patch decoder层蒸馏，建议独立辅助前向共享同一x_t/t、让教师多见骨架，选一两个层经归一化/投影后单向回归teacher stopgrad，不同时强制所有层相同。专用共享x_t/t会改变此前两路独立采样的实验设置，需单独声明/消融；保留各路GT epsilon任务。不直接对齐原始attention图（token数和可见性可能不同），位置/层语义必须对应。

其他额外视角候选：①教师见更长时间/更密帧，学生见短片段/稀疏帧，目标先用全局动作或明确重叠时段，注意阶段不完全可预测；②教师见两个人，学生仍person0，对齐全局交互特征而非强迫恢复不可见第二人轨迹，空人物槽需排除；③教师用joint+bone+velocity，学生只joint，派生运动视角提供结构归纳偏置但没有新增原始信息；④教师同时见更多骨架+文本、学生各只见本模态，为两路提供共同融合目标。这需要一个实际受到GT监督的在线融合分支再EMA；未训练的无梯度融合层不会自动变成可靠教师。多mask/多时段教师平均可以降低目标采样方差，属于目标平滑而非额外原始信息。不一口气合并以上候选；先检验可见信息增加能否带来收益。

原论文依据：data2vec让masked输入预测完整输入的上下文latent，并考察教师上下文大小（https://proceedings.mlr.press/v162/baevski22a/baevski22a.pdf）；I-JEPA采用context预测target block特征并强调target/context采样设计（https://arxiv.org/abs/2301.08243）；DINO包含动量encoder与multi-crop（https://arxiv.org/abs/2104.14294）。以上骨架/文本适配及推荐比例是本项目设计推断，不是这些论文已经验证的MacDiff结果。

## 2026-10-19 历史教师候选：预训练双教师，已被用户最新约束替代

用户询问如何为native/T→S两路提供教师信息。目前只讨论，不新增训练方法/配置，不承诺LP收益。一个完整候选是从已训练的T23 PT checkpoint中冻结两部分作为双教师：骨架encoder E_bar与已优化的文本remap R_bar；不使用LP分类头，不加载S→T decoder或动态目标库。优先等待T23重跑复核，再决定是否采用。第一版只做全局特征，教师eval/no_grad且完全冻结，不先引入EMA/七区域融合。

骨架教师看相同crop/person0/A空间的完整干净骨架，无mask/小扰动，得到256维z_skeleton；文本教师读取现有CLIP global缓存，经固定R_bar得到512维z_text。学生仍是native masked encoder E_S和在线文本remap R_S/T→S decoder，各自保留GT epsilon监督。增加的两个交叉特征目标为：`L_S<-T = MSE(normalize(P_S(E_S(masked_aug_x))), stopgrad(normalize(R_bar(c))))`；`L_T<-S = MSE(normalize(P_T(R_S(c))), stopgrad(normalize(E_bar(clean_full_x))))`。P_S为256→512辅助头，P_T为512→256辅助头，两者可训练，教师不能反传；LP只取E_S，不带辅助头。两空间分别回归，不把256维和512维特征直接相加。native明确收到固定文本教师的信息，在线remap明确收到固定骨架教师的信息；没有两名随机学生互相定义移动目标。

这是latent蒸馏，不从epsilon反算x0，目标不含扩散t的inverse-SNR放大；native/T→S的噪声监督仍可独立全t。新增特征loss的权重未选定，不自动沿用旧1/1/1。与现有decoder distill不同：旧分支冻结decoder主体/训练随机head并做noise2joint；此提案冻结encoder/remap教师、学生decoder保持原训练。与旧fixed_clip/S→T diffusion不同，来源是已训练remap快照且监督直接进入encoder特征，但固定CLIP和EMA remap已经尝试过，不能把它包装成新增外部语义知识。

关键边界：仅让两个学生分别回归同一个骨架教师z_skeleton，native梯度不依赖文本，不自动实现文本→encoder指导；因此上面的具体候选采用交叉教师目标。T23的高LP只支持其骨架encoder作为候选，不证明其remap更有判别力，也不能保证新文本教师更可靠或超过教师LP；完整骨架教师有额外计算。冻结/EMA只提供稳定性，不等于目标正确。双教师和两个辅助loss均尚未选定/实现，不能据论文类比承诺收益。

参考原论文：data2vec用完整输入的上下文latent监督masked输入（https://proceedings.mlr.press/v162/baevski22a.html）；Mean Teacher使用模型权重EMA（https://arxiv.org/abs/1703.01780）。这里的冻结预训练教师为本项目候选设计，并非上述论文已验证MacDiff文本收益；旧remap EMA/逐样本文本目标平滑与整个encoder teacher不是同一实现。

本轮静态核对：新原版3层配置的模型字段与T22历史配置语义一致；与主5层模型配置补齐既有one_person=True/self_shift=False默认后仅decoder_depth不同。三条PT/LP命令均通过实际argparse解析，核对seed/轮数/batch/累积/lr与floor/新目录/新checkpoint路径/六个端口/no-share三层和1/.1权重。检查不导入Torch；本地无PyYAML，配置比较采用已知两层YAML结构静态读取，不称实际PyYAML/Torch/GPU验证。未新增测试或启动训练。

## 2026-10-19 最新回传：nscale一致性T25，口述LP best85.44%

用户明确确认85.44是下方nscale新组（两路3层、主要权重1/1/1）的LP最高值，登记T25口述结果。没有新PT/LP附件，不补推best epoch/last/末20均值、完整轮数和服务器实际参数。约等于T22原版3层85.462154313，较T23旧no-share全3层86.280931625低.84pp，本次没有显示新增一致性的LP收益；数值尺度修复不能视为表示改进或全程数值稳定验证。

当前应收紧结论：T23单次+.818777pp、T21单次+.218341pp是正向观察，尚无同协议独立PT重复确认其稳健性；T24/T25没有延续收益，也不能直接证明所有差异都是原生噪声。T20同一PT权重/同seed重复LP逐epoch完全一致，只说明那组LP可复现，不估计PT训练方差；末20轮std不作跨seed方差。不安排用户已拒绝的seed1，也不把固定CLIP/条件打乱等已做诊断包装成新方向。助手建议暂不继续扩展一致性和凭loss降低堆模块；用户尚未选择新实验/取消旧任务，本次不改代码/配置、不启动/重跑训练。下方命令保留运行身份，不是默认重跑建议。

## 2026-10-18 当前修复：全时间步双向一致性按噪声尺度归一化

用户问如何修复epoch8的巨大一致性尖峰。已修改现有新方法的模型/入口/YAML/测试，保留两路完整0–999独立t与噪声、双侧梯度、各自GT epsilon监督、masked位置和条件drop筛选、三个主要权重1/1/1、骨架uni.02。原版、旧text分支、decoder distill和公共engine均未改动；没有启动/重启训练或新服务器成绩。

每个参与样本先计算 `scale=sqrt(k_native²+k_text²+1e-6)`，`k=sqrt((1-alpha_bar)/alpha_bar)`，再以FP32计算 `masked_mean(((x0_native-x0_text)/scale)²)`，最后对参与样本平均。等价于每样本raw MSE除以两路inverse-SNR之和；必须先按样本归一化再平均，不能先对batch求raw均值再除一个batch系数。此为一致性loss定义的修改，保留全t但不再对极高噪声样本赋予巨大隐含权重；epsilon残差在归一化差里的系数绝对值均不大于1。没有改原native/T2S epsilon损失、没有套用原MinSNR选项、没有截断坐标或detach任一侧。loss_consistency现记录归一化值（乘外层权重前），fraction定义不变；不能把新旧consistency数值直接当表示质量比较。

YAML模型字段 `consistency_normalization: noise_scale_v1` 在入口缺省时也自动补入并保存。原有严格恢复检查会拒绝旧raw-MSE checkpoint，因此此次从头训练，不加resume。输出根换为 `output_dir/ntu60_xsub_text_consistency_nscale_dec3_remap2048_ep400_seed0_20261018`，保留失败任务目录。同步 `model/transformer_macdiff_text_consistency.py`、`main_pretrain_text_consistency.py`、`config/ntu60_xsub_joint/pretrain_madiff_text_consistency.yaml`；`tests/test_text_consistency.py`供服务器运行回归检查。

非有限loss时抛出的异常带rank、native/骨架uni/T2S/consistency/total、两路t、condition_kept，以及预测/encoder/remap的非有限元素数量，避免rank1的print被静默。正常训练日志保持原有精简字段。

新单行PT&&LP命令（Linux仓库根运行，不需cd）：

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10285 main_pretrain_text_consistency.py --config config/ntu60_xsub_joint/pretrain_madiff_text_consistency.yaml --lambda_text_to_skeleton 1 --lambda_consistency 1 --batch_size 32 --accum_iter 1 --epochs 400 --warmup_epochs 20 --min_lr_epochs 20 --lr 0.001 --min_lr 0.0005 --mask_ratio 0.9 --motion_aware_tau -1 --seed 0 --output_dir output_dir/ntu60_xsub_text_consistency_nscale_dec3_remap2048_ep400_seed0_20261018/pt --log_dir output_dir/ntu60_xsub_text_consistency_nscale_dec3_remap2048_ep400_seed0_20261018/pt/tensorboard && OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10286 main_linprobe.py --config config/ntu60_xsub_joint/linprobe_madiff.yaml --finetune output_dir/ntu60_xsub_text_consistency_nscale_dec3_remap2048_ep400_seed0_20261018/pt/checkpoint-399.pth --batch_size 64 --accum_iter 1 --epochs 100 --warmup_epochs 0 --min_lr_epochs 10 --lr 0.1 --min_lr 0 --seed 0 --dist_eval --output_dir output_dir/ntu60_xsub_text_consistency_nscale_dec3_remap2048_ep400_seed0_20261018/lp --log_dir output_dir/ntu60_xsub_text_consistency_nscale_dec3_remap2048_ep400_seed0_20261018/lp/tensorboard
```

本地验证：新方法20项中10通过、10因无Torch跳过；与既有text/cache/累积策略合计65项中28通过、37因无Torch/PyYAML跳过。纯标量回归直接使用实际扩散日程和实际scale函数，核验t999误差放大被消除、全部选定端点组合的残差系数界限，以及入口对旧raw恢复拒绝/新定义接受。3个Python文件的3.8语法、配置模型字段与构造器对应、新命令PT CLI/输出目录/LP权重路径及PT/LP floor静态检查通过，不是实际PyYAML解析。Torch测试已改为归一化公式/每样本平均/双侧梯度并增加rank1异常检查，但本地无法执行这些项，不称CUDA/AMP/DDP或服务器稳定性通过。默认Python缺NumPy的既有测试导入失败后，使用bundled有NumPy解释器完成上述检查，未安装依赖。以下raw MSE定义、旧命令和“尚未修复”表述保留历史身份，以本节最新修复为准。

## 2026-10-18 运行回传：raw x0一致性出现巨大尖峰，epoch8退出

用户已在服务器执行下方新方法1/1/1命令。附件 `C:/Users/97537/.codex/attachments/2e01d102-ea57-44e8-9966-82d1b0cbcb97/已粘贴的文本.txt`，SHA256 `8ed7fa6547c5038bdd8b2afada49e4053f8bf6c9daf416144bbbc3d5bf2e15c2`。epoch8 step400：native累计均值.6815、T2S 1.1941、consistency 768.7751、total 770.6706；consistency近期20步中位数仅2.8900，step280累计均值曾963.0015，显示严重尖峰。skel_uni累计均值.9928，是骨架token间cos²而非已移除的text uniformity；提示样本内token近共线，不足以确认跨样本坍塌。

rank1正退出码11与公共engine非有限loss保护吻合；其print默认被非master日志抑制，附件未含失败批次NaN/Inf具体分支。未见OOM证据，不能把launcher traceback或之前随机打印的t563当作根因。下方PT&&LP命令因PT失败不会继续LP，尚无完整新成绩，不登记完成的T25。

raw x0误差会按k=sqrt((1-alpha_bar)/alpha_bar)放大epsilon预测残差，两路独立t时一致性为||kS*rS-kT*rT||²。实际inverse_cosine/tau1/1000步的k²：t500=.9774、t990=55.17、t998=499.61、t999=500606.93，最后一个时间步的放大尤其明显。当前helper已FP32还原并算MSE；外层1/1/1仍可能令一致性占总loss数值99.75%，不代表测得梯度占比。尚缺失败批次t/张量有限性，精确非有限来源未捕获。

下一步可讨论保留全t和双侧梯度，将每样本一致性除以kS²+kT²+epsilon以限制噪声尺度放大；这是待用户选择的loss定义变更。本次仅记录分析，未改变raw MSE、全t、默认权重或训练代码，不启动/重启任务。以下“未运行”描述是实现时状态，本节真实失败回传优先；下方原命令保留身份，不代表建议原样重跑。

## 2026-10-18 新方法已实现：native + T→S，双向骨架重建一致性

用户已明确要求新建方法、替换S→T文本去噪。独立入口 [main_pretrain_text_consistency.py](D:/program/MacDiff/main_pretrain_text_consistency.py)、模型 [transformer_macdiff_text_consistency.py](D:/program/MacDiff/model/transformer_macdiff_text_consistency.py)、配置 [pretrain_madiff_text_consistency.yaml](D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_text_consistency.yaml)。原版、既有no-share、decoder蒸馏及公共engine/入口均未改动；没有新实验成绩/编号，没有启动训练。

用户选择：**两侧均独立采样完整0–999时间步与高斯噪声，所有时间步都参加一致性，不采用300–700筛选、SNR重加权或输出裁剪。**独立抽样的整数t可能偶然相同，不强制排斥相等。两路对应同一裁剪后的干净骨架、同一person0/A归一化；native仍由随机mask后的骨架encoder条件去噪，T→S由在线remap的global+6local文本条件去噪。native/T→S各3层、remap512→2048→512，保留T23公共PT400/总64/lr.001/min_lr5e-4、LP100/总128/lr.1/min_lr0、seed0，AMP和骨架uniformity.02保持。

总loss为 `native_epsilon_MSE + lambda_text_to_skeleton * T2S_epsilon_MSE + lambda_consistency * masked_MSE(x0_native, x0_text) + .02 * skel_uni`；T→S默认权重1。两侧各自使用对应的真实epsilon监督；一致性从两个噪声预测分别经 `_predict_xstart_from_eps` 还原归一化x0，FP32计算masked位置MSE，两侧均不detach，因此会同时更新encoder/native decoder和remap/T→S decoder。MSE只写一次，不重复加反向项，不反A归一化、不做DDIM多步采样。native保留condition dropout.1，被drop的样本只参与各自GT训练、不参与一致性；日志consistency_fraction反映参加比例。

**用户随后明确选择三个主要loss默认权重均为1：native/T→S/consistency=1/1/1，骨架uniformity仍为.02。**YAML顶层 `lambda_consistency: 1.0`，模型默认与入口未指定时的回退值也为1；可用 `--lambda_consistency WEIGHT` 覆盖。负数/非有限值拒绝；0仍允许做“native+T→S，无一致性”的基线。全时间步的原始x0 MSE保留高噪声端的误差放大，代码没有暗中改变用户选择。

不构造S→T decoder，不保存/读取逐样本文本目标库，不做text target EMA或text uniformity。直接复用现有七句缓存和其身份检查，无需重新生成文本。新增日志仅loss_diff/loss_uniformity/loss_text_to_skeleton/loss_consistency/consistency_fraction及公共LR/GPU日志，不恢复用户已删除的energy/variance/valid token统计。移除目标库后没有其读写/快照耗时字段。

模型保留native encoder state_dict命名，LP使用根入口/主配置。`--resume`只接受本方法完整checkpoint，严格检查模型参数（含一致性权重和decoder深度），再沿用现有缓存身份/人物对齐/优化器/scaler恢复；不允许把T23/T24或native checkpoint当作本方法resume。新方法使用独立输出目录，旧权重和目标库不复用。还没有实际Torch/CUDA/DDP训练验证。

以下是用户已选默认1/1/1的一条完整PT&&LP命令，无需额外设置权重变量。若比较不同权重，另选各自未使用的输出根目录。

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10285 main_pretrain_text_consistency.py --config config/ntu60_xsub_joint/pretrain_madiff_text_consistency.yaml --lambda_text_to_skeleton 1 --lambda_consistency 1 --batch_size 32 --accum_iter 1 --epochs 400 --warmup_epochs 20 --min_lr_epochs 20 --lr 0.001 --min_lr 0.0005 --mask_ratio 0.9 --motion_aware_tau -1 --seed 0 --output_dir output_dir/ntu60_xsub_text_consistency_dec3_remap2048_ep400_seed0_20261018/pt --log_dir output_dir/ntu60_xsub_text_consistency_dec3_remap2048_ep400_seed0_20261018/pt/tensorboard && OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10286 main_linprobe.py --config config/ntu60_xsub_joint/linprobe_madiff.yaml --finetune output_dir/ntu60_xsub_text_consistency_dec3_remap2048_ep400_seed0_20261018/pt/checkpoint-399.pth --batch_size 64 --accum_iter 1 --epochs 100 --warmup_epochs 0 --min_lr_epochs 10 --lr 0.1 --min_lr 0 --seed 0 --dist_eval --output_dir output_dir/ntu60_xsub_text_consistency_dec3_remap2048_ep400_seed0_20261018/lp --log_dir output_dir/ntu60_xsub_text_consistency_dec3_remap2048_ep400_seed0_20261018/lp/tensorboard
```

新增测试 [test_text_consistency.py](D:/program/MacDiff/tests/test_text_consistency.py) 包含定义/CLI/恢复、全时间步两次独立加噪、raw x0公式、双侧梯度、条件丢弃/空参与集、person0/缓存padding及LP加载。无Torch部分可直接运行；模型/梯度测试需在Torch环境运行 `python -m unittest tests.test_text_consistency -v`。本地两个解释器均无Torch，不能把跳过项称为GPU/DDP已验证。

本地验证：新方法17项测试中8通过、9因缺Torch跳过；既有text/cache/累积策略45项中18通过、27因缺Torch/PyYAML跳过。新Python文件通过3.8语法检查，YAML顶层字段/模型字段与实际parser/构造器静态对应；随后默认权重改为1/1/1，默认与覆盖检查同步更新。未声称实际PyYAML/Torch导入、CUDA/AMP/DDP或服务器训练通过。

## 2026-10-18 最新：5/5/3已回传T24，未带来LP收益

用户明确确认此次为下文10月16日5/5/3组。T24 best85.522804492%@89、last85.383309101、末20均值85.414240689±.054243770；相对T23全3层best−.758127134pp、末20−.675642897pp。PT/LP LR逐epoch相同、text uni0/权重1/.1/完整40091目标库核验通过。native+.02*skel_uni、T→S、S→T末20均值分别降低4.739631%、3.808544%、26.559421%，说明更低训练目标未转成更好的LP。

本次仅两套骨架decoder同时加深，不能分开归因native或T→S；减少encoder条件依赖、remap目标变易等为待验证机制，不能确诊shortcut/过拟合。T23保留为当前最高best候选，T21后期均值仍最高；不自动恢复text uni、改LR/权重、加dropout或启动新矩阵。完整统计见[Stage1结果](D:/program/MacDiff/STAGE1_EXPERIMENT_RESULTS.md:9)及[no-share分析](D:/program/MacDiff/tools/vlm_pilot/NOSHARE_NORMB_LOG_ANALYSIS.md:3)。当前共24条Stage1记录，5/5/3不再待结果；decoder蒸馏仍未回传。

## 2026-10-16运行身份：no-share 5/5/3，已于10月18日完成T24

用户已要求该组命令；新增独立配置 [pretrain_madiff_text_sentence_sample_target_blend_noshare_dec5_st3.yaml](D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_text_sentence_sample_target_blend_noshare_dec5_st3.yaml)，保留T23原配置/权重/目标库。native/T→S/S→T=5/5/3；text decoder hidden512/output_norm=none、remap2048、text uni0、骨架uni.02、权重1/.1、A/person0、七句v3/sample_target_blend、PT400/双卡每卡32/accum1/seed0/lr.001/min_lr.0005；LP100/每卡64/accum1/lr.1/min_lr0/checkpoint399。

相对T23仅模型参数 `decoder_depth: 3→5`（同时控制native和独立T→S的深度），`text_decoder_depth: 3`保持；其余模型/数据/优化参数相同，仅更换PT/LP输出目录。两套骨架decoder同时加深，不能分别归因各自贡献；PT floor继续5e-4。从头训练，不传resume/finetune给PT，不复用T23目标库。以下命令已由用户执行并于10月18日回传T24，保留核对身份，不重复启动。

历史运行命令：从仓库根运行，PT/LP均使用根目录主文件和主配置：

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10283 main_pretrain.py --config config/ntu60_xsub_joint/pretrain_madiff_text_sentence_sample_target_blend_noshare_dec5_st3.yaml --lambda_text_to_skeleton 1 --lambda_skeleton_to_text 0.1 --batch_size 32 --accum_iter 1 --epochs 400 --warmup_epochs 20 --min_lr_epochs 20 --lr 0.001 --min_lr 0.0005 --mask_ratio 0.9 --motion_aware_tau -1 --seed 0 --output_dir output_dir/ntu60_xsub_noshare_dec5_st3_h512_remap2048_textuni0_ep400_minlr5e4_pt32_lp64_seed0_20261016/pt --log_dir output_dir/ntu60_xsub_noshare_dec5_st3_h512_remap2048_textuni0_ep400_minlr5e4_pt32_lp64_seed0_20261016/pt/tensorboard && OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10284 main_linprobe.py --config config/ntu60_xsub_joint/linprobe_madiff.yaml --finetune output_dir/ntu60_xsub_noshare_dec5_st3_h512_remap2048_textuni0_ep400_minlr5e4_pt32_lp64_seed0_20261016/pt/checkpoint-399.pth --batch_size 64 --accum_iter 1 --epochs 100 --warmup_epochs 0 --min_lr_epochs 10 --lr 0.1 --min_lr 0 --seed 0 --dist_eval --output_dir output_dir/ntu60_xsub_noshare_dec5_st3_h512_remap2048_textuni0_ep400_minlr5e4_pt32_lp64_seed0_20261016/lp --log_dir output_dir/ntu60_xsub_noshare_dec5_st3_h512_remap2048_textuni0_ep400_minlr5e4_pt32_lp64_seed0_20261016/lp/tensorboard
```

## 近期结果速查（T14–T28）

no-share深度按native/T→S/S→T排序；完整末20为LP epoch80–99均值。T27仅收到0–95、T28仅0–89，各自标注76–95/70–89，不能与其他行80–99直接配对。T15/T16/T19为口述，其余证据及不同协议详见结果表，不能作跨配置单因素排序。T21为remap512/text uni.02，T23为remap2048/text uni0。

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

T23相对配对T22：best+.818777pp、末20+.698083pp；T23相对T21：best+.060650pp、末20−.044578pp。T24相对T23：best−.758127pp、末20−.675643pp。当前最高best仍为T23，最高已知末20均值仍为T21；5/5/3已完成，decoder蒸馏没有本次新回传结果。


## 2026-10-16 最新结果：T23 no-share已回传，T22/T23配对完成

LP best **86.280931625%@73**，last **86.080786054%**，末20 **86.089883586±0.065868988%**。相对T22原版3层：best **+0.818777313pp**、last **+0.745997098pp**、末20 **+0.698083473pp**；相对T21旧no-share5层：best **+0.060650170pp**、末20 **-0.044577872pp**，峰值略高但后期整体略低，不能称全面超过旧组。

本次no-share按既有全3层/decoder512/remap2048/text uni0/PT floor5e-4定义登记T23；完整PT400/LP100日志与T22的学习率逐轮一致，text uni全程0，总loss组合与1/.1权重一致。具体模型/入口/seed身份仍缺Namespace独立核验。完整统计、原始附件SHA及解释见[Stage1结果](D:/program/MacDiff/STAGE1_EXPERIMENT_RESULTS.md)与[no-share分析](D:/program/MacDiff/tools/vlm_pilot/NOSHARE_NORMB_LOG_ANALYSIS.md)。

当前配对已完成，不再等待、重启或重复运行本页旧命令。以下段落保留当时计划/身份，其“待回传”状态已被本节替代。以后任何新运行命令遵守用户10月15日约定：只用仓库根主入口/主配置，不用handoff_artifacts中的入口。随后已选5/5/3新组，见顶部当前计划；不跑seed1。decoder主体蒸馏是独立分支，其运行状态不能从本次no-share日志推断。


## 2026-10-15 结果：配对原版3层已完成T22，仅新no-share待回传

用户提供原版新3层完整PT400/LP100：T22 best85.462154313%@90、last85.334788956、末20均值85.391800113±.045036580。相对T20原版5层/min_lr1e-5，best−.539786500pp、末20−.458515284pp，100轮LP都低于T20对应轮；PT末20loss=.016801853122，比T20高15.290570%。LP逐epoch LR与T20完全相同，PT末20floor5e-4由日志确认。完整统计、原始附件和SHA见[结果表](D:/program/MacDiff/STAGE1_EXPERIMENT_RESULTS.md:9)。

此次同时改变深度与PT floor，不能单独确认3层或5e-4造成下降；用户PT5e-4/LP0偏好保留，不自动改回。若以后隔离深度，可固定5e-4补5层，尚未选定/执行。当前只待下文no-share全3层/decoder512/remap2048/text uni0结果，应先与T22配对评估文本收益。原版命令保留身份，不重复启动；no-share的服务器运行进度未知，不推断未开跑或替用户重启。

## 2026-10-14 当前选择：原版/no-share 全decoder3配对，文本decoder512/remap2048，关闭text uniformity，PT min_lr统一5e-4

用户已明确确认：以后仅PT的min_lr统一5e-4，LP保持0。下一组为原版与no-share各一条PT&&LP命令；均双卡、seed0、A、one_person=True、PT不旋转、PT400/每卡32/accum1/lr.001/min_lr.0005/warmup20/min_lr_epochs20；LP100/每卡64/accum1/lr.1/min_lr0/warmup0/min_lr_epochs10，checkpoint399。PT有效batch64、LP128。

原版PT用隔离官方入口和既有[p1 decoder3配置](D:/program/MacDiff/handoff_artifacts/macdiff_reproduction_20261008/configs/p1_decoder3_pretrain.yaml)，该历史候选YAML中的min_lr1e-5由CLI显式覆盖为5e-4，不改官方副本。no-share用已更新的[decoder3配置](D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_text_sentence_sample_target_blend_noshare_decoder3.yaml)：decoder_depth=3控制native和T→S，text_decoder_depth=3控制S→T；用户了解MLP为4H后，将text_decoder_hidden_dim改回512，text_hidden_dim保持2048，text_decoder_output_norm保持none；权重1/.1、七句v3/sample_target_blend。两组LP均使用T20官方隔离入口及A配置。

此前256→512改的是text_decoder_hidden_dim，控制S→T decoder hidden、自注意力/骨架cross-attention/MLP及输入输出投影；当前保持512，内部MLP宽度为4×512=2048，输入/输出为512→512→512，骨架memory为256→512。text_hidden_dim单独控制remap的512→H→512中间层，当前2048；remap为512→2048→512，这个中间层不再乘4。text_input_dim仍为缓存每句512维，dim_feat仍为骨架encoder/native/T→S的256维；不重生成缓存/改变目标向量维度。LP不使用文本decoder。此前同日两个hidden均2048只是未执行的中间计划，以本段最新选择为准。

用户随后明确要求去掉text uniformity：当前no-share的lambda_text_uniformity=0.0；模型在权重为0时跳过masked_text_uniformity_loss计算，日志loss_text_uniformity=0表示关闭，不表示句向量达到正交。骨架lambda_loss_uni=.02、T→S=1/S→T=.1保留，总loss为native+.02*skel_uni+T→S+.1*S→T。remap仍通过T→S训练，sample_target_blend更新公式不变；不添加dropout或跨样本正则。

随后按用户要求移除text_energy、text_batch_variance、text_valid_tokens的计算与返回指标，控制台和epoch JSON不再输出这三项；分支loss、LR及训练状态日志保留。同步更新后的模型文件即可，命令无需为这次日志调整再次改动；历史日志与分析保持原身份。

本次no-share输出根改为包含h512_remap2048_textuni0的新目录，区别于此前同日开启text uniformity的计划；不复用旧PT输出/目标库、不自动resume。原版命令保持原身份，10月15日已回传T22。更新既有配置、模型零权重分支与记录时没有启动训练；no-share需同步模型文件和YAML到服务器，是否已同步/开跑未知。

从服务器仓库根执行；no-share先同步本次更新的model/transformer_macdiff_text.py和YAML，并确保命令引用的既有隔离入口存在。输出目录pt/lp分离，新的目标库目录不复用。助手未启动训练；原版已登记T22，no-share新配置无回传结果。

原版（已完成T22，保留核对身份，不重复启动）：

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10277 handoff_artifacts/upstream_692888f/main_pretrain.py --config handoff_artifacts/macdiff_reproduction_20261008/configs/p1_decoder3_pretrain.yaml --batch_size 32 --accum_iter 1 --epochs 400 --warmup_epochs 20 --min_lr_epochs 20 --lr 0.001 --min_lr 0.0005 --mask_ratio 0.9 --motion_aware_tau -1 --seed 0 --output_dir output_dir/ntu60_xsub_native_dec3_ep400_minlr5e4_pt32_lp64_seed0_20261014/pt --log_dir output_dir/ntu60_xsub_native_dec3_ep400_minlr5e4_pt32_lp64_seed0_20261014/pt/tensorboard && OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10278 handoff_artifacts/upstream_692888f/main_linprobe.py --config handoff_artifacts/upstream_692888f/config/ntu60_xsub_joint/linprobe_madiff.yaml --finetune output_dir/ntu60_xsub_native_dec3_ep400_minlr5e4_pt32_lp64_seed0_20261014/pt/checkpoint-399.pth --batch_size 64 --accum_iter 1 --epochs 100 --warmup_epochs 0 --min_lr_epochs 10 --lr 0.1 --min_lr 0 --seed 0 --dist_eval --output_dir output_dir/ntu60_xsub_native_dec3_ep400_minlr5e4_pt32_lp64_seed0_20261014/lp --log_dir output_dir/ntu60_xsub_native_dec3_ep400_minlr5e4_pt32_lp64_seed0_20261014/lp/tensorboard
```

no-share：

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10279 main_pretrain.py --config config/ntu60_xsub_joint/pretrain_madiff_text_sentence_sample_target_blend_noshare_decoder3.yaml --lambda_text_to_skeleton 1 --lambda_skeleton_to_text 0.1 --batch_size 32 --accum_iter 1 --epochs 400 --warmup_epochs 20 --min_lr_epochs 20 --lr 0.001 --min_lr 0.0005 --mask_ratio 0.9 --motion_aware_tau -1 --seed 0 --output_dir output_dir/ntu60_xsub_noshare_all_dec3_h512_remap2048_textuni0_ep400_minlr5e4_pt32_lp64_seed0_20261014/pt --log_dir output_dir/ntu60_xsub_noshare_all_dec3_h512_remap2048_textuni0_ep400_minlr5e4_pt32_lp64_seed0_20261014/pt/tensorboard && OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10280 handoff_artifacts/upstream_692888f/main_linprobe.py --config handoff_artifacts/upstream_692888f/config/ntu60_xsub_joint/linprobe_madiff.yaml --finetune output_dir/ntu60_xsub_noshare_all_dec3_h512_remap2048_textuni0_ep400_minlr5e4_pt32_lp64_seed0_20261014/pt/checkpoint-399.pth --batch_size 64 --accum_iter 1 --epochs 100 --warmup_epochs 0 --min_lr_epochs 10 --lr 0.1 --min_lr 0 --seed 0 --dist_eval --output_dir output_dir/ntu60_xsub_noshare_all_dec3_h512_remap2048_textuni0_ep400_minlr5e4_pt32_lp64_seed0_20261014/lp --log_dir output_dir/ntu60_xsub_noshare_all_dec3_h512_remap2048_textuni0_ep400_minlr5e4_pt32_lp64_seed0_20261014/lp/tensorboard
```

10月13日5层/min_lr1e-5的旧no-share已于10月14日回传完整PT400/LP100，登记T21；运行身份与命令保留如下，不再待结果或重复启动。新3层原版/no-share两组共享PT floor，比较它们的文本辅助差异；与T20相比还改变了decoder深度和PT floor，与T21相比还扩大remap、关闭text uniformity，不能只归因一个因素。

## 2026-10-14 已完成：T21，旧5层no-share双卡公共配置

用户确认native/T→S/S→T各5层、text decoder512/none、remap512、PT32/accum1、LP64/accum1、seed0、A、权重1/.1、checkpoint399、PT min_lr1e-5。新日志不属于顶部3层/5e-4/remap2048计划。LP best86.220281456%@88、last86.147501239、末20均值86.134461457±.052393987；相对T20 best+.218340643pp、末20+.284146061pp，相对T12 best仅+.024260pp、末20+.133734pp。PT/LP逐epoch LR与T20完全相同，公共配置更匹配；源码/DDP/额外任务/RNG仍有差异，单次结果不证明跨seed显著。

S→T最低.009536486467@116→末轮.029400900783（3.083倍），回升比T12更大，但LP后期更好；不能据此直接确认过拟合或增加dropout。原始附件、SHA与详细统计更新于[既有no-share分析](D:/program/MacDiff/tools/vlm_pilot/NOSHARE_NORMB_LOG_ANALYSIS.md)及结果表；本节为当时21条记录的历史状态。10月15日新原版3层已回传T22，仅新no-share待结果。没有新建归档目录/复制日志/启动训练。

## 2026-10-14 此前MAMP对齐讨论（历史分析，已被顶部新选择替代）

MAMP官方NTU60脚本为decoder3、min_lr5e-4、PT400、PT总128/LP总256、PT无AMP、motion-aware mask；LP为256维global pooling、无head dropout、full cosine。MacDiff为diffusion AdaLN decoder、默认5、min_lr1e-5、随机mask、AMP；LP保留关节6400维/dropout.3且末10轮lr0。MAMP论文NTU60的3/5层同为84.9，正文总体默认5；MacDiff3层最佳86.4、5层85.9。

当前noshare仍按10月13日已选T20 profile：5层、min_lr1e-5、PT400/总64、LP100/总128、seed0、S→T5/512/none，暂无回传结果。MAMP分析不自动改配置或重启训练，不跑seed1。统一LP head协议、骨架3层配对、单独min_lr5e-4均为候选，未获选定。完整来源和分析写入[既有实现审计](D:/program/MacDiff/handoff_artifacts/native_implementation_audit_20261011/README.md)，没有新增artifact文件。

## 2026-10-13 历史选择：noshare对齐官方双卡配置，已完成T21

用户明确选择noshare，保持T20公共超参数：PT400/每卡32/accum1、LP100/每卡64/accum1、seed0、骨架decoder5、A、PT不旋转、lr1e-3/min_lr1e-5。文本沿用已完成T12定义：T→S1/S→T0.1、native/T→S5层、S→T5/512/none、七句v3/sample_target_blend/no-share。相对T12，主要将PT accum2→1使有效PT128→64；LP仍有效128。PT使用当前text入口，LP使用T20隔离官方入口及配置。

现有noshare YAML直接用CLI覆盖权重与batch，不新增配置或artifact文件。新输出目录的pt/lp分离，LP载入checkpoint399。用户要求的一条命令：

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10275 main_pretrain.py --config config/ntu60_xsub_joint/pretrain_madiff_text_sentence_sample_target_blend_noshare.yaml --lambda_text_to_skeleton 1 --lambda_skeleton_to_text 0.1 --batch_size 32 --accum_iter 1 --epochs 400 --lr 0.001 --min_lr 0.00001 --mask_ratio 0.9 --seed 0 --output_dir output_dir/ntu60_xsub_noshare_dec5_ep400_pt32_lp64_seed0_20261013/pt --log_dir output_dir/ntu60_xsub_noshare_dec5_ep400_pt32_lp64_seed0_20261013/pt/tensorboard && OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10276 handoff_artifacts/upstream_692888f/main_linprobe.py --config handoff_artifacts/upstream_692888f/config/ntu60_xsub_joint/linprobe_madiff.yaml --finetune output_dir/ntu60_xsub_noshare_dec5_ep400_pt32_lp64_seed0_20261013/pt/checkpoint-399.pth --batch_size 64 --accum_iter 1 --epochs 100 --lr 0.1 --min_lr 0 --seed 0 --dist_eval --output_dir output_dir/ntu60_xsub_noshare_dec5_ep400_pt32_lp64_seed0_20261013/lp --log_dir output_dir/ntu60_xsub_noshare_dec5_ep400_pt32_lp64_seed0_20261013/lp/tensorboard
```

命令当时已静态核对；10月14日用户已回传完整结果并确认身份，登记T21，不再待结果。PTseed1候选已由用户取消。

## 2026-10-13 T20结果：官方默认T20已完成，best86.001941%

T20一份PT400及两次LP100完整日志已保存。两LP都seed0、目录lp/lp1，用户确认参数相同，附件逐字节/逐字段一致：best86.001940813%@95、last85.783600214、末20均值85.850315396±0.072069144。当前20条配置/成绩记录，第二次重复不计T21。PT末20 lr1e-5已日志核验，官方默认组不再待结果。见[分析归档](D:/program/MacDiff/handoff_artifacts/official_dualgpu_results_20261013/README.md)。

这显示同seed LP重跑精确复现，不估计跨seed方差。PTseed1建议已被用户明确取消，当前选择见顶部noshare组；文本T15/T16保留，cache不改。

## 2026-10-12 T20之前：T19同旋转encoder的LP64已回传85.89%

T19仅复用T18 checkpoint499，LP每卡batch128→64，其余继承保存配置，口述LP85.89%，约高0.10pp。完整日志、best/末轮口径、epoch与actual args/seed未核验。这是不同LP batch，不是同配置重复；不能据此称seed随机性很大。当前19条数值记录，T19不新增PT。隔离官方双卡默认PT&&LP随后于10月13日回传T20完整日志；没有新选定的文本下一组，不改cache。

## 2026-10-12 T19之前：原版旋转对照已完成

T17不旋转best85.638039799%@64；T18旋转best85.789665239%@88；best+0.151625439pp、末20均值+0.126455600pp。用户已回传summary，原生decoder3/PT500旋转对照不再待结果，不能重复安排同两组训练。详见[结果归档](D:/program/MacDiff/handoff_artifacts/native_rotation_results_20261012/README.md)。

建议下一步复用T18 checkpoint499，仅LP每卡128→64、accum1不变，原configs/lp.json继续控制seed/LR/增强等；用户随后已要求此LP命令和新的隔离官方默认PT&&LP命令，LP64随后已回传T19/85.89%，新官方PT&&LP暂无结果。此测试用于检查LP组织贡献，不是纯BN消融。T15/T16文本85.84/85.86仍保留；没有新选定的文本下一组，本次未改文本配置或cache。

## 2026-10-11 状态记录：text decoder3已回传85.86%；旋转另测原版MacDiff

用户回传当前no-share骨架decoder3组的LP成绩 **85.84%（口述，T15）**；完整日志、best/末轮口径、对应epoch、末20统计及最终实际args尚未提供。native/T→S各3层、S→T仍5层/512/none、权重1/.1、A、PT不旋转的身份按上一会话上下文登记，最终实际batch/accum不从YAML或失败启动命令补填。

**2026-10-11已回传该组：T16，LP85.86%（用户口述）。**骨架decoder_depth保持3，S→T text_decoder_depth从5改为3；相对T15口述85.84%高0.02个百分点，尚不能确认深度收益。其余参数原计划继承T15，实际args/seed、完整日志及统计口径未核验。该组不再列为待结果，尚无新选定的文本下一组；本次仅登记，没有修改配置/模型或启动训练。

**随机旋转改为在原版MacDiff单独对照，不叠加文本实验。**用户最新要求已完成脚本：原版两组均decoder3/PT500、lr1e-3/min_lr1e-5、A、双卡PT64/accum1；先不旋转PT+checkpoint499 LP，再旋转PT+checkpoint499 LP；LP100/lr.1/双卡128/accum1一致。唯一组间配置差异为PT random_rot；完成后自动输出两组best LP acc及百分点差。脚本与说明见 [原版旋转对照](D:/program/MacDiff/tools/MACDIFF_ROTATION_ABLATION.md)。该准备阶段状态已被10月12日T17/T18结果替代；旧文本旋转控制计划保留为历史，不再是当前选择。

此前current.shared.json导致的FileExistsError是10月9日历史尝试。现在已报告decoder3成绩，不再把它当当前未成功训练的证据；修复方法、最终目录与实际运行参数仍未核验。成绩记录见 [Stage1汇总](D:/program/MacDiff/STAGE1_EXPERIMENT_RESULTS.md) 和 [handoff](D:/program/MacDiff/handoff.md:5)。

## 2026-10-09 历史计划：decoder3之后独立测试PT随机旋转（顺序已被上文替代）

用户明确要求：“把随机旋转的事情记录一下，下次实验试试”。**先完成当前no-share骨架decoder3、PT不旋转的实验，再以该组为控制，仅开启PT随机旋转。**目前只登记待实验，不修改当前训练YAML，不启动新训练；不能将此计划当成已完成结果。

### 已确认的论文/代码差异

- [论文第9页](https://lehongwu.github.io/ECCV24MacDiff/macdiff-paper.pdf#page=9)写明使用random crop、random rotation与小Gaussian noise（sigma=.005）。公开NTU60 XSub基础PT YAML的random_rot被注释；Feeder构造默认False。当前T12/no-share decoder3 PT同样未开启。
- 基础LP训练random_rot=True、测试False。LP冻结encoder，因此LP阶段旋转不等同于PT阶段用旋转训练encoder。
- [实现](D:/program/MacDiff/feeder/tools.py:236)绕XYZ各取uniform(-.3,.3)弧度（每轴约±17.2度）；同一片段所有帧/关节/人物共享矩阵。旋转在模型A标准化之前执行。
- [Feeder](D:/program/MacDiff/feeder/feeder_ntu.py:222)先crop/resize，再公共random_rot，然后复制encoder输入并加小噪声；native干净骨架和encoder输入共享同次旋转。source_rot只额外旋转encoder输入，是另一个因素，保留False。flip也保留False。

### 下一组的单因素约定

在独立的新PT配置的train_feeder_args中只增加random_rot: True。no-share、native/T→S各3层、S→T5层/512/none、权重1/.1、A、PT400、lr1e-3/min_lr1e-5、cache v3、sample_target_blend更新比.1、uniformity、seed与父控制相同。LP维持父控制的checkpoint399、100轮、lr.1、局部batch/累积/卡数、训练旋转True与测试False。使用新的PT/LP输出目录，不能复用已有目标库目录。

**以实际完成的无旋转控制命令/args为准固定batch组织。**此前保存的decoder3命令是PT每卡32/accum2；最新报错里的用户尝试命令是每卡64/accum1，双卡有效batch均128，但不能自动视为完全相同。该次尝试因已有current.shared.json在目标库prepare阶段退出，尚无成功训练结果；旋转实验须匹配最终成功控制的实际microbatch/accum，而不能再叠加修改。

文本cache先保持现有版本。旋转保留左右关节身份；若caption含相机/世界方向描述，固定文本可能与旋转后的骨架有偏差，这是待验证的可能性，不自动重生成cache。几何增强可能改善泛化，但未证实能补到论文86.4；比较LP best、last、末20均值及主要PT loss，而不只比较去噪loss。

### 2026-10-09启动失败记录（历史，当前已报告T15成绩）

最近报错为FileExistsError: Refusing to reuse target bank，指向output_dir/ntu60_xsub_macdiff_sentence_sampletarget01_noshare_st512_t2s1_s2t01_dec3_pt400/text_target_bank/current.shared.json。未传resume时，prepare拒绝覆盖已存在描述文件；不是随机旋转造成的报错。后续从头训练优先换未使用的输出目录，保留旧文件；若要续跑，需要同次checkpoint及配对的checkpoint-X-target-bank.sqlite，current.shared.json本身不能当完整恢复快照。本会话未删除文件、未修改恢复保护，未确认用户是否已修复启动。

## 2026-10-07历史选择：No-share 的 S→T 0.1 → 0.5（最新选择见上文）

用户最新决定继续文本版no-share，并明确选择S→T=0.5，替代助手建议的0.2。保持T→S=1、归一化A、512/无末端输出LN、sample_target_blend及目标更新比例0.1不变。已完成父对照LP best86.196021%、末20均值86.000728%；shared512口述best85.79%。原版B本次best85.183164%，不叠加到文本版。完整分析见 [NOSHARE_NORMB_LOG_ANALYSIS.md](D:/program/MacDiff/tools/vlm_pilot/NOSHARE_NORMB_LOG_ANALYSIS.md)。

沿用配置：[pretrain_madiff_text_sentence_sample_target_blend_noshare_st02.yaml](D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_text_sentence_sample_target_blend_noshare_st02.yaml)。文件名和YAML默认权重仍为0.2，由CLI显式覆盖为0.5；无需修改该文件。从头PT400、LP100；双卡PT每卡32/accum2，LP每卡64/accum1，有效batch均128，seed0，LP lr0.1/dist_eval。复用现有cache，输出与日志目录全部使用_s2t05；助手未启动训练，用户尚未提供进度或结果。参数更改不能当作原0.1实验的完整resume。

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10260 main_pretrain.py --config config/ntu60_xsub_joint/pretrain_madiff_text_sentence_sample_target_blend_noshare_st02.yaml --lambda_text_to_skeleton 1 --lambda_skeleton_to_text 0.5 --batch_size 32 --accum_iter 2 --epochs 400 --seed 0 --output_dir output_dir/ntu60_xsub_macdiff_sentence_sampletarget01_noshare_st512_t2s1_s2t05 --log_dir output_dir/ntu60_xsub_macdiff_sentence_sampletarget01_noshare_st512_t2s1_s2t05/tensorboard && OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10261 main_linprobe.py --config config/ntu60_xsub_joint/linprobe_madiff.yaml --finetune output_dir/ntu60_xsub_macdiff_sentence_sampletarget01_noshare_st512_t2s1_s2t05/checkpoint-399.pth --output_dir output_dir/ntu60_xsub_macdiff_sentence_sampletarget01_noshare_st512_t2s1_s2t05_lp_399_bs64 --log_dir output_dir/ntu60_xsub_macdiff_sentence_sampletarget01_noshare_st512_t2s1_s2t05_lp_399_bs64/tensorboard --batch_size 64 --accum_iter 1 --epochs 100 --lr 0.1 --seed 0 --dist_eval
```

判断先看LP best、末20轮均值与末轮，再看S→T后期曲线、native/T→S是否变化。更低文本loss不能单独证明表征改善；单次训练差异仍需后续确认。

## 以下保留2026-10-04历史规划，当前状态以上述选择为准

用户已授权将骨架输入归一化 A/B 加入待测试变量。本文件记录候选方案与建议顺序；未切换训练配置、未实现新模型结构、未启动服务器训练。用户现已提供512修复后的400epoch预训练日志，并口述LP best=85.79%；尚无本轮完整LP日志。日志分析见 [ST512_TRAINING_LOG_ANALYSIS.md](D:/program/MacDiff/tools/vlm_pilot/ST512_TRAINING_LOG_ANALYSIS.md)。

会话结束时用户决定：checkpoint-150的LP首轮仅72多，低于399的首轮74.65，不再继续该LP；继续S→T=1实验，随后根据结果测试归一化B。150的最终LP未完成，不能把首轮当最终分数。以下路线已按这一最新选择更新。

## 当时的共享对照组

| 项目 | 固定设置 |
|---|---|
| 数据/文本 | NTU60 XSub，当前 global + 六部位 v3 cache，person0 |
| 文本干净目标 | sample_target_blend，RMS，逐样本0.9旧目标+0.1当前remap |
| S→T decoder | hidden512，output_norm=none，depth5 |
| 骨架 decoder | share=True；共享原生重建与T→S主体 |
| 文本方向权重 | T→S=1，S→T=0.1；CLI显式覆盖，sentence YAML的T→S默认仍为0.1 |
| 归一化 | A：mean=[-0.0058,-0.1333,-0.0246]，var=[0.0206,0.0805,0.0218] |
| 扩散 | 各分支epsilon；现有inverse_cosine/1000步及均匀时间步不变 |
| 其他 | crop=[0.5,1]，mask=0.9，现有uniformity，不加跨样本目标 |
| 训练 | 双卡，每卡batch32/accum2，400epoch，seed0 |
| LP | 双卡，每卡batch64/accum1，100epoch，lr0.1，seed0，dist_eval；checkpoint-399 |

旧256组T→S=0.1/1的LP best分别为85.8018%/85.8928%，只差0.091个百分点；两者S→T均0.1。它们不能替代512对照组，也不能证明哪种设置稳定更好。

E0最新结果：400条epoch0..399连续且有限，LR与旧256/T→S=1一致；loss核算确认T→S=1/S→T=0.1。S→T最低0.00996099在epoch142，末轮0.02788974，远低于旧0.5结构限制，但后期回升到最低的2.80倍。native/T→S最后50轮均值0.01435490/0.01989607，与旧256的0.01439415/0.01991729几乎相同。用户报告LP best85.79%，相对旧256同权重85.8928%低约0.103个百分点，未显示收益；单次且缺LP完整日志，不能判定显著退化。

## 待测试变量

| 编号 | 单变量对照 | 要回答的问题 | 当前实现状态 |
|---|---|---|---|
| E0 | 当前512对照组 | 解除约0.502输出子空间下限后，是否实际改善encoder的LP？ | 已完成用户提供的PT日志分析；LP best85.79%，完整LP日志未提供 |
| E1 | S→T权重0.1 → 1（用户当前继续） | 512修复后，较强的S→T监督是否改善LP？ | 已提供命令；尚无本轮日志/结果，替代此前建议0.3，不做多档权重扫描 |
| E2 | share=True → False | T→S=1时，两种条件共同更新骨架decoder是否妨碍原生任务与表征学习？ | 模型支持；需要独立no-share YAML和输出目录 |
| E3 | 骨架归一化A → B（用户下一项） | 改变骨架信号尺度和有效扩散SNR，是否改善LP？ | 模型支持；需要成对的PT/LP YAML，父对照权重待E1结果决定 |
| E4 | 仅S→T epsilon → v | 调整不同时间步对干净文本恢复的监督权重，是否提高文本任务的表征收益？ | 未实现；需要S→T专用参数，不能把全局diff_prediction直接改为v |
| E5 | S→T无区域偏置 → 六部位软区域偏置 | 让局部文本优先读取对应关节区域，是否使六部位监督更有用？ | 未实现；需要保留随机mask后的原始关节身份、全局key与空区域回退 |

B为pretrain/LP中注释的#new组：mean=[-0.0024,-0.2132,-0.0446]，var=[0.0525,0.1527,0.0513]。它不同于NTU60 XSub finetune的另一组#new（统计报告中的C）。完整8组及全部50处位置见 [INPUT_NORMALIZATION_AUDIT.md](D:/program/MacDiff/tools/vlm_pilot/INPUT_NORMALIZATION_AUDIT.md)。

A→B使同一骨架的XYZ变化幅度成为原来的约0.626/0.726/0.652；固定t下以中心化信号方差衡量的有效SNR成为约0.392/0.527/0.425。它是训练超参数候选，尚不能确证是当前训练数据的正确统计。预训练与LP必须使用同组，从头预训练，独立目录，复用现有文本cache。

## 建议顺序与条件

1. E0已表现出输出瓶颈解除、去噪显著改善但LP未改善。用户已尝试150 checkpoint的LP并决定不再继续；仅首轮72多，不能据此证明它最终比399更差，不再安排继续150或扫描checkpoint。
2. 当前承接E1：保持512/share=True/A/T→S=1，仅把S→T从0.1改为1，替代此前建议0.3。这条分支直接向encoder传递文本监督。较低loss不证明0.1太小，512之前的S→T=1实验也不能作为相同结构的结论。该轮不同时改T→S、不做多档扫描，从头预训练；命令已提供，尚无结果，不重复启动同目录。
3. **随后根据E1结果测试E3归一化B，先于no-share。** 如果权重1有确认的收益，以该组为父对照；如果未改善，可保留原E0的S→T=0.1。当前尚未锁定B的权重。创建独立、成对的PT/LP YAML，只改骨架mean/var，从头训练，复用文本cache；不改正在跑的基线。它可能使epsilon更容易预测而降低loss，不能把这种下降当作文本语义增强。若它有效，再考虑拆分mean/var或噪声日程影响，第一轮不拆成多组。
4. E2 no-share保留为B之后的备选，届时明确父对照。历史share/no-share几乎持平的实验使用旧文本/旧目标机制、T→S=0.1，不能排除当前设置下的任务干扰。no-share会增加参数，同时保留T→S→remap→逐样本目标→S→T→encoder的间接路径，因此收益只能支持这一设计整体，不能单独证明梯度冲突。
5. 如果现有开关的简单对照没有明确LP改善，再讨论是否实现E4，改变S→T任务本身。v=a_t*epsilon-sigma_t*text_target；骨架原生/T→S保持epsilon，目标bank和干净文本来源不变。初轮沿用明确父对照的S→T外层权重，但epsilon与v的raw loss不可直接比较，其梯度与有效监督权重也不等价。该方案不增加信息，效果是待验证假设。
6. E5保留为后续备选，初轮仅修改S→T读出，T→S维持原样。encoder已经混合全身信息，因此区域bias只是读取先验，不是严格局部特征隔离；错误部位描述及静态模板也可能被强化。

当前最小路线是 **E1权重1结果 → E3归一化B**。E2/E4/E5只是备选，不要求将E0–E5全部跑完，也不沿用旧的“150 LP → 0.3 → no-share”优先级。

## 对照与判定

- 首轮每次只变一个因素，明确父对照；各候选使用独立PT/LP输出目录，不覆盖旧checkpoint/目标bank。
- E1/E2/E3首轮都可各自与E0比较；确认收益后建立新的对照组，再测下一因素。不要把所有修改一次叠加，否则无法判断收益来源。
- 相同epoch、LP协议及batch/累积设置；记录实际Git SHA与model_args。不同share结构或归一化实验从头训练，不称为旧实验完整resume。
- 比较LP best、最后10轮均值和最后结果，并保留分支原始loss解释任务变化。较低noise MSE、较高variance/energy不能代替LP。
- 如果只提高约0.05–0.1个百分点，先视为尚未确认的候选。必要时以额外LP seed检查head随机性；这不能替代额外预训练seed。确认有意义的候选后再做完整重复，不先铺开全组合。
- 已做的固定t/噪声条件打乱、方差/相似度诊断不再列为默认下一步；固定CLIP干净目标已试，不作为新方向；不加入对比学习、batch关系蒸馏或其他跨样本内容。

## 用户最新选择与命令

LP best已更正为85.79%。用户不再继续checkpoint-150的LP，当前继续T→S=1/S→T=1、共享512、归一化A的实验；之后根据结果测试B。每卡预训练32/accum2，LP64/accum1；无cd，预训练与LP用&&连接。

以下是已经提供的S→T=1预训练与LP命令，供核对实际运行，不默认重复启动同目录：

~~~bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10254 main_pretrain.py --config config/ntu60_xsub_joint/pretrain_madiff_text_sentence_sample_target_blend_shared.yaml --lambda_text_to_skeleton 1 --lambda_skeleton_to_text 1 --batch_size 32 --accum_iter 2 --epochs 400 --seed 0 --output_dir output_dir/ntu60_xsub_macdiff_sentence_sampletarget01_shared_st512_t2s1_s2t1 --log_dir output_dir/ntu60_xsub_macdiff_sentence_sampletarget01_shared_st512_t2s1_s2t1/tensorboard && OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10255 main_linprobe.py --config config/ntu60_xsub_joint/linprobe_madiff.yaml --finetune output_dir/ntu60_xsub_macdiff_sentence_sampletarget01_shared_st512_t2s1_s2t1/checkpoint-399.pth --output_dir output_dir/ntu60_xsub_macdiff_sentence_sampletarget01_shared_st512_t2s1_s2t1_lp_399_bs64 --log_dir output_dir/ntu60_xsub_macdiff_sentence_sampletarget01_shared_st512_t2s1_s2t1_lp_399_bs64/tensorboard --batch_size 64 --accum_iter 1 --epochs 100 --lr 0.1 --seed 0 --dist_eval
~~~
