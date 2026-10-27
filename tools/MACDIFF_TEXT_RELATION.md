# 文本关系分布蒸馏骨架：保留T→S，替换S→T diffusion

**2026-10-27实现边界更正：**用户追问一级为何不按原论文实现。当前B是工程上为保持EMA参数同形而采用的“细patch embedding均值合并＋随机70%”近似，不能把它称为原论文独立大patch投影已实现。真正12帧独立投影尚未实现，其与4帧学生投影的初始化/更新规则仍需明确；不是EMA原则上不允许大patch。默认A保持原生MacDiff＋完整EMA key全局InfoNCE＋T→S＋全局关系KL，见下方v2定义。此更正不修改模型、默认配置或启动训练。

## 2026-10-27可选一级消融：大时间patch＋随机mask70%，默认仍完整key v2

用户澄清实验层级：主方案是完整骨架EMA key；一级额外消融只借用“大patch＋教师随机mask70%”；运动差分输入与Motion Compass掩码作为这项消融之上的二级消融，分别评估及可选组合。**默认入口/config仍是下方完整key v2，不自动切到本消融。**

| 层级 | 教师输入 | 时间尺度 | 教师mask | 状态 |
|---|---|---|---|---|
| 主方案A | 原始骨架 | 4帧/细token | 无，750 token | 已实现，默认 |
| 一级B | 原始骨架 | 3个相邻4帧embedding合成12帧粗token | 随机70%，75 token | 本次已实现，独立配置 |
| 二级B+M | 在B上改为运动差分 | 同B | 随机70% | 候选，未实现/未启动 |
| 二级B+C | 在B上只改mask优先级 | 同B | Motion Compass-aware | 候选，未实现/未启动 |
| 二级B+M+C | 在B上同时加入运动输入与MCAM | 同B | Motion Compass-aware | 候选，未实现/未启动 |

二级不默认混入一级，不能把一级结果归给运动输入或MCAM。用户没有要求本次实现这些二级项，后续需按论文公式复核其输入/方向强度/概率采样，再独立提供配置。学生侧、T→S、global KL及队列/LP都保持当前定义。

### 论文依据与本项目适配边界

已读取本地ACA²Net原文并渲染核对PDF第4页图2/教师式5–7及第6页IV-B：论文90帧、学生时间patch3/mask90%，教师时间patch9/mask70%，两路都是75个token，教师实际输入为运动差分且使用MCAM。原文第10页解释教师70%mask是效果与计算量的折中。原文：[本地PDF](/D:/program/paper/Asymmetric_Context-Guided_Adaptive_Alignment_Network_for_Skeleton-Based_Action_Recognition.pdf)。

当前MacDiff固定120帧、学生patch4，因此仿照3倍时间尺度可得：学生120/4×25×.1=75；教师120/12×25×.3=75。**75个教师token是更粗的时间token，不是75个原网格细token。**它们名义覆盖225个细patch、约为学生可见坐标位置的3倍，但少于完整教师750细patch；不能保证压缩后的信息质量一定更高。

独立的12帧Conv/Linear投影与学生4帧投影参数形状不同，不能直接套用当前EMA。为保留同架构EMA、只改变教师视图的受控适配，本版使用：
1. 用同形EMA的4帧patch投影得到[B,30,25,D]。
2. 加原时空位置embedding；同一关节的相邻3个时间patch连同位置embedding取均值，得到[B,10,25,D]。
3. 展平250个粗token，逐样本随机保留75个，再进入8层EMA Transformer和norm/GAP/L2。
4. 该粗key作为InfoNCE正样本并进入原共享骨架queue，与配对文本同步入队。

这等价于一种由细patch权重约束的12帧线性投影，不是ACA²Net独立训练的9帧MotionEmbed；平均可能损失细节，本消融不冒称原样复现。相邻组不跨关节，位置也按组平均，无新可训练权重、无需resize EMA参数。教师mask与学生mask独立，教师保留视图不保证包含学生全部可见位置。key attention从750降到75 token，attention矩阵元素约为原教师的1/100；4帧embedding仍计算完整网格，不能把该比例当整网加速或显存实测。

### 配置、恢复与验证

沿用同一模型/入口，新增[可选配置](/D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_text_relation_teacher_coarse70_decoder3.yaml)和[专用测试](/D:/program/MacDiff/tests/test_text_relation_teacher_coarse.py)。一级身份 `relation_teacher_version=coarse3_mean_random70_ema_gap_shared_queue_kl_v3`，明确绑定 `teacher_view=coarse_mask`、`teacher_temporal_group=3`、`teacher_mask_ratio=.7`；完整key v2仍默认full/group1/mask0，旧model_args不添加冗余新字段，保持旧完整key恢复兼容。跨full/coarse的resume会被完整model_args检查拒绝，不能混装两种历史key队列。

除此之外参数不变：A/person0、encoder8/native decoder3/T→S decoder3、query可见token GAP256、global KL/配对65536队列、CL1e-4/温度.07、EMA.999、KL.1/两侧温度.1、uni.02、T→S1/七句/remap2048/no S→T。PT400/总64/floor5e-4/noAMP/seed0→原第8层LP100/总128/floor0/6400维，读本组checkpoint399。

本地一级新增7项：5通过、2因缺Torch跳过；一级+v2+v1+累积37项为28通过/9跳过。通过含实际生产粗粒度forward的NumPy执行、250→75随机选择、同关节时间分组/平均位置、默认full保持与跨视图resume拒绝。真实75query/75key的反向、EMA投影形状/更新及checkpoint2项待服务器执行；未验证GPU/DDP。相关整合回归与静态配置检查见handoff，不由NumPy或mock声称autograd/NCCL通过。

同步更新 model/transformer_macdiff_text_relation_teacher.py、main_pretrain_text_relation_teacher.py、tests/test_text_relation_teacher.py，加上本次新配置/测试；此前v2的基类关系模型/入口/公共engine/main及缓存支持文件仍须为当前版本。服务器可先运行：

```bash
python -m unittest tests.test_text_relation_teacher_coarse tests.test_text_relation_teacher tests.test_text_relation tests.test_pretrain_accumulation_policy -v
```

以下为**可选B**的独立PT&&LP命令，端口10327/10328、新目录；默认主方案A仍使用下方v2配置及10325/10326，助手未启动任一训练、不新增成绩编号：

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10327 main_pretrain_text_relation_teacher.py --config config/ntu60_xsub_joint/pretrain_madiff_text_relation_teacher_coarse70_decoder3.yaml --teacher_view coarse_mask --teacher_temporal_group 3 --teacher_mask_ratio 0.7 --relation_bank_size 65536 --lambda_contrastive 0.0001 --ema_momentum 0.999 --contrast_temperature 0.07 --lambda_relation 0.1 --relation_temperature_text 0.1 --relation_temperature_skeleton 0.1 --lambda_text_to_skeleton 1 --lambda_skeleton_to_text 0 --disable_amp --batch_size 32 --accum_iter 1 --epochs 400 --warmup_epochs 20 --min_lr_epochs 20 --lr 0.001 --min_lr 0.0005 --mask_ratio 0.9 --motion_aware_tau -1 --seed 0 --output_dir output_dir/ntu60_xsub_textrel_coarse3mean_m70_q65536_cl1e4_kl01_gap256_dec3_noamp_seed0_20261027/pt --log_dir output_dir/ntu60_xsub_textrel_coarse3mean_m70_q65536_cl1e4_kl01_gap256_dec3_noamp_seed0_20261027/pt/tensorboard && OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10328 main_linprobe.py --config config/ntu60_xsub_joint/linprobe_madiff.yaml --finetune output_dir/ntu60_xsub_textrel_coarse3mean_m70_q65536_cl1e4_kl01_gap256_dec3_noamp_seed0_20261027/pt/checkpoint-399.pth --batch_size 64 --accum_iter 1 --epochs 100 --warmup_epochs 0 --min_lr_epochs 10 --lr 0.1 --min_lr 0 --seed 0 --dist_eval --output_dir output_dir/ntu60_xsub_textrel_coarse3mean_m70_q65536_cl1e4_kl01_gap256_dec3_noamp_seed0_20261027/lp --log_dir output_dir/ntu60_xsub_textrel_coarse3mean_m70_q65536_cl1e4_kl01_gap256_dec3_noamp_seed0_20261027/lp/tensorboard
```


## 2026-10-27最新v2：完整EMA key队列由InfoNCE和全局KL共用

用户进一步明确：“原生MacDiff接key encoder做teacher对比学习，就用它的完整骨架特征queue，文本侧建立对应queue；KL仍全局，query为每批mask后10%可见token的全局池化”。本版据此实现；不加入此前讨论的六部位local KL，也不另建一个骨架关系库。旧v1 masked-online入库配置/命令仍保留在后文，不能混同实验身份。

### 计算定义

原生encoder8/256在120帧、时间patch4、25关节下有750个token。query encoder只编码随机保留的75个增强视图token；key encoder编码同一crop内全部750个干净骨架token（不是整段原始视频，不加入diffusion噪声）。两侧均第8层norm后GAP：

```text
q_i = normalize(mean_visible E_query(mask(x_aug_i)))       # 256维，有梯度
k_i = normalize(mean_all E_key(x_clean_i))                 # 256维，no_grad/EMA/eval
y_i = normalize(sample_target_blend_global_text_i)         # 512维，detach

Q_S[j] = 历史完整骨架 k_j
Q_T[j] = 与同一sample_id配对的历史文本 y_j

P_S(i,j) = softmax_j(cos(q_i,Q_S[j]) / tau_S)
P_T(i,j) = softmax_j(cos(y_i,Q_T[j]) / tau_T)
L_KL = mean_i KL(stopgrad(P_T(i,:)) || P_S(i,:))

L_CL = InfoNCE(q_i, 当前完整k_i为正样本, 同一个Q_S作为历史负样本)
L = L_S2S + 1*L_T2S + 1e-4*L_CL + .1*L_KL + .02*Uni_skeleton
```

InfoNCE和KL实际访问同一个 `relation_skeleton_keys` buffer、同一IDs；没有额外 `contrast_queue`。两个损失均排除历史同sample_id的全部副本；当前完整k_i在InfoNCE中保留为正样本。KL两边的参照列一一对应，排除self后不足2个参照的行暂不提供KL，空队列的InfoNCE只有正样本，首步为0。

入库的是(k_i,y_i,sample_id)，**不是masked query q_i**。两个FIFO容量/IDs/指针/覆盖时序统一，当前默认65536，沿用之前AA0的骨架对比学习容量，文本队列扩到同容量。容量只由 `relation_bank_size` 设置，两个损失共同使用，不再另设contrast_queue_size。

完整key encoder是同架构EMA副本，只复制骨架encoder/patch嵌入/norm/位置参数；冻结且始终eval，无投影头、AA对齐或层混合适配器。仅在成功完整optimizer步之后提交本窗口已有的完整key/配对文本，并更新一次EMA：E_key←.999*E_key+.001*E_query。AMP跳步、空提交及不完整累积尾窗均不更新队列或EMA。调用顺序为提交该forward的旧EMA key，再更新EMA权重；不是优化后额外重算一次key。

原生main在DDP参数同步后、resume载入前调用initialize_pretraining_state，将fresh key复制到同步后的online encoder；完整resume随后恢复已保存EMA/队列/teacher_updates。当前main明确broadcast_buffers=False，队列通过成功步后的变长all_gather保持跨rank配对一致，不依赖buffer广播。旧v1公共engine的supports_relation_memory提交开关继续复用，本轮不再修改engine。

### 256维与6400维

用户仍保留维度方案讨论空间。按本次明确的“可见token全局平均”，query自然为256维，key也采用同样的GAP256，配置 `relation_feature_mode=gap`。6400维需要按25关节各自池化再拼接；90% mask下还需定义不可见关节的处理，属于另一种读出，本版不擅自加入。暂不提供6400训练模式，其他feature_mode会明确拒绝，不能只换队列维数或重复/补零冒充同一种特征。

LP保持已有第8层完整输入、时间/人物平均后25关节拼接6400维和原分类头；这与训练辅助分支GAP256的读出分开记录。预训练本身仍有局部token条件，不能称原生MacDiff只训练一个256维向量。MAMP官方LP使用GAP256、finetune用关节拼接+MLP，参见[官方LP配置](https://raw.githubusercontent.com/maoyunyao/MAMP/main/config/ntu60_xsub_joint/linprobe_t120_layer8.yaml)及[读出代码](https://github.com/maoyunyao/MAMP/blob/main/model/transformer.py)。

### 参数、状态与验证

新模型身份 `model.transformer_macdiff_text_relation_teacher.Transformer`，方法版本 `relation_teacher_version=full_ema_gap_shared_queue_global_kl_v2`；底层关系KL沿用v1实现。默认EMA=.999、CL温度.07/权重1e-4，KL两侧温度.1/权重.1。保留T→S权重1、独立原生结构decoder3、remap隐藏2048、七句v3缓存和逐样本目标值平滑.1；无S→T decoder，不接local KL、OTFM、多层diffusion或固定预训练teacher蒸馏。

两个FP32特征队列分别64 MiB（骨架65536×256）和128 MiB（文本65536×512），另有约.5 MiB IDs及指针；这是buffer静态大小，不是实测峰值显存，另有EMA encoder和完整key前向开销。无新增可训练投影/decoder，EMA参数冻结；online encoder/文本T→S分支仍正常训练。现有loss/熵/队列指标保留，另记loss_contrastive、loss_contrastive_weighted、contrast_positive_cosine；teacher_updates计数随checkpoint保存。

新模型/入口/配置与v1独立，完整resume拒绝v1或旧纯骨架teacher等方法冒充；K/EMA/CL/KL温度和权重/数据缓存/训练协议均绑定。保持原逐样本文本目标库的同checkpoint快照要求。旧控制组defaults与状态格式保持，公共基类仅新增两个缺省无效的key/辅助loss扩展方法。

本地v2新增12项：9通过、3因缺Torch跳过；v2+v1+累积共30项，23通过/7跳过。通过含生产forward NumPy适配执行（masked query、完整clean key输入、只前向一次query encoder、配对入库、CL/KL总loss）、完整key处理所有token、InfoNCE共享同一队列及self排除、成功步EMA时序、恢复隔离。3项新真实Torch检查覆盖5可见token vs50完整token的小模型、真实反向/EMA更新与配对入队、完整state/旧LP兼容，待服务器运行；NumPy不验证autograd，未验证实际GPU/NCCL/DDP。所有相关回归145项为74通过/71缺Torch跳过，teacher旧测试的导入路径需按原tests发现方式或整合runner处理。本地Python3.8语法与参数/配置/命令静态检查通过，不等于真实PyYAML或训练运行。

### 同步与运行

新文件：
- model/transformer_macdiff_text_relation_teacher.py
- main_pretrain_text_relation_teacher.py
- config/ntu60_xsub_joint/pretrain_madiff_text_relation_teacher_decoder3.yaml
- tests/test_text_relation_teacher.py

同时同步本轮修改的 model/transformer_macdiff_text_relation.py、main_pretrain_text_relation.py、tests/test_text_relation.py；沿用此前当前engine_pretrain.py、main_pretrain.py、util/misc.py和文本缓存/目标库支持文件。服务器先检查：

```bash
python -m unittest tests.test_text_relation_teacher tests.test_text_relation tests.test_pretrain_accumulation_policy -v
```

新输出目录、端口10325/10326；encoder8/native decoder3/T→S decoder3，PT400/两卡总64/floor5e-4/seed0/noAMP→LP100/总128/floor0/原第8层6400读出，从本组checkpoint399启动LP：

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10325 main_pretrain_text_relation_teacher.py --config config/ntu60_xsub_joint/pretrain_madiff_text_relation_teacher_decoder3.yaml --relation_bank_size 65536 --lambda_contrastive 0.0001 --ema_momentum 0.999 --contrast_temperature 0.07 --lambda_relation 0.1 --relation_temperature_text 0.1 --relation_temperature_skeleton 0.1 --lambda_text_to_skeleton 1 --lambda_skeleton_to_text 0 --disable_amp --batch_size 32 --accum_iter 1 --epochs 400 --warmup_epochs 20 --min_lr_epochs 20 --lr 0.001 --min_lr 0.0005 --mask_ratio 0.9 --motion_aware_tau -1 --seed 0 --output_dir output_dir/ntu60_xsub_textrel_fullkey_q65536_cl1e4_kl01_gap256_dec3_noamp_seed0_20261027/pt --log_dir output_dir/ntu60_xsub_textrel_fullkey_q65536_cl1e4_kl01_gap256_dec3_noamp_seed0_20261027/pt/tensorboard && OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10326 main_linprobe.py --config config/ntu60_xsub_joint/linprobe_madiff.yaml --finetune output_dir/ntu60_xsub_textrel_fullkey_q65536_cl1e4_kl01_gap256_dec3_noamp_seed0_20261027/pt/checkpoint-399.pth --batch_size 64 --accum_iter 1 --epochs 100 --warmup_epochs 0 --min_lr_epochs 10 --lr 0.1 --min_lr 0 --seed 0 --dist_eval --output_dir output_dir/ntu60_xsub_textrel_fullkey_q65536_cl1e4_kl01_gap256_dec3_noamp_seed0_20261027/lp --log_dir output_dir/ntu60_xsub_textrel_fullkey_q65536_cl1e4_kl01_gap256_dec3_noamp_seed0_20261027/lp/tensorboard
```

助手未启动训练，新组无成绩、不预分配T33。以下是旧v1的定义、检查与原命令，供独立对照追溯。


2026-10-27：用户提出“remap后的文本与骨架各建memory bank，KL让骨架模仿文本的相似结构”，并明确选择**保留T→S扩散训练remap，只替换S→T**。已提供独立模型、入口、配置和测试。旧S→T diffusion、OTFM采样对照、后三层diffusion是各自独立实验，本实现不混接。助手未启动训练，无新成绩或实验编号。

## 监督和梯度

- 保留原生S→S diffusion及skeleton uniformity=.02。
- 保留T→S diffusion、独立的原生结构decoder3及remap512→2048→512，权重1；沿用七句v3缓存（global+六local）、concat_memory、A/person0。
- 删除S→T TextNoiseDecoder的构造和训练；关系分支没有decoder、投影头或新可训练参数。
- 首轮只比较全局关系。骨架s为第8层norm后可见token的GAP（256维），文本y为现有sample_target_blend平滑global目标（512维），相似度前各自L2归一化。七句local继续参与T→S，六个local的关系KL未加入。

两个FIFO记忆库按相同slot和sample_id保存历史配对骨架/文本特征。对于当前样本i，j遍历两侧共同有效的同一批参照样本，并排除所有sample_id=i的列：

```text
P_T(i,j) = softmax_j[cos(stopgrad(y_i), stopgrad(K_T[j])) / tau_T]
P_S(i,j) = softmax_j[cos(s_i, stopgrad(K_S[j])) / tau_S]
L_relation = mean_i sum_j P_T(i,j) * (log P_T(i,j) - log P_S(i,j))
L_total = L_S2S + 1*L_T2S + 0.1*L_relation + 0.02*Uni_skeleton
```

KL方向是KL(P_T || P_S)，只有当前骨架query接收此损失梯度，直接回到第8层GAP和encoder；文本query、两个库都detach。remap仍由T→S训练，KL不反向推动文本目标迎合骨架。两边特征维度无需一样，因为对齐的是同一K个参照样本上的概率分布。没有温度平方补偿、额外对比loss或标签监督。

文本teacher沿用旧最佳dec3的逐样本目标值平滑定义：每次成功优化步后，以新remap输出对已访问样本更新 y_i←.9*y_i+.1*remap(CLIP_i)；首次访问未初始化目标时沿用既有固定CLIP fallback。关系query和入库文本均取本次forward开始时读取的global平滑目标，不是在线r的未平滑输出，也不是remap参数EMA。这样保留旧目标稳定方式；它与新加入的两个关系FIFO库是不同的状态。

## 记忆库和训练时序

方法身份 `paired_global_fifo_text_to_skeleton_kl_v1`。默认K=4096对、tau_T=tau_S=.1、lambda_relation=.1；这些是首轮起始设置，未调参，不声称与旧S→T噪声MSE的.1等效。

- 两个库共享一组IDs和环形指针，容量4096；这是最近历史配对条目的FIFO，**不是全训练集每ID一个槽的索引库**。两侧同时入库、同时覆盖。重复访问可以留下多个同ID条目，但query会在两侧排除该ID的全部副本。
- 条目存L2归一化的、已detach的当前forward特征；先backward/optimizer成功，再提交该累积窗口的条目。库不保留计算图，不增加encoder EMA或额外骨架前向。
- 跳过AMP优化步和未完成的累积尾窗均不入库。text sample target bank仍按既有成功优化步更新，两类状态都不在forward里提交。
- DDP在成功步后按rank顺序all_gather IDs/骨架/文本，支持各rank有效样本数不同；两侧共同更新，保证相同列含义。没有实际多卡验证，不把mock gather当作NCCL验证。
- 空库和排除self后不足两个参照的行暂不提供KL；没有epoch级预热，第一次成功步入库后即可开始。默认全局batch64时约64个成功步装满（有效person0较少时稍慢）。
- 两个特征库、IDs、指针均为持久buffer，跟随设备移动并保存在model checkpoint；256/512维、K4096的FP32特征约12 MiB，另有IDs。参数量不含这些buffer，训练无需新增OT或其他库。
- 当前原生main明确broadcast_buffers=False，配对库依靠成功步all_gather保持跨rank一致；当前实现不承诺降低通信耗时或实测显存，是否加速须服务器测量。

## 日志与判读

记录loss_relation、loss_relation_weighted、relation_bank_fill、relation_anchor_count、relation_valid_fraction、relation_teacher_entropy、relation_teacher_entropy_fraction、relation_student_entropy，前面由公共main加train_。

teacher_entropy_fraction是每行文本分布熵除以该行有效参照数的log，再取均值；接近1说明文本关系较平，较小的KL本身不能证明表示变好。冷启动无有效行时指标记0，必须结合valid_fraction/bank_fill看。目标是冻结encoder后的LP，KL数值和生成重建质量不能替代LP。

本方案属于跨模态关系蒸馏。可参考[SEED](https://arxiv.org/abs/2101.04731)的相似度分布蒸馏与[ReSSL](https://arxiv.org/abs/2107.09282)的关系分布学习；本实现用remapped text目标与配对骨架库，并非这两篇方法的完整复现，也不是OPD。

## 文件、恢复和验证

服务器同步：
- model/transformer_macdiff_text_relation.py
- main_pretrain_text_relation.py
- config/ntu60_xsub_joint/pretrain_madiff_text_relation_decoder3.yaml
- tests/test_text_relation.py
- **本次engine_pretrain.py**：新特性开关传入sample_indices、收集pending配对条目，并仅在成功完整优化步后提交。旧文本/FM/teacher路径开关缺省False。
- 此前已使用的main_pretrain.py、util/misc.py、文本模型/缓存/target bank支持文件保持现有版本。

完整resume需本方法checkpoint及既有逐样本文本目标库的同checkpoint快照；关系FIFO已经在model state_dict中。入口绑定模型身份、完整model_args（包括K/温度/权重）、数据/缓存路径、日程/精度等，公共main还核对文本缓存identity及目标库元数据。旧S→T/native/FM/teacher权重不能冒充完整续训。新PT目录若已有log/checkpoint会拒绝覆盖。未保存RNG、数据/服务器源码未hash，不保证逐位复现。

本地新测试13项：9通过、4因无PyTorch跳过；通过含生产forward的NumPy适配执行、KL方向、跨维度对应、参照同序、排除全部self/空槽、FIFO环绕/配对与成功步提交、入口/恢复/目录保护。4项真实Torch覆盖梯度隔离、无S→T完整前向/encoder与remap反向、队列checkpoint/LP兼容、mock变长gather，待服务器运行。NumPy不验证autograd，mock不验证真实DDP。既有文本/一致性/FM/梯度累积回归也已检查，详情见handoff。

服务器先执行：

```bash
python -m unittest tests.test_text_relation tests.test_pretrain_accumulation_policy -v
```

PT为encoder8/native decoder3/T→S decoder3，400轮、两卡每卡32、总64、accum1、lr.001/floor5e-4、warmup20/floor20、seed0、noAMP；LP100、总128、lr.1/floor0，只读第8层原LP特征与原分类头。历史T23为AMP，精度差异仍需区分；新组未有成绩。新端口10323/10324，新输出目录如下：

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10323 main_pretrain_text_relation.py --config config/ntu60_xsub_joint/pretrain_madiff_text_relation_decoder3.yaml --lambda_relation 0.1 --relation_bank_size 4096 --relation_temperature_text 0.1 --relation_temperature_skeleton 0.1 --lambda_text_to_skeleton 1 --lambda_skeleton_to_text 0 --disable_amp --batch_size 32 --accum_iter 1 --epochs 400 --warmup_epochs 20 --min_lr_epochs 20 --lr 0.001 --min_lr 0.0005 --mask_ratio 0.9 --motion_aware_tau -1 --seed 0 --output_dir output_dir/ntu60_xsub_textrel_kl01_k4096_t01_dec3_noamp_seed0_20261027/pt --log_dir output_dir/ntu60_xsub_textrel_kl01_k4096_t01_dec3_noamp_seed0_20261027/pt/tensorboard && OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10324 main_linprobe.py --config config/ntu60_xsub_joint/linprobe_madiff.yaml --finetune output_dir/ntu60_xsub_textrel_kl01_k4096_t01_dec3_noamp_seed0_20261027/pt/checkpoint-399.pth --batch_size 64 --accum_iter 1 --epochs 100 --warmup_epochs 0 --min_lr_epochs 10 --lr 0.1 --min_lr 0 --seed 0 --dist_eval --output_dir output_dir/ntu60_xsub_textrel_kl01_k4096_t01_dec3_noamp_seed0_20261027/lp --log_dir output_dir/ntu60_xsub_textrel_kl01_k4096_t01_dec3_noamp_seed0_20261027/lp/tensorboard
```

