# Stage1 文本实验待测变量与顺序（更新2026-10-18）

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

## 近期结果速查（T14–T24）

no-share深度按native/T→S/S→T排序；末20为LP epoch80–99均值。T15/T16/T19为口述，其余证据及不同协议详见结果表，不能作跨配置单因素排序。T21为remap512/text uni.02，T23为remap2048/text uni0。

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
