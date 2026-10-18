# 原版 decoder 主体蒸馏：随机 encoder + 随机输出 head

实现日期：2026-10-15。独立入口 `main_pretrain_decoder_distill.py`，不修改原版、文本版、OSE、官方隔离副本或其配置/训练入口。本地没有启动训练。

## 初始化与训练范围

单个模型继承 `model.transformer_macdiff.Transformer`，保持原有 state_dict 键名。

| 部分 | 初始化 | 训练状态 |
|---|---|---|
| encoder `joints_embed/blocks/norm/pos_embed/temp_embed` | 原版随机初始化 | 更新 |
| decoder `decoder_embed/decoder_blocks/decoder_norm/decoder_pos_embed/decoder_temp_embed` | 原版 PT checkpoint | 冻结，eval |
| head `decoder_pred`，默认 Linear(256,12)，3084参数 | 原版 Xavier 随机初始化，不加载教师 head | 更新 |

冻结主体包括 AdaLN 的 `fc_z/fc_t`、attention、MLP 和 LayerNorm。原 encoder 不参与教师前向，也不向学生加载权重。新 head 需要重新学习噪声读出，初始化时的组合不具有完整原生成器的输出能力。

源 checkpoint 默认使用已有 T20 原版 decoder5/checkpoint399，配置已填入历史启动命令记录的路径 `output_dir/native_official_dec5_ep400_bs32_seed0_20261012/pt/checkpoint-399.pth`，也可通过 `--decoder_checkpoint` 覆盖。路径依据 `handoff_artifacts/native_rotation_results_20261012/commands.json` 和 `clean_default_pt_lp.sh`；尚未直接访问服务器文件。所有主体键和形状必须严格匹配，多余或缺失的 decoder 层也报错。源 args 存在时核验 native 模型身份、epsilon 参数化、attention heads、激活函数、diff_steps、noise schedule、归一化和 self_shift。纯 state_dict 没有这些元数据，只能校验张量键/形状，启动日志明确标记 metadata=unavailable。只加载信任的本地训练 checkpoint。

## 完整数据流

1. Feeder 从 train split 读取 `[B,3,T,25,2]`。沿用 valid_crop_resize，裁剪区间 `[0.5,1]`，输出120帧；PT不旋转、不flip。干净 target 和增强 source 使用相同裁剪，只有 source 加 joint noise `[1,0.005]`。
2. 模型固定取 person0。两路变成 `[B,120,25,3]`，按 A 的 `(x-mean)/sqrt(var)` 归一化。target 是归一化后的干净骨架，不是加入增强噪声的副本；人物空槽处理继承原版。
3. source 经学生 patch embedding：4帧×1关节×3坐标=12维输入，映射256维，得到 `[B,30,25,256]`。加学生随机初始化、可训练的时空位置编码，展平750tokens。
4. 随机mask90%，保留75tokens。学生8层Transformer和最终LN输出 `[B,75,256]`。对75个局部特征求均值得 `[B,1,256]`。
5. 沿用 build_global_local_condition：可见位置填局部特征，675个被mask位置填全局均值特征；按 ids_restore 还原时空位置，得到条件 `[B,750,256]`。
6. 每个样本从300到700（包括两端）独立均匀采样 t；diffusion仍为1000步 inverse-cosine τ=1。对干净 target 重新采样标准高斯 epsilon，形成 `x_t=sqrt(alpha_bar_t)*x0+sqrt(1-alpha_bar_t)*epsilon`，形状 `[B,120,25,3]`。采样器返回的重要性权重仍按 native 前向惯例忽略；训练分布就是中等噪声区间，不补偿成0–999全区间目标。
7. `x_t` 经冻结的 decoder embedding、位置编码和5层条件Transformer；AdaLN在每层接收 z 和 t，输出经冻结的最终LN得到 `[B,750,256]`。condition dropout和layer dropout都显式为0。每轮 engine 调用 train(True) 后，冻结主体仍保持eval。
8. 新训练 head 映射256→12，输出噪声预测 `[B,750,12]`。沿用 noise2joint，把 `patchify(x_t)` 与预测噪声组合为 `x0_hat=(x_t-sqrt(1-alpha_bar_t)*epsilon_hat)/sqrt(alpha_bar_t)`。没有DDIM迭代，也没有采样裁剪；原前向的 x_t 与扩散系数为FP32，坐标还原/MSE在FP32中完成。
9. 沿用 native forward_loss：12个坐标取均值，只在被mask的675个位置计算重建MSE，逐样本平均后取batch均值。总loss为 `masked_coordinate_MSE + 0.02 * skeleton_token_uniformity`，uniformity仍来自学生75个可见token。
10. backward 更新学生 encoder 和随机 head。冻结 decoder 的参数不求梯度，但其计算保留对 z 的梯度，所以重建信号能经过整个冻结主体抵达学生。预处理、加噪和 target 位于原版 no_grad 段中；decoder前向不位于no_grad中。

新方法严格保留原生 masked-only loss；没有改成全750位置MSE。没有引入原 encoder 特征回归、JEPA或文本目标。

## 2026-10-16：Gong 等 AAAI-25 论文适用性检查

用户提供的 [Rethinking Masked Data Reconstruction Pretraining for Strong 3D Action Representation Learning](D:/DL/08371-GongT.pdf)，PDF第4页公式4/5及第7页表5/6，讨论了两个问题：短时坐标目标中的运动变化弱，以及重建任务缺少样本间判别约束。本次只做文献和代码检查，没有修改模型、配置或启动训练。

**不是同一种归一化实现，但相关风险没有被排除。**论文批评的是每个关节的短时间片把时间和x/y/z合并求一个mean/std；NAT则对每个坐标轴单独在该片段的时间维归一化。当前模型在加噪前使用固定的逐轴A统计量，`patchify`只重排，`forward_loss`直接坐标MSE，没有patch统计量归一化。因此没有论文公式4的同款处理，也没有NAT对细小时间变化的强化。固定逐轴仿射变换不抹去运动信息，但不会专门放大各patch的时间变化。4帧内的姿态均值与时间变化都贡献坐标误差；实际哪项主导需测量，不能仅凭目标形式认定已发生退化。

**扩散结构是重要差别。**令a=alpha_bar_t，则 `x0_hat-x0=-sqrt((1-a)/a)*(epsilon_hat-epsilon)`，同一mask下 `L_x0=((1-a)/a)*L_epsilon`，不含uniformity。因此本方法不是论文的普通MAE直接坐标回归；当前inverse-cosine的t=300/500/700对应a约.612368/.505715/.396131、权重约.633006/.977399/1.524417。时间步加权不等于NAT，但也不能忽略随机噪声目标和noisy输入而直接套用论文的退化结论。冻结decoder提供预训练先验，不保证当前encoder保留教师的动作判别能力。

**样本间判别目标仍缺失。**当前total是重建加.02骨架token uniformity；`token_uniformity_loss`在每个样本内的可见token之间计算cos²，不进行跨样本比较，不等同于论文的momentum encoder + InfoNCE。可以确认缺少该机制，不能由此确认LP必然差或表征坍塌。论文表5的NTU60 XSub 74.8→85.1及表6的85.1→86.9只属于其协议，不能当作当前distill的预期增益。

验证建议尚未实现：在相同mask和t分组下，分别统计4帧patch均值误差与去时间均值后的动态误差（两者相加严格等于原坐标MSE），同时检查正常/打乱条件的重建差异。若要做最小消融，保持A归一化、q_sample、t范围和冻结decoder输入空间不变，保留原坐标loss，仅在loss端增加对预测与target一致定义的逐轴时间NAT辅助项；静止片段必须设方差下限。只把target替换为NAT坐标、却保留物理x0预测，会使两侧定义不一致。任何收益均需实际PT/LP验证；当前没有本分支训练结果，也没有本地Torch/checkpoint梯度测量。

## 训练、恢复与线性评估

默认 PT400、warmup20、末20轮floor，AdamW lr1e-3/min_lr5e-4/wd.05/betas(.9,.95)，AMP开，双卡每卡32/accum1=global64，seed0。head与encoder沿用同一优化器和LR。native timm 参数分组跳过 requires_grad=False 参数；偏置/LN沿用不衰减规则。尾部累积策略、DDP等沿用当前 main_pretrain/engine_pretrain。

以下命令只供用户在服务器运行。源权重已填入T20历史命令的路径；LP使用根目录 `main_linprobe.py` 和 `config/ntu60_xsub_joint/linprobe_madiff.yaml`。今后新运行命令统一使用仓库根入口和主配置，不使用 `handoff_artifacts` 下的入口或配置。PT、LP输出目录独立。默认输出目录用于首次运行，已有实验不要从头覆盖，恢复应显式使用 `--resume`。

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10281 main_pretrain_decoder_distill.py --config config/ntu60_xsub_joint/pretrain_madiff_decoder_distill.yaml --decoder_checkpoint output_dir/native_official_dec5_ep400_bs32_seed0_20261012/pt/checkpoint-399.pth --batch_size 32 --accum_iter 1 --epochs 400 --warmup_epochs 20 --min_lr_epochs 20 --lr 0.001 --min_lr 0.0005 --mask_ratio 0.9 --motion_aware_tau -1 --seed 0 --output_dir output_dir/ntu60_xsub_decoder_distill_dec5_randhead_t300_700_ep400_seed0/pt --log_dir output_dir/ntu60_xsub_decoder_distill_dec5_randhead_t300_700_ep400_seed0/pt/tensorboard && OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10282 main_linprobe.py --config config/ntu60_xsub_joint/linprobe_madiff.yaml --finetune output_dir/ntu60_xsub_decoder_distill_dec5_randhead_t300_700_ep400_seed0/pt/checkpoint-399.pth --batch_size 64 --accum_iter 1 --epochs 100 --warmup_epochs 0 --min_lr_epochs 10 --lr 0.1 --min_lr 0 --seed 0 --dist_eval --output_dir output_dir/ntu60_xsub_decoder_distill_dec5_randhead_t300_700_ep400_seed0/lp --log_dir output_dir/ntu60_xsub_decoder_distill_dec5_randhead_t300_700_ep400_seed0/lp/tensorboard
```

恢复使用同一独立入口和配置，将 `--decoder_checkpoint ...` 替换成 `--resume .../pt/checkpoint-X.pth`，两者不同时指定。恢复只接受本方法的训练checkpoint，核验时间步区间和损失定义；直接用恢复文件构造冻结主体，再由原入口恢复完整学生/head/optimizer/scaler/epoch，不要求原教师文件仍存在。没有承诺跨环境或resume逐位复现。

保存仍使用原生checkpoint格式。LP直接读取本方法训练出的encoder，decoder/head键是unexpected keys并被现有strict=False忽略；LP有自己的分类head。首轮LP沿用现有100轮、总batch128、lr.1/min_lr0和A归一化。教师预训练预算与学生400轮分别记录。

## 验证

`python -m unittest discover -s tests -p test_decoder_distill.py -v`

测试包含无Torch可运行的实际checkpoint筛选/元数据检查，以及需要Torch的CPU前向、单独重建梯度、优化后冻结权重不变、head/encoder更新、train/eval、采样端点、MSE公式、LP加载和无需原教师文件的恢复检查。没有Torch时后者明确skip，不能据此称实际autograd/AMP/DDP通过。

本地结果：新检查7通过、9因无Torch跳过；原有累积策略5项与旋转runner8项通过。新Python文件通过3.8语法检查，独立CLI与原生parser的参数兼容检查通过。当前两个本地Python环境均无Torch，尚未执行实际checkpoint张量加载、CPU/CUDA模型前向、autograd或DDP，也没有实际NPZ/服务器源权重可做完整训练检查。
