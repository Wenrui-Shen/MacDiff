# 骨架输入均值/方差配置审查（2026-10-04）

> 2026-10-07补充：以下31 YAML/B生效0/NTU60 XSub全A是2026-10-04的历史扫描快照。当前有35份YAML，新增no-share、no-share_st02及原版B的PT/LP四份；`pretrain_madiff_norm_b.yaml`和`linprobe_madiff_norm_b.yaml`已启用B，原版B也已取得LP85.183164%。本次没有重写旧快照的逐项计数。最新实验与保存缺项见 [实验结果总表](D:/program/MacDiff/EXPERIMENT_RESULTS.md)。

范围：`config/` 全部31份YAML；按 input_mean 与紧邻 input_var 成对统计，包含注释。共8组不同值、50次成对出现，其中31次生效、19次注释。模型构造器默认 mean=[0,0,0]、var=[1,1,1] 不计入配置分组；这些YAML均显式指定了一组生效值。未修改训练配置或模型。

这里的生效仅指 YAML 未注释；self_shift=True 时模型先按样本中心化并强制 input_mean=[0,0,0]，input_var 仍使用配置值。

## 不同数值组

| 组 | input_mean | input_var | 生效次数 | 注释次数 | 出现位置含义 |
|---|---|---|---:|---:|---|
| A | [-0.0058, -0.1333, -0.0246] | [0.0206, 0.0805, 0.0218] | 28 | 1 | 旧组；多数配置共用 |
| B | [-0.0024, -0.2132, -0.0446] | [0.0525, 0.1527, 0.0513] | 0 | 6 | pretrain/LP 的 #new；多个协议共用候选值 |
| C | [-0.0034, -0.1322, -0.0271] | [0.088325, 0.106987, 0.065367] | 0 | 1 | NTU60 XSub finetune 的 #new 候选值 |
| D | [-0.0027, -0.1353, -0.0244] | [0.0211, 0.0826, 0.0241] | 0 | 2 | NTU60 XView 注释候选值 |
| E | [-0.0021, -0.2198, -0.0404] | [0.0298, 0.1424, 0.0358] | 1 | 0 | NTU120 XSub finetune 生效值 |
| F | [-0.0016, -0.2247, -0.0321] | [0.0306, 0.1451, 0.0398] | 1 | 0 | NTU120 XSet finetune 生效值 |
| G | [-0.0686, -0.0241, 1.513] | [0.2834, 0.2447, 2.0166] | 0 | 5 | 配置注释标为 pkuv1 2man |
| H | [-0.0459, -0.068, 1.3642] | [0.2356, 0.2351, 1.9492] | 1 | 4 | 配置注释标为 pkuv2 2man |

## 每份配置的分组与状态

| 配置 | 生效组 | 注释候选组 | 显式 self_shift | 显式 one_person |
|---|---|---|---|---|
| [config/ntu120_xset_joint/finetune_madiff.yaml](D:/program/MacDiff/config/ntu120_xset_joint/finetune_madiff.yaml) | F | 无 | 未设置 | 未设置 |
| [config/ntu120_xset_joint/linprobe_madiff.yaml](D:/program/MacDiff/config/ntu120_xset_joint/linprobe_madiff.yaml) | A | B | 未设置 | 未设置 |
| [config/ntu120_xset_joint/pretrain_madiff.yaml](D:/program/MacDiff/config/ntu120_xset_joint/pretrain_madiff.yaml) | A | B | 未设置 | 未设置 |
| [config/ntu120_xsub_joint/finetune_madiff.yaml](D:/program/MacDiff/config/ntu120_xsub_joint/finetune_madiff.yaml) | E | 无 | 未设置 | 未设置 |
| [config/ntu120_xsub_joint/linprobe_madiff.yaml](D:/program/MacDiff/config/ntu120_xsub_joint/linprobe_madiff.yaml) | A | B | 未设置 | 未设置 |
| [config/ntu120_xsub_joint/pretrain_madiff.yaml](D:/program/MacDiff/config/ntu120_xsub_joint/pretrain_madiff.yaml) | A | B | 未设置 | 未设置 |
| [config/ntu60_xsub_joint/finetune_madiff.yaml](D:/program/MacDiff/config/ntu60_xsub_joint/finetune_madiff.yaml) | A | C | 未设置 | 未设置 |
| [config/ntu60_xsub_joint/linprobe_madiff.yaml](D:/program/MacDiff/config/ntu60_xsub_joint/linprobe_madiff.yaml) | A | B | 未设置 | 未设置 |
| [config/ntu60_xsub_joint/pretrain_madiff.yaml](D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff.yaml) | A | B | 未设置 | 未设置 |
| [config/ntu60_xsub_joint/pretrain_madiff_ose_peer.yaml](D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_ose_peer.yaml) | A | 无 | 未设置 | True |
| [config/ntu60_xsub_joint/pretrain_madiff_stage2.yaml](D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_stage2.yaml) | A | 无 | False | True |
| [config/ntu60_xsub_joint/pretrain_madiff_stage2_dense_ose.yaml](D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_stage2_dense_ose.yaml) | A | 无 | False | False |
| [config/ntu60_xsub_joint/pretrain_madiff_text.yaml](D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_text.yaml) | A | 无 | 未设置 | True |
| [config/ntu60_xsub_joint/pretrain_madiff_text_rms.yaml](D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_text_rms.yaml) | A | 无 | 未设置 | True |
| [config/ntu60_xsub_joint/pretrain_madiff_text_rms_ema_bidirectional.yaml](D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_text_rms_ema_bidirectional.yaml) | A | 无 | 未设置 | True |
| [config/ntu60_xsub_joint/pretrain_madiff_text_rms_ema_bidirectional_shared.yaml](D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_text_rms_ema_bidirectional_shared.yaml) | A | 无 | 未设置 | True |
| [config/ntu60_xsub_joint/pretrain_madiff_text_rms_sample_target_blend.yaml](D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_text_rms_sample_target_blend.yaml) | A | 无 | 未设置 | True |
| [config/ntu60_xsub_joint/pretrain_madiff_text_rms_sample_target_blend_shared.yaml](D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_text_rms_sample_target_blend_shared.yaml) | A | 无 | 未设置 | True |
| [config/ntu60_xsub_joint/pretrain_madiff_text_rms_st01.yaml](D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_text_rms_st01.yaml) | A | 无 | 未设置 | True |
| [config/ntu60_xsub_joint/pretrain_madiff_text_sentence_fixed_rms.yaml](D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_text_sentence_fixed_rms.yaml) | A | 无 | 未设置 | True |
| [config/ntu60_xsub_joint/pretrain_madiff_text_sentence_sample_target_blend_shared.yaml](D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_text_sentence_sample_target_blend_shared.yaml) | A | 无 | 未设置 | True |
| [config/ntu60_xsub_joint/pretrain_madiff_text_st01.yaml](D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_text_st01.yaml) | A | 无 | 未设置 | True |
| [config/ntu60_xview_joint/finetune_madiff.yaml](D:/program/MacDiff/config/ntu60_xview_joint/finetune_madiff.yaml) | A | D | 未设置 | 未设置 |
| [config/ntu60_xview_joint/linprobe_madiff.yaml](D:/program/MacDiff/config/ntu60_xview_joint/linprobe_madiff.yaml) | A | D | 未设置 | 未设置 |
| [config/ntu60_xview_joint/pretrain_madiff.yaml](D:/program/MacDiff/config/ntu60_xview_joint/pretrain_madiff.yaml) | A | 无 | 未设置 | 未设置 |
| [config/pkuv1_xsub_joint/finetune_madiff.yaml](D:/program/MacDiff/config/pkuv1_xsub_joint/finetune_madiff.yaml) | A | 无 | 未设置 | 未设置 |
| [config/pkuv1_xsub_joint/linprobe_madiff.yaml](D:/program/MacDiff/config/pkuv1_xsub_joint/linprobe_madiff.yaml) | A | G,H | True | 未设置 |
| [config/pkuv1_xsub_joint/pretrain_madiff.yaml](D:/program/MacDiff/config/pkuv1_xsub_joint/pretrain_madiff.yaml) | A | G,H | True | 未设置 |
| [config/pkuv2_xsub_joint/finetune_madiff.yaml](D:/program/MacDiff/config/pkuv2_xsub_joint/finetune_madiff.yaml) | A | G,H | True | 未设置 |
| [config/pkuv2_xsub_joint/linprobe_madiff.yaml](D:/program/MacDiff/config/pkuv2_xsub_joint/linprobe_madiff.yaml) | A | G,H | 未设置 | 未设置 |
| [config/pkuv2_xsub_joint/pretrain_madiff.yaml](D:/program/MacDiff/config/pkuv2_xsub_joint/pretrain_madiff.yaml) | H | A,G | 未设置 | 未设置 |

## 所有出现位置

### A：旧组；多数配置共用

| 配置 | 状态 | mean/var 行号 |
|---|---|---|
| [config/ntu120_xset_joint/linprobe_madiff.yaml](D:/program/MacDiff/config/ntu120_xset_joint/linprobe_madiff.yaml:50) | 生效 | 50 / 51 |
| [config/ntu120_xset_joint/pretrain_madiff.yaml](D:/program/MacDiff/config/ntu120_xset_joint/pretrain_madiff.yaml:43) | 生效 | 43 / 44 |
| [config/ntu120_xsub_joint/linprobe_madiff.yaml](D:/program/MacDiff/config/ntu120_xsub_joint/linprobe_madiff.yaml:50) | 生效 | 50 / 51 |
| [config/ntu120_xsub_joint/pretrain_madiff.yaml](D:/program/MacDiff/config/ntu120_xsub_joint/pretrain_madiff.yaml:42) | 生效 | 42 / 43 |
| [config/ntu60_xsub_joint/finetune_madiff.yaml](D:/program/MacDiff/config/ntu60_xsub_joint/finetune_madiff.yaml:54) | 生效 | 54 / 55 |
| [config/ntu60_xsub_joint/linprobe_madiff.yaml](D:/program/MacDiff/config/ntu60_xsub_joint/linprobe_madiff.yaml:51) | 生效 | 51 / 52 |
| [config/ntu60_xsub_joint/pretrain_madiff.yaml](D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff.yaml:42) | 生效 | 42 / 43 |
| [config/ntu60_xsub_joint/pretrain_madiff_ose_peer.yaml](D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_ose_peer.yaml:41) | 生效 | 41 / 42 |
| [config/ntu60_xsub_joint/pretrain_madiff_stage2.yaml](D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_stage2.yaml:39) | 生效 | 39 / 40 |
| [config/ntu60_xsub_joint/pretrain_madiff_stage2_dense_ose.yaml](D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_stage2_dense_ose.yaml:38) | 生效 | 38 / 39 |
| [config/ntu60_xsub_joint/pretrain_madiff_text.yaml](D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_text.yaml:40) | 生效 | 40 / 41 |
| [config/ntu60_xsub_joint/pretrain_madiff_text_rms.yaml](D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_text_rms.yaml:40) | 生效 | 40 / 41 |
| [config/ntu60_xsub_joint/pretrain_madiff_text_rms_ema_bidirectional.yaml](D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_text_rms_ema_bidirectional.yaml:40) | 生效 | 40 / 41 |
| [config/ntu60_xsub_joint/pretrain_madiff_text_rms_ema_bidirectional_shared.yaml](D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_text_rms_ema_bidirectional_shared.yaml:40) | 生效 | 40 / 41 |
| [config/ntu60_xsub_joint/pretrain_madiff_text_rms_sample_target_blend.yaml](D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_text_rms_sample_target_blend.yaml:40) | 生效 | 40 / 41 |
| [config/ntu60_xsub_joint/pretrain_madiff_text_rms_sample_target_blend_shared.yaml](D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_text_rms_sample_target_blend_shared.yaml:40) | 生效 | 40 / 41 |
| [config/ntu60_xsub_joint/pretrain_madiff_text_rms_st01.yaml](D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_text_rms_st01.yaml:40) | 生效 | 40 / 41 |
| [config/ntu60_xsub_joint/pretrain_madiff_text_sentence_fixed_rms.yaml](D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_text_sentence_fixed_rms.yaml:41) | 生效 | 41 / 42 |
| [config/ntu60_xsub_joint/pretrain_madiff_text_sentence_sample_target_blend_shared.yaml](D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_text_sentence_sample_target_blend_shared.yaml:41) | 生效 | 41 / 42 |
| [config/ntu60_xsub_joint/pretrain_madiff_text_st01.yaml](D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_text_st01.yaml:40) | 生效 | 40 / 41 |
| [config/ntu60_xview_joint/finetune_madiff.yaml](D:/program/MacDiff/config/ntu60_xview_joint/finetune_madiff.yaml:47) | 生效 | 47 / 48 |
| [config/ntu60_xview_joint/linprobe_madiff.yaml](D:/program/MacDiff/config/ntu60_xview_joint/linprobe_madiff.yaml:47) | 生效 | 47 / 48 |
| [config/ntu60_xview_joint/pretrain_madiff.yaml](D:/program/MacDiff/config/ntu60_xview_joint/pretrain_madiff.yaml:45) | 生效 | 45 / 46 |
| [config/pkuv1_xsub_joint/finetune_madiff.yaml](D:/program/MacDiff/config/pkuv1_xsub_joint/finetune_madiff.yaml:50) | 生效 | 50 / 51 |
| [config/pkuv1_xsub_joint/linprobe_madiff.yaml](D:/program/MacDiff/config/pkuv1_xsub_joint/linprobe_madiff.yaml:48) | 生效 | 48 / 49 |
| [config/pkuv1_xsub_joint/pretrain_madiff.yaml](D:/program/MacDiff/config/pkuv1_xsub_joint/pretrain_madiff.yaml:46) | 生效 | 46 / 47 |
| [config/pkuv2_xsub_joint/finetune_madiff.yaml](D:/program/MacDiff/config/pkuv2_xsub_joint/finetune_madiff.yaml:50) | 生效 | 50 / 51 |
| [config/pkuv2_xsub_joint/linprobe_madiff.yaml](D:/program/MacDiff/config/pkuv2_xsub_joint/linprobe_madiff.yaml:48) | 生效 | 48 / 49 |
| [config/pkuv2_xsub_joint/pretrain_madiff.yaml](D:/program/MacDiff/config/pkuv2_xsub_joint/pretrain_madiff.yaml:46) | 注释 | 46 / 47 |

### B：pretrain/LP 的 #new；多个协议共用候选值

| 配置 | 状态 | mean/var 行号 |
|---|---|---|
| [config/ntu120_xset_joint/linprobe_madiff.yaml](D:/program/MacDiff/config/ntu120_xset_joint/linprobe_madiff.yaml:47) | 注释 | 47 / 48 |
| [config/ntu120_xset_joint/pretrain_madiff.yaml](D:/program/MacDiff/config/ntu120_xset_joint/pretrain_madiff.yaml:46) | 注释 | 46 / 47 |
| [config/ntu120_xsub_joint/linprobe_madiff.yaml](D:/program/MacDiff/config/ntu120_xsub_joint/linprobe_madiff.yaml:47) | 注释 | 47 / 48 |
| [config/ntu120_xsub_joint/pretrain_madiff.yaml](D:/program/MacDiff/config/ntu120_xsub_joint/pretrain_madiff.yaml:45) | 注释 | 45 / 46 |
| [config/ntu60_xsub_joint/linprobe_madiff.yaml](D:/program/MacDiff/config/ntu60_xsub_joint/linprobe_madiff.yaml:48) | 注释 | 48 / 49 |
| [config/ntu60_xsub_joint/pretrain_madiff.yaml](D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff.yaml:45) | 注释 | 45 / 46 |

### C：NTU60 XSub finetune 的 #new 候选值

| 配置 | 状态 | mean/var 行号 |
|---|---|---|
| [config/ntu60_xsub_joint/finetune_madiff.yaml](D:/program/MacDiff/config/ntu60_xsub_joint/finetune_madiff.yaml:51) | 注释 | 51 / 52 |

### D：NTU60 XView 注释候选值

| 配置 | 状态 | mean/var 行号 |
|---|---|---|
| [config/ntu60_xview_joint/finetune_madiff.yaml](D:/program/MacDiff/config/ntu60_xview_joint/finetune_madiff.yaml:50) | 注释 | 50 / 51 |
| [config/ntu60_xview_joint/linprobe_madiff.yaml](D:/program/MacDiff/config/ntu60_xview_joint/linprobe_madiff.yaml:49) | 注释 | 49 / 50 |

### E：NTU120 XSub finetune 生效值

| 配置 | 状态 | mean/var 行号 |
|---|---|---|
| [config/ntu120_xsub_joint/finetune_madiff.yaml](D:/program/MacDiff/config/ntu120_xsub_joint/finetune_madiff.yaml:45) | 生效 | 45 / 46 |

### F：NTU120 XSet finetune 生效值

| 配置 | 状态 | mean/var 行号 |
|---|---|---|
| [config/ntu120_xset_joint/finetune_madiff.yaml](D:/program/MacDiff/config/ntu120_xset_joint/finetune_madiff.yaml:45) | 生效 | 45 / 46 |

### G：配置注释标为 pkuv1 2man

| 配置 | 状态 | mean/var 行号 |
|---|---|---|
| [config/pkuv1_xsub_joint/linprobe_madiff.yaml](D:/program/MacDiff/config/pkuv1_xsub_joint/linprobe_madiff.yaml:51) | 注释 | 51 / 52 |
| [config/pkuv1_xsub_joint/pretrain_madiff.yaml](D:/program/MacDiff/config/pkuv1_xsub_joint/pretrain_madiff.yaml:49) | 注释 | 49 / 50 |
| [config/pkuv2_xsub_joint/finetune_madiff.yaml](D:/program/MacDiff/config/pkuv2_xsub_joint/finetune_madiff.yaml:53) | 注释 | 53 / 54 |
| [config/pkuv2_xsub_joint/linprobe_madiff.yaml](D:/program/MacDiff/config/pkuv2_xsub_joint/linprobe_madiff.yaml:51) | 注释 | 51 / 52 |
| [config/pkuv2_xsub_joint/pretrain_madiff.yaml](D:/program/MacDiff/config/pkuv2_xsub_joint/pretrain_madiff.yaml:49) | 注释 | 49 / 50 |

### H：配置注释标为 pkuv2 2man

| 配置 | 状态 | mean/var 行号 |
|---|---|---|
| [config/pkuv1_xsub_joint/linprobe_madiff.yaml](D:/program/MacDiff/config/pkuv1_xsub_joint/linprobe_madiff.yaml:54) | 注释 | 54 / 55 |
| [config/pkuv1_xsub_joint/pretrain_madiff.yaml](D:/program/MacDiff/config/pkuv1_xsub_joint/pretrain_madiff.yaml:52) | 注释 | 52 / 53 |
| [config/pkuv2_xsub_joint/finetune_madiff.yaml](D:/program/MacDiff/config/pkuv2_xsub_joint/finetune_madiff.yaml:56) | 注释 | 56 / 57 |
| [config/pkuv2_xsub_joint/linprobe_madiff.yaml](D:/program/MacDiff/config/pkuv2_xsub_joint/linprobe_madiff.yaml:54) | 注释 | 54 / 55 |
| [config/pkuv2_xsub_joint/pretrain_madiff.yaml](D:/program/MacDiff/config/pkuv2_xsub_joint/pretrain_madiff.yaml:52) | 生效 | 52 / 53 |

## 数值比较与运行差异

A与D接近，D/A方差约为[1.0243,1.0261,1.1055]。B与E/F的Y均值、Y方差接近，但X/Z方差不同；它们不是同组。E与F也仅接近而不相同。C的均值接近A，但方差明显更大，特别是X轴，不能与B混为一组。G/H的Z均值约1.4–1.5、Z方差约2，与NTU各组尺度明显不同。数值接近不能证明统计来自同一数据集或相同预处理。

当前所有NTU60 XSub配置（包括文本、OSE、Stage2）均生效A，pretrain与LP一致。NTU60 XView也全用A。NTU120 XSub/XSet的pretrain/LP用A，finetune分别用E/F。PKUv1全部用A；PKUv2 pretrain用H、LP/finetune用A。后两种pretrain与finetune/LP不同的情况必须连同self_shift和是否冻结encoder解释，不能仅凭不同数值判为错误。

B在NTU60 XSub、NTU120 XSub、NTU120 XSet的pretrain/LP共6处出现且全部注释；C仅在NTU60 XSub finetune一处注释。Git初始提交0aafec2已经包含B/C；未找到计算脚本、统计输出或来源记录。不能确证B源自NTU120，也不能确证C就是按NTU60 XSub当前训练预处理计算的。

## A换B对当前训练的影响

代码按 (raw-mean)/sqrt(var) 标准化。下面公式使用当前NTU60设置self_shift=False，并固定同一原始坐标。

| 轴 | A标准差 | B标准差 | B归一化值相对A的比例 | B归一化值的额外偏移 | 中心化方差/SNR相对A |
|---|---:|---:|---:|---:|---:|
| X | 0.143527 | 0.229129 | 0.626403 | -0.014839 | 0.392381 |
| Y | 0.283725 | 0.390768 | 0.726070 | +0.204469 | 0.527177 |
| Z | 0.147648 | 0.226495 | 0.651883 | +0.088302 | 0.424951 |

即 z_B=[0.626403,0.726070,0.651883]*z_A+[-0.014839,0.204469,0.088302]，为逐通道仿射变换。各轴坐标变化幅度缩小27.4%–37.4%；固定时间步下，以中心化信号方差定义的有效SNR降到39.2%–52.7%。均值偏移会影响二阶能量，因此这里的SNR比例不用于未经中心化的原点总能量。

C与B并不等价。A换C的坐标变化幅度比例是[0.482938,0.867426,0.577496]，额外偏移[-0.008075,-0.003363,+0.009778]；其方向间缩放差异更大。

当前骨架扩散 q_sample=a_t*z0+sigma_t*epsilon；epsilon仍是单位高斯。A换B改变骨架encoder输入、原生骨架重建与T→S任务的有效噪声强度，位置/时间embedding与内容幅度的相对关系也会变化。S→T仍预测文本噪声，文本目标的尺度不会直接按骨架方差比例改变，但encoder条件及T→S训练remap/共享decoder产生的间接路径会受影响。

原生与T→S均回归epsilon，loss的目标单位未改变，不能机械把loss乘以old_var/new_var。较小的干净信号可能让模型更容易从noisy输入估计噪声；更低的噪声MSE不能单独证明语义条件使用更强或LP更好。仿射变换不增加样本信息，但优化、正则及扩散训练不是对该变换自动不变，性能可能升也可能降，幅度需实测。

input_mean/input_var是普通Python属性，没有注册为parameter/buffer。LP先用自己的YAML构造模型，再load checkpoint[model]；更改LP统计值会实际生效，旧checkpoint不会自动覆盖它。当前LP冻结encoder，head中的BatchNorm不能保证补偿前端归一化造成的特征变化；仅在旧checkpoint上换LP统计，不能验证新归一化预训练的效果。

建议保留当前512修复实验的A基线。若比较B，使用独立输出目录，从头预训练并让该实验的LP同样使用B，其他训练权重、share、batch、crop、噪声日程与seed固定；当前缓存内容不因这两个骨架输入参数而改变，不需要重生成文本cache。重算数据统计时应明确train split、当前裁剪、person0/两人、空人物、时间padding及旋转口径，仅凭历史#new标签不能认定数值正确。

## 代码证据

- [model/transformer_macdiff.py](D:/program/MacDiff/model/transformer_macdiff.py:363)：var开平方得到std；self_shift覆盖mean。
- [model/transformer_macdiff.py](D:/program/MacDiff/model/transformer_macdiff.py:475)：骨架归一化。
- [guided_diffusion/gaussian_diffusion.py](D:/program/MacDiff/guided_diffusion/gaussian_diffusion.py:201)：q_sample。
- [model/transformer_macdiff_text.py](D:/program/MacDiff/model/transformer_macdiff_text.py:306)：T→S噪声回归；316行S→T文本噪声回归。
- [model/transformer_downstream.py](D:/program/MacDiff/model/transformer_downstream.py:322)：LP归一化参数不是state_dict buffer。
- [main_linprobe.py](D:/program/MacDiff/main_linprobe.py:210)：先按YAML构造模型后加载权重；243行冻结encoder。
