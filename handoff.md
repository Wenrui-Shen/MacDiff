# MacDiff Joint-aware ReSA+OSE Stage2 交接

更新时间：2026-08-22

## 1. 当前任务

在原生 MacDiff NTU60 XSub Stage1 checkpoint 上增加一个独立的100-epoch
ReSA+OSE Stage2，并用原项目 `linprobe2` 协议评估。当前重点不是继续扩展模型，
而是解决：Stage2 的 ReSA 已经恢复有效，但训练后的 LP accuracy 低于 Stage1
checkpoint 的约85.86%。

Stage1 权重必须使用：

```text
./output_dir/ntu60_xsub_macdiff/checkpoint-399.pth
```

不要改回 `ntu60_xsub_ose`。

## 2. 已完成内容

### 2.1 Stage2训练流程

- 新增 `main_pretrain_stage2.py`、`model/transformer_stage2.py`、
  `config/ntu60_xsub_joint/pretrain_madiff_stage2.yaml` 和启动脚本。
- 仅从Stage1转移在线骨架encoder；diffusion decoder、optimizer、EMA和旧OSE状态
  均不转移。
- Stage2包含ReSA、独立OSE projector、K=2 Joint-only原型、mixed prototype loss和
  mixed instance loss。
- K=2表示每类Joint样本独立增强两次，共两个Joint embedding。Motion/Bone构造和
  对应EMA exemplar分支已经删除，不能再按旧的六embedding语义理解。
- 双卡使用标准DDP；关系矩阵、mixed permutation和instance keys跨卡构造；
  projector/predictor的BN会转换为SyncBatchNorm，使无标签分支统计对应全局batch128。
- fresh run若输出目录是 `./output_dir/` 下明确的子目录，会自动删除并重建；
  resume不会删除。
- 每10轮保存完整Stage2 checkpoint和仅含 `encoder_q` 的LP backbone。

### 2.2 已明确移除的内容

用户明确要求下列内容不要保留：

- activation checkpoint / `checkpoint_blocks`；
- encoder chunk；
- checkpoint引入的手动梯度同步和BN广播；
- Stage2 queue及所有enqueue、buffer、日志字段。

不要擅自重新加入。

### 2.3 当前mask协议

- `mask_ratio=0.9`，每条样本保留75/750 token。
- 同一个无标签view的online/EMA共享同一组mask indices。
- 两个Joint-only exemplar增强分别使用独立mask；不再存在Motion/Bone mask。
- 仍是全局随机75个token；尚未改成每joint固定抽3个token。用户要求本轮只改
  joint-aware pooling，其他先不改。

### 2.4 Joint-aware改造

发现现有LP并不是256维全局均值：

- `linprobe_madiff.yaml` 使用 `protocol: linprobe2`；
- 下游先平均person/time，保留25个joint，再展平为 `25 x 256 = 6400` 维；
- LP linear head前还有无仿射BatchNorm。

原Stage2却将所有可见time/joint token一起平均成256维，导致ReSA Sinkhorn目标
接近均匀。现在Stage2已改成：

1. 根据mask保留下来的原始flattened token id恢复joint id；
2. 每个joint内部对可见时间token求均值；
3. 缺失joint填零；
4. 按joint顺序展平为6400维；
5. ReSA/OSE projector输入相应改为6400维。

没有改变mask采样、loss权重或温度。对应测试位于 `tests/test_stage2.py`。

### 2.5 ReSA 4.8问题结论

双卡每卡64时全局关系batch为128。ReSA raw CE满足：

```text
ReSA = H(Sinkhorn target) + KL(target || prediction)
ln(128) = 4.852
```

旧全局均值版本日志中 `H` 接近4.8且 `KL` 很小，说明target和prediction都接近
均匀，ReSA几乎没有有效梯度。Joint-aware后用户确认ReSA已经正常。

不要再把raw ReSA约4.8简单解释为loss权重错误；必须分别看H和KL。

### 2.6 Joint-only原型与SyncBN改造

2026-08-22新增两项对照修改：

- `ExemplarProvider`只产生K个Joint增强，不再构造Motion和Bone；
- 原型直接对K个归一化Joint anchor取均值；当前K=2即两个Joint anchor；
- 删除已经无调用的teacher Motion/Bone exemplar projection和JMB融合逻辑；
- DDP默认`sync_batchnorm: True`，在包装DDP前转换模型中的BN；
- 额外Joint view仍使用batch statistics但不更新长期running buffers，且该逻辑兼容
  `SyncBatchNorm`；
- mask协议名更新为`shared_qk_joint_v1`，因此不能resume旧JMB Stage2 checkpoint。

### 2.7 Stage2增强关闭

为隔离MacDiff Stage1/Stage2增强分布跳变，当前配置已将
`augmentation_probability`设为`0.0`。temporal crop、shear和rotation均不执行；
基础`base_p_interval: [0.95]`裁剪/resize仍保留。无标签view A/B和K=2 Joint
exemplar仍分别抽取独立的90%随机mask，因此训练并非两个完全相同的输入分支。

## 3. 当前阻塞点：ReSA正常但LP下降

Stage1 checkpoint的LP约85.86，joint-aware Stage2训练后LP仍下降。需要同时验证
backbone更新速度、稀疏mask关系目标和原型质量。用户补充PSTL也存在Stage1 AdamW、
Stage2 SGD且表现良好的先例，因此不要把“AdamW切换SGD”本身当作主要原因。
当前仍保留低backbone LR对照，因为数值0.25相对MacDiff预训练LR仍可能造成漂移：

- MacDiff Stage1：AdamW，backbone LR `1e-3`；
- MacDiff finetune：LR `5e-4`；
- 当前Stage2：SGD，backbone LR `0.25`，无warmup；
- fresh ReSA/OSE heads同样使用LR `0.25`。

Joint-aware之前ReSA是均匀退化目标，梯度接近零；joint-aware后ReSA真正产生梯度，
`0.25` 很可能快速改写已经能LP 85.86的encoder。因此下一实验只降低backbone
LR，不动head LR和其他协议：

```text
backbone lr = 0.001
head lr     = 0.25
```

Stage2导出的backbone不包含6400维projector，因此LP下降不是checkpoint key或head
shape加载错误，而是encoder参数本身发生了不利漂移。

## 4. 新增LP sweep脚本

`script_linprobe_stage2_sweep.sh` 会依次对
`checkpoint-010-backbone.pth` 到 `checkpoint-100-backbone.pth` 跑完整100轮LP。

特点：

- 每个Stage2 checkpoint使用独立LP目录和TensorBoard；
- `main_linprobe.py --output_dir ""`，避免每个LP epoch保存完整模型导致巨大磁盘占用；
- console日志保存在各自run目录；
- 从 `Max accuracy` 提取每个checkpoint的best acc；
- 最后打印10个best acc和overall best；
- CSV保存在 `${LP_ROOT}/best_acc_summary.csv`；
- 若LP_ROOT已存在会拒绝运行，避免混合两次实验。换一个 `LP_ROOT` 即可。

旧LR=0.25 Stage2的sweep命令：

```bash
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 BATCH_SIZE=64 LP_ROOT=./output_dir/ntu60_xsub_macdiff_stage2_seed0_lp_sweep bash script_linprobe_stage2_sweep.sh ./output_dir/ntu60_xsub_macdiff_stage2_seed0
```

当前Joint-only + SyncBN低LR实验完成后的sweep命令：

```bash
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 BATCH_SIZE=64 LP_ROOT=./output_dir/ntu60_xsub_macdiff_stage2_jointonly_noaug_syncbn_lr1e3_lp_sweep bash script_linprobe_stage2_sweep.sh ./output_dir/ntu60_xsub_macdiff_stage2_jointonly_noaug_syncbn_lr1e3
```

## 5. 低backbone LR训练命令

`script_pretrain_stage2.sh` 现在支持 `BACKBONE_LR` 和 `HEAD_LR` 环境变量。

```bash
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 MASTER_PORT=10237 BATCH_SIZE=64 BACKBONE_LR=0.001 HEAD_LR=0.25 OUTPUT_DIR=./output_dir/ntu60_xsub_macdiff_stage2_jointonly_noaug_syncbn_lr1e3 OMP_NUM_THREADS=1 bash script_pretrain_stage2.sh ./output_dir/ntu60_xsub_macdiff/checkpoint-399.pth
```

## 6. 下一步计划

1. 从Stage1 fresh run当前Joint-only + no-augmentation + SyncBN版本，不能resume任何
   旧JMB或有增强Stage2。
2. 优先用backbone LR `0.001`、head LR `0.25`训练，并逐10轮运行LP sweep。
3. 如果需要严格拆变量，再用同一LR分别运行`sync_batchnorm=False`或K=1；不要把
   多项改动混进同一个对照。
4. 比较相同epoch：
   - 若旧实验第10轮即大降而低LR保持，确认是catastrophic backbone drift；
   - 若两者均前期高、后期下降，缩短Stage2或选择最佳中间checkpoint；
   - 若低LR仍持续下降，再记录encoder parameter drift、ReSA/OSE backbone梯度范数
     与梯度余弦，判断目标冲突。
5. 在上述对照完成前，不继续修改Sinkhorn温度、loss权重或mask策略，
   避免同时改变多个变量。

### 6.1 OSE-only低LR对照

低backbone LR最终checkpoint的LP达到85.22；若完整sweep确认这就是最高值，下一项
单变量对照为关闭ReSA梯度、保留全部OSE目标：

```text
resa_weight          = 0.0
ose_lambda           = 1.0
ose_mix_proto_weight = 1.0
ose_mix_ins_weight   = 1.0
backbone lr          = 0.001
head lr              = 0.25
```

启动命令：

```bash
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 MASTER_PORT=10237 BATCH_SIZE=64 OMP_NUM_THREADS=1 bash script_pretrain_stage2_ose_only.sh ./output_dir/ntu60_xsub_macdiff/checkpoint-399.pth
```

该实验仍计算并记录ReSA指标用于诊断，但其权重为零，不向encoder或ReSA head提供
梯度。必须从Stage1 fresh run，不能resume ReSA+OSE checkpoint。

OSE-only最终checkpoint的LP为85.02，低于相同低LR的ReSA+OSE 85.22。该0.20pp
差距说明ReSA可能有轻微帮助，但仍可能处于LP随机波动范围，不能视为显著结论。

### 6.2 ReSA+OSE teacher温度0.06对照

保留低LR完整ReSA+OSE，只将OSE EMA teacher温度从0.04提高到0.06：

```text
resa_weight = 1.0
ose_tau_s   = 0.1
ose_tau_t   = 0.06
backbone lr = 0.001
head lr     = 0.25
```

启动命令：

```bash
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 MASTER_PORT=10237 BATCH_SIZE=64 OMP_NUM_THREADS=1 bash script_pretrain_stage2_resa_ose_taut006.sh ./output_dir/ntu60_xsub_macdiff/checkpoint-399.pth
```

必须fresh run并使用独立输出目录；除`ose_tau_t`外不要同时改变mask、loss权重、K、
SyncBN或augmentation协议。

该实验最终checkpoint的LP为85.12，低于`tau_t=0.04`完整ReSA+OSE的85.22；0.10pp
差距处于LP波动量级，没有证据表明继续增大teacher温度能解决当前瓶颈。

### 6.3 Stage1特征几何诊断

新增`diagnose_stage1_geometry.py`，冻结Stage1 `checkpoint-399.pth`并在不训练的情况
下检查：

- full-token与90% mask特征的余弦一致性；
- 两次独立mask之间的余弦一致性；
- full/masked空间的同类、异类余弦及其间隔；
- 当前单exemplar和masked K=2 exemplar的最近原型准确率；
- `tau_t=0.04/0.06/0.1`对应的原型分布置信度与熵；
- 全局随机75-token mask导致的缺失joint数量。

原型准确率和温度统计位于冻结encoder输出空间，不经过尚未训练的OSE projector，
因此用于判断Stage1几何是否适合单exemplar OSE，不等同于训练后OSE teacher的实测值。

默认平衡抽取4096个非exemplar训练样本，使用单卡且不更新权重：

```bash
CUDA_VISIBLE_DEVICES=0 python diagnose_stage1_geometry.py --checkpoint ./output_dir/ntu60_xsub_macdiff/checkpoint-399.pth --config ./config/ntu60_xsub_joint/pretrain_madiff_stage2.yaml --batch_size 32 --max_samples 4096 --output ./output_dir/stage1_geometry.json
```

本地工作区没有checkpoint和NTU数据，因此结果必须在训练服务器生成。

Stage1的4096样本诊断结果：

- full/masked cosine为0.7581，两次masked cosine为0.6882；
- 每个masked样本平均缺失1.016个joint，67.18%的样本至少缺失一个joint；
- full同异类cosine gap为0.0301，masked后降到0.0214；
- full单exemplar最近原型准确率17.63%，masked K=2仅7.79%；
- masked K=2在`tau=0.04`时平均top-1 confidence仅8.46%，entropy为3.759，
  相对`ln(60)=4.094`仍接近均匀，因此现有证据不支持teacher target过尖。

### 6.4 Stage1/Stage2同样本同mask对比

新增`compare_stage1_stage2_geometry.py`，在相同4096个样本和相同随机mask下对比：

- Stage1与Stage2各自的全部6.3指标；
- full/masked表征的逐样本余弦和linear CKA；
- encoder整体及逐block参数漂移；
- 若输入完整Stage2 checkpoint，再测EMA encoder/projector相对online K=2 prototype的
  实际伪标签准确率、置信度和熵（BN使用checkpoint running statistics）。

```bash
CUDA_VISIBLE_DEVICES=0 python compare_stage1_stage2_geometry.py --stage1_checkpoint ./output_dir/ntu60_xsub_macdiff/checkpoint-399.pth --stage2_checkpoint ./output_dir/ntu60_xsub_macdiff_stage2_jointonly_noaug_syncbn_lr1e3/checkpoint-100.pth --config ./config/ntu60_xsub_joint/pretrain_madiff_stage2.yaml --batch_size 32 --max_samples 4096 --output ./output_dir/stage1_vs_stage2_geometry.json
```

优先传入完整`checkpoint-100.pth`而不是`-backbone.pth`，否则无法诊断真实OSE teacher
projector。

## 7. 绝对不要再踩的坑

1. 不要把LP 85.86理解成256维全局均值有效；LP2实际用6400维joint-aware特征、
   全750 token以及分类头前BN。
2. 旧K=2六embedding版本使用100% token时，6GB显卡即使每卡batch16也OOM；当前
   Joint-only虽减少了原型分支，但本轮对照仍保持10%输入，不要同时切100%。
3. 不要通过继续减普通batch掩盖原型分支的固定显存开销。
4. 不要重新加入checkpoint blocks、encoder chunk、手动梯度同步或Stage2 queue。
5. 不要把Stage1路径写回 `ntu60_xsub_ose`；必须是 `ntu60_xsub_macdiff`。
6. `batch_size` 是每GPU大小；双卡每卡64才是全局128。
7. 不要用旧的256维Stage2 checkpoint resume joint-aware版本；projector shape不兼容。
8. mask协议变化或模型结构变化后必须fresh run；当前协议是
   `shared_qk_joint_v1`。
9. 不要只看raw ReSA；同时看 `cluster_entropy(H)` 和 `cluster_kl`。
10. 不要因为ReSA数值正常就断言表征更适合分类；最终必须逐checkpoint LP验证。

## 8. 验证状态

- `py_compile`静态语法检查通过。
- `git diff --check` 通过，仅有Windows工作区LF/CRLF提示。
- 本机默认Windows Python launcher不可执行，Codex bundled Python没有PyTorch，
  因此没有在本机运行GPU/PyTorch单测。服务器启动脚本会先运行
  `python -m unittest tests.test_stage2`。
- PyTorch 1.8不支持SyncBatchNorm的CPU forward；相关测试已拆成CPU普通BN行为测试
  与不执行forward的SyncBN结构转换测试。正式双卡SyncBN仍在CUDA DDP中运行。
