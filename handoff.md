# MacDiff Joint-aware ReSA+OSE Stage2 交接

更新时间：2026-08-21

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
- Stage2包含ReSA、独立OSE projector、K=2原型、mixed prototype loss和mixed
  instance loss。
- K=2表示每类独立增强两次；每次包含Joint在线anchor以及Motion/Bone EMA特征，
  共六个embedding。
- 双卡使用标准DDP；关系矩阵、mixed permutation和instance keys跨卡构造。
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
- 每个K=2组内部Joint/Motion/Bone共享对齐mask；两次独立增强使用不同mask。
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

## 3. 当前阻塞点：ReSA正常但LP下降

Stage1 checkpoint的LP约85.86，joint-aware Stage2训练后LP仍下降。最可能原因是
直接照搬AimCLR/ST-GCN的Stage2学习率破坏了MacDiff Transformer backbone：

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

低LR新实验完成后的sweep命令：

```bash
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 BATCH_SIZE=64 LP_ROOT=./output_dir/ntu60_xsub_macdiff_stage2_seed0_lr1e3_lp_sweep bash script_linprobe_stage2_sweep.sh ./output_dir/ntu60_xsub_macdiff_stage2_seed0_lr1e3
```

## 5. 低backbone LR训练命令

`script_pretrain_stage2.sh` 现在支持 `BACKBONE_LR` 和 `HEAD_LR` 环境变量。

```bash
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 MASTER_PORT=10237 BATCH_SIZE=64 BACKBONE_LR=0.001 HEAD_LR=0.25 OUTPUT_DIR=./output_dir/ntu60_xsub_macdiff_stage2_seed0_lr1e3 OMP_NUM_THREADS=1 bash script_pretrain_stage2.sh ./output_dir/ntu60_xsub_macdiff/checkpoint-399.pth
```

## 6. 下一步计划

1. 对现有LR=0.25实验的10/20/.../100 backbone运行LP sweep，得到下降曲线。
2. 用backbone LR `0.001`、head LR `0.25`从Stage1重新训练，不能resume旧Stage2。
3. 对低LR实验运行同样LP sweep。
4. 比较相同epoch：
   - 若旧实验第10轮即大降而低LR保持，确认是catastrophic backbone drift；
   - 若两者均前期高、后期下降，缩短Stage2或选择最佳中间checkpoint；
   - 若低LR仍持续下降，再记录encoder parameter drift、ReSA/OSE backbone梯度范数
     与梯度余弦，判断目标冲突。
5. 在上述对照完成前，不继续修改Sinkhorn温度、loss权重、mask策略或原型结构，
   避免同时改变多个变量。

## 7. 绝对不要再踩的坑

1. 不要把LP 85.86理解成256维全局均值有效；LP2实际用6400维joint-aware特征、
   全750 token以及分类头前BN。
2. 不要再用100% token训练当前K=2六分支Stage2；6GB显卡即使每卡batch16也OOM。
3. 不要通过继续减普通batch解决全输入OOM；60类六原型是固定显存开销。
4. 不要重新加入checkpoint blocks、encoder chunk、手动梯度同步或Stage2 queue。
5. 不要把Stage1路径写回 `ntu60_xsub_ose`；必须是 `ntu60_xsub_macdiff`。
6. `batch_size` 是每GPU大小；双卡每卡64才是全局128。
7. 不要用旧的256维Stage2 checkpoint resume joint-aware版本；projector shape不兼容。
8. mask协议变化或模型结构变化后必须fresh run。
9. 不要只看raw ReSA；同时看 `cluster_entropy(H)` 和 `cluster_kl`。
10. 不要因为ReSA数值正常就断言表征更适合分类；最终必须逐checkpoint LP验证。

## 8. 验证状态

- Python AST静态语法检查通过。
- `git diff --check` 通过，仅有Windows工作区LF/CRLF提示。
- 本机默认Windows Python launcher不可执行，Codex bundled Python没有PyTorch，
  因此没有在本机运行GPU/PyTorch单测。服务器启动脚本会先运行
  `python -m unittest tests.test_stage2`。
