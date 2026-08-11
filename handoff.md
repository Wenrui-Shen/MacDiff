# MacDiff + OSE 跨样本扩散改造交接文档

更新时间：2026-08-10  
工作目录：`D:\program\MacDiff`

## 1. 任务背景与最终目标

本轮工作是在 MacDiff 骨骼动作自监督预训练框架中加入论文 OSE（One-Shot
Exemplars for Class Grounding in Self-Supervised Learning）的原型学习，并用
OSE 给出的类邻域为 MacDiff 提供跨样本 diffusion target。

用户的明确要求是：

1. 参考项目内 `MACDIFF_OSE_CROSS_INSTANCE_DESIGN.md`、论文
   `D:\program\AimCLR\Cui 等 - 2026 - ONE-SHOT EXEMPLARS FOR CLASS GROUNDING IN SELF-SUPERVISED LEARNING.pdf`
   和官方 AimCLR 仓库中的 OSE 实现检查并修改本项目；
2. 做最小修改，不做兼容性分支或旧参数别名；
3. 使用 MAMP 的数据处理流程，raw data 固定放在：
   - `../data/nturgbd_raw`
   - `../data/nturgbd_raw_120`
   - `../data/pku_raw`
4. 所有处理结果放到 `../data/MAMP/`，训练、线性探测和微调配置也从这里读取；
5. 非 exemplar 的真实标签绝不能参与原型、分配、选邻居、门控、loss 或反向传播；
6. 可以使用真实标签做训练决策之后的离线诊断；
7. 最新阶段要求：epoch 0-99 只优化 MacDiff 原生 loss，epoch 100 起才加入
   OSE loss 和跨样本重建。

## 2. 当前采用的方法设计

### 2.1 两阶段训练

专用配置为 `config/ntu60_xsub_joint/pretrain_madiff_ose_peer.yaml`，当前总计
400 epochs，并加入：

```yaml
ose_start_epoch: 100
```

阶段定义如下：

```text
epoch 0-99:
    loss = L_diff + lambda_uniformity * L_uniformity
    self target = 100%
    不编码 exemplar
    不计算 OSE prototype loss
    不构建或使用 neighbor_map
    EMA encoder 和 Queue 仅做无梯度预热

epoch 100:
    使用前 100 epochs 预热的 Queue 构建第一份慢原型和 neighbor_map
    启用 L_OSE = L_align + L_dispersion
    计划 self=0.9, peer=0.1

epoch 100-399:
    peer 概率在 OSE 活跃区间内线性从 0.1 增加到 0.9
    self 概率从 0.9 降到 0.1
    每个 epoch 默认刷新一次慢路由状态

epoch 399:
    计划 self=0.1, peer=0.9
```

这里“前 100 epochs 只用 MacDiff 原生 loss”是优化目标层面的严格保证。
EMA/Queue 在后台预热，但没有梯度，也不改变在线 encoder；保留预热是为了让
epoch 100 能立即建立 neighbor map，而不是到 epoch 101 才首次出现 peer。

学习率阶段与 OSE 阶段独立：当前配置 `warmup_epochs=20`，之后余弦下降，最后
20 epochs 保持 `min_lr`。绝对不要用 `warmup_epochs` 代替
`ose_start_epoch`。

### 2.2 OSE 快状态与慢状态

快状态（每次成功 optimizer step 更新）：

- EMA/momentum encoder，固定 momentum `0.999`；
- 一个所有类别共享的 FIFO Queue，大小 `32768`；
- Queue 每项保存归一化 teacher feature 和不可变 dataset ID；
- DDP 中先 all-gather 各 rank 的 feature/ID，再让每个 rank 入队，保证 Queue
  状态一致。

动态可微原型（OSE 活跃后每个 iteration 重建）：

- 每类一个在线 exemplar embedding；
- 从当前原始 Queue 中按 OSE discriminative score 选邻居；
- score 为
  `alpha * sim(exemplar_c, q) - (1-alpha) * max_other_class_sim`；
- `alpha=0.75`，P1 规则先让每个 Queue slot 只归属一个类，再做每类 Top-K；
- skeleton/AimCLR 参数为 `topk=4`、`tau_s=0.1`、`tau_t=0.04`；
- prototype 由 exemplar 和邻居按与 exemplar 的相似度 softmax 加权得到；
- 不对最终 prototype 再归一化，这是为了匹配官方 AimCLR OSE P1；
- `L_proto = L_align + L_dispersion`；动态 prototype 依赖在线 exemplar，所以
  dispersion 对 encoder 有梯度。

慢状态（默认每个 OSE epoch 边界刷新）：

- 使用去重 Queue 快照，只保留每个 dataset ID 最新 feature；
- 用 EMA exemplar feature 构建冻结 routing prototype；
- 保存每类 P1 Top-K 的不可变 dataset IDs；
- 构建 `neighbor_map[source_id] -> [candidate_peer_ids]`；
- 该 map 在完整 epoch 内冻结，只负责 DataLoader 的 peer 路由，不参与 loss；
- DDP 只广播慢状态 tensor，各 rank 本地重建 Python map，避免 NCCL 上
  `broadcast_object_list` 的风险。

### 2.3 跨样本 diffusion 路由

OSE 活跃后每个样本实际使用 peer 的条件是：

```python
use_peer = requested_peer & has_peer & confident
```

- `requested_peer` 来自当前 epoch 的线性计划概率；
- `has_peer` 表示冻结 neighbor map 中存在候选；
- `confident` 表示当前 EMA teacher 对动态 prototype 的最大类别概率不低于
  `ose_assignment_confidence=0.8`；
- 任一条件失败都回退到 self reconstruction；
- self 使用 MacDiff 原始 global-local condition；
- peer 使用 source 的 global-only condition，避免把 source 的局部时间位置强行
  对齐到另一个动作样本；
- diffusion 仍保持原来的 mask、noise schedule、prediction target 和
  reconstruction loss。

## 3. 已完成的代码修改

### 3.1 OSE 核心

新增 `model/ose_memory.py`：

- 共享 FIFO Queue；
- Queue 去重快照；
- OSE discriminative neighbor score；
- P1 互斥邻居所有权；
- 动态 prototype 与 paper-style soft teacher alignment；
- 有梯度的 dispersion；
- epoch 冻结 routing prototype 和 neighbor IDs；
- 基于 dataset ID 的 neighbor map；
- Queue fill、prototype cosine、snapshot version 等诊断。

修改 `model/transformer_macdiff.py`：

- 新增 EMA `MomentumSkeletonEncoder`；
- 新增 OSE 初始化、EMA 更新、入队、exemplar 安装和慢状态刷新接口；
- forward 改为 source/source_aug/peer 路由形式；
- epoch 0-99 跳过 exemplar encoder 和 prototype loss；
- epoch 100 后计算动态 OSE loss 和 peer 路由；
- peer 计划概率在 `[ose_start_epoch, epochs-1]` 内重新从 0.1 插值到 0.9；
- OSE 的低温 logits/prototype 路径强制 fp32，避免 AMP 数值问题；
- 返回实际 `use_peer` mask，仅供训练引擎做标签诊断；
- 新增 `ose_active`、各 loss、计划/实际 peer 比例和置信度日志。

修改 `engine_pretrain.py`：

- 支持新的 feeder/model 返回结构；
- 正确处理 gradient accumulation 最后一个不足窗口；
- 只在成功 optimizer step 后更新 EMA 和 Queue；
- accumulation window 内所有 microbatch teacher feature 都会入队，不再只取最后
  一个 microbatch；
- DDP all-gather feature 与 dataset ID；
- 修复曾经存在的 DDP 死锁风险：所有 rank 都执行 aux all-reduce，不能只在
  rank 0/TensorBoard 分支内做 collective；
- 加入实际跨样本 target 的离线标签准确率计数。

修改 `feeder/feeder_ntu.py`：

- 抽出 `_load_processed(index)`，避免递归调用 `__getitem__`；
- 新增 `FeederOSE`；
- exemplar 从 unlabeled sampler 和 Queue 中排除；
- 使用不可变的原始 dataset ID；
- 每次从冻结 candidate pool 随机选一个 peer；
- 无候选时返回 source copy 和 `has_peer=False`；
- source/peer 真实标签只作为离线诊断输出。

修改 `main_pretrain.py`：

- 严格加载一类一个 exemplar 的 JSON；
- 要求 exemplar mapping 完整覆盖数据集所有类别、索引互异且 exemplar 标签正确；
- 加入全部 OSE 参数和启动检查；
- 加入 `ose_start_epoch`，要求在 `[0, epochs)`；
- epoch 100 前不刷新慢状态；epoch 100 或 resume 到 OSE 阶段的首个 epoch 必定
  刷新；之后按 `ose_refresh_interval` 刷新；
- DDP 使用 `broadcast_buffers=False`，避免每次 forward 广播整个大 Queue；
- refresh 时广播慢 tensor，并本地重建 neighbor map；
- 输出/记录离线邻居正确率。

### 3.2 真实标签离线检查

已经加入两个指标，二者都不参与训练决策：

1. `offline_neighbor_label_accuracy`
   - 对冻结 `neighbor_map` 的全部 source-to-candidate 边计数；
   - 正确条件是 source 和 candidate 的真实类别相同；
   - 同时记录 `offline_neighbor_edge_count`。
2. `offline_cross_reconstruction_label_accuracy`
   - 只统计实际 `use_peer=True` 的重建；
   - 正确条件是 source 和最终选中 peer 的真实类别相同；
   - DDP 下先 all-reduce 正确数和总数，再计算准确率；
   - 同时记录 `offline_cross_reconstruction_count`。

标签只在 feeder 和训练引擎中出现，模型 forward 没有 label 参数，且标签不会进入
prototype、assignment、neighbor selection、confidence gate、loss 或 backward。
分母为 0 时 accuracy 记 0，同时 count 也为 0，不能把这种 0 当成有效测量。

注意：“跨样本重建标签正确率”目前衡量的是被选作 reconstruction target 的 peer
是否与 source 同类，不是 decoder 输出经过分类器后的动作识别准确率。项目当前没有
冻结的动作分类器，若用户将来要求后者，需要额外设计离线 classifier evaluation，
绝不能偷换概念。

### 3.3 数据处理与路径

数据脚本已根据 MAMP 补充/修改：

- `data/ntu/*`
- `data/ntu120/*`
- `data/pku_v1/pku_gendata.py`
- `data/pku_v2/pku_gendata.py`

README 的 Data Preparation 已写明目录和执行顺序。预期输入：

```text
../data/nturgbd_raw/
../data/nturgbd_raw_120/
../data/pku_raw/v1/
../data/pku_raw/v2/
```

预期输出：

```text
../data/MAMP/ntu/
../data/MAMP/ntu120/
../data/MAMP/pku_v1/
../data/MAMP/pku_v2/
```

所有已检查的 pretrain、linprobe、finetune YAML 已切到 `../data/MAMP/...`。

README 已根据 MAMP 的流程补充命令。数据脚本来自/参考
`https://github.com/maoyunyao/MAMP`，不要再恢复原项目旧 data path。

### 3.4 配置和入口

新增：

- `config/ntu60_xsub_joint/pretrain_madiff_ose_peer.yaml`
- `config/exemplar_indices.example.json`

example JSON 只有 3 个演示条目，不能直接训练 NTU60。真实训练文件应为：

```text
config/ntu60_xsub_joint/exemplar_indices.json
```

且必须有完整 60 类映射。`script_pretrain_madiff.sh` 已改为使用专用 OSE 配置。

按用户“不要做兼容性修改”的要求，没有保留旧 forward 签名、旧参数别名或同时支持
旧 feeder 的 fallback。当前 `main_pretrain.py` 明确要求 `FeederOSE`。不要为了让旧
pretrain YAML 继续走同一入口而悄悄加 try/except 兼容分支；若以后要保留 baseline，
应由用户明确决定独立入口或独立文件。

## 4. 论文与官方实现核对结果

官方 AimCLR 仓库检查时使用的 commit：

```text
9cf5dfcfa39a4f3a40a3bf9081e4eccb8ff2c33e
```

已经确认：

- 官方 `net/ose_aimclr.py` 使用在线 exemplar 构建动态 prototype；
- teacher prototype detach；
- OSE loss 是 alignment + dispersion；
- 一个共享 Queue；
- P1 是先做 Queue slot 的类别互斥所有权，再做类内 Top-K；
- skeleton 配置是 Queue 32768、momentum 0.999、Top-K 4、alpha 0.75、
  tau_s 0.1、tau_t 0.04、lambda 1；
- 官方 AimCLR skeleton 实验后半程才启用 OSE。当前项目根据用户最新要求采用固定
  `ose_start_epoch=100`，不是官方的 epoch 150；
- 论文图像实验的 Queue/Top-K/momentum schedule 与 skeleton 配置不同，不要把
  ImageNet/CIFAR 数值误抄到本项目；
- 没有实现论文的 `L_mix`。MacDiff 的 cross-instance diffusion 已承担跨样本正则，
  再加 `L_mix` 会超出“最小修改”并混淆实验归因。

## 5. 当前卡点与未完成事项

当前没有逻辑设计上的硬阻塞，但缺少运行环境和真实数据，因此尚未完成训练级验证：

1. 当前可用 Python 环境没有安装 PyTorch，也没有项目训练依赖；
2. `../data/MAMP/ntu/NTU60_XSub.npz` 在检查时不存在；
3. 完整的 NTU60 exemplar JSON 尚未提供/生成；
4. 因此没有跑过单 GPU 两阶段 smoke test；
5. 没有跑过多 GPU/NCCL smoke test；
6. 所有改动仍在 working tree，尚未 stage、commit 或 push；
7. `handoff.md`、设计文档、OSE 配置和 `model/ose_memory.py` 等目前是 untracked
   文件，接手后不要误删。

已经完成的验证只有：

- `feeder/feeder_ntu.py`
- `engine_pretrain.py`
- `main_pretrain.py`
- `model/transformer_macdiff.py`
- `model/ose_memory.py`

以上文件的 Python AST 解析通过；

- 阶段边界检查通过：epoch 99 为 `(self=1, peer=0)`，epoch 100 为
  `(self=0.9, peer=0.1)`，epoch 399 为 `(self=0.1, peer=0.9)`；
- 静态标签隔离检查通过，model 文件中没有 `source_labels`/`peer_labels`；
- `git diff --check` 通过，只有 Windows CRLF 提示，没有 whitespace error；
- 因缺 PyTorch，以上不等于运行正确。

## 6. 下一步计划（按优先级）

### P0：准备最小可运行输入

1. 按 README 运行 MAMP 数据处理流程，至少生成
   `../data/MAMP/ntu/NTU60_XSub.npz`；
2. 从训练 split 生成完整 60 类、每类一个 immutable dataset index 的
   `config/ntu60_xsub_joint/exemplar_indices.json`；
3. 固定 exemplar 选择随机种子并保存映射，不要每次运行重新随机选；
4. 安装与本项目匹配的 PyTorch/CUDA 和 requirements。

### P1：单 GPU smoke test

建议先临时用小数据/短 epoch 配置验证边界，不要直接烧完整 400 epochs：

1. 验证 epoch `< ose_start_epoch`：
   - `ose_active=0`；
   - `loss_ose_* = 0`；
   - `p_peer_planned=0`；
   - `peer_fraction_effective=0`；
   - Queue fill ratio 持续增加；
   - offline neighbor/cross count 都为 0；
2. 验证 epoch `== ose_start_epoch`：
   - 启动时立即生成慢 snapshot 和 neighbor map；
   - `ose_active=1`；
   - `p_peer_planned=0.1`；
   - OSE loss 有限且有梯度；
   - neighbor label accuracy/count 开始出现；
3. 验证一个 optimizer accumulation 尾窗口，确保 feature/ID 数量一致；
4. 检查 Queue wraparound 和 dataset ID 对齐；
5. 检查 loss、prototype、confidence 是否出现 NaN/Inf。

短跑时可把 `epochs` 和 `ose_start_epoch` 同比例缩小，但不要提交这种临时配置。

### P2：多 GPU smoke test

至少跑 2 GPU、2 个 epoch 跨过 OSE 启动边界，检查：

- 所有 rank Queue pointer/count/IDs 完全一致；
- 不发生 collective 次序不一致或死锁；
- slow prototype tensor 广播后一致；
- 各 rank 本地构建的 neighbor map 长度和内容一致；
- 离线 cross accuracy 使用全局正确数/总数，而不是 rank 比例平均；
- `broadcast_buffers=False` 保持不变。

### P3：实验与风险诊断

先观察以下日志，再决定是否调整阈值或 Top-K：

- `queue_fill_ratio`
- `prototype_cosine_mean/max`
- `ose_confidence`
- `above_confidence_fraction`
- `neighbor_map_entries`
- `peer_fraction_effective`
- `offline_neighbor_label_accuracy`
- `offline_cross_reconstruction_label_accuracy`
- 对应的两个 count

如果 neighbor 正确率高但实际 cross 正确率明显低，优先检查“快 teacher 类别”和
“慢 neighbor map 路由类别”是否发生漂移，不能直接怪 diffusion decoder。

## 7. 已知风险和建议但尚未实现的改进

以下是明确知道、但本轮没有改的内容。不要在没有实验数据前擅自一次性全加：

1. **快慢类别一致性缺失**
   - 当前慢 map 根据刷新时的 class assignment 选择 candidate pool；
   - forward 只检查当前 teacher confidence，没有检查当前 teacher class 是否仍与慢
     route class 一致；
   - 可能出现慢状态指向 A 类、当前 teacher 高置信预测为 B 类，却仍使用 A 类 peer；
   - 若离线 cross accuracy 暴露这个问题，最小修复是让 neighbor map 同时携带
     `routing_class`，feeder 返回它，forward 加
     `teacher_assignment == routing_class`；
   - 这项改进尚未实现。
2. **双重置信度过滤可能过严**
   - 构建慢 map 时已经用 `confidence >= 0.8`；
   - forward 又用当前 teacher confidence `>=0.8`；
   - `tau_t=0.04` 下 softmax confidence 未必校准；
   - 不要直接删除所有门控。先看 coverage、effective peer fraction 和真实标签离线
     指标，再决定保留哪一层或调整阈值。
3. **Top-K=4 可能太小**
   - 对完整骨骼序列重建而言，每类只有最多 4 个 peer，可能产生固定模板记忆；
   - 后续可做 K=4/8/16 消融，不要未经实验就改默认值。
4. **每步编码所有 exemplar 的开销**
   - NTU60 每步额外编码 60 个样本，NTU120 是 120 个；
   - 当前这么做是为了保持动态 prototype 对在线 encoder 可微；
   - 不要随手缓存/detach 在线 exemplar，否则 dispersion 会再次失去梯度。
5. **resume 随机性不完全可复现**
   - checkpoint 恢复 Queue/EMA/slow tensor，但未专门保存 Python/NumPy/DataLoader
     worker 的精确 RNG 进度；
   - resume 后随机 peer 序列可能与不中断运行不同。
6. **完整输出语义未评估**
   - 目前真实标签诊断只评价 target selection；
   - 若要评价 decoder 重建结果是否保持动作类别，需要冻结 classifier 或独立评估
     协议，不能复用当前 target label accuracy 的名字冒充。

## 8. 绝对不要再踩的坑

1. **不要把冻结慢原型的 dispersion 当成可训练 loss。**
   冻结 prototype 没有梯度，只会改变 loss 数值。当前 dispersion 必须基于每步由在线
   exemplar 重建的动态 prototype；teacher side detach，student prototype side保持梯度。
2. **不要让非 exemplar 真实标签进入训练。**
   标签只能在路由完成后用于离线计数，绝不能用于 prototype、pseudo-label、neighbor
   pool、confidence gate、loss 或采样决策。
3. **不要用 Queue slot 当样本身份。**
   FIFO 会覆盖和绕回，neighbor map 必须存 immutable dataset ID。
4. **不要每 iteration 重建 Python neighbor map。**
   动态 prototype loss可以每步重建；routing prototype/map 必须慢刷新并在一个 epoch
   内冻结，否则 worker/DDP 状态不稳定且开销很大。
5. **不要在 NCCL 中广播 Python object map。**
   广播 tensor 慢状态，然后每个 rank 本地确定性重建 map。
6. **不要把 collective 放进 rank-0-only 或 TensorBoard-only 分支。**
   所有 rank 必须以相同顺序执行 all-reduce/all-gather，否则会死锁。
7. **不要在 gradient accumulation 时只入队最后一个 microbatch。**
   必须缓存完整 accumulation window，成功 optimizer step 后一起入队；AMP overflow
   时不能更新 EMA/Queue。
8. **不要让 DDP 默认每次 forward 广播大 Queue buffer。**
   当前必须保留 `broadcast_buffers=False`，Queue 已通过显式 all-gather 同步。
9. **不要把 `warmup_epochs` 当 OSE 启动 epoch。**
   LR warmup 当前是 20；OSE 启动是独立的 `ose_start_epoch=100`。
10. **不要让 epoch 100 延迟到 epoch 101 才有 peer。**
    前 100 epochs 保留无梯度 EMA/Queue 预热，epoch 100 边界先 refresh 再训练。
11. **不要沿用旧的全程 peer 插值。**
    当前 epoch 0-99 强制 `(self=1, peer=0)`；插值分母是
    `epochs - ose_start_epoch - 1`，epoch 100 必须精确为 peer 0.1。
12. **不要重新加兼容性参数和 try/except fallback。**
    用户明确要求不做兼容性修改。发现旧配置不能走新入口时应明确报告，而不是静默走
    另一套 loss/forward。
13. **不要把官方图像实验参数和 skeleton 参数混用。**
    当前默认遵循 AimCLR skeleton：Queue 32768、Top-K 4、momentum 0.999、
    tau_s 0.1、tau_t 0.04。
14. **不要在没有真实数据和 PyTorch 的情况下宣称训练通过。**
    目前只有静态检查，运行验证仍是下一会话最重要的工作。
15. **不要清理或覆盖当前 dirty worktree。**
    这些修改都是本任务成果，包含大量 tracked 修改和若干 untracked 新文件。禁止
    `git reset --hard`、`git checkout --` 或批量删除。

## 8.1 2026-08-11 显存测试补充

在交接文档初稿之后又加入了以下功能：

- `--ose_start_epoch N` 已有参数继续保留，可从命令行覆盖 YAML；
- 新增 `--ose_exemplar_checkpoint`，默认关闭，只 checkpoint 在线 exemplar
  encoder 的 8 个 Transformer blocks；
- 项目固定 PyTorch 1.8.1，因此使用 reentrant checkpoint；block 输入通过 patch
  embedding 后已经 `requires_grad=True`，不要在进入 checkpoint 前 detach；
- DDP 改为 `find_unused_parameters=False`。所有在线 encoder/decoder 参数每步都会
  使用，这也是 PyTorch 1.8 reentrant checkpoint 在 DDP 下的重要前提；
- 新增每 epoch 的 `cuda_peak_allocated_mb` 和 `cuda_peak_reserved_mb`，DDP 记录
  所有 rank 的最大值；
- 新增诊断参数 `--max_train_steps`，0 表示完整 epoch；
- 新增 `script_profile_ose_memory.sh`，单 GPU 顺序测试 checkpoint 关闭/开启；
  默认 `ose_start_epoch=0`、2 epochs、每 epoch 20 steps。第二个 epoch 会覆盖
  Queue 非空和 neighbor map 刷新后的路径；
- 可通过环境变量 `PROFILE_EPOCHS`、`PROFILE_STEPS`、`PROFILE_BATCH_SIZE`、
  `PROFILE_NUM_WORKERS` 和 `CUDA_VISIBLE_DEVICES` 调整测试；
- checkpoint 不减少 EMA/Queue/相似度矩阵的常驻或临时显存，只减少 exemplar
  encoder 反向图中保存的 block 内部 activation，并会增加一次 backward 重算。
- 新增 `generate_ose_exemplar_indices.py`。它不依赖 PyTorch，默认从
  `../data/MAMP/ntu/NTU60_XSub.npz` 的 `y_train` 以 seed 0 为每类选择一个不可变
  训练索引，并写入 `config/ntu60_xsub_joint/exemplar_indices.json`。脚本严格检查
  one-hot 标签、x/y 样本数、连续类 ID、索引唯一性和索引标签一致性。

## 9. 当前工作树提示

本轮没有创建 commit。`git status --short` 显示 README、数据脚本、多个配置、训练
框架文件均已修改；以下关键文件目前为 untracked：

```text
MACDIFF_OSE_CROSS_INSTANCE_DESIGN.md
config/exemplar_indices.example.json
config/ntu60_xsub_joint/pretrain_madiff_ose_peer.yaml
model/ose_memory.py
handoff.md
```

接手后第一步应先运行：

```powershell
git status --short
git diff --check
```

随后只检查和补充，不要回滚任何现有修改。若需要提交，先逐文件确认 scope，再由用户
明确授权/要求 commit 或 push。
