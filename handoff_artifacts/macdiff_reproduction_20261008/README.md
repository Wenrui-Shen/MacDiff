# MacDiff 论文、官方实现与当前复现审计

日期：2026-10-08。目标：NTU60 XSub、joint 单流、冻结 encoder 的 linear probe（LP）。本报告仅完成静态检查、既有日志复核和配置准备，没有产生新的训练成绩。

官方固定提交：[692888f1e2a4227216511aac6733dff31b587213](https://github.com/LehongWu/MacDiff/commit/692888f1e2a4227216511aac6733dff31b587213)，2025-07-06。本地 HEAD：38016d82ef40db3f3f8e7779a6be49725ea10892；审查对象是当前工作区，包含已有扩展代码。论文来源为用户的 D:/program/paper/macdiff.pdf，已提取全文并渲染检查第 7、9、10、14 页；补充材料取自[作者项目页 PDF](https://lehongwu.github.io/ECCV24MacDiff/macdiff-supp.pdf)。

:codex-file-citation{path="D:/program/paper/macdiff.pdf" purpose="source"}

## 1. 最重要的解释：86.4 与默认 5 层之间有实验口径疑点

- 主论文第 10 页表 1：NTU60 XSub LP 为 **86.4%**，joint 单流。
- 第 10 页默认结构：encoder 8 层、decoder 5 层。
- 第 14 页表 10：decoder 2/3/4/5 层分别为 **84.8/86.4/86.0/85.9%**；正文称 3 层表现最好，但考虑生成能力，默认用 5 层。
- 官方基础 YAML 和当前基础 YAML 都是 decoder_depth=5。
- 历史 T01 约 85.86% 与论文 5 层的 85.9% 接近；然而 T01 的实际 checkpoint args、精确 LP 日志和服务器代码身份尚缺，不能用当前默认值补填其历史参数。

**合理假设是目标成绩与默认结构的口径需要进一步确认。不能确认主表 86.4 必定来自 3 层，也不能直接宣布历史 T01 已完整复现。** 最有依据的结构消融是 5 对 3 层，其他条件完全一致。decoder_depth 的变化必须重新预训练；LP 不加载 decoder，不能靠改 LP 配置让现有 encoder 变成“3 层训练结果”。

## 2. 论文与官方代码逐项对照

| 项目 | 论文描述 | 官方实际设置 | 判断及优先级 |
|---|---|---|---|
| PT epoch | 第 10 页：500 | NTU60 XSub YAML 和脚本：400 | 明确不一致；高优先级 |
| PT random rotation | 第 9 页：random crop + rotation + Gaussian noise | XSub PT 的 random_rot 被注释；feeder 默认 False | 明确不一致；高优先级。LP 训练旋转是 True |
| PT batch | 第 10 页：4 卡，总 batch 128 | 官方 4×32×accum1=128 | 官方脚本符合。当前 native 脚本默认 2×32×1=64 |
| PT lr / min_lr | 第 10 页：1e-3 → 1e-5 | YAML min_lr=5e-4，但脚本覆写 1e-5 | 按脚本符合；直接按 YAML 启动会不同 |
| encoder / embedding | 8 层、256 维、8 heads、MLP 1024 | 相同 | 未发现差异 |
| temporal input / patch | 300 → crop/resize 120，patch 长 4 | window 120、t_patch_size 4 | 相同；细节另见数据审计 |
| mask | 第 14 页表 7：random、90% 最优 | mask_ratio=0.9、motion_aware_tau=-1 | 相同；750 token 保留 75 |
| diffusion | 第 7 页：epsilon、1000 步、gamma_t=1 | noise、1000、默认权重为 1 | 相同 |
| noise schedule | 第 14 页表 8：inverse cosine、tau=1 | ['inverse_cosine', 1] | 相同；和字符串 inverse_cosine 数值一致 |
| conditioning | 第 6–7 页：AdaLN、global-local、drop condition 0.1 | 相同 | 未发现语义差异 |
| data normalization | 第 7 页式 5：用训练集 mean/std | 写死的 A；有 #new B 但未启用 | 无统计生成脚本或明确统计范围，不能核实 A/B 来源 |
| diffusion objective | 第 7 页式 6：完整 epsilon 向量误差范数 | 仅 masked patch 的 MSE | 论文未明示的实现选择；可消融，不能据此断言 bug |
| uniformity penalty | 第 7 页用 global-local 设计讨论 token uniformity；未给额外惩罚项公式 | 总损失 = diffusion MSE + 0.02×token_uniformity_loss | 论文未明示的实现选择；可消融 |
| LP | 第 10 页：冻结 encoder，SGD、100 epoch、lr=0.1 | 冻结 encoder，SGD(momentum=.9, wd=0)，100 epoch、lr=.1 | 主协议相同；head、BN、batch 和调度未在论文中充分说明 |

官方代码来源：

- [NTU60 XSub PT YAML](https://github.com/LehongWu/MacDiff/blob/692888f1e2a4227216511aac6733dff31b587213/config/ntu60_xsub_joint/pretrain_madiff.yaml)、[PT 启动脚本](https://github.com/LehongWu/MacDiff/blob/692888f1e2a4227216511aac6733dff31b587213/script_pretrain_madiff.sh)
- [LP YAML](https://github.com/LehongWu/MacDiff/blob/692888f1e2a4227216511aac6733dff31b587213/config/ntu60_xsub_joint/linprobe_madiff.yaml)、[LP 入口](https://github.com/LehongWu/MacDiff/blob/692888f1e2a4227216511aac6733dff31b587213/main_linprobe.py)
- [PT 损失与 forward](https://github.com/LehongWu/MacDiff/blob/692888f1e2a4227216511aac6733dff31b587213/model/transformer_macdiff.py#L537)、[下游 head](https://github.com/LehongWu/MacDiff/blob/692888f1e2a4227216511aac6733dff31b587213/model/transformer_downstream.py#L85)

## 3. 当前配置是否符合官方

全部 18 份对应的 PT/LP/FT 基础 YAML 已逐项读取和对照：仅 30 个 data_path 字段迁移为 ../data/MAMP/...。统一路径、换行和尾部空白后，连注释也一致。完整协议/归一化矩阵见[此前审计](../upstream_config_audit_20261008.md)。该 18 份结论不包含本目录新准备的消融文件。

模型与运行过程另有本地扩展，必须核实实际入口。当前原生模型 model.transformer_macdiff.Transformer 在 enable_ose=False 时进入 forward_macdiff；原生模型不使用 text cache。下游模型全文与官方一致。静态 AST 比较忽略 docstring 后，初始化权重、随机 mask、patchify、forward_loss、时间步 sampler 更新方法均一致；forward_encoder 增加可选 activation checkpoint，原生调用默认关闭。global-local 条件构建与官方含义相同。未进行端到端 Tensor 数值对照。

[verification.json](verification.json) 保存上述静态比较、beta 数值对照、batch 推算和 T13 原始日志复核结果。噪声 schedule 的两个函数 AST 一致，字符串 inverse_cosine 与列表 ['inverse_cosine',1] 的 1000 步 beta 最大绝对差为 0。

### one_person

- PT 默认 one_person=True；基础 YAML 的 #one_person: False 是注释。实际固定截取 person0，绝非随机选择人。
- LP 无 one_person 构造参数；两个人物槽位各自经过共享 encoder，再在 head 中聚合。空人物未被过滤。
- linprobe2 对 M 与 T 取均值，保留 25 个关节，再展平为 25×256=6400 维进入线性分类器。
- 原生 PT 改为 one_person=False 可以做消融，但 uniformity 项仍只取 person0，diffusion MSE 则使用两个人物。应记录此不对称，不能称为“均衡双人训练”。
- 下游只用 person0 的消融需要增加明确的输入选择实现；直接在 LP model_args 写 one_person 会因不支持该参数而失败。

### #new input_mean / input_var

NTU60 XSub 基础 PT/LP 都启用 A；B 的两行是注释：

| 组 | mean | var |
|---|---|---|
| A | [-0.0058,-0.1333,-0.0246] | [0.0206,0.0805,0.0218] |
| B | [-0.0024,-0.2132,-0.0446] | [0.0525,0.1527,0.0513] |

feeder normalization=False 不会关闭模型内部标准化；PT 与 LP 都执行 (x-mean)/sqrt(var)。这些统计量不是 state_dict buffer，加载 checkpoint 不会自动带入。因此 A/B 的完整比较必须配套 PT 和 LP；拿 A 的 checkpoint 仅切 B 的 LP 是输入分布错配诊断，不能验证 B 预训练收益。

B 相比 A 会改变平移和幅度。扣除各组均值后，同一坐标偏差在 B 空间中的幅度约为 A 的 [0.6264,0.7261,0.6519]。diffusion 注入的 epsilon 仍为单位 Gaussian，所以有效信噪关系也改变；不能推断“B 方差更大所以直接降低 epsilon MSE”。小噪声由 feeder 在标准化前添加，故归一化也影响 encoder 小噪声的有效幅度。#new 标签不能证明 B 更新、更准确或属于某个特定数据集。

## 4. LP 运行细节：先复用 checkpoint 排查

当前 main_linprobe.py:238–247 给 6400 维输入加 BatchNorm1d(affine=False, eps=1e-6)，然后冻结 encoder，仅训练 head。engine_linprobe.py:33–35 用 model.eval() 与 head.train(True)，行为符合冻结评估。

1. **BN 用每卡的即时 batch。** 没有 SyncBatchNorm。官方 4 卡×64，与 T13 的 2 卡×128，虽同为全局 batch 256，BN 统计范围不同。梯度累积不合并 BN 的即时统计；DDP 的 buffer 广播也不等同 SyncBN。
2. **lr=.1 是绝对 LR。** YAML 已设 lr，只有 lr=None 时才按 blr×global_batch/256 换算。因此改 batch 不会自动缩放实际 LR。
3. **最后 10 轮 lr=0。** 默认 min_lr_epochs=10，warmup 被 YAML 改为 0；余弦在约第 90 轮降为 0，之后保持。BN 仍更新 running statistics，因此最后 10 轮准确率可变化。T13 的 epoch92 best 不能解释为该轮 FC 又获得新的梯度更新。
4. 官方 dist_eval 在 NTU60 XSub 的 16487 个测试样本上补 1 个重复样本（2 或 4 卡）。理论最大准确率差约 0.0061 个百分点，不能解释约 0.5 个百分点。
5. 2 卡×64×accum2 可保持局部 BN batch64 与全局梯度 batch256，但不与 4 卡运行逐位等价。40091 个训练样本下每卡 313 个 microbatch，LP 当前代码不提交末尾单独剩下的累积 microbatch；其 BN 前向仍发生。必须固定此运行方式再比较各 PT 消融。

推荐在同一份原生 A checkpoint 上先做有限的 LP 对照：

| 运行 | 每卡 batch / 双卡累积 | 全局 batch | LR | 目的 |
|---|---|---|---|---|
| L0 | 64 / 2 | 256 | .1 | 两卡参考 LP；有上述末尾累积差异 |
| L1 | 128 / 1 | 256 | .1 | 与 L0 比，检查 microbatch/BN 组织敏感性 |
| L2 | 64 / 1 | 128 | .1 | 与 L0 比，检查梯度 batch 和更新次数影响 |
| L3/L4（按需） | 沿用固定 batch | 固定 | .05 / .2 | 检查线性头 LR 是否影响复现差距 |
| L5（按需） | 沿用固定 batch | 固定 | .1，min_lr_epochs=0 | 检查最后 10 轮 lr=0 的日程选择 |

L1 同时改变 BN batch、microbatch 分组和 BN 更新频率，不是纯 BN 消融。若先找到重要差异，再为纯 BN 研究增加独立实现选项。不要同时扫描 head 类型、归一化、人物数量并从 test best 反推因果。所有比较使用同一 checkpoint 身份、评估 crop 和 seed，并记录 best、last、后期均值；单次运行的末 20 个 epoch 不是随机种子置信区间。

## 5. T01 / T12 / T13 现有结果的可比性

详见 [EXPERIMENT_RESULTS.md](../../EXPERIMENT_RESULTS.md) 与 [NormB 日志分析](../../tools/vlm_pilot/NOSHARE_NORMB_LOG_ANALYSIS.md)。

- T01：约 85.86%，原版历史摘要；实际 args 不齐，不能把当前 decoder=5、batch 或 min_lr 填成已知历史事实。
- T12：86.196021%，文本 no-share 方法，属于扩展方法成绩，不能当作原生 MacDiff 的复现结果。
- T13：85.183164% @LP92；原始 JSON PT0–399、LP0–99 均完整。PT 后期 train_lr 实际为 .0005；LP90–99 实际为 0；可训练参数 384060，符合 6400×60+60 的 linprobe2 head。
- T13 保存命令为 PT 两卡×128×accum1，总 batch256、min_lr5e-4；LP 两卡×128×accum1，总 batch256。batch 身份依赖保存命令，日志没有完整 args 或服务器 SHA。
- 按 40091 个训练样本、DistributedSampler、drop_last 推算：官方 PT 每轮 313 次优化器更新；T13 每轮 156 次。400 轮分别约 125200 与 62400 次。这是训练预算推算，不是日志实测的 successful AMP step 数。
- T13 与历史 T01、文本组同时存在模型/统计量/batch/LR 等差异；85.18 不能单独判定 B 有害。缺失控制组是同 T13 协议的原生 A。

## 6. 数据预处理：哪些相同，哪些需要检查数据实物

| 数据集 | 当前离线流程 | 与复现相关的细节 |
|---|---|---|
| NTU60 | 读 skeleton→按 bodyID 整理、去空帧/噪声人物→合成最多两个人物→clip 平移→补齐时间→按 XSub/XView 输出 NPZ | 平移原点为 person0 首个有效帧 joint2；单人第二槽位置零。定义的 frame_translation 尺度函数未在主流程调用 |
| NTU120 | 相同的读/去噪/平移/补齐流程，XSub/XSet | align_frames 对单人把 person0 复制到 person1，区别于 NTU60 的空槽位；LP 会受此人物槽位规则影响 |
| PKU v1/v2 | 按官方 split 文件取视频→按标签 start/end 切动作→剔除求和为零实例→截前 300 帧/补零→51 类 onehot NPZ | 当前脚本没有 NTU 式人物去噪或 clip 原点平移；start:end 为 Python 半开切片，需要对原始标签的帧编号约定另核验 |

NTU 核心读骨架和去噪逻辑与官方继承的 MAMP 流程一致，本地主要迁移路径/输出目录。在线共用 feeder.Feeder，将 NPZ 的 [N,T,150] 重排为 [N,3,T,25,2]。训练随机取有效长度的 50%–100%（最少约 64 帧，受实际有效帧数上限约束），再 resize120；验证中心取 95%，无随机旋转。标准统计文件下 NTU60 XSub 为 train40091/test16487。

必须对实际用于训练的 NPZ 核查：

- 样本数、label 范围 0–59、onehot 行和、split 身份、文件 SHA256、NaN/Inf、人物槽位空/复制比例。
- 当前 feeder 用坐标求和 !=0 判有效帧，且按连续有效前缀裁剪；检测是否有内部空洞、求和抵消的非空帧。
- 官方继承的 get_raw_skes_data.py:74–75 用“该 body 上次位置+1”记再次出现的位置，未直接用当前全局有效帧号。当一个 body 暂缺而另一个仍在时，有压缩该 body 时间轴的可能。应先量化此情形；直接修复再和官方比较，会同时改变数据协议。
- 对 training split 分别统计 person0/两槽位、含零/排除空槽位、含 padding/有效帧、resize120 之后的 mean/var。与 A/B 比较时必须写明统计范围，不用“统计看起来接近”代替数据版本证据。
- PKU 的 51 类代码与论文文字 52 categories 之间存在口径差别；需结合标签文件核验，不能因此给 NTU60 LP 定因。

本地默认数据路径 D:/program/data/MAMP/ntu/NTU60_XSub.npz 当前不存在，也没有本次复现 checkpoint 可直接检查。因此没有跑这些数据实物检查或训练，无法最终排除服务器数据版本差异。

另有两份官方 NTU120 FT num_classes=60 的问题，及 PKU/NTU120 的跨阶段归一化差异；它们已列在前一份审计，**不是这次 NTU60 XSub LP 的直接原因**。



## 7. 已准备的单因素消融与执行顺序

[manifest.json](manifest.json) 给出每个配置相对控制组的生效字段差异、sha256、配套 LP 和运行 profile。所有 YAML 在 configs/ 内，当前原有 config/ 和模型没有修改。

| 配置 ID | 相对 R0 的改变 | 配套 LP | 建议顺序/成本 |
|---|---|---|---|
| r0_pretrain | 原生控制：A、one_person=True、depth5、rotationFalse、400 epoch、PT lr1e-3/min_lr1e-5、有效 batch128 | lp_a | 先有可信控制；可复用身份完全确认的等价 checkpoint |
| p1_decoder3_pretrain | decoder_depth 5→3 | lp_a | 优先；需要新 PT |
| p2_epochs500_pretrain | 400→500，总日程从初始化按 500 轮规划 | lp_a | 优先；需要新 PT |
| p3_rotation_pretrain | PT random_rot False→True | lp_a | 优先；需要新 PT |
| p4_uniformity0_pretrain | lambda_loss_uni .02→0 | lp_a | 次优先；需要新 PT |
| p5_norm_b_pretrain | mean/var A→B，作为同一归一化因素组 | lp_b | 次优先；PT/LP 必须配对 |
| p6_two_person_pretrain | one_person True→False；uniformity 仍仅 person0 | lp_a | 后续；需要新 PT，算量增大 |
| p7_paper_combination_pretrain | depth3+500轮+rotationTrue | lp_a | 可选组合，不能用于单因素归因 |
| c1_t13_matched_a_pretrain | 用 A，固定 T13 保存命令中的 PT/LP batch256、PT min_lr5e-4 | lp_a，但 LP 每卡128 | 若要判断已有 T13 的 B 作用，应补此控制组 |

推荐顺序：

1. 先核验 checkpoint/data/args 身份；如有可用原生 checkpoint，复用其 encoder 做 L0/L1/L2 的 LP 检查。不能用文本 T12 的 checkpoint 代替原生控制组。
2. 固定一个 LP 协议，取得 R0，再做 P1（depth3）。
3. 在 R0 上独立做 P2（500轮）和 P3（rotation）。若预算只够少量实验，先完成这些直接有论文依据的因素。
4. 再做 P4、P5；B 的现有 T13 若值得继续查，补 C1 比较能直接利用已完成结果。
5. 最后 P6 或加入全位置 MSE。全位置 MSE 没有现成 YAML 开关，必须先增加可选实现、保持 masked 默认值，并验证损失位置范围；此次没有修改模型。
6. 找到有效因素后再跑组合及至少 3 个 seed。主表没有给 NTU60 全监督 LP 的随机种子置信区间，不能把半个百分点机械认定为 bug，也不能用单次后期 epoch 方差代替重复实验方差。

**500 轮消融不等于拿已完成 400 轮的 checkpoint 追加 100 轮。** 余弦轨迹在前 400 轮已经不同。可以从按 500 轮规划的中断 checkpoint 恢复；若从完成的 400 轮追加，应另命名为 continuation，不能作为本文 P2。

## 8. 可直接采用的运行命令

以下为服务器 Bash 命令示例，工作目录必须是项目根目录。当前本机未执行。每个 variant/seed 单独使用输出目录；CLI 会覆盖 YAML，因此不要套用旧脚本中的 --epochs 400 或 --min_lr 5e-4 到所有变体。PT 保持官方默认 AMP=True；LP 保持 False。

### 四卡参考 R0

有效 PT batch128、LP batch256；LP 每卡 BN batch64。这与官方启动脚本的卡数和 batch 组织相同。

~~~bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=10274 main_pretrain.py \
  --config handoff_artifacts/macdiff_reproduction_20261008/configs/r0_pretrain.yaml \
  --output_dir output_dir/audit_r0_seed0 \
  --log_dir output_dir/audit_r0_seed0/tensorboard
~~~

~~~bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=10275 main_linprobe.py \
  --config handoff_artifacts/macdiff_reproduction_20261008/configs/lp_a.yaml \
  --finetune output_dir/audit_r0_seed0/checkpoint-399.pth \
  --output_dir output_dir/audit_r0_seed0_lp \
  --log_dir output_dir/audit_r0_seed0_lp/tensorboard
~~~

P1/P3/P4/P6 换对应 PT YAML 和独立目录，末轮仍 399；P2/P7 末轮为 **499**；P5 要配 lp_b.yaml。

### 双卡 R0 替代方案

保持相同 PT 全局 batch；LP 保持局部 BN batch64 和全局梯度 batch256。含第 4 节说明的 DDP/BN/末尾累积差异，因此与四卡参考应分别标记。

~~~bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.launch --nproc_per_node=2 --master_port=10274 main_pretrain.py \
  --config handoff_artifacts/macdiff_reproduction_20261008/configs/r0_pretrain.yaml \
  --batch_size 32 --accum_iter 2 \
  --output_dir output_dir/audit_r0_2gpu_seed0 \
  --log_dir output_dir/audit_r0_2gpu_seed0/tensorboard
~~~

~~~bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.launch --nproc_per_node=2 --master_port=10275 main_linprobe.py \
  --config handoff_artifacts/macdiff_reproduction_20261008/configs/lp_a.yaml \
  --finetune output_dir/audit_r0_2gpu_seed0/checkpoint-399.pth \
  --batch_size 64 --accum_iter 2 \
  --output_dir output_dir/audit_r0_2gpu_seed0_lp \
  --log_dir output_dir/audit_r0_2gpu_seed0_lp/tensorboard
~~~

LP L1 只把上述 LP 的 batch/accum 改为 128/1；L2 改为 64/1，并使用各自独立输出目录。其它参数包括输入 A、head、checkpoint、crop 和 seed 全部固定。

C1 使用 c1_t13_matched_a_pretrain.yaml、两卡、PT batch128/accum1，LP 使用 lp_a.yaml 并显式 --batch_size 128 --accum_iter 1。它对应保存命令的 T13 控制，不能标成官方 batch128 的 R0。

## 9. 每次实验必须留下的最小证据

- PT/LP 完整 Namespace、实际模型打印、服务器 Git SHA 和未提交修改、Python/PyTorch/CUDA/timm 版本。
- NPZ 路径与 SHA256、实际 train/test 样本数、类数；记录是否另做了人物排序/中心化/尺度缩放。
- checkpoint 路径、epoch、SHA256、保存的 args；统计 state_dict 的 blocks 与 decoder_blocks 层号范围，以实际权重核验 encoder/decoder 深度。
- PT 世界大小、每卡 batch、累积、effective batch、AMP、planned epochs、LR/min_lr/warmup/min_lr_epochs、mean/var、one_person、rotation、mask/tau、lambda_uniformity。
- LP 的同类执行参数、head feature dimension、BN 配置、训练/验证 crop、best/last；固定评价 checkpoint，不混入半监督生成增强结果。
- 若 loader 报 missing encoder keys，必须先处理；标准 LP 只允许 head 缺键，decoder 的 unexpected keys 属于预训练到下游结构差别，应保存完整 load_state_dict 消息。

配置准备检查已完成：11 份文件的相对控制组差异、重复生效键、顶层 argparse 字段、模型构造器参数均通过静态核验。当前环境无可用的训练数据/checkpoint，未用训练环境 PyYAML/模型启动运行验证；manifest 中明确标注 training_executed=False。

