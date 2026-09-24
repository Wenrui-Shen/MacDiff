# MacDiff Stage1 文本扩散交接：RMS0.1 基线与 EMA remap 双向实验

更新时间：2026-09-24。本文写给完全没有上下文的新会话，是当前唯一优先交接。旧的
`handoff_stage2_legacy_20260905.md`、`handoff_vlm_pilot_legacy.md` 和历史交付包仅供追溯，
不得用其中的旧状态覆盖本文。

本页第1节是最新状态；后续章节保留以前的实现细节和历史命令，其中“当前RMS 1.0”
指当时的实验，不再是下一步。

## 1. 一页结论：现在在做什么、做到哪里、下一步是什么

任务是在 NTU60 XSub 的原始 MacDiff Stage1 上加入由骨架可视化描述得到的文本监督，
希望改善骨架 encoder 的动作表示和下游 linear probe（LP）。这不是 Stage2/OSE，不是
RGB监督，也不是文本生成。下游始终只使用骨架 encoder。

实验路线已经从复杂到简单逐项收缩：

| 实验 | 文本目标/分支 | S→T权重 | 最终LP best | 状态 |
|---|---|---:|---:|---|
| 原始MacDiff | 无文本 | 0 | 历史记录约85.86% | 已完成，比较协议有差异 |
| 第一版双向 | 可训练remap；T→S＋S→T | 1.0 | 83.03% | 已完成，效果下降 |
| 固定CLIP原尺度 | 去掉T→S和目标MLP；S→T | 1.0 | 84.64% | 已完成 |
| 固定CLIP原尺度权重消融 | 同上 | 0.1 | **85.95%** | 已完成 |
| 固定CLIP RMS目标 | 每个固定CLIP token加噪前无参数RMS归一化；S→T | 1.0 | 约83.7% | 已完成 |
| **固定CLIP RMS 0.1** | 同上 | **0.1** | **85.82%** | **用户暂定基线** |
| EMA remap双向，独立decoder | RMS0.1＋文本remap＋EMA文本目标＋T→S＋文本uniformity | 0.1 | 未运行 | 本地代码/配置已备 |
| EMA remap双向，共享decoder | 上一行只将`share_skeleton_decoder`设为`True` | 0.1 | 未运行 | 消融配置已备 |

正确/打乱骨架条件诊断已完成：epoch399、t100时，原尺度0.1的S→T MSE仅增加
`0.000021`，RMS0.1增加`0.002005`（0.3795%）。RMS确实提高了S→T对配对骨架的
依赖，但RMS1.0依赖最强而LP较差。因此暂不追加诊断，按用户指定继续双向实验。

新模式 `ema_remap`：在线512→512残差remap随机初始化；只有EMA目标支路初始化为
恒等映射，因而第一个S→T目标仍是固定RMS CLIP。T→S用在线remap文本预测骨架噪声。
`share_skeleton_decoder=False`时T→S有独立骨架decoder；`True`时复用原生decoder主体，
两组其余设置相同。S→T仍保持权重0.1，EMA目标每个成功optimizer step后以0.999动量更新。
文本local remap特征另加0.02的masked token uniformity。T→S权重暂定0.1；
新实验必须从头训练，不可把RMS0.1 checkpoint当成`--resume`。

服务器需同步 `model/transformer_macdiff_text.py`、`engine_pretrain.py`、`main_pretrain.py`、
`util/misc.py`、两份 `pretrain_madiff_text_rms_ema_bidirectional*.yaml` 配置。
工具没有服务器SSH，尚未代用户运行新实验。

从服务器项目根目录启动新预训练：

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10242 main_pretrain.py --config config/ntu60_xsub_joint/pretrain_madiff_text_rms_ema_bidirectional.yaml --batch_size 64 --accum_iter 1 --output_dir output_dir/ntu60_xsub_macdiff_rms_st01_ema_t2s01 --log_dir output_dir/ntu60_xsub_macdiff_rms_st01_ema_t2s01/tensorboard --skip_text_cache_validation
```

epoch399后做同协议LP：

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10243 main_linprobe.py --config config/ntu60_xsub_joint/linprobe_madiff.yaml --finetune output_dir/ntu60_xsub_macdiff_rms_st01_ema_t2s01/checkpoint-399.pth --output_dir output_dir/ntu60_xsub_macdiff_rms_st01_ema_t2s01_lp_399 --log_dir output_dir/ntu60_xsub_macdiff_rms_st01_ema_t2s01_lp_399/tensorboard --batch_size 32 --accum_iter 1 --epochs 100 --lr 0.1 --seed 0 --dist_eval
```

共享decoder消融的预训练（与上方不共享组使用不同目录）：

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10244 main_pretrain.py --config config/ntu60_xsub_joint/pretrain_madiff_text_rms_ema_bidirectional_shared.yaml --batch_size 64 --accum_iter 1 --output_dir output_dir/ntu60_xsub_macdiff_rms_st01_ema_t2s01_shared --log_dir output_dir/ntu60_xsub_macdiff_rms_st01_ema_t2s01_shared/tensorboard --skip_text_cache_validation
```

共享组epoch399的LP：

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10245 main_linprobe.py --config config/ntu60_xsub_joint/linprobe_madiff.yaml --finetune output_dir/ntu60_xsub_macdiff_rms_st01_ema_t2s01_shared/checkpoint-399.pth --output_dir output_dir/ntu60_xsub_macdiff_rms_st01_ema_t2s01_shared_lp_399 --log_dir output_dir/ntu60_xsub_macdiff_rms_st01_ema_t2s01_shared_lp_399/tensorboard --batch_size 32 --accum_iter 1 --epochs 100 --lr 0.1 --seed 0 --dist_eval
```

## 2. 环境、路径和用户偏好

- 本地项目：`D:/program/MacDiff`，Windows PowerShell。
- 服务器项目：`/home/user9/public3/swr/MacDiff`，用户终端曾显示
  `ubuntu@user9`；conda环境 `macdiff`，Python3.8、旧PyTorch API。
- 服务器有两张RTX4090 24GB。没有已配置的SSH工具，用户手动同步文件和运行命令。
- 用户中文沟通，偏好完整单行Linux命令，不喜欢重复确认、重建缓存和环境折腾。
- 只列需要同步的文件及用途，不再生成增量zip，不擅自提交或清理工作区。

服务器相对项目根目录的重要路径：

| 内容 | 路径 |
|---|---|
| NTU60数据 | `../data/MAMP/ntu/NTU60_XSub.npz` |
| 描述JSONL | `vlm_pilot/ntu60_xsub_train_person_captions_8b_single_gpu.jsonl` |
| v2 CLIP缓存 | `vlm_pilot/ntu60_xsub_clip_cache_v2` |
| 原始MacDiff输出 | `output_dir/ntu60_xsub_macdiff` |
| 第一版双向输出 | `output_dir/ntu60_xsub_macdiff_bidirectional_tokens` |
| 固定CLIP原尺度1.0输出 | `output_dir/ntu60_xsub_macdiff_fixed_clip` |
| 固定CLIP原尺度0.1输出 | `output_dir/ntu60_xsub_macdiff_fixed_clip_st01` |
| 固定CLIP RMS1.0输出 | `output_dir/ntu60_xsub_macdiff_fixed_clip_rms` |
| 固定CLIP RMS0.1基线输出 | `output_dir/ntu60_xsub_macdiff_fixed_clip_rms_st01` |
| EMA remap双向输出 | `output_dir/ntu60_xsub_macdiff_rms_st01_ema_t2s01` |
| EMA remap双向共享decoder输出 | `output_dir/ntu60_xsub_macdiff_rms_st01_ema_t2s01_shared` |

## 3. 文本和缓存已经完成：绝对不要重做

NTU60 XSub train共有40091个样本。Qwen3-VL-8B已为每个可见人物生成英文描述，
不含动作类别标签或RGB。历史18个GIF错误已补齐，accepted_unique=40091、missing=[]。
JSONL按 `sample_index` 取最后一条accepted记录；不能用物理行数判断缺失。

缓存协议是 `macdiff_clip_token_cache_v2`：

- `person_features.npy`：FP32 `[N,2,512]`，每人物句向量。
- `token_features.npy`：FP16 `[N,2,77,512]`，逐token特征，约5.89GiB。
- `person_valid.npy`、`token_mask.npy`、`token_ids.npy`：人物有效性、有效位置和token ID。
- local token保留正文＋真实EOS，排除BOS、padding和空人物。
- global和每个有效local token在 `cache_clip_text.py::encode_batch` 中已经逐向量L2归一化。
- global句向量为FP32，local token存FP16，因此local范数有很小量化误差。
- 缓存有manifest、文件大小和SHA256；已有缓存已成功训练完整400轮，不需要重编码。

当前正式读取器是 `util/person_text_cache.py::load_person_token_cache`，逐人物配对。
`one_person=True` 时只取骨架person0及其描述，禁止用person1替换空person0，禁止两人物
文本拼接或平均。

每人物local有效长度为小写 `k`，理论上 `k≤76`。训练batch只动态padding到当前
batch最大 `K_batch`，不会固定复制到76。日志 `text_valid_tokens≈20.15` 是人均有效
local token数，不含global；平均实际有效文本位置约21.15。padding不参与attention或loss。

## 4. 当前RMS模型的精确数据流

### 4.1 骨架原生分支

Feeder对时间段裁剪50%～100%并resize到120帧。encoder输入有Gaussian joint noise
std=0.005；扩散目标使用同一裁剪但不加该扰动。无vel/bone。

骨架只取person0：

`[B,3,120,25,2] → [B,120,25,3]`

模型用固定数据集统计量逐坐标通道标准化：

[
x'_c=(x_c-\mu_c)/\sqrt{\mathrm{var}_c}
]

- mean = `[-0.0058, -0.1333, -0.0246]`
- var = `[0.0206, 0.0805, 0.0218]`
- std约 `[0.1435, 0.2837, 0.1476]`
- `self_shift=False`

标准化后骨架每坐标通道设计尺度约为单位方差，再按相同inverse-cosine schedule加标准高斯
噪声：

[
x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,quad \epsilon\sim N(0,1)
]

120帧×25关节，patch=(4帧,1关节)，共750个patch；mask90%，encoder保留75个。
骨架encoder是8层、256维、8heads，输出 `latent[B,75,256]` 和平均池化
`pooled[B,1,256]`。原生5层骨架decoder预测12维patch噪声，只在675个masked位置计算MSE。
原生条件有10%整份dropout，uniformity权重0.02。

### 4.2 固定CLIP RMS文本目标

当前RMS配置：

```yaml
text_target_mode: fixed_clip
text_target_norm: rms
lambda_text_to_skeleton: 0.0
lambda_skeleton_to_text: 1.0
```

没有T→S分支，不创建目标remap MLP、目标LayerNorm、目标人物/位置embedding或
`text_skeleton_decoder`。目标不是冻结随机MLP，而是直接读取缓存CLIP。

对global句向量和每个有效local token，在加噪前用FP32执行：

$$
\operatorname{RMS}(x)=\sqrt{\frac{1}{D}\sum_i x_i^2},\qquad
\hat{x}=x/\operatorname{RMS}(x)
$$

无affine、无可训练参数、不减均值、目标detach，padding保持0。缓存已经L2归一化，因此对
精确单位范数512维向量：

$$
\operatorname{RMSNorm}(x)=\sqrt{512}\,x\approx22.627x
$$

这不是第二次L2；L2把整个向量范数设为1，RMS把每通道平均能量设为1，使整个向量范数为
`sqrt(512)`。若从原始CLIP投影重新设计，可以直接RMS一次；当前缓存已是L2，所以训练侧
RMS等价地转换尺度，并纠正FP16 local token的小范数误差。CLIP方向和余弦关系不变。

目标memory为：

- global `[B,1,512]`
- 动态local `[B,K_batch,512]`
- 拼接 `[B,K_batch+1,512]`

每人物的所有文本token共用一个独立采样的文本扩散时间步，噪声逐token/channel独立。
S→T仍预测标准高斯噪声，不改为直接回归干净CLIP。

### 4.3 文本decoder与梯度

文本decoder：

```text
noisy text [B,K+1,512]
→ Linear 512→256
→ 加decoder自有global/person/original-position结构embedding
→ 5个TextNoiseBlock
→ LayerNorm + Linear 256→512
→ 预测epsilon
```

每个block先以文本状态为Q、骨架 `[pooled; latent]` 为K/V做cross-attention，再用骨架条件
和时间步做FeatureModulation，然后做masked文本self-attention和FFN。结构embedding只属于
decoder输入，不进入干净目标。

总loss：

[
L=L_{native}+0.02L_{uniformity}+1.0L_{S\to T}
]

梯度：

- native＋uniformity更新骨架encoder。
- native更新骨架decoder。
- S→T更新骨架encoder和文本decoder。
- S→T不更新固定CLIP目标，也不进入骨架decoder。

## 5. 已完成实验及证据

### 5.1 第一版双向remap（历史）

第一版同时训练native、T→S、S→T。CLIP经可训练
`Linear512→512→256 + GELU + 无affine LayerNorm`，并加入人物/位置结构。
最终LP best为83.03%，整体下降。主要问题是一次加入组件过多，难以归因；remap内容在早期
强烈收缩，T→S与native共享decoder，S→T与native共享encoder。

只读诊断已在checkpoint 0/200/350运行。固定256条person0描述、batch8、3 repeats、
t=100/500/900；正确与打乱严格共用target/noise/t/mask，仅交换条件。

S→T打乱骨架memory后的MSE相对增幅：

| checkpoint | t=100 | t=500 | t=900 |
|---|---:|---:|---:|
| 0 | 约0 | 约0 | 约0 |
| 200 | 10.95% | 7.87% | 2.24% |
| 350 | 24.62% | 21.11% | 11.50% |

200/350的CI均为正，说明后期remap模型确实使用骨架条件；但LP仍差，证明“正确骨架降低
文本去噪loss”不等于学到分类有用语义。错误条件效应也可能来自样本细节、姿态和裁剪差异。

历史诊断脚本 `diagnose_text_conditioning.py` 当前只支持remap checkpoint，不能直接用于
fixed/RMS模型，后续需适配而不是强行strict=False。

### 5.2 固定CLIP原尺度、S→T权重1.0（已完成）

用户提供完整训练日志：
`C:/Users/97537/.codex/attachments/daee5726-57b6-4f86-931c-6b560db50527/pasted-text.txt`

日志411行由三段组成：0～5、0～4两次中断，最后0～399完整轨迹。分析必须取每个epoch最后
出现记录，即最后完整400轮。

结果：

- LP best：84.64%，比第一版83.03%提高1.61pp。
- epoch399 native部分：
  `.01436548 + .02*.01632162 = .01469191`
- epoch399 S→T原始MSE：`.50322537`
- epoch399总loss：`.51791728`
- 原版epoch399 loss：`.01452074`，本次native部分高约1.18%。
- S→T从epoch0的1.15828降至epoch19的.50454，之后基本停在.503附近。
- `text_energy=.001953125=1/512`、`text_batch_variance≈.000258`全程固定。
- 绝对方差小主要来自L2尺度；相对方差约13.2%，不是训练坍缩。
- 双卡每卡batch64、accum1，日志313步/epoch，有效batch128。
- GPU峰值allocated约18100MiB/卡、reserved约19878MiB/卡。

未缩放CLIP每维RMS约0.044，标准高斯噪声std=1。当前inverse-cosine schedule下，t=0文本
信号和噪声幅度已约相等，随后噪声迅速占优；例如t=332信号约.034、噪声约.640。这可能
让decoder主要根据noisy text估计epsilon，但没有fixed模型的打乱骨架诊断，不能直接断言
它忽略骨架。

历史原版Stage1 LP记录约85.86%，但原版预训练有效batch64，本次有效batch128；历史LP本身
也可能使用不同卡数/有效batch。因此84.64与83.03的比较更直接，84.64与85.86不是严格
同口径因果比较。若要确认，应用完全相同LP命令重跑原版checkpoint-399。

## 6. 当前待跑的两个配置

### 6.1 当前优先：RMS目标、权重1.0

配置：`config/ntu60_xsub_joint/pretrain_madiff_text_rms.yaml`

与已完成原尺度1.0实验相比，只增加 `text_target_norm: rms` 并使用独立输出目录；
目标内容、decoder内部256维、S→T权重、数据、训练轮数和batch均不变。这是纯尺度消融。

启动命令（服务器项目根目录）：

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10238 main_pretrain.py --config config/ntu60_xsub_joint/pretrain_madiff_text_rms.yaml --batch_size 64 --accum_iter 1 --output_dir output_dir/ntu60_xsub_macdiff_fixed_clip_rms --log_dir output_dir/ntu60_xsub_macdiff_fixed_clip_rms/tensorboard --skip_text_cache_validation
```

从头训练，不加旧resume。成功标志：

- args打印 `text_target_mode='fixed_clip'`
- args打印 `text_target_norm='rms'`
- `lambda_text_to_skeleton=0.0`
- `lambda_skeleton_to_text=1.0`
- 首个batch `text_energy≈1.0`
- 双卡每卡batch64时约313步/epoch

同组续训才可追加：

```text
--resume output_dir/ntu60_xsub_macdiff_fixed_clip_rms/checkpoint-实际编号.pth
```

### 6.2 已备但非当前优先：原尺度权重0.1

配置：`config/ntu60_xsub_joint/pretrain_madiff_text_st01.yaml`

只把原尺度fixed-CLIP的 `lambda_skeleton_to_text` 从1.0改为0.1，输出目录独立。日志中的
`loss_skeleton_to_text`仍显示乘权重前MSE，只有总loss和反传乘0.1。它用于判断共享
encoder上的辅助梯度是否过强，不能与RMS变化混为同一实验。

命令：

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10236 main_pretrain.py --config config/ntu60_xsub_joint/pretrain_madiff_text_st01.yaml --batch_size 64 --accum_iter 1 --output_dir output_dir/ntu60_xsub_macdiff_fixed_clip_st01 --log_dir output_dir/ntu60_xsub_macdiff_fixed_clip_st01/tensorboard --skip_text_cache_validation
```

## 7. LP评估与公平比较

当前fixed/RMS预训练采用双卡每卡64、有效batch128。LP命令此前确定为双卡每卡32、
有效batch64、100 epochs、绝对lr=.1、seed0、`linprobe2`。LP只加载骨架encoder；文本
decoder为unexpected keys被忽略，missing keys应只有分类head。

RMS checkpoint-399的双卡LP命令：

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10239 main_linprobe.py --config config/ntu60_xsub_joint/linprobe_madiff.yaml --finetune output_dir/ntu60_xsub_macdiff_fixed_clip_rms/checkpoint-399.pth --output_dir output_dir/ntu60_xsub_macdiff_fixed_clip_rms_lp_399 --log_dir output_dir/ntu60_xsub_macdiff_fixed_clip_rms_lp_399/tensorboard --batch_size 32 --accum_iter 1 --epochs 100 --lr 0.1 --seed 0 --dist_eval
```

`main_linprobe.py`每轮在test split评估并报告max，因此比较必须统一使用best-vs-best或
last-vs-last，不能混用。若需要严格比较历史原版，使用完全相同命令参数，仅替换：

- finetune为 `output_dir/ntu60_xsub_macdiff/checkpoint-399.pth`
- output/log目录换成独立基线路径

## 8. 缓存启动校验和多卡运行

默认完整缓存校验会：

1. hash骨架NPZ；
2. hash包括约5.89GiB token特征在内的所有缓存文件；
3. 分块扫描token数值、padding、EOS和范数。

双卡两个rank都会做一遍，只有rank0打印，可能很久没有新日志。现有缓存已验证并完成过
训练，可使用 `--skip_text_cache_validation`。快速模式仍检查manifest协议、complete、
样本数、文件大小和五个NPY的shape/dtype，但不能检测同大小内容篡改。

单卡/双卡可在epoch checkpoint之间切换resume，但数据顺序和随机轨迹会改变，且只恢复到
下一epoch，不恢复epoch内batch。保持当前有效batch128：

| 卡数 | 每卡batch | accum_iter |
|---|---:|---:|
| 单卡 | 64 | 2 |
| 双卡 | 64 | 1 |

RMS只能resume RMS同组checkpoint。checkpoint校验缓存身份、人物协议、T→S/S→T权重、
decoder共享设置、`text_target_mode` 和 `text_target_norm`。旧checkpoint缺
`text_target_norm`时按 `none` 处理，因此不能误接RMS。

## 9. 已修复和已解释的问题

- 空person0：某些随机裁剪后person0为空但person1非空。现在只排除该row的全部loss，
  不用person1替代；整batch无有效保留人物才报错。`empty_skeleton_persons`是每iteration
  数量的epoch平均，不是样本比例，历史频率极低。
- 随机 `pred/target` 打印：来自原生骨架 `forward_loss` 中0.2%随机打印，和文本目标无关。
  没打印不表示没有训练。
- OSE参数出现在args：`main_pretrain.py`是共用入口，会打印全部默认参数。只要
  `enable_ose=False`就不启用；text cache与OSE同时开启还会直接报错。
- 端口占用：误按Ctrl+Z会让分布式进程处于 `T/Tl` 暂停态但继续占端口。可用 `fg`
  恢复。若要结束，先TERM再CONT让暂停进程处理退出；不要看到Address already in use就反复
  启动更多进程。
- `torch.distributed.launch`的CalledProcessError通常只是父进程汇总，真正错误在它之前。
- 训练日志可能append多段重启记录，必须按epoch非连续处切段，不能直接把所有行当一条曲线。

## 10. 本轮本地改动、同步清单和验证

当前与RMS实验直接相关、需要同步服务器的文件：

| 文件 | 用途 |
|---|---|
| `model/transformer_macdiff_text.py` | 固定CLIP目标的 `none/rms` 模式；逐token无参数RMS |
| `main_pretrain.py` | 将 `text_target_norm` 写入checkpoint args |
| `util/misc.py` | resume时校验目标归一化模式 |
| `config/ntu60_xsub_joint/pretrain_madiff_text_rms.yaml` | RMS、S→T权重1.0正式配置 |

其他本地文件：

- `config/ntu60_xsub_joint/pretrain_madiff_text_st01.yaml`：原尺度0.1权重消融。
- `tests/test_macdiff_text.py`：新增RMS能量、方向、padding、detach和resume测试。
- 本 `handoff.md`：交接文档。

当前 `git status --short` 在写本文前为：

```text
 M handoff.md
 M main_pretrain.py
 M model/transformer_macdiff_text.py
 M tests/test_macdiff_text.py
 M util/misc.py
?? config/ntu60_xsub_joint/pretrain_madiff_text_rms.yaml
?? config/ntu60_xsub_joint/pretrain_madiff_text_st01.yaml
```

不要回滚、覆盖或宣称已提交这些改动。

本地验证：

- `python -m unittest tests.test_macdiff_text tests.test_clip_text_cache -q`：32项通过。
- 从正式RMS YAML构造B=1、120帧、25关节、6个local token模型，CPU前向/反向通过。
- 实际进入文本 `q_sample` 的global/local有效token每通道能量为1。
- 所有可训练参数梯度有限。
- 仅CPU验证，尚未服务器GPU/AMP/显存实测。

本地Python：
`C:/Users/97537/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe`

临时依赖通过：
`$env:TEMP/macdiff-text-test-py312` 和
`$env:TEMP/macdiff-clip-test-py312`。

## 11. 绝对不要再踩的坑

1. 不重新生成40091条描述，不重建已完成的v2缓存。
2. 不恢复跨人物拼接、两人物平均或空person0自动换person1；训练必须逐人物严格配对。
3. 不把双视图GIF的front/side误当成两个人。
4. 不把 `text_valid_tokens≈20` 当作固定K或总序列长度；它是平均local有效数，batch动态pad。
5. 不把fixed-CLIP的绝对方差 `.000258` 与LayerNorm remap的方差直接比较；先除以各自energy。
6. 不把 `text_energy=1` 当成语义丰富或没有坍缩的证明；它只说明RMS尺度正确。
7. 不把S→T loss下降或打乱条件loss上升直接等同于LP改善；条件可能传递分类无关细节。
8. 不用第一版remap诊断脚本强行加载fixed/RMS checkpoint；需先适配模型和目标构造。
9. 不把三任务总loss与原版native loss比较。当前应比较
   `loss_diff + .02*loss_uniformity` 与原版train_loss。
10. 不把S→T原始MSE数值当成梯度大小；权重和AdamW会改变共享encoder上的相对作用。
11. 不把原尺度0.1和RMS同时修改后称为单因素实验。当前RMS配置刻意保持权重1.0。
12. 不resume原尺度checkpoint到RMS实验，也不resume第一版双向checkpoint到fixed实验。
13. 不声称历史85.86与当前84.64严格同口径；预训练和LP有效batch存在差异。
14. 不把checkpoint-0当随机初始化；它已经完成第1轮。
15. 不因完整缓存校验耗时就重建缓存；可信现有缓存用快速校验开关。
16. 不因args打印OSE默认值就认为OSE参与训练，检查 `enable_ose`。
17. 不误按Ctrl+Z后立刻重启同端口；先检查 `ps`、`ss` 和进程状态。
18. Bash脚本需LF、无BOM；当前RMS应使用本文直接命令，旧
    `script_pretrain_macdiff_text.sh`硬编码的是原尺度基础YAML。
19. 不生成新zip，不擅自删除用户未跟踪文件、tmp或历史artifact。
20. 不声称服务器训练、GPU显存或LP已经运行，除非用户提供日志确认。

## 12. 下一会话接手时先做的检查

1. 先问用户RMS文件是否已同步、训练是否已启动，不要重复实现。
2. 若用户给启动日志，检查 `text_target_norm='rms'`、权重1.0和 `text_energy≈1`。
3. 若启动报state mismatch，确认没有加原尺度 `--resume`。
4. 若显存OOM，不先改模型；优先每卡batch32、accum2保持有效batch128，并记录这会增加
   每epoch更新次数，和batch64/accum1并非完全等轨迹。
5. 若用户给RMS训练日志，先按重启段切分，再比较native同口径、S→T平台和LP。
6. 若要判断骨架条件是否真正被使用，适配诊断并保持noisy text、target、noise、t完全相同，
   只打乱骨架memory；分t=100/500/900报告相对MSE增幅和CI。
