# 原生 MacDiff 的坐标空间 Optimal Transport Flow Matching 基线

## 2026-10-27新增：Inverse Cosine信噪比映射，只改变OT时间采样

用户已要求“改一个试试”。新配置为 [pretrain_madiff_flow_matching_inverse_cosine.yaml](/D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_flow_matching_inverse_cosine.yaml)，仍走原FM入口和原encoder8/decoder3，方法路径版本仍为 `conditional_ot_noise_to_data_v2`，采样身份单独保存为 `flow_time_sampling=inverse_cosine_snr_v1`。旧配置和模型缺省仍为 `uniform`，T29成绩属于旧均匀采样，不能归给本次改动。本次无成绩、不预分配实验编号，助手未启动训练。

### 采样与目标

使用生产 `get_named_beta_schedule(['inverse_cosine', 1], 1000)`，按原生beta上限0.999离散化后累乘得到1000个alpha_bar，包含原生首步已有噪声及末步clipping；不是另写连续近似或普通Cosine。逐样本等概率抽参考索引u=0…999，计算：

```text
a = sqrt(alpha_bar[u])
b = sqrt(1-alpha_bar[u])
t = a / (b + (1-sigma_min)*a)
x_t = t*x + [1-(1-sigma_min)*t]*epsilon
target_velocity = x-(1-sigma_min)*epsilon
embedding_time = 999*t
L = masked_velocity_MSE + 0.02*uniformity
```

因此 `[t/(1-(1-sigma_min)*t)]² = alpha_bar[u]/(1-alpha_bar[u])`，匹配的是骨架/噪声的**系数信噪比**；旧A归一化后实际骨架方差不等于1，不能把该值当实测信号方差。u越大，对应OT时间t越小；decoder始终接收OT物理时间t，而不是u。只改变采样密度，速度不乘dt/du，也不乘999，不增加逆采样概率补偿；否则会改变本次拟验证的权重。OT路径、速度目标、masked归约和输入幅度保持原样，不直接切换成diffusion路径/epsilon损失。

这是借鉴 [SD3论文第3.1节CosMap的SNR映射思路](https://arxiv.org/html/2403.03206v1#S3.SS1)为本项目推导的Inverse Cosine版本，不冒称论文已有同名算法；原论文CosMap对应普通Cosine。不是logit-normal/Mode采样，也不是mini-batch OT重配对。

默认sigma_min=1e-4时，FP32采样表t范围约0.00141136–0.95767021、等概率均值0.50170809；1000点系数SNR处于[.1,10]的比例90.2%，旧均匀OT约51.95%。与原生表的FP32 SNR最大相对数值误差约1.45e-6。该区间仅是诊断定义，不是已验证最佳学习窗口。表为不持久化buffer，跟随设备移动，没有可训练参数或新增state_dict键；较大的sigma_min若达不到参考日程的最高SNR，直接拒绝，避免截断后假称完全匹配。

### 实验设置、日志与续训

其余与T29预期协议相同：A/person0/120帧/随机mask.9/global-local条件/condition dropout.1，无teacher、文本或后三层辅助监督。PT400、两卡每卡32、accum1、总batch64、lr.001、min_lr5e-4、warmup20、floor20、seed0、全程noAMP。LP用原main_linprobe.py和linprobe_madiff.yaml，第8层原特征，只训练原分类头，100轮/总batch128/lr.1/min_lr0。LP加载本组checkpoint-399.pth。

新日志 `train_time_mean` 与 `train_snr_middle_fraction` 可检查采样是否生效；后者epoch均值应在0.902附近波动，旧uniform预期约0.52。它们只是指标，不加入loss。原loss项保持 `train_loss_flow_matching` 与 `train_loss_uniformity`。

采样身份写入checkpoint args/model_args，完整resume拒绝跨uniform与inverse_cosine_snr_v1切换；旧v2 checkpoint缺该字段只解释为uniform，仍可按原均匀方法续训。新PT目录已有log/checkpoint会拒绝覆盖。恢复校验沿用完整方法/数据配置/日程/精度检查，仍不保证RNG逐位复现，也未验证服务器数据内容或源码hash。

服务器同步本次4个运行/检查文件：模型、FM入口、新YAML、tests/test_flow_matching.py；公共engine/main/util依赖沿用此前FM版本。未同步过FM的服务器还需按下面历史“文件与运行”节同步既有支持文件。先在已有Torch环境运行：

```bash
python -m unittest tests.test_flow_matching tests.test_pretrain_accumulation_policy -v
```

本地该命令共27项：21通过、6因无Torch跳过；新增验证含1000点逐点SNR对应、离散采样接口、实际生产forward的映射时间/路径/embedding/原速度损失、CLI覆盖、旧uniform恢复兼容及跨采样恢复拒绝、目录保护。6项真实Torch参数/梯度/设备移动/LP/engine测试待服务器执行，本地未运行GPU或DDP。Python3.8语法及实际参数解析的静态配置/命令检查通过，不等于真实PyYAML/Torch训练运行。

### 完整PT成功后接LP的一行命令

从Linux服务器仓库根目录执行，使用新目录和10321/10322端口：

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10321 main_pretrain_flow_matching.py --config config/ntu60_xsub_joint/pretrain_madiff_flow_matching_inverse_cosine.yaml --flow_time_sampling inverse_cosine_snr_v1 --flow_sigma_min 0.0001 --disable_amp --batch_size 32 --accum_iter 1 --epochs 400 --warmup_epochs 20 --min_lr_epochs 20 --lr 0.001 --min_lr 0.0005 --mask_ratio 0.9 --motion_aware_tau -1 --seed 0 --output_dir output_dir/ntu60_xsub_otfm_invcos_snr_dec3_noamp_seed0_20261027/pt --log_dir output_dir/ntu60_xsub_otfm_invcos_snr_dec3_noamp_seed0_20261027/pt/tensorboard && OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10322 main_linprobe.py --config config/ntu60_xsub_joint/linprobe_madiff.yaml --finetune output_dir/ntu60_xsub_otfm_invcos_snr_dec3_noamp_seed0_20261027/pt/checkpoint-399.pth --batch_size 64 --accum_iter 1 --epochs 100 --warmup_epochs 0 --min_lr_epochs 10 --lr 0.1 --min_lr 0 --seed 0 --dist_eval --output_dir output_dir/ntu60_xsub_otfm_invcos_snr_dec3_noamp_seed0_20261027/lp --log_dir output_dir/ntu60_xsub_otfm_invcos_snr_dec3_noamp_seed0_20261027/lp/tensorboard
```

以下2026-10-22/23内容保留旧uniform方法、T29成绩和原命令的历史身份。


2026-10-22：按用户补充的“要做Optimal flow matching”，当前采用Lipman原始FM论文的 **OT条件高斯路径**，在单路原生MacDiff上替换diffusion并评估encoder LP。方法身份为 `conditional_ot_noise_to_data_v2`，默认 `flow_sigma_min=1e-4`，独立入口、模型和配置；现有diffusion、文本和teacher配置继续保留。助手未启动训练。

**2026-10-23结果更新：**用户回传完整PT400/LP100，登记T29。LP best84.631246997%@90、80–99 mean84.512372650%，比历史原生dec3 T26分别低1.079572997/.932799616pp。PT末20velocity=.087960651531、uniformity=.033783732637、total=.088636326182，日志连续有限、loss组合与日程一致；实际Namespace/服务器SHA/LP加载输出仍缺，不由JSON补齐。OT uniform t与原inverse_cosine的噪声难度分布及速度/epsilon隐含时间权重不同，OT生成优势不保证LP；原因未定位，未改采样/权重/归一化。详细对照、系数SNR分布、原附件和SHA见[Stage1顶部T29](D:/program/MacDiff/STAGE1_EXPERIMENT_RESULTS.md:5)。下文为既有方法定义与命令，不是再次安排训练。本地仍无真实Torch/CUDA/DDP执行；已收到用户训练/LP/显存日志，但未收到下文5项服务器单测输出或耗时。

## 实际改动

设 `x` 为同一 crop 的干净骨架，经原生 A 归一化后形状为 `[B,120,25,3]`；保留 person0。每个样本独立取 `t ~ Uniform[0,1)`、`ε ~ N(0,I)`：

```text
sigma_min = 0.0001
sigma_t = 1 - (1-sigma_min)*t
x_t = sigma_t * ε + t*x       # t=0: ε; t=1: x + sigma_min*ε
u_t = x - (1-sigma_min)*ε     # 相对于物理时间 t 的配对速度目标
v_hat = decoder(x_t, z, timestep_embedding(999*t))
L_FM = mean_B [ sum_p mask_p * mean_coordinate((v_hat_p - patchify(u_t)_p)^2)
                / sum_p mask_p ]
L_total = L_FM + 0.02 * token_uniformity_loss(encoder_visible_tokens)
```

插值与速度目标对应 [Flow Matching for Generative Modeling，第4.1节Example II，公式20–23](https://arxiv.org/html/2210.02747v2#S4.SS1)。其OT性质是每个目标对应的 `N(0,I) → N(x,sigma_min² I)` 的最优传输位移插值；各条件路径混合后，不由此保证学到从噪声到整个骨架分布的全局最优传输。网络在重复采样和平方误差下学习条件期望速度，不涉及MeanFlow的区间平均速度。

**与上一版的关系：** `linear_noise_to_data_v1` 的 `(1-t)*ε+t*x`、`x-ε` 本来就是这条OT条件路径在 `sigma_min=0` 的形式，不能称其为“非OT路径”。本次明确记录OT方法身份并提供终点噪声参数；默认1e-4是本实验选择，不能声称仅改名或这一微小变化会改善LP。路径与目标同步变化，训练直接计算导数 `x-(1-sigma_min)*ε`，不在接近终点时用除法恢复速度。

**与mini-batch OT-CFM的区别：** [Tong等作者实现](https://github.com/atong01/conditional-flow-matching/blob/main/torchcfm/conditional_flow_matching.py)将原论文路径称为 `TargetConditionalFlowMatcher`，而 `ExactOptimalTransportConditionalFlowMatcher` 会另求batch间端点OT耦合。用户已明确选择前者：噪声→对应骨架，保持原生配对。噪声仍独立采样，骨架、增强骨架与encoder条件保持原样本配对，不新增batch匈牙利/Sinkhorn匹配、跨rank配对或OT求解器依赖；不要把本版本标成mini-batch OT-CFM。

`999*t` 仅用于已有正弦时间编码，保持原生约0–999的输入范围；**不取整，也不把速度目标乘999**。该编码系数是可记录的 `flow_time_scale`，默认999。物理插值和速度目标始终使用 `[0,1]` 的t；`flow_sigma_min`可通过CLI覆盖，允许范围 `[0,1)`，0用于终点无残余噪声的极限形式。

`z` 仍由增强骨架的 masked encoder 产生：750个位置中75个可见位置用对应 local token，675个 masked 位置用可见 token 的 GAP global，经 `ids_restore` 恢复原顺序，再由原生每层 FeatureModulation 逐位置调制。FM loss 经 decoder→调制→encoder 反传，uniformity 也直接作用于 encoder。

| 项目 | 本次设置 |
|---|---|
| 学习空间 | 归一化骨架坐标；不扩散/插值 encoder 特征 |
| 编码器与解码器 | encoder8 / decoder3，均256维；时间embedding64；输出仍每patch12维 |
| 输入和增强 | 与 native decoder3 配置相同，120帧、25 joints、时间patch4、person0、同crop小joint扰动 |
| mask / 条件 | 随机mask .9，motion_aware_tau=-1；global/local调制、condition dropout .1保持原生 |
| 路径 | 原论文OT条件路径，sigma_min=1e-4；t=1为骨架加该标准差的残余噪声 |
| 目标与权重 | masked OT velocity MSE；无diffusion schedule、离散t采样或MinSNR；uni权重.02 |
| PT | 400轮、两卡每卡32、accum1，总batch64；LR .001、warmup20、min_lr .0005、floor20；seed0 |
| 精度 | 新入口和YAML默认关闭AMP；命令显式 `--disable_amp`，关闭PT autocast和GradScaler |
| LP | 原 `main_linprobe.py`，100轮、两卡每卡64，总batch128；LR .1、warmup0、min_lr0、floor10；seed0 |

训练仍是每个样本随机取一个t、一次decoder前向；LP只加载encoder，不求解ODE。没有新增可训练模块，encoder/decoder的参数名和形状沿用原生，旧LP加载方式可直接复用。为复用原生初始化，子类构造时先创建后丢弃参数无关的diffusion helper；训练不调用它们，也不保存它们到state_dict。

与原 inverse-cosine epsilon 基线比较时，路径、时间采样和回归目标一起改变，故不把跨方法loss数值高低当作表示质量。uni仍为.02，但它相对主损失的尺度未必与旧epsilon目标相同；首轮保持该权重，先看LP。

## 文件与运行

- 模型：[transformer_macdiff_flow_matching.py](D:/program/MacDiff/model/transformer_macdiff_flow_matching.py)。
- 入口：[main_pretrain_flow_matching.py](D:/program/MacDiff/main_pretrain_flow_matching.py)。
- 配置：[pretrain_madiff_flow_matching.yaml](D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_flow_matching.yaml)。
- 检查：[test_flow_matching.py](D:/program/MacDiff/tests/test_flow_matching.py)。
- 公共engine仅增加通用辅助指标读取/记录分支，FM不进入teacher、EMA、Queue或文本target更新逻辑；完整累积窗口策略不变。

服务器同步上述新文件和当前 `engine_pretrain.py`；若服务器未同步此前的关闭AMP补丁，还需当前 `main_pretrain.py` 与 `util/misc.py`。无需修改文本缓存或LP文件，也无需安装OT求解包。使用原仓库已配置的Python3.8/Torch1.8环境和NTU数据路径。服务器可先执行：

```bash
python -m unittest tests.test_flow_matching tests.test_pretrain_accumulation_policy -v
```

从服务器仓库根目录运行下面完整一行，PT成功后自动接LP；输出使用新的FM目录，不复用历史checkpoint：

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10309 main_pretrain_flow_matching.py --config config/ntu60_xsub_joint/pretrain_madiff_flow_matching.yaml --flow_sigma_min 0.0001 --disable_amp --batch_size 32 --accum_iter 1 --epochs 400 --warmup_epochs 20 --min_lr_epochs 20 --lr 0.001 --min_lr 0.0005 --mask_ratio 0.9 --motion_aware_tau -1 --seed 0 --output_dir output_dir/ntu60_xsub_native_otfm_dec3_noamp_seed0_20261022/pt --log_dir output_dir/ntu60_xsub_native_otfm_dec3_noamp_seed0_20261022/pt/tensorboard && OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10310 main_linprobe.py --config config/ntu60_xsub_joint/linprobe_madiff.yaml --finetune output_dir/ntu60_xsub_native_otfm_dec3_noamp_seed0_20261022/pt/checkpoint-399.pth --batch_size 64 --accum_iter 1 --epochs 100 --warmup_epochs 0 --min_lr_epochs 10 --lr 0.1 --min_lr 0 --seed 0 --dist_eval --output_dir output_dir/ntu60_xsub_native_otfm_dec3_noamp_seed0_20261022/lp --log_dir output_dir/ntu60_xsub_native_otfm_dec3_noamp_seed0_20261022/lp/tensorboard
```

此命令从头训练，encoder8/decoder3；新目录含`_otfm_`，端口10309/10310，替代上一版`_fm_`目录和10307/10308命令，不据此推断旧命令是否已运行。`log.txt`新增 `train_loss_flow_matching`、`train_loss_uniformity`，总 `train_loss` 应等于二者按1/.02加权之和；TensorBoard辅助项前缀为 `flow_matching/`。

## 断点与验证边界

完整续训使用此专用入口的 `--resume`，要求checkpoint同时含model、optimizer、scaler、epoch、args；在原生main载入前核对FM方法、完整model_args（包括flow_sigma_min）、数据配置以及主要训练参数，拒绝原diffusion/text/teacher和上一版`linear_noise_to_data_v1` checkpoint冒充当前OT续训。若只想初始化旧encoder，那属于另一组warm start实验，当前入口不提供。不要绕过专用入口直接用公共main加载同形状的旧权重。

续训校验不是逐位复现承诺：原生checkpoint未保存RNG，数据路径/配置也不是数据内容hash；实际卡数、环境及数据内容仍需保持一致。训练精度改变会被专用入口的完整resume检查拒绝。

本地OT-FM 16项检查：11通过，5因缺PyTorch跳过；公共梯度累积5项通过，共16通过/5跳过。已通过：实际生产OT插值/损失/完整forward方法的NumPy执行、t=0/1端点和有限差分、论文欧拉场与采样导数等价、sigma_min=0退化形式、person0/归一化/增强配对、连续时间编码、masked归约、辅助指标分支、CLI覆盖和resume定义。NumPy检查不等于真实Torch执行。

待在服务器执行的5项真实Torch检查：原生参数/初始化一致性、完整反向传播及encoder/decoder梯度、真实decoder路径/时间输入、现有LP键兼容、公共engine单步和指标记录。测试使用Torch1.8提供的 `torch.testing.assert_allclose`（[对应官方源码](https://raw.githubusercontent.com/pytorch/pytorch/v1.8.1/torch/testing/__init__.py)）。另已做Python3.8语法、配置/命令静态核对及diff空白检查；本地无真实Torch/CUDA/DDP执行。10月23日已收到用户PT400/LP100及显存日志，见顶部；未收到这5项服务器单测输出或耗时。

既有teacher回归按其发现模式运行 `python -m unittest discover -s tests -p test_skeleton_teacher.py -v`，11通过/11缺Torch跳过；检查了公共engine在成功完整优化步后才提交辅助状态的既有行为。它与上面的FM检查分开计数。
