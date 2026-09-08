# 冻结 Stage1 的类别关系读出比较

本实验回答：Stage1 已有的 LP 可分信息，能否在每类只有一个标注 exemplar 的条件下，通过更合适的距离或线性读出被利用？

这是独立诊断，不训练 backbone，不改 Stage2，不使用完整标签训练 LP teacher。
特征提取使用原项目 **LP 的确定性评估输入及分类器前特征**；不宣称重跑了原版 LP 的随机增强/分类器训练配方。

## 固定协议

- NTU60 XSub，Joint 输入。主比较不加入 JMB、queue、随机 projector 或 Stage2 mask。
- Stage1 encoder 永久 eval、no-grad；提取前后校验全部 BN/其他 buffers 不变。
- train 与官方 held-out test 分开缓存，原始浮点特征不提前 L2 normalize。
- 默认 exemplar seed 为 0/1/2，仅代表 exemplar 选择随机性，不是三个独立 Stage1 训练。
- 每个 seed 的 60 个 exemplar 在所有读出方法之间完全一致。先在完整训练集选 exemplar，再做可选的快速子集抽样。
- 所有 seed 的 exemplar 并集从统计拟合和 train-query 指标中排除；每个方法/seed 的拟合集及 query 集相同。
- 均值、方差仅由剩余训练特征估计。PCA 在其中按固定 seed=17、无标签均匀抽取最多 4096 条拟合。
- test 特征与标签均不参与统计量、PCA、ridge 权重拟合。train-query 标签也仅用于评分。
- train-query 是利用同一无标签训练分布的 transductive 诊断；主要泛化结果看 held-out test。
- 不依据 test 指标选择 alpha、温度、PCA 维度、seed 或方法；预设方法全部报告。
- 跨项目相同 seed 不保证是同一条骨架序列。跨项目逐样本比较须核对原始样本名/映射；本实验默认只保证项目内部配对。

## 比较矩阵

令 H 为冻结 backbone 的原始特征，mu/sigma 来自无标签训练统计。

| transform / head | 计算 | 要回答的问题 |
|---|---|---|
| raw / cosine | normalize(H) 对单 exemplar 的余弦 | 原始 one-shot 几何有多可靠 |
| centered / cosine | normalize(H-mu) | 公共偏置是否掩盖类别方向 |
| standardized / cosine | normalize((H-mu)/sigma) | 通道尺度是否干扰相似度 |
| pca / cosine | 标准化后投影到 PCA-256，再归一化 | 去掉低方差子空间的影响 |
| whitened / cosine | 同一 PCA 子空间内正则化白化，再归一化 | 控制维度后白化本身的影响 |
| standardized / ridge | 归一化标准特征上的 one-shot ridge | 固定输入，学习分类边界的作用 |
| linear_standardized / ridge | 未 L2 归一化的标准特征上的 one-shot ridge | 保留长度信息，测试对原始 H 真正仿射的读出 |

PCA 默认 rank=256；样本不足时 rank=min(256,N_fit_pca-1,D)，实际 components/特征值会保存。
PCA 使用一个 power iteration 的固定种子随机近似；pca/whitened 两组共享同一组 components。
它是 **PCA 子空间白化**，不是保持原始维度的完整 ZCA。不能把 raw 到 whitened 的全部差异归因于白化。

白化分母为 sqrt((1-rho)*eigenvalue_i + rho*mean(eigenvalues))，rho=0.1。
标准化方差下限为 max(mean(variance)*1e-6,1e-12)，防止常数维度爆炸。

两种 ridge 均固定报告 alpha=0.1/1/10，不进行搜索或自动选择。
默认每个分支有 5 个 cosine + 6 个 ridge = 11 个方法，三个 exemplar seed 共 33 行。

Ridge 拟合：
min_{W,b} ||E W + b - I_C||_F^2 + lambda ||W||_F^2，
lambda = alpha*trace(E_centered E_centered^T)/C。
E 只含每类一个 exemplar，按 class_ids 排序；偏置不惩罚，求解 C×C 对偶系统。
这是 one-shot 最小二乘分类器，不是用全训练标签训练的 softmax LP；其准确率不能直接冒充原文 LP。

## 两步运行

在 MacDiff 根目录：

~~~bash
CUDA_VISIBLE_DEVICES=0 python compare_stage1_readouts.py extract --checkpoint ./output_dir/ntu60_xsub_macdiff/checkpoint-399.pth --config ./config/ntu60_xsub_joint/linprobe_madiff.yaml --cache-dir ./output_dir/stage1_readout_cache --batch-size 4
~~~

~~~bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 python compare_stage1_readouts.py compare --cache-dir ./output_dir/stage1_readout_cache --output-dir ./output_dir/stage1_readout_comparison
~~~

需要覆盖数据位置时增加 --data-path /path/to/NTU60_XSub.npz。
直接使用原 Transformer downstream + ActionHeadLinprobe2，将 fc 换为 Identity。
encoder 严格加载原生 MacDiff Stage1；每人完整 750 tokens，处理双 person，
平均 person/time、保留 25 joints，输出 6400 维 J 特征。
不调用 feature_only=True（它会返回 256 维全局均值），不实例化 diffusion decoder 或 Stage2 heads。

输入固定 LP 测试配方：中心 95% 裁剪、120 帧，无旋转和噪声。
缓存位于 LP BN/Linear 之前；统计标准化是本实验的训练集变换，不是加载已训练 LP 的 BN。
显存不足时只降低提取 batch-size；eval 模式下不会因此改变 BN 统计。

运行前在项目自身的 Python/PyTorch 环境验证：

~~~bash
python -m unittest tests.test_stage1_readout tests.test_stage1_readout_torch -v
~~~

compare 只需要 NumPy，缓存可以移到 CPU 机器分析。限制 BLAS 线程数的
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 可按 CPU 配置调整。
提取不使用 AMP，固定 FP32，保证各方法复用相同数值；原始缓存通常需要约 1.5–2 GB 磁盘，
比较时建议预留至少 8 GB 可用主存。提取阶段还需原数据加载空间。

先检查链路时，在 extract 命令增加 --max-train 512 --max-eval 256 --workers 0，
并使用独立的新 cache/output 路径。小样本结果只证明流程可用。
正式比较默认 --max-train 0 --max-eval 0，即完整数据。
子集由不读取 query 标签的均匀抽样得到，因此 quick test 不保证类别均衡。

所有 cache/output 路径必须不存在。不会删除、覆盖或自动续跑旧目录；
中断的 cache 没有最后写入的 manifest，compare 会拒绝读取。中断后换新路径重新提取。
完成 cache 可重复比较，换新的 output-dir 即可，无须重新 GPU 前向。

## 复用现有 Stage2 exemplar

默认选择规则与两个项目 Stage2 相同：np.random.RandomState(seed)，按类别顺序各选一条。
若已有指定 exemplar，extract 时传入 --exemplar-seeds 0 --exemplar-caches PATH。
SCD 支持原 Stage2 NPZ，MacDiff 支持原 Stage2 JSON；多 seed 时按相同顺序传多个路径。

缓存会验证 seed、样本总数、class_ids、索引边界及索引对应的标签。
原始样本顺序必须与创建 Stage2 cache 时一致；不能重排数据后继续使用旧索引。
实际 seed→原始索引写入 manifest；样本名另存，便于人工核对。不会写回 Stage2 exemplar 文件。

## 输出和解读

cache-dir：

- manifest.json：完成标记、checkpoint/数据路径与大小/mtime、模型和提取协议、exemplar 原始索引。
- train/eval_indices.npy、train/eval_names.npy、train/eval_labels.npy。
- train/eval_BRANCH.npy：未经归一化的 FP32 特征，支持 mmap。

output-dir：

- comparison.md：逐方法测试 Top1 的 seed mean/sample-std，以及相对同 seed raw 的配对差。
- comparison.json：每 seed 的 support、train_unlabeled、eval 指标，以及设定和源 manifest SHA256。
- BRANCH_transform.npz：统计量、PCA components/特征值、正则化分母、PCA 拟合行号。
- fit_original_indices.npy：统计拟合所用的原始训练样本索引。

完整指标包括 Top1、Top5、macro Top1、confidence、entropy、NLL、ECE、
真实类概率 margin、active classes、预测类别分布及 confusion matrix。
准确率单位为百分数，差值为百分点；confidence/ECE 为 0–1，熵使用自然对数。

余弦分数默认温度 0.1，ridge 分数默认温度 1.0。温度不影响 Top1/Top5，但显著影响
confidence、entropy、NLL、ECE；不同 score family 的置信度不宜直接作优劣判断。
support 准确率只帮助发现记忆化，不能代表泛化能力。

观察顺序：

1. centered/standardized 比 raw 稳定改善：支持公共方向或尺度干扰距离的解释。
2. pca 和 whitened 相同 rank 下比较，才可讨论白化收益。
3. standardized ridge 对 standardized cosine：检查边界能否从同一归一化输入提取更多信息。
4. linear_standardized ridge 对 standardized ridge：检查是否需要保留特征长度。
5. support 高而 train/test 低：one-shot 边界代表性不足，不能直接当作高质量 Stage2 teacher。
6. 高 confidence/低 entropy 而低 Top1：错误目标变尖，不是关系质量提高。
7. 所有 one-shot 方法仍远低于完整 LP：一个 exemplar 可能不足以识别表征中的判别方向。

MacDiff 重点看：6400 维 raw 几何经中心化/标准化后能否恢复，PCA 是否丢掉有用信息，以及线性读出相对 exemplar 方向能否提取更多类别信息。原记录 85.86% 来自完整标签 LP，不代表该特征的 one-shot 相似度准确率。

本轮固定 clean Joint exemplar 和 clean query，以隔离读出方式。
增强稳定性、JMB 融合、邻居传播，以及使用这些关系训练 Stage2 后的 LP，是后续独立实验。
完整 LP teacher 可作为另行明确标记的额外监督诊断；本工具不会训练或加载它，也不读取 Stage2。

## 验证边界

本地 NumPy 数值/缓存集成测试可运行；Torch 测试需要项目服务器的依赖。
Torch 测试检查提取不改变权重/BN，并检查适配器输出与原 LP 特征一致。
正式 NTU 特征提取和最终准确率仍须在含数据与 Stage1 checkpoint 的服务器运行。
