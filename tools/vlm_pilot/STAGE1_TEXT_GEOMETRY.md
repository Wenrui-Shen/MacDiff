# Stage1 与骨架描述文本的类别结构预实验

第一轮只做 NTU60 XSub train 的类别均衡诊断，不更新权重，不接入训练损失。

- 同一批样本分别通过 Stage1 backbone 和冻结 CLIP 文本 encoder。
- 每个 batch 随机选 16 类，每类 8 个不同样本，共 128 个；默认固定 seed=42、100 个 batch。
- 对所有 checkpoint 复用同一个 plan，文本按原始 NPZ `sample_index` 对齐。
- 先取 checkpoint 0、10、20、50、100、200、399。文件中的 epoch 从 0 起算；checkpoint-0 已经训练了一个 epoch。
- 骨架采用原生 linprobe2 的完整双人、750 token/person、6400 维特征：分类层换成 Identity，eval 关闭 dropout。输入沿用 LP 的中心 95% 裁剪并 resize 到 120 帧，不做随机增强或 mask。
- 文本只取 `texts[].text`。每个人的 CLIP 投影后向量先 L2 归一化，人物间平均后再次 L2 归一化。双人关系可能受平均聚合影响；本轮不是人物级分类。
- 使用官方 `openai/clip-vit-base-patch32`。不需要生成模型 Qwen，也不需要 vLLM/AutoAWQ。
- 两种特征在各自空间算余弦相似度，无需投影到相同维数。标签只用于采样和指标。

## 服务器准备

同步最新仓库（入口 `compare_stage1_text_geometry.py` 复用 `compare_stage1_readouts.py`、`stage1_readout.py` 和现有 model/feeder 模块）。使用已装有 Torch、Transformers、NumPy、PyYAML 的环境，例如现有 skeleton_vlm；无需升级 Torch/Transformers。

在 `/home/user9/public3/swr/MacDiff` 执行。以下每条命令均为单行。

下载一次官方权重与 tokenizer。只下载 PyTorch 格式及配置，约 610 MB，不下载 TensorFlow/Flax 的重复权重：

```bash
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_HUB_DISABLE_XET=1 python -u -c "from huggingface_hub import snapshot_download; print(snapshot_download('openai/clip-vit-base-patch32', local_dir='/home/user9/public3/swr/models/clip-vit-base-patch32', allow_patterns=['*.json','merges.txt','pytorch_model.bin'], max_workers=2))"
```

来源与接口：[官方权重](https://huggingface.co/openai/clip-vit-base-patch32/tree/main)、[下载文档](https://huggingface.co/docs/huggingface_hub/guides/download)。如果访问 Hugging Face 失败，先处理下载连接，不需要更换推理框架。

本脚本直接实例化 CLIP 的文本结构，从本地官方 tensor state dict 严格加载文本权重和投影层；忽略视觉权重。支持本地 safetensors 或官方 pytorch_model.bin。不允许文本权重缺失，也不静默截断超过 77 tokens 的描述。

先运行校验（在服务器还会检查现有骨架提取路径）：

```bash
python -m unittest tests.test_stage1_text_geometry tests.test_stage1_readout_torch -v
```

## 固定采样和执行

生成不可变的采样清单，此步不需要 CLIP，也不占用 GPU。若实际文件路径不同，修改 data-path / captions。可以在 `--captions` 后列多个生成分片，但只允许同一模型、同一提示词 hash。重复样本保留给定文件顺序中最后一条 accepted 记录；invalid 不覆盖 accepted。

```bash
python compare_stage1_text_geometry.py plan --data-path ../data/MAMP/ntu/NTU60_XSub.npz --captions vlm_pilot/ntu60_xsub_train_person_captions_8b_single_gpu.jsonl --output-dir vlm_pilot/stage1_text_geometry_pk128 --num-batches 100 --classes-per-batch 16 --samples-per-class 8 --seed 42
```

plan.json 保存原始样本索引、标签、冻结文本、全部 batch、每类有效文本覆盖率和数据来源。每类至少需要 8 个 accepted 样本，不静默丢掉覆盖不足的类别。若 JSONL 最后一行还在写入导致 JSON 不完整，会明确报错，待生成写入完成再运行。以后新增的文本不会自动改变已经固定的 plan。

单卡执行 7 个 checkpoint，顺序加载、保存特征缓存。GPU 1 在进程内编号是 cuda:0：

```bash
OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 python -u compare_stage1_text_geometry.py run --output-dir vlm_pilot/stage1_text_geometry_pk128 --clip-model /home/user9/public3/swr/models/clip-vit-base-patch32 --checkpoints output_dir/ntu60_xsub_macdiff/checkpoint-{0,10,20,50,100,200,399}.pth --micro-batch-size 4
```

上面 checkpoint 列表使用 Bash 花括号展开。提取 micro-batch=4 与相似度 batch=128 是两个参数；可以因显存将前者减为 2。重跑同一 run 命令会复用完整特征缓存，中途未完成的 checkpoint 从头提取；不需要重新生成 plan。代码、模型或配置变更后应使用新的实验目录，避免混用缓存。

如先试通整条路径，可将 plan 的 num-batches 改为 2、output-dir 换一个新目录，并仅运行 checkpoint-0 与 checkpoint-399；这只是冒烟测试，不能代表完整类别统计。

## 结果解释

- `summary.csv`：文本参考和每个 checkpoint 的 class-macro AUC、P@1、P@5、同类／异类余弦及 gap。
- `summary.json`：逐类、逐 batch 指标，文本减骨架的差值，以及逐 batch 配对 AUC 差值。
- `text_metrics.json` / `checkpoint-*_metrics.json`：独立结果；`.npy` 为归一化特征缓存，行序对应 plan.samples。
- 若环境已有 matplotlib，会额外保存 `geometry_curve.png`。图横轴是实际已完成 epoch 数，即 checkpoint 编号 + 1。

主要指标为逐 anchor 的 AUC：同类伙伴比分别的异类伙伴更相似的比例，相等计 0.5，随机基线为 0.5。最近邻 P@1/P@5 对边界相同分数取平均命中率，不依赖样本排序打破平局。均衡 batch 中随机近邻同类比例为 7/127。

首先对每个 anchor 计算，按 batch 内类别取均值，再按所有出现的类别等权汇总。反复采样共用样本，因此 batch 标准差只表示本次采样波动，不作为独立观测置信区间。

文本更高的 AUC/P@k 才支持更可靠的类别关系，单独更高的同类余弦不足以支持假设。本轮使用有有效描述的训练集子集，结论限于该条件，不直接证明测试泛化或接入文本后训练会提升。
