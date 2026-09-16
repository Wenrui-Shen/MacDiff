# MacDiff：骨架文本与 Stage1 类别结构预实验交接

更新时间：2026-09-16。写给完全没有上下文的新会话。

## 0. 最新进展：已授权实现文本条件 Stage1（优先于后文旧任务状态）

用户在原预实验负结果后明确要求：先写全量冻结 CLIP 缓存脚本，再把全局文本条件接入原始 MacDiff Stage1。已确定 MLP remap、三个条件噪声预测目标、反向文本去噪在加噪前 stop-gradient；三个 loss 从第一步共同参与，固定权重后续消融，不加 loss warm-up。整段描述暂作为全局语义条件，保持原始时间裁剪。

本地新增 `cache_clip_text.py`、`util/clip_text_cache.py`、`model/transformer_macdiff_text.py`、`config/ntu60_xsub_joint/pretrain_madiff_text.yaml`、`script_pretrain_macdiff_text.sh`，接入 `main_pretrain.py` 和 `engine_pretrain.py`。完整设计、缓存/训练单行命令见 **`tools/vlm_pilot/STAGE1_TEXT_DIFFUSION.md`**。

- 缓存覆盖全部原始 x_train 行，按最后 accepted 去重；逐人保存投影后 FP32 L2 CLIP 特征和人物 mask。严格检查覆盖、提示词/生成模型版本、token 上限；支持中断续跑和完整缓存复用，有 SHA256 身份与完整性检查。
- 训练时有效人物文本取均值再 L2，形成全局 e；`r = LN(MLP(e))`，512→512→256，最后 LN 无 affine 参数、FP32。缓存之前不会做这个可训练 remap。
- 保留原生骨架 encoder/decoder；额外独立骨架 decoder 用 r 预测骨架噪声，更新 remap；额外条件残差 MLP decoder 以骨架全局特征预测 r 的噪声，r 在加噪前 detach。一次骨架编码，三个目标都预测 epsilon。
- 两个新增权重默认 1.0；没有证明最优。文本沿用 1000 步 inverse_cosine、均匀采样。保留原始 uniformity、学习率 warm-up；后者与新增 loss 调度无关。
- 原始训练配置实际默认 `one_person=True`，新配置保留，并未顺带改为双人训练（此前几何提取用双人是另一协议）。全局文本汇总全部描述。
- 关闭 T→S 时 remap 冻结在随机初始化，此消融不等于直接恢复原 CLIP；两个新权重均为零时调用原始 forward。resume 校验缓存身份与固定权重，不能直接用原始 checkpoint 完整恢复三任务训练。
- 本地缓存/几何 14 项测试、真实 PyTorch CPU 模型/梯度/训练引擎/恢复身份 8 项测试、原有 encoder 提取 2 项回归测试，共 24 项通过。生产 YAML/参数解析与完整尺寸 CPU 前向（1×750×12 输出）也通过。为验证，在 Windows TEMP/macdiff-text-test-py312 独立目录准备了 torch 2.5.1+cpu 和 PyYAML，用 PYTHONPATH 临时启用；没有改服务器或原 Python 环境。`util/misc.py` 的 torch._six.inf 改为 math.inf，兼容新旧 Torch。
- 尚未运行真实 CLIP 全量编码、服务器 CUDA AMP、4090 显存或正式训练，不能声称获得收益。用户需复制本地改动到服务器后执行。
- 本轮开始时已有用户修改 `engine_linprobe.py`、`main_linprobe.py` 和未跟踪 `script_linprobe_stage2_early_final.sh`，本轮不改它们。

以下第 1～8 节保留此前预实验背景和真实结果；其中“尚未授权训练设计/当前优先错误分析”等描述已被本节和新用户指令更新，不应阻止继续当前 Stage1 文本方案。

## 1. 当前任务及结论

用户想验证：原始骨架渲染成 GIF 后，由 Qwen3-VL 生成的动作文本，经预训练 CLIP 文本 encoder 编码，是否能在一个 batch 内提供比早期骨架 encoder 更可靠的类别相似度结构，为后续训练提供关系监督。

用户明确选择：**原始 MacDiff Stage1，不是 Stage2；关系矩阵 batch=128；先只做按类别均衡采样。**

当前已经跑完预实验并分析了结果。最重要结论是：**当前生成文本＋CLIP＋人物平均聚合的方案，没有整体优于早期骨架，近邻纯度明显更低。** 文本有类别信息，但不支持直接作为更可靠的全局关系监督。

现在不是卡在运行、下载或生成，而是需要解释负结果，判断是否存在局部互补性。下一步已建议小规模近邻错误分析，尚未实施。用户没有要求开始新训练或反复换模型直到支持假设。

原 handoff 已备份为 `handoff_stage2_legacy_20260905.md`，只作历史背景。里面的 Stage2/OSE “唯一基线”“严格执行顺序”不是当前任务。`handoff_vlm_pilot_legacy.md` 中三视图、只做单样本的状态也已过时。

## 2. 机器、环境、沟通习惯

- 本地 Windows 工作区：`D:\program\MacDiff`。工具改的是本地，不能声称已修改服务器。
- 服务器：`ubuntu@user9`，项目 `/home/user9/public3/swr/MacDiff`，用户 home 实际是 `/home/ubuntu`。
- 本会话没有可调用的服务器 SSH 配置；用户复制文件、执行命令、回传日志。
- 用户使用中文，喜欢**完整单行命令**，很反感无必要的升级环境、反复确认、反复重跑。
- 两张 RTX 4090 24GB，仅 GPU 1 用于本任务。`CUDA_VISIBLE_DEVICES=1` 后进程内是 `cuda:0`。
- 服务器 `skeleton_vlm`：Torch 2.5.1+cu121、Transformers 4.57.1；此前确认 qwen-vl-utils 0.0.14、numpy 1.26.4。
- 驱动 535.54.03、glibc 2.27、kernel 5.4.0-150-generic。user9 可能为容器，隔离边界未最终确认。
- 本地测试 Python：`C:/Users/97537/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe`。有 NumPy、Pillow，没有 Torch/Transformers，不能本地跑真实模型。

服务器关键路径（除绝对路径外均相对项目根目录）：

```text
数据：../data/MAMP/ntu/NTU60_XSub.npz
GIF：vlm_pilot/ntu60_xsub_train_rendered_v3_2view_smooth_w5
文本：vlm_pilot/ntu60_xsub_train_person_captions_8b_single_gpu.jsonl
提示词：tools/vlm_pilot/skeleton_motion_prompt_v1.txt
Qwen：/home/user9/public3/swr/models/Qwen3-VL-8B-Instruct
CLIP：/home/user9/public3/swr/models/clip-vit-base-patch32
Stage1：output_dir/ntu60_xsub_macdiff/checkpoint-*.pth
预实验：vlm_pilot/stage1_text_geometry_pk128
```

## 3. 文本生成已全量完成

NTU60 XSub train 共 40,091 个样本，输入预渲染双视图 GIF（front XY、side ZY），平滑 w5、32 帧、8 fps。两视图不是两个人；红色 person_index=0，蓝色 person_index=1。

Qwen 为每个可见人物选择一个主要运动部位，生成 main_part、motion、beginning、middle、end、interaction、text。每人 text 最多 35 个英文单词；不提供类别标签/RGB，不允许猜测物体、意图、场景。该描述可能丢失全身动作细节，这是待分析的瓶颈，不是已证实原因。

全量提示词 SHA256：

```text
90ff138534aebf5fcb1c19607e4bb8d1ce1b46d9c8cbcdfe257b726e717f9f90
```

最初 accepted=40073、invalid=0、pipeline_error=18；18 个均是 GIF 帧数不匹配，发生在推理之前。修复补跑后用户实际确认：

```text
accepted_unique: 40091
missing: []
```

**不要重新全量生成。** JSONL 追加成功结果，旧的 18 条错误仍存在。统计必须按 sample_index 找 accepted 去重，不能把历史错误数当作当前缺失，也不能按行数算样本数。accepted 仅表示通过脚本校验，不保证语义正确。

生成入口 `tools/vlm_pilot/caption_qwen3vl_rendered_train_transformers.py` 曾被加入 vLLM/AWQ 兼容逻辑。用户因加载变慢要求回退，已恢复到 Git `b6c54dd` 原始 Transformers 版本，之后仅加入 GIF 恢复和 `--check_gifs`。不要重新引入放弃的加载逻辑。

### GIF 修复（已在服务器完成验证）

17 个读到 31 帧，sample=33369 读到 30 帧，元数据都是 32 帧。sample=7030 实测 `31 frames; durations={120:30,250:1}; total=3850 ms`。

渲染每帧 duration=round(1000/8)=125ms；GIF 存储粒度 10ms，普通帧成为 120ms，相同相邻帧由 Pillow 合并后成为 250ms。

load_gif_frames 新逻辑只在帧数不足时按原始 sample_fps 推断重复次数，并严格要求：每帧时长等于 `floor(repeats*duration_ms/10)*10`，重复次数合计等于 expected_frames，才复制相应画面恢复。其他不一致仍报错。恢复在内存中，不覆盖 GIF/元数据，不影响正常 32 帧。

`--check_gifs --resume` 可以不加载模型检查所有未成功样本。服务器实际 `checked=18, passed=18, failures=[]`，之后 18 个已补生成成功。`tests/test_rendered_gif_recovery.py` 5 项本地通过，含原生 writer→reader 像素顺序往返。

18 个索引：7030,11328,11329,11392,12467,18484,21558,25662,29256,33369,34209,35687,35746,35779,35791,36988,38952,39266。

## 4. 预实验代码与协议

入口 `compare_stage1_text_geometry.py`，协议 `stage1_caption_balanced_geometry_v1`。

依赖/说明：

- `compare_stage1_readouts.py`：严格加载原生 Stage1 encoder。
- `stage1_readout.py`：fingerprint 等辅助函数。
- `model/transformer_downstream.py`：原生 linprobe2 forward。
- `feeder/feeder_stage2.py`：只调用 get_base_sample 做确定性处理，并非训练 Stage2。
- `tests/test_stage1_text_geometry.py`：8 项采样、指标、对齐、缓存测试本地通过。
- `tests/test_stage1_readout_torch.py`：本地缺 Torch 跳过；没有用户单独运行它的结果。用户真实提取已经跑通，但不能说所有 GPU 单测已通过。
- `tools/vlm_pilot/STAGE1_TEXT_GEOMETRY.md`：完整使用说明。

### 采样和提取

- 原始 NPZ sample_index 对齐，只用 train、同生成模型/提示词 hash 的 accepted 描述；重复取最后一条 accepted，invalid 不覆盖 accepted。
- 当前 plan 在全部 40091 个样本补齐后生成。100 batches，每批无放回随机选 16 类，每类无放回取 8 个，seed=42，跨 batch 可重复；union=11003 个样本。
- plan.json 冻结索引、标签、文本、全部 batch、60 类覆盖率和来源信息。不要覆盖或重新抽样。
- 所有 checkpoint 和文本使用完全相同的 plan；特征分别缓存。
- 骨架：eval，关闭 mask/随机增强，中心95%裁剪、resize120帧、双人、750 token/person。原生 linprobe2 head.fc=Identity，平均人物/时间，保留关节，得到25×256=6400维。无监督分类器、LP BN 或 Stage2 projector。
- **不要改成 feature_only=True**：它还平均关节，输出256维，会改变比较口径。
- 文本：冻结 CLIPTextModelWithProjection，取投影后 text_embeds。只编码每人 texts[].text，各人先L2、人物间平均、再L2。双人仍是一个样本。超长文本明确报错，不静默截断。
- 本实验用 Transformers 格式的官方 `openai/clip-vit-base-patch32`，本地 pytorch_model.bin（weights_only=True读取 tensor state dict）或 safetensors，严格加载文本键、不加载视觉部分。无需官方 clip 包。
- 两种特征各自 L2 后算空间内 cosine，维数不需要一致，不直接算骨架-文本 cosine。
- 提取 micro-batch=4 与关系矩阵 batch=128 独立，减小前者不改变实验。

### 指标和统计口径

- Anchor AUC：每个 anchor 的同类正伙伴比分别的异类负伙伴更相似的比例；相等计0.5；排除自身；随机基线0.5。
- P@1/P@5：最相似1/5个样本的同类比例；边界相同分数平均命中，避免按索引打破平局。本协议随机基线7/127=5.5118%。
- same_cosine、different_cosine、gap 只辅助解释，不能靠跨空间绝对值判断谁更可靠。
- 先每个anchor、再batch内类别平均、最后按出现类别等权汇总（class-macro）。
- batch_std/配对差值是描述性统计；batch共享样本，不是100次独立试验，不据此伪造CI/p值。
- 还没有做：普通随机batch、逐样本错误近邻导出、人物数分组、triplet“文本纠正骨架/反向误导”分析。不要声称已实现。

用户已成功运行：

```bash
python compare_stage1_text_geometry.py plan --data-path ../data/MAMP/ntu/NTU60_XSub.npz --captions vlm_pilot/ntu60_xsub_train_person_captions_8b_single_gpu.jsonl --output-dir vlm_pilot/stage1_text_geometry_pk128 --num-batches 100 --classes-per-batch 16 --samples-per-class 8 --seed 42
```

输出 `100 batches x 128; 11003 unique samples`。

```bash
OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 python -u compare_stage1_text_geometry.py run --output-dir vlm_pilot/stage1_text_geometry_pk128 --clip-model /home/user9/public3/swr/models/clip-vit-base-patch32 --checkpoints output_dir/ntu60_xsub_macdiff/checkpoint-{0,10,20,50,100,200,399}.pth --micro-batch-size 4
```

花括号是 Bash 展开。checkpoint编号0是已经训练完一个epoch，不是随机初始化；绘图使用编号+1。

服务器输出包括 plan.json、text_features.npy/json、各checkpoint特征npy与身份json、text_metrics.json、各checkpoint metrics.json、summary.json/csv；有matplotlib时另生成geometry_curve.png。

缓存校验包含plan、代码、模型/配置信息。run重跑复用完整缓存，未完成checkpoint从头提取。新提取方式需明确新协议/目录；不要删旧缓存，更不要分析时重提取全部7个模型。

## 5. 已得到的真实结果

完整原始 summary 已保存至本仓库：**`handoff_artifacts/stage1_text_geometry_summary.json`**。无需让用户重复上传。

原始附件：`C:/Users/97537/.codex/attachments/44ef8996-6a4a-42a2-89fc-0173975a0880/pasted-text.txt`。

本地有全部逐类/逐batch指标，但没有服务器plan、原始特征、GIF或逐样本文本，无法只凭summary重建错误邻居。

| 特征 | AUC | P@1 | P@5 | same cosine | different cosine | gap |
|---|---:|---:|---:|---:|---:|---:|
| CLIP文本 | 0.667381 | 0.191414 | 0.159642 | 0.908117 | 0.866681 | 0.041435 |
| Stage1 0 | 0.664143 | 0.347901 | 0.239186 | 0.976648 | 0.969527 | 0.007121 |
| Stage1 10 | 0.785167 | 0.456910 | 0.336177 | 0.959433 | 0.928223 | 0.031210 |
| Stage1 20 | 0.752455 | 0.436217 | 0.305546 | 0.941285 | 0.907098 | 0.034188 |
| Stage1 50 | 0.744836 | 0.453236 | 0.316345 | 0.928847 | 0.888792 | 0.040055 |
| Stage1 100 | 0.737276 | 0.468434 | 0.323633 | 0.921494 | 0.879805 | 0.041690 |
| Stage1 200 | 0.741194 | 0.487118 | 0.335788 | 0.923336 | 0.884006 | 0.039330 |
| Stage1 399 | 0.774436 | 0.522610 | 0.367702 | 0.952073 | 0.927153 | 0.024920 |

进一步统计：

| checkpoint | 文本AUC更高类数/60 | 文本P@1更高类数 | 文本P@5更高类数 | 文本AUC更高batch数/100 |
|---|---:|---:|---:|---:|
| 0 | 17 | 4 | 8 | 57 |
| 10 | 6 | 1 | 1 | 0 |
| 20 | 11 | 1 | 4 | 0 |
| 50 | 14 | 0 | 4 | 2 |
| 100 | 14 | 0 | 5 | 3 |
| 200 | 11 | 0 | 5 | 3 |
| 399 | 11 | 0 | 4 | 0 |

已经告知用户的结论：

- 相对checkpoint-0，文本AUC只+0.003238，但P@1低15.65个百分点；文本P@1/P@5分别仅1个batch占优。
- 对checkpoint-10和399，全部100个batch的AUC/P@1/P@5都是骨架更高。
- 相对最终checkpoint-399，文本P@1低33.12个百分点；60类中没有任何一类文本P@1更高。
- 文本AUC>0.5、P@1>5.51%，有类别信息，不是完全无用。
- 文本gap=0.0414反而高于最终骨架0.0249，而排序/近邻更差；只报gap会误导。
- 相对最终骨架，文本AUC优势最大的0-based类ID包括57、55、53、56、52、54、50、58，这些类P@1仍弱。尚未核对类别名，不要凭记忆贴动作名称。
- 原假设在当前流水线、已测checkpoint、均衡train batch条件下不成立，不等于否定所有文本编码器/描述方式或未测的最早训练updates。
- 不要因checkpoint-10 AUC高于最终就说模型退化：最终P@1/P@5更高，指标衡量不同。

## 6. 当前卡点与下一步

用户已经完成真实提取并上传结果，无下载/执行阻塞。此前网络异常是否恢复未知，但CLIP权重已准备成功，不能再当作当前卡点。

最后建议了以下错误分析，**尚未开始实现**：

1. 与用户讨论并确定查看逐样本近邻案例。复用已有plan和特征缓存，不重新生成或提取。
2. 在相同batch中固定规则抽取“文本近邻异类、骨架近邻同类”和反向情况，优先看checkpoint-0及399。不能只挑支持假设的例子。
3. 导出anchor/近邻的sample_index、0-based标签、cosine、两侧描述、GIF路径，再小批查看图像。需用户提供服务器plan/缓存或运行分析脚本；当前本地summary不够。
4. 区分可能瓶颈：生成描述丢失关键动作细节；CLIP未区分方向/顺序差异；双人平均丢失角色关系；粗粒度人数/姿态主导相似度。这些目前只是待检验解释。
5. 错误分析支持后，再讨论字段组合、人物聚合、文本编码器或局部关系使用。保存相同采样清单、明确变更；不要不断换模型直到得到正结果。
6. 用真实标签判断哪些关系可靠可以做诊断，但不能直接充当无监督训练中的筛选策略。

当前优先级是理解负结果，不是开始全局文本蒸馏、重做40091条文本、切换大模型或恢复旧Stage2训练。

## 7. 不要再踩的坑

### 模型和环境

- 已尝试Qwen3-VL-30B-A3B-Instruct-AWQ并在4090单卡OOM，用户明确回退8B。A3B是每token激活参数，不是只需加载3B权重。
- vLLM0.11需要Torch2.8，与服务器glibc2.27的可用wheel不匹配。官方pip只显示到Torch2.6是平台筛选，不是换镜像就能解决。
- 当时Docker/Apptainer/Singularity均未安装，用户担心影响服务器。不要擅自安装系统服务、升级glibc/驱动或重启网络。
- AutoAWQ0.2.9停更，TF4.57.1缺PytorchGELUTanh；曾做进程内兼容。Qwen3-VL MoE的独立专家checkpoint与TF堆叠专家Parameter不匹配，触发缺失权重随机初始化而极慢；拆专家后开始加载但仍OOM。该路线已放弃。
- 当时AWQ GEMM不支持auto device_map中的CPU/disk offload；不能无依据承诺24GB能加载。
- 旧faulthandler的Timeout栈是诊断，不是每分钟终止模型。用户很反感持续输出；当前原版加载路径已无此逻辑。
- 环境保留AutoAWQ不等于当前8B/CLIP使用它，不要无故卸载/降级。
- GAP/CLIP仓库的clip目录是代码，clip.load首次下载缓存到~/.cache/clip，后续本地校验读取。用户默认缓存不存在，但后续已另行准备Transformers权重。
- 官方CLIP .pt与本脚本Transformers .bin/config/tokenizer不能改名互换。已跑通后不要无必要切换加载方式。

### 网络与文件传输

- 服务器HF/GitHub/镜像/ModelScope/百度均解析失败。resolv.conf=127.0.0.53，但resolved确实监听；上游8.8.8.8、114.114.114.114、fe80::216:3eff:fe4d:a68b。
- nslookup直连223.5.5.5超时，curl绕过DNS/代理访问1.1.1.1也超时，用户说校园网认证页打不开。更倾向机房网络/路由/认证故障，不是Git权限或Python问题。
- 提供过有条件的临时DNS设置，没有证据执行；不要说已更改或已修复网络。
- 用户通过Windows自带mstsc连接远程桌面，不是SSH软件。thinclient_drives只有.clipboard，发生Bad file descriptor。最终 **本地文件复制→Ubuntu文件管理器粘贴**成功，已经用于修复脚本和权重传输；不要再反复要求git pull/SFTP。
- 粘贴到真实项目或Desktop目录，不是thinclient_drives目录。
- 本地py不存在，python是WindowsApps占位程序，无输出不表示成功。最后使用Windows自带curl.exe下载7个CLIP文件再复制到服务器；不需要安装PyTorch。

### 实验解释

- 标签仅用于采样/诊断，不回填类别名，不根据标签删除坏描述来抬指标。
- 不混淆accepted、同类cosine、gap、AUC、近邻纯度和LP准确率。
- 不把6400维原生特征换成256维全局平均后仍称同一实验。
- 不把人物拆成独立样本，不混淆micro-batch、梯度累积batch和关系矩阵batch。
- 这轮是训练集关系分析，不能说验证了测试泛化或接入文本后的训练收益。

## 8. 新会话快速开始

先读第1、5、6节，再读 `handoff_artifacts/stage1_text_geometry_summary.json` 和 `compare_stage1_text_geometry.py`。用户接下来需要解释当前文本近邻错误来源，不需要从头准备环境和生成数据。

写交接时无新训练/推理进程在本地启动。仓库修改/提交状态以git status为准，不自动提交。
