# 文本到缓存到 Stage1 训练：检查与新版接入（2026-10-01）

检查范围是当前本地代码和 Git 中的历史改动。本地没有服务器数据、实际 captions/cache 或 PyTorch；因此这是代码级检查和 NumPy/模拟编码器验证，不能替代服务器全量数据校验或 GPU/DDP smoke。

## 1. 旧链路和实际修正

旧 accepted JSONL → `compare_stage1_text_geometry.read_captions` → `cache_clip_text.py` → v2 NPY + manifest → `util/person_text_cache.py` → `engine_pretrain.py` 按 feeder 返回的原始索引查表 → `transformer_macdiff_text.Transformer` → 骨架 encoder → 下游 LP。

| 项目 | 历史或当前情况 | 能否在缓存生成阶段处理 |
|---|---|---|
| 人物配对 | 早期用两人句向量聚合、拼接两人 local，再给保留的骨架使用。`879c232` 增加逐人 reader 和模型逐人 memory；现在 person0 只配 person0，双人按 sample/person 展开。 | 保存与校验原始人物槽位可以提前；训练中的逐人选择和空 crop 过滤必须保留。只改缓存不能修正旧模型跨人物混合的计算。 |
| 文本尺度 | v2 缓存正确保存 L2 向量；512 维单位 L2 向量平均通道能量约 1/512。后来固定 CLIP 与 remap 模式在训练中增加 RMS，使固定目标能量约 1。 | 可把固定向量的 RMS 前移到新缓存。它是扩散目标尺度的选择，不是 CLIP 提取算错；新缓存不能与原尺度实验混用。 |
| local 定义 | `b3eda18` 从句向量增加词元缓存。每个 BPE 末层状态经 CLIP projection 得到向量，这不代表每个词元都是一个完整动作/部位语义。 | 新版独立编码六个完整部位句子，分别取 projected `text_embeds`。不能靠平均旧 BPE 或重命名槽位得到新的部位描述。 |
| EOS 重复 | v2 保留真实 EOS；该句 EOS 的投影特征与该句 global 向量相同（FP16 有小量误差，已有测试覆盖）。因此 global 内容又作为一个 local 进入模型。 | 新版一个 global 句向量加六个不同部位句向量，消除由相同句子的 EOS 引起的结构性重复。保留旧 v2 定义以支持历史实验。 |
| 覆盖、顺序与身份 | v2 按 sample_index 写行、缺样本报错、最后一条 accepted 覆盖之前 accepted，混合模型/提示词报错；保存 dataset/captions/CLIP 的 SHA256。 | 新版同样验证全部原始训练行，并拒绝重复 index 和部位缺失。另在编码前对照实际 x_train 人物槽位，避免把“有文本”直接当作“有骨架”。 |
| 空人物与 padding | v2 提取已排除 BOS/padding、空人物置零。训练进一步在标准化前过滤空骨架，覆盖随机裁剪后 person0 为空的情况。 | 固定空人物和六个槽位可提前规范；随机 crop 的空人物无法在静态缓存中判断。 |
| 输入与目标库 | 固定 CLIP 通过在线 remap 进入 T→S；S→T 用固定/EMA/逐样本历史目标。逐样本方案在成功 optimizer step 后更新 0.9 old + 0.1 current remap。 | 缓存只保存固定 CLIP；在线 remap、目标历史、AMP 跳步和 checkpoint 配对均必须留在训练。 |

当前 v2 链路已应用人物配对、padding、空 crop、RMS 和 resume 身份检查，未发现可以仅通过重写固定缓存解决的剩余索引/人物配对错误。仍需服务器检查真实文本质量和文件一致性。

## 2. 新版 v3 句级缓存（已接入代码）

`cache_clip_motion_text.py` 接收新生成的 text-only `.json` 数组或 `.jsonl`，并读取同名 `.metadata.json` 获取模型、提示词和 revision。不同 caption 版本不能混合；任何训练样本缺失都不能生成 complete 缓存。

每个人编码七个独立句子：global、head、torso、left_arm、right_arm、left_leg、right_leg。固定 CLIP eval/inference，每个句子独立检查 BOS/content/EOS 与 77-token 上限，不静默截断、不拼成一条长描述。只保存各句的 projected pooled `text_embeds`，逐向量 RMS 后保存 FP32，未添加人物/部位 embedding，未经过可训练 remap。

缓存定义为 `macdiff_clip_sentence_cache_v3`，典型维度为：

- `person_features.npy`: `[N,2,512]`，global 句向量。
- `token_features.npy`: `[N,2,6,512]`，六个部位句向量。文件名沿用现有接口，但含义是句子，不是 BPE。
- `person_valid.npy`: `[N,2]`，对照实际 x_train 的人物非零槽位。
- `token_mask.npy`: `[N,2,6]`，有该人物即六个部位都有效，静止部位的姿态句也有效。
- `samples.json` 和 `manifest.json`：描述副本、部位顺序、归一化、输入与实现身份、进度及校验和。

v3 不保存 BPE token_ids，因为六个槽位与分词器位置无关。`context_length=7` 是模型的 global + 六部位结构容量；`clip_context_length=77` 是 CLIP 句子编码上限，二者不能混用。

40091 行、两个人槽位全部分配时，global/local FP32 特征约 1.07 GiB；person0 的七向量目标库约 0.535 GiB（另有元数据）。现有 reader 自动识别 v2/v3，v2 原有提取文件和协议未改写。v3 通过 `text_positions=1..6` 固定绑定六个部位，复用现有位置 embedding；这不是时间位置，也不意味着关节到部位的硬 attention mask 已实现。

已有训练接口得到 `[B,512]` global 和 `[B,6,512]` local；双人时得到 `[2B,...]`。固定 RMS、参数 EMA 和逐样本目标库都可使用这一接口。训练中的 RMS 操作保留，作用于已 RMS 的输入近似幂等，维持旧代码的稳定性；动态混合目标不能强行重新归一化，否则会改变用户确认的 0.9/0.1 公式。

如果缓存阶段报告人物槽位不匹配，应检查相关 sample 的渲染覆盖和文字。32 帧采样可能没有展示原序列中短暂出现的人；此时应修正该样本的渲染/生成结果，不能用另一人的描述填空。

## 3. 缓存不能解决、仍需实验处理的事项

1. **整段文本与随机时间裁剪。** 当前 p_interval=[0.5,1]；六个部位的整段主要动作可能不出现在某次 crop 中。全局描述也有此问题。可另做完整序列 p_interval=[1] 消融，或今后增加有时间边界的文本再按 crop 匹配。静态六部位缓存无法推断每次 crop 可见的动作。
2. **左右侧语义。** 当前渲染只是前/侧投影和红蓝人物颜色，没有逐关节的左/右提示。新版格式检查无法证明 VLM 的 left_arm/right_arm 内容识别正确，应抽检。若启用 feeder.flip，需要同时交换对应部位并处理方向文字；目前没有 flip 标记供训练映射，所以新版配置明确 flip=False，训练启动也拒绝 v3 搭配 flip=True。随机旋转与方向文字也需要独立考虑。
3. **global/local 的实际 loss 权重。** S→T 仍一次平均所有有效向量。旧平均约 20 个 local 时 global 约占 1/21，新六部位时是 1/7。因此外层 S→T 权重相同也不是完全相同的内部监督分配；新配置保留现有公式，便于先跑通。以后可独立实验 `w_g L_g + w_l mean(L_region)`。
4. **uniformity。** 当前仅同一样本 local 的平方余弦，含对角项。六向量的对角下限是 1/6，旧约二十词元约 1/20，日志不能直接比较。强迫左右臂/腿的同步动作句互相正交可能不符合语义。fixed RMS 首轮不启用文本 uniformity；sample_target 配置保留原 0.02，之后可单独做权重为 0 的消融。
5. **S→T 输出维度。** 首轮256隐藏、512输出带末端LayerNorm和仿射线性头，对各向同性噪声有约0.502的期望MSE下限。减少token数不会改变这个问题；后续用户已授权修复，最新sentence配置改为512/无末端LN，见第6节。
6. **渲染信息。** 现有视频逐帧减去 person0 的 root，未展示绝对位移；global 描述无法恢复被去掉的世界坐标运动。当前缓存不会给缺失信息补造特征。
7. **原始数据来源证据。** 新缓存会绑定当前 NPZ 的 SHA256并校验其人物槽位，但已有渲染 provenance 主要是路径/大小/mtime，不能单凭新缓存证明旧 captions 一定来自内容相同的 NPZ。首次应核对生成时实际数据文件；后续训练则用缓存 dataset SHA256严格校验。

## 4. 运行与新训练

用户当前选择保留原 uniformity，不实施跨样本方差正则。2026-10-01 修改了训练输出：不再打印或记录空骨架数量，仍在内部过滤空人物；取消 `text_target_drift_mse`；`text_batch_variance` 改为在线 remap 后的 global 与所有有效 local 向量合并后的逐通道总体方差均值（不含结构 embedding、padding 和非活动人物）。fixed_clip 没有 remap，统计固定输入。该值是每卡当前 microbatch 的合并统计，日志沿用原有各卡/各步汇总；包含部位之间的差异，不能单独证明跨样本没有坍缩。`text_energy` 仍是目标 global 能量，不能把两者当作同一组向量的指标。

先同步 `cache_clip_motion_text.py`、`util/structured_text_cache.py`、`util/person_text_cache.py`、`main_pretrain.py`、`model/transformer_macdiff_text.py`、`engine_pretrain.py` 和两个新的 sentence YAML。sample_target 配置还依赖上次的共享内存性能版文件。首次真实训练保留完整 cache validation；通过后才使用 `--skip_text_cache_validation` 的 header/size 快速模式。该模式不能发现同大小内容篡改。

在之前能提取 CLIP 的环境中生成新缓存（每 batch 16 人，即 112 句）：

```bash
cd /home/user9/public3/swr/MacDiff && OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 python -u cache_clip_motion_text.py --data-path ../data/MAMP/ntu/NTU60_XSub.npz --captions vlm_pilot/ntu60_xsub_global_local_v2/captions.json --clip-model /home/user9/public3/swr/models/clip-vit-base-patch32 --output-dir vlm_pilot/ntu60_xsub_clip_sentence_cache_v3 --batch-size 16 --resume
```

中断后原命令继续；先 flush 每人的七向量，再更新 completed_persons；已完成缓存验证后复用，无需加载 CLIP。更改文本、元数据、模型或实现时用新输出目录。

在原 MacDiff 环境中做固定 RMS 两卡 smoke：

```bash
cd /home/user9/public3/swr/MacDiff && OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10252 main_pretrain.py --config config/ntu60_xsub_joint/pretrain_madiff_text_sentence_fixed_rms.yaml --batch_size 2 --accum_iter 1 --epochs 1 --warmup_epochs 0 --min_lr_epochs 0 --max_train_steps 2 --num_workers 0 --output_dir output_dir/ntu60_xsub_sentence_fixed_smoke --log_dir output_dir/ntu60_xsub_sentence_fixed_smoke/tensorboard
```

smoke 通过后，先跑固定 RMS、S→T 权重 0.1，比较文本版本；再跑与当前逐样本方案相同的共享 decoder 配置。这两种配置区别是已有的训练目标方案，不应混为文本版本的单变量对照：

```bash
cd /home/user9/public3/swr/MacDiff && OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10252 main_pretrain.py --config config/ntu60_xsub_joint/pretrain_madiff_text_sentence_fixed_rms.yaml --batch_size 32 --accum_iter 2
```

```bash
cd /home/user9/public3/swr/MacDiff && OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10254 main_pretrain.py --config config/ntu60_xsub_joint/pretrain_madiff_text_sentence_sample_target_blend_shared.yaml --batch_size 32 --accum_iter 2
```

使用新的 output_dir 从头训练；旧 checkpoint 的 cache 身份、结构 embedding 大小和目标库不匹配，不能当作新方案的完整 resume。新实验中断后仍使用该实验自己的 checkpoint；sample_target 还需对应目标库快照。LP 仍只加载骨架 encoder，新文本不进入下游推理。

CPU 验证命令：

```bash
python -m unittest tests.test_structured_text_cache tests.test_clip_text_cache tests.test_sample_text_target_bank tests.test_shared_memory_text_target_bank -v
```

覆盖新 JSON/JSONL 与 metadata、整句编码、六部位/双人顺序、RMS、NPZ 人物校验、覆盖与重复拒绝、过长句拒绝、生成中断恢复、旧 v2 reader 回归、SQLite/RAM 六部位目标更新和配对快照恢复。另有 PyTorch S→T 梯度检查，本机缺少 Torch 时明确跳过。

本次连同旧 geometry/model 回归共收集 63 项：33 项 CPU 检查通过，30 项依赖 PyTorch/Transformers 或 PyYAML 的检查跳过。新增代码通过 Python 3.8 语法检查和 CLI help 检查。服务器实际 CLIP 编码、全量缓存、训练前向/梯度及两卡 DDP 尚未验证。

## 5. 2026-10-02：首轮结果与padding检查

用户已提供新版400轮预训练、100轮LP日志，LP best 85.801795%（epoch87）、Top5 97.531538%；这一轮T→S/S→T均为0.1。没有明显训练中断或非有限值。与历史85.82%和原MacDiff约85.86%接近；历史LP batch等协议不完全一致，不能据此判定新版文本有提升或下降。

77只用于离线CLIP句子编码的上限，`padding=True` 补到当前编码batch最长句。v3缓存只有六个local句向量，配置context_length=7，训练attention实际收到global+6local；模型默认77仅保留旧v2兼容，不会自动补齐。共享目标库按有效人物/向量打包，也没有77槽位。

本次修复/优化：

- `util/person_text_cache.py`：全六句或全空的v3人物批量读取，保持原样本/人物顺序、单人只读取person0、缺失槽位清零；v2和部分mask继续原压缩路径。
- `util/sample_text_target_bank.py`：先按active过滤空crop人物，部分local有效时仅对有效向量归一化/remap，再散射回原槽位；全有效六句仍走dense路径。保留post-step权重、重复样本更新次序、0.9old+0.1online公式及快照格式；共享内存后端继承同一实现。
- `engine_linprobe.py`：两个evaluate入口把CE按实际batch样本数累计，避免小末batch和完整batch等权造成偏差。Top1/Top5原本已正确加权，不受修改影响。

不修改cache生成/身份绑定的四份源码，不需重生成cache；不修改模型参数形状，同实验checkpoint兼容。新版有效人物本来没有文本padding，性能收益主要是CPU逐行打包和少量空crop的无效remap，尚无GPU/epoch提速证据。

当前回归75项：45项CPU通过、30项依赖跳过；新增4项reader、5项目标更新、3项LP聚合检查。目标更新和LP测试以NumPy-backed Torch API运行实际方法，未验证真实CUDA/AMP/DDP；六份改动Python通过3.8 AST检查，diff whitespace检查通过。

结果仍需关注：S→T最后0.510913接近256→512仿射输出的各向同性噪声期望MSE下限约0.5；text uni最后0.167875接近六句下限1/6，仅说明同样本部位接近正交，不能排除各部位跨样本变成固定原型；合并variance=0.926353可由部位差异维持，也不能排除这一情况。首轮LP best的train CE=0.466444/test CE=4.616537，需logits/错误置信度分析；不能凭此直接认定BN或标签错误。完整描述与0.5..1随机时间crop/90%mask的监督可见性仍未检验。暂未扩大decoder、改uniformity/crop或添加正则。

## 6. S→T维度瓶颈修复（用户授权）

用户随后要求修复并询问512或1024。采用512：两份sentence YAML显式设置 `text_decoder_hidden_dim: 512`、`text_decoder_output_norm: 'none'`，并使用独立 `_st512` 输出/log目录。仅取消decoder最后的输出LayerNorm，block内归一化不变。原LayerNorm将H维表示约束到H−1维仿射空间，因此只把宽度改512仍有1/512的小秩限制；取消末端LN后，512→512线性头可覆盖完整噪声空间。

以512目标、256骨架、5层、7向量计算，512新decoder约27,595,776参数，1024约107,621,888参数（保留LN的计算，去除仅少2048）；1024约为512的3.9倍，先以512解除瓶颈再考虑容量消融。这是结构分析，不是GPU性能或LP增益证据。

模型新增可选 `text_decoder_output_norm`，默认保留历史 `layernorm` 和256宽度；输出仍采用Sequential，线性层保持 `output.1` 参数名。历史256配置/checkpoint可复现；新配置不能完整resume旧256模型，需要新预训练。旧checkpoint仍可按原方式做LP。cache来源身份、目标递推、loss权重、骨架encoder及共享骨架decoder不变。更新双卡训练/LP串联命令见handoff第6.4节。

结构修复回归共81项：49项CPU通过、32项依赖跳过。新增 `tests.test_text_decoder_width` 的4项CPU检查通过，运行实际decoder构造器及NumPy-backed输出层，覆盖512全部输出方向、历史LayerNorm、非法norm和两份配置；2项真实Torch完整前向/梯度及checkpoint检查因缺依赖跳过。模型/测试通过Python3.8 AST和diff whitespace检查，未做服务器AMP/DDP/显存或LP验证。
