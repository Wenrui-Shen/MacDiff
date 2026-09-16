# Stage1 多 token 文本条件与双向噪声预测（v2）

本实现使用全部 NTU60 XSub train accepted 描述；不重新生成文本、不用类别标签选择文本、不启用 OSE/Stage2。原始 MacDiff 配置和模型仍可单独运行。以下命令均在服务器项目根目录执行。

当前采用双向多 token。正向将全局 token 与局部 token 拼接；反向恢复同一套全局＋局部表示，文本 decoder 为 5 层 Transformer，所有有效 token 一起计算平均 MSE。`model_args.share_skeleton_decoder` 控制两个骨架去噪过程是否共享主体，当前 YAML 为 True。缓存仍用 v2，无需重新编码已有 v2 缓存。

## 1. 缓存冻结 CLIP

在已跑通 CLIP 提取的 skeleton_vlm 环境中运行（不要求训练环境安装 Transformers）：

```bash
OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 python -u cache_clip_text.py --data-path ../data/MAMP/ntu/NTU60_XSub.npz --captions vlm_pilot/ntu60_xsub_train_person_captions_8b_single_gpu.jsonl --clip-model /home/user9/public3/swr/models/clip-vit-base-patch32 --output-dir vlm_pilot/ntu60_xsub_clip_cache_v2 --batch-size 64 --resume
```

- 固定本地 HF CLIP 权重，eval + inference_mode，投影后 text_embeds，FP32，每人 L2 归一化。
- 按 x_train 原始行号写入 `[N,2,512]`，空人物为零，另存 `[N,2]` person_valid。
- 同一次 CLIP 前向另取最终层 `last_hidden_state`，对每个 token 使用冻结的 `text_projection` 投影，再逐 token L2，保存 FP16 `[N,2,77,512]`。投影与句向量相同，不意味着每个词已获得与句向量相同的跨模态对齐质量。
- token mask `[N,2,77]` 保留内容 token 和真实 EOS，排除 BOS、padding、空人物；无效特征置零。另存 int32 token IDs，无效位置为 -1。人物维和原始 token 位置保留在数组布局中。
- 同样本重复记录取最后一条 accepted；历史 invalid/pipeline_error 不覆盖成功记录。缺失任何训练样本、混合模型/提示词版本、超过 CLIP token 上限均报错。
- 输出 `person_features.npy`、`person_valid.npy`、`token_features.npy`、`token_mask.npy`、`token_ids.npy`、`samples.json`、`manifest.json`。最后一个文件记录数据/文本/模型/实现的 SHA256、提取进度和完成标记。
- 每个 microbatch 先 flush 数组、再原子更新进度。`--resume` 续跑不完整缓存；完整缓存只校验后复用，不加载 CLIP。改变输入/实现后使用新目录，不覆盖旧缓存。
- 数据文件校验是内容 SHA256，与路径无关；因此复制到新位置不会仅因路径或修改时间不同而失效。首次校验需顺序读取 NPZ 和模型权重。
- 缓存保存的是 remap 前的特征。训练不加载 CLIP，也不缓存持续更新的 remap 输出。
- 40,091 样本的固定长 FP16 token 数组约 5.89 GiB，另外保存原有句向量及少量元数据。训练使用 mmap，每个 batch 只读取对应样本并打包有效 token，不将全量 token 特征载入内存/显存。启动时会读取并校验文件。
- 旧 v1 句向量不能恢复多 token，不能作为 v2 缓存使用；需要从已生成描述重新编码 CLIP，无需重新生成描述。旧目录不覆盖。

## 2. 训练定义

训练时对有效人物的句向量取均值并再次 L2，得到全局条件 e。同时按人物与原 token 位置顺序打包有效文本 token，得到 `[B,K,512]`；K 为当前 batch 中最大的有效 token 总数（最多 152），另传有效 mask、人物 ID 和原 token 位置。不对 token 平均，不做逐人骨架匹配或时间阶段强对齐。

共享两层 remap MLP：512→512→256，中间 GELU。全局路径 `r = LN(MLP(e))`；token 路径 `R = LN(MLP(tokens) + person_embedding + position_embedding)`，输出 `[B,K,256]`。人物与位置 embedding 可学习、std=0.02 初始化；最后 LN 无 affine 参数并用 FP32。无效输入在 remap 前清零、输出也清零，且被注意力 key padding mask 屏蔽。

1. 原始分支：`D_S(x_t,t,h)` 预测骨架噪声，保留原有全局/局部条件、masked MSE、0.02 token uniformity、0.1 条件丢弃率。
2. 文本→骨架：`D_TS(x_t,t,r,R,mask_text)` 预测相同骨架噪声。与原始分支复用本次 x_t、t、noise、mask，使用由共享开关决定的骨架 decoder。每个 block 新增一个 cross-attention 读取器：Q 来自当前带噪骨架 decoder 状态的 LN，K/V 来自拼接 memory `[r;R]`，形状 `[B,K+1,256]`，全局 token 始终有效，局部 padding 屏蔽。读取结果 `[B,750,256]` 直接形成 z，不再额外相加全局 r，再送入原有 FeatureModulation → self-attention → FeatureModulation → MLP。它不读取干净骨架 encoder h；更新 remap、人物/位置 embedding 和该 decoder。
3. 骨架→文本：`M=[r;R]` 在加噪前整体 detach。每个样本采一个时间步，所有 token 共用该时间步，噪声按 token/通道独立采样。骨架 memory 为全局有效人物池化＋可见 encoder token，默认单人 `[B,76,256]`，不 detach；空人物 token 屏蔽。文本 decoder 每层以带噪文本状态为 Q、骨架 memory 为 K/V，读取条件后按 MacDiff 的 FeatureModulation → self-attention → FeatureModulation → FFN 顺序更新。输出 `[B,K+1,256]` 噪声。

反向使用自身全局类型/人物/位置 embedding 提供结构信息，无干净文本内容输入。self-attention 不用因果 mask，使用有效 token mask。padding 在输入、层输出和损失中排除。损失为 `MSE(pred[valid], noise[valid])`，所有有效全局/局部 token 和通道一起平均，不单独加权，也不按样本长度归一化，长描述有更多有效项。反向更新骨架 encoder 和文本 decoder，不更新 remap 及正向人物/位置 embedding。

每次只编码一次骨架。正式配置三个 decoder 都为 5 层。文本 decoder 使用支持 padding mask 的 PyTorch 多头注意力，复用 MacDiff FeatureModulation 和 FFN 顺序，非原始骨架 decoder 类的直接替换。

`L = L_diff + 0.02 L_uniformity + lambda_text_to_skeleton L_TS + lambda_skeleton_to_text L_ST`

两个新增权重默认均为 1.0，只是起始消融值，不表示已验证最优。三个目标从第一步同时参与，没有 loss warm-up/分阶段训练。保留原来的学习率 warm-up，这与 loss 权重调度不同。文本沿用 1000 步 inverse_cosine 日程和均匀时间步采样。

原始训练模型 `one_person` 默认 True（只编码第一个人），新配置明确保留该值；文本条件包含样本内所有有效描述。若将来设置 False，两个骨架人物读取同一个样本级文本 memory，各自的 Q 不同；新增骨架去噪损失排除空人物，反向条件仅池化有效人物，原始分支保持原有人物处理。数据裁剪仍为 50%～100%，整段文本作为全局弱条件。

日志分别记录 `loss_diff`、`loss_uniformity`、`loss_text_to_skeleton`、`loss_skeleton_to_text`、`text_energy`、`text_batch_variance` 和 `text_valid_tokens`。energy/variance 仍对应全局 r；token 数是每个样本有效 token 总数的 batch 均值。这些指标不能代替下游评估。

## 3. 运行

使用原来能够运行 MacDiff Stage1 的训练环境，不要求切换到 skeleton_vlm。

先做两个真实 GPU 训练 step（独立输出目录，不占用正式 checkpoint）：

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 python -u main_pretrain.py --config config/ntu60_xsub_joint/pretrain_madiff_text.yaml --batch_size 2 --accum_iter 1 --max_train_steps 2 --epochs 1 --warmup_epochs 0 --min_lr_epochs 0 --num_workers 0 --output_dir output_dir/ntu60_xsub_macdiff_bidirectional_tokens_smoke --log_dir output_dir/ntu60_xsub_macdiff_bidirectional_tokens_smoke/tensorboard
```

正式训练（仅 GPU 1，batch size 64、梯度累积 1，有效 batch 64；显存峰值尚未在服务器验证）：

```bash
bash script_pretrain_macdiff_text.sh
```

固定权重消融必须使用独立输出目录，例如：

```bash
OUTPUT_DIR=output_dir/ntu60_xsub_macdiff_bidirectional_tokens_w01 bash script_pretrain_macdiff_text.sh --lambda_text_to_skeleton 0.1 --lambda_skeleton_to_text 0.1
```

权重为零会关闭对应分支并冻结无训练来源的参数。特别是 `lambda_text_to_skeleton=0` 时 remap 保持随机初始化且冻结，这个消融**不等同于直接恢复原始 CLIP 特征**。两个权重都为零直接调用原始 forward。一般原始基线仍使用原来的脚本。

恢复训练使用本方案同架构、同缓存、同 loss 权重的 checkpoint：

```bash
bash script_pretrain_macdiff_text.sh --resume output_dir/ntu60_xsub_macdiff_bidirectional_tokens/checkpoint-100.pth
```

请指定真实存在的文件；实际保存按原有频率为 checkpoint-0、10、20…以及最后一轮。不能把原始 Stage1 或旧单向量或旧反向全局 decoder 版本 checkpoint 当作本方案完整 resume；新增模块及 optimizer 状态不同。下游线性评估仍可提取原生 encoder，新增模块保持独立前缀。

## 4. 验证

```bash
python -m unittest tests.test_clip_text_cache tests.test_stage1_text_geometry tests.test_macdiff_text tests.test_stage1_readout_torch -v
```

本地共 31 项测试通过。缓存测试覆盖续跑、完整性检查、人物/位置/EOS 保留，并用真实 Hugging Face 小型随机 CLIP 验证 safetensors 和 bin 两种权重加载、编码、落盘及 EOS 特征一致性（没有下载官方预训练权重）。训练测试覆盖三个损失、梯度隔离、padding 不影响输出、有效 token 影响预测、全局 token 注意力梯度、反向统一 MSE 与 padding 屏蔽、人物元数据、空人物、关闭分支、原始模型等价性、encoder 转移、训练引擎索引查表和恢复身份检查。生产 YAML/参数解析和完整尺寸 CPU 前向也通过，输出为 `[1,750,12]`。真实预训练 CLIP 全量数据、CUDA AMP 和 batch 64 显存尚需在服务器验证。

## 5. 骨架 decoder 共享开关

配置 `model_args.share_skeleton_decoder: True` 共享骨架输入投影、时空位置编码、5 层调制/self-attention/FFN、末尾归一化和输出层；False 使用独立副本。文本 cross-attention 读取器仍独立，反向文本 decoder 不共享。模型构造默认 False 以兼容旧调用，正式 YAML 已明确设 True。

共享仍执行两次条件前向，两个损失共同更新主体；文本→骨架无直接骨架 encoder 梯度。主体只在原生名称下注册一次，避免 checkpoint 和优化器重复参数。关闭文本→骨架分支仅冻结其独立参数，原生主体继续训练。完整 resume 要求共享设置相同；旧 checkpoint 未记录此选项时按 False 处理。消融请使用不同 output_dir/log_dir。已有 v2 CLIP 缓存可复用，batch 64 不变。
