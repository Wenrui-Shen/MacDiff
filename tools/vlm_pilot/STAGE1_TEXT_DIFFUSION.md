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

## 2. 当前训练定义：人物一一对应

训练通过 `util/person_text_cache.py` 读取原 v2 文件，不修改缓存提取代码或缓存身份。`one_person=True` 时仅取人物 0 的句向量和 token，人物 1 的长度不影响 K、特征不参与任何文本目标。`one_person=False` 时按 sample0/person0、sample0/person1、sample1/person0…展开为 B×2 行；不在 token 维拼接不同人物，也不跨人物平均句向量。

每行是一个骨架及其描述：句向量 `[A,512]`；正文＋EOS token `[A,K,512]`，K 为所读取人物描述的最大有效长度，K≤76，保留原人物编号/位置及 padding mask。A 在模型过滤空骨架前为 B 或 2B；过滤后为有效骨架人数。有效骨架缺少对应文本时明确报错，不用另一个人的描述替代。空骨架不参与三个损失或 uniformity。

共享两层 remap MLP：512→512→256，中间 GELU。全局 `r=LN(MLP(该人物句向量))`；局部 `R=LN(MLP(tokens)+person_embedding+position_embedding)`。LN 无 affine、FP32。拼接 `M=[r;R]` 得 `[A,K+1,256]`，每个人有自己的全局 token，padding 屏蔽。

1. 原始骨架去噪：每个人独立编码为 75 个可见 token，加平均池化全局特征；恢复 750 位置条件后预测骨架噪声，masked MSE 只计算遮挡位置。保留 0.1 条件丢弃和 0.02 uniformity。
2. 文本→骨架：使用同一人的 M、同一份骨架噪声/time/mask。每层以当前带噪骨架状态为 Q、该人物文本 M 为 K/V，cross-attention 输出直接用作 MacDiff 调制条件，不额外广播相加 r。共享开关控制是否复用原生骨架 decoder 主体；不读取骨架 encoder 输出。
3. 骨架→文本：该人物 `M.detach()` 后加噪；同一人的全局池化＋75 个可见 encoder token 形成 `[A,76,256]` 骨架 memory，不跨人物汇总，也不 detach 骨架特征。5 层文本 Transformer 每层 cross-attention 读取→调制→masked self-attention→调制/FFN，输出 `[A,K+1,256]` 噪声。每个人独立采样一个时间步，所有文本 token 共用该时间步，各 token/通道噪声独立。decoder 自有全局类型/人物/位置 embedding，无干净文本内容额外输入。

反向损失为 `MSE(pred[valid], noise[valid])`，所有有效全局/局部 token 与通道一次平均，不单独加权。正向损失优化 remap；反向加噪前 detach 保证不优化 remap。每个保留人物的骨架只编码一次，三个任务同时参与，均预测噪声。正式 encoder 8 层，骨架/文本 decoder 均 5 层，256 维，8 heads。

`L = L_native + 0.02 L_uniformity + 1.0 L_TS + 1.0 L_ST`

保留 1000 步 inverse_cosine、均匀时间步采样和原学习率 warm-up。当前仍 `one_person=True`、共享骨架 decoder、batch 64。日志 text_energy/text_batch_variance 对应逐人全局 r；text_valid_tokens 对应参与文本任务人物的有效局部 token 数平均值。文本仍描述整段动作，不随骨架时间裁剪重新生成。

旧跨人物混合文本训练 checkpoint 不允许直接完整 resume；现在校验人物配对协议与 one_person 设置。新输出目录使用 `output_dir/ntu60_xsub_macdiff_person_text`，缓存仍使用原 v2 目录。

## 3. 运行

使用原来能够运行 MacDiff Stage1 的训练环境，不要求切换到 skeleton_vlm。

先做两个真实 GPU 训练 step（独立输出目录，不占用正式 checkpoint）：

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 python -u main_pretrain.py --config config/ntu60_xsub_joint/pretrain_madiff_text.yaml --batch_size 2 --accum_iter 1 --max_train_steps 2 --epochs 1 --warmup_epochs 0 --min_lr_epochs 0 --num_workers 0 --output_dir output_dir/ntu60_xsub_macdiff_person_text_smoke --log_dir output_dir/ntu60_xsub_macdiff_person_text_smoke/tensorboard
```

正式训练（仅 GPU 1，batch size 64、梯度累积 1，有效 batch 64；显存峰值尚未在服务器验证）：

```bash
bash script_pretrain_macdiff_text.sh
```

固定权重消融必须使用独立输出目录，例如：

```bash
OUTPUT_DIR=output_dir/ntu60_xsub_macdiff_person_text_w01 bash script_pretrain_macdiff_text.sh --lambda_text_to_skeleton 0.1 --lambda_skeleton_to_text 0.1
```

权重为零会关闭对应分支并冻结无训练来源的参数。特别是 `lambda_text_to_skeleton=0` 时 remap 保持随机初始化且冻结，这个消融**不等同于直接恢复原始 CLIP 特征**。两个权重都为零直接调用原始 forward。一般原始基线仍使用原来的脚本。

恢复训练使用本方案同架构、同缓存、同 loss 权重的 checkpoint：

```bash
bash script_pretrain_macdiff_text.sh --resume output_dir/ntu60_xsub_macdiff_person_text/checkpoint-100.pth
```

请指定真实存在的文件；实际保存按原有频率为 checkpoint-0、10、20…以及最后一轮。不能把原始 Stage1 或旧单向量或旧反向全局 decoder 版本 checkpoint 当作本方案完整 resume；新增模块及 optimizer 状态不同。下游线性评估仍可提取原生 encoder，新增模块保持独立前缀。

## 4. 验证

```bash
python -m unittest tests.test_clip_text_cache tests.test_stage1_text_geometry tests.test_macdiff_text tests.test_stage1_readout_torch -v
```

本地共 35 项测试通过，包含人物配对、空人物、缺失描述与单人模式排除第二人。缓存测试覆盖续跑、完整性检查、人物/位置/EOS 保留，并用真实 Hugging Face 小型随机 CLIP 验证 safetensors 和 bin 两种权重加载、编码、落盘及 EOS 特征一致性（没有下载官方预训练权重）。训练测试覆盖三个损失、梯度隔离、padding 不影响输出、有效 token 影响预测、全局 token 注意力梯度、反向统一 MSE 与 padding 屏蔽、人物元数据、空人物、关闭分支、原始模型等价性、encoder 转移、训练引擎索引查表和恢复身份检查。生产 YAML/参数解析和完整尺寸 CPU 前向也通过，输出为 `[1,750,12]`。真实预训练 CLIP 全量数据、CUDA AMP 和 batch 64 显存尚需在服务器验证。

## 5. 骨架 decoder 共享开关

配置 `model_args.share_skeleton_decoder: True` 共享骨架输入投影、时空位置编码、5 层调制/self-attention/FFN、末尾归一化和输出层；False 使用独立副本。文本 cross-attention 读取器仍独立，反向文本 decoder 不共享。模型构造默认 False 以兼容旧调用，正式 YAML 已明确设 True。

共享仍执行两次条件前向，两个损失共同更新主体；文本→骨架无直接骨架 encoder 梯度。主体只在原生名称下注册一次，避免 checkpoint 和优化器重复参数。关闭文本→骨架分支仅冻结其独立参数，原生主体继续训练。完整 resume 要求共享设置相同；旧 checkpoint 未记录此选项时按 False 处理。消融请使用不同 output_dir/log_dir。已有 v2 CLIP 缓存可复用，batch 64 不变。
