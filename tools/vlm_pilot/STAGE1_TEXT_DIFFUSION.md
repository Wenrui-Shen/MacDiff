# Stage1 全局文本条件与双向噪声预测

本实现使用全部 NTU60 XSub train accepted 描述；不重新生成文本、不用类别标签选择文本、不启用 OSE/Stage2。原始 MacDiff 配置和模型仍可单独运行。以下命令均在服务器项目根目录执行。

## 1. 缓存冻结 CLIP

在已跑通 CLIP 提取的 skeleton_vlm 环境中运行（不要求训练环境安装 Transformers）：

```bash
OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 python -u cache_clip_text.py --data-path ../data/MAMP/ntu/NTU60_XSub.npz --captions vlm_pilot/ntu60_xsub_train_person_captions_8b_single_gpu.jsonl --clip-model /home/user9/public3/swr/models/clip-vit-base-patch32 --output-dir vlm_pilot/ntu60_xsub_clip_cache_v1 --batch-size 64 --resume
```

- 固定本地 HF CLIP 权重，eval + inference_mode，投影后 text_embeds，FP32，每人 L2 归一化。
- 按 x_train 原始行号写入 `[N,2,512]`，空人物为零，另存 `[N,2]` person_valid。
- 同样本重复记录取最后一条 accepted；历史 invalid/pipeline_error 不覆盖成功记录。缺失任何训练样本、混合模型/提示词版本、超过 CLIP token 上限均报错。
- 输出 `person_features.npy`、`person_valid.npy`、`samples.json`、`manifest.json`。最后一个文件记录数据/文本/模型/实现的 SHA256、提取进度和完成标记。
- 每个 microbatch 先 flush 数组、再原子更新进度。`--resume` 续跑不完整缓存；完整缓存只校验后复用，不加载 CLIP。改变输入/实现后使用新目录，不覆盖旧缓存。
- 数据文件校验是内容 SHA256，与路径无关；因此复制到新位置不会仅因路径或修改时间不同而失效。首次校验需顺序读取 NPZ 和模型权重。
- 缓存保存的是 remap 前的特征。训练不加载 CLIP，也不缓存持续更新的 remap 输出。

## 2. 训练定义

训练时对有效人物的 CLIP 向量取均值并再次 L2，得到每个样本唯一的全局条件 e。保留每人缓存，便于以后改变聚合方式。当前不做逐人语义匹配、阶段对齐或按标签采样。

`r = LayerNorm(MLP(e))`：512→512→256、GELU、最后 LayerNorm 无 affine 参数，最后归一化用 FP32。两条文本分支用同一个 r。

1. 原始分支：`D_S(x_t,t,h)` 预测骨架噪声，保留原有全局/局部条件、masked MSE、0.02 token uniformity、0.1 条件丢弃率。
2. 文本→骨架：`D_TS(x_t,t,r)` 预测相同骨架噪声。与原始分支复用本次 x_t、t、noise、mask，使用独立的同结构骨架 decoder；r 广播到全部时空位置，不输入 h，无额外全局条件丢弃。更新 remap 和该 decoder。
3. 骨架→文本：先 `target = r.detach()`，独立采样 u、epsilon_r，`r_u = q_sample(target,u,epsilon_r)`；条件残差 MLP `D_ST(r_u,u,h_global)` 预测 epsilon_r。更新骨架 encoder 和文本 decoder，不更新 remap。向量 MSE 对 batch/维度取均值，不对 r_u 额外归一化。

每次只编码一次骨架。文本 decoder 默认隐藏维数 256、3 个条件残差 MLP 块；每个块用骨架条件与时间步生成缩放/偏置。

`L = L_diff + 0.02 L_uniformity + lambda_text_to_skeleton L_TS + lambda_skeleton_to_text L_ST`

两个新增权重默认均为 1.0，只是起始消融值，不表示已验证最优。三个目标从第一步同时参与，没有 loss warm-up/分阶段训练。保留原来的学习率 warm-up，这与 loss 权重调度不同。文本沿用 1000 步 inverse_cosine 日程和均匀时间步采样。

原始训练模型 `one_person` 默认 True（只编码第一个人），新配置明确保留该值；全局文本仍聚合样本内所有有效描述。若将来设置 False，文本条件广播给两个人；新增骨架去噪损失排除空人物，反向条件仅池化有效人物，原始分支保持原有人物处理。数据裁剪仍为 50%～100%，整段文本作为全局弱条件。

日志分别记录 `loss_diff`、`loss_uniformity`、`loss_text_to_skeleton`、`loss_skeleton_to_text`、`text_energy` 和 `text_batch_variance`，最后两项检查尺度和跨样本退化，不能代替下游评估。

## 3. 运行

使用原来能够运行 MacDiff Stage1 的训练环境，不要求切换到 skeleton_vlm。

先做两个真实 GPU 训练 step（独立输出目录，不占用正式 checkpoint）：

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 python -u main_pretrain.py --config config/ntu60_xsub_joint/pretrain_madiff_text.yaml --batch_size 2 --accum_iter 1 --max_train_steps 2 --epochs 1 --warmup_epochs 0 --min_lr_epochs 0 --num_workers 0 --output_dir output_dir/ntu60_xsub_macdiff_text_smoke --log_dir output_dir/ntu60_xsub_macdiff_text_smoke/tensorboard
```

正式训练（仅 GPU 1，microbatch 16 × 梯度累积 4 = 有效 batch 64，与原始启动脚本双卡各 32 相同；显存峰值尚未在服务器验证）：

```bash
bash script_pretrain_macdiff_text.sh
```

固定权重消融必须使用独立输出目录，例如：

```bash
OUTPUT_DIR=output_dir/ntu60_xsub_macdiff_text_w01 bash script_pretrain_macdiff_text.sh --lambda_text_to_skeleton 0.1 --lambda_skeleton_to_text 0.1
```

权重为零会关闭对应分支并冻结无训练来源的参数。特别是 `lambda_text_to_skeleton=0` 时 remap 保持随机初始化且冻结，这个消融**不等同于直接恢复原始 CLIP 特征**。两个权重都为零直接调用原始 forward。一般原始基线仍使用原来的脚本。

恢复训练使用本方案同架构、同缓存、同 loss 权重的 checkpoint：

```bash
bash script_pretrain_macdiff_text.sh --resume output_dir/ntu60_xsub_macdiff_text/checkpoint-100.pth
```

请指定真实存在的文件；实际保存按原有频率为 checkpoint-0、10、20…以及最后一轮。不能把原始 Stage1 checkpoint 当作本方案完整 resume；新增模块及 optimizer 状态不同。下游线性评估仍可提取原生 encoder，新增模块保持独立前缀。

## 4. 验证

```bash
python -m unittest tests.test_clip_text_cache tests.test_stage1_text_geometry tests.test_macdiff_text -v
```

缓存测试用模拟 encoder 验证数据流，不等同于真实 CLIP 推理；训练测试使用真实 PyTorch 小模型，覆盖三个损失、梯度隔离、空人物、关闭分支、原始模型等价性、encoder 转移、训练引擎索引查表和恢复身份检查。真实 CLIP 权重、完整数据、CUDA AMP 和显存需在服务器验证。
