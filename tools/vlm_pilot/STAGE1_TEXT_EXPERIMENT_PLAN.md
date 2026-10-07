# Stage1 文本实验待测变量与顺序（2026-10-07）

## 当前选择：No-share 的 S→T 0.1 → 0.5

用户最新决定继续文本版no-share，并明确选择S→T=0.5，替代助手建议的0.2。保持T→S=1、归一化A、512/无末端输出LN、sample_target_blend及目标更新比例0.1不变。已完成父对照LP best86.196021%、末20均值86.000728%；shared512口述best85.79%。原版B本次best85.183164%，不叠加到文本版。完整分析见 [NOSHARE_NORMB_LOG_ANALYSIS.md](D:/program/MacDiff/tools/vlm_pilot/NOSHARE_NORMB_LOG_ANALYSIS.md)。

沿用配置：[pretrain_madiff_text_sentence_sample_target_blend_noshare_st02.yaml](D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_text_sentence_sample_target_blend_noshare_st02.yaml)。文件名和YAML默认权重仍为0.2，由CLI显式覆盖为0.5；无需修改该文件。从头PT400、LP100；双卡PT每卡32/accum2，LP每卡64/accum1，有效batch均128，seed0，LP lr0.1/dist_eval。复用现有cache，输出与日志目录全部使用_s2t05；助手未启动训练，用户尚未提供进度或结果。参数更改不能当作原0.1实验的完整resume。

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10260 main_pretrain.py --config config/ntu60_xsub_joint/pretrain_madiff_text_sentence_sample_target_blend_noshare_st02.yaml --lambda_text_to_skeleton 1 --lambda_skeleton_to_text 0.5 --batch_size 32 --accum_iter 2 --epochs 400 --seed 0 --output_dir output_dir/ntu60_xsub_macdiff_sentence_sampletarget01_noshare_st512_t2s1_s2t05 --log_dir output_dir/ntu60_xsub_macdiff_sentence_sampletarget01_noshare_st512_t2s1_s2t05/tensorboard && OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10261 main_linprobe.py --config config/ntu60_xsub_joint/linprobe_madiff.yaml --finetune output_dir/ntu60_xsub_macdiff_sentence_sampletarget01_noshare_st512_t2s1_s2t05/checkpoint-399.pth --output_dir output_dir/ntu60_xsub_macdiff_sentence_sampletarget01_noshare_st512_t2s1_s2t05_lp_399_bs64 --log_dir output_dir/ntu60_xsub_macdiff_sentence_sampletarget01_noshare_st512_t2s1_s2t05_lp_399_bs64/tensorboard --batch_size 64 --accum_iter 1 --epochs 100 --lr 0.1 --seed 0 --dist_eval
```

判断先看LP best、末20轮均值与末轮，再看S→T后期曲线、native/T→S是否变化。更低文本loss不能单独证明表征改善；单次训练差异仍需后续确认。

## 以下保留2026-10-04历史规划，当前状态以上述选择为准

用户已授权将骨架输入归一化 A/B 加入待测试变量。本文件记录候选方案与建议顺序；未切换训练配置、未实现新模型结构、未启动服务器训练。用户现已提供512修复后的400epoch预训练日志，并口述LP best=85.79%；尚无本轮完整LP日志。日志分析见 [ST512_TRAINING_LOG_ANALYSIS.md](D:/program/MacDiff/tools/vlm_pilot/ST512_TRAINING_LOG_ANALYSIS.md)。

会话结束时用户决定：checkpoint-150的LP首轮仅72多，低于399的首轮74.65，不再继续该LP；继续S→T=1实验，随后根据结果测试归一化B。150的最终LP未完成，不能把首轮当最终分数。以下路线已按这一最新选择更新。

## 当时的共享对照组

| 项目 | 固定设置 |
|---|---|
| 数据/文本 | NTU60 XSub，当前 global + 六部位 v3 cache，person0 |
| 文本干净目标 | sample_target_blend，RMS，逐样本0.9旧目标+0.1当前remap |
| S→T decoder | hidden512，output_norm=none，depth5 |
| 骨架 decoder | share=True；共享原生重建与T→S主体 |
| 文本方向权重 | T→S=1，S→T=0.1；CLI显式覆盖，sentence YAML的T→S默认仍为0.1 |
| 归一化 | A：mean=[-0.0058,-0.1333,-0.0246]，var=[0.0206,0.0805,0.0218] |
| 扩散 | 各分支epsilon；现有inverse_cosine/1000步及均匀时间步不变 |
| 其他 | crop=[0.5,1]，mask=0.9，现有uniformity，不加跨样本目标 |
| 训练 | 双卡，每卡batch32/accum2，400epoch，seed0 |
| LP | 双卡，每卡batch64/accum1，100epoch，lr0.1，seed0，dist_eval；checkpoint-399 |

旧256组T→S=0.1/1的LP best分别为85.8018%/85.8928%，只差0.091个百分点；两者S→T均0.1。它们不能替代512对照组，也不能证明哪种设置稳定更好。

E0最新结果：400条epoch0..399连续且有限，LR与旧256/T→S=1一致；loss核算确认T→S=1/S→T=0.1。S→T最低0.00996099在epoch142，末轮0.02788974，远低于旧0.5结构限制，但后期回升到最低的2.80倍。native/T→S最后50轮均值0.01435490/0.01989607，与旧256的0.01439415/0.01991729几乎相同。用户报告LP best85.79%，相对旧256同权重85.8928%低约0.103个百分点，未显示收益；单次且缺LP完整日志，不能判定显著退化。

## 待测试变量

| 编号 | 单变量对照 | 要回答的问题 | 当前实现状态 |
|---|---|---|---|
| E0 | 当前512对照组 | 解除约0.502输出子空间下限后，是否实际改善encoder的LP？ | 已完成用户提供的PT日志分析；LP best85.79%，完整LP日志未提供 |
| E1 | S→T权重0.1 → 1（用户当前继续） | 512修复后，较强的S→T监督是否改善LP？ | 已提供命令；尚无本轮日志/结果，替代此前建议0.3，不做多档权重扫描 |
| E2 | share=True → False | T→S=1时，两种条件共同更新骨架decoder是否妨碍原生任务与表征学习？ | 模型支持；需要独立no-share YAML和输出目录 |
| E3 | 骨架归一化A → B（用户下一项） | 改变骨架信号尺度和有效扩散SNR，是否改善LP？ | 模型支持；需要成对的PT/LP YAML，父对照权重待E1结果决定 |
| E4 | 仅S→T epsilon → v | 调整不同时间步对干净文本恢复的监督权重，是否提高文本任务的表征收益？ | 未实现；需要S→T专用参数，不能把全局diff_prediction直接改为v |
| E5 | S→T无区域偏置 → 六部位软区域偏置 | 让局部文本优先读取对应关节区域，是否使六部位监督更有用？ | 未实现；需要保留随机mask后的原始关节身份、全局key与空区域回退 |

B为pretrain/LP中注释的#new组：mean=[-0.0024,-0.2132,-0.0446]，var=[0.0525,0.1527,0.0513]。它不同于NTU60 XSub finetune的另一组#new（统计报告中的C）。完整8组及全部50处位置见 [INPUT_NORMALIZATION_AUDIT.md](D:/program/MacDiff/tools/vlm_pilot/INPUT_NORMALIZATION_AUDIT.md)。

A→B使同一骨架的XYZ变化幅度成为原来的约0.626/0.726/0.652；固定t下以中心化信号方差衡量的有效SNR成为约0.392/0.527/0.425。它是训练超参数候选，尚不能确证是当前训练数据的正确统计。预训练与LP必须使用同组，从头预训练，独立目录，复用现有文本cache。

## 建议顺序与条件

1. E0已表现出输出瓶颈解除、去噪显著改善但LP未改善。用户已尝试150 checkpoint的LP并决定不再继续；仅首轮72多，不能据此证明它最终比399更差，不再安排继续150或扫描checkpoint。
2. 当前承接E1：保持512/share=True/A/T→S=1，仅把S→T从0.1改为1，替代此前建议0.3。这条分支直接向encoder传递文本监督。较低loss不证明0.1太小，512之前的S→T=1实验也不能作为相同结构的结论。该轮不同时改T→S、不做多档扫描，从头预训练；命令已提供，尚无结果，不重复启动同目录。
3. **随后根据E1结果测试E3归一化B，先于no-share。** 如果权重1有确认的收益，以该组为父对照；如果未改善，可保留原E0的S→T=0.1。当前尚未锁定B的权重。创建独立、成对的PT/LP YAML，只改骨架mean/var，从头训练，复用文本cache；不改正在跑的基线。它可能使epsilon更容易预测而降低loss，不能把这种下降当作文本语义增强。若它有效，再考虑拆分mean/var或噪声日程影响，第一轮不拆成多组。
4. E2 no-share保留为B之后的备选，届时明确父对照。历史share/no-share几乎持平的实验使用旧文本/旧目标机制、T→S=0.1，不能排除当前设置下的任务干扰。no-share会增加参数，同时保留T→S→remap→逐样本目标→S→T→encoder的间接路径，因此收益只能支持这一设计整体，不能单独证明梯度冲突。
5. 如果现有开关的简单对照没有明确LP改善，再讨论是否实现E4，改变S→T任务本身。v=a_t*epsilon-sigma_t*text_target；骨架原生/T→S保持epsilon，目标bank和干净文本来源不变。初轮沿用明确父对照的S→T外层权重，但epsilon与v的raw loss不可直接比较，其梯度与有效监督权重也不等价。该方案不增加信息，效果是待验证假设。
6. E5保留为后续备选，初轮仅修改S→T读出，T→S维持原样。encoder已经混合全身信息，因此区域bias只是读取先验，不是严格局部特征隔离；错误部位描述及静态模板也可能被强化。

当前最小路线是 **E1权重1结果 → E3归一化B**。E2/E4/E5只是备选，不要求将E0–E5全部跑完，也不沿用旧的“150 LP → 0.3 → no-share”优先级。

## 对照与判定

- 首轮每次只变一个因素，明确父对照；各候选使用独立PT/LP输出目录，不覆盖旧checkpoint/目标bank。
- E1/E2/E3首轮都可各自与E0比较；确认收益后建立新的对照组，再测下一因素。不要把所有修改一次叠加，否则无法判断收益来源。
- 相同epoch、LP协议及batch/累积设置；记录实际Git SHA与model_args。不同share结构或归一化实验从头训练，不称为旧实验完整resume。
- 比较LP best、最后10轮均值和最后结果，并保留分支原始loss解释任务变化。较低noise MSE、较高variance/energy不能代替LP。
- 如果只提高约0.05–0.1个百分点，先视为尚未确认的候选。必要时以额外LP seed检查head随机性；这不能替代额外预训练seed。确认有意义的候选后再做完整重复，不先铺开全组合。
- 已做的固定t/噪声条件打乱、方差/相似度诊断不再列为默认下一步；固定CLIP干净目标已试，不作为新方向；不加入对比学习、batch关系蒸馏或其他跨样本内容。

## 用户最新选择与命令

LP best已更正为85.79%。用户不再继续checkpoint-150的LP，当前继续T→S=1/S→T=1、共享512、归一化A的实验；之后根据结果测试B。每卡预训练32/accum2，LP64/accum1；无cd，预训练与LP用&&连接。

以下是已经提供的S→T=1预训练与LP命令，供核对实际运行，不默认重复启动同目录：

~~~bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10254 main_pretrain.py --config config/ntu60_xsub_joint/pretrain_madiff_text_sentence_sample_target_blend_shared.yaml --lambda_text_to_skeleton 1 --lambda_skeleton_to_text 1 --batch_size 32 --accum_iter 2 --epochs 400 --seed 0 --output_dir output_dir/ntu60_xsub_macdiff_sentence_sampletarget01_shared_st512_t2s1_s2t1 --log_dir output_dir/ntu60_xsub_macdiff_sentence_sampletarget01_shared_st512_t2s1_s2t1/tensorboard && OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10255 main_linprobe.py --config config/ntu60_xsub_joint/linprobe_madiff.yaml --finetune output_dir/ntu60_xsub_macdiff_sentence_sampletarget01_shared_st512_t2s1_s2t1/checkpoint-399.pth --output_dir output_dir/ntu60_xsub_macdiff_sentence_sampletarget01_shared_st512_t2s1_s2t1_lp_399_bs64 --log_dir output_dir/ntu60_xsub_macdiff_sentence_sampletarget01_shared_st512_t2s1_s2t1_lp_399_bs64/tensorboard --batch_size 64 --accum_iter 1 --epochs 100 --lr 0.1 --seed 0 --dist_eval
~~~
