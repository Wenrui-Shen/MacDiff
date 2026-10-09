# No-share 文本配置与原生 MacDiff 参数审计

日期：2026-10-08。读取当前工作区全部 12 份 pretrain_madiff_text*.yaml、原生/文本模型、训练入口和目标库实现。此次只检查并保存报告，没有修改配置、模型或启动训练。

原生比较基准为 config/ntu60_xsub_joint/pretrain_madiff.yaml。前一轮已确认其生效字段与[官方固定提交配置](https://github.com/LehongWu/MacDiff/blob/692888f1e2a4227216511aac6733dff31b587213/config/ntu60_xsub_joint/pretrain_madiff.yaml)仅 data_path 不同。完整逐字段结果在 [matrix.json](matrix.json)。

## 1. 当前究竟是哪份 no-share

- 已完成 T12 对照：pretrain_madiff_text_sentence_sample_target_blend_noshare.yaml，实际 T→S=1、S→T=0.1，PT400+LP100。YAML 自身两项默认都是 0.1，T→S=1 来自 CLI。已有完整日志的损失核算支持 1/0.1 生效。
- 当前保存计划：pretrain_madiff_text_sentence_sample_target_blend_noshare_st02.yaml。文件默认为 T→S=1、S→T=0.2；保存命令明确覆写为 **1/0.5**，输出目录后缀为 _s2t05。尚无回传结果，不把计划参数当成实际 checkpoint 身份。
- 两者都是 A、one_person=True、sample_target_blend、七句 v3 cache、S→T hidden512/output_norm=none、share_skeleton_decoder=False。
- st02 与父 YAML 的生效参数差别是 T→S .1→1、S→T .1→.2、batch64→32、accum1→2，以及输出/日志目录。按实际保存命令与已完成 T12 比较，计划只上调 S→T .1→.5，T→S 保持 1，PT32×accum2 保持不变。
- 同名 sentence shared 与 sentence no-share 两份 YAML，排除输出/日志目录后，模型及训练参数仅 share_skeleton_decoder True→False 不同。

来源：[当前实验计划](D:/program/MacDiff/tools/vlm_pilot/STAGE1_TEXT_EXPERIMENT_PLAN.md:3)、[T12 日志分析](D:/program/MacDiff/tools/vlm_pilot/NOSHARE_NORMB_LOG_ANALYSIS.md:50)。

## 2. 三个 decoder，不能把它们混成一套

文本模型继承 MacDiff，并先执行 super().__init__(**kwargs)。当前 no-share 同时训练以下三套 decoder：

| 项目 | 原生骨架 decoder | T→S decoder | S→T decoder |
|---|---|---|---|
| 作用 | 骨架条件下预测骨架 diffusion epsilon | 文本条件下预测骨架 epsilon | 骨架条件下预测文本 epsilon |
| 参数所有者 | decoder_* | text_skeleton_decoder.*，no-share 独立复制 | text_noise_decoder.*，始终独立 |
| 深度参数 | decoder_depth=5 | 从原生 decoder_blocks 复制，仍由 decoder_depth=5 控制 | text_decoder_depth=5 |
| hidden width | dim_feat=256 | 同原生，256 | text_decoder_hidden_dim=512 |
| heads / head width | 8 / 32 | 8 / 32；额外 text reader 也是 8 heads | 从原生首个 decoder block 取 8 / 64 |
| MLP hidden | mlp_ratio×256=1024 | 从原生复制，1024 | 代码固定 4×512=2048 |
| noisy query token | 120/4×25=750 个骨架 patch | 同样 750 个骨架 patch | 当前 v3 为 global+6部位，共 7 个文本向量 |
| condition | 75 个 visible latent + pooled global 还原为 750 位置的 global-local 条件 | 每层 cross-attention 读取 global+6部位文本；512→256 投影 | 每层 cross-attention 读取 pooled+75 visible skeleton，256→512 投影 |
| time embedding | 64 | 64 | 64 |
| modulation | 原版 FeatureModulation/AdaLN | 复制原版 FeatureModulation/AdaLN | 复用 FeatureModulation，采用同一调制公式 |
| 输出维度/每 token | 4×1×3=12 个骨架坐标噪声 | 同样 12 | 512 维文本噪声 |
| 最后一层归一化 | decoder_norm=LayerNorm(256) | 复制的 LayerNorm(256) | output_norm=none：Identity→Linear(512,512) |
| condition dropout | uncond_ratio=.1，逐样本整条条件丢弃 | 没有相应 condition dropout | 没有相应 condition dropout |
| diffusion | epsilon、1000 步、inverse cosine tau1 | 复用相同 noisy skeleton、epsilon 与 t | 同一 diffusion/sampler 定义，另采文本 t 和 epsilon |
| MSE 位置 | 675 个 masked 骨架 patch | 使用同一骨架 mask，仅 masked patch | 所有 valid 文本 token/通道，无另一个 90% text mask |
| 直接更新骨架 encoder | 是 | 否 | 是，经 skeleton memory |

因此 **S→T 是新增的不同结构，不能说它只是原版 decoder 从 256 改成 512。** 它新增 reader、文本 self-attention、文本/person 位置提示，并改变最终归一化、condition dropout 和 loss 范围。T→S 的骨架主体来自原版，仍额外增加逐层 text reader。

代码依据：[T→S 构造和 forward](D:/program/MacDiff/model/transformer_macdiff_text.py:18)、[S→T block/decoder](D:/program/MacDiff/model/transformer_macdiff_text.py:59)、[原生 decoder](D:/program/MacDiff/model/transformer_macdiff.py:329)。

### no-share 的精确含义

share_skeleton_decoder=False 对 T→S 的 decoder_embed、decoder_blocks、decoder_norm、decoder_pred 做 deepcopy，并独立 clone 时空 embedding。复制主体初始值来自原生 decoder，随后参数分别更新；新增 text_readers 是额外模块。它不是重新训练一个独立 skeleton encoder。

share=True 时 T→S 复用原生骨架主体，text reader 仍属于额外模块；S→T 无论 share=True/False 都独立。no-share 没有切断所有文本对骨架训练的影响：S→T 仍直接训练 encoder，T→S 和文本 uniformity 仍训练在线 remap，在线 remap 又在 step 后更新 S→T 保存目标。

若同时把两个文本方向权重设为 0，forward 会回退原生 forward_macdiff，并跳过文本 uniformity。作为关闭文本的控制时还应把 lambda_text_uniformity 设为0：否则构造器因该正则权重仍保留 remap.requires_grad=True，但 forward 并不使用它，当前 DDP(find_unused_parameters=False) 存在未用参数风险。这是静态推断，未运行验证；当前两个方向非零不触发该情形。文本模型构造时额外消耗 RNG，因此同 seed 不能直接假定整段训练逐位等价于纯原生模型。

## 3. 原生参数是否被文本配置偷偷改动

全部 12 份文本 YAML 的以下生效设置，与原生 NTU60 XSub 一致：

| 参数组 | 一致的值 |
|---|---|
| encoder | depth8、dim_feat256、heads8、mlp_ratio4 |
| 原生 decoder | decoder_depth5、dim_feat256、MLP1024、heads8 |
| 输入与 patch | dim_in3、120帧、25关节、patch_size1、t_patch_size4 |
| skeleton dropout | drop_rate/attn_drop_rate/drop_path_rate 全0 |
| modulation | layer_mask_ratio0、uncond_ratio.1（仅原生 forward） |
| diffusion | noise、1000、inverse_cosine tau1、默认无 SNR reweight |
| skeleton normalization | A mean/var、self_shift=False |
| 人物 | one_person=True；原生默认 True，文本显式 True |
| mask | mask_ratio.9、motion_aware_tau=-1，即随机 mask |
| 骨架 uniformity | lambda_loss_uni=.02 |
| feeder | crop[.5,1]、joint_noise[1,.005]、其余三类噪声0、bone/vel/flip/rotation均False、normalization=False |

唯一书写区别是原生 qk_scale: None、文本 qk_scale: null。PyYAML 将前者视为字符串、后者为 None；但当前 Attention 完全忽略 qk_scale，始终使用 head_dim**-.5，故不会导致这组训练的实际缩放差异。

“原生结构参数一致”不表示文本版训练轨迹一致：目标函数增加两项任务与文本 uniformity；有条件 dropout 范围差异；文本 forward 在标准化前检测 active people，排除空人物行的 native/text loss。原生 forward 未做相同的空行过滤。当前 one_person=True 时若 person0 始终有效，uniformity 公式/样本范围一致；若未来两人训练，文本 uniformity 对全部 active rows，而原生仍只取 person0。

来源：[文本原生分支](D:/program/MacDiff/model/transformer_macdiff_text.py:353)、[骨架标准化](D:/program/MacDiff/model/transformer_macdiff.py:475)、[原生 loss/人物](D:/program/MacDiff/model/transformer_macdiff.py:744)。

## 4. 每个 text 参数实际影响哪里

| 参数 | 当前值 | 作用范围和含义 |
|---|---|---|
| text_input_dim | 512 | CLIP/global/local 向量维度，也是当前 S→T epsilon 输出维度；不改变 skeleton encoder 的256维 |
| text_hidden_dim | 512 | 在线 ResidualTextRemap 的中间层，512→512→512，带 residual 与 unit RMS；不是骨架 decoder hidden |
| text_decoder_hidden_dim | 512 | S→T 内部 hidden 和投影维度；不改变 T→S 骨架主体宽度 |
| text_decoder_depth | 5 | 只控制 S→T 层数 |
| decoder_depth | 5 | 原生层数；T→S 复制相同层数及每层配置。改成3会同时改变这两套 |
| text_decoder_output_norm | none | 只去掉 S→T 输出头前最后一个 LN；query norm、两处 modulation LN 都保留，原生/T→S 最终 LN 也保留 |
| text_context_length | 7 | v3 global+6部位句子的位置范围；六个 local 的 position IDs 为1..6。不是77个 BPE token |
| text_target_mode | sample_target_blend | 按原始样本 ID 保存 global/local 目标，供 S→T 使用；T→S 使用在线 remap |
| text_target_norm | rms | 固定 CLIP 输入与在线 remap 的 unit RMS；不是 XYZ mean/var，也不是 text output head 的 LN |
| text_target_update_ratio | .1 | 当前出现样本：saved_target=.9×old+.1×step后在线remap。与 S→T loss 的 .1/.5 权重无关 |
| text_target_momentum | 默认.999，当前不使用 | 只用于历史 ema_remap 参数 EMA；sample_target_blend 不使用该数值 |
| lambda_text_uniformity | .02 | 对在线 remap 的6个有效 local 文本向量做 token uniformity；不包含global，不直接作用于 skeleton encoder |
| lambda_loss_uni | .02 | 对 skeleton encoder latent 做原版 uniformity；与上一项是两个正则 |
| share_skeleton_decoder | False | 只决定原生与 T→S 是否共享骨架 decoder 主体；S→T 始终独立 |
| text_target_bank_backend | shared_memory | 目标库的存储/访问方式；与 sqlite 的更新公式一致，不是另一种 target_mode |
| text_cache | ...sentence_cache_v3 | global+六部位，原 NPZ 人物槽位对齐；必须与 feature_dim512、context7 匹配 |

v3 部位为 head、torso、left_arm、right_arm、left_leg、right_leg。当前本机没有该 cache，表格按配置与 reader/validator 的预期协议核查；既有 T12 日志的 valid local tokens=6 可作为过去运行的补充证据，不能替代新实验 cache manifest。

**RMS 的边界：** 初始目标及在线 remap 输出为 unit RMS；target bank 的 .9/.1 混合后不会再次强制 RMS=1，读取也不重新归一化。因此 saved target 的 energy 可略低于1，这是当前实现，不应将 text_target_norm=rms 解读为目标库每一步严格重新 RMS。

### 尚未独立暴露的参数

- T→S 深度/hidden：跟随原生 decoder；没有独立 t2s_decoder_depth/hidden 配置项。
- S→T num_heads：从原生 decoder 首层读取，当前8；没有独立 text_num_heads。
- S→T MLP ratio：硬编码4；改原生 mlp_ratio 不会改变它。
- S→T reader/self-attention dropout 都硬编码0，modulation layer_mask_ratio也硬编码0，无 DropPath；原生 drop_*、layer_mask_ratio 不控制这些新增组件。
- T→S 新增 reader dropout 硬编码0；复制的骨架主体才继承原生 drop_*、MLP、layer_mask_ratio。
- uncond_ratio 控制原生 forward_decoder；两个文本 decoder 没有读取它。
- 当前 persistent sample target engine 要求 one_person=True。仅改文本 YAML one_person=False 不能完成双人 sample_target_blend 消融，engine 会拒绝。
- 输出头归一化、缓存维度、目标模式都由相应显式字段决定，不能因文件名带 rms/noshare/st02 代替读取有效配置。

## 5. 总损失及权重优先级

当前保存计划的总 loss：

~~~text
L = L_native
  + 0.02 * L_skeleton_uniformity
  + 0.02 * L_text_uniformity
  + 1.0  * L_T2S
  + 0.5  * L_S2T
~~~

已完成 T12 的最后一项为 .1×L_S2T。方向权重是乘 raw loss 后相加；S→T weight=.5 不代表它贡献总梯度的50%。当前 S→T 对所有 valid global/local token 同权平均，global 1 个、local 6 个，因此在均方误差平均中 global 占1/7、全部local占6/7；并非 global/local 各占一半。

权重解析：CLI → 顶层 YAML → 构造器默认。main_pretrain 将两个顶层 lambda 明确覆写进 model_args。因各文件均有顶层 lambda，单改 model_args 内同名字段可能被顶层值盖掉；应使用当前命令的两个显式 CLI 参数。

参数依据：[main_pretrain 覆写](D:/program/MacDiff/main_pretrain.py:361)、[总损失](D:/program/MacDiff/model/transformer_macdiff_text.py:432)、[S→T loss](D:/program/MacDiff/model/transformer_macdiff_text.py:316)、[目标更新](D:/program/MacDiff/util/shared_memory_text_target_bank.py:245)、[engine 更新条件](D:/program/MacDiff/engine_pretrain.py:154)。

## 6. 12 份文本 YAML 的分组，避免拿错历史配置

所有组的原生 encoder/decoder 均为8/5层、256维，skeleton A；表中权重是 YAML 默认，实际 CLI 需另看。

| 文件名组（共同前缀 pretrain_madiff_text） | 数量 | 目标模式/文本归一化 | context | S→T hidden/末端LN | share | YAML T→S/S→T |
|---|---:|---|---:|---|---|---|
| .yaml、_st01.yaml | 2 | fixed_clip / none | 77 | 256 / 有 | T→S关闭，flag=False | 0/1、0/.1 |
| _rms.yaml、_rms_st01.yaml | 2 | fixed_clip / rms | 77 | 256 / 有 | T→S关闭，flag=False | 0/1、0/.1 |
| _rms_ema_bidirectional[ _shared ].yaml | 2 | ema_remap / rms，参数EMA | 77 | 256 / 有 | False/True | .1/.1 |
| _rms_sample_target_blend[ _shared ].yaml | 2 | sample_target_blend / rms，逐样本目标 | 77 | 256 / 有 | False/True | .1/.1 |
| _sentence_fixed_rms.yaml | 1 | fixed_clip / rms | 7 | 512 / 无 | T→S关闭 | 0/.1 |
| _sentence_sample_target_blend_shared、_noshare.yaml | 2 | sample_target_blend / rms | 7 | 512 / 无 | True/False | .1/.1；T11/T12实际1/.1 |
| _sentence_sample_target_blend_noshare_st02.yaml | 1 | sample_target_blend / rms | 7 | 512 / 无 | False | 1/.2；当前命令1/.5 |

旧 context77 配置与当前七句配置同时改变 token 定义、decoder hidden 和末端 LN；不能据文件中都写 depth5 就当同一结构。fixed_clip 组 T→S=0，不构造 remap/T→S 主体，部分 remap/share 参数即使被接受也不参与训练。

## 7. 训练参数与官方运行口径

- 当前 st02 PT 为400轮、warmup20、AdamW(.9,.95)、wd.05、LR1e-3→1e-5、AMP 默认True，与文本父对照相同。
- 全部文本 YAML 的 LR1e-3/min_lr1e-5 一致；原生 YAML min_lr5e-4，但官方 .sh 传1e-5。按脚本比，文本 min_lr 没有改变官方计划；按纯 YAML 比则不同。
- 已完成 T12 和当前计划 PT 两卡×32×accum2=128，与官方四卡×32×accum1同全局 batch，非逐位等价。
- 父 sentence no-share YAML 写64×accum1；当前 st02 写32×accum2。两卡下均128，但 microbatch/RNG组织不同，实际对照按保存命令32×2。
- 当前文本 LP 仍使用基础 linprobe_madiff.yaml，encoder256、linprobe2、BN6400。没有把文本 decoder 或512维文本特征用于 LP。
- 当前文本 LP 两卡×64×accum1=128，官方脚本四卡×64=256；head 的每卡BN batch均64，梯度batch和更新次数不同。
- 论文500轮/rotation，以及原生 decoder3/5 的口径疑点见[论文审计](D:/program/MacDiff/handoff_artifacts/macdiff_reproduction_20261008/README.md)。本次没有将这些因素叠加进 no-share。

## 8. 适合下一步消融的明确结论

1. 当前 no-share 的原生骨架结构与官方默认一致。先固定已完成 T12 的完整执行参数；当前计划仅提高 S→T 权重为.5。
2. 若要试论文3层，直接改 decoder_depth 会同时改原生与 T→S；必须把它标成“两个骨架 decoder 深度一起变化”，或先增加独立 T→S 深度选项才能控制归因。
3. text_decoder_depth=5→3 是另一项单独的 S→T decoder 深度实验，不等于原版3层 decoder 消融。
4. 文本条件 dropout、S→T MLP ratio/dropout、global/local loss配比目前未独立暴露；它们确实与原版行为不同，但尚无证据证明是性能变化原因。增加开关应保持当前默认再做单因素实验。
5. 修改 text_target_norm、text_decoder_output_norm 和 skeleton input_mean/var 是三类不同的归一化因素；避免捆绑改动。
6. 仅改变 text_hidden_dim 研究的是 remap；改变 text_decoder_hidden_dim 研究的是 S→T 容量；改变 dim_feat 则会动 encoder、原生/T→S及下游权重形状，不适合作为文本 remap 的替代开关。

验证范围：静态源代码/继承默认值/12份配置逐字段对照，shared/no-share与st02差异核验。未导入 Torch 实例化模型，未检查服务器实际 checkpoint/cache，未运行新训练。已有用户文件保持原样，仅新增本目录报告和矩阵。

