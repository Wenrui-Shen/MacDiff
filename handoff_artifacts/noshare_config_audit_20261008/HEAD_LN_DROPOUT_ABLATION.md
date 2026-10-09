# Decoder 输出头、最终 LayerNorm 与条件 dropout 消融

日期：2026-10-08。依据当前 no-share 代码静态核查。本文件是实验设计；没有修改模型、配置或运行训练。

## 三套 decoder 都有输出 projection

当前 patch_size=1、t_patch_size=4、dim_in=3、dim_feat=256、text_input_dim=512、text_decoder_hidden_dim=512。

| 分支 | 输出链路 | 输出头 | 代码 |
|---|---|---|---|
| 原生骨架 | decoder blocks → LayerNorm(256) → Linear(256,12) | decoder_pred，bias=True | model/transformer_macdiff.py:339、346、695 |
| T→S no-share | decoder blocks → LayerNorm(256) → Linear(256,12) | 独立 deepcopy 的 decoder_pred | model/transformer_macdiff_text.py:25、56 |
| S→T | decoder blocks → Identity → Linear(512,512) | output.1，bias 默认True | model/transformer_macdiff_text.py:101–115 |

S→T 的 text_decoder_output_norm=none 只令 output.0 为 Identity，**没有删除 projection**。layernorm 模式把 output.0 换为 LayerNorm，output.1 的 Linear 保留。三者都是线性输出投影，不是额外的两层非线性 MLP。文本分支的输入/condition 维度投影是另一类适配层，不应与末端输出头混淆。

## 为什么不能仅因原版有 LN 就给文本分支强制加回

LayerNorm 先令 H 维向量零均值，所以归一化结果位于最多 H−1 维的线性子空间。可学习 affine 再接 Linear，会得到最多 H−1 维的仿射输出子空间。

- 骨架 H=256，输出D=12；H−1=255大于12，并不强制损失某个输出维度。
- 旧文本 H=256，输出D=512，末端LN会限制最多255个输出方向。对每通道标准 Gaussian epsilon、通道平均MSE，其维度下界至少 (512−255)/512≈0.501953。
- 当前文本 H=512，输出D=512，若加末端LN，最多511个方向，对应的维度下界至少1/512≈0.001953。
- 当前 H=512、none、Linear512→512，没有上述强制的线性输出维度缺失。

这是从 LN 零均值约束与当前 Linear/epsilon 定义推得的结构下界，不是训练最终 loss 或 LP 的预测；优化、噪声条件下的可辨识性等仍会影响结果。LN 对尺度与训练稳定性的影响需要实验，不能单凭维度推断 LP。

因此当前 none 是有结构依据的选择，但 **512/none 对 512/layernorm** 仍值得独立消融。不要用旧256/LN对新512/none宣称纯LN实验，也不要用改 norm_layer 的方式同时去掉 encoder/block 内部归一化。

## 最小消融组

先固定一份有结果的 no-share 权重基线。已完成 T12 为 T→S=1、S→T=.1；当前保存的1/.5计划沿用原计划，若选择其作为结构基线，应先取得对应 control。以下各行必须与同一权重、batch、LR日程、cache、target ratio、seed及LP协议比较。

| ID | 原生 LN | T→S LN | S→T LN | 原生 condition drop | T→S condition drop | S→T condition drop | 相对基线改变 |
|---|---|---|---|---:|---:|---:|---|
| B0 | 有 | 有 | 无 | .1 | 0 | 0 | 当前no-share |
| A1 | 有 | 有 | 有 | .1 | 0 | 0 | 只给S→T加最终LN，hidden仍512 |
| A2 | 有 | 有 | 无 | .1 | .1 | 0 | 只给T→S加逐样本condition dropout |
| A3 | 有 | 有 | 无 | .1 | 0 | .1 | 只给S→T加逐样本condition dropout |
| A4（按结果追加） | 有 | 有 | 无 | .1 | .1 | .1 | 两个文本方向同时drop；检查交互，不作为单因素 |
| A5（后续） | 有 | 无 | 无 | .1 | 0 | 0 | 只去T→S末端LN，不改原生LN |

优先完成A1/A2/A3。A1已有 text_decoder_output_norm 开关；A2/A3及A5没有现成独立开关，需要先增加实现并保持当前默认（drop=0、T→S LN开启），再准备各自配置。

现有 uncond_ratio=.1 只在原生 forward_decoder 中生效。nn.MultiheadAttention(dropout=.1) 会丢弃 attention 权重，并不等于 condition dropout；layer_mask_ratio 逐层丢调制条件，也不同于全样本、全层同一次条件丢弃。

## 两个文本 dropout 的准确实现定义

为尽量与原版“每个样本整条条件丢弃”对应：

1. 每次该 decoder forward，对每个样本采一次 keep，形状[N,1,1]，该 decoder 的全部层复用它。
2. 各层 cross-attention 产生 conditioning z 后、传入 FeatureModulation 前，执行 z=z*keep。
3. T→S 丢弃的是 text conditioning；S→T 丢弃的是 skeleton conditioning。保留 noisy target、t、decoder 自己的 position/person 提示和有效token mask。
4. 不应把一整行 memory 的 key_padding_mask 设为全True，避免 attention 无有效key；也不要把 detach 当成条件丢弃，detach只断梯度但仍提供信息。
5. 当前 condition reader/输入投影均可能有bias，输出 z 置零比单纯把原始memory输入置零更直接对应原生 z=0 的定义。

S→T 条件被丢弃的样本，该辅助任务不会经 condition reader 给 encoder 提供梯度。因此固定外层loss权重时，dropout会同时改变条件学习与encoder收到监督的频率。若A3明显变化，再考虑匹配条件样本的权重对照；不要第一轮同时改dropout和lambda，混淆因素。

## 判断依据与执行阶段

这些开关影响 encoder 的预训练，必须重新PT后采用同一LP协议评价。仅给已完成 checkpoint 换 decoder 配置、再做LP，不能检验它们对已学encoder的影响。

主指标为NTU60 XSub LP best、last、后期均值，并对有效差异追加至少3个seed。辅助记录各分支raw MSE、加权loss、目标energy、目标更新比例和NaN/AMP情况；若加dropout，记录实际keep率。噪声MSE降低不是LP必然改善的证据。

T→S最终LN和原生最终LN可以随后做定向诊断；不必为了三个任务表面一致而一次修改全部decoder。原生输出12维、文本输出512维，条件接口与输出统计不同，本来就可能需要不同选择。

