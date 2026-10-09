# Decoder3 / A 原版日志分析与 no-share 下一组（2026-10-09）

本轮结果登记为 **T14：原版 MacDiff，A，用户报告 decoder3，PT500，checkpoint-499 LP100，best 85.704755% @94**。PT/LP 都已完成。用户已更正实际 LP 加载的是 **checkpoint-499.pth**，按更正后的命令登记；不需要重复评价499来修正 checkpoint 编号。

## 1. 完整性与精确结果

来源：[本轮600行附件](C:/Users/97537/.codex/attachments/f6643024-0636-4677-ab2b-60b92d4ec967/已粘贴的文本.txt)。500行PT epoch0–499 + 100行LP epoch0–99，连续、无重复、无缺轮、全部数值有限。原文件SHA256及精确统计在 [summary.json](D:/program/MacDiff/handoff_artifacts/decoder3_20261009/summary.json)，复现脚本为 [analyze_logs.py](D:/program/MacDiff/handoff_artifacts/decoder3_20261009/analyze_logs.py)。

| 指标 | 结果 |
|---|---:|
| LP best | **85.7047549825684% @94** |
| LP last @99 | 85.5955846991023% |
| LP epoch80–99 mean ± population SD | **85.63591705546224 ± 0.038371717383694964%** |
| LP epoch90–99 mean ± SD | 85.64228532097977 ± 0.03093158238404497% |
| best轮 Top5 | 97.42236779743696% |
| 与论文86.4之差 | 0.695245个百分点 |
| 与T12 best之差 | 0.491266个百分点 |
| 与T12末20均值之差 | 0.364811个百分点 |
| PT loss @399 / @499 | 0.0157702993 / 0.0156150659 |
| PT 最低loss | 0.0154325114 @438 |
| PT 最后50轮 loss mean ± SD | 0.0155809998 ± 0.0000750828 |

epoch SD只是一次运行内的后期波动，不是多seed置信区间。T12与本轮不是同协议的纯文本收益对照。

## 2. 用户确认的执行设置

| 阶段 | 实际命令显式值 | 配置/日志补充 |
|---|---|---|
| PT | 双卡，每卡64，accum1，有效128；epochs500、seed0；基础pretrain_madiff.yaml | A与decoder3按用户说明；日志确认峰值lr约1e-3、warmup20、末端5e-4 |
| LP | 双卡，每卡128，accum1，有效256；epochs100、lr.1、seed0、dist_eval | 加载checkpoint-499；基础linprobe_madiff.yaml，A |
| 目录 | PT/LP均为output_dir/ntu60_xsub_macdiff_decoder3 | 日志与TensorBoard混在一起；部分PT权重已被LP同名文件覆盖 |

命令没有显式 --min_lr，因此沿用基础YAML的5e-4，而不是script_pretrain_madiff.sh中的1e-5覆写。PT random_rot也没有CLI覆写；若服务器除decoder_depth之外的YAML与本地相同，PT rotation=False。实际服务器配置/SHA/数据身份仍未独立核验。

LP参数量384060 = 6400×60 + 60，与linprobe2的60类线性头一致；这支持分类头尺寸正确，不单独证明所有加载/冻结行为正确。

## 3. 曲线含义及复现差距

**最明确的剩余差异是预训练学习率日程。** [主论文第10页](https://lehongwu.github.io/ECCV24MacDiff/macdiff-paper.pdf#page=10)写明PT500、有效batch128，AdamW lr从1e-3降到1e-5。本轮的轮均lr为：epoch199 8.34526e-4、299 6.67088e-4、399 5.36842e-4、480–499 5e-4。末端比论文写法高50倍，整个后半程也不同。因此这次3层+500轮并未把所有已知训练设置与论文对齐。

这只能定位一个可控因素，不能证明5e-4导致了0.695个百分点缺口，也不能保证改到1e-5就达到86.4。此前历史T01约85.86缺实际完整args，而且本轮还变了epoch/LP batch等，不能解释为“decoder3比5层差”。[论文第14页表10](https://lehongwu.github.io/ECCV24MacDiff/macdiff-paper.pdf#page=14)报告3层86.4、5层85.9；这是作者的消融结果，不是任意运行必然满足的关系。

**PT已进入平台，没有发散信号。** epoch399到499 loss仅下降0.9843%，末50轮波动很小。当前native只记录总loss，即diffusion MSE + lambda×skeleton uniformity；没有分别记录两个分量，也没有按diffusion t分桶，不能仅凭它判断encoder条件使用程度或把0.0156当纯噪声MSE。它也不能和T12含多个任务的总loss直接比较。

**LP正常上升后进入平台。** epoch0为77.2986，9为83.0058，49为85.1164，79为85.5956；epoch80–89均值85.62955，90–99均值85.64229，后10轮仅提高0.01274个百分点。best94与末20均值相差0.06884个百分点，没有明显后期崩塌，也没有充分依据只延长LP。

LP epoch90–99 lr=0，FC停止梯度更新，但head BatchNorm仍在train模式更新running mean/var，所以best@94不表示该轮线性权重继续优化。这是当前代码既有日程。train CE降到约0.16，test CE约2.33；较高test CE表示部分错误预测可能很自信，不单独证明训练失败或encoder坍塌。

**本轮与T12的LP不同。** 本轮每卡128、有效256；T12每卡64、有效128。BatchNorm局部batch、更新次数也不同，lr都为绝对.1。T12更高的best/末20均值是描述性事实，不能把全部差距归于文本。若要量化文本收益，可复用当前499 encoder另跑T12的LP64/accum1，不需要重新PT；暂未启动/安排该补充运行。

![PT/LP曲线](D:/program/MacDiff/handoff_artifacts/decoder3_20261009/comparison.png)

## 4. 输出目录与checkpoint身份

main_linprobe.py每轮使用misc.save_model写checkpoint-{epoch}.pth；PT每10轮及末轮保存。本次复用同一个目录意味着PT的checkpoint-0、10、20、…、90会被LP同编号文件覆盖，log.txt追加LP行，TensorBoard也混用。**PT399与499不在LP0–99覆盖范围内**。已存在的文件不要清空或移动；后续新实验使用独立PT/LP目录。

decoder3目录名不改变模型。decoder=3依用户说明登记，本轮JSON epoch日志没有打印实际层数。用于只读核验的 [inspect_checkpoint.py](D:/program/MacDiff/handoff_artifacts/decoder3_20261009/inspect_checkpoint.py) 可在服务器读取499的saved args与state_dict；native decoder若为3层，decoder_blocks索引应为[0,1,2]。本地没有Torch/服务器checkpoint，未运行实物核验。

显存附加观察：本轮前400轮allocated与reserved记录恰好与T13对应轮完全一致；这不是证明层数未变的证据，历史运行身份与峰值影响因素尚不齐，层数应直接查看checkpoint，而不能用显存猜。

## 5. 用户已选下一组：相对T12只改变骨架decoder深度

新配置：[pretrain_madiff_text_sentence_sample_target_blend_noshare_decoder3.yaml](D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_text_sentence_sample_target_blend_noshare_decoder3.yaml)。

- native / T→S：**5→3**。no-share T→S复制native主体，所以这一个参数同时改变两套骨架decoder。
- S→T：仍5层、hidden512、outputnorm=none；不改text_decoder_depth。
- 权重1/0.1、A、v3七句cache、sample_target_blend更新比.1、uniformity、encoder8/256保持T12。
- PT400、双卡32×accum2=128、lr1e-3/min_lr1e-5、seed0；LP100、双卡64×accum1=128、lr.1、seed0、checkpoint399，均保持T12。
- 新YAML中T→S权重/batch/accum默认值从父文件库存值改为T12实际CLI值，属于让配置自包含，不是相对实际T12新增实验因素。
- 没有同时加500轮、S→T权重.5、rotation、B、LN、768等其他因素。此前1/.5计划保留为待结果/未确认运行；不能与本组混登记。

训练单行命令已保存为 [launch_noshare_decoder3.sh](D:/program/MacDiff/handoff_artifacts/decoder3_20261009/launch_noshare_decoder3.sh)，PT与LP用&&连接、无cd、独立输出目录。先将新YAML同步到服务器再执行。本地仅准备配置/分析文件，没有启动训练、改模型代码或提交推送。

它能回答“两套骨架decoder变浅对现有no-share方法是否有帮助”，不能单独分离native与T→S贡献。论文的native3层效果不能保证迁移到本方法；也不能预言本组一定优于86.196。