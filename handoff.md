# MacDiff 骨架表征 + 文本特征监督方案交接

更新时间：2026-08-19

> 这是当前方案的权威交接。此前的骨架视频/VLM pilot 详细记录已保存在
> `handoff_vlm_pilot_legacy.md`，需要查环境、渲染和 Qwen 推理细节时再读；不要把旧文档中“文本尚未接入”的状态当成当前设计结论。

## 1. 我们正在做什么

目标是在不改变 MacDiff 现有单人骨架建模方式的前提下，用 VLM 生成的样本级文本作为辅助语义监督，增强 90% mask 后剩余 75 个 skeleton token 的表征，并间接改善它们池化得到的下游表示。

当前主线不是生成自然语言，也不是照搬 AIMv2 做多模态自回归。已经收敛到的核心思路是：

1. 保留 MacDiff 原有 skeleton diffusion 预训练路径。
2. 每次仍固定 mask 90%，750 个 skeleton patch 中保留 75 个进入骨架 encoder。
3. 新增一个 learnable query token，通过 cross-attention 从这 75 个 skeleton token 中读取信息。
4. 将 query 输出投影到冻结文本 encoder 的连续特征空间，预测当前人物对应的文本特征。
5. 文本只是特征级监督；不生成离散单词，不做 Text AR。
6. 双人动作继续沿用当前“每个人作为一个独立单人骨架样本前向”的方式，只在数据层把完整动作描述拆成两个人各自的描述并分别配对。

当前没有要求修改代码；本轮完成的是论文调研、方案排除和设计收敛。下一会话应先核对代码和数据路径，再实现。

## 2. 当前 MacDiff 基线必须先理解清楚

相关代码：

- `model/transformer_macdiff.py`
- `config/ntu60_xsub_joint/pretrain_madiff.yaml`
- 其他数据集同名 `pretrain_madiff.yaml`

已确认的关键事实：

- NTU 设置下骨架长度 120 帧、25 个关节，时间 patch size 为 4，因此共有 `30 × 25 = 750` 个 skeleton patch。
- 配置中的 `mask_ratio: 0.9`，所以 encoder 每次只看到 75 个 patch。
- `forward_encoder()` 对可见 token 做编码，并在 `model/transformer_macdiff.py` 约 655 行以可见 token 的 mean 得到 `cls_token`。
- diffusion decoder 的直接训练目标是噪声，不是直接输出干净骨架坐标。
- 更严谨的说法是“对 masked skeleton patch 做条件去噪”，不要再笼统写成“decoder直接重建骨架”。

若

```text
x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * epsilon
```

decoder 学习 `epsilon_hat = D(x_t, t, Z_visible)`。它预测的是噪声；由噪声可以反推出 `x_0_hat`，所以只能在“训练作用/最终恢复意义”上称为间接骨架恢复。

## 3. 已经确定的主方案

### 3.1 单人样本

对一个人的骨架 `X`：

```text
X
  -> 固定 90% random mask
  -> 75 个 visible skeleton tokens
  -> 现有 skeleton encoder
  -> Z_s: [75, d_s]
```

原有 diffusion 分支保持不变。

新增文本分支只做：

```text
learnable query q
  --cross-attention(K=Z_s, V=Z_s)--> query feature
  --projection--> predicted text feature
```

文本目标：

```text
person-specific caption
  -> frozen CLIP-like text encoder
  -> continuous target feature t
```

然后对 predicted text feature 和 `t` 做特征级匹配。具体采用 cosine、MSE、JEPA-style loss 还是带对比项，尚未由用户最终指定；不要在实现前自行添加复杂损失。

必须保持的边界：

- 当前只确定一个 learnable query token，不要擅自扩展成 7 个语义 query。
- 不要增加 Text AR。
- 不要生成离散文本 token。
- 不要把真实文本特征喂给 predictor；文本只作为目标。
- 不要擅自增加 CLS 对齐、额外对比头、query self-attention 等用户没有提出的模块。

### 3.2 双人动作：数据级展开，不改模型

这是用户刚刚明确确认的设计，必须按这个理解：

当前双人动作本来就会把两个人的骨架拆开，分别走完整的单人骨架前向。因此文本侧也做同样的数据级拆分。

原始双人样本：

```text
(X_person1, X_person2, full_caption)
```

VLM 将完整描述拆成：

```text
caption_person1
caption_person2
```

数据层展开为两个互相独立的训练样本：

```text
(X_person1, caption_person1)
(X_person2, caption_person2)
```

两者分别执行相同的单人流程：独立 mask、独立 skeleton encoder forward、独立 diffusion loss、独立 query cross-attention、独立文本特征预测。模型权重共享，但一次前向只看一个人的骨架。

单人动作只产生：

```text
(X_person1, caption_person1)
```

不要对不存在的 person2 构造空字符串文本目标，也不要对全零第二人骨架做一次无意义前向。CLIP 对空字符串仍会产生非零向量，不能把它当作“无目标”。

这个方案的含义是学习每个人自身可观察到的动作/角色，不在模型内部融合两个人，也不显式加入相对位置、person embedding 或交互模块。用户的目的正是复用现有单人建模，不要擅自改成双人联合 encoder。

必须保证 `caption_person1/person2` 与骨架 person slot 对应。现有骨架视频中 person1/person2 使用红/蓝区分，详细渲染约定见历史交接；真正落盘时仍要核对 feeder 中 person 维度和 VLM 输出人物标识的映射。

## 4. 文本数据当前怎么来

此前已经完成骨架视频到 VLM 文本的 pilot：

- root-centered 三视图骨架视频。
- person1 红色，person2 蓝色。
- Qwen3-VL 2B/8B 单样本推理已跑通。
- 当前较合理的输入基线是 32 个实际帧、`sample_fps=8`。
- 详细路径、命令、环境和 prompt 见 `handoff_vlm_pilot_legacy.md`。

旧 prompt 主要生成一条整体描述。为了当前双人数据级展开，下一步要让 VLM 在双人样本中输出两条明确对应 person1/person2 的描述；单人样本只输出一条。不要把动作类别标签写进 prompt，也不要让文件名或 metadata 泄漏类别。

用户担心 VLM 文本表述多样，因此当前不要求模型还原原句，而是让冻结 text encoder 把不同表述映射成连续语义特征，再让 skeleton query 预测该特征。

## 5. 论文调研已经得到的结论

### 5.1 AIM/AIMv2

用户提供的论文：

```text
D:/program/paper/Fini 等 - Multimodal Autoregressive Pre-training of Large Vision Encoders.pdf
```

它用视觉 patch 和文本 token 的自回归目标预训练视觉 encoder，但它不能直接证明“固定只看 10% 可见 patch，让这 10% 始终承担其余 90% 的 AR 重建”可行。

经典 teacher-forced AR 中，只有第一个待预测 token 真正只依赖初始前缀；后续 token 会看到越来越多真实前序目标。因此“AIM 从完整/前缀序列做 AR”和“MacDiff 固定 75 个可见 token 并行解释 675 个 masked token”不是同一个问题。

### 5.2 高 mask MAE 与 generalized AR

调研过 OmniMAE、MaskGIT、MAGE、MAR：

- 图像保留 10% 做 MAE/迭代 masked prediction 是可行的先例。
- 但它们通常并行预测或逐轮增加上下文，不是让所有目标始终只依赖固定 10%。
- MAR 最接近“连续 latent + masked autoregressive”，但生成上下文会逐步扩充，并用 diffusion 建模连续 latent。
- 没找到一篇同时满足“固定 10% 可见、严格逐 token AR 重建剩余 90%、且以 encoder 表征学习为主要目的”的直接先例。

### 5.3 MAP 论文的准确理解

MAP：`Masked Autoregressive Pretraining`，CVPR 2025，目标是预训练 Mamba-Transformer 混合视觉骨干。

它不是普通逐 patch AR，而是分层结构：

- 对整张图随机 mask。
- 一行作为一个 local region。
- 行内 masked token 并行预测，类似 local MAE。
- 行与行之间使用 row-wise causal attention，强调与 Mamba raster scan 顺序一致。
- 目标是 normalized pixel patch，使用 MSE。

论文关键消融：

- ViT 更受益于 MAE；Mamba 更受益于与扫描顺序一致的 AR。
- pure AR pilot 的最佳 mask 是 20% masked，不是 90%；70% masked 已下降。
- 完整 MAP 的最佳 mask ratio 是 50%；25% 和 75% 都更差，论文没有证明 90% mask 合适。
- MAP 的 decoder mask 优于纯 AR、纯 MAE 和 local MAE。
- 论文表 7 中 diffusion loss 只有 83.3，而 normalized-pixel MSE 达到 84.9；作者据此认为重建生成质量与 encoder 表征质量并不等价。

官方代码还显示：

- encoder 会 drop masked token。
- decoder 一次前向使用按行 block-causal mask。
- 它没有像语言模型那样把前面 masked 位置的 ground-truth pixel 逐个 teacher-force 给后面位置，因此不要把 MAP 描述成严格 token-by-token 生成式 AR。
- 当前官方代码默认从约 30%-70% mask 分布采样一个比例，而 CVPR 正文实验写的是 50%；如果以后复现，必须明确记录“按论文”还是“按代码”。
- 可变 mask ratio 在训练中通常是每个样本或 batch 只采一次，不意味着同一样本必须多次 encoder forward。

## 6. 为什么当前主线不做 AR

### 6.1 Skeleton MAE/diffusion 与 Skeleton AR 的差别

两者都以恢复骨架为训练信号，但条件分解不同：

- 并行 masked reconstruction：675 个目标都依赖同一组 75 个 visible encoder features，所有损失直接给这 75 个 token 压力。
- teacher-forced AR：越靠后的目标越依赖先前真实骨架 patch，可能减弱对最初 75 个 visible features 的依赖。

MacDiff 的 skeleton patch 同时具有时间轴和关节轴，没有图像 raster order 那么自然。强行把 750 个 patch 排成一条严格因果序列，顺序本身就会成为额外假设。

### 6.2 Text AR 被明确否决

当前文本目标是一个冻结 text encoder 输出的连续向量，不是要生成一句离散自然语言。没有自然的 `next token` 任务，因此 Text AR 没必要。不要在新会话重新提出：

- 多个语义字段连续 AR；
- teacher forcing 前序文本向量；
- 离散语言 decoder。

当前替代方案就是一个 learnable query 通过 cross-attention 读取 75 个 skeleton token，然后直接预测对应人物的文本特征。

## 7. 另存的研究 idea：完全 AR 路线

用户要求保留，但它不是当前主线，也不要未经确认实现。

idea 的核心是 AIMv2-style full autoregressive pretraining：以 75 个可见 skeleton token 作为初始条件，为其余 skeleton patch 规定顺序，通过 causal decoder 逐步预测；如果扩展为真正多模态生成，还可以在 skeleton 序列之后生成离散文本 token。

需要记住：

- 如果采用 teacher forcing，只有开头真正只依赖 10%；后续上下文会增长。
- 若所有 675 个目标始终只看最初 75 个而不反馈前序真实/预测 patch，那更接近带 causal mask 的并行重建，不是标准 AR。
- 连续骨架 AR 可以直接回归坐标/特征，也可以每步用 diffusion 建模条件分布；AR 与 diffusion 理论上是两个维度，并不逻辑冲突。
- 但在本项目中，同一批 masked skeleton 同时上 diffusion 和 AR 会目标重叠、计算变重且难以归因。
- 如果以后验证完全 AR，优先把它作为“替换 diffusion decoder”的独立对照，而不是直接堆在主模型上。
- 固定 90% mask 做严格 skeleton AR 没有现成证据，且 MAP 的消融反而提示高 mask 会损害 AR。

此 idea 暂时只记录，不进入下一步实现计划。

## 8. 已经完成了什么

1. 骨架三视图渲染和 Qwen3-VL 单样本 caption pilot 已跑通。
2. 8B 比 2B 更适合当伪文本教师；旧 caption validator 存在“空字段也 accepted”的问题，尚需修。
3. 阅读并分析 AIMv2 的视觉/文本 AR 训练方式。
4. 调研高遮挡 MAE、MaskGIT、MAGE、MAR 和 MAP，澄清 10% 可见与严格 AR 的差别。
5. 澄清 MacDiff diffusion decoder 直接预测噪声，而不是直接输出骨架。
6. 排除 Text AR。
7. 确定 learnable query + cross-attention + text latent prediction 的主线。
8. 确定双人动作采用数据级拆分的两个独立单人前向，不改现有单人骨架模型。
9. 将“完全 AR”保留为独立研究 idea，不进入当前实现。

## 9. 当前卡在哪里

没有理论死锁，当前停在实现前的接口确认阶段：

1. 需要检查 feeder/训练循环中现有双人骨架到底在哪一层被拆成单独 person forward，不能只凭讨论假设。
2. 需要修改 VLM 输出格式，使双人样本稳定地产生两条可与 person slot 对应的描述。
3. 需要决定冻结 text encoder 的具体模型和文本特征匹配损失；用户尚未指定，下一会话不要自行扩大方案。
4. 需要决定文本特征在线编码还是离线预计算，并设计 `(sample_id, person_index)` 到文本特征的索引。
5. 尚未实现 learnable query cross-attention head，也没有相应单元测试或训练实验。

## 10. 下一步计划

按以下顺序推进，先确认再改代码：

1. 运行 `git status --short`，保护现有未提交改动。
2. 阅读 `model/transformer_macdiff.py` 的 `forward_encoder()`、主训练 `forward()` 和 feeder/processor 中 person 维度处理，画出当前单人/双人样本的真实张量流。
3. 读取 `tools/vlm_pilot/skeleton_motion_prompt_v0.txt` 和 caption 脚本，确定最小改动：单人输出一条 person caption，双人输出两条 person caption，并保存明确的 person 索引。
4. 在数据层将双人样本展开为两个 `(skeleton_person, text_person)` 训练记录；单人只保留一个记录。
5. 选定并冻结 text encoder，先验证同一人物不同合理措辞在特征空间中的稳定性。
6. 实现最小文本预测头：一个 learnable query、cross-attention 读取 75 个 skeleton token、一个输出投影。
7. 只加入用户确认的单一特征匹配 loss，并与现有 diffusion loss组合；不要附带新增模块。
8. 添加最低限度测试：
   - 90% mask 后 query 的 K/V 长度确为 75。
   - 单人样本只产生一个文本目标。
   - 双人样本被展开成两个独立 forward。
   - person1/person2 文本没有串位。
   - 不存在空文本 embedding 监督。
   - 文本损失梯度能够回传到 skeleton encoder 的 75 个 token。
9. 先做小规模过拟合/损失下降检查，再做完整预训练。

## 11. 绝对不要再踩的坑

1. **不要再把当前主方案说成多模态 AR decoder。** 当前是 learnable query cross-attention 预测连续文本特征。
2. **不要重新加入 Text AR。** 用户已明确认为没有必要。
3. **不要擅自增加 7 个语义 query、CLS 对齐、额外对比头或其他未提出模块。** 用户要求只保留其明确提到的设计。
4. **不要修改现有单人骨架建模来联合编码两个人。** 双人动作在数据层拆成两个独立单人样本前向。
5. **不要给单人动作的第二人构造空字符串目标。** 空字符串 text encoder 输出不是“空”。
6. **不要对全零 person2 骨架做训练前向。** 单人样本只产生一个有效记录。
7. **不要让两个人的文本与 skeleton person slot 串位。** 这是双人方案是否成立的首要数据问题。
8. **不要把双人完整描述原样同时监督两个单人骨架。** 必须先拆成各自人物描述。
9. **不要说 diffusion decoder 直接重建骨架。** 它直接预测噪声，骨架恢复是间接去噪意义。
10. **不要默认 AR 和 diffusion 必须绑定或必须互斥。** 理论上正交；工程主线目前保留 diffusion、不做 AR。
11. **不要把 AIMv2 的完整/增长前缀 AR 等同于固定 10% 可见 token 对全部 90% 目标的条件预测。**
12. **不要把 MAP 说成 90% mask 的逐 token AR。** MAP 最佳 50% mask、行内并行、行间 causal，而且官方实现不是语言模型式 teacher forcing。
13. **不要认为可变 mask ratio 必须让同一样本多次 encoder forward。** 通常每个样本/batch只采一次；但当前主线仍固定 90%。
14. **不要把动作标签或类别名交给 VLM。** 文本必须从骨架视频本身生成，避免标签泄漏。
15. **不要在未修复空 caption 校验前全量生成文本库。**
16. **不要覆盖用户现有未提交改动。** 所有后续代码修改前先重新检查状态。

## 12. 新会话最短启动方式

先读本文件，然后执行：

```powershell
git status --short
rg -n "forward_encoder|forward_decoder|mask_ratio|one_person|person" model feeder config
rg --files tools/vlm_pilot vlm_pilot
```

接着只做两件只读工作：

1. 找出双人骨架当前被拆成单人前向的准确代码位置和张量形状。
2. 找出 caption JSON 与样本 ID/person slot 的现有映射方式。

确认后再提出最小实现 patch。

如果新会话只能记住一句话：

> 主线是“固定 90% mask 的现有 MacDiff diffusion + 单个 learnable query cross-attend 75 个 skeleton token并预测对应人物的连续文本特征”；双人动作在数据层拆成两条独立的单人骨架-单人文本样本，完全 AR 只作为另一个未实现 idea 保留。
