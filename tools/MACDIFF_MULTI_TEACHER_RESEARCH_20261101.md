# MacDiff 多教师蒸馏文献调研

日期：2026-11-01。范围：无标签表征蒸馏、多个冻结预训练教师、异构编码器、骨架自监督；优先核对论文正文、正式会议页和作者代码。下文把论文事实与对本项目的迁移推断分开。此次仅调研，没有修改模型或训练配置，没有启动新实验。

## 结论与优先阅读顺序

有直接相关工作。最贴近“MacDiff的掩码建模表征＋SCD的对比学习表征”分工的是 **Hybrid Distillation（ICLR 2024）**；多个固定教师各保留自己的特征空间、用学生专属投影头分别拟合，可参考 **AM-RADIO（CVPR 2024）与UNIC（ECCV 2024）**。我们遇到的特征尺度问题，与 **PHI-S（2024预印本）**研究的问题直接相关。骨架领域最有针对性的先例是 **IKEM（ICASSP 2024）**的六流教师到三流学生关系蒸馏。

本次未找到与“冻结MacDiff＋冻结SCD、训练一个随机MacDiff学生、逐层与最终输出分别监督、NTU60 XSub冻结LP”完全一致的已验证方案；这不是对首创性的系统性证明。已有文献足以支持开展原型实验，但不能承诺NTU LP收益。

| 工作 | 真正的多个冻结教师→单一学生表征 | 主要可借鉴之处 | 对本项目的边界 |
|---|---|---|---|
| Hybrid Distillation | 是，MAE＋DeiT/CLIP→随机ViT | MIM教师负责中间token关系，另一教师负责最后层特征 | 图像、原文全ViT；不是已验证的GCN→Transformer；教师来源含监督/图文 |
| AM-RADIO | 是，CLIP＋DINOv2＋SAM | 学生侧teacher-specific投影；不平均教师原始hidden | 无标签蒸馏不等于教师无标签；不是骨架实验 |
| UNIC | 是，多预训练ViT→一个ViT | 教师特征标准化、独立投影、teacher dropping | 教师含监督模型；同完整图像输入，不是90%骨架mask |
| PHI-S | 是，多教师表征蒸馏的分布平衡研究 | 处理MSE对教师方差的隐式加权 | 不是简单逐维白化；图像实验，不能直接保证骨架收益 |
| IKEM的关系蒸馏阶段 | 六模态教师→三流学生 | 纯骨架SSL、冻结教师、传递跨样本关系 | 均为ST-GCN，学生仍三流；不是单joint异构学生 |
| CoMAD | 是，MAE＋MoCo v3＋iBOT→ViT-Tiny | 多种SSL教师、非对称mask和共识融合 | 2025预印本；学生已有预训练，主要top1是微调；适配器可训练 |

## 1. Hybrid Distillation: Connecting Masked Autoencoders with Contrastive Learners

- 标题/作者/年份：Bowen Shi、Xiaopeng Zhang、Yaoming Wang、Jin Li、Wenrui Dai、Junni Zou、Hongkai Xiong、Qi Tian；ICLR 2024。
- 摘要：把掩码建模与对比学习教师的互补表征蒸馏给随机学生。
- 背景：单一特征蒸馏难同时保持局部多样性与全局判别性。
- 方法：冻结MAE及DeiT或CLIP，前者提供中间层token关系，后者提供最终feature maps，以Smooth-L1等项组合；教师完整前向，学生采用渐进冗余token masking。
- 结果：优势更突出于检测、分割与迁移；原文同分布ImageNet分类增益有限，不能将其微调结果当作冻结LP。
- 贡献/局限：提供“不同教师监督不同层次”的直接先例，但原文模型均为ViT，教师来源也不是全为纯SSL。
- 关键词：混合蒸馏；MIM；CL；token关系；渐进mask。

一手来源：[ICLR论文页](https://proceedings.iclr.cc/paper_files/paper/2024/hash/5598cf1b2905a26ddb863e6705588327-Abstract-Conference.html)、[正式论文§3.1–3.2、§4.2](https://proceedings.iclr.cc/paper_files/paper/2024/file/5598cf1b2905a26ddb863e6705588327-Paper-Conference.pdf)、[作者代码](https://github.com/lygsbw/hybriddistill)。

迁移推断：MacDiff可承担token/中间结构监督，SCD承担最终时空特征监督。MacDiff是条件去噪式骨架预训练，不应直接等同于MAE；SCD的最终池化特征也不是原文CL教师的patch feature maps。角色分工可借鉴，具体loss和token对应仍需重设计。

## 2. AM-RADIO: Agglomerative Vision Foundation Model Reduce All Domains Into One

- 标题/作者/年份：Mike Ranzinger、Greg Heinrich、Jan Kautz、Pavlo Molchanov；CVPR 2024。
- 摘要：聚合CLIP、DINOv2、SAM等冻结教师的不同能力，训练统一学生。
- 背景：教师的训练任务、输出分布和优势不同，不能自然合并为一组分类标签。
- 方法：随机学生通过各教师专属的学生侧MLP适配头，分别拟合summary及空间特征，使用cosine与Smooth-L1等目标；蒸馏阶段不使用数据标签。
- 结果：在多种分类、密集预测及视觉语言任务中保留或改善教师能力，并研究高效学生架构。
- 贡献/局限：支持跨任务教师特征的分头迁移；教师含图文和分割监督，不能将整个知识来源称为纯SSL。
- 关键词：多教师；冻结编码器；独立适配头；表征聚合。

一手来源：[论文§3.1–3.4](https://arxiv.org/html/2312.06709)、[CVPR正式论文](https://openaccess.thecvf.com/content/CVPR2024/papers/Ranzinger_AM-RADIO_Agglomerative_Vision_Foundation_Model_Reduce_All_Domains_Into_One_CVPR_2024_paper.pdf)、[作者代码](https://github.com/NVlabs/RADIO)。

迁移推断：即便MacDiff与SCD最终维度不同，也可以由学生分出独立投影头，各自进入教师空间。投影放在学生侧、冻结教师目标，是一个清楚且易控制的首版方案；推理/LP仍可仅使用学生主干。

## 3. UNIC: Universal Classification Models via Multi-teacher Distillation

- 标题/作者/年份：Mert Bülent Sariyildiz、Philippe Weinzaepfel、Thomas Lucas、Diane Larlus、Yannis Kalantidis；ECCV 2024。
- 摘要：把多个互补预训练模型的知识整合为一个通用分类表征。
- 背景：直接相加多教师loss会受到输出统计差异和学习难度差异影响。
- 方法：教师特征标准化、教师专属projector、学生中间层ladder projectors，以及平衡教师贡献的teacher dropping；蒸馏阶段丢弃标签。
- 结果：单学生在多类迁移任务保留或改善教师优势；评价包含冻结编码器的线性读出。
- 贡献/局限：提供多教师平衡和特征统计处理证据；教师含监督模型，主体实验为同架构ViT、同完整输入。
- 关键词：标准化；teacher dropping；projector；泛化。

一手来源：[论文§3.1–3.5](https://arxiv.org/html/2408.05088v1)、[ECCV正式论文](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/00581.pdf)、[作者代码](https://github.com/naver/unic)。

迁移推断：先做两个教师的分项loss及尺度/梯度检查，再决定是否需要teacher dropping。不能按原始MSE大小直接说某教师更可靠，也不必第一版就加入动态gate。

## 4. PHI-S: Distribution Balancing for Label-Free Multi-Teacher Distillation

- 标题/作者/年份：Mike Ranzinger、Jon Barker、Greg Heinrich、Pavlo Molchanov、Bryan Catanzaro、Andrew Tao；2024 arXiv预印本，本次核验页面未标正式会议录用。
- 摘要：研究无标签多教师蒸馏中教师激活统计对学生的影响。
- 背景：相同loss系数并不意味着相同监督强度，高方差教师可被隐式放大。
- 方法：比较统计归一化；PHI-S通过中心化、PCA与Hadamard正交变换及统一标量缩放平衡通道方差，保留可逆映射。
- 结果：在其多教师图像实验中优于所比较的归一化方案。
- 贡献/局限：直接提示尺度平衡的重要性；不是逐轴whitening，也不能认为LayerNorm天然解决跨教师分布差异。
- 关键词：激活统计；尺度平衡；PCA；Hadamard；无标签蒸馏。

一手来源：[摘要与作者](https://arxiv.org/abs/2410.01680)、[正文引言、§2.2.5及实验](https://arxiv.org/html/2410.01680v1)。

迁移推断：本项目A2/T尺度尚未实测确认；该论文说明这是需要排查的一般风险，不是对本次异常的直接诊断。多教师首版至少应记录各教师/学生的特征范数，并使用明确的归一化或标准化；不把完整PHI-S当作无需验证的必要组件。

## 5. Elevating Skeleton-Based Action Recognition with Efficient Multi-Modality Self-Supervision

- 标题/作者/年份：Yiping Wei、Kunyu Peng、Alina Roitberg、Jiaming Zhang、Junwei Zheng、Ruiping Liu、Yufan Chen、Kailun Yang、Rainer Stiefelhagen；ICASSP 2024，常按IKEM称呼。
- 摘要：利用骨架派生的互补模态并将较大模型知识蒸馏到较小多流模型。
- 背景：不同模态质量不一致，直接交换知识可能传递错误。
- 方法：六模态教师冻结后，向joint/bone/motion三流学生迁移embedding及anchor/positive/negative关系；采用学生MLP和memory bank。
- 结果：NTU60 XSub线性评价中六流教师81.2%、三流学生80.7%，基线3s-CrosSCLR为77.8%；这里学生没有超过六流教师。
- 贡献/局限：是纯骨架SSL的多源知识压缩证据；各流均为ST-GCN，学生仍三流，并非MacDiff+SCD异构教师到单joint学生。
- 关键词：骨架；跨模态关系；冻结教师；多流；知识压缩。

一手来源：[论文及venue](https://arxiv.org/abs/2309.12009)、[正文§2.3、§3及表1](https://arxiv.org/html/2309.12009)、[作者代码](https://github.com/desehuileng0o0/IKEM)。

迁移推断：不同维度、不同坐标空间可以通过对应样本的关系迁移知识。它支持我们保留关系蒸馏为候选，但不能说已有相同异构组合的成功复现。

## 6. CoMAD: A Multiple-Teacher Self-Supervised Distillation Framework

- 标题/作者/年份：Sriram Mandalika、Lalitha V；2025 arXiv预印本，本次核验版本为v1。
- 摘要：融合MAE、MoCo v3与iBOT教师，训练紧凑学生。
- 背景：SSL教师有互补性，简单固定融合可能不足。
- 方法：非对称mask、teacher侧Linear+LayerNorm适配、共识gate及双层KL；“parameter-free”不能理解为整个训练没有可学习适配器。
- 结果：原文报告ImageNet top1为75.4%，但这是后续有标签fine-tune，不是冻结encoder LP。
- 贡献/局限：提供多SSL教师和不同可见性的参考；学生从21K→1K蒸馏checkpoint初始化，不是随机初始化的纯SSL学生，mask也不是直接删除token。
- 关键词：多SSL教师；非对称mask；共识；适配；KL。

一手来源：[摘要](https://arxiv.org/abs/2508.04816)、[原文Adapter Projection、Asymmetric Masking及Experimental Setup](https://arxiv.org/html/2508.04816v1)。

迁移推断：可作后续算法参考，但证据适配度低于前述正式发表的基础方案；不优先照搬其teacher侧可训练适配器和复杂gate。

## 旁证与排除项

- [CrosSCLR，CVPR 2021](https://arxiv.org/abs/2104.14466)：joint/bone/motion多流在线协作和知识交换，不是多个固定教师→一个学生。
- [TMS-CL，ESWA 2026](https://www.sciencedirect.com/science/article/pii/S0957417426001181)：teacher→middle→student层级迁移，适合讨论输入/容量差，但不是两个独立异构教师并行融合；不依据摘要补写具体loss。
- [OFA-KD，NeurIPS 2023](https://arxiv.org/abs/2310.19444)：跨架构表征对齐的重要旁证，但不是纯SSL多教师设定。
- [CA-MKD，ICASSP 2022](https://arxiv.org/abs/2201.00007)：教师可靠性权重依赖ground-truth labels，不能直接搬进当前无标签预训练。

## 对MacDiff＋SCD的首版建议（本项目推断，未实现）

1. 停用外推残差，先以已有效的普通蒸馏为参照；不要把早期随机快照当作第三个知识教师。
2. 只用两个已有自监督预训练教师：固定MacDiff T20负责token/中间层结构，固定SCD负责最后池化特征，例如其时空特征拼接。SCD的CL projection向量不是60类概率，不能默认做分类KL。
3. 保留两个独立监督空间：MacDiff分支继续自己的对齐，SCD分支增加学生侧投影P，把学生最终特征映到SCD空间。分别计算loss再加权，不先平均两位教师未经对齐的hidden，不计算T_MacDiff−T_SCD。
4. SCD目标先用冻结输出，采用归一化后的cosine或有明确尺度统计的回归；不默认加入teacher-side可学习remap。两项loss应分开记录，并检查对学生的实际影响，不能把等权标量当成梯度平衡。
5. 输入可见性要单独决定。SCD使用正常完整骨架；若学生仍只看10%，就额外引入了partial→full预测难度。若先用完整学生输入验证异构互补，需配置同样完整输入的单教师基线，不能把改可见性的收益全部归于多教师；也需考虑完整序列反向传播的开销。
6. 对齐同一时间crop及人物可见性，分别保持教师预训练归一化/帧长。当前本地SCD为64帧/两人物布局，MacDiff为120帧/person0；直接喂同一数值tensor不等于输入语义相同。
7. 最小目标可写为L=L_MacDiff_KD+beta*(1−cos(P(z_S),stopgrad(z_SCD)))。这是依据RADIO式独立投影给出的原型，不是声称复现Hybrid Distill；若投影吸收了差异而主干无益，再比较IKEM式关系迁移。

先回答“两位教师是否有互补判别信息”，再决定是否做Hybrid Distill式token关系、更复杂的teacher dropping或PHI-S。不要预设架构不同就必然互补，也不要承诺可叠加教师各自的LP分数。

## 判读与创新边界

“多教师蒸馏”“MIM＋CL双教师分工”“多教师归一化＋独立projector”均已有明确先例，本身不能单独作为强新颖性主张。本项目若继续，需验证骨架中跨架构表征、人物/可见性信息差以及教师互补是否真实改善同协议LP；图像任务的检测/微调收益不能代替这个证据。

## 后续核验：教师原学习率与Hybrid的token选择

以下对比区分论文报告LR、按batch缩放后的实际LR和具体checkpoint证据。除本项目T20的实际日志外，外部教师主要能确认原模型公开标准recipe，不能宣称逐一核验了蒸馏论文所加载权重的完整训练历史。

| 模型/阶段 | 原训练LR与batch | 蒸馏对照及边界 |
|---|---|---|
| MAE | base 1.5e-4 × batch/256；B4096→实际2.4e-3 | Hybrid 6e-4为0.25倍；但MAE同B1024规则恰好也是6e-4 |
| DeiT-B | base 5e-4 × batch/512；B1024→实际1e-3 | Hybrid 6e-4为0.6倍 |
| OpenAI CLIP-B/16、L/14 | 主训练5e-4、4e-4，B32768 | Hybrid 6e-4反而为1.2、1.5倍；L/14@336的2e-5是额外高分辨率阶段 |
| DINO、iBOT | base 5e-4 × batch/256；B1024→实际2e-3 | UNIC论文报告3e-4为0.15倍；当前代码默认另按batch缩放，不能混用 |
| DeiT-III-B | 主训练LAMB 3e-3/B2048；后续224微调AdamW 1e-5/B512 | 不能将不同优化器、训练阶段压成一个教师LR |
| dBOT-B | 预训base1.5e-4/B256，B4096→实际2.4e-3；分类微调base3e-4/B256，B1024→最高组1.2e-3 | UNIC使用dBOT-ft，不能只引用其自监督预训LR；微调另有layer decay |
| DINOv2-g | AdamW 3.5e-4/B3072 | AM-RADIO AdamW1e-3数值约为2.86倍 |
| SAM-H | AdamW 8e-4/B256，layer decay0.8 | AM-RADIO 1e-3数值为最高组的1.25倍 |
| 本项目MacDiff T20 | 实际峰值1e-3、floor1e-5，global B64 | 历史masked师生蒸馏峰值1e-3、floor5e-4、global B256；新完整教师组已按用户改为峰值1e-4、floor1e-5，即峰值为教师0.1倍、floor相同 |
| 本地SCD README推荐配置 | SGD .01/B64/451epoch，351轮降到.001 | 历史学生AdamW .001数值低10倍；新完整教师组AdamW .0001数值低100倍；优化器不同，且尚未选定SCD权重，不能当作其实际checkpoint协议 |

来源：[MAE附录A.1](https://arxiv.org/html/2111.06377#A1.SS1)、[DeiT §6](https://arxiv.org/html/2012.12877#S6)、[CLIP附录F](https://arxiv.org/html/2103.00020#A6)、[DINO §3.2](https://arxiv.org/html/2104.14294)、[iBOT §3.2](https://arxiv.org/html/2111.07832)、[DeiT-III命令](https://github.com/facebookresearch/deit/blob/main/README_revenge.md)、[dBOT预训](https://github.com/liuxingbin/dbot/blob/main/PRETRAIN.md)、[dBOT微调](https://github.com/liuxingbin/dbot/blob/main/DOWNSTREAM.md)、[DINOv2附录B](https://arxiv.org/html/2304.07193)、[SAM附录A](https://arxiv.org/html/2304.02643)。T20本地证据为`handoff_artifacts/official_dualgpu_results_20261013/README.md`与PT日志；SCD为`D:/Program/SCD-NET/SCD-Net/README.md`的推荐命令。

UNIC补充：论文附录A报告LR3e-4/global B512；[公开main_unic.py](https://github.com/naver/unic/blob/main/main_unic.py#L235)将默认args.lr3e-4再乘global_batch/256，README默认4×128时实际峰值6e-4。前次汇总表的3e-4是论文报告值，不能据此断言当前默认代码实际也为3e-4。PHI-S蒸馏使用LAMB1e-3，不是AdamW；其DFN CLIP不能直接套OpenAI CLIP的训练recipe。IKEM的KD独立LR、CoMAD三位实际教师各自LR未足够披露，不补猜倍率。

Hybrid的progressive指单次教师前向的不同深度，而非epoch调度。依正式论文§3.1–3.2式1/4与表8，两个教师完整处理x；在MIM教师的0、L/3、2L/3三个深度，从当前候选位置中剔除与其均值token最相似的前K%。最终位置集合V从学生输入端就固定保留，不是学生在各层继续删token。K=.3时可见比例为.7³=.343，约65.7%mask；原文较好的折中不是90%mask。

位置对应的实现解释：教师特征F_T完整前向后取F_T[V]，学生只输出对应k个token；教师softmax前关系R=QK^T取R_T[V,V]与学生k×k关系比较。双轴gather是将论文掩码简记落实为张量操作的解释，作者未提供代码以进一步核验。索引对齐只解决形状和时空位置，无法消除教师看到被删上下文、学生看不到的信息差。

对本项目的迁移推断：同架构MacDiff可先做教师750token完整前向，再对输出按学生ids_keep gather；旧默认路径在block前gather，师生都只看75token。用户随后已授权实现`teacher_full_visible=True`新路径与独立ep100配置，先保持随机90%mask/逐层MSE/lambda1，只改变教师可见性，尚未运行；具体命令和检查见`handoff.md`顶部。若采用原文约34%可见，学生约257token，不能继续称为当前90%mask的同一协议。接近均值的骨架token未必无用，原文图像选择准则对动作/关节覆盖的作用未验证，因此teacher选择mask仍留作后续。SCD保留自己的正常结构化完整输入，最终全局表示做batch样本关系蒸馏，不要求与MacDiff逐token对应；这是前面投影方案之外、当前优先讨论的无扩维候选，尚未实现。不要混淆Hybrid的单样本token×token关系与SCD候选的batch样本×样本关系。

## 后续核验：学生蒸馏轮数相对教师原训练预算

用户明确主要目标是达到/超越教师性能，先以T20约86%的冻结LP为参照，不将额外因果消融作为当前运行的前提。以下教师数字来自对应原模型公开标准配方；Hybrid没有公开可锁定的教师权重文件，UNIC仅部分下载入口能落实，不能把所有标准配方都当作具体checkpoint训练日志的直接证据。

| 对照 | 教师原训练 | 学生蒸馏 | 轮数/步数比较及边界 |
|---|---|---|---|
| Hybrid / MAE | 主结果1600epoch，IN1K | 主实验300epoch，IN1K | 18.75%，少1300轮；学生随机初始化 |
| Hybrid / DeiT-B | 300epoch，IN1K | 300epoch，IN1K | 相同轮数 |
| Hybrid / CLIP | 400M图文对上32epoch；L14@336另1epoch | IN1K上300epoch；IN21K版本40epoch | 数据不同，不能按300/32解释训练量；基础样本抽取量粗算384M/12.8B约3%，不等于FLOPs比 |
| UNIC / DINO、iBOT ViT-B/16 | 各400epoch，IN1K | 基础100、完整含teacher dropping为200epoch | 分别25%、50%；800epoch是ViT-S，不可套给ViT-B |
| UNIC / DeiT-III-B | 800epoch主训练＋20epoch224分辨率适配 | 200epoch | 约24.4%；官方下载文件deit_3_base_224_1k.pth |
| UNIC / dBOT-ft分类配方 | 2×800=1600epoch预训练＋100epoch有监督微调 | 200epoch | 对完整阶段数值约11.8%；不能误用dense任务3×800配方 |
| IKEM | 六流教师300epoch | 三流学生150epoch | 50%；原文未明确学生随机/已有权重初始化 |
| CoMAD | 正文称MAE/MoCov3/iBOT各300epoch | 追加KD300epoch | 新增阶段100%，学生此前已有21K→1K distilled初始化；正文与表1中MoCov3 baseline800epoch有口径差异 |
| AM-RADIO / DINOv2-g | 625kstep/B3072，另10k高分辨率适配 | 600kstep，通用B1024 | 主step数约96%，不能当作相同数据/计算预算 |
| AM-RADIO / SAM-H | SAM阶段90kstep/B256，约2个SA1B epoch，且此前已有MAE初始化 | 600kstep | 比SAM这个阶段多约6.67倍step；不能把教师此前MAE成本删掉 |
| PHI-S / DINOv2-g | 625kstep主训练 | 消融300k；正式模型3×300k=900k | 分别约48%、144%step；学生已有TIMM预训练初始化 |
| 本项目MacDiff | T20 400epoch/global B64 | 当前100epoch/global B256 | epoch为25%，相同数据量下更新次数约1/16；不等于必须补足相同步数 |

来源：[Hybrid §4.1及附录A.3](https://proceedings.iclr.cc/paper_files/paper/2024/file/5598cf1b2905a26ddb863e6705588327-Paper-Conference.pdf)、[MAE §4.2](https://arxiv.org/html/2111.06377#S4.SS2)、[DeiT §6](https://arxiv.org/html/2012.12877#S6)、[CLIP §2.5](https://arxiv.org/html/2103.00020#S2.SS5)、[UNIC附录A](https://arxiv.org/html/2408.05088v1#A1)、[iBOT §3.2及附录B表10](https://arxiv.org/html/2111.07832)、[DeiT-III配方](https://github.com/facebookresearch/deit/blob/main/README_revenge.md)、[UNIC的DeiT-III下载脚本](https://github.com/naver/unic/blob/main/scripts/teachers/deit-iii.sh)、[dBOT预训](https://github.com/liuxingbin/dbot/blob/main/PRETRAIN.md)、[dBOT分类微调](https://github.com/liuxingbin/dbot/blob/main/DOWNSTREAM.md)、[IKEM §3.2](https://arxiv.org/html/2309.12009)、[CoMAD实验设置](https://arxiv.org/html/2508.04816)、[AM-RADIO §4](https://arxiv.org/html/2312.06709)、[PHI-S §3/附录A.7](https://arxiv.org/html/2410.01680)、[DINOv2附录B](https://arxiv.org/html/2304.07193)、[SAM附录A](https://arxiv.org/html/2304.02643)。

达到/超过教师的证据需按评价协议解释：Hybrid的85.1对CLIP83.6/MAE83.6是监督fine-tune，LP附表仅提供MAE67.8、Hybrid(MAE+DeiT)80.9/(MAE+CLIP)80.1，未并列对应CL教师LP，不能直接说LP超最强教师。UNIC采用冻结特征LogReg；四教师学生ImageNet83.8仍略低于最佳教师84.0，但迁移平均75.1高于最佳教师72.4、ADE39.6高于36.6。IKEM NTU60XS学生80.7低于六流教师81.2，XV86.0低于86.9。CoMAD的75.4为Tiny监督微调，ViT-B84.7也不能当作随机Tiny冻结LP超教师的证据。

对当前训练预算的判断（项目推断）：100轮是第一轮性能尝试，而不是文献保证的充分训练。学生更新次数较少且峰值LR从教师1e-3降至1e-4；若100轮未达到教师，不足以排除路线，后续200/400轮可作为独立日程候选，是否延长应参考同协议LP及不同checkpoint趋势，不能只看训练MSE/cos。此处仅分析，没有修改100轮配置或启动额外训练。
