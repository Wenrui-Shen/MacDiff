# MacDiff 会话交接（2026-10-10）

本文写给完全没有上下文的新会话。用户在服务器运行训练；本地助手负责代码/论文审查、日志分析、准备配置及提供命令。先读第 0 节最新状态，再按用户继续讨论的方向读后面的记录。写入前的完整交接已备份到 [2026-10-07 交接归档](D:/program/MacDiff/handoff_artifacts/handoff_before_20261008_session_close.md)。

## 0. 2026-10-10 最新状态（优先于下文历史计划）

- **T15：当前 no-share 骨架 decoder3 组，用户口述 LP 成绩 85.84%。**按上一会话上下文登记为 native/T→S 各3层、S→T text decoder 5层/hidden512/output_norm=none，T→S=1、S→T=0.1、归一化A、PT不旋转。85.84来自本次用户反馈；未提供完整PT/LP日志，best/末轮统计口径、best epoch、last及末20均值未核实。最终实际batch/accum、checkpoint、完整args及服务器SHA也未核验，不用库存YAML或失败尝试命令补填。
- **当前下一组：用户自己将 text_decoder_depth 从5改为3试验。**以上述85.84组为控制，骨架 decoder_depth 保持3（native/T→S不再改），仅S→T文本decoder深度5→3；hidden512/output_norm=none、权重、cache/target、归一化和实际PT/LP协议保持控制组设置。新组暂未回传成绩；本次助手只登记，没有修改训练YAML/模型或启动训练。
- **随机旋转方向已明确改为原版MacDiff。**用户要求准备原版不旋转/旋转的顺序复现脚本；两组共同decoder3、PT500、lr1e-3/min_lr1e-5，双卡PT64/accum1，checkpoint499 LP100、lr.1、双卡LP128/accum1；仅PT random_rot不同，LP训练旋转True/测试False。脚本已完成并静态/CPU验证，训练由用户在服务器执行，尚无这两组成绩。见 [启动脚本](D:/program/MacDiff/script_pretrain_madiff_rotation_ablation.sh)、[说明及核对参数](D:/program/MacDiff/tools/MACDIFF_ROTATION_ABLATION.md)。此原版对照不叠加文本分支，不以T15或text decoder3为旋转控制。
- 10月9日current.shared.json/FileExistsError保留为历史失败尝试。现在用户已回传decoder3成绩，不能再将其描述为当前仍未成功训练；具体处理方式及最终目录未核验。T14是原版decoder3/PT500的85.704755%，与本次文本no-share的T15/85.84%是不同实验，不互相覆盖。
- 已同步 [Stage1成绩](D:/program/MacDiff/STAGE1_EXPERIMENT_RESULTS.md)、[实验总表](D:/program/MacDiff/EXPERIMENT_RESULTS.md) 和 [当前计划](D:/program/MacDiff/tools/vlm_pilot/STAGE1_TEXT_EXPERIMENT_PLAN.md)。

### 2026-10-09 状态快照（历史，以本节顶部为准）

- 新结果 **T14：原版/A、用户报告decoder3、PT500、checkpoint-499 LP100，best85.7047549825684% @94；last85.5955846991023%，末20均值85.63591705546224±0.038371717383694964%**。完整PT500+LP100合并附件已分析，无缺轮/重复/非有限值。
- 用户更正实际LP命令为checkpoint-499。**之前给的399命令失效，不能再说85.70漏评了最后100轮，不能重复要求跑499。**
- 实际命令：PT双卡64/accum1=128，500轮；LP双卡128/accum1=256，100轮，lr.1，seed0。PT日志确认min_lr=5e-4（基础YAML值，命令未覆写），不是论文/脚本的1e-5。A与decoder3依用户说明，actual args/state_dict/server SHA未实物核验。
- PT/LP用了同一output_dir/ntu60_xsub_macdiff_decoder3，日志/TensorBoard混用；LP每轮保存，所以PT0/10/…/90被同编号LP覆盖。PT399/499不受LP0–99覆盖。后续使用独立目录，不清理已有文件。
- PT399→499 loss仅降0.9843%，末50稳定；LP已平台，90–99 lr0但BN仍更新。不能凭loss或这一次结果保证更低min_lr/3层能达到86.4。T12与本组LP batch/BN不同，不能把全部差值归于文本。
- **用户已选当前decoder3实验：以已完成T12（1/.1、PT400、A）为控制，只改decoder_depth5→3；native与no-share T→S都会变3，S→T维持5层/512/none。PT32×accum2、min_lr1e-5、LP64×accum1、seed0保持T12。**此前1/.5计划仍缺结果，但不自动叠加到本组。
- 新独立配置：[noshare_decoder3 YAML](D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_text_sentence_sample_target_blend_noshare_decoder3.yaml)；[中文分析](D:/program/MacDiff/handoff_artifacts/decoder3_20261009/README.md)、[统计JSON](D:/program/MacDiff/handoff_artifacts/decoder3_20261009/summary.json)、[曲线](D:/program/MacDiff/handoff_artifacts/decoder3_20261009/comparison.png)、[单行PT&&LP命令](D:/program/MacDiff/handoff_artifacts/decoder3_20261009/launch_noshare_decoder3.sh)、[用户更正后的原命令](D:/program/MacDiff/handoff_artifacts/decoder3_20261009/commands.json)。
- 新YAML把库存父文件的权重/batch/accum默认值设成T12实际命令，实验因素仍只有骨架深度。未改原配置/模型，未启动服务器训练、未提交/推送。本地依然无Torch/PyYAML；仅做标准库日志分析、AST/文本配置检查、ReportLab+PDFium绘图及视觉核验。

### 2026-10-09 追加：下次单独测试PT随机旋转

用户已明确要求记录并在下次实验测试。当前no-share decoder3组继续PT random_rot=False；完成后以其为控制，**仅开启train_feeder_args.random_rot=True**。其余深度、权重1/.1、A、PT400、min_lr1e-5、cache/target/seed、实际PT batch/accum与LP协议全部固定。source_rot/flip保持False；LP训练旋转True、测试False保持不变。详见 [最新实验计划](D:/program/MacDiff/tools/vlm_pilot/STAGE1_TEXT_EXPERIMENT_PLAN.md:3)。仅登记计划，未改YAML或启动旋转训练，没有成绩。

公共random_rot在crop后、模型标准化前执行；XYZ每轴角度±.3弧度，同片段所有帧/关节/人物共享R，encoder输入与diffusion干净骨架同转。它与source_rot（仅encoder额外旋转）不同。固定caption的视角方向语义可能受影响，属待验证推断，现有cache不自动重生成。不能保证旋转解释全部复现缺口或必然提高LP。

**2026-10-09启动尝试状态（历史，当前已报告T15成绩）：**用户回传的decoder3 PT命令实际batch64/accum1（原保存命令为32/2），双卡有效128；因已有current.shared.json、未传resume，在prepare阶段报FileExistsError，尚无成功PT/LP新成绩。不要说当前已完成、不要把失败尝试的batch或库存默认当最终控制的实际参数。以后旋转组匹配最终成功控制的actual args。该报错与旋转无关；换未使用的目录从头训练，或用同次模型+target-bank配对快照正式resume；不删除旧目录、不绕过复用保护。当前未确认用户是否已处理该启动障碍。

## 1. 2026-10-08 接手结论（历史状态，以第0节为准）

我们有两条相关但必须区分的工作线：

1. **原版 MacDiff 复现审查**：用户复现不到论文性能，已明确目标是 **NTU60 XSub、joint 单流、linear probe（LP）**。检查 NTU60/NTU120/PKU 预处理、官方配置的 one_person 和 mean/var、论文与公开代码的差异，找出可控消融因素。
2. **文本扩展方法**：逐样本/逐人物的 global + 六部位文本辅助骨架预训练，最终仍用只输入骨架的 LP。当前方法是 **no-share、512 维七句文本、sample_target_blend、骨架归一化 A**。

接手时必须知道：

- v3 七句 CLIP cache **已经生成并在服务器训练成功**，不是待生成阶段。本地没有该 cache，不能把本地缺文件解释为服务器尚未生成。
- 已完成 no-share 对照 **T12：T→S=1、S→T=0.1**，PT400 + LP100，LP best **86.196021% @75**；末轮 85.892771%，末 20 轮 86.000728±0.061392%。有完整 PT/LP 日志。
- 截至10月8日用户确定的下一组是 **no-share、T→S=1、S→T=0.5**；10月9日当前新选择见第0节。用名字含 st02 的 YAML，但 CLI 覆盖为 0.5，目录后缀必须是 _s2t05。尚无回传结果，助手没有启动训练。
- 原版归一化 B 实验 **T13 已完成**，LP best **85.183164% @92**。它同时使用 PT/LP 有效 batch256、PT min_lr=5e-4，不能当纯 A/B 消融。
- 本轮确认三套 decoder **都有带 bias 的线性输出 projection**；S→T 的 output_norm=none 只是没有最终 LN，投影仍是 Linear(512,512)。
- 最新用户疑问是“512→512 是否太接近，是否值得改”。助手建议保留当前 512/none 为基线，可只改 S→T hidden 为 **768** 做容量消融；这是建议，**用户尚未决定实施，未改代码/配置**。不能将它自动替代已选的权重 0.5 计划。
- 本轮已完成官方/论文/文本配置静态审查，并新增报告与 11 份原生消融 YAML。**没有修改原有训练配置、模型或启动任何新训练。**
- 当前主要卡在**缺服务器实际 args/SHA、数据及 checkpoint 身份、匹配的控制组结果**；没有证据认定某个 decoder 差异就是性能原因，也没有一个已确认的训练报错等待修复。

## 2. 环境、用户偏好与操作边界

| 项目 | 信息 |
|---|---|
| 本地仓库 | D:/program/MacDiff，Windows PowerShell |
| 本地 HEAD | 38016d82ef40db3f3f8e7779a6be49725ea10892；本轮核验过 |
| Git remote | git@github.com:Wenrui-Shen/MacDiff.git |
| 官方比较版本 | LehongWu/MacDiff，692888f1e2a4227216511aac6733dff31b587213，2025-07-06 |
| 服务器仓库 | /home/user9/public3/swr/MacDiff |
| 服务器数据 | ../data/MAMP/ntu/NTU60_XSub.npz，预期 train40091/test16487 |
| 服务器实际版本 | 尚无当前 SHA/完整 Namespace；不要把历史 edc924a 当当前版本 |
| 历史训练环境 | conda macdiff，Python3.8、PyTorch1.8.1+cu111、两张 RTX4090 24GB；实际环境以服务器为准 |
| 本地 Python | C:/Users/97537/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe |
| 本地依赖 | 2026-10-08 复核：有 numpy/pypdf/pdfplumber，缺 torch/yaml/transformers |
| 本地数据边界 | D:/program/data/MAMP/ntu/NTU60_XSub.npz 不存在；没有可直接检查的实际复现 checkpoint |
| 服务器访问 | 本会话没有 SSH，用户自己执行 Linux 命令 |
| 用户论文 | D:/program/paper/macdiff.pdf，18 页主论文；已提取全文并渲染检查第 7、9、10、14 页 |

用户用中文；给训练命令要**单行 Linux、不要 cd，PT 与 LP 用 && 连接**，通常双卡。当前文本 PT 每卡32/accum2，LP每卡64/accum1，二者有效 batch128；原版 B 按用户要求每卡128/accum1，有效256。不能为了贴近官方四卡脚本擅自改用户的文本 LP 协议。

本地默认 sandbox 的 Windows helper 初始化曾失败；require_escalated 的 exec_command 经自动审查后可用。写文件使用 PowerShell 单引号 here-string，避免内容插值；长命令曾超过 Windows 命令长度上限，应分段。无需为写文档安装 Torch。静态 AST/NumPy 检查不等于真实 autograd/CUDA/AMP/DDP 验证。

用户明确约束：暂不做对比学习、跨样本关系蒸馏或其他跨样本内容；保留现有 uniformity。固定 CLIP 干净目标、固定噪声的条件打乱、方差/相似度诊断已经试过，不要重新当作默认新方向。v 参数化、区域 bias 仅讨论，未实现。

## 3. 已有结果与证据等级

完整历史成绩查 [实验结果总表](D:/program/MacDiff/EXPERIMENT_RESULTS.md)，只看 Stage1 查 [Stage1 汇总](D:/program/MacDiff/STAGE1_EXPERIMENT_RESULTS.md)。

| 实验 | 已知成绩 | 证据与限制 |
|---|---|---|
| T01 历史原版 MacDiff | 约85.86% | 历史摘要，缺实际 args/完整 LP/服务器 SHA；不要以当前默认值补填历史参数 |
| 旧256/LN、T→S=.1/S→T=.1 | 85.801795% | 与七句512组有目标/结构等差异 |
| 旧256/LN、T→S=1/S→T=.1 | 85.892771% | 不能作为纯 LN 或纯 hidden 对照 |
| T11 七句512/shared、1/.1 | **85.79%** | 用户更正后的口述 best；不是85.75，缺完整LP |
| 七句512/shared、1/1 | 用户报告 LP 明显下降 | 没有精确 best，不能编数值或当成 no-share 权重1结果 |
| T12 七句512/no-share/A、1/.1 | **86.196021% @75** | 完整 PT400/LP100；末20均值86.000728，std .061392 |
| T13 原版/B/batch256/min_lr5e-4 | **85.183164% @92** | 完整 PT/LP；末轮85.037603，末20均值85.110990，std .043181 |
| 当前 no-share/A、1/.5 | 尚无结果 | 已保存命令；未确认服务器实际开始/完成，助手未执行 |

T12 相比 T11 口述 best 高 .406021 个百分点，仅单次候选收益，不能宣称显著或稳定。T12 是扩展方法，不能当原生 MacDiff 复现成绩。T13 缺同协议原版 A 对照，不能单独归因于 B。

用户试过 shared512 的 checkpoint-150 LP，仅首轮“72多”，399 首轮口述74.65，**决定不继续150 LP**。不能把首轮当最终成绩，也不要再催跑完。当前通常评价 checkpoint-399。

历史512训练：S→T MSE 最低在 epoch142 为 .00996099，末轮 .02788974，后期回升；native/T→S 仍下降。no-share 末50轮 S→T 比 shared 低24.52%，后期回升仍在。低 loss、较高 variance 或 energy 不能证明 LP 改善，也不能据 loss 占比断言 encoder 梯度不足。

## 4. 当前文本方法的完整含义

逐原始样本、逐人物生成 global + 六句 local，顺序固定：
head、torso、left_arm、right_arm、left_leg、right_leg。front/side 是同一人的两个视角，不是两个人。缓存查表必须用原始 NPZ sample index/person slot，不能用 shuffle 后 batch 行号。

当前缓存：服务器项目下 vlm_pilot/ntu60_xsub_clip_sentence_cache_v3。每句独立用冻结 CLIP 的 pooled projected text_embeds 编码，512维、FP32、unit RMS：

- person_features.npy：[N,2,512]，global。
- token_features.npy：[N,2,6,512]，完整句向量；名字 token 不表示 BPE 词元。
- person_valid.npy：[N,2]；token_mask.npy：[N,2,6]，另有 samples.json/manifest.json。
- 训练 sequence 是 **7 个向量**，77 只是离线 CLIP 句长上限。缓存约1.07GiB，person0目标库约.535GiB，另有元数据。

当前 one_person=True，固定 person0；空 person0 过滤，不用 person1 替代。v3 flip=False。骨架 input120帧、patch4×1、750tokens、mask90%保留75；S→T condition 是 pooled+75 visible，共76个 skeleton tokens，encoder 仍256维。

sample_target_blend 是**保存目标值递推，不是 remap 参数 EMA**：
```text
initial_target_i = RMS(CLIP_i)
成功 optimizer step 后，用 step 后的 online remap：
saved_target_i = 0.9 * old_target_i + 0.1 * online_remap(RMS(CLIP_i))
```
global/有效local 都更新；AMP 跳步不更新，累积窗口在成功 step 后处理。混合后的 bank **不重新 RMS**，energy 可低于1。text_target_momentum 默认 .999 仅用于旧 ema_remap，当前模式不使用。

三条任务：

| 分支 | 主体/条件 | 输出头及归一化 | 直接训练 encoder |
|---|---|---|---|
| native 骨架 | 5层、hidden256；骨架 global-local 条件 | LN256→Linear256→12 | 是 |
| T→S | no-share deepcopy native 主体、时空embedding；额外逐层文本reader | LN256→Linear256→12 | 否，属已确认设计 |
| S→T | 独立5层、hidden512；骨架reader+文本self-attention、MLP2048 | Identity→Linear512→512 | 是，经 skeleton memory |

native 和 T→S 共用同一个 noisy skeleton、t、epsilon 和 mask，仅对 masked patches 求 MSE。S→T 对 detached saved text 独立采 t/epsilon，对全部 valid global/local 通道求 MSE；global1个、local6个，同权平均，非各占一半。

share_skeleton_decoder 只控制 native 与 T→S 是否共享骨架 decoder 主体，**S→T 始终独立**。no-share 复制初始 native 参数后独立更新，仍保留 T→S→remap→saved target→S→T→encoder 的间接路径。

总 loss：native + .02×skel_uni + .02×text_uni + λT2S×T2S + λS2T×S2T。当前计划权重1/.5，已完成T12为1/.1。text_uni只含同样本有效local两两cosine平方、含对角，不含global/跨样本；六句理论下限1/6。text_batch_variance统计在线global+local，text_energy统计保存global，口径不同。

LP 只加载骨架 encoder，不读 text/remap/decoder/bank。linprobe2 对人物和时间平均，保留25关节，得到6400维，再 BatchNorm+Linear60；不是 global256 或文本512读出。

参数作用：text_hidden_dim 是 remap 中间层；text_decoder_hidden_dim 只改 S→T；text_decoder_depth 只改 S→T。decoder_depth 同时改 native 与复制的 T→S。dim_feat 会改 encoder、两套骨架 decoder 和下游形状，不要用它替代文本容量开关。


## 5. 官方代码与论文审查：已确认结论

官方源：[LehongWu/MacDiff 固定提交](https://github.com/LehongWu/MacDiff/tree/692888f1e2a4227216511aac6733dff31b587213)。补充材料：[作者 supplement](https://lehongwu.github.io/ECCV24MacDiff/macdiff-supp.pdf)。详见 [原版复现审计](D:/program/MacDiff/handoff_artifacts/macdiff_reproduction_20261008/README.md)。

### 论文成绩与公开默认设置的口径

- 主论文第10页表1报告 NTU60 XSub LP **86.4%**；默认 encoder8/decoder5、hidden256、heads8、MLP1024。
- 第14页表10 decoder2/3/4/5层成绩 **84.8/86.4/86.0/85.9%**，正文称3层最好，但默认5层兼顾生成。
- 官方及本地基础 YAML 均5层。历史T01约85.86接近论文5层85.9，但 T01身份不全。
- **不能断言主表86.4必定使用3层，也不能宣布T01已完整复现。** 有根据的实验是5 vs 3，重新PT、固定LP。

| 因素 | 论文 | 官方/本地基础设置 | 判断 |
|---|---|---|---|
| PT epoch | 500 | XSub YAML及脚本400 | 明确差异 |
| PT rotation | crop + rotation + Gaussian noise | random_rot注释，默认False | 明确差异；LP训练rotation=True |
| PT batch | 四卡总128 | 官方4×32=128；当前native脚本2×32=64 | 实际入口/卡数决定 |
| PT min_lr | 1e-5 | YAML5e-4，官方/current脚本覆写1e-5 | 不要只看YAML或只看脚本 |
| diffusion loss | 式6写完整epsilon误差 | masked patch MSE + .02 token_uniformity | 论文未充分明示；待消融，不能直接叫bug |
| LP | frozen encoder、SGD100、lr.1 | 符合；head/BN/batch/末轮日程更细 | 需固定评价口径 |

1000步、epsilon、inverse_cosine tau1、mask.9、AdaLN/global-local、native条件drop.1与论文一致。本轮比较的初始化、随机mask、patchify、forward_loss、sampler更新等原生方法（忽略docstring的AST）与官方一致；forward_encoder新增可选activation checkpoint，原生调用默认关闭。schedule函数AST一致，字符串 inverse_cosine 与 ['inverse_cosine',1] 的beta最大差0。**这不是端到端Tensor/梯度等价验证。**

全部 **18份官方基础PT/LP/FT YAML** 与本地对应文件比较，仅30处data_path迁移；统一路径/换行/尾空白后包括注释均一致。本地基础扩展前目录共35配置，官方18，其余17为文本/其他扩展；本轮11个新消融文件位于handoff_artifacts，未改原config。下游模型全文规范化后与官方一致。

### one_person 的实际范围

- 官方 native PT 构造器默认 True；YAML的 #one_person: False 只是注释。固定选择person0，不随机、不自动选择更活跃的人。
- 基础 LP 没有 one_person 构造参数；读两个人物槽位，在head聚合，空槽未过滤。直接给LP model_args添加该参数会失败。
- native PT one_person=False 可测试，但diffusion使用两人，uniformity仍仅person0，不能称为均衡双人训练。
- 当前文本 sample_target_blend engine **要求 one_person=True**；直接改文本YAML为False会被拒绝，双人目标库需另实现。

### #new mean/var 与跨协议矩阵

实际标准化为 (raw-mean)/sqrt(var)，在模型内部执行。feeder normalization=False **不关闭它**。mean/var 是普通Python属性，不在state_dict里；加载checkpoint不会替换LP配置中的统计量。

| 组 | mean | var |
|---|---|---|
| A（当前文本与NTU60基础） | [-.0058,-.1333,-.0246] | [.0206,.0805,.0218] |
| B（PT/LP中的注释#new） | [-.0024,-.2132,-.0446] | [.0525,.1527,.0513] |

B相对A在扣除各自均值后的XYZ幅度约变为 [.6264,.7261,.6519]；epsilon仍单位Gaussian，因此有效SNR改变。不是直接把noise MSE按var缩放。没有找到统计生成脚本或统计范围证据，**#new不证明B更正确或来自NTU120**。NTU60 FT另一个#new不是B，勿混淆。

官方生效组合：

| 协议 | PT | LP | FT |
|---|---|---|---|
| NTU60 XSub/XView | A，shiftFalse | A，shiftFalse | A，shiftFalse |
| NTU120 XSub/XSet | A，shiftFalse | A，shiftFalse | 另两组E/F，shiftFalse |
| PKUv1 CrossSubject/CrossView | A，shiftTrue | A，shiftTrue | A，shiftFalse |
| PKUv2 CrossSubject/CrossView | H，shiftFalse | A，shiftFalse | A，shiftTrue |

self_shift=True 是按每个人T/V中心化、模型mean置0而保留var；不能当无归一化。E/F/H精确值和各文件位置见 [全配置审计](D:/program/MacDiff/handoff_artifacts/upstream_config_audit_20261008.md)。官方两份NTU120 FT num_classes=60与120类不符，是继承问题，但**不是当前NTU60 LP原因**。

A/B收益必须从头成对PT/LP比较。只给A checkpoint的LP换B属于输入错配诊断。T13还有batch/LR差异，缺同协议A控制；新的C1配置已准备，但未跑。

### 数据预处理检查

| 数据集 | 已读代码流程 | 差异/待核实 |
|---|---|---|
| NTU60 | raw skeleton→bodyID人物整理/去噪→最多两人→clip平移→pad→XSub/XView NPZ | origin为person0首个有效帧joint2；单人第二槽置零；frame_translation缩放函数未在主流程调用 |
| NTU120 | 同类去噪/clip平移/pad→XSub/XSet | 单人第二槽复制person0，区别于NTU60空槽 |
| PKU v1/v2 | 按官方split、标签start/end截动作→去坐标求和为0实例→前300帧截断/补零→51类onehot | 没有NTU式去噪/clip平移；start:end半开切片，原标签帧编号约定尚未核验 |

共用在线feeder将 [N,T,150] 变为 [N,3,T,25,2]；训练crop有效长度50%–100%再resize120，验证中心取95%。代码继承官方/MAMP主要逻辑，本地主要迁移路径。

需检查服务器实物：NPZ SHA、split/count/classes/onehot、NaN/Inf、空/复制人物、内部空帧；有效帧判断采用坐标求和!=0，且crop假定有效前缀，存在抵消/空洞风险。继承的raw读取按“该body上次位置+1”记录再次出现的位置：另一人仍在时某人暂缺可能压缩该人的时间轴；**仅发现潜在条件，未量化，不要先改数据再称同协议复现**。PKU代码51类与论文52 categories也需对照标签，不能给NTU60定因。

### LP 中容易忽略的实际运行差异

- head为 BatchNorm1d(6400,affine=False,eps=1e-6)+FC；无SyncBN。encoder eval/frozen、head train正确。
- 官方4×64=global256；文本2×64=global128；T13两卡每卡128虽然global256，**BN局部batch与官方不同**。累积或DDP buffer广播不等于SyncBN。
- YAML已有lr=.1，故为绝对LR；改batch不会自动按blr缩放。
- min_lr_epochs默认10，LP最后10轮lr=0；BN running stats仍更新，best@92不能解释为FC该轮继续更新。
- dist_eval把16487补为16488（1个重复），最大影响约.0061个百分点，不能解释.5个百分点。
- 双卡64×accum2可保留global256/局部BN64，但每卡313microsteps有末尾未提交累积，BN前向仍发生；并非四卡逐位等价。
- 按样本数推算官方PT400约125200optimizer steps，T13约62400；这是预算推算，非实测成功AMP steps。

## 6. 三套 decoder 的差异与最新宽度讨论

详见 [no-share完整审计](D:/program/MacDiff/handoff_artifacts/noshare_config_audit_20261008/README.md) 和 [输出头/LN/dropout消融设计](D:/program/MacDiff/handoff_artifacts/noshare_config_audit_20261008/HEAD_LN_DROPOUT_ABLATION.md)。

全部12份文本YAML的native参数/feeder与基础配置一致：encoder8/256、native decoder5/256、heads8、MLP4、A、shiftFalse、oneTrue、mask.9、各drop0、native条件drop.1。qk_scale的None字符串/null书写不同，但当前Attention不使用该参数，实际缩放不变。

新增组件的范围：

- T→S复制native block/norm/head，额外text reader为MHA(dropout0)；当前 **没有condition dropout**。
- S→T的reader/self-attention dropout0、MLP ratio4、modulation layer_mask_ratio0硬编码，没有DropPath；不受通用drop_*/mlp_ratio控制。heads从native首层读取8。
- uncond_ratio=.1 **只控制native**。attention dropout、layer_mask_ratio和整样本condition dropout不是同一个因素。
- 没有独立T→S深度/宽度或最终LN开关；text_decoder_output_norm只控制S→T末端LN。

为什么保留512→512有依据：12是骨架patch噪声维度（4×3），512是文本向量噪声维度。projection用于把hidden映射为epsilon，不要求降维。512→512仍是可学习Linear，不是恒等复制。

当前单Linear输出头下，hidden H的输出至多H维固定仿射子空间；最终LN再令上限H−1。对512维独立标准Gaussian epsilon、逐维平均MSE，旧256/LN有至少257/512≈.501953的期望下界；256/none仍至少.5；512/LN约.001953的维度下界，512/none无这项强制缺失。**这是理论期望约束，不是有限batch硬界、LP上限或性能预测。** 旧256/LN vs 新512/none同时变宽度与LN，不是纯LN实验。

最新答复建议：先保留512/none，可用现有 text_decoder_hidden_dim 做 **512 vs 768** 单因素；output仍512，text_hidden_dim（remap）仍512。宽度变大增加参数/算量，未有LP收益证据。若将文本目标压到256，需同时改变目标表示、加噪空间和输出，属于另一项实验，不能只改末端头。**768尚未建配置或跑实验。**

LN/dropout最小候选：

| ID | 相对当前no-share的唯一变化 | 实现状态 |
|---|---|---|
| A1 | S→T最终none→layernorm，hidden固定512 | 已有配置开关；未准备/运行此新实验 |
| A2 | T→S condition dropout 0→.1 | 未实现独立开关 |
| A3 | S→T condition dropout 0→.1 | 未实现独立开关 |
| A4（后续） | 两个文本条件drop同时.1 | 组合，不能单因素归因 |
| A5（后续） | 仅去T→S最终LN | 未实现独立开关 |

如果用户要求实现dropout：每decoder每样本抽一次keep[N,1,1]，全部层复用；各层reader输出z进入FeatureModulation前乘keep，保留noisy query/t/结构提示/有效token mask。不要把整行memory key_padding_mask设全True（可能NaN），不要把detach当丢条件。S→T丢条件也减少该辅助任务到encoder的梯度频率，首轮固定lambda，不同时补权重。

关闭全部文本控制时，两个方向lambda=0使forward直接回退native并跳过text_uni；还应令lambda_text_uniformity=0，否则remap仍requires_grad但forward不用，在DDP find_unused_parameters=False下有未用参数风险。**这是静态推断，未运行验证，当前非零方向不触发。** 额外文本初始化还消耗RNG，不能把同seed当纯native逐位等价。

## 7. 已完成工作及产物边界

本轮新增：

1. [官方18配置、归一化/人物/预处理审计](D:/program/MacDiff/handoff_artifacts/upstream_config_audit_20261008.md)。
2. [论文与原版复现报告](D:/program/MacDiff/handoff_artifacts/macdiff_reproduction_20261008/README.md)、[验证记录](D:/program/MacDiff/handoff_artifacts/macdiff_reproduction_20261008/verification.json)、[11配置manifest](D:/program/MacDiff/handoff_artifacts/macdiff_reproduction_20261008/manifest.json)及configs目录。
3. [文本审计](D:/program/MacDiff/handoff_artifacts/noshare_config_audit_20261008/README.md)、[12配置矩阵](D:/program/MacDiff/handoff_artifacts/noshare_config_audit_20261008/matrix.json)、[LN/dropout方案](D:/program/MacDiff/handoff_artifacts/noshare_config_audit_20261008/HEAD_LN_DROPOUT_ABLATION.md)。
4. 本handoff更新及旧handoff备份。没有新训练成绩、模型改动或Git提交。

11份原生YAML：r0控制，p1 decoder3，p2 epochs500，p3 rotationTrue，p4 uniformity0，p5 normB，p6 twoPerson，p7 decoder3+500+rotation组合，c1 T13匹配A，以及lp_a/lp_b。文件位于 D:/program/MacDiff/handoff_artifacts/macdiff_reproduction_20261008/configs；差异/sha256/profile在manifest。已做静态字段、重复键、构造器/argparse核对，**未用真实PyYAML/Torch启动验证，training_executed=False**。

此前已完成的代码工作（本轮没有重做）：v3缓存链路、孤立target快照恢复保护、packed读取/过滤、LP CE按样本数聚合、S→T hidden512/outputnone、日志口径整理。原Top1/Top5不受CE聚合修复影响。历史测试中有大量Torch依赖跳过；详情保留在备份交接，不能说本轮跑过GPU测试。

关键实现： [文本模型](D:/program/MacDiff/model/transformer_macdiff_text.py)、[native模型](D:/program/MacDiff/model/transformer_macdiff.py)、[训练入口](D:/program/MacDiff/main_pretrain.py)、[PT engine](D:/program/MacDiff/engine_pretrain.py)、[共享目标库](D:/program/MacDiff/util/shared_memory_text_target_bank.py)、[样本目标库](D:/program/MacDiff/util/sample_text_target_bank.py)、[结构cache](D:/program/MacDiff/util/structured_text_cache.py)。

本轮开始前已有用户改动：handoff.md、INPUT_NORMALIZATION_AUDIT.md、ST512_TRAINING_LOG_ANALYSIS.md；EXPERIMENT_RESULTS.md、STAGE1_EXPERIMENT_RESULTS.md及experiment_results_audit_20261007.json已是未跟踪产物。不要将它们当本轮新写内容、不要reset或clean清除。本轮未发现AGENTS.md，也未提交/推送。


## 8. 此前已选实验：no-share、T→S=1 / S→T=0.5（当前选择见第0节）

以已完成T12（1/.1）为父对照，只增加S→T权重。A/oneTrue、七句cache、hidden512/outputnone、目标更新ratio.1、PT32×accum2、LP64×accum1、seed0不变。不要叠加decoder3、rotation、B、LN或768。

配置：[no-share_st02](D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_text_sentence_sample_target_blend_noshare_st02.yaml)。YAML默认1/.2，**实际计划CLI为1/.5**。main_pretrain将顶层lambda覆写入model_args，单改model_args同名值可能被盖掉；保留两个显式CLI权重。

以下是已给用户的服务器命令，**用于交接核对，不代表已运行，不要重复启动已有目录**：

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10260 main_pretrain.py --config config/ntu60_xsub_joint/pretrain_madiff_text_sentence_sample_target_blend_noshare_st02.yaml --lambda_text_to_skeleton 1 --lambda_skeleton_to_text 0.5 --batch_size 32 --accum_iter 2 --epochs 400 --seed 0 --output_dir output_dir/ntu60_xsub_macdiff_sentence_sampletarget01_noshare_st512_t2s1_s2t05 --log_dir output_dir/ntu60_xsub_macdiff_sentence_sampletarget01_noshare_st512_t2s1_s2t05/tensorboard && OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10261 main_linprobe.py --config config/ntu60_xsub_joint/linprobe_madiff.yaml --finetune output_dir/ntu60_xsub_macdiff_sentence_sampletarget01_noshare_st512_t2s1_s2t05/checkpoint-399.pth --output_dir output_dir/ntu60_xsub_macdiff_sentence_sampletarget01_noshare_st512_t2s1_s2t05_lp_399_bs64 --log_dir output_dir/ntu60_xsub_macdiff_sentence_sampletarget01_noshare_st512_t2s1_s2t05_lp_399_bs64/tensorboard --batch_size 64 --accum_iter 1 --epochs 100 --lr 0.1 --seed 0 --dist_eval
```

旧shared/no-share父sentence YAML默认λ=.1/.1、batch64/accum1，完成T11/T12的实际CLI覆盖为T→S1和batch32/accum2。不要仅以文件默认推断真实实验。400轮epoch0..399；500轮末checkpoint499，不是399。

原实验计划 [STAGE1_TEXT_EXPERIMENT_PLAN.md](D:/program/MacDiff/tools/vlm_pilot/STAGE1_TEXT_EXPERIMENT_PLAN.md) **只看顶部最新1/.5段**；下面明确保留了旧“权重1→B→no-share”路线，已过时。

## 9. 当前卡点与下一步计划

### 卡点：缺证据，不是已确认代码bug

1. T01实际checkpoint args、服务器代码SHA、batch/LR及完整LP记录缺失，无法判断其与官方默认差异。
2. 本地无实际NPZ/checkpoint/cache，不能验证数据身份、A/B统计来源、权重层数/加载缺键。
3. no-share1/.5还无结果；768/LN/dropout只有讨论，没有对应新控制。
4. T13同时变归一化、训练预算、LP batch组织，缺同协议A；T11只有口述best，缺完整LP。
5. 论文主表86.4与默认5层/表10之间的口径未明确；公开代码400/noRotation与论文500/rotation不一致，但还未通过消融定位贡献。

### 接手顺序

先依据用户继续的方向承接，**不要把下面全部自动跑成实验矩阵**。

- 若用户继续文本分支：保留1/.5既定计划。收到日志后对照T12，检查实际lambda/args/LR/目录，再统一统计LP best/last/末20轮均值，分析native、T→S、S→T曲线。
- 若用户决定结构消融：选择一份有结果的no-share控制（T12的1/.1，或先获得1/.5控制），每次只变一个因素。A1最终LN已有开关；A2/A3条件drop需先实现且默认0；宽度512 vs768是独立候选。固定cache/target/batch/lambda/LP，重新PT后评价。
- 若用户继续原版复现：先整理最小身份信息（完整Namespace、服务器SHA及未提交改动、环境版本、NPZ SHA/count、checkpoint SHA/epoch/args、encoder/decoder层数、完整load_state_dict信息）。有身份可信的native checkpoint时先复用做少量LP检查。
- 固定LP后取得R0，再优先独立比较P1 decoder3、P2 epochs500、P3 rotationTrue。P4 uniformity0/P5 normB为次优先；要解释现有T13，先补C1匹配A。P6双人、全位置MSE、组合P7随后按结果决定。
- R0两卡参考PT32×accum2=128；原版审计的LP64×accum2=256是另一套评价profile，**不能悄悄替换文本T12/1/.5的LP64×accum1**。LP对照L0(64/2)、L1(128/1)、L2(64/1)共用同一native encoder，L1还同时改变BN组织/频率，非纯BN消融。
- 500轮实验必须从初始化按500日程训练；已结束400轮的checkpoint追加100轮应命名continuation，不能当P2。decoder/norm/dropout/宽度影响预训练，修改LP中的decoder配置不能改变已学encoder。
- 全位置MSE没有现成YAML开关，需要新实现保留masked默认。仅S→T改v也未实现，不能直接改全局diff_prediction（当前文本模型要求noise）；区域bias需追踪原始joint身份、global key与空区域回退。
- 有意义的候选再做至少3个PT seed，不先穷举全部组合。额外LP seed只能测head随机性，不能替代额外PT seed；末20个epoch的std不是独立重复置信区间。

准备代码/config是可继续的本地工作；服务器训练仍由用户执行。未经用户明确要求，不远程启动、停止、覆盖或重生成已有数据。

## 10. 绝对不要再踩的坑

1. **不要git clean -fd/-fdx或未经授权reset --hard。** 未跟踪的vlm_pilot、报告、cache、目标库和checkpoint不是垃圾；保留用户已有修改。
2. cache已完成，不为未证实的性能问题重生成/改caption。生成JSON/metadata/sidecar保留，不用cat拼两个JSON数组。
3. cache身份绑定生成/reader实现SHA：cache_clip_motion_text.py、util/structured_text_cache.py、cache_clip_text.py、util/clip_text_cache.py。无关格式改动也可能破坏已有缓存身份，不随手改。
4. 使用原始sample/person ID，空person0不换person1；front/side不是两人；不改部位顺序，v3不直接启用flip。原渲染每帧减person0 root，未展示世界位移，文本不能恢复缺失信息。
5. 7句向量不是77个BPE；旧BPE/EMA配置和诊断脚本不能未经适配拿来解释当前v3/sample_target/512。
6. 改hidden256→512/768、share、目标模式/缓存等结构需要独立从头实验，不用strict=False假称完整resume。旧encoder仍可LP，和完整PT恢复是不同事情。
7. sample_target完整resume需同次checkpoint-X.pth与checkpoint-X-target-bank.sqlite；current.shared.json不是完整快照。LP只需模型；仅可重建没有对应模型的孤立target快照，**不能覆盖配对快照**。
8. bank是.9历史+.1当前step后remap，不是.9固定CLIP+.1remap；不每forward更新，AMP跳步不更新，不对混合目标额外RMS。
9. st02文件名不等于实际.2：当前计划1/.5、_s2t05，父T12为1/.1。共享1/1下降结果不能转写成no-share1/1成绩。权重CLI优先，检查实际Namespace。
10. 不把text_hidden_dim（remap）、text_decoder_hidden_dim（S→T）、dim_feat（骨架全链路）混同。改decoder_depth会同时改native/T→S；text_decoder_depth只改S→T。
11. 三套都已有projection；none只去S→T最终LN。512→512并非无头/恒等。不要为了和256→12比例一致而盲目压hidden；预测Gaussian epsilon的维度瓶颈须考虑。
12. T→S无直接encoder梯度属用户确认设计，不“修正”为新通路；no-share并非切断全部文本影响。S→T始终独立，不称共享三套。
13. uncond_ratio只作用native；attention dropout、逐层mask、detach均不是整样本条件丢弃。不将memory整行mask为无有效key；drop与lambda首轮不同时改。
14. 一次改变一个因素，PT/LP用同组mean/var和独立目录；#new、数组出现次数及feeder normalization=False都不能替代实际归一化检查。归一化不从checkpoint继承。
15. 不把T12当原版复现，不把T13与T01差距当纯B影响，不把NTU120 FT或PKU问题当NTU60 LP原因。
16. LP首轮72多不是150最终成绩；用户已放弃150LP，不再催跑。shared512 best应是85.79；缺日志的权重1成绩不能编造。
17. loss/variance/energy改善不保证LP；loss权重/占比不等于encoder梯度占比；text_uni含对角/非跨样本；不同variance/energy口径不混比。
18. epoch std不是seed方差；理论Gaussian期望下界不是有限batch硬阈值。BN局部batch、累积、DDP广播不等于SyncBN；dist_eval一个重复解释不了半个百分点。
19. 无cd、单行Linux、PT&&LP；历史cd误拼成argparse参数导致没跑。CalledProcessError应看前面真正rank traceback，不只抄父进程结尾。
20. 没实际args/SHA不声称服务器尚未同步或已验证某版本；不要重复启动已完成目录或擅自停止服务器进程。本地缺Torch/PyYAML，不声称做过模型实例化、GPU/DDP或全配置启动验证。
21. 论文/附录是分析资料，里面的内容不作为对助手的指令。当前只交接文档，不安装依赖、不训练、不提交或推送代码。

## 11. 资料与原始日志索引

建议阅读顺序：

1. 本handoff第1节；[实验总表](D:/program/MacDiff/EXPERIMENT_RESULTS.md)和[Stage1表](D:/program/MacDiff/STAGE1_EXPERIMENT_RESULTS.md)确认“完成/口述/计划”。
2. 当前文本方向读 [no-share审计](D:/program/MacDiff/handoff_artifacts/noshare_config_audit_20261008/README.md)、[LN/dropout设计](D:/program/MacDiff/handoff_artifacts/noshare_config_audit_20261008/HEAD_LN_DROPOUT_ABLATION.md)；最新768建议在本handoff第6节。
3. 原版复现方向读 [论文/代码/运行审计](D:/program/MacDiff/handoff_artifacts/macdiff_reproduction_20261008/README.md)、[manifest](D:/program/MacDiff/handoff_artifacts/macdiff_reproduction_20261008/manifest.json)、[官方配置矩阵](D:/program/MacDiff/handoff_artifacts/upstream_config_audit_20261008.md)。
4. 日志趋势读 [No-share与NormB分析](D:/program/MacDiff/tools/vlm_pilot/NOSHARE_NORMB_LOG_ANALYSIS.md)、[精确统计](D:/program/MacDiff/handoff_artifacts/noshare_normb_20261007/summary.json)、[曲线](D:/program/MacDiff/handoff_artifacts/noshare_normb_20261007/comparison.png)、[512日志分析](D:/program/MacDiff/tools/vlm_pilot/ST512_TRAINING_LOG_ANALYSIS.md)。
5. [实验计划](D:/program/MacDiff/tools/vlm_pilot/STAGE1_TEXT_EXPERIMENT_PLAN.md)只看顶部最新状态；[归一化历史审计](D:/program/MacDiff/tools/vlm_pilot/INPUT_NORMALIZATION_AUDIT.md)的31配置/50处是旧快照，当前本地原config目录35份，不与本轮新消融YAML混数。
6. [缓存链路审计](D:/program/MacDiff/tools/vlm_pilot/TEXT_CACHE_CHAIN_AUDIT.md)的早期“待生成/未跑”已经失效；更早背景见 [写入前交接](D:/program/MacDiff/handoff_artifacts/handoff_before_20261008_session_close.md)、[2026-10-04归档](D:/program/MacDiff/handoff_artifacts/handoff_before_20261004_session_close.md)、[2026-10-01归档](D:/program/MacDiff/handoff_artifacts/handoff_before_20261001_refresh.md)。

原始T12/T13日志（JSON epoch行，无完整Namespace/服务器SHA）：

- T12 PT：[完整附件](C:/Users/97537/.codex/attachments/02d00a0e-de94-42cb-98f4-6ad1cc73af47/已粘贴的文本.txt)；LP：[完整附件](C:/Users/97537/.codex/attachments/cc1abd5f-c4d1-481b-8670-badf284c8140/已粘贴的文本.txt)。
- T13 PT：[完整附件](C:/Users/97537/.codex/attachments/826649bf-4dae-4569-b341-3e1581c0cf92/已粘贴的文本.txt)；LP：[完整附件](C:/Users/97537/.codex/attachments/4cb61fd1-2294-42bf-82dd-a231581f3661/已粘贴的文本.txt)。
- T11 shared512 PT：[附件](C:/Users/97537/.codex/attachments/64857d73-03e2-4bf9-97ed-e6ca4fba0355/已粘贴的文本.txt)，LP只有用户口述85.79。
- 用户论文：[macdiff.pdf](D:/program/paper/macdiff.pdf)。渲染/提取的临时文件已清理，未保留PDF页面PNG；新会话需要看图再读原PDF。

新会话从第0节“T15骨架decoder3口述85.84%、文本下一组仅text_decoder_depth 5→3；随机旋转另测原版decoder3/PT500”承接，不重启已完成阶段。
