# MacDiff 会话交接（更新2026-10-29）

**本文件开头的10月29日“接手更新”优先，其后的10月28日“接手摘要”保留方法定义；后面的“历史逐轮记录”仅供追溯。历史中的“当前/默认/最新/下一步”、成绩数量、端口和未完成状态可能已经过时，不能覆盖最新更新。固定CLIP版本仍未收到启动或训练结果；新SNR采样OTFM已回传T34。**

## 2026-10-29接手更新：三层最佳no-share＋新读取命令；新OTFM结果T34

**最新已授权并完成：全量文本类内/类间审查。**用户随后提供桌面 `C:/Users/97537/Desktop/ntu60_xsub_captions_by_class.txt`（31855815 bytes），明确要求检查类内和类间错误，已在本地实际解析，不再停留在“只汇总/真实文本缺失”。原始索引0–40090、40091样本、60类及每类计数一致，47608个人物条目均有七句。主要统计按person0：8662条不同global，2521个跨类精确同文组涉及32948样本（82.18%）；七句整套有2522个跨类精确相同组涉及10345样本（25.80%）。这些是文本重复/信息冲突比例，不是经GIF确认的错误率。人工核对发现A030键盘同类有下蹲/跳舞/行走，A009站起写降低身体，A036摇头写head全视频不动，A055拥抱双方写镜像舞蹈等；关键词统计只作线索，不冒充错误计数。完整60类表和具体ID/原文见[审查报告](/D:/program/MacDiff/tools/vlm_pilot/NTU60_XSUB_CAPTION_AUDIT.md)；证据统计、类表CSV、5043组global/七句跨类同文及全部ID在 `handoff_artifacts/caption_text_audit_20261029/`。一套七句行走描述366样本跨20类，包含train_623(A024)、1523(A026)、3908(A027)；一句泛泛dynamic movement原文905样本跨53类。文本质量问题已不是只看两对的猜测，但真实NPZ/GIF仍未提供，不能定位为VLM、渲染、索引错位中的唯一根因。优先回看8/29/60/623/5052/54的原始行→已有GIF→caption，不自动重生成缓存或改训练。原输入SHA与统计定义已保存，未复制原文件、未改模型/描述/cache、未新增训练编号，仍34条。

**此前步骤已完成：按类别汇总全部文本。**用户先把范围收窄为“先按类别把所有的文本汇总”，随后自行运行并回传完整汇总文件，又明确要求检查类内/类间错误；后续审查授权与结果以上段为准。分组脚本本身仍只做整理：根入口 `group_ntu60_captions_by_class.py`、测试 `tests/test_group_ntu60_captions.py`，从现有v3缓存 `vlm_pilot/ntu60_xsub_clip_sentence_cache_v3/samples.json` 读取全部描述，按同一NPZ的y_train/原sample_index分成A001–A060，写到一个 `output_dir/ntu60_xsub_captions_by_class.txt`。保留原ID、全部人物槽的global及六local，类内按ID排序；显示0起点类别ID及官方类名。只读y_train、不解压x_train、不依赖Torch/GPU、不改原文本/cache、拒绝覆盖输出，要求完整原始行覆盖。完整单行命令见[文本README](/D:/program/MacDiff/tools/vlm_pilot/README.md)。4项合成集成/完整性测试通过，Python3.8语法与CLI核验通过；本地已收到真实文本，但真实NPZ/GIF仍缺。无新增训练编号。

**第二对诊断更新，解释优先于下段第一对：**用户又回传T23 epoch399现场CLIP，seed699171498：[附件](/C:/Users/97537/.codex/attachments/6237e2cf-5cf7-4554-9fb2-af46ad459674/已粘贴的文本.txt)。train_27667/A008坐下描述屈膝/降低臀部，train_4325/A028电话描述抬右臂与头部；这次文本不同。CLIP cosine=.908754881→remap=.974887726（+.066132845），unit-L2=.427188762→.224108341（缩小47.5388%），原CLIP中更不相近的第二对经remap后反而比第一对更近。支持这对差异被remap压缩、相似度排序已改变，不能把上一对“变化小”泛化为整体几何保持，也未能确诊总体坍塌或旧KL组原因。用户问异类同描述：v2生成prompt不输入类别、仅骨架运动摘要，validator主要验证结构/人物/非空，accepted不等于动作准确；第一对敲键盘/推人均写行走仍须看原始GIF与global/local。CLIP现场与缓存吻合不验证骨架→文本正确，当前按原始sample_index读写未见直接错位证据，旧服务器产物尚未核验。详细比较、原因边界和SHA见关系方法文档“第二对实测”节。未改生成/缓存/模型或启动训练，仍34条训练结果。

**最新诊断回传：**两样本脚本已在服务器 `skeleton_vlm` 环境现场CLIP推理＋T23 epoch399在线remap正常跑完，原输出见 [附件](/C:/Users/97537/.codex/attachments/c519670f-0d42-4264-b69a-c554f84161bd/已粘贴的文本.txt)。seed258876072，train_17540/类别29(A030敲键盘)与train_23232/类别51(A052推人)，两条global描述却都讲自然行走。CLIP pair cosine=.965108963→remap=.965999933，只增.000890970；现场CLIP与缓存逐维最大差≤3.815e-6。先核查这两条原始行/GIF/global及六部位文本，不由一对样本定总体坍塌、索引错位或语义保持；固定CLIP同样无法自动补回描述漏掉的主要动作。remap同一样本前后cos .22777/.27687是跨空间坐标变化，不直接证明语义破坏；单维−7在unit-RMS下合法。细节/官方类别来源/附件SHA保存在关系方法文档“服务器实测”节。此前macdiff环境traceback缺最后异常，未确定缺包或兼容性根因；切环境后本次已完成，不自动安装依赖或修改旧Torch autocast代码，FutureWarning非运行失败。真实用户运行不冒充此前跳过的单测输出；无新训练编号，仍34条。

**本日后续新增请求已实现：**用户要求随机两个样本的CLIP/remap特征诊断。新增根入口 `diagnose_clip_remap_pair.py` 和 `tests/test_clip_remap_pair.py`；完整缓存/现场CLIP两种一行命令与输出定义见[关系方法文档的10月29日额外诊断](/D:/program/MacDiff/tools/MACDIFF_TEXT_RELATION.md)。每次OS新seed、有效person0中两个不同原始训练ID，打印train_<index>/sample_index、类别0/1起点ID、全局描述、CLIP与在线remap的首尾原值及abs、完整带符号cosine/L2及相似度变化。默认复用训练CLIP缓存；--clip-model用同一SHA本地CLIP重新编码并与缓存核对，--seed可复查。直接严格加载生产ResidualTextRemap的checkpoint权重，不构造encoder/decoder，不用EMA/平滑target/queue替代，不写诊断文件。没有remap的fixed_clip组明确拒绝。10项9通过/1本地缺Torch跳过；后续服务器现场CLIP已成功，见上方最新回传，端到端单测仍缺实际输出。本次仅额外诊断，不新增训练编号，仍34条结果。

用户本轮明确要求“最佳no-share全3层＋新读取”的完整命令，同时提供新OTFM PT/LP要求分析。已完成两项，未启动任何训练。**本节覆盖下方摘要中“最近请求”、33条计数及新SNR OTFM未回传的旧状态；固定CLIP关系教师仍是另一项已准备方案，未被本组三层文本命令改写或取消。**

- 新OTFM登记 **T34**，当前 **34条配置/成绩记录**。PT0–399/LP0–99完整连续有限；LP best **85.231683676%@71**、last85.031538116%、末20 **85.049126665±.075286871%**。
- 对旧uniform T29：best **+.600436679pp**、末20 **+.536754015pp**，100轮LP全部更高；对原生dec3 T26仍低 **.479136318pp best/.396045600pp末20**，末20全部低。SNR中间比例实测400轮平均 **90.1815%**，支持新采样生效；并非只改了名字。新旧PT采样期望不同，loss下降37.4%不是同分布质量提升；T26历史AMP与新OTFM预期noAMP边界保留。
- 最佳no-share三层＋新读取复用 `pretrain_madiff_text_global_local_decoder3.yaml`：native/T→S/S→T各3层、encoder8/256、remap2048、S→T512/none、text uni0、权重1/.1、七句v3/sample_target_blend、global/time调制＋六local attention；仍为原S→T diffusion，无固定CLIP KL/teacher/FM。
- 命令用 **10331/10332**、独立 `output_dir/ntu60_xsub_noshare_globalmod_localattn_all_dec3_h512_remap2048_textuni0_noamp_seed0_20261029/{pt,lp}`，PT400/global64/lr.001→.0005/noAMP/seed0→原第8层LP100/global128/.1→0/checkpoint399；保留原6400读出。相对noAMP T23复跑只改读取方式，相对历史T23还含精度差异。CLI覆盖旧YAML默认输出，不需新配置/重建cache；新目标库配独立目录。
- 完整一行命令保存在[当前计划顶部](/D:/program/MacDiff/tools/vlm_pilot/STAGE1_TEXT_EXPERIMENT_PLAN.md:3)；新读取文本组尚无结果，不预编号。命令需同步当前text模型/main/engine/misc、YAML和依赖；不从头覆盖已有输出或复用旧库。
- T34详细统计、原附件/SHA、完整性及比较在[Stage1结果顶部](/D:/program/MacDiff/STAGE1_EXPERIMENT_RESULTS.md:5)，[FM文档](/D:/program/MacDiff/tools/MACDIFF_FLOW_MATCHING.md:3)和总表已同步。实际服务器Namespace/源码与checkpoint身份仍缺；本次只解析日志和核对命令，未运行Torch/GPU/DDP或安装依赖。



## 接手摘要：我们正在做什么

用户在NTU60 XSub上研究MacDiff骨架自监督预训练，主要以骨架encoder冻结后的linear probe（LP）准确率评价效果。近期尝试了骨架EMA教师对比学习、后三层diffusion辅助监督，以及用文本与骨架各自的记忆库相似度分布做KL蒸馏。

**最新明确选择：使用冻结的原始CLIP文本特征作为关系教师，整个删除T→S和remap。** 当前目标为：

`L = L_native_diffusion + 0.02*L_skeleton_uniformity + 1e-4*L_InfoNCE + 0.1*KL(stopgrad(P_CLIP) || P_S)`

上一轮助手曾建议“只换KL文本来源，保留T→S”；用户随后明确“换成clip固定特征就不用T到S了”，已按后者实现独立版本。旧remap/fullkey主线A及可选coarse教师实验保留作对照，旧入口的默认配置没有被偷偷改成新方法。用户本轮最后要求**给出一行PT&&LP命令，并把完整交接写进本文件**；该要求已落实，不应接着自动启动训练或新增消融。

## 当前版本的完整定义

| 项目 | 已实现的固定CLIP版本 |
|---|---|
| 模型 / 入口 | `model.transformer_macdiff_clip_relation.Transformer` / `main_pretrain_clip_relation.py` |
| 配置 / 方法身份 | `config/ntu60_xsub_joint/pretrain_madiff_clip_relation_decoder3.yaml` / `fixed_clip_full_ema_gap_shared_queue_kl_v1` |
| 骨架主干 | 原生encoder8层、dim256；原生MacDiff diffusion decoder3层、预测epsilon、inverse_cosine schedule；A归一化/person0 |
| 学生关系query | 120帧、时间patch4×单关节，共750细token；随机mask90%后75可见token；第8层norm/GAP得到256维q |
| 骨架key | 同一crop的完整原始骨架，750token全部进入EMA encoder，eval/no_grad，GAP/L2得到256维k；EMA=.999 |
| InfoNCE | 当前完整k是正样本，历史骨架队列为负样本；温度.07、权重1e-4 |
| 固定文本 | 现有七句v3缓存中person0的**全局句向量**512维，detach/L2；不经过remap/逐样本平滑，不是六部位均值，不在线运行CLIP |
| 关系KL | 文本query对文本queue得到P_CLIP，骨架q对骨架queue得到P_S；两侧温度.1/.1，KL方向为文本→骨架，权重.1 |
| 配对记忆库 | K65536；骨架bank同一buffer供InfoNCE和KL使用，另有配对文本bank；同sample_id、槽位、pointer；排除所有历史self副本 |
| 状态更新 | 成功且完整optimizer step后配对all_gather/enqueue一次，再更新EMA一次；AMP跳步、不完整累积尾窗不提交；DDP buffers不靠自动广播同步 |
| 文本训练分支 | 没有remap、T→S/S→T decoder、文本person/position embedding、文本projection或文本可训练参数；不建逐样本文本目标库 |
| 数据读取 | 每批只读/传输全局文本；原完整缓存启动校验和mmap结构保留，不能说启动时完全不访问局部文件 |
| LP | 原`main_linprobe.py`+`linprobe_madiff.yaml`；只用在线学生encoder第8层，完整输入，25关节×256=6400维；不融合teacher/文本/后三层 |
| 日程 | PT400轮/双卡每卡32/总64、lr.001→.0005、warmup20、最后20轮floor、noAMP；LP100轮/每卡64/总128、lr.1→0、warmup0、最后10轮floor、dist_eval；seed0 |
| 输出与端口 | 本文命令中的独立`cliprel...20261028/{pt,lp}`目录；PT10329、LP10330；LP读取本组checkpoint-399.pth |

512维文本与256维骨架无须投影到同维：比较的是各自在相同K个样本上的关系概率。CLIP缓存里的RMS缩放不改变其L2归一化后的余弦结构。文本queue仍随样本FIFO入队，“固定特征”不表示停止更新队列。

## 已完成的代码、记录与验证

新增的4个文件：

- [新模型](/D:/program/MacDiff/model/transformer_macdiff_clip_relation.py)：直接继承原生MacDiff，不先构造旧文本模型再把权重置零。
- [新入口](/D:/program/MacDiff/main_pretrain_clip_relation.py)：固定CLIP协议与参数验证；拒绝非零T→S/S→T权重、跨方法resume和覆盖已有输出。
- [新配置](/D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_clip_relation_decoder3.yaml)：原生decoder3，现有v3缓存，独立输出目录。
- [新增测试](/D:/program/MacDiff/tests/test_clip_relation.py)：11项，覆盖全局reader/engine路由、固定教师、精确目标、队列配对/时序、恢复与LP兼容等。

修改的公共代码只有：

- [engine_pretrain.py](/D:/program/MacDiff/engine_pretrain.py)：仅对`text_global_only=True`模型选择全局reader；其他文本模型保持旧路径。
- [person_text_cache.py](/D:/program/MacDiff/util/person_text_cache.py)：新增`get_global_batch`，不读取局部token数组。

服务器还须有此前当前版本的`model/transformer_macdiff_text_relation.py`（KL/gather）、`model/transformer_macdiff_text_relation_teacher.py`（完整key encoder/InfoNCE）、`main_pretrain_text_relation.py`（parser/resume定义）及其公共依赖。新模型只复用其定义，不构造旧remap/文本decoder。**不能只复制新YAML就运行，也不能省略新engine/cache修改。**

验证结果（均来自上一轮实际执行；本轮交接不声称重新跑过GPU测试）：

- 新增11项：**8通过、3因本地没有PyTorch跳过**。
- 相关9模块共127项：**72通过、55缺Torch跳过**。
- 通过的NumPy适配检查执行了生产forward和队列代码，验证固定全局CLIP来源、单次学生/完整key/原生decoder调用和精确loss；它们不能替代真实autograd。
- 实际PT/LP parser配静态YAML子集核对了参数、总batch、端口目录及checkpoint-399衔接；本地没有实际PyYAML/Torch训练。
- 新文件和公共修改通过Python3.8语法检查；diff空白检查通过。
- 方法文档和实验计划已同步最新定义：[关系方法](/D:/program/MacDiff/tools/MACDIFF_TEXT_RELATION.md)、[Stage1计划](/D:/program/MacDiff/tools/vlm_pilot/STAGE1_TEXT_EXPERIMENT_PLAN.md)。
- 当前共有**33条配置/成绩记录**，不是33次完整独立PT；固定CLIP新组没有成绩，不预分配T34。结果表未因新方法而虚增实验结果。

## 当前卡在哪里

没有已定位但未修复的实现错误。尚未完成的是**真实服务器环境验证和实验结果**：

1. 本地缺Torch/PyYAML/timm，也没有此次实际NPZ/cache/checkpoint；真实参数清单、反向传播、优化器→EMA更新、checkpoint→LP的3项Torch测试未执行。
2. 未验证真实双卡DDP/NCCL；CPU/Torch单测通过也不能直接声称双卡正确。
3. 用户没有回传新组Namespace、源码身份或日志，**不知道它是否启动、跑到哪、能否提升LP**；助手没有服务器SSH，也没有启动或停止任务。
4. 旧remap组只收到epoch38/39片段，缺完整PT/LP成绩。文本teacher高熵不等于已经证实remap坍塌；固定CLIP能否改善关系结构仍待验证。

## 下一步接手顺序

1. 先确认用户当前诉求；若用户只问解释或发日志，直接处理，不自行发起训练/新矩阵。
2. 若准备运行新组，由用户在服务器同步上述新4文件、公共engine/cache修改及已有依赖；保持现有CLIP缓存和数据。
3. 在服务器现有macdiff环境先执行下方测试。若失败，按真实traceback修复；不要为了绕过错误删resume保护、恢复T→S/remap或改AMP/温度/损失定义。
4. 由用户执行本文单行PT&&LP命令。已有log/checkpoint时按协议完整resume或另选未使用目录，不覆盖已有实验。
5. 收到早期日志后先核对实际model/version、fixed_clip、CL/KL项、队列填充、自ID过滤、teacher entropy/温度及有限值。冷队列阶段KL=0、InfoNCE没有负样本时=0正常；总loss删除了旧T→S项，不能拿旧total=.0487要求新组相同。
6. 如果固定CLIP分布仍平，优先对**相同样本和锚点**检查余弦跨度、top-k概率质量/熵；教师温度.1/.05/.04是后续候选，**尚未授权自动改当前实验**。不靠增大KL权重制造目标区分度。
7. 收到完整LP后再统计best/last/末20均值与std，核对PT/LP实际身份与既有结果重复后决定编号；不得把文本来源变化的效果与删除T→S的效果分别归因。
8. OPD/固定预训练encoder蒸馏等待用户参考文献；大patch真投影、motion差分/MCAM、局部KL、LP融合都不是本次已授权实现的下一步。

## 最近结果与做出本次选择的依据

| 实验/证据 | 当前可靠记录与边界 |
|---|---|
| 原生dec3 T26 | LP best85.710819994%；历史AMP等协议与新noAMP有差异 |
| 文本dec3 T23 | best86.280931625%；文本dec5 T21 best86.220281456%；不能把不同配置best当随机重复 |
| AA0/CL1e-4 T30、T31 | 用户确认两次各自从头PT→LP的LP best86.03/85.87，均值85.95%；口述，无逐epoch日志 |
| AA0/CL1e-3 T32 | PT400完整，LP附件只有0–89；收到的best85.286268831%@72，不能写最终LP100成绩 |
| 后三层diffusion T33 | 用户口述LP85.71%；best/last、实际版本/seed/完整日程未核验；与T26两位小数相同不等于精确持平 |
| 旧remap关系主线A epoch38/39 | total及分项正常；teacher H=11.0899，K约65534.864的最大熵11.0903376，接近均匀；CL确实存在 |
| 固定CLIP新组 | 已实现/给命令，尚无运行或LP结果 |

旧remap片段的epoch38均值：native=.0207、T→S=.0259、uni=.0232、KL=.0062、CL=9.7559；加权合计.04865959≈打印total .0487。teacher entropy fraction实际约.9999605，四位小数显示1.0000；目标总体区分度弱，但不能确诊特征坍塌或唯一归因温度。K约65536时InfoNCE均匀参考约11.09，9.76不能按绝对值判失败。原附件与完整诊断见关系方法文档；不把旧组片段误当固定CLIP运行日志。

## 踩过的坑：本次接手绝对不要再踩

1. **最新方法必须无T→S/remap。**不拿下方历史“默认A保留T→S”覆盖用户最后选择；不把只设lambda=0却仍构造/计算的旧文本模型当新实现。
2. **CL一直保留。**旧v1基类辅助函数返回0不代表v2/new实际没有InfoNCE；检查实际入口、子类覆盖及总loss。新固定CLIP配置CL权重仍1e-4，不擅自升1e-3。
3. **固定CLIP queue不能与旧remap queue混用。**不跨方法resume，不用strict=False冒充完整PT恢复；新的模型名/参数身份/输出目录保护必须保留。原有sample_target_blend恢复仍须与模型checkpoint对应的目标库快照，新fixed_clip不需要目标库。
4. **样本与人物必须正确配对。**原始dataset ID而非batch行号；骨架/文本同槽位同ID，过滤全部self副本；空person0不偷偷换person1；front/side不是两个人。
5. **成功步时序不能改错。**队列存完整EMA key而非masked q；同批旧key提交后再EMA；跳步/累积尾窗不更新；两个队列不能独立打乱；不凭DDP buffer广播假设一致。
6. **global的含义别混淆。**本次文本取一个全局句向量，六部位仅存在原缓存中；七句是全局+六部位，不是77个BPE。骨架KL只用可见token GAP256；LP6400是另一读出，不能临时加投影/缺失关节补齐/局部KL。
7. **dec3/dec5始终指原生MacDiff decoder深度。**不要按S→T文本decoder深度分组，不把辅助方法混入原生/普通文本四组均值。
8. **PT与LP分目录。**历史T14同目录覆盖过早期PT checkpoint；400轮用399、500轮用499。LP必须读取本组权重，不能复制旧目录/端口后漏改finetune。
9. **归一化不会从checkpoint自动继承。**当前PT/LP用A的固定XYZ mean/var；feeder normalization=False并不关闭模型内部归一化。缓存RMS是逐向量尺度；不是每patch XYZ分别标准化。测过的新数据统计没有自动应用。
10. **不要误判日志。**括号前是窗口中位数，括号内是epoch累计均值；不相加分项中位数验证total。mask*pred/target的整行0通常是可见位置被打印mask置零。小KL/低diffusion loss/有梯度不等于LP会提高，标量loss占比不等于梯度占比。
11. **成绩证据要分层。**口述best/last未知不补猜；附件只90轮不写完整100轮；std末20轮不是跨seed方差；不同batch/AMP/floor/decoder不是随机重复。T20同一次PT的重复LP不新增独立PT编号。
12. **后三层版本已经是完全共享norm/decoder。**v3第6/7/8层都用同一个原生LayerNorm及decoder，分别监督；LP只第8层。不再加两层辅助LN、三套decoder或LP融合，也不把85.71口述直接判成方法无效。
13. **粗patch可选B只是均值合并近似。**3个相邻细时间embedding按同关节取均值后随机mask70%，不是ACA²Net独立大patch投影原样实现；motion差分和MCAM是更后一级候选，不能混进主线。
14. **不要重生成已有描述/cache或改缓存身份文件。**缓存来源脚本SHA与manifest绑定；无关格式改动也可能破坏身份。保留metadata/sidecar/JSON结构。caption/render做过居中，不能假定文本能还原未呈现世界位移。
15. **模型比较别靠印象。**原生MacDiff PT也利用局部可见tokens，并非只训练256全局向量；MAMP官方linprobe为GAP256，不是本仓库linprobe2的6400。
16. **本地通过不等于真实Torch/GPU/DDP通过。**不要安装依赖或启动训练来补做未授权运行；服务器历史Python3.8/Torch1.8.1，避免引入仅新版语法/API。现有老测试有从tests顶层导入的情况，整合运行必要时把tests加入sys.path，别误判为生产回归。
17. **保留工作区与用户产物。**不git clean/reset、删移artifact/cache/checkpoint、不commit/push、不清理未跟踪文件；本轮新文件还未跟踪，要同步它们。已有log/checkpoint/目标库拒绝覆盖是保护，不绕过。
18. **不用旧计划自动续跑。**用户自行执行Linux命令；助手不推断服务器状态，不擅自启动/停止/监控训练，不创建额外日志副本或handoff备份，不因会话结束创建定时任务/新会话。

## 环境与工作习惯

本地工作区`D:/program/MacDiff`，Windows PowerShell；用户在Linux仓库（历史路径`/home/user9/public3/swr/MacDiff`）自行执行命令，助手没有SSH。历史训练环境conda macdiff / Python3.8 / Torch1.8.1+cu111 / 双RTX4090，当前实际环境以用户回传为准。数据为`../data/MAMP/ntu/NTU60_XSub.npz`，train40091/test16487；缓存为`vlm_pilot/ntu60_xsub_clip_sentence_cache_v3`。

本地可用Python为`C:/Users/97537/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe`（此前检查Python3.12、有NumPy、无Torch/PyYAML/timm），用-B避免新增pycache；不靠系统Python猜依赖。先看git status，保留当前未提交/未跟踪工作；本轮没有commit/push。禁止根据历史HEAD或旧“未提交”文字推断最新Git状态。

用户偏好中文、直接动手、少反复确认；运行命令必须**单行Linux，不加cd，PT与LP用&&连接**。本地文件链接用绝对路径。优先更新已有文档，不为每次审查新建备份或复制附件。开始做工具操作前简要说明，长任务保持进展沟通。

## 最新运行命令（从头PT → 第8层LP）

先在服务器现有环境检查：

```bash
python -m unittest tests.test_clip_relation tests.test_text_relation tests.test_text_relation_teacher tests.test_pretrain_accumulation_policy -v
```

以下为用户本轮索要的完整一行；双卡0,1，端口10329/10330，未加cd，不带多余的两个文本diffusion权重0参数。需要先同步文件，数据/cache保持已有路径。**助手未执行这条训练命令。**

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10329 main_pretrain_clip_relation.py --config config/ntu60_xsub_joint/pretrain_madiff_clip_relation_decoder3.yaml --relation_bank_size 65536 --lambda_contrastive 0.0001 --ema_momentum 0.999 --contrast_temperature 0.07 --lambda_relation 0.1 --relation_temperature_text 0.1 --relation_temperature_skeleton 0.1 --disable_amp --batch_size 32 --accum_iter 1 --epochs 400 --warmup_epochs 20 --min_lr_epochs 20 --lr 0.001 --min_lr 0.0005 --mask_ratio 0.9 --motion_aware_tau -1 --seed 0 --output_dir output_dir/ntu60_xsub_cliprel_fullkey_q65536_cl1e4_kl01_gap256_dec3_noamp_seed0_20261028/pt --log_dir output_dir/ntu60_xsub_cliprel_fullkey_q65536_cl1e4_kl01_gap256_dec3_noamp_seed0_20261028/pt/tensorboard && OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10330 main_linprobe.py --config config/ntu60_xsub_joint/linprobe_madiff.yaml --finetune output_dir/ntu60_xsub_cliprel_fullkey_q65536_cl1e4_kl01_gap256_dec3_noamp_seed0_20261028/pt/checkpoint-399.pth --batch_size 64 --accum_iter 1 --epochs 100 --warmup_epochs 0 --min_lr_epochs 10 --lr 0.1 --min_lr 0 --seed 0 --dist_eval --output_dir output_dir/ntu60_xsub_cliprel_fullkey_q65536_cl1e4_kl01_gap256_dec3_noamp_seed0_20261028/lp --log_dir output_dir/ntu60_xsub_cliprel_fullkey_q65536_cl1e4_kl01_gap256_dec3_noamp_seed0_20261028/lp/tensorboard
```

主要记录入口：[关系方法与命令](/D:/program/MacDiff/tools/MACDIFF_TEXT_RELATION.md)、[实验结果总表](/D:/program/MacDiff/EXPERIMENT_RESULTS.md)、[Stage1结果](/D:/program/MacDiff/STAGE1_EXPERIMENT_RESULTS.md)、[Stage1计划](/D:/program/MacDiff/tools/vlm_pilot/STAGE1_TEXT_EXPERIMENT_PLAN.md)、[多层监督](/D:/program/MacDiff/tools/MACDIFF_MULTILAYER_DIFFUSION.md)。

## 历史逐轮记录（以下旧状态不能覆盖以上接手摘要）

### 最新实现：2026-10-28固定CLIP关系教师，删除T→S/remap

用户明确“换成clip固定特征就不用T到S了”。已实现独立模型`model/transformer_macdiff_clip_relation.py`、入口`main_pretrain_clip_relation.py`、配置`config/ntu60_xsub_joint/pretrain_madiff_clip_relation_decoder3.yaml`、测试`tests/test_clip_relation.py`；完整方法/同步文件/命令见[文档顶部](/D:/program/MacDiff/tools/MACDIFF_TEXT_RELATION.md:3)。这取代此前仅换KL来源却保留T→S的建议；旧remap主线A与配置作为控制保留。新组未启动、无新成绩ID，仍33条记录。

新身份`fixed_clip_full_ema_gap_shared_queue_kl_v1`，直接继承原生MacDiff：S→S原生decoder3+uni.02、完整750token EMA key（.999）的InfoNCE（.07/1e-4）、固定原始CLIP全局句向量作teacher的KL（T/S温度.1/.1、权重.1）。学生10%可见75token/第8层GAP256；文本为现有v3缓存person0全局512维/L2，未经过remap/平滑，不是六部位平均。骨架一个65536 queue供CL与KL共用，配对CLIP queue同ID/槽位/指针；成功完整步提交后更新EMA，self副本全部排除。不存在remap、T→S/S→T decoder或文本可训练模块；非零文本diffusion权重直接拒绝。原生可训练参数应保持一致（实际Torch参数清单测试待服务器）。LP仍第8层完整骨架/6400维/原入口。

公共改动仅`util/person_text_cache.py`新增get_global_batch，以及`engine_pretrain.py`对text_global_only模型选择全局reader；旧文本路径保持。新组fixed_clip模式不建立逐样本文本目标库，每步不读/传输局部句向量或更新remap；仍保留原完整缓存启动校验/mmap结构。新模型复用旧关系KL/gather和完整key encoder/InfoNCE函数，因此服务器相关旧文件须当前版本，但不会构造其文本模块。已有旧主线A运行日志仍是真实旧方案证据，不改名为固定CLIP结果；新组同时换teacher并去掉T→S，不能仅归因于文本来源。

新11项8通过/3无Torch跳过；相关9模块127项72通过/55跳过。生产forward NumPy、全局cache/engine路由、精确loss、配对queue wrap/跳步/EMA时序、fixed不建目标库、旧方法resume隔离、实际PT/LP parser配静态YAML子集、checkpoint路径与Python3.8语法已核对。真实梯度/参数清单/EMA优化/checkpoint→LP的3项待服务器，未声称GPU/DDP通过，未安装依赖或启动训练。

新PT400/总64/.001→.0005/warm20/floor20/noAMP/seed0→LP100/总128/.1→0/warm0/floor10/dist_eval；端口10329/10330，目录`output_dir/ntu60_xsub_cliprel_fullkey_q65536_cl1e4_kl01_gap256_dec3_noamp_seed0_20261028/{pt,lp}`，LP读本组checkpoint-399.pth。新入口拒绝旧remap resume/旧queue及改变方法定义，命令不再带lambda_text_to_skeleton或lambda_skeleton_to_text的0参数。原始CLIP结构是否更有信息仍需熵/近邻诊断与LP验证，未保证收益。

### 最新日志诊断：2026-10-28主线A epoch38/39片段，teacher关系近乎均匀

用户提供KL/fullkey主线A的控制台片段，79行、17条进度记录，epoch38 step160–625及epoch39 step0/40/80/120，含epoch38最终Averaged stats；原附件路径/SHA及详细计算见[方法文档顶部](/D:/program/MacDiff/tools/MACDIFF_TEXT_RELATION.md:3)。已见CL/KL实际运行指标，不再说没有任何KL训练日志。尚无KL LP成绩或完整PT/实际Namespace/服务器代码SHA，不新增成绩ID，不推断片段之后的进度。当前记录仍33条（含T33多层口述）。未改训练代码/温度/权重或启停训练。

打印数值均有限、total配方在四位小数精度内吻合：epoch38均值S→S .0207、T→S .0259、uni .0232*.02、KL .0062*.1、CL9.7559*1e-4，合计.04865959 vs total .0487。InfoNCE确实存在，加权约.000976；K65536的均匀logits参考约11.09，CL9.76不能按绝对值判异常。满队列/valid_fraction1、非self锚点约65534.864；少于65536是self历史副本过滤。pred/target打印整行0来自原生mask乘法，不是坍塌证据。括号前是窗口中位数，括号内是epoch累计均值，不相加分项中位数验证total。

**核心疑点：**teacher_entropy11.0899几乎等于ln(65534.864)=11.0903376；entropy_fraction实际约.9999605，四位小数显示1.0000。teacher相对均匀分布的KL约.000438 nats（均值/打印精度近似），关系区分度很弱；student_entropy11.0840也较高。KL=.0062很小不证明语义蒸馏成功，目标若近均匀可能主要拉平骨架关系；不能据熵确诊remap/骨架坍塌，也不能归因唯一温度。建议后续先对同样本/锚点比较CLIP→在线remap→平滑target→文本queue的方差/余弦分布和时滞，再离线扫teacher温度.1/.05/.04/保持student.1，看熵与top-k质量；不先盲目增大KL权重。ReSSL锐化可作参考，非当前跨模态方法已验证处方。

此段epoch38约187.33s/.2992s每步，allocated12796.9MiB/reserved13426MiB；target读取.58s/更新2.05s占约1.4%。epoch39首步加载2.17s导致高ETA，后续回归约.30s。只记录这段实测，不外推整次训练时间/显存趋势，不把日志当本地Torch单测或真实DDP一致性验证。

### 最新回传：2026-10-28后三层diffusion口述85.71%，T33；KL主线A保留InfoNCE

用户口述“后三层diffusion的跑出来85.71”，登记T33，当前33条配置/成绩记录。先记LP分数85.71%，best/last、对应epoch、原始PT/LP日志和末20统计未提供；预期为此前共享norm/decoder的v3，但实际Namespace/服务器版本/seed/checkpoint加载仍待日志。与原生dec3 T26 best85.710819994%按两位小数相同，当前没有可见提升证据；不把口述两位小数当精确持平，不在best/last及AMP/noAMP差异未对齐时断言无效。不自动调权重/融合LP/重跑，T33不混入原生/文本四组均值。详见[Stage1最新](/D:/program/MacDiff/STAGE1_EXPERIMENT_RESULTS.md:5)、[总表](/D:/program/MacDiff/EXPERIMENT_RESULTS.md:5)、[多层文档顶部](/D:/program/MacDiff/tools/MACDIFF_MULTILAYER_DIFFUSION.md:3)。旧“多层无成绩/32条/不预登记T33”仅为历史状态。

用户另问“KL为什么没有对比学习”，已回传其使用的主线A命令：main_pretrain_text_relation_teacher.py + pretrain_madiff_text_relation_teacher_decoder3.yaml、10325/10326、lambda_contrastive=.0001、lambda_skeleton_to_text=0。当前v2明确保留CL：teacher模型覆盖skeleton_auxiliary_loss，shared_queue_info_nce得到raw CL，乘1e-4作为auxiliary，由基类forward总loss加上；base v1的zero辅助函数不是v2实际调用。日志为train_loss_contrastive/train_loss_contrastive_weighted/contrast_positive_cosine。用户随后明确“我以为没有，是我理解错了”，无需排查日志故障或改权重。只确认命令与当前源码定义，未拿到服务器日志/源码，不猜运行进度，也不登记KL成绩。

**2026-10-27最新解释更正：**用户追问一级为什么不像原论文实现、默认A现在是什么。已重新核对代码/原文：当前B只实现“3个细patch embedding均值合并＋随机70%”，是助手为保留EMA同形参数自行选的近似；不能当成用户所指的独立大patch投影已实现。真正12帧投影（36→256）与4帧学生投影（12→256）不能直接逐项EMA，需另定义其学习/初始化/更新；这不是EMA原则上不能搭配大patch。当前模型/配置未因本次解释更改。默认A仍是S→S diffusion/native dec3＋完整750token EMA key的全局InfoNCE＋独立T→S dec3训练remap＋全局文本关系KL；query75token GAP256，两个配对65536queue，LP仅学生第8层6400读出。不要把B近似的实现状态误写成原论文大patch已复现，也不要擅自把默认A换掉。

### 最新消融层级：2026-10-27默认完整key；可选大patch＋随机70%；运动与MCAM是其二级消融

用户最初提出按AA论文教师路先大patch再mask70%，获得更多时间覆盖的75token；异步回答选只改大patch+随机70%，随后再次明确“运动差分和motion compass mask作为额外消融的额外消融”。最终层级已记录在[方法文档顶部](/D:/program/MacDiff/tools/MACDIFF_TEXT_RELATION.md:3)：主方案A仍完整骨架key不mask（v2）；一级B仅大时间尺度+随机70%mask；二级B+M/B+C/B+M+C分别考察运动输入/MCAM/组合，二级本次仅列候选，未实施，不把运动或MCAM混入B。默认入口及默认YAML仍A，未启动任何训练、无新结果或实验编号。

已实现B：[可选配置](/D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_text_relation_teacher_coarse70_decoder3.yaml)、[专用测试](/D:/program/MacDiff/tests/test_text_relation_teacher_coarse.py)，复用并扩展v2模型/入口。版本coarse3_mean_random70_ema_gap_shared_queue_kl_v3，显式teacher_view=coarse_mask、teacher_temporal_group=3、teacher_mask_ratio=.7；full要求v2/group1/mask0，新view字段不冗余加入旧full的model_args，保留旧full恢复。跨full/coarse恢复被严格args校验拒绝，即使参数形状相同也不能混用历史key队列。

关键适配：原文ACA²Net是90帧、学生patch3/.9、教师独立patch9/.7（运动输入+MCAM），两者75token；本地PDF第4页图2/式5–7、第6页IV-B已用pypdf与Poppler渲染核对。MacDiff120帧/学生patch4时，对应教师12帧尺度。但独立12帧投影与学生4帧投影不满足直接EMA形状对应，因此B先用相同EMA细patch投影得到[B,30,25,D]，加时空位置后按**同关节相邻3个时间embedding取均值**，成为[B,10,25,D]；250粗token随机保留75，再进入EMA Transformer/norm/GAP/L2。这样参数/EMA完全同形、无新增投影权重；相当于受细patch权重约束的12帧投影，**不是ACA²Net独立MotionEmbed原样复现**，平均可能损失细节，不能称一定信息更优。75粗token覆盖225个细patch，约为学生75细patch的3倍，但少于完整教师750细patch；教师/学生mask独立，不保证视图包含关系。

其余保持v2：在线75token GAP256、共享InfoNCE/KL骨架queue及配对文本queue65536、EMA.999、CL.07/1e-4、KL两侧.1/权重.1、T→S1/七句/remap2048、uni.02、A/person0/noAMP，encoder8/native decoder3/T→S decoder3。PT400/总64/floor5e-4/seed0→原第8层LP100/总128/floor0/6400读出。可选B端口10327/10328、目录output_dir/ntu60_xsub_textrel_coarse3mean_m70_q65536_cl1e4_kl01_gap256_dec3_noamp_seed0_20261027/{pt,lp}，LP读本组checkpoint399；A仍10325/10326和原fullkey目录。

新增7项为5通过/2缺Torch跳过；B+v2+v1+累积37项为28通过/9跳过；相关整合回归152项为79通过/73跳过。生产粗key forward的NumPy检查验证先时间分组/平均位置、再250→75随机采样、每batch无重复位置及关节不混合；CLI/完整恢复隔离、默认full不变、实际parser静态配置/命令/75对75形状/Python3.8语法与启动print也已核对。2项真实Torch75/75反向、EMA同形更新/checkpoint待服务器执行；未运行GPU/DDP，也未安装依赖。服务器同步更新的v2模型/入口/测试及新B配置/测试，先python -m unittest tests.test_text_relation_teacher_coarse tests.test_text_relation_teacher tests.test_text_relation tests.test_pretrain_accumulation_policy -v。论文只读核对，未修改源PDF。

### 最新用户定义与实现：2026-10-27完整EMA key的同一骨架queue，同时用于InfoNCE和全局关系KL（v2）

用户澄清原生MacDiff接完整骨架key encoder做teacher对比学习，直接复用其骨架queue，并建立对应文本queue；KL仍global，query为masked输入10%可见token的全局池化。已实现[新模型](/D:/program/MacDiff/model/transformer_macdiff_text_relation_teacher.py)、[新入口](/D:/program/MacDiff/main_pretrain_text_relation_teacher.py)、[新配置](/D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_text_relation_teacher_decoder3.yaml)、[新测试](/D:/program/MacDiff/tests/test_text_relation_teacher.py)，完整定义和PT&&LP命令见[方法文档顶部v2](/D:/program/MacDiff/tools/MACDIFF_TEXT_RELATION.md:3)。旧v1文件保留控制组；本组未启动、无新结果/编号，仍32条配置/成绩记录。

query=normalize(第8层75个可见token GAP)，256维有梯度；key=normalize(EMA encoder对同一crop完整750token的第8层GAP)，256维no_grad/eval；文本query/key为现有remap逐样本平滑global目标512维、detach。InfoNCE以当前完整key为正样本、历史完整骨架queue为负样本；KL的P_S也使用**同一个relation_skeleton_keys buffer**，P_T使用配对文本queue，按同sample_id列对应。没有另一个contrast_queue或独立骨架关系queue。两个损失排除历史self全部副本，InfoNCE当前key正对保留；入库(k_i,y_i,id)，不是masked q_i。

两个queue共用IDs/环形指针/覆盖时序，默认K65536与旧AA0对比队列相同（旧v1是4096）；仅用relation_bank_size一个容量参数。EMA=.999、CL温度.07/权重1e-4；KL温度两侧.1/权重.1，T→S权重1、skeleton uni.02。无AA对齐/适配器，无S→T decoder、local KL、额外投影头、OTFM或多层监督。用户说256/6400可再考虑；按其明确的可见token平均定义当前用GAP256，relation_feature_mode=gap；6400需要另定义按关节池化及缺失关节处理，目前未实现，不能宣称可切换。LP仍第8层完整输入的25关节拼接6400维及原head。

成功完整optimizer步后，一次all_gather并配对提交当前forward已有完整keys，再更新一次EMA；AMP跳步/空提交/不完整累积尾窗不更新。key只前向一次、online masked encoder只前向一次；EMA在DDP同步后由initialize_pretraining_state复制online，再由full resume恢复保存值。当前main的broadcast_buffers=False；queue由成功步gather一致，不依赖buffer广播。EMA参数冻结，teacher_updates及EMA/两个queue/IDs/指针全部进入checkpoint。保留原文本目标库同checkpoint快照要求。

模型新身份model.transformer_macdiff_text_relation_teacher.Transformer，relation_teacher_version=full_ema_gap_shared_queue_global_kl_v2；入口绑定完整model_args及协议，拒绝v1/纯teacher等跨方法resume，新目录已有log/checkpoint拒绝覆盖。新输出output_dir/ntu60_xsub_textrel_fullkey_q65536_cl1e4_kl01_gap256_dec3_noamp_seed0_20261027/{pt,lp}，端口10325/10326；encoder8/native decoder3/T→S decoder3、A/person0、PT400/总64/floor5e-4/noAMP/seed0→LP100/总128/floor0/checkpoint399。队列FP32特征静态大小64+128=192MiB，另约.5MiB IDs，不等于实测峰值显存，EMA完整前向也有额外开销。

本轮基类model/transformer_macdiff_text_relation.py只加缺省masked-query key和zero辅助loss的两个扩展方法，旧v1默认/状态保持；main_pretrain_text_relation.py增加可复用的指定模型身份/参数解析，仍保留旧缺省；tests/test_text_relation.py适配扩展并保留旧生产forward检查。engine本轮未再改，但服务器仍需此前当前engine/main/util及文本缓存支持文件。同步新4文件及修改的基类模型/入口/测试。

v2新增12项为9通过/3缺Torch跳过；v2+v1+累积30项为23通过/7跳过；相关回归145项为74通过/71跳过。通过含生产forward NumPy适配、可见query/完整clean key/共享CL-KL队列/精确loss/配对payload、全token key前向、self排除、成功步EMA时序、恢复隔离和实际参数解析静态检查。真实Torch小模型5 vs50 token、真实反向/EMA优化与队列、完整checkpoint/旧LP兼容3项待服务器执行，未验证GPU/NCCL/DDP。服务器先python -m unittest tests.test_text_relation_teacher tests.test_text_relation tests.test_pretrain_accumulation_policy -v。未安装依赖或启动训练。

读出讨论已核对：原生MacDiff预训练保留可见local tokens并以GAP填masked条件，并非只训练一个256向量；本仓库LP linprobe2用6400。MAMP官方LP配置protocol=linprobe，对人物/时间/关节全部平均得256；其关节拼接用于finetune MLP。原先建议六部位local KL已被用户本次明确“KL还是全局”取代，不擅自接入。

### 最新实现：2026-10-27文本/骨架配对记忆库关系KL，替换S→T diffusion

用户提出remap后文本与骨架各建memory bank，以当前样本对记忆库的相似度分布做KL，让骨架模仿文本结构；并在异步澄清中明确选择“保留T→S，只替换S→T”。已完成[独立模型](/D:/program/MacDiff/model/transformer_macdiff_text_relation.py)、[入口](/D:/program/MacDiff/main_pretrain_text_relation.py)、[配置](/D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_text_relation_decoder3.yaml)、[测试](/D:/program/MacDiff/tests/test_text_relation.py)及[方法/完整一行PT&&LP命令](/D:/program/MacDiff/tools/MACDIFF_TEXT_RELATION.md)。旧文本/OTFM/多层实现保持独立，本组未启动、无结果、不预分配T33；当前仍32条配置/成绩记录。

保留原生S→S、T→S（独立原生结构decoder3、concat_memory、七句缓存、remap隐藏2048）、skeleton uni.02；不构造S→T TextNoiseDecoder，不加投影头。新关系分支首轮global-only：骨架query为第8层norm后可见token GAP256，文本teacher为既有sample_target_blend的平滑global512；各自L2归一化后，分别与自身模态库求cos/temperature，再按相同参照样本列softmax。L_relation=KL(stopgrad(P_T)||P_S)，只更新当前骨架query/encoder；文本与两个历史库均detach，remap继续由T→S训练。两边维度无需相同。L=L_S2S+1*L_T2S+.1*L_relation+.02*uni，text uni0。lambda_relation初始.1、两侧temperature.1只是首轮设置，不能与旧S→T MSE权重.1等同。

关系库为4096对FIFO，两个特征buffer共享IDs/指针，按sample_id排除两侧当前样本的全部副本；不是全数据集每ID一槽。空库/少于2个非self参照的行暂不提供KL。特征取本次forward的detach值，成功完整optimizer步后才入库；AMP跳步和不完整累积尾窗不提交。DDP按rank gather成对更新、支持不同有效行数。model checkpoint保存两个库/IDs/指针；原逐样本文本平滑目标库仍用旧shared-memory/SQLite快照，其首次访问固定CLIP fallback与.1值更新定义保留。无encoder EMA或额外骨架前向。

公共engine本次只增加supports_relation_memory开关：传sample_indices、收集result[4]及成功步提交；旧路径缺省False。新模型/入口需要同步**当前engine_pretrain.py**。新增日志loss_relation/loss_relation_weighted、bank_fill/有效参照数/有效行比例、teacher/student熵和teacher_entropy_fraction；后者接近1表示教师关系较平，KL低不等于LP提升。方法类似SEED/ReSSL关系分布蒸馏，但不是其完整复现，也不是OPD。

本组encoder8/native decoder3/T→S decoder3、A/person0、mask.9、seed0、PT400/总64/lr.001/floor5e-4/noAMP→原第8层LP100/总128/lr.1/floor0，读本组checkpoint399。新目录output_dir/ntu60_xsub_textrel_kl01_k4096_t01_dec3_noamp_seed0_20261027/{pt,lp}，端口10323/10324。完整resume绑定模型/参数/温度/K/数据和缓存路径/日程/精度，要求同checkpoint的文本目标库快照；拒绝旧S→T等方法冒充续训，新PT已有log/checkpoint拒绝覆盖。不保证RNG逐位复现，未核实服务器代码/数据内容。

本地新分支13项为9通过/4缺Torch跳过；与既有text/consistency/FM/multilayer/teacher/累积回归共133项为65通过/68跳过。teacher旧测试使用tests目录的顶层导入，整合runner需将tests加sys.path，或按原发现模式单独运行，勿误记为生产回归错误。通过含生产forward NumPy适配、KL方向/同序/跨维度/self排除、FIFO/成功步提交与恢复校验；4项新真实Torch梯度/完整模型/队列与LP/checkpoint/mock gather待服务器执行。未验证真实GPU/NCCL/DDP，也未安装依赖。服务器先运行python -m unittest tests.test_text_relation tests.test_pretrain_accumulation_policy -v。

### 最新实现：2026-10-27 OTFM匹配Inverse Cosine信噪比的采样对照

用户在讨论Inverse Cosine与旧OTFM差异后要求“改一个试试”，已实现只改时间采样的单因素对照，独立于下方后三层diffusion。沿用原FM模型/入口，新增[配置](/D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_flow_matching_inverse_cosine.yaml)，详细定义/检查/完整一行PT&&LP命令见[FM方法文档顶部](/D:/program/MacDiff/tools/MACDIFF_FLOW_MATCHING.md:3)。旧配置和默认仍uniform，T29=84.631246997%仍属于旧均匀采样。

新model_args/CLI为flow_time_sampling=inverse_cosine_snr_v1；从生产inverse_cosine(1)的1000步beta（含末步.999 clipping）累乘得到alpha_bar，均匀抽索引，a=sqrt(alpha_bar)、b=sqrt(1-alpha_bar)、t=a/[b+(1-sigma_min)*a]，沿用x_t=t*x+[1-(1-sigma_min)*t]*noise和target=x-(1-sigma_min)*noise。decoder时间仍999*t，不是参考索引；不乘映射导数或补偿采样概率。系数SNR分布逐点匹配，中间[.1,10]占90.2%，旧uniform约51.95%；FP32最大相对SNR误差约1.45e-6。映射仅改变采样，输入幅度/速度损失仍不同于原生epsilon diffusion，不承诺LP追平。

原encoder8/decoder3、A/person0、mask.9、uni.02、sigma_min1e-4、seed0、noAMP；PT400/总64/lr.001/floor5e-4，LP100/总128/lr.1/floor0/原第8层分类头，读本组checkpoint399。新目录output_dir/ntu60_xsub_otfm_invcos_snr_dec3_noamp_seed0_20261027/{pt,lp}，端口10321/10322。采样表为非持久化buffer，无新增参数/state_dict键；日志增加train_time_mean和train_snr_middle_fraction。完整resume拒绝跨采样，旧v2缺字段只视为uniform；新PT目录拒绝覆盖已有log/checkpoint。

本地FM+累积测试27项，21通过/6缺Torch跳过；通过含生产forward NumPy执行、逐点SNR、CLI/恢复兼容与隔离/目录保护。6项真实Torch（含新采样的参数一致性/反向/设备移动）待服务器执行，未运行GPU/DDP。当前仍32条配置/成绩记录；新方法无成绩、不预分配T33，未启动训练或推断服务器状态。服务器同步FM模型、FM入口、新YAML和tests/test_flow_matching.py，沿用已有公共支持文件。

### 最新结果与实现：2026-10-27 T32为AA0/CL1e-3；后三层监督→第8层LP已准备

用户要求分析记录本次teacher AA0/CL1e-3，并完成此前选定“encoder第6/7/8层各接diffusion分别监督，LP仅第8层”。用户此前说固定预训练encoder蒸馏有参考文献，等准备实现时再讨论；该蒸馏未授权实施、不与新多层监督混接。

**T32新结果：**PT0–399完整，LP附件只0–89共90轮；均连续、有限。当前best85.286268831%@72、epoch89=85.219553633%、70–89 mean85.122816606±.080791743%。缺90–99，不写最终LP100/80–99均值。total=native+.001*CL，最大残差6.784075159e-10，AA0组合核验。相对T26同0–89 best−.424551163pp、70–89均值−.329936927pp，末20共同窗口每轮都低；对旧CL1e-4两次口述best平均85.95%暂低.663731169pp。native PT末20较T26仅高.474134%，提高CL没有转成已收到LP收益；标量CL占total约2.53%不等于梯度份额，不确诊冲突/坍塌。实际Namespace/服务器版本/checkpoint加载仍缺；新组给定noAMP、历史原生AMP的边界保留。完整数值/原附件/SHA见[Stage1顶部](/D:/program/MacDiff/STAGE1_EXPERIMENT_RESULTS.md:5)。现在32条配置/成绩记录；旧31/29及CL1e-3待结果为历史状态，T30/T31仍保留旧CL1e-4身份。

**新实现：**[专用模型](/D:/program/MacDiff/model/transformer_macdiff_multilayer.py)、[根目录入口](/D:/program/MacDiff/main_pretrain_multilayer.py)、[专用配置](/D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_multilayer.yaml)、[测试](/D:/program/MacDiff/tests/test_multilayer_diffusion.py)、[说明及完整一行PT&&LP命令](/D:/program/MacDiff/tools/MACDIFF_MULTILAYER_DIFFUSION.md)。公共main/engine/native/text/LP代码未因本次修改。

encoder8/256，用户进一步指定后三层6/7/8共享同一原生decoder3/256，分别前向计算loss；v3中三层统一调用原self.norm（含共享weight/bias），再进入同一个decoder，没有额外norm/decoder参数，state_dict键/形状和参数量与原生完全一致；中间特征不detach。三路共享同一t/epsilon/x_t/mask/ids_restore和一次condition dropout。默认L=L8+.1*(L6+L7)/2+.02*uni8，辅助两路各.05；CLI --lambda_aux_diffusion可覆盖并保存。默认PT关闭AMP，decoder/loss FP32；仅纯骨架diffusion，无teacher/text/FM/OSE。LP继续原main_linprobe.py/linprobe_madiff.yaml，只读取第8层与原6400维分类头，不融合。

PT400/两卡每卡32/accum1/总64/seed0/lr.001/min_lr5e-4/warmup20/floor20；LP100/每卡64/总128/lr.1/min_lr0/warmup0/floor10，读本组checkpoint399。端口10319/10320，输出output_dir/ntu60_xsub_multilayer678_sharednormdec3_aux01_lp8_noamp_seed0_20261027/{pt,lp}。新PT目录已有log/checkpoint会拒绝覆盖，完整resume核验方法/参数/数据配置/日程/精度等。代码命令就绪但助手未启动训练、不预登记T33，不推断服务器同步/运行状态。

本地多层+累积测试23项：16通过、7缺Torch跳过。通过包含生产forward的NumPy适配执行、三路共享状态/权重组合、CLI/恢复/覆盖保护、累积逻辑；7项真实Torch梯度/共享参数与加权梯度累积/原生零权重兼容/LP加载测试已写，因本机无Torch未执行，不能说CUDA/DDP已验证。服务器可先运行python -m unittest tests.test_multilayer_diffusion tests.test_pretrain_accumulation_policy -v，再运行文档内PT&&LP；没有安装依赖、生成权重或启停进程。

### 最新结果与统计：2026-10-26 teacher AA0两次从头完成，按原生decoder归四组

用户本次明确86.03%/85.87%为**两次从头PT→LP的LP best**，新增T30/T31，均值85.95%。不是同一PT重复LP；没有新日志，last/末20/实际seed/Namespace/checkpoint身份不补填。当前31条配置/成绩记录，后面的29条和“AA0未回传”均为历史状态。原生dec5新floor重复、新文本5层/读取配置仍未收到新成绩，不能将AA0两次误当其结果。

**dec3/dec5仅指原生MacDiff的decoder_depth。** T15 3/3/5归文本dec3；T24 5/5/3归文本dec5，不能因text_decoder_depth=3归错。全部历史已记录LP分数均值：原生dec3 85.6992%（6条，含T19复用PT的新LP配置）；原生dec5 85.5926%（2条，含T13归一化B）；文本dec3 85.8552%（4条，含T25一致性）；文本dec5约85.3934%（13条，含全部旧BPE/fixed/EMA/shared/no-share变体）。T05约数，T15/T16/T19旧口述best/last待核。这些均值不是同配置独立PT重复；T20相同重复LP只计一次，T01原生约85.86%深度未确认单列，teacher/FM不混入四组。

近期七句no-share子集：文本dec3=T15/T16/T23，均值85.9936%；文本dec5=T12/T21/T24，均值85.9797%，仍含协议差异。当前AA0优先对原生dec3 T22/T26（PT400/总64/floor5e-4），best均值85.586487154%，因此AA0平均+.363512847pp；对T26单次两次分别+.319180006/+.159180006pp。AA0关闭AA、保留InfoNCE，预期noAMP，历史原生为AMP；只记小幅正向观察，稳定方法净收益尚未确认。原生5/A T20=86.001940813%，与AA0均值仅差−.051940813pp，但不是同深度方法对照。

完整四组逐条表、末20平均及来源已放[Stage1结果顶部](/D:/program/MacDiff/STAGE1_EXPERIMENT_RESULTS.md:5)，[总记录](/D:/program/MacDiff/EXPERIMENT_RESULTS.md:5)和[teacher文档](/D:/program/MacDiff/tools/MACDIFF_SKELETON_TEACHER.md:5)同步。用户当前要统计和判断收益；本次未改训练代码/配置、未启动新训练，不自动扩大实验矩阵。

### 最新存档：2026-10-23 S→T监督改进候选已统一记录

用户明确要求把此前S→T改进讨论记录起来，已整理到[文本计划：S→T监督改进统一入口](/D:/program/MacDiff/tools/vlm_pilot/STAGE1_TEXT_EXPERIMENT_PLAN.md:3)。三个方向分别为：A骨架条件进入冻结T→S decoder、用GT骨架epsilon反传encoder；B骨架GAP经轻量256→512头回归固定配对CLIP特征（cosine）；C骨架条件的自回归描述生成。A/B已讨论机制，C只被用户提问后即切换查看结果，尚未展开，不能当成已选方案。统一记录现状、公式、梯度归属、原CLIP/remap条件空间接口及待定项；新读取全5层只改T→S读取，仍用现有S→T diffusion，不包含A/B/C。

当前只是讨论存档，没有选定监督/冻结时机/模型/权重或启动实现、训练；仍29条成绩。自回归若继续，需现存captions.json的原文和所选tokenizer的ID，当前七句512维缓存不等于词表token，也不把六部位当时间序列。完整冻结decoder/语言模型都须保留对骨架条件的梯度；去噪/回归/生成loss下降不保证LP收益。单独teacher AA0（CL=1e-4）命令仍见[teacher文档](/D:/program/MacDiff/tools/MACDIFF_SKELETON_TEACHER.md:11)，不混称为文本语义对齐。

### 最新配置补充：2026-10-23新读取全5层已准备，未启动

用户要求“那再给我一个新读取配置的”，按最近全5层设置新建[新读取全5层YAML](/D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_text_global_local_decoder5.yaml)。对旧全5层配置仅改文本读取为global_mod_local_attention_v1和output/log路径；encoder8、三套decoder5/remap2048/S→T512/none/text uni0/骨架uni.02、七句v3、权重1/.1、A/person0/noAMP均保持，不接讨论中的冻结交互/直接回归/自回归。旧配置保留，新缓存无需重建，新目标库/独立PT-LP目录。

PT400/两卡每卡32/accum1/总64/lr.001/min_lr5e-4/warmup20/floor20；LP100/两卡每卡64/总128/lr.1/min_lr0/warmup0/floor10/seed0，读取本次checkpoint399。命令端口10313/10314，输出`output_dir/ntu60_xsub_noshare_globalmod_localattn_all_dec5_h512_remap2048_textuni0_noamp_seed0_20261023/{pt,lp}`；完整PT&&LP命令见[文本计划顶部](/D:/program/MacDiff/tools/vlm_pilot/STAGE1_TEXT_EXPERIMENT_PLAN.md:3)。服务器同步新YAML及当前文本模型/main/engine/misc。已核对真实CLI解析和静态配置差异，未真实PyYAML/Torch/GPU执行。没有启动、未预登记T30，仍29条成绩；本地输出路径不存在不保证服务器未使用，服务器已有同目录则另换新路径，不删除/复用旧库。

### 最新讨论：2026-10-23通过冻结文本条件骨架decoder优化encoder，尚未实现

用户提出：用两路各自的diffusion，交互时骨架特征输入冻结T→S decoder预测噪声，优化骨架encoder，替换当前S→T文本去噪辅助监督。当前按`骨架条件→骨架去噪`及`文本条件→骨架去噪`两路理解；交互为`MSE_masked(D_T_frozen(x_t,t;P_S(E_S(masked_aug_x))),epsilon)`，交互梯度更新encoder/适配器，不更新D_T。冻结参数须保留输入autograd；不能no_grad整个decoder。256→512及global/六local接口需适配，完整冻结包括reader/head，不等于旧native decoder-body冻结方案。

“只在交互时冻结、文本本路继续训练”与“同run先训练后固定快照”是不同协议；后者仅是便于隔离变量的候选建议，用户未选冻结时机/权重/接口，也未要求实现或使用外部预训练teacher。去噪可用条件不保证与配对文本严格语义对齐，decoder可能忽略条件或接受偏离文本分布的条件，不能承诺LP收益。详细讨论已记[现有文本计划顶部](/D:/program/MacDiff/tools/vlm_pilot/STAGE1_TEXT_EXPERIMENT_PLAN.md:3)，未改训练代码/配置或新增实验ID；仍29条。

随后用户追问能否骨架回归文本实现语义对齐。已讨论固定CLIP配对目标＋骨架GAP/Linear256→512的归一化cosine回归：`L_sem=1-cos(P(E_S),stopgrad(CLIP))`，更新encoder/骨架头，native保留；属于此前候选细化，未实现/选权重，不把旧fixed-CLIP文本去噪成绩当成本直接回归已验证。对齐的是投影后表示，字幕区分度/crop/person0和适配器容量仍影响LP。若同时接冻结D_T，原CLIP语义预测须经配套冻结R_T转成decoder条件空间并保留输入梯度，不能把原CLIP与remap后的条件混用。对比/关系蒸馏仅为其他选项，不自动堆入新loss。

### 最新结果补充：2026-10-23原生OTFM已回传，T29，低于扩散dec3

用户本次要求“看handoff。otfm跑出来了，怎么跟扩散差这么多”。已解析PT400/LP100两份完整连续有限日志，登记T29，总计29条配置/成绩记录。预期为此前原生OT条件FM v2/sigma1e-4/encoder8/decoder3/A/person0/noAMP命令；实际Namespace、服务器源码/checkpoint SHA及LP加载输出仍缺。不要继续说FM没有结果，也不要拿文本T23或native5 T20当纯方法对照。

T29 LP best **84.631246997%@90**、last84.588791878、80–99 mean **84.512372650±.055134978%**。相对原生dec3 T26（85.710819994/85.445172265）best−1.079572997pp/后期均值−.932799616pp，100轮LP全部低于T26；相对T22 best−.830907316pp/末20−.879427463pp。PT/LP逐epoch LR与T26完全相同、LP参数384060；不是只差峰值或少跑轮数。T29预期noAMP、T22/T26为此前AMP协议，已安排的同精度native3仍缺结果，不能把全部差异归为FM。

PT末20total=.088636326182、velocity=.087960651531、uniformity=.033783732637；total=velocity+.02*uni最大残差1.315e-9，持续下降后平台，未见发散。速度MSE不与epsilon loss直接比。路径导数/符号/999仅编码经本地复核未见明显错误；OT原论文生成性质不保证冻结encoder LP。实际训练/测试CE均比T26低，不能只凭acc低认定过拟合或坍塌。

**具体可检验差异：**用生产inverse_cosine函数重算，系数SNR .1–10占原扩散90.2%，uniform t OT只51.9546%，两端各约24%（sigma1e-4、100万t中点积分；系数SNR不含真实骨架方差，阈值仅示意分布差异）。同一OT路径有`epsilon_hat=x_t-t*v_hat`及`epsilon_error²=t²*velocity_error²`，均匀velocity MSE还改变时间权重。两点均是候选机制，尚非掉点因果证明；不能直接断言decoder绕过encoder。优先看已选同精度native3，并用现有checkpoint固定batch/t/noise/mask做时间分桶条件置零/打乱损失与encoder梯度诊断；若继续调整，优先仅改采样分布的单因素对照，未实现或启动新实验，不接文本/teacher、不改归一化。

完整数值、原附件路径/SHA及数学说明已写入[Stage1顶部T29](/D:/program/MacDiff/STAGE1_EXPERIMENT_RESULTS.md:5)、[总表](/D:/program/MacDiff/EXPERIMENT_RESULTS.md:5)、[FM文档](/D:/program/MacDiff/tools/MACDIFF_FLOW_MATCHING.md:5)。本地本次重跑FM+累积21项，16通过/5缺Torch跳过；不能把用户训练回传当成这5项单测输出。仅更新这四份既有文档，未改训练代码/配置、未复制日志、未启停训练。以下旧“尚无FM成绩/28条”等状态以本补充为准。

### 0.00 本次会话结束交接：任务全景与最新全5层命令（2026-10-23）

#### 我们在做什么，用户最后要求什么

主目标是提升**原生MacDiff骨架encoder的NTU60 XSub joint-only线性评估LP**。已尝试文本no-share辅助预训练、纯骨架EMA teacher；目前新增原生单路OT条件Flow Matching基线，并审查输入归一化。PT是预训练，LP是冻结encoder、只训练分类头的线性评估。训练由用户在Linux服务器执行，助手负责本地代码、审计、解释日志和给完整命令。

用户最后要求两件事：①给“noshare全5层”的命令；②写本交接。当前为全5层准备的是**保留最近T23设置，只把native/T→S/S→T三套decoder均设为5层**：encoder仍8层/256维，remap512→2048→512，S→T宽512/末端none，text uniformity=0，骨架uniformity=.02，T→S=1/S→T=.1，七句v3/sample_target_blend，A归一化/person0，PT关闭AMP、floor5e-4。文本读取仍为旧`concat_memory`；没有混入新global/local读法、FM或teacher。这是结合用户最近设置给出的运行定义，尚无实测结果，**不是历史T21精确复跑**；T21当时remap512、text uni.02、PT floor1e-5且开AMP。

新增的唯一必要配置为[全5层YAML](/D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_text_sentence_sample_target_blend_noshare_decoder5.yaml)。旧5层YAML和T23三层YAML均不改。`decoder_depth=5`同时控制native与no-share T→S，`text_decoder_depth=5`控制S→T；不是把encoder也改成5层。服务器需同步此YAML，以及已讨论的当前`main_pretrain.py`、`util/misc.py`、`model/transformer_macdiff_text.py`、`engine_pretrain.py`等对应代码。新配置/命令均明确关闭AMP，不用`--enable_amp False`。端口10311/10312只是本次选择，不能保证服务器端口未占用。

从服务器仓库根执行的一行PT400 **&&** LP100命令（两张卡、PT每卡32/总64、LP每卡64/总128、accum均1、seed0；新输出目录/新目标库；不自动resume）：

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10311 main_pretrain.py --config config/ntu60_xsub_joint/pretrain_madiff_text_sentence_sample_target_blend_noshare_decoder5.yaml --disable_amp --lambda_text_to_skeleton 1 --lambda_skeleton_to_text 0.1 --batch_size 32 --accum_iter 1 --epochs 400 --warmup_epochs 20 --min_lr_epochs 20 --lr 0.001 --min_lr 0.0005 --mask_ratio 0.9 --motion_aware_tau -1 --seed 0 --output_dir output_dir/ntu60_xsub_noshare_all_dec5_h512_remap2048_textuni0_noamp_seed0_20261023/pt --log_dir output_dir/ntu60_xsub_noshare_all_dec5_h512_remap2048_textuni0_noamp_seed0_20261023/pt/tensorboard && OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10312 main_linprobe.py --config config/ntu60_xsub_joint/linprobe_madiff.yaml --finetune output_dir/ntu60_xsub_noshare_all_dec5_h512_remap2048_textuni0_noamp_seed0_20261023/pt/checkpoint-399.pth --batch_size 64 --accum_iter 1 --epochs 100 --warmup_epochs 0 --min_lr_epochs 10 --lr 0.1 --min_lr 0 --seed 0 --dist_eval --output_dir output_dir/ntu60_xsub_noshare_all_dec5_h512_remap2048_textuni0_noamp_seed0_20261023/lp --log_dir output_dir/ntu60_xsub_noshare_all_dec5_h512_remap2048_textuni0_noamp_seed0_20261023/lp/tensorboard
```

本次已用AST提取两入口的真实`get_args_parser`解析整条命令，核对轮数/batch/accum/seed/LR与floor/AMP/权重、两端口、PT/LP分目录及checkpoint399。新配置对T23三层配置的静态字段差异仅为两处decoder深度、显式旧读取模式/关闭AMP及输出路径；没有Torch/PyYAML/GPU执行。新输出目录本地不存在不等于服务器不存在；若服务器曾用该目录，换新目录，不删除已有目标库。

#### 已完成的工作：实现、结果、诊断分开

| 分支/事项 | 已完成什么 | 现在的证据边界与入口 |
|---|---|---|
| 历史no-share文本 | 七句缓存、在线remap、独立T→S与S→T、逐样本目标库及LP；T21/T23等已记录 | 加上本日OTFM T29，共29条配置/成绩记录，含口述和不完整LP，不是29次独立完整PT；见[Stage1结果](/D:/program/MacDiff/STAGE1_EXPERIMENT_RESULTS.md)和[总记录](/D:/program/MacDiff/EXPERIMENT_RESULTS.md) |
| 新文本读取 | `global_mod_local_attention_v1`：global独立用于每层调制，六个local仅作cross-attention K/V；保留七句缓存 | 独立[global/local配置](/D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_text_global_local_decoder3.yaml)，未接新监督，未收到PT/LP结果；旧默认仍concat_memory |
| 2025/2026监督方案调研 | 已阅读/核实GFP、MLS³L、AMR、TranCLR相关材料，列出关系蒸馏、运动加权、固定运动摘要预测候选 | [方案与来源](/D:/program/MacDiff/tools/vlm_pilot/STAGE1_TEXT_EXPERIMENT_PLAN.md)；候选未实现，不能自动加进loss或称已验证/首创 |
| 原生OT条件FM | 独立entry/model/config/tests和[方法文档/命令](/D:/program/MacDiff/tools/MACDIFF_FLOW_MATCHING.md)；连续时间速度预测替换epsilon预测，原encoder/decoder和mask结构保留 | 当前`conditional_ot_noise_to_data_v2`、sigma_min=1e-4；用户已明确选择原论文噪声→对应骨架OT条件路径，不重新询问、不改成batch OT。T29完整PT400/LP100，best84.631246997%，见顶部 |
| 归一化统计脚本 | [脚本](/D:/program/MacDiff/tools/compute_skeleton_normalization.py)流式只读x_train，比较person0/all、raw/model输入、self_shift和空人物帧；已收到六协议控制台并核算、记录 | [完整审查](/D:/program/MacDiff/tools/vlm_pilot/INPUT_NORMALIZATION_AUDIT.md)；未收到服务器JSON/CSV/实际NPZ，只是归一化诊断，不算新LP成绩，不新增T29；未改任何既有归一化参数 |
| EMA teacher | 当前v3学生单侧adapter、AA1、CL1e-4，成功优化步才更新teacher/queue，公共PT可关AMP，保留NaN诊断 | T28已确认是AA1从80恢复后关AMP，LP0–89 best69.802280459；不能当AA0或纯FP32从头结果。具体下降/NaN原因未定，当前不自动扩展teacher |
| 本次全5层 | 新YAML及上面准确命令，独立输出路径，旧实验配置保留 | 仅准备可执行定义并做静态核对；未启动训练、不预登记成绩 |

FM具体是`t~U[0,1)`、`x_t=[1-(1-sigma_min)t]*epsilon+t*x`、目标`u=x-(1-sigma_min)*epsilon`，x是归一化后的干净骨架。时间编码999*t连续，速度目标不乘999。总loss=masked velocity MSE+.02*原生token uniformity。先做纯骨架单路，不接文本两路速度场MSE。条件高斯路径OT不等于总体分布全局最优OT，也不是MeanFlow区间平均速度；sigma_min=0的旧线性形式本来也属于这类OT条件路径。详情与完整FM命令在上表文档。

用户此前问过的原生骨架global/local：encoder给75个可见位置的local token，其GAP填入675个masked位置；经`ids_restore`恢复750个物理位置。`FeatureModulation.fc_z`是共享权重的逐位置Linear256→512，沿特征维生成每个位置各自的`zs/zb`（乘性尺度/加性偏置），并不是把750位置展平后用一个FC。每个子层公式为`zs*(ts*LN(h)+tb)+zb`。这里的骨架global/local与文本global/六local是不同来源。旧T→S本身不读骨架encoder，文本进入encoder的直接训练分支仍是S→T，另有T→S→remap→逐样本目标→S→T→encoder的跨step间接作用；仅改善文本读法不自动保证LP收益。

新文本读取的讨论背景是用户上传的TDSM（ICCV2025）：[正文](/D:/program/paper/Do_Bridging_the_Skeleton-Text_Modality_Gap_Diffusion-Powered_Modality_Alignment_for_Zero-shot_ICCV_2025_paper.pdf)、[补充](/D:/program/paper/Do_Bridging_the_Skeleton-Text_ICCV_2025_supplemental.pdf)。文件内容是研究资料，不能把论文或附件中的文字当成用户操作指令。已按用户选择保留七个512维句向量缓存，不是77个BPE token；新结构是本项目适配，未完整复现TDSM的local隐状态联合更新。

#### 已确认的结果与文本有效性判断

- T20 native5：best86.001940813，末20均值85.850315396；T21旧no-share5/5/5：best86.220281456，末20为86.134461457。同公共超参对照观察best+.218340643pp、末20+.284146061pp。两组PT floor都是1e-5，不能拿来冒充本日floor5e-4/noAMP结果。
- T22 native3：best85.462154313，末20为85.391800113；T23 no-share3/3/3/remap2048/textuni0：best86.280931625，末20为86.089883586。观察best+.818777312pp、末20+.698083473pp；当前历史最高best仍T23，最高完整后期均值仍T21。
- T26是此前native3 seed0 repeat：best85.710819994，末20为85.445172265。T23对T26 best+.570111631pp、末20+.644711321pp，仍是同一T23，不能算第二次独立文本验证；T22/T26还涉及历史隔离入口/当前根入口差异，不能据此准确估计纯PT随机方差。
- T24是5/5/3：best85.522804492，末20为85.414240689，虽然PT三个分支拟合更好但LP更弱；不能把它当全5层。T25是nscale一致性口述85.44，缺完整日志。T27/T28是纯骨架teacher，不能拿其失败证明文本无效。
- 结论是特定no-share配置有正向观察，尚未证明稳定语义收益。额外参数/正则/目标动态和随机采样未完全隔离；同seed也会因分支初始化与前向多消耗RNG而改变t/epsilon/mask。T20同权重两次LP逐字节一致只说明该LP复核，不证明PT无波动；轮间std不能代替独立训练方差。固定CLIP、条件打乱和geometry诊断已有，别再包装成未做过的新实验。

#### 最新归一化解释：用户刚问过，下一会话不要再混淆

现有固定输入归一化与新统计的公式完全相同：`(XYZ-mean)/sqrt(var)`，区别是所用常数。A为mean`[-.0058,-.1333,-.0246]`、var`[.0206,.0805,.0218]`。本次NTU60 XSub训练集经当前干净PT变换、person0、包含实际输入位置得到mean`[-.003405887,-.220585,-.04651074]`、var`[.04780319,.159242,.0464151]`。套旧A后std约1.523/1.406/1.459。它说明旧A未把这份实际输入变为严格0均值/单位方差，**不证明旧A是bug或新统计必提高LP**。

person0/person1是数组最后一维预留的两个“人物存储位置”。person0为第一个位置，person1可能是真实第二人、全零填充，或NTU120本地预处理复制的第一个人。`one_person=True`指每个样本只取person0，**不是过滤掉双人动作**；双人握手仍参加PT，只保留第一人的骨架。原生Diffusion/FM PT取person0；LP将两槽分别用同一encoder编码，再平均人物特征。750是每个人120帧/时间patch4×25关节的位置数，与人物槽不是一回事。

全零人物槽也计入mean/var会改变统计。例如person0坐标[1,3]均值2/方差1，与空person1合起来[1,3,0,0]均值1/方差1.5。保留模型实际输入全部位置与排除整个人物帧都是脚本的对照口径；不删除单个合法0坐标、不因person0空就换人。PKUv2旧H几乎匹配两槽含空位，但不匹配当前person0，且本地PKUv2 PT=H/LP=A值得优先核对；旧H出处还没被证明。PKUv1/v2有66/24条非前缀时间padding，风险已记录未修。

若未来决定换统计，只用对应训练split拟合一次，PT/LP训练/LP测试复用同一套，不能把test混进估计，也不能因LP读两槽就另换all统计。已有checkpoint LP保持其PT输入变换。mean/var不是state_dict buffers，LP YAML会独立决定，不能以“加载PT权重”假设自动一致；feeder normalization=False也不代表模型内A关闭。LN归一化hidden token通道、LP BN归一化分类头输入，都不能当输入XYZ变换的等价替代。

#### 当前卡在哪里、接下来如何推进

没有需要用户确认才能写代码的权限卡点。本次命令与交接已准备；剩余限制是**缺本地Torch环境和服务器实物/新结果**。本地bundled Python有NumPy，无Torch/PyYAML/timm，因此FM、文本新读取及teacher的GPU/DDP/autograd/显存/LP载入未在本地实跑。FM及累积回归此前16通过/5跳过；文本相关59项17通过/42跳过；归一化11项9通过/2跳过。通过的是实际函数NumPy执行、CLI/AST与静态检查，不能称真实Torch训练通过。不要擅自安装依赖。

下一会话优先按用户新消息推进：

1. 本日全5层若收到结果，核对实际Namespace、三套5层、remap2048/textuni0、concat_memory、disable_amp、PT/LP路径和checkpoint399，再追加结果；仅给命令不等于用户已执行。
2. FM已收到T29完整PT400/LP100，best84.631246997%。先按顶部解释时间/噪声分布和监督权重差异、结合已安排同精度native3与条件诊断；原命令10309/10310、`ntu60_xsub_native_otfm_dec3_noamp_seed0_20261022`仅作身份追溯，不重复安排首轮、不接文本监督。
3. 10月21日已选的native3/native5/no-share T23三组noAMP repeat和独立teacher AA0，均未收到可确认身份的结果。命令见[Stage1计划](/D:/program/MacDiff/tools/vlm_pilot/STAGE1_TEXT_EXPERIMENT_PLAN.md)与[teacher文档](/D:/program/MacDiff/tools/MACDIFF_SKELETON_TEACHER.md)，不猜是否运行，不再问用户是否要这些命令；T26不是这次noAMP repeat。
4. 归一化若继续分析，先用用户服务器已经生成的JSON/CSV取得完整精度和raw视图，别默认重扫六数据集。当前未授权换归一化配置或重训新统计组；先讨论后实施。新文本监督也仍待讨论。
5. 只在用户切回时继续caption：NTU60旧缓存保留，其他划分“GIF生成完”不等于描述完成；独立XSet入口及SHA保护见下方0.9和[README](/D:/program/MacDiff/tools/vlm_pilot/README.md)。

#### 绝对不要再踩的坑与工作区保护

- **身份准确：**旧T21、新全5层、本日命令、T23三层、T24混合5/5/3、新global/local、FM和teacher是不同定义。已有29条记录，T29为本日OTFM完整结果；后续未回传不预编号T30。不能从当前YAML反推历史真实实参。
- **输出路径：**PT/LP分目录，LP读本次PT的checkpoint-399；不要用旧成绩目录跑新方案。T14曾因共目录让LP覆盖早期PT文件。no-share完整resume的checkpoint与对应`checkpoint-X-target-bank.sqlite`要配套；不得删保护、删库、强覆旧库，LP不需目标库。
- **AMP与resume：**公共`--disable_amp`同时关PT autocast/scaler，默认旧native/no-share仍开；LP默认autocast关但历史scaler保留。T28中途关AMP不算从头FP32。不能跨teacher v1/v2/v3、AA定义、文本读取模式或FM身份完整resume；RNG未保存，不承诺逐bit续跑。
- **实验推断：**loss降低、梯度有通路、condition被使用都不等于LP提高；97% loss数值不是97%梯度。v2双侧FC可共同趋零已验证数学反例，但没有证明T27实物坍塌；NaN可由FP16路径触发不代表已定位T28实际失败算子。不要自行加clip/LN/AA权重来掩盖问题。
- **数据/随机性：**train-only统计且PT/LP一致；人物槽/真实人数分清，person0不换person1；同seed不保证跨架构相同噪声，不能用单次结果或末20std宣称跨seed显著。用户不跑seed1/2，不自动扩展矩阵。
- **操作习惯：**中文解释，Linux单行PT&&LP，不加cd；global batch=每卡batch×GPU数×accum，别擅改accum。用户自行安排GPU，两张卡被多组共用不代表可以并跑。
- **保护成果：**工作区很多修改和未跟踪文件（FM/teacher/归一化/文本/caption）都是成果，不git reset/clean、不删checkpoint/cache/官方隔离副本、不commit/push、不安装包、不远程启停训练、不复制日志新建重复归档。只更新必要现有文档；本次新增一个运行所需YAML。

环境：本地`D:/program/MacDiff`、Windows PowerShell；服务器历史路径`/home/user9/public3/swr/MacDiff`，数据`../data/MAMP/ntu/NTU60_XSub.npz`。历史Python3.8/Torch1.8.1+cu111/双4090，当前服务器版本未核实。bundled Python为`C:/Users/97537/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe`。没有SSH或本次实物NPZ/checkpoint，不装依赖、不猜GPU状态。原始归一化附件路径/SHA和六协议向量在紧接下方；更细的历史实现、成绩、命令、论文与caption索引继续保留，避免新会话从零审计。

### 最新补充：六协议归一化统计已回传，仅分析记录（2026-10-22）

用户明确统计只能来自train集，并要求“区分one person和非one person，写脚本算然后跟现有对比”。已新增`tools/compute_skeleton_normalization.py`与`tests/test_skeleton_normalization.py`，完整NTU60/六协议命令及输出口径补入[既有归一化审查顶部](/D:/program/MacDiff/tools/vlm_pilot/INPUT_NORMALIZATION_AUDIT.md)。默认仅流式读取NPZ `x_train`，输出person0/all × raw_valid/model_input × self_shift关/开 × 保留全部位置/排除空人物帧统计，保留合法0坐标、不把person0换为person1；模型视图复用原生crop/resize/配置flip与rotation，排除source_aug与diffusion/FM噪声。默认每样本一次seed0采样，可提高draws，不能声称精确遍历所有增强。

报告写入独立`output_dir/normalization_audit/...`，JSON+CSV对比当前PT及同目录LP/FT，列出normalized_mean/std、方差比例与输入变换差异；self_shiftTrue时有效mean为0。归一化不自动写回配置；旧checkpoint的LP仍沿用其PT统计，不因LP两个人物而换成all统计。NPZ按块流式读取、不用假的NPZ mmap；支持训练专用NPY mmap，Fortran NPZ明确拒绝要求先提取。完整扫描与max-samples预览区分，零样本/空槽/有效帧规则差异有计数。

本地11项检查9通过/2缺PyYAML或Torch跳过，涵盖真实NPZ流式读、NPY、训练集隔离、人物/空槽/中心化、数值与完整CLI输出；Python3.8语法通过。随后用户回传六协议服务器控制台：各组摘要完整，scanned=40091/37646/63026/54468/18840/5299，person0/all计数分别严格为N*3000/N*6000；全零样本、整段person0为空、有效帧计数规则差异均0。未收到JSON/CSV、NPZ或服务器源码/配置SHA，不能称本地独立重算或所有服务器单测通过。

当前PT的model_input/person0/model_positions：NTU60 XSub measured mean=`[-.003405887,-.220585,-.04651074]`、var=`[.04780319,.159242,.0464151]`；用旧A后mean=`[.01668058,-.3076392,-.1483983]`、std=`[1.523333,1.406471,1.459155]`，方差/A约2.3205/1.9782/2.1291。其他NTU三个协议person0也偏离单位方差，完整六协议向量和对比已记录归一化审查顶部。

PKUv2的H几乎匹配all含空槽（std≈1.0018/1.0003/.9914），却不匹配当前person0（Z mean=.9324、std=.3698）。PKUv2 PT=H/LP=A的输入变换差异仍待核查，不自动改配置。两槽空人物帧比例：NTU60 XSub/XView40.9593%/41.1163%、NTU120 XSub/XSet.2650%/.2426%、PKUv1/v2 45.9394%/49.2579%。NTU120本地预处理会复制单人到第二槽，非零槽不能直接当真实双人。PKUv1有66条、PKUv2有24条非前缀时间padding；原生crop用有效帧数量截取前缀，存在边界，未修数据。

这是seed0、每样本一次的PT干净变换采样，控制台只显示model_input/当前self_shift且约7位有效数字；raw_valid及另一self_shift在服务器报告内未上传。NTU60 XSub无旋转、其余五组有旋转，跨组差异含变换差异。所有数字已核对内部矩关系，但不能据此证明换归一化提高LP或解释文本收益/原生随机性；不新增T29，仍28条成绩。所有现有训练配置、FM/文本代码和成绩不变，不自行改旧checkpoint的LP归一化。

原始附件：[d7a88672控制台](/C:/Users/97537/.codex/attachments/d7a88672-8604-41db-aa66-e513fd8c1e1e/已粘贴的文本.txt)，SHA256=`75627e10b77ea1a2ab2fb20adfb7b728e8395556acc7577e6e34e5173668327f`。不复制日志/新建归档目录；需要完整精度和raw视图时优先核对用户服务器本次输出目录中的报告，不默认重跑。

### 0.0 最新授权已实现：原生单路 MacDiff → OT条件 Flow Matching（2026-10-22）

用户先要求“感觉flow matching可以提上日程了，帮我修改一个替换原版macdiff的吧”，随后补充“要做Optimal flow matching”，并已明确选择“原始FM论文的OT条件路径：噪声→对应骨架，保持原生配对”。当前实现与该选择一致，方法身份`conditional_ot_noise_to_data_v2`、`flow_sigma_min=1e-4`，无需再询问OT定义，不接mini-batch最优配对。保持独立纯骨架单路，不接文本/teacher或两路速度一致性；文本新监督仍待讨论。

新增`main_pretrain_flow_matching.py`、`model/transformer_macdiff_flow_matching.py`、`config/ntu60_xsub_joint/pretrain_madiff_flow_matching.yaml`、`tests/test_flow_matching.py`；方法文档和完整Linux PT400 `&&` LP100一行命令见[MACDIFF_FLOW_MATCHING.md](/D:/program/MacDiff/tools/MACDIFF_FLOW_MATCHING.md)。当前PT/LP端口10309/10310，全新目录`output_dir/ntu60_xsub_native_otfm_dec3_noamp_seed0_20261022/{pt,lp}`；替代之前线性v1的10307/10308和`_fm_`目录命令。助手未启动训练，不预登记成绩编号，不推断用户是否已运行旧命令。

物理时间`t~Uniform[0,1)`，`sigma_t=1-(1-sigma_min)*t`、`x_t=sigma_t*epsilon+t*normalized_clean_skeleton`，速度目标`normalized_clean_skeleton-(1-sigma_min)*epsilon`。t=0噪声、t=1骨架+sigma_min残余噪声；时间编码输入`999*t`保持连续，目标不乘999。masked velocity MSE按坐标→masked patch→样本归约，另加原生`.02*token_uniformity`，无离散diffusion采样、schedule或MinSNR。原`linear_noise_to_data_v1`本来就是OT条件路径sigma_min=0的形式，不能误说其“非OT”。当前没有mini-batch OT耦合；条件高斯OT不保证总体分布的全局OT。不是MeanFlow区间平均速度。

完整复用原生encoder8/decoder3/256维、person0、A归一化、同crop增强、随机mask .9（750位置/75可见）、ids_restore/global-local逐位置调制和condition dropout .1。PT默认/命令均关AMP，双卡每卡32/accum1、400轮、LR.001/warm20/min_lr.0005/floor20/seed0；LP沿原入口每卡64、100轮、LR.1/warm0/min_lr0/floor10。没有新增可训练模块，encoder参数名/形状与原生一致；真实初始化/梯度/LP加载测试仍待服务器执行。FM主目标尺度改变，不能直接拿loss大小判断相对epsilon的收益，uni系数首轮不调。

公共`engine_pretrain.py`仅新增`supports_pretrain_metrics`的辅助日志读取/前缀分支，FM不触发teacher/文本更新；原生基类、LP及已有文本结构未改。子类复用原生构造，随后丢弃参数无关的diffusion helpers，训练不调用。服务器同步新文件+当前engine；若缺此前noAMP补丁，还需当前main_pretrain.py和util/misc.py。

专用入口校验完整FM resume，拒绝旧diffusion/text/teacher及线性v1 checkpoint、model_args（含sigma_min）或主要数据/训练协议变化；不用公共main绕过检查。RNG未保存，数据内容未hash，卡数仍须一致，不声称逐位复现。日志新增`train_loss_flow_matching`/`train_loss_uniformity`，total=前者+.02*后者。

本地OT-FM16项：11通过、5缺Torch跳过；累积回归5项通过，合计16通过/5跳过。实际forward/OT插值/损失NumPy执行、OT端点/有限差分/欧拉场等价/零sigma形式、连续t、person0/归一化/配对、masked归约、日志、CLI/resume已检查；Python3.8语法、配置/命令静态核对和diff空白检查通过。无真实Torch/CUDA/DDP/显存/LP实测；**不能把实现完成说成已跑通PT或提高LP**。此前28条成绩和repeat状态不变。

### 0.0a 此前新增：已改文本读取；新监督仍待讨论（2026-10-21）

用户在上述交接后明确要求“先把文本读取方式改了”，并调研2025/2026论文后再讨论监督。已在既有`model/transformer_macdiff_text.py`新增`text_conditioning_mode=global_mod_local_attention_v1`：global512单独投影256，与时间步MLP相加，逐层生成shift/scale/gate；六个local独立投影后只作为cross-attention K/V，每层再做骨架self-attention和FFN。没有照搬TDSM的local隐状态联合更新。T→S仍不读encoder特征；S→T/目标detach/目标库递推/loss权重未改，没有接入新监督。

新增一个必要的新配置`config/ntu60_xsub_joint/pretrain_madiff_text_global_local_decoder3.yaml`，沿用T23缓存、维度、深度、1/.1权重/数据/LR，显式关闭AMP以符合用户最新选择，新输出目录独立。旧配置默认concat_memory、T23/noAMP repeat命令保留原架构。已有七句缓存无需重建；新输出目标库从头创建，不复用或删除旧库。main保存实际读取模式，misc在state加载前双向拒绝不同模式完整resume；缺字段旧checkpoint按concat_memory。

本地相关59项测试：17通过/42缺Torch跳过；新增实际forward NumPy路由/padding与双向resume检查3项通过，新增真实Torch梯度/共享/损失等5项未执行。Python3.8语法与diff空白检查通过，无Torch/CUDA/DDP/显存/LP实测，未启动训练或编号新成绩。

2025/2026调研及具体公式写入[现有Stage1计划顶部](/D:/program/MacDiff/tools/vlm_pilot/STAGE1_TEXT_EXPERIMENT_PLAN.md)。主要来源GFP(ICCV2025)、MLS³L(KBS2025，摘要/引言)、AMR(CVPR2026)、TranCLR(2026公开稿)。待讨论候选为固定文本软关系蒸馏、运动信息加权native epsilon去噪、部位×多时间尺度固定运动摘要预测；都未实现，不自动扩大成新实验。T28/四组repeat状态继续以下文记录为准。

### 0.1 我们在做什么，用户最后确认了什么

主线是**在原生MacDiff上提升NTU60 XSub joint-only的骨架encoder线性评估（LP）性能**。此前做过文本no-share辅助预训练、原生复现审计，以及结合全局InfoNCE和位置对应跨层AA的纯骨架EMA教师。最新已新增顶部0.0的原生FM单路基线；以下0.2–0.5继续记录此前teacher方法，不代表FM已接入teacher。

用户已明确授权实现该骨架方法，并先后要求：对齐论文中的跨层W/adapter定义；随后去掉教师侧adapter、把AA从.1改为1；出现NaN后关闭AMP；另给独立AA=0组和三组原生/no-share的无AMP repeat命令。旧“只分析、不新增对比学习”的约定已在这个已授权骨架方法范围内被后续指令替代，不能据此拒绝已有工作，也不能扩大成新的文本方案。

**此前日志身份确认：新上传的两份日志是teacher v3、AA=1，从checkpoint-80恢复后关闭AMP。已登记T28。**这不是待回答的问题，不要再询问同一身份。最新代码请求已经转向顶部0.0的原生FM替换；不要继续自动扩展teacher实验。

### 0.2 当前实现：v3，不能与v1/v2混用

| 项目 | 当前生效定义 |
|---|---|
| 方法身份 | `raw_full_position_student_adapter_aa1_v3` |
| 骨干 | 学生/教师encoder均8层、维度256；native decoder 3层、维度256 |
| 输入 | 同一crop、120帧、25 joints、时间patch4；one_person=True，保留person0；A归一化 |
| 学生 | 小joint扰动后的骨架，随机mask .9，75个可见token；native diffusion照旧 |
| 教师 | 同crop原始完整骨架750 token，无学生小扰动/扩散噪声/mask；eval/no_grad，从学生初始化后EMA .999 |
| InfoNCE | 学生末层GAP/L2 query、教师全局key；温度.07、FIFO65536、权重1e-4；按原始sample ID排除自己的历史负样本；无projection head |
| AA | 教师第4–8层按学生`ids_keep`取同物理位置；学生8层用无约束、可正可负的8×5 W混合；每个目标层一个学生侧256→256 Linear（FC） |
| AA目标与归约 | `mean_j SmoothL1(A_j(sum_i W_ij S_i), detach(T_j[ids_keep]))`；各项mean归约；无额外AA LayerNorm，无教师侧adapter |
| 初始化 | W为Normal(std=.02)，学生FC为Xavier/零bias；不是softmax W或恒等FC |
| 总loss | `native epsilon MSE + .02*骨架uniformity + 1e-4*InfoNCE + lambda_alignment*AA`；v3默认AA=1，独立对照CLI覆盖0 |
| 更新时机 | 只有成功、完整optimizer step后更新EMA/全rank队列；AMP跳步和不足累积窗口的尾部不提交辅助状态 |
| 下游 | 原`main_linprobe.py`加载学生同名encoder键，冻结encoder，只训分类head；不选教师 |

AA和InfoNCE从epoch0、所有样本启用，不依赖扩散t或native condition dropout筛选。W与学生FC由AdamW优化，教师encoder无反向、仅EMA。原生native loss仍含uniformity，不能把`loss_native`称为纯epsilon误差。辅助loss日志为外层加权前的值。

v1曾有softmax W、额外LN等自定义；v2改成无约束W、无额外LN，但学生/教师两侧各有可训练FC，AA=.1。**v2双侧FC共同趋0即可使AA为0而encoder无需改善，已用实际AA函数数值反例验证这个退化通路；没有证明T27 checkpoint实际发生坍塌。**教师侧FC直接由优化器训练是原文未写清处的实现选择，不是作者明确规定。用户随后选择当前v3，移除该直接通路，不保证整个EMA系统不会退化或LP提高。

论文对应关系已核查并写入方法文档：Gong论文提供EMA/完整骨架/GAP/InfoNCE思路；ACA²Net提供跨层W/adapter思路。`ids_keep`同位置匹配是本方法适配；ACA²Net未明确该位置gather或教师EMA更新。原文的运动教师、mask、帧数、batch及重建目标等与本方法仍有差异，不能称为两文原样复现，也不能说“除了两个点其余全对齐”。详见[方法和论文对照](/D:/program/MacDiff/tools/MACDIFF_SKELETON_TEACHER.md)。

### 0.3 已完成的代码与AMP修正

| 文件 | 职责/当前状态 |
|---|---|
| [专用入口](/D:/program/MacDiff/main_pretrain_skeleton_teacher.py) | 注册教师模型，解析CLI覆盖，严格检查完整resume定义，默认enable_amp=False，记录数值策略 |
| [教师模型](/D:/program/MacDiff/model/transformer_macdiff_skeleton_teacher.py) | EMA encoder、InfoNCE队列、位置AA、成功步提交、native FP32包装及非有限诊断 |
| [教师配置](/D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_skeleton_teacher.yaml) | 当前v3、默认AA1/CL1e-4、enable_amp=False、PT400、默认`_fp32_noamp`输出目录 |
| [公共PT入口](/D:/program/MacDiff/main_pretrain.py) | 可选初始化钩子；公共`--disable_amp`；所有PT方法用`NativeScaler(enabled=args.enable_amp)` |
| [公共PT engine](/D:/program/MacDiff/engine_pretrain.py) | 仅标记模型走teacher aux/sample-ID/成功步提交；保留官方完整累积窗口更新策略 |
| [Scaler](/D:/program/MacDiff/util/misc.py) | enabled参数默认True；disabled时非有限梯度范数在optimizer.step前抛错，不clip、不自动跳batch |
| [教师测试](/D:/program/MacDiff/tests/test_skeleton_teacher.py) | 静态/实际函数NumPy检查和真实Torch/CUDA用例；本地Torch部分跳过 |

服务器首次v3在epoch89约600/626后因native=NaN退出，失败batch CL=2.761038065、AA=1.763285041有限；这是非有限值，不是OOM。原生decoder先做FP16 QK乘积再缩放、modulation多重乘法存在溢出路径，有源码和有限输入数值反例支持，但没有失败batch/checkpoint复现，**具体NaN算子未查实**。

已先在teacher子类覆盖native decoder/diffusion loss：关闭局部autocast、输入转FP32再调用原方法；细分`decoder_prediction`/`diffusion_loss`/`total_loss`诊断，包含rank/sample IDs/dtype/非有限数/absmax/RMS，finally清理临时引用，不置零loss或吞异常。随后按用户要求关闭整个teacher PT的autocast和GradScaler。v3 model_args/参数结构与原v3兼容，v1/v2仍拒绝完整resume。

**当前AMP规则：**公共PT解析器定义`--disable_amp`，teacher入口继承，不能两处重复注册。原生/no-share默认AMP=True，teacher默认False；三组repeat必须显式带`--disable_amp`。该开关同时关闭PT autocast和scaler。`--enable_amp False`不是关法。FP32参数/输入且关闭autocast时走FP32路径；未修改TF32设置。LP协议没改，LP默认autocast=False，但历史GradScaler仍保留，不能声称LP scaler也关闭。

从原始v3 `checkpoint-80.pth`恢复会从81继续，`epochs`仍400以保留原日程；完整恢复model/optimizer/EMA/Queue/scaler。disabled scaler忽略旧AMP scaler状态，保存新状态为空字典；args数值版本为`full_fp32_no_amp_v1`（若打开AMP则为`native_decoder_loss_fp32_v1`）。这是精度改变的续训，不是原AMP逐位恢复；RNG未保存，数据/行序及全部顶层协议也未与resume身份绑定。已确认T28按此精度转换续训，但实际服务器源码/具体补丁版本与LP加载输出仍缺。

**同步服务器必须成套：**专用entry/model/YAML、公共main/engine、util/misc和测试。只复制模型会缺sample ID/成功步钩子；只同步公共main或旧teacher入口可能重复注册参数。未改原生模型基类的算法数学或LR函数。

### 0.4 最新结果与证据范围

当前共**28条配置/成绩记录**，不等于28次独立初始化或完整PT→LP100。历史结果和全部附件SHA在[Stage1结果](/D:/program/MacDiff/STAGE1_EXPERIMENT_RESULTS.md)及[总审计](/D:/program/MacDiff/EXPERIMENT_RESULTS.md)。

| ID | 身份与日志范围 | LP best (%) | 后期LP均值 (%) |
|---|---|---:|---:|
| T20 | native decoder5、PT min_lr1e-5，完整LP100 | 86.001940813 @95 | 80–99：85.850315396 |
| T21 | 旧no-share 5/5/5、remap512、text uni.02，完整LP100 | 86.220281456 @88 | 80–99：86.134461457 |
| T22 | native decoder3、PT min_lr5e-4，完整LP100 | 85.462154313 @90 | 80–99：85.391800113 |
| T23 | 最佳no-share 3/3/3、remap2048、text uni0，完整LP100 | **86.280931625 @73** | 80–99：86.089883586 |
| T24 | no-share 5/5/3，完整LP100 | 85.522804492 @89 | 80–99：85.414240689 |
| T25 | nscale一致性，仅用户口述best | 85.44 | 缺日志 |
| T26 | native decoder3 seed0 repeat，PT400/LP100 | 85.710819994 @73 | 80–99：85.445172265 |
| T27 | teacher v2、双侧FC/AA.1，PT400、LP仅0–95 | 85.377244084 @90 | 76–95：85.220766640 |
| T28 | teacher v3 AA1，从80关AMP续训；PT附件81–399、LP0–89 | **69.802280459 @87** | 70–89：69.109958775 |

**这些行窗口不同，不能直接用后期均值作配对。**T28与T26共同LP0–89的best差−15.908539535个百分点；共同70–89均值为69.109958775 vs 85.452753533，差−16.342794758个百分点。T27共同0–89 best为85.346918988，不能拿epoch90的85.377244084混入同范围比较。当前最高历史best仍T23，最高已知完整后期均值仍T21。

T28 PT81–399共319行、LP0–89共90行，连续无重复、上传数值均有限；未收到PT0–80/LP90–99，不补猜最终LP100或服务器进程状态。所有319轮满足`total=native+1e-4*CL+AA`，最大绝对残差2.585072e-9，确定不是AA0结果；已有PT和LP各自LR与T26同epoch逐项相同，LP可训练参数始终384060。

T28 PT末20（380–399）均值：native=.022601757304、CL=3.677871770895、AA=.841239377104、total=.864208921464。AA占total标量97.3421%、约native的37.2201倍，**不是encoder梯度占比**。native较T26末20高33.8512%；CL从81轮2.617841290到末20上升40.4925%，AA下降40.8602%，总loss下降主要来自AA。LP train/test CE也均高于历史同窗口，不能只凭test曲线说过拟合。整体峰allocated7142.120117MiB（约6.975GiB）、reserved7736MiB（约7.555GiB）；没有耗时或独立teacher显存剖析，不能把不同run峰差全算作teacher/AMP开销。

T28原始附件（不复制到新目录）：

- PT：[81–399轮](/C:/Users/97537/.codex/attachments/69c66ad7-e69b-4f32-aa0c-944c9384a6ea/已粘贴的文本.txt)，SHA256=`ebf703d1f56ca1b892284db1683010478ebe4216056540d22de208ff84c91193`。
- LP：[0–89轮](/C:/Users/97537/.codex/attachments/69f299ee-0550-4a25-ac55-2a4622db49c9/已粘贴的文本.txt)，SHA256=`230f728f205d7153ca79ba2eb51a6902f739c580c229994b62d2f25489819375`。

### 0.5 当前卡点：能跑完续训，但表示效果明显变差，原因尚未定

1. **当前AA/联合目标是首要排查对象，未确诊梯度冲突。**没有分支encoder梯度范数/夹角、W/adapter范数、特征RMS/样本方差或匹配AA0结果；不能把loss数值占比当梯度主导证据，也不能把低AA等同好表示或确认坍塌。
2. **关闭AMP后续训81–399有限，没有带来好的LP结果；这不证明AMP是此前故障的唯一根因或此次掉点原因。**T28前0–80已受AA1/原AMP训练，关AMP不会撤销历史更新；它不是全程FP32实验。
3. **LP实际权重加载身份仍缺。**本地源码加载学生同名键，EMA有独立前缀，encoder冻结、head训练，静态未发现新增误读teacher路径；没有服务器Namespace/checkpoint SHA/load_state_dict输出，不能证明服务器载入了预期权重。
4. 本地没有Torch/PyYAML/timm、此次NPZ或checkpoint，没有SSH；无法做真实GPU/DDP/梯度、跨rank EMA/Queue及权重实物验证。附件结束不代表服务器停止，保存命令不等于实际运行参数。

### 0.6 下一步已经选定：四组命令，勿再混淆repeat

**repeat只指原生decoder3、原生decoder5、最佳no-share T23全3层，不是teacher。**独立teacher AA0是第四组。2026-10-21新noAMP三组repeat及AA0目前均未收到可确认身份的结果，不推断是否已启动，也不预登记完成ID。历史T26是此前AMP的dec3 repeat，不能冒充这次新noAMP组。

四组均从头seed0、PT400→LP100、新输出目录，无PT resume。共同PT：双卡每卡32、accum1、总64、lr.001、warmup20、min_lr5e-4/末20 floor、mask.9/tau−1；LP：双卡每卡64、总128、lr.1、warmup0、min_lr0/末10 floor、dist_eval、只加载同组PT checkpoint-399。原版decoder5是统一高floor的新基线，不是T20的1e-5重复。

| 已给命令 | 输出根目录（仓库相对路径） | PT/LP端口 |
|---|---|---|
| 原生dec3、关闭AMP | `output_dir/ntu60_xsub_native_dec3_repeat_noamp_seed0_20261021` | 10301 / 10302 |
| 原生dec5、关闭AMP、floor5e-4 | `output_dir/ntu60_xsub_native_dec5_repeat_noamp_minlr5e4_seed0_20261021` | 10303 / 10304 |
| 最佳no-share全3层、关闭AMP | `output_dir/ntu60_xsub_noshare_all_dec3_t23_repeat_noamp_seed0_20261021` | 10305 / 10306 |
| teacher v3 AA0、CL1e-4、关闭AMP | `output_dir/ntu60_xsub_skeleton_teacher_v3_aa0_dec3_noamp_seed0_20261021` | 10299 / 10300 |

完整单行PT&&LP命令已保存，接手后直接读取，不要从历史端口/目录手工拼错：三组repeat在[Stage1计划顶部](/D:/program/MacDiff/tools/vlm_pilot/STAGE1_TEXT_EXPERIMENT_PLAN.md)，AA0在[教师方法文档顶部](/D:/program/MacDiff/tools/MACDIFF_SKELETON_TEACHER.md)。均用GPU0,1，由用户自行安排资源，助手未启动。

no-share选的是T23：native/T→S/S→T各3层、七句v3/sample_target_blend、remap2048、S→T decoder512/none、权重T→S=1/S→T=.1、text uniformity=0、骨架uniformity=.02。不是T21旧5层、T24混深度或T25新一致性。每组用新目标库，复用原文本cache。

AA0显式`--lambda_alignment 0 --lambda_contrastive 0.0001 --disable_amp`；YAML/模型默认AA仍1，CLI仅覆盖这一组。当前为保持DDP图仍计算AA，因此原始`loss_alignment`可能非零，但对总loss及AA反向贡献为0，不承诺省计算/显存。**AA0仍有EMA教师和InfoNCE，不是纯原生MacDiff。**沿已给命令从头训练，不能用AA1 checkpoint当纯AA0对照；若恢复LP，支持当前AA的不利影响，但与T28仍有精度历史/重启差异，不能单独证明梯度冲突。

新会话按用户新消息推进。收到结果先核对身份、轮数、重复、loss组合、同epoch LR和实际LP加载，再登记和用共同窗口比较；缺失轮不补造。若用户继续诊断，优先利用现有checkpoint和固定小批数据检查尺度/样本方差、分支encoder梯度及LP加载身份；不要自行增加AA权重/LN/clip、重启大实验或扩展多seed矩阵。用户已选AA0和三组repeat，不需要再问“是否要这四组命令”。

### 0.7 已验证什么，绝不能夸大成什么

此前本地教师测试22项：**11通过、11因缺Torch跳过**；累积策略5项通过，合计16通过/11跳过。通过项包含真实AA函数和FP32包装函数的NumPy执行、CLI/YAML覆盖/AA0/模型定义校验、真实main的scaler选择、disabled scaler非有限范数在step前拒绝的控制流；Python3.8语法和四组命令参数/目录/checkpoint配对检查通过。真实Torch autograd、旧scaler恢复、CUDA溢出、EMA/Queue/DDP和LP实物测试仍未执行。

服务器测试命令（交给用户，不在缺依赖本地强行安装）：`python -m unittest discover -s tests -p test_skeleton_teacher.py -v`。本次纯交接不重跑训练测试，也不把T28日志有限当成全部状态测试通过。

官方审计已确认AMP和梯度累积原本就存在，不是本次新增。官方PT AMP默认True/LP False；原生不足累积窗口尾梯度丢弃，当前已恢复此策略，不自动补尾step。官方PT DDP为find_unused_parameters=True/broadcast_buffers默认True，当前为False/False，这是已有差异，未按用户指令改回，不能称整个仓库逐位同官方或据此认定分差原因。更早“Scaler与官方AST完全一致”的记录只对应新增enabled/非有限守卫之前的快照，不能作为当前代码描述。

### 0.8 踩过的坑：接手绝对不要再踩

1. **运行身份不混用。**T27是v2/双侧FC/AA.1；T28是v3/学生FC/AA1/resume80后关AMP。不能把T28当AA0、原生repeat或从头FP32；不用日志缺尾判断服务器状态。
2. **AA标量、收敛和梯度是不同证据。**小AA不证明好表示，97% loss数值不证明97% encoder梯度；双侧可训练目标可一起缩小，不能再无依据恢复该设计。
3. **完整teacher必须保持原位置顺序。**原生mask_ratio=0也会全序列随机置换，不能直接这样取`ids_keep`当同位置；不能把跨层W说成位置匹配算法。
4. **EMA/Queue初始化与提交时机固定。**从学生复制；初始化在resume载入前，加载后不能重置；只成功完整step提交，不能每microbatch/AMP跳步更新；空队列首窗口InfoNCE=0正常。
5. **AMP关法与同步成套。**用公共`--disable_amp`，不是`--enable_amp False`；同时关PT autocast/scaler。teacher入口不重复注册；LP scaler未改；不要声称旧官方AMP缺陷已证实。
6. **resume不能随便改模型定义。**完整恢复校验model/model_args，包含AA权重；AA1改AA0不属于现有允许的完整resume。跨v1/v2/native/text不能恢复成v3；严格加载学生encoder迁移也不是完整PT恢复。未保存RNG，不承诺逐位续跑。
7. **PT/LP必须分目录。**T14共目录曾让LP checkpoint0–99覆盖早期PT0–90。400轮末399、500轮末499；400后续100不等同从头500日程。LP只读本组新PT权重。
8. **目标库保护不删。**no-share完整PT resume要同次checkpoint-X与checkpoint-X-target-bank.sqlite配对；current.shared.json不是配套快照。已有库拒绝复用时换新输出目录，不删保护/强覆旧库。LP不需要目标库。
9. **相同seed不等于独立seed方差或逐位一致。**同seed重复是执行复核；encoder额外初始化/前向会消耗RNG，历史隔离入口和当前根入口不同，cuDNN/环境也需核验。epoch末20 std不能代替跨训练方差；best差异不自动是稳定收益。
10. **文件名/YAML/命令不是实际Namespace。**权重、floor、batch必须核实；global batch=每卡batch×卡数×accum，不能擅自改accum来“恢复官方”。保持用户所选seed0，不自动安排seed1/2。
11. **归一化和person身份不能暗改。**A在模型内，feeder normalization=False不代表关闭A；mean/var不是state_dict参数，PT/LP要一致。person0空不擅换person1，sample ID不能用shuffle batch行号。更多数据疑点先查实物，不先修再称同协议。
12. **命令格式固定。**中文解释；Linux单行、不加cd、PT和LP用&&。CalledProcessError要看前面的rank真实traceback。四组共用两张卡，不推断可以同时跑。
13. **保护未提交工作。**不git clean/reset、删移checkpoint/cache/artifact、不提交推送、不安装依赖或远程启停。未跟踪teacher文件是成果；工作区其他caption/text consistency修改不是垃圾。用户嫌文件多，优先维护现有文档，不新建重复归档/日志/目录。
14. **caption/cache沿用完整身份。**已有cache绑定实现SHA，无关格式修改也可破坏身份；不重生成旧NTU60文本，不用cat拼JSON数组。XSet新独立入口与旧benchmark入口不混用同一计划。
15. **当前选择优先。**不恢复旧text uniformity、seed1、768、GT warmup、decoder蒸馏或其他仅讨论候选；不把旧“尚无v3结果”继续当当前事实。

### 0.9 环境、并行遗留工作与接手索引

本地`D:/program/MacDiff`，Windows PowerShell；服务器仓库`/home/user9/public3/swr/MacDiff`，数据`../data/MAMP/ntu/NTU60_XSub.npz`（历史预期train40091/test16487）。历史服务器环境Python3.8/Torch1.8.1+cu111、双RTX4090，最新实际版本未提供。助手没有SSH或此次实物NPZ/checkpoint。Bundled Python：`C:/Users/97537/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe`，有NumPy、无Torch/PyYAML/timm；系统Python另查，不能照搬旧依赖结论。

`git status`当前有公共main/engine/util/misc、结果/计划文档、text consistency/caption/render等已修改文件；teacher entry/model/YAML/tests/方法文档及native decoder3配置等仍为未跟踪。**本会话没有commit/push，接手不能依赖HEAD认为实现已入库。**同步前按功能检查文件，不清除其他任务成果。

caption并行遗留：旧NTU60 XSub文本/cache已完成并沿用；用户此前“都生成完了”只指其他划分的骨架GIF，不代表描述已生成。XSet已新增独立`tools/vlm_pilot/run_caption_ntu120_xset.py`，render阶段不依赖完整源文本，caption阶段才校验/绑定XSub、XView、NTU120 XSub三源及人物槽/模型/prompt/revision和SHA。新XSet与旧`--benchmark`入口不能混同一计划；不改旧renderer/cache实现SHA迫使已渲染数据重跑。按README原入口做XView→NTU120 XSub→独立XSet描述，PKU独立，最后分别CLIP编码；本次没有启动这些任务，后续仅在用户切回时处理。

优先阅读：本第0节 → [教师方法/论文审计/AA0命令](/D:/program/MacDiff/tools/MACDIFF_SKELETON_TEACHER.md) → [Stage1结果](/D:/program/MacDiff/STAGE1_EXPERIMENT_RESULTS.md) → [当前三组repeat命令](/D:/program/MacDiff/tools/vlm_pilot/STAGE1_TEXT_EXPERIMENT_PLAN.md) → [实验总审计](/D:/program/MacDiff/EXPERIMENT_RESULTS.md)。caption入口说明见[README](/D:/program/MacDiff/tools/vlm_pilot/README.md)。

官方隔离源码仍被历史命令引用：[upstream_692888f](/D:/program/MacDiff/handoff_artifacts/upstream_692888f/ISOLATION_README.md)，固定官方提交`692888f1e2a4227216511aac6733dff31b587213`；不得删除。论文路径：`D:/DL/08371-GongT.pdf`、`D:/program/paper/Asymmetric_Context-Guided_Adaptive_Alignment_Network_for_Skeleton-Based_Action_Recognition.pdf`、`D:/program/paper/macdiff.pdf`。更多原生/MAMP/旋转/文本历史审计索引保留在下方旧版附录，不用重新从零审计。

## 历史附录：旧版交接正文，仅作追溯

以下保留旧版第1–9节，避免丢失早期设计、实验、caption与审计资料；部分段落已追加过新结果，但整体未逐段重写。**其中所有“当前/最新/下一步”、实验数、测试数、AMP/scaler行为、端口和命令均是历史语境，不作为当前操作指令。当前生效信息统一见上方第0节。**

### 1. 接手先看：当前选择和最后一个问题

**2026-10-19最新：用户已明确授权在原生MacDiff上实现骨架EMA教师＋全局InfoNCE＋位置对应的自适应跨层对齐，并要求完整数据流。**此前“只分析不实现”“不做新增对比/跨样本”的约束已被这次明确指令在该骨架方法范围内替代；继续不评估教师encoder、不使用文本/预训练教师、不启动服务器训练。新增专用入口`main_pretrain_skeleton_teacher.py`、模型`model/transformer_macdiff_skeleton_teacher.py`、配置`config/ntu60_xsub_joint/pretrain_madiff_skeleton_teacher.yaml`和测试`tests/test_skeleton_teacher.py`，完整实现/数据流/单行PT&&LP命令见[方法说明](D:/program/MacDiff/tools/MACDIFF_SKELETON_TEACHER.md)。原生基类训练数学不变；新模型调用其forward_macdiff，公共main仅增DDP后/resume前可选初始化钩子，engine仅为带标记的新模型记录aux/传sample IDs及成功优化步后提交EMA和Queue；旧native/text/OSE不进入新钩子。既有三组seed0复跑命令和caption流程保留。

当前v3默认native decoder3，学生/教师encoder8×256，教师读同crop完整原始骨架750token（无小扰动/扩散噪声/mask，eval/no_grad），学生沿用小扰动/.9mask保留75token。学生全局池化query和教师全局key做InfoNCE（温度.07/FIFO65536，raw sample index排除自己历史视图、不用标签；空队列首完整优化窗口为0）。教师第4–8层按学生ids_keep取同位置75token；学生8层经8×5无约束W直接线性组合，每目标层仅学生侧一个256→256 Linear adapter，直接与detach后的原始教师特征做FP32 SmoothL1，无额外LN。W及学生adapter由AdamW训练，教师encoder冻结/EMA。总loss=native epsilon MSE+.02骨架uni+1e-4 InfoNCE+1 AA。辅助loss全部样本/从epoch0启用，不受native condition dropout或t筛选影响，native全t/drop.1保留。教师从当前随机学生复制，成功完整optimizer step后EMA.999和全rank keys/IDs入队；AMP跳步/不完整尾窗口不更新。完整state和严格model_args恢复，拒绝旧v1/v2；LP只读学生，原linprobe2/LP min_lr0，PT5e-4。

历史v1联合验证为21通过/16缺Torch跳过；当前v3含精度保护后为教师11通过/11跳过、累积5通过，共16通过/11跳过，Python3.8语法及实际CLI检查通过。本地无Torch/PyYAML/timm，不能称实际前向、梯度、DDP、AMP或GPU验证。T27为旧v2结果，T28为v3从80关AMP续训，LP截至89 best69.8023，未显示提升。论文对应关系：ACA²Net没有明确描述按原始时空坐标gather同位置token，其W是跨层矩阵；v3保留物理位置匹配，使用无约束W/学生FC/无额外LN，按用户新选择去掉教师FC、AA改1。ACA²Net正文未明确EMA规则，本方法EMA依据Gong第一篇；论文PDF在D:/program/paper/Asymmetric_Context-Guided_Adaptive_Alignment_Network_for_Skeleton-Based_Action_Recognition.pdf，第一篇在D:/DL/08371-GongT.pdf。

**2026-10-19文本流程澄清并修正：用户“都生成完了”仅指前面骨架GIF渲染完成，新划分尚未生成描述。**旧NTU60 XSub的完整文本/cache此前已通过检查，继续复用。上一答把XSet完整源描述当作其渲染前提，依赖放早；渲染去重本身只需ID和坐标。已新增独立 `tools/vlm_pilot/run_caption_ntu120_xset.py`（默认stage=render），让XSet在XView/NTU120 XSub没有文本时先完成骨架去重/新增22503条GIF，保存plan与source_person_slots。其caption阶段才校验三份完整源文本/人物槽/model/prompt/revision，并冻结文本和metadata SHA到reuse_caption_sources.json，最终metadata也保存绑定；续跑拒绝源变化，caption阶段不加载NPZ数组。渲染/描述都使用这个独立入口，输出仍为vlm_pilot/ntu120_xset/；旧--benchmark XSet入口只给已存在旧计划保留，两种入口不可混用同一计划。此前已渲染的XView/NTU120 XSub/PKU仍用原入口--stage caption；本次没有修改三个原入口、公共dataset pipeline、原renderer或cache实现，保护已有manifest的实现SHA身份，不要求重新渲染。README4.1/4.2已改为全部先渲染，再按XView→NTU120 XSub→XSet顺序生成描述，PKU独立，最后分别CLIP编码。合成数据验证源文本全不存在时可渲染/续跑、缺源时caption明确等待、后补文本三源复用、无NPZ读取的caption续跑、metadata稳定及源变化/坐标冲突拒绝；未启动服务器任务。

**2026-10-19当前任务：用户已选择原版decoder3、原版decoder5及最佳no-share再做完整PT→LP，并讨论两路教师信息来源。**用户明确两次回答：保持seed0完整重复以检查同设置复现性；三组PT min_lr统一5e-4、LP仍0，5层作为新基线而非T20历史1e-5重复。原版3层参照T22=85.462154313，no-share用T23=86.280931625的3/3/3、remap2048/text decoder512/none、text uni0、GT权重1/.1/sample_target_blend，不用T25新一致性。encoder都仍8层。共同PT400/双卡每卡32/accum1/总64/lr.001/warmup20/floor20/seed0，LP100/每卡64/总128/lr.1/warmup0/floor10/min_lr0/seed0/dist_eval/checkpoint399。不跑seed1/seed2，不换成跨seed实验。

已新增根配置 `config/ntu60_xsub_joint/pretrain_madiff_decoder3.yaml`；原版5层复用pretrain_madiff.yaml，no-share复用既有decoder3 YAML。三条单行PT&&LP命令已放[计划顶部](D:/program/MacDiff/tools/vlm_pilot/STAGE1_TEXT_EXPERIMENT_PLAN.md:3)，全用根主入口/配置，端口10287–10292，三组各自新输出目录及新目标库，无PT resume/旧权重复用。没有启动/重跑/检查服务器进程；待真实回传，不登记完成的新ID。相同seed重跑只检查执行可复现性，不估计独立seed方差；T22历史用隔离入口、现在按用户约定用根入口，历史代码/环境变化仍需考虑，不能把分差一律归因随机采样。保存实际Namespace/代码版本/加载路径与完整PT/LP日志，比较best/last/末20。

**历史教师讨论（已被顶部纯骨架教师实现替代）：用户明确不用预训练好的MacDiff作教师，利用额外可见信息，从头学习。**当时只讨论、不实现，提出学生见10%骨架、EMA教师见50%～100%、GT warmup后latent蒸馏等候选；这些并非当前实现的开关。当前已授权并实现原始完整骨架教师/InfoNCE/位置AA，从epoch0启用，无text/T→S/GT warmup。保留的原则是教师初始化复制学生并随成功优化步EMA更新，不能冻结随机网络；更多上下文才是额外信息，EMA仅平滑，不能保证目标正确或防坍塌。

历史文本教师候选（不是当前实现）：更多骨架→E_bar→监督在线R_S（512→256辅助头）→EMA R_bar文本教师→监督masked E_S（256→512辅助头），学生各自GT epsilon监督保留、CLIP缓存不变，不额外加载PT checkpoint。只分别回归共同骨架教师不能建立文本到encoder路径。decoder层对齐自身不增加信息；独立t/噪声的隐状态直接逐层MSE有匹配干扰风险。更多/更长骨架、双人物、joint/bone/velocity、融合teacher等也是旧候选。详细历史设计和data2vec、I-JEPA、DINO来源见[现有计划](D:/program/MacDiff/tools/vlm_pilot/STAGE1_TEXT_EXPERIMENT_PLAN.md)。旧“不做对比/跨样本”约定在当前已授权的骨架InfoNCE范围内被替代，不恢复text uniformity，不承诺LP收益。

静态核对已通过：新增原版3层配置与T22历史模型参数语义一致；与主5层补齐既有默认参数后只有decoder_depth差异。三条PT/LP命令实际argparse解析、seed/轮数/batch/累积/lr/floor/新目录/新checkpoint/端口与no-share旧方法身份核对通过。无PyYAML/Torch本地依赖，配置比较为静态已知结构读取，不称导入/训练/GPU验证；未写新测试。

**2026-10-19最新：用户回传85.44，并明确确认是nscale新一致性组（native/T→S各3层、权重1/1/1）的LP最高值，登记T25口述结果。**尚无完整PT/LP日志、实际Namespace/权重加载身份或checkpoint SHA；不补推best epoch/last/末20均值、400/100轮是否完整、loss稳定性和实际训练参数。该次登记后共25条配置/成绩记录（含口述）；10月20日T26/T27追加后为27条，10月21日T28追加后为28条，最高best仍T23=86.280931625，最高已知后期均值仍T21。T25约比T22原版3层85.462154313低.02pp、比T23旧no-share全3层低.84pp，本次没有显示LP收益；不能把“修复了显式数值放大”当成表示改善。

用户质疑新增组件无效/可能只是MacDiff原生波动。应承认证据边界：T23相对T22单次记录best+.818777pp、末20+.698083pp，T21相对T20 best+.218341pp，是观察到的正向结果，尚非独立PT重复验证的稳定方法收益；T24/T25显示新改动不稳健，不能笼统称所有方法有效或全部只是噪声。T20相同PT checkpoint/同seed重复LP两次逐epoch完全相同，仅说明那组LP可复现，不估计跨PT训练方差；同一run末20 std也不能作跨seed方差。用户已说不跑seed1，不据此自动安排重复PT。固定CLIP/条件打乱/geometry诊断已有，不能当全新实验建议。助手建议暂不扩展一致性、停止把decoder loss降低或梯度通路存在当语义增益证据；这是建议，不是用户已取消/暂停任务。本次只更新4份既有结果/计划/交接，不改训练代码/权重、不启动/重跑任务。

**2026-10-18最新修复：用户问如何修复一致性尖峰，已在新方法中加入每样本noise-scale归一化。**修改既有模型/入口/YAML/测试，不改原版、旧text方法、distill或公共engine。保留完整0–999、两路独立t/噪声、双侧反传、同一masked位置、condition dropout/有效样本筛选和native/T2S/consistency=1/1/1、骨架uni.02。新一致性为每样本 `masked_MSE(x0_native,x0_text)/(k_native²+k_text²+1e-6)`，其中k²=(1-alpha_bar)/alpha_bar；实现先用FP32将x0差除以sqrt分母，再平方并按patch/参与样本平均，避免先平方再缩放。改变的是一致性loss定义，不是限制t、裁剪骨架或直接改低外层权重；日志loss_consistency现为归一化后的值，fraction仍是有效样本里native条件未drop的比例。新旧loss数值不可直接当训练质量比较。

模型构造器和YAML新增 `consistency_normalization: noise_scale_v1`，入口也补入该字段保存到checkpoint model_args；严格model_args恢复检查因此拒绝缺少该字段的旧raw-MSE checkpoint。此次应从头训练，默认输出换到 `output_dir/ntu60_xsub_text_consistency_nscale_dec3_remap2048_ep400_seed0_20261018/{pt,lp}`，不覆盖失败目录，不自动恢复/启动任务。新单行PT&&LP命令见[计划顶部](D:/program/MacDiff/tools/vlm_pilot/STAGE1_TEXT_EXPERIMENT_PLAN.md:3)，仅需同步模型、入口、YAML；测试文件供服务器检查。异常时模型抛FloatingPointError，包含rank、各分支loss、两路t、条件保留标志与预测/encoder/remap非有限元素数量，非master也会显示traceback；正常日志不增加用户已删的统计字段。

验证：新方法20项中10通过、10因无Torch跳过；与既有text/cache/累积策略合计65项中28通过、37因缺Torch/PyYAML跳过。实际日程+实际scale函数的纯标量端点/残差界限、入口旧raw恢复拒绝/新定义接受检查通过；3个Python文件的3.8语法、模型配置字段与构造器、命令CLI/新目录/LP权重路径和PT/LP floor静态检查通过，不是实际PyYAML解析。不能称Torch前向/双侧梯度/优化/异常诊断或CUDA/AMP/DDP数值稳定已验证。默认Python缺NumPy导致两份既有测试导入失败，已切换有NumPy的bundled Python完成上述检查；未安装依赖。当前尚无修复后服务器成绩；只定位并修复显式高噪声放大，不保证LP提升或排除其他数值故障。下文raw MSE/“未实现归一化”等是修复前历史，以本节为准。

**2026-10-18最新回传：新双向x0一致性方法在epoch8中途退出，尚无完整PT/LP成绩，不登记完成的T25。**原始附件 `C:/Users/97537/.codex/attachments/2e01d102-ea57-44e8-9966-82d1b0cbcb97/已粘贴的文本.txt`，SHA256 `8ed7fa6547c5038bdd8b2afada49e4053f8bf6c9daf416144bbbc3d5bf2e15c2`。命令确认lambda_T2S/lambda_consistency=1/1、每卡32、PT400/floor5e-4；epoch8 step400累计均值native=.6815、skel_uni=.9928、T2S=1.1941、consistency=768.7751、total=770.6706，一致性约占总loss数值99.75%，不是测得的梯度占比。consistency近期20步中位数2.8900而累计均值768.7751，step280累计均值曾963.0015，说明少数巨大尖峰；日志括号外为近期20步中位数，括号内为本rank当前epoch累计均值。skel_uni为归一化token间cos²均值，接近1提示样本内token近共线，不能据此认定跨样本完全坍塌。

rank1返回正退出码11，与现有engine_pretrain.py:149–152非有限loss保护的sys.exit(11)吻合；非master print默认被屏蔽，能解释为什么未看到“Loss is nan/inf”。附件没有实际失败批次分支值，尚不能确认NaN/Inf首先发生在哪个张量；无OOM证据，launcher traceback不是根因定位。pred/target及t563是原有随机debug打印，不是已定位的故障批次。

日程代码核算：inverse_cosine/tau1/1000步时，x0还原误差平方系数k²=(1-alpha_bar)/alpha_bar，在t500约.9774、t990约55.17、t998约499.61、t999约500606.93。独立两路raw一致性为||kS*rS-kT*rT||²（r为epsilon预测残差），高噪声末端会强烈放大误差；外层1/1/1不代表有效尺度相同。这解释尖峰的机制，但缺少失败批次t/分支有限性，不能声称已捕获精确非有限链条。当前helper已经FP32还原/算MSE，不能把“改FP32”当新增修复。可讨论保留全t、按每样本kS²+kT²+epsilon归一化一致性；该方案会改变loss定义，尚未实现/用户尚未选择。用户已明确全t和默认1/1/1，本次不擅改权重、t范围或loss。只更新既有handoff/计划，无训练代码改动/训练启动。

**2026-10-18最新开发：用户要求新建方法，以native与T→S的双向骨架重建一致性替换S→T文本去噪。**已新增根入口 `main_pretrain_text_consistency.py`、模型 `model/transformer_macdiff_text_consistency.py`、配置 `config/ntu60_xsub_joint/pretrain_madiff_text_consistency.yaml`、测试 `tests/test_text_consistency.py`，详细定义/运行命令更新在[现有计划顶部](D:/program/MacDiff/tools/vlm_pilot/STAGE1_TEXT_EXPERIMENT_PLAN.md:3)。原版/text旧方法、distill、公共main_pretrain/engine均未修改，没有启动训练/新训练结果，不登记T25。

用户明确选择**完整时间步都做一致性，随后指定native/T→S/consistency三个默认权重均为1**，不是助手提出的300–700筛选/默认.1。新模型两侧从现有完整0–999日程分别独立抽t和高斯噪声；相同干净骨架/crop/person0/A空间，经各自decoder预测epsilon并一步还原归一化x0，FP32在同一encoder masked位置做一次MSE，两侧都不detach。保留native/T→S各自GT epsilon监督、骨架uniformity.02；native条件drop.1保留，被drop的样本不参与一致性。无t筛选/SNR重加权/坐标裁剪/反归一化/文本目标EMA；不构造S→T decoder和逐样本文本目标库。native/T→S默认3层、remap2048、文本缓存512/global+6local、PT400/总64/min_lr5e-4/seed0保持T23公共设置，LP走根入口/主配置。

一致性权重现已按用户选择设为1：YAML顶层lambda_consistency=1.0，模型默认和入口回退值均为1；CLI `--lambda_consistency WEIGHT` 可覆盖，不再要求额外填值或设置环境变量。native固定系数1，T→S默认1，骨架uniformity保持.02；负数/非有限值拒绝，0可做无一致性对照。新增日志为native/T→S/骨架uni/consistency loss和consistency_fraction，不恢复energy/variance/valid token。无目标库，不读写旧bank/snapshot。encoder键名兼容既有LP；resume只接受同方法完整checkpoint并检查model_args（含权重/深度）及原有cache身份，不把T23/T24当resume。Torch前向/双侧梯度/优化与LP加载因本地无Torch尚未验证。文本生成和distill分支仍保持下文既有状态。

新方法最终测试为8通过/9因无Torch跳过；既有text/cache/累积策略18通过/27因无Torch或PyYAML跳过。新增3个Python文件的3.8语法、配置字段与实际CLI/构造器对应、文档检查通过；不是完整PyYAML导入或Torch/CUDA/DDP验证。只新增4个必要实现/配置/测试文件并更新现有handoff/计划，没有新建artifact归档/复制日志。

**2026-10-18最新：用户问新日志“为什么这么低”，并明确确认对应10月16日no-share 5/5/3组，登记T24。**best85.522804492%@89、last85.383309101、末20均值85.414240689±.054243770；相对T23全3层best−.758127134pp、末20−.675642897pp，末20全部更低。PT/LP逐epoch LR与T23完全相同，text uni全程0、loss组合1/.1、目标库epoch1起40091、PT400/LP100连续有限已核验；本地两份YAML的训练参数仅native/T→S decoder_depth3→5，S→T仍3，用户确认其余沿用T23。完整服务器Namespace、代码/数据/checkpoint SHA和加载信息未提供，不称实物身份全部独立核验。T24已完成，不再等待/重跑；当前24条Stage1记录，最高best仍T23、最高后期均值仍T21。

T24末20 native+.02*skel_uni=.016053768166（比T23低4.739631%）、T→S=.021204194818（低3.808544%）、S→T=.007920559358（低26.559421%），训练目标更好但LP更弱。LP末20train CE也从.358558020升至.401845749，不能直接解释成LP训练过拟合。较深native可能减少对encoder条件依赖、较深T→S改变remap/动态目标，这些只是与日志相容的机制假设；T→S不直接对骨架encoder反传，两套decoder同时变化，尚未定位各自贡献，不能声称shortcut已证实。不能再把此次下降归因新改了batch/floor/remap/text uni。根LP与隔离LP本地静态比较未发现改变双卡Top1流程的差异，test loss meter计数略不同；不等同服务器数值等价验证。详细统计/附件SHA见[no-share分析顶部](D:/program/MacDiff/tools/vlm_pilot/NOSHARE_NORMB_LOG_ANALYSIS.md:3)及[Stage1结果](D:/program/MacDiff/STAGE1_EXPERIMENT_RESULTS.md:9)。本次只更新5份既有文档，不改训练代码/配置、不复制日志/新建目录、不启动训练；caption/distill其他分支状态保持下文记录，不因T24推断进度。

**2026-10-17当前问题：用户觉得新划分渲染慢，已核对与原NTU60 XSub一致性。**相同输入/参数/环境时新旧直接复用相同render_sample_frames/save_gif/make_metadata，默认32帧8fps、960×360、font18、2view/root-centered、median3/Savgol5/poly2/gap2、GIF optimize=False/disposal2一致。本地3种300帧合成样本（单人、复制人物槽、双人）两轮，旧1.3719/1.4336s、新1.4370/1.4117s，每例GIF字节/config/render信息完全一致；不是服务器速度。单例绘制平滑.078–.106s、GIF编码.346–.404s，cProfile主要在Pillow逐帧quantize（约.302s/32帧）。慢的已确认差异：新NTU首次需额外读源NPZ/描述并做ID坐标去重；原逐样本日志改成每100样本一次；新渲染固定CPU单进程，gpus只给后续VLM，原默认也单进程但支持手动num_shards并行；两者NPZ x_train均整体加载，原mmap_mode不是真正NPZ memmap。不同数据源有效长度/人物槽影响代价，NTU120复制槽参与2人渲染。已异步询问用户慢在Loading/Checking还是Rendered及最近日志，尚未收到本问题的具体耗时，不可断定服务器主因或给总耗时保证。新增同输入GIF字节回归测试、README4.4及读最近20个render_seconds的只读命令；此次不修改绑定计划身份的渲染/生成实现、不启动/重启服务器任务，已在运行的任务可继续。若要优化并行/日志，需先考虑当前已有manifest的实现SHA保护，不能随意改代码后让用户丢弃产物重来。

**2026-10-17服务器只读检查已有真实回传。**用户附件 `C:/Users/97537/.codex/attachments/78295564-7e84-45ac-8ff1-46d2dc4f0183/已粘贴的文本.txt` 为macdiff环境运行check_text_pipeline.py的完整输出。六份NPZ均data=ok：NTU60 XSub40091/16487、XView37646/18932、NTU120 XSub63026/50919、XSet54468/59477、PKUv1 XSub18840/2704、PKUv2 XSub5299/1613。NTU60 XSub render/caption/cache均ok，描述40091条/47608人物槽、Qwen3-VL-8B、prompt SHA debaff8e850927390c407da845387d405000de3f78c5126f49b23e97e1bc29ce（与本地v2 prompt字节SHA一致）、revision null；cache512维/context7、full_npz_and_feature_hashes=true、complete=true。GIF本轮未解码，不称语义质量重新验证。其余五份在默认目录caption/cache缺失。助手发现旧检查器无复用计划时将完整train误报为“新增渲染/缺失数”，已修正：新划分无manifest报告not_planned/expected_new_renders=null，另列前置源齐全后的统计预计值；原NTU60 XSub无新版manifest仍检查既有全量GIF。实际待新增预计XView10970、NTU120 XSub22935、XSet22503（待前两份完整描述）、PKUv1 18840、PKUv2 5299。下一步先为XView/NTU120 XSub/PKU执行--stage render，再Qwen生成描述；XSet等待依赖描述完成后开始，最后各自CLIP编码。不得重做NTU60 XSub、把旧全量渲染计数当作重复生成要求，或根据此输出推断其他目录产物/正在运行进程。PKU仅核对NPZ头数量与后续自洽性，原始受试者划分仍未独立审查。没有助手启动服务器任务或改动实际产物。

**2026-10-17最新范围：检查NTU60 XSub/XView、NTU120 XSub/XSetup、PKU XSub（v1/v2分开）。**项目cross-setup文件名/参数为XSet。`run_caption_ntu120.py`已补 `--benchmark XSet`，按偶数setup训练、奇数测试、setup-major行序；默认复用完整NTU60 XSub→XView→NTU120 XSub文本并逐ID/坐标核验，缺源直接报错，不静默重复生成。NTU120 XSet train54468/test59477，仅复用NTU60 XSub为16087；前述三个源合计31965，真正新增22503。NTU60 XSub40091/16487（已有，不重做），XView37646/18932（复用26676、新增10970），NTU120 XSub63026/50919（复用40091、新增22935）。XSet独立保存于 `vlm_pilot/ntu120_xset/`，其cache在该目录 `clip_sentence_cache_v3/`。推荐先完成XView/NTU120 XSub文本，再为XSet先render后caption、最后独立CLIP编码；`--stage all`仍只前两步。PKU v1/v2沿用各自XSub NPZ，不混用文本/索引/cache；原始PKU split文件及实际NPZ不在本地，数量/划分不能凭代码声称实物已核验。新增只读 `tools/vlm_pilot/check_text_pipeline.py` 一次检查六份NPZ和render/caption/cache；默认完整cache hash，`--stages data`只查数量，`--quick`降为cache size/header，`--check_gifs`额外解码帧数；缺失/不完整/错误返回非零，不启动模型或写文件。当前本地六份真实NPZ均缺失，已实际运行头信息检查确认missing，不能报告服务器产物完成。完整命令和表见 [README 4.2/4.3](D:/program/MacDiff/tools/vlm_pilot/README.md)。本地合成数据验证三源复用、setup行重排、复制人物槽、完整cache衔接和只读检查器；四份cache身份绑定实现/既有cache/训练代码均未改，无新增训练结果。

**2026-10-17补充已实现：NTU60 XView也需要描述，三阶段流程已整理并检查。**新增 `tools/vlm_pilot/run_caption_ntu60_xview.py`，复用NTU120入口的ID/坐标核验逻辑；NTU统计按XSub subject-major和XView camera2→3顺序恢复行号。XView训练37646、测试18932；与已生成XSub训练40091相交26676，需新增10970（XSub测试受试者的camera2/3，在XView属训练集；NTU120 XSub也不覆盖）。仅新增骨架渲染/调用VLM，最终完整文本与metadata保存到 `vlm_pilot/ntu60_xview/`，不将其用于XSub训练。用户明确要求先保存渲染，再保存描述，再CLIP编码cache：[README 4.2](D:/program/MacDiff/tools/vlm_pilot/README.md)已给XView/NTU120/PKU的 `--stage render`、`--stage caption` 和四条 `cache_clip_motion_text.py` 命令；`--stage all`仍仅前两步，不自动CLIP编码。各自cache保存于各dataset根的 `clip_sentence_cache_v3/`，完整合并文本配对应NPZ，不能用只含新增的generated文本或共用XSub cache。本次检查修复公共pipeline和原dual-GPU launcher的无新增合并仍刷新metadata时间戳问题，避免后续CLIP cache来源SHA被无谓改变；未修改四份绑定cache身份的实现、既有NTU60文本/cache或训练代码。本地真实GIF+模拟VLM/CLIP三阶段衔接通过，含无新增描述后已完成cache仍可复用；服务器真实NPZ/GPU全量任务未启动。

**2026-10-17新增任务已实现：沿用已完成的 NTU60 流程，为 NTU120、PKU 补独立描述脚本。**入口为 `tools/vlm_pilot/run_caption_ntu120.py` 和 `tools/vlm_pilot/run_caption_pku.py`，共用 `dataset_caption_pipeline.py`；32帧/8fps/双视图/原平滑参数、v2 global+6local prompt、Qwen3-VL-8B/双卡/1024token均沿用。仅针对XSub的x_train，不读y_train，不改原NTU60脚本、描述、NPZ或训练配置。NTU120按statistics的原始骨架ID和subject-major行顺序映射，并核对完整坐标hash；当前统计63026行中40091复用NTU60、22935新增，仅新增行渲染/推理。NTU120单人复制槽将原person0文本复制至person1，保留原数据与人物槽对齐，不再调用VLM。最终 `vlm_pilot/ntu120_xsub/captions.json` 为复用+新增，`generated/captions.json` 仅新增。PKU默认依次v1/v2，各自保存于 `vlm_pilot/pku_v1_xsub/`、`vlm_pilot/pku_v2_xsub/`，包含独立rendered/generated/最终JSON及metadata。支持all/render/caption、dry_run、merge_only和自动续跑；源描述/model/prompt/NPZ身份变化拒绝混用。命令和可改路径见 [VLM README 4.1](D:/program/MacDiff/tools/vlm_pilot/README.md)。本地用bundled NumPy/Pillow验证实际GIF渲染与模拟GPU worker，未运行真实VLM、服务器全量生成或CLIP缓存；没有新训练结果。

**2026-10-16最新问题：核查distill是否存在Gong等AAAI-25论文的问题。**已阅读用户本地 `D:/DL/08371-GongT.pdf`（Rethinking Masked Data Reconstruction Pretraining for Strong 3D Action Representation Learning）。当前distill是固定逐轴A归一化，没有论文公式4的patch内时间/坐标混合mean/std，也没有公式5 NAT的短时运动强化；相关弱运动监督风险未排除，但不能认定已退化。重要差别是noise2joint逐样本严格等价于 `((1-alpha_bar_t)/alpha_bar_t)*masked_epsilon_MSE`，不是普通MAE直接坐标回归；t300–700不等于NAT。现有token uniformity只在样本内，不提供论文的跨样本InfoNCE约束。建议先分解4帧patch均值/动态误差和检查条件依赖，若做NAT消融仅在loss端加辅助项，不改变冻结decoder输入/扩散空间；尚未实现或启动。详细证据和限制补入[现有distill说明](D:/program/MacDiff/tools/MACDIFF_DECODER_DISTILL.md)。没有distill训练结果，本次未修改模型或配置。

**2026-10-16补充因素对照分析（10月18日补T24）：**已在[Stage1结果表](D:/program/MacDiff/STAGE1_EXPERIMENT_RESULTS.md)新增LP best因素归因表。整套文本辅助T20→T21为+.218341pp、T22→T23为+.818777pp；原版PT旋转T17→T18为+.151625pp；同encoder LP总batch256→128约+.10pp（后者口述）。七句512/none共享→不共享约+.406021pp，T→S权重.1→1为+.090975pp；PT accum2→1有效batch128→64仅+.024260pp，S→T5→3仅口述+.02pp。深度/floor、remap/text uni等组合不能拆分为独立加分；旧EMA no-share为−.01pp，不能把no-share收益泛化。所有数字是单次观测差，不可相加预测下一组成绩，尤其不能把+.818777加到旧原版86.0上。5/5/3已回传T24，相对T23 best−.758127pp、末20−.675643pp；两套骨架decoder同时加深，不能单独分配各自贡献，不据结果自动改训练参数。


**2026-10-16选择、10月18日已完成T24：no-share native/T→S/S→T=5/5/3。**新增 `config/ntu60_xsub_joint/pretrain_madiff_text_sentence_sample_target_blend_noshare_dec5_st3.yaml`，相对T23仅decoder_depth由3改5，S→T保持3，remap2048/text uni0/权重1/.1/PT floor5e-4及其余设置保持。native/T→S/S→T=5/5/3；text decoder hidden512/output_norm=none、remap2048、text uni0、骨架uni.02、权重1/.1、A/person0、七句v3/sample_target_blend、PT400/双卡每卡32/accum1/seed0/lr.001/min_lr.0005；LP100/每卡64/accum1/lr.1/min_lr0/checkpoint399。独立输出根 `output_dir/ntu60_xsub_noshare_dec5_st3_h512_remap2048_textuni0_ep400_minlr5e4_pt32_lp64_seed0_20261016`，根目录main_pretrain.py→main_linprobe.py，端口10283/10284；历史单行PT&&LP命令及近期统计见[当前计划](D:/program/MacDiff/tools/vlm_pilot/STAGE1_TEXT_EXPERIMENT_PLAN.md)。原T23 YAML未改，从头训练、不resume/复用旧目标库。用户已回传完整T24日志，不再待结果。恢复骨架侧容量没有带来本次LP收益，两套decoder同时变化仍不能分离其贡献。


**2026-10-16最新结果：新no-share已完成，登记T23；原版T22/no-share T23全3层配对均已回传。**best86.280931625%@73、last86.080786054、末20均值86.089883586±.065868988；相对T22 best+.818777313pp、末20+.698083473pp，末20每轮都高；相对T21旧5层best仅+.060650170pp、末20反而−.044577872pp，不能说新组全面优于旧组。PT/LP逐epoch LR与T22完全相同，text uni全400轮为0、分支组合核验为1/.1。S→T末20=.010784990420（T21=.029032381955），native+.02*skel_uni=.016852515289（T22=.016801853122），因此LP改善不是伴随更低原生重建目标。运行定义按此前唯一待回传配置和用户上下文；日志无Namespace，实际层数/宽度、batch/seed、checkpoint/LP入口和服务器SHA仍未独立核验，不因用户后来要求根入口而反推此次实际LP入口。当前共23条Stage1记录，峰值最高T23、后期均值最高仍T21。完整统计和两份原始附件SHA见[Stage1结果](D:/program/MacDiff/STAGE1_EXPERIMENT_RESULTS.md)及[no-share分析](D:/program/MacDiff/tools/vlm_pilot/NOSHARE_NORMB_LOG_ANALYSIS.md)。本次只更新5份现有文档，不改模型/配置、不复制日志或新建归档目录、不启动训练。decoder蒸馏分支仍仅按已有实现记录，用户未回传其结果，不能推断是否运行。



**最新运行命令约定（2026-10-15用户明确要求）：以后新提供的PT/LP命令只使用仓库根目录主入口和主配置，不再使用 `handoff_artifacts` 下面的入口或配置。**本蒸馏方法PT用根目录 `main_pretrain_decoder_distill.py`，LP用根目录 `main_linprobe.py` 与 `config/ntu60_xsub_joint/linprobe_madiff.yaml`。T20源权重已从既有 `native_rotation_results_20261012/commands.json` 和 `clean_default_pt_lp.sh` 查到记录路径：`output_dir/native_official_dec5_ep400_bs32_seed0_20261012/pt/checkpoint-399.pth`；已填入新蒸馏YAML和说明命令，不再要求用户替换占位路径。历史实验命令保留其当时运行身份，不代表以后继续使用隔离入口。

**2026-10-15新增研究分支，用户已要求实现：随机encoder + 冻结原版decoder主体 + 随机可训练输出head。**独立入口 `main_pretrain_decoder_distill.py`、模型 `model/transformer_macdiff_decoder_distill.py`、配置 `config/ntu60_xsub_joint/pretrain_madiff_decoder_distill.yaml`，详细数据流/单行PT&&LP命令见 [decoder蒸馏说明](D:/program/MacDiff/tools/MACDIFF_DECODER_DISTILL.md)。没有修改原版、text、OSE、官方隔离代码或既有配置/入口，也未启动训练、生成权重或登记新实验ID。原有no-share随后于10月16日回传T23，见顶部最新结果。

新方法只从显式 `--decoder_checkpoint` 加载 `decoder_embed/decoder_blocks/decoder_norm/decoder_pos_embed/decoder_temp_embed`；encoder和 `decoder_pred` 不加载，沿用原版随机初始化并训练。冻结主体保持eval但保留对条件的autograd；输出head为256→12。每样本t在300–700（含端点）均匀采样，沿用1000步inverse-cosine和native `noise2joint`：一步预测epsilon、还原x0，仍只计算被mask位置的坐标MSE，保留.02骨架uniformity。person0、A归一化、mask.9、PT400/总64/lr.001/min_lr.0005、seed0，condition dropout显式0。默认decoder5，建议用已有T20；源权重路径已从T20历史启动命令查到，见顶部最新约定；没有直接访问服务器文件。严格核验主体键/形状和可读取的attention heads、扩散/归一化元数据；`--resume`只接受本方法完整checkpoint，恢复无需原教师文件。标准state_dict兼容既有LP。无Torch测试7项通过；CPU模型/梯度/更新/LP/恢复9项因本地无Torch跳过；原有累积5项和旋转runner8项通过，不能称CUDA/AMP/DDP已验证。完整记录见上述说明与 `tests/test_decoder_distill.py`。

**当前decoder3配对已完成：原版T22、no-share T23，seed0按既有命令身份，PT min_lr统一5e-4。**原版骨架decoder3；no-share的native/T→S/S→T三套decoder全部3层，用户了解MLP为4H后，将text_decoder_hidden_dim改回512，text_hidden_dim保持2048，即S→T hidden512/output_norm=none、remap hidden2048；缓存文本输入/输出仍512维。两组均PT400、每卡32、accum1、lr1e-3/min_lr5e-4；LP100、每卡64、accum1、lr.1；PT不旋转、归一化A、one_person=True、checkpoint399。no-share权重T→S=1、S→T=.1，沿用七句v3缓存和sample_target_blend。历史配对命令的原版PT/LP使用官方隔离入口，no-share PT使用文本代码；T23实际LP入口未从日志核验。历史命令见第6节，新命令遵守顶部根入口约定；T22/T23均已完成，不自动重跑。

**用户最新反馈“原版从5层到3层反而下降”：T22 best85.462154313%@90、last85.334788956、末20均值85.391800113±.045036580。**相对T20原版5层，best−.539786500pp、末20−.458515284pp，100轮LP均低于T20对应轮；PT末20总loss=.016801853122，比T20高15.290570%。但同时把PT min_lr从1e-5改为5e-4，不是深度单因素，不能直接确认3层本身更差或高floor是定因。新no-share应与T22配对。完整统计、两份原始附件与SHA见[Stage1最新结果](D:/program/MacDiff/STAGE1_EXPERIMENT_RESULTS.md:9)。本次仅更新4份现有文档，不改训练配置/模型，不启动实验。

**最新纠正：用户强调早期已把PT每卡64、双卡总128对齐论文，仍然偏低。**助手上一答仅列论文500/global128与T22的400/global64，遗漏了已完成T14/T17/T18，不足以解释整个复现差距。T14 decoder3/PT500/global128/floor5e-4=85.704755%；T17/T18按保存协议decoder3/PT500/global128/floor1e-5，不旋转85.638040%、旋转85.789665%；T19同T18 encoder将LP总batch256→128，口述85.89%。不要再次把总batch128、500轮、floor1e-5或旋转当成尚未试过且能保证补齐的方案，也不能认定floor5e-4是所有旧3层偏低的原因。当前是论文3层收益尚未复现、原因未定位；完整服务器运行身份仍缺，现有跨配置分数不构成深度单因素证据。本次补充只更新既有handoff/Stage1结果中的解释。

用户已明确确认：以后仅PT的min_lr统一用5e-4，LP保持0。第6节两条新3层命令符合该偏好，不因T22下降自动改回。10月13日5层/min_lr1e-5的旧no-share已回传完整PT400/LP100，登记T21：best86.220281456%@88、last86.147501239、末20均值86.134461457±.052393987。用户确认旧5层/decoder512/remap512/PT32 accum1/LP64/seed0/权重1/.1身份；不能把它当新3层/min_lr5e-4/remap2048的结果。新no-share已于10月16日完成登记T23，不能自动停进程或重复启动。

用户明确说“不跑seed1”，不要继续安排 seed1。旧的 no-share 1/.5、768宽度、LN/dropout、MAMP完整协议等只是历史计划或候选，均不是当前已选任务。不要根据旧文档自动启动。

**最新文本修订：用户已明确要求去掉text uniformity。**当前no-share decoder3配置lambda_text_uniformity=0.0，模型在0权重时跳过该正则计算；骨架lambda_loss_uni=.02保持。当前总loss为native+.02*skel_uni+T→S+.1*S→T。输出根新增textuni0，命令见第6节。服务器需同步模型文件与YAML；日志loss_text_uniformity=0代表关闭，不代表正交达成。T21和其他历史文本结果仍按当时text uni=.02身份保存，不能反推成无text uni；最新无text uni的no-share已回传T23，日志全程text uni为0。

用户随后要求简化训练日志：当前模型不再返回text_energy、text_batch_variance、text_valid_tokens，也不再执行对应统计。engine自动汇总模型返回的loss指标，因此控制台和epoch JSON的train_text_energy/train_text_batch_variance/train_text_valid_tokens均不再产生；保留分支loss、LR、GPU峰值/计时和目标库状态。历史T21/T12日志与数值分析保留原字段，不回写。现有测试改为直接检查真实目标RMS和S→T有效句数，不依赖已删除的日志字段。验证：AST检查仅返回5个分支loss指标、不调用被移除的统计、token校验保留；现有test_macdiff_text/test_structured_text_cache共40项，13通过、27因无Torch跳过，不称完整模型/autograd已验证。只同步模型文件即可应用日志调整，命令不变。

**最近问题已核实：MacDiff one_person=False将人物展开成N*M，各自前向、共享参数，最后平均扩散loss；uniformity仍只person0。**LP先分别编码，再平均人物/时间特征。没有改人物开关。MAMP官方NTU配置无one_person开关；NTU feeder保留两个输入人物槽（可能含零填充），PT也展开N*M分别编码并预测运动；其LP把人物、时间、关节一起平均成256维。

文本侧此前256→512改的是`text_decoder_hidden_dim`，影响S→T文本decoder主宽度；当前512，MLP中间层为4×512=2048。`text_hidden_dim`直接控制remap中间层，当前2048，即512→2048→512，不再乘4；`text_input_dim=512`是缓存句向量维度，`dim_feat=256`是骨架侧宽度。此前同日两个hidden都2048仅是未执行的中间计划，已被当前decoder512/remap2048替代。

MacDiff官方 native PT构造器默认 `one_person=True`，forward仅取 `[...,0:1]`，固定person0；不是随机选人，也不是自动选更活跃的人。基础LP仍读取两个人物槽，没有 one_person 构造参数。当前文本持久目标库明确要求 one_person=True，直接改为False会报错，需要先设计双人物目标库/文本对齐。原生PT可以另测False，但现有uniformity仍只用person0，不能描述成完全对称的双人训练。这个人物差异已确认；它对LP分数的贡献尚未验证。

固定来源：[MAMP PT构造器与forward](https://github.com/maoyunyao/MAMP/blob/09e6252a5c7d061c226395d25e6aa47a69e7e58d/model_mamp/transformer.py#L457)、[MAMP NTU feeder](https://github.com/maoyunyao/MAMP/blob/09e6252a5c7d061c226395d25e6aa47a69e7e58d/feeder/feeder_ntu.py#L67)。本地限制在 [engine_pretrain.py](D:/program/MacDiff/engine_pretrain.py:118)、[共享目标库](D:/program/MacDiff/util/shared_memory_text_target_bank.py:33)。本次更新既有no-share decoder3 YAML与模型零权重分支、记录与命令；没有启动实验。

### 2. 我们在做什么

当前主要有三条关联任务，最新执行优先级见第0节：

1. **纯骨架EMA教师改善学生LP。**新方法已经实现、尚无真实Torch/GPU验证和训练结果。以原生MacDiff为学生，完整骨架EMA教师同时提供InfoNCE和位置对应的多层特征目标；不用预训练教师，不评教师encoder。
2. **复现原生MacDiff的NTU60 XSub joint-only线性评估。**论文主表86.4%，第三方双3090口述86.2%，我们多组原生结果在85.4–86.0附近。逐项核对官方代码、论文、实际lr/batch/decoder/epoch/增强/LP协议，定位差距；目前没有证据确认是硬件或某个实现bug。官方AMP/累积本次已核实原本存在，见第0节。
3. **保留Stage1文本no-share历史实验与三组seed0复跑。**T23 best86.280932%，比配对T22 best高.818777pp、末20均值高.698083pp；尚非跨PT重复验证的稳定收益。T24的5/5/3和T25新一致性均已回传（T25口述）。当前讨论不继续文本交互；已选三组seed0复跑命令保留，不自动扩大实验矩阵。

旧Stage2 OSE/ReSA工作不是当前任务，背景保留在 [Stage2旧交接](D:/program/MacDiff/handoff_stage2_legacy_20260905.md) 与 [OSE设计](D:/program/MacDiff/MACDIFF_OSE_CROSS_INSTANCE_DESIGN.md)。用户现已明确授权骨架InfoNCE/教师多层对齐，旧“不做新增对比/跨样本”在此范围内失效；text uniformity仍关闭，骨架uniformity保留。固定CLIP目标、固定噪声条件打乱、方差/相似度诊断已经试过，不要重复当作新方向。v参数化、区域bias只讨论过，未实现。

#### 环境和边界

| 项目 | 当前信息 |
|---|---|
| 本地工作区 | `D:/program/MacDiff`，Windows PowerShell |
| 当前本地HEAD | `796964ad04005c273b5fd6f55004a2bd84c74e8f`，2026-10-19会话结束核验；不是本助手提交的本轮修改 |
| Git remote | `git@github.com:Wenrui-Shen/MacDiff.git` |
| 官方MacDiff比较提交 | `692888f1e2a4227216511aac6733dff31b587213` |
| 官方MAMP比较提交 | `09e6252a5c7d061c226395d25e6aa47a69e7e58d`（2023-09-22） |
| 用户服务器仓库 | `/home/user9/public3/swr/MacDiff` |
| 服务器数据路径 | `../data/MAMP/ntu/NTU60_XSub.npz`，预期train40091/test16487 |
| 历史服务器环境 | conda macdiff，Python3.8、Torch1.8.1+cu111、双RTX4090；最新实际版本待核验 |
| 访问方式 | 助手无SSH，用户自己执行Linux命令 |
| 本地验证能力 | 当前用`C:/Users/97537/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe`，有NumPy、缺Torch/PyYAML/timm；系统C:/Python314还缺NumPy。只做AST/标准库/标量控制流，未做CUDA/DDP/模型运行验证 |
| 本地数据 | 没有此次训练的实际NPZ/cache/checkpoint可直接验算 |
| 用户论文 | [macdiff.pdf](D:/program/paper/macdiff.pdf)；原页面渲染临时文件未保留 |

默认Windows sandbox helper曾初始化失败；本会话 exec_command 用 require_escalated 经自动审查后执行。写文档不需要安装依赖。其他解释器的依赖应现查，不要沿用旧环境结论。

用户用中文，通常双卡；训练命令要求**单行Linux、不加cd，PT和LP用&&连成一条**。没有授权助手启动/停止服务器任务、提交推送、清理数据或重生成缓存。用户嫌handoff_artifacts新文件太多：今后优先更新现有文档/结果表，不为每次检查新建目录、复制日志或增加备份。

### 3. 已完成的工作与结论

#### 原生实现审计与尾窗口恢复

对照固定官方提交的92个源文件，2026-10-11审计时55份规范化后一致，37份不同（包括文档、路径和扩展）。22项原生核心AST检查通过；diffusion全部15个文件、下游模型、LR函数、原生初始化/attention/MLP/损失/采样核心、feeder旋转函数等未发现改变原生算法的已证实差异。详见 [完整实现审计](D:/program/MacDiff/handoff_artifacts/native_implementation_audit_20261011/README.md) 与 [audit.json](D:/program/MacDiff/handoff_artifacts/native_implementation_audit_20261011/audit.json)。静态相同不证明CUDA前向/梯度逐位一致。

已知剩余差异：当前PT DDP用 `find_unused_parameters=False, broadcast_buffers=False`；官方为True及默认True。**用户只要求恢复梯度尾窗口，两个DDP开关没有改回。**原生模型以LayerNorm为主，关闭OSE/text时无对应额外参数/Queue，没有据此确认分差原因；文本辅助分支则确有额外参数和RNG消耗。LP的DDP设置未改。native forward公开默认mask/tau由.8/.75改为.9/-1，但engine显式传参；整数gather索引repeat→expand数学值相同，未做实际梯度等价验证；可选encoder activation checkpoint默认关闭。

**已按用户要求把PT不足一个累积窗口的尾梯度改回官方丢弃策略。**`engine_pretrain.py` native/text及OSE路径都 loss/=accum_iter，仅完整窗口末step；尾部到下轮zero_grad时丢弃。尾部不更新text bank/EMA或OSE teacher/Queue，AMP跳步的成功step保护保留。累积1没有尾窗口；标准NTU60双卡32/accum2共626microsteps，也是整除。验证：5项实际引擎语句的标准库控制流测试+原8项旋转runner测试，13/13通过；这不是Torch autograd/DDP测试。

原生AMP：PT默认True，LP默认False；仍用GradScaler。seed默认0，每rank使用seed+rank，Torch/NumPy初始化、worker Python/NumPy seed与sampler.set_epoch逻辑保持官方；cudnn.benchmark=True，不应承诺跨环境严格确定性。seed影响初始化、shuffle/增强、mask、扩散噪声/t、dropout等；新架构额外初始化消耗RNG，所以同seed不等于原生与文本逐位同轨迹。

#### 官方隔离副本与旋转脚本

为了干净官方对照，创建了 [upstream_692888f](D:/program/MacDiff/handoff_artifacts/upstream_692888f/ISOLATION_README.md)：92份官方源文件+manifest/README，只有3份例外：util/misc.py的torch._six.inf改同值math.inf；两份NTU60 PT/LP YAML仅迁移现有数据路径。官方DDP、模型、engine保持原样。**没有切分支或覆盖当前扩展代码。**从仓库根调用副本main文件，脚本目录优先导入副本模块；不要删除这个目录，当前命令还在用它。

已完成原生decoder3/PT500随机旋转两组的自动串行runner：no_rot PT→LP，再rot PT→LP，自动输出best acc/差值；唯一组间因素是PT random_rot。它校验配置、代码/数据身份和样本数，拒绝复用输出根目录，不自动resume。入口为 [script_pretrain_madiff_rotation_ablation.sh](D:/program/MacDiff/script_pretrain_madiff_rotation_ablation.sh)，实现 [run_macdiff_rotation_ablation.py](D:/program/MacDiff/tools/run_macdiff_rotation_ablation.py)，说明 [MACDIFF_ROTATION_ABLATION.md](D:/program/MacDiff/tools/MACDIFF_ROTATION_ABLATION.md)。**两组已经有结果，不再等待或重跑。**

当前 `script_pretrain_madiff.sh` 曾按双卡默认参考调整为2×32/accum1；官方原脚本是4×32。LP原脚本仍是四卡模板。实际入口/卡数/CLI决定batch和lr，不能只看文件名或YAML。

#### 日志核验与资料整理

完整维护 [EXPERIMENT_RESULTS.md](D:/program/MacDiff/EXPERIMENT_RESULTS.md) 和 [STAGE1_EXPERIMENT_RESULTS.md](D:/program/MacDiff/STAGE1_EXPERIMENT_RESULTS.md)。当前24条Stage1配置/成绩记录，不代表24次均有完整实参验证；T20第二次LP不另编号，T21是独立no-share5层PT+LP，T22是新原版3层/min_lr5e-4 PT+LP，T23是全3层/remap2048/text uni0的no-share，T24是保留T23其余设置的5/5/3组。旧文档正文有过期“待结果”和实验数量，读取顶部最新日期并以本交接为准。

handoff_artifacts保存官方源码副本、精确日志/统计、实现审计、历史交接和候选配置，不是自动启动的新实验。此前盘点约159文件/8.35MB，最近多数新增来自官方94文件副本。不要因“文件多”直接删除或移动，且不要继续无必要扩充。

### 4. 已有成绩：完成、口述与待测严格分开

accuracy单位为百分数，差值为百分点（pp）；末20标准差是population std，不是跨seed方差或置信区间。完整历史T01–T10查结果表，这里列当前决策相关记录。

| ID/组 | best LP acc (%) | 关键身份与证据 |
|---|---:|---|
| T01 历史原版 | 约85.86 | 缺完整args/日志/服务器SHA，不以现有默认补填 |
| T11 shared512 | 85.79 | 用户更正后的口述best；**不是85.75**；LP全文缺失 |
| T12 no-share512/A/1/.1 | 86.196021074 @75 | 完整PT400+LP100；native/T→S5、S→T5/512/none；PT2×32×accum2=128，LP2×64=128 |
| T13 native/B | 85.183164 @92 | 完整日志；batch/LR/预算也不同，不能当纯A/B对照 |
| T14 native/A/decoder3/PT500 | 85.704754983 @94 | 完整日志；PT2×64=128、LP2×128=256；min_lr实际5e-4；**用户确认LP已用checkpoint499** |
| T15 no-share骨架decoder3 | 85.84 | 口述；native/T→S3，S→T5；完整args与best/last口径未核验 |
| T16 T15再改text_decoder_depth3 | 85.86 | 口述；骨架仍3，S→T5→3；仅+.02pp，不证明收益 |
| T17 native decoder3/PT500/no_rot | 85.638039799 @64 | 用户summary JSON；正确min_lr1e-5，PT总128/LP总256 |
| T18 同协议/rot | 85.789665239 @88 | 用户summary JSON；与T17只改PT旋转，best+.151625439pp |
| T19 T18同checkpoint499、LP64/卡 | 85.89 | 口述；LP有效batch256→128，其他继承保存配置；best/last、epoch/实际seed未核验 |
| T20 隔离官方双卡默认 | 86.001940813 @95 | 完整PT400+两次LP100；5层/PT总64/LP总128，seed0重复；详见下文 |
| T21 10月13日no-share双卡5层 | 86.220281456 @88 | 完整PT400+LP100；用户确认5层/decoder512/remap512/PT32 accum1/LP64/seed0/1/.1；PT floor1e-5，旧计划已完成 |
| T22 原版decoder3/min_lr5e-4 | 85.462154313 @90 | 完整PT400+LP100；新配对原版已完成，PT总64/LP总128，seed0/checkpoint399按用户及既有命令身份；日志核验floor5e-4 |
| T23 no-share全decoder3配对 | 86.280931625 @73 | 完整PT400/LP100；末20 86.089883586±.065868988；decoder512/remap2048/text uni0/PT floor5e-4按既有命令；与T22配对完成 |
| T24 no-share 5/5/3 | 85.522804492 @89 | 用户确认身份；完整PT400/LP100；末20 85.414240689±.054243770；比T23 best−.758127pp、末20−.675643pp；已完成 |
| T26 native decoder3 repeat | 85.710819994 @73 | 完整PT400/LP100；末20 85.445172265±.054504224；比T22 best+.248666pp、末20+.053372pp；根入口/seed0身份按既定命令与用户说明 |
| T27 teacher v2（LP截至95） | 85.377244084 @90 | PT400/LP附件96轮；76–95均值85.220766640；同窗口比T26低.225012pp；v2由用户确认，不称LP100完成 |
| T28 teacher v3 AA1（LP截至89） | 69.802280459 @87 | 用户确认checkpoint-80后关闭AMP续训；PT附件81–399/LP90轮；70–89均值69.109958775；不是独立从头FP32 |

T12：last85.892771，末20均值86.000728±.061392；是文本扩展结果，不能标原版复现。它与T20的PT batch等不同，不能把差距全归文本收益。T13 B组缺同协议A控制。

T21：last86.147501239、末20均值86.134461457±.052393987；T21−T20 best+.218340643pp、last+.363901024pp、末20+.284146061pp。末20每轮均高于T20对应轮，但不是独立PT重复，不能据轮间std宣称跨seed显著。PT/LP逐epoch LR均与T20完全相同；完整Namespace、服务器源码/数据/checkpoint SHA及环境仍未独立核验。

T21末20 native MSE+.02 skel_uni=.014517515377，T20=.014573484324，仅低.384%；S→T最低.009536486467@116→末轮.029400900783，是最低的3.083倍，后期回升比T12更大而LP后期更高。不能把文本MSE当LP代理，也不能据训练loss回升确认过拟合或直接加dropout。原始PT/LP附件与SHA、完整比较更新在[既有no-share分析](D:/program/MacDiff/tools/vlm_pilot/NOSHARE_NORMB_LOG_ANALYSIS.md)，没有新建归档目录/复制日志。新3层/5e-4/remap2048已回传T23，见顶部最新结果。

T22 PT400/LP100连续且有限，LP逐epoch LR与T20完全相同，参数384060；末20 train CE=.334522595087（T20=.402430325676）、test CE=4.684069968878（T20=4.493198760020）。当前冻结特征在该LP协议下测试泛化较弱，但不能仅据LP训练/测试差距确诊PT过拟合。PT末20保持5e-4，较旧组末端50倍，后期loss平稳；无分支loss，不能拆解native MSE/骨架uniformity。若以后隔离深度可固定5e-4补5层，尚未选定/执行，不自动安排新矩阵。

T17/T18精确数据：

| 指标 | no_rot (T17) | rot (T18) |
|---|---:|---:|
| best | 85.63803979946768 | 85.78966523882603 |
| best epoch | 64 | 88 |
| last | 85.47428436686494 | 85.61984477443409 |
| last20 mean | 85.55767833815219 | 85.68413393778572 |
| last20 std | .042406355924753954 | .04483192483658093 |

best差+.15162543935835515pp，末20均值差+.12645559963353037pp。旋转有小幅正收益，单独不足以补齐复现差距。服务器输出根为 `/home/user9/public3/swr/MacDiff/output_dir/native_rotation_dec3_ep500_20261011_012631_490772`，下有no_rot/rot及各自pt/lp。保留 [原summary与分析](D:/program/MacDiff/handoff_artifacts/native_rotation_results_20261012/README.md)。只收到summary，不是完整epoch日志/run_manifest/Namespace；不能把脚本默认seed当已核验实际seed。

#### T20：官方默认PT与同seed两次LP已完成

按用户描述和所给命令：隔离官方692888f、native/A、decoder5、PT400/每卡32/accum1/seed0/no rotation/lr.001/min1e-5；checkpoint399；LP100/每卡64/accum1/seed0/lr.1/dist_eval。两次目录分别lp与lp1。

- PT epochs0–399完整、有限；last loss=.014521730446313231，末20 loss=.01457348432352457±.00008287206650271168；最后20轮lr1e-5由日志核验。
- 两次LP epochs0–99完整：**best86.00194081283784 @95；last85.7836002144404；末20均值85.8503153964306±.07206914375114691。**可训练参数384060，符合6400→60 FC。
- 两份LP100轮所有字段、原始字节完全一致，长度20024，SHA256=`f9ff33cdba3667ff1dd6fd84435fefccb021416b12b2111d610e9d1bd8032018`。用户明确确认两次seed0、不同输出目录、算法命令相同；新上传lp1再次核验仍同SHA。**不要继续无依据怀疑粘贴重复，也不要把T20重复LP另编号；T21是另外已完成的no-share训练。**
- 这证明本次同seed LP重复无可见差异；不测跨LP/PT seed方差，也不证明checkpoint文件逐位相同。T19改变batch，不是同配置随机重复，不能据85.79→85.89说LP随机性很大。
- LP90–99轮lr=0，但head BN仍train并更新running statistics，所以best@95正常；不能说FC95轮仍在更新。
- 距第三方86.2约.198059187pp。完整日志不含完整Namespace/数据/代码/权重SHA/运行版本，不能宣布所有实际参数已独立核验。

全文归档：[T20 README](D:/program/MacDiff/handoff_artifacts/official_dualgpu_results_20261013/README.md)、[summary.json](D:/program/MacDiff/handoff_artifacts/official_dualgpu_results_20261013/summary.json)，含pt.jsonl、lp1/lp2及重上传证据。不再等待这组结果。

第三方参考：用户确认另有人双RTX3090、原MacDiff默认配置、PT每卡32（默认accum1→总64）复现86.2。LP“估计默认每卡64→总128”仍是推测，没有链接/命令/日志；不把seed/各具体实参当已核验，也不增加项目实验ID。没有证据把我们差距归因4090/3090或双卡硬件。

### 5. MAMP与MacDiff对齐分析（已完成分析，未变更实验）

| 项目 | MAMP官方NTU60 XSub | MacDiff官方公开配置/脚本 | T20/10月13日no-share旧计划 |
|---|---|---|---|
| encoder | 8层/256/heads8/MLP1024、120帧、patch4×1 | 相同公共骨干 | 相同 |
| 骨架decoder | 3层/256 | 5层/256 | 5；no-share T→S也5 |
| PT epoch/warmup | 400/20 | 400/20，论文写500 | 400/20 |
| PT lr/min_lr | .001/.0005 | YAML .0005，脚本显式覆盖.00001，论文也是.00001 | .001/.00001 |
| PT日程 | warmup后cosine到400 | cosine到380、最后20轮保持floor | MacDiff日程 |
| 官方PT batch | 4×32=128 | 4×32=128 | 2×32=64，accum1 |
| 官方LP batch | 2×128=256 | 4×64=256 | 2×64=128，accum1 |
| PT AMP | 默认False | 默认True | True |
| PT rotation/mask | rot=True，motion-aware tau=.80，mask.9 | 默认不旋转，随机mask tau=-1；论文写旋转 | 不旋转，random mask |
| 人物/目标 | 保留两人物槽，预测patch标准化运动 | 默认person0，A归一化骨架扩散epsilon | person0/A |
| LP读出 | linprobe：平均人物/时间/关节→256；BN+FC，head无dropout | linprobe2：平均人物/时间，保留25关节→6400；dropout.3/BN+FC | MacDiff读出 |
| LP lr/schedule | .1/100轮，cosine到100 | .1/100轮，最后10轮lr0 | MacDiff日程 |

MAMP运动重建decoder和MacDiff带噪骨架+t+encoder条件/AdaLN扩散decoder任务不同；不能只因层数相同就宣称同容量/同协议。MAMP论文NTU60 decoder3/5均84.9，正文整体默认5；MacDiff论文表10 decoder2/3/4/5分别84.8/86.4/86.0/85.9，5层兼顾生成。**不能断言MacDiff主表86.4一定就是decoder3的完整训练配置。**

min_lr5e-4是1e-5的50倍，但1e-5是MacDiff论文/脚本明确选择，不能因为MAMP是baseline就认定错误。只改floor仍保留MacDiff末20轮平台，不等于完整MAMP日程；full cosine还需单独处理min_lr_epochs=0（LP对应0）。完整公平MAMP比较还涉及人物、目标、mask、AMP、batch、LP读出/dropout等，不能把多个因素一起改后只归因decoder或floor。

已有checkpoint可做256维LP读出对照，无需重训encoder；这改变整个head协议（包括dropout），不是纯维度消融。用户10月14日已选native/no-share全decoder3配对并统一PT floor5e-4，详见第6节。完整MAMP统一LP仍是候选，**不直接改one_person=False**。详细固定来源、配置与日程分析在 [现有实现审计的MAMP章节](D:/program/MacDiff/handoff_artifacts/native_implementation_audit_20261011/README.md:83)，未复制MAMP仓库。

### 6. 当前no-share方法与准确运行命令

#### 结构和维度：新会话不要混淆

固定缓存为服务器 `vlm_pilot/ntu60_xsub_clip_sentence_cache_v3`，逐原始sample/person生成global+六部位（head、torso、left_arm、right_arm、left_leg、right_leg），每句冻结CLIP pooled projected embedding512维。global `[N,2,512]`，local `[N,2,6,512]`，有效mask与sample manifest配套。训练sequence为**7个句向量**，不是77个BPE；77只是离线CLIP句长上限。front/side是同一个人的视角，不是两个人。

当前person0固定，空person0过滤、不换person1，flip=False。骨架120帧/patch4×1=750tokens，mask90%保留75，S→T skeleton memory为pooled+75visible共76tokens，encoder维度256。

| 参数/分支 | 含义 |
|---|---|
| text_input_dim=512 | 每个缓存文本句向量的特征宽度，不是句数/长度 |
| text_hidden_dim=2048 | residual text remap中间层宽度，512→2048→512 |
| text_decoder_hidden_dim=512 | 独立S→T Transformer主宽度512，MLP中间层2048，输出仍512 |
| text_decoder_depth=3 | 只改S→T text decoder层数 |
| decoder_depth=3 | 改native骨架decoder及其no-share T→S副本 |
| dim_feat=256 | 骨架encoder/decoder及下游特征形状，不是单独文本容量开关 |
| share_skeleton_decoder=False | native与T→S decoder分开；S→T始终独立 |

native/T→S为主宽度256、MLP中间层1024、末LN+Linear256→12（4帧×3坐标）；S→T当前主宽度512，输入512→512、骨架memory256→512、内部MLP2048、末Identity+Linear512→512。`output_norm=none`只去S→T末端LN，保留内部norm与projection；512→512 Linear是可学习输出头，不是恒等映射。旧256/LN预测512维独立Gaussian epsilon有维度瓶颈理论期望下界；这不是LP上限或有限batch硬界，旧256/LN vs512/none也不是单独LN消融。当前缓存与持久目标仍为512维，无需重生成缓存。

T→S和native共用noisy skeleton/t/epsilon/mask，对masked patches求MSE；T→S无直接skeleton encoder梯度是用户确认设计，不能擅自“修复”。S→T用独立text扩散t/epsilon、detached saved targets，通过skeleton memory训练encoder。no-share不等于切断全部文本影响，仍有T→S→remap→saved targets→S→T→encoder的间接路径。

sample_target_blend保存**目标值递推**，不是remap参数EMA：初始RMS(CLIP)，成功optimizer step之后用更新后的online remap计算 `saved_target=.9*old_saved_target+.1*remap(RMS(CLIP))`。global和有效local都更新；AMP跳步/不足累积窗口不更新；混合后不额外RMS。旧ema_remap的momentum.999不是当前模式。缓存约1.07GiB，person0目标约.535GiB加元数据。

当前总loss为native+.02 skel_uni+λT2S*T2S+λS2T*S2T，lambda_text_uniformity=0.0。历史启用的text_uni是同样本有效local之间cos²平均、含对角、不含global/跨样本；6句时理论下限1/6，但它无条件压低合理的部位语义相似性，用户据此要求关闭。保留可选计算以复现历史配置，0权重时不计算、日志loss_text_uniformity为0。text_energy/text_batch_variance/text_valid_tokens已按用户要求停止统计输出；历史分析里online variance与saved global energy的统计口径不同。uncond_ratio.1仅作用native；当前文本reader没有独立整样本condition dropout，attention dropout不等于条件丢弃。

#### 2026-10-14命令，10月15日原版已完成T22：全decoder3配对

两组均A、one_person=True、PT不旋转，PT400/每卡32/accum1/lr.001/min_lr.0005/warmup20/min_lr_epochs20；LP100/每卡64/accum1/lr.1/min_lr0/warmup0/min_lr_epochs10，加载checkpoint399。有效PT batch64、LP128；AMP、随机mask.9、骨架uniformity和其余增强保持前组，no-share关闭text uniformity。LP只加载骨架encoder，没有decoder层数参数。

原版复用既有[p1 decoder3配置](D:/program/MacDiff/handoff_artifacts/macdiff_reproduction_20261008/configs/p1_decoder3_pretrain.yaml)，其中历史min_lr1e-5由CLI显式覆盖；官方隔离源码/配置不改。no-share已更新既有[decoder3 YAML](D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_text_sentence_sample_target_blend_noshare_decoder3.yaml)：native/T→S/S→T三套3层，S→T512/none、remap2048、lambda_text_uniformity=0、骨架uniformity=.02、权重1/.1、七句v3/sample_target_blend。输出根包含h512_remap2048_textuni0，不复用此前计划的PT/目标库目录，不自动resume。T15/T16/T21保留历史运行身份，不从新版YAML反推。

从服务器仓库根执行，先同步更新的model/transformer_macdiff_text.py与no-share YAML并确保既有隔离入口/p1配置存在。PT/LP目录分离；助手未启动训练，不重复使用目标库目录。

原版（已完成T22，保留用于核对，不重复启动）：

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10277 handoff_artifacts/upstream_692888f/main_pretrain.py --config handoff_artifacts/macdiff_reproduction_20261008/configs/p1_decoder3_pretrain.yaml --batch_size 32 --accum_iter 1 --epochs 400 --warmup_epochs 20 --min_lr_epochs 20 --lr 0.001 --min_lr 0.0005 --mask_ratio 0.9 --motion_aware_tau -1 --seed 0 --output_dir output_dir/ntu60_xsub_native_dec3_ep400_minlr5e4_pt32_lp64_seed0_20261014/pt --log_dir output_dir/ntu60_xsub_native_dec3_ep400_minlr5e4_pt32_lp64_seed0_20261014/pt/tensorboard && OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10278 handoff_artifacts/upstream_692888f/main_linprobe.py --config handoff_artifacts/upstream_692888f/config/ntu60_xsub_joint/linprobe_madiff.yaml --finetune output_dir/ntu60_xsub_native_dec3_ep400_minlr5e4_pt32_lp64_seed0_20261014/pt/checkpoint-399.pth --batch_size 64 --accum_iter 1 --epochs 100 --warmup_epochs 0 --min_lr_epochs 10 --lr 0.1 --min_lr 0 --seed 0 --dist_eval --output_dir output_dir/ntu60_xsub_native_dec3_ep400_minlr5e4_pt32_lp64_seed0_20261014/lp --log_dir output_dir/ntu60_xsub_native_dec3_ep400_minlr5e4_pt32_lp64_seed0_20261014/lp/tensorboard
```

no-share：

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10279 main_pretrain.py --config config/ntu60_xsub_joint/pretrain_madiff_text_sentence_sample_target_blend_noshare_decoder3.yaml --lambda_text_to_skeleton 1 --lambda_skeleton_to_text 0.1 --batch_size 32 --accum_iter 1 --epochs 400 --warmup_epochs 20 --min_lr_epochs 20 --lr 0.001 --min_lr 0.0005 --mask_ratio 0.9 --motion_aware_tau -1 --seed 0 --output_dir output_dir/ntu60_xsub_noshare_all_dec3_h512_remap2048_textuni0_ep400_minlr5e4_pt32_lp64_seed0_20261014/pt --log_dir output_dir/ntu60_xsub_noshare_all_dec3_h512_remap2048_textuni0_ep400_minlr5e4_pt32_lp64_seed0_20261014/pt/tensorboard && OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10280 handoff_artifacts/upstream_692888f/main_linprobe.py --config handoff_artifacts/upstream_692888f/config/ntu60_xsub_joint/linprobe_madiff.yaml --finetune output_dir/ntu60_xsub_noshare_all_dec3_h512_remap2048_textuni0_ep400_minlr5e4_pt32_lp64_seed0_20261014/pt/checkpoint-399.pth --batch_size 64 --accum_iter 1 --epochs 100 --warmup_epochs 0 --min_lr_epochs 10 --lr 0.1 --min_lr 0 --seed 0 --dist_eval --output_dir output_dir/ntu60_xsub_noshare_all_dec3_h512_remap2048_textuni0_ep400_minlr5e4_pt32_lp64_seed0_20261014/lp --log_dir output_dir/ntu60_xsub_noshare_all_dec3_h512_remap2048_textuni0_ep400_minlr5e4_pt32_lp64_seed0_20261014/lp/tensorboard
```

两条命令也保存在[实验计划顶部](D:/program/MacDiff/tools/vlm_pilot/STAGE1_TEXT_EXPERIMENT_PLAN.md)。通过标准库/AST提取实际get_args_parser验证CLI、配置字段、decoder构造对应关系、公共参数与checkpoint399/目录匹配；最新关闭text uni后执行实际模型loss语句，验证0权重跳过计算、.02保留可选历史路径，骨架/T→S/S→T项保留。默认解释器无Torch/NumPy/PyYAML，bundled解释器有NumPy但无Torch/PyYAML；未做完整模型/CUDA/autograd/DDP运行，也未启动训练。新配对的公共PT超参数一致，但PT源码/DDP/辅助任务/RNG仍有差异；与T20比较又同时改变了深度和floor，相对T21还改变remap/text uni，不称单因素。

#### 10月13日旧5层计划：已完成T21，以下保留运行身份与命令

现有 [no-share YAML](D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_text_sentence_sample_target_blend_noshare.yaml) 的顶层默认lambda=.1/.1，命令**显式覆盖1/.1和batch32**。main_pretrain把顶层lambda写入model_args，只改model_args同名值可能被覆盖。

公共设置：A mean=[-.0058,-.1333,-.0246]、var=[.0206,.0805,.0218]；PT400/warmup20/min_lr_epochs20/wd.05/AdamW betas(.9,.95)，mask.9、motion_aware_tau=-1、diffusion epsilon/inverse_cosine tau1/1000steps、native condition drop.1、Gaussian joint noise[1,.005]、PT random_rot=False/source_rot=False/flip=False。LP frozen encoder/linprobe2/6400维/head dropout.3/BN+FC，100轮/warmup0/min_lr_epochs10/SGD momentum.9/wd0，训练旋转True、测试False。

PT用当前文本代码，LP用与T20相同的隔离官方入口；公共超参数对齐，**不是PT源码逐位相同**（DDP开关、辅助任务、RNG等仍不同）。相对T12，PT accum2→1把global128降到64，LP仍128；因此新组对照T20，不能把与T12差值称文本结构单因素。

以下是T21已完成组此前提供的命令，保留用于核对身份，**不要自动重复启动**。PT/LP独立输出目录，400轮载入399：

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10275 main_pretrain.py --config config/ntu60_xsub_joint/pretrain_madiff_text_sentence_sample_target_blend_noshare.yaml --lambda_text_to_skeleton 1 --lambda_skeleton_to_text 0.1 --batch_size 32 --accum_iter 1 --epochs 400 --lr 0.001 --min_lr 0.00001 --mask_ratio 0.9 --seed 0 --output_dir output_dir/ntu60_xsub_noshare_dec5_ep400_pt32_lp64_seed0_20261013/pt --log_dir output_dir/ntu60_xsub_noshare_dec5_ep400_pt32_lp64_seed0_20261013/pt/tensorboard && OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10276 handoff_artifacts/upstream_692888f/main_linprobe.py --config handoff_artifacts/upstream_692888f/config/ntu60_xsub_joint/linprobe_madiff.yaml --finetune output_dir/ntu60_xsub_noshare_dec5_ep400_pt32_lp64_seed0_20261013/pt/checkpoint-399.pth --batch_size 64 --accum_iter 1 --epochs 100 --lr 0.1 --min_lr 0 --seed 0 --dist_eval --output_dir output_dir/ntu60_xsub_noshare_dec5_ep400_pt32_lp64_seed0_20261013/lp --log_dir output_dir/ntu60_xsub_noshare_dec5_ep400_pt32_lp64_seed0_20261013/lp/tensorboard
```

同一命令已存 [实验计划顶部当前选择](D:/program/MacDiff/tools/vlm_pilot/STAGE1_TEXT_EXPERIMENT_PLAN.md)。本地检查过CLI/配置/Bash语法，未在Torch环境运行导入。需要同步隔离副本至服务器；目前没有完整服务器身份信息，不推断已经同步或未同步。

### 7. 当前卡在哪里、下一步做什么

当前优先卡点是**新骨架教师缺真实Torch/GPU验证与运行结果**，其次是历史复现缺完整实际运行身份；没有已定位的AMP/梯度累积错误。第0节为当前接手计划，以下历史审计仍有效：

- 原版3层T22、no-share全3层T23、5/5/3的T24均已完成；T21是旧5层，结果不能混用。decoder蒸馏分支未收到结果，不猜服务器运行状态。
- T20/T21/T22/T23/T24日志完整，但完整Namespace、服务器当前源码/未提交改动、NPZ/checkpoint SHA与软件版本缺失。旋转组只summary，T15/T16/T19只有口述；不能从库存YAML补齐实际参数。
- 论文86.4与公开默认5层/400/no rotation、表10不同decoder成绩存在口径差；第三方86.2无公开实参。现有证据不足以确定硬件、seed、DDP、person数、LP头或某个单因素解释差距。
- 本地没有实际权重/数据和Torch，无法端到端比较前向、梯度、AMP/DDP或数据身份。

接手顺序：

1. 先按用户下一条消息继续。若继续骨架教师，优先按第0节用T27现有checkpoint诊断AA尺度/方差/梯度，补真实测试与加载身份，不自动重训、不评教师。若问实验状态，只确认真实回传，**不自动启停进程、不跑seed1、不擅自重跑旧实验**；用户已另选三组seed0完整重复，原版3层已完成T26，另两组命令保留但尚未收到新成绩。
2. T24已完成日志核算和T23/T21比较；下一步按用户新指示。若继续核验身份，补actual args、checkpoint399、T23三套3层/T24的5/5/3、text decoder512/remap2048、PT总64/LP总128、seed0与实际LP入口，不能从当前配置补填；若收到新日志先查重复/运行身份，再决定是否新增ID。不要自动新增消融矩阵或把条件依赖假设当已证实原因。
3. 若继续定位原生复现差距，先补完整身份：Namespace、服务器git SHA+dirty diff、Torch/CUDA/cuDNN/timm、NPZ SHA/count/classes/有限值/人物空槽、checkpoint SHA/epoch/args与load_state_dict信息。优先利用现有encoder做用户选定的LP检查，避免无根据重训。
4. MAMP对齐若获选定，先明确目标是复现各自原协议，还是共同评价协议。LP256读出/head/dropout、骨架3层、min_lr5e-4、人物数等拆成可解释控制；双人物sample target须先实现/验证，不直接改False。没有选择前只保留候选，不穷举矩阵。
5. 512vs768、文本最终LN/condition dropout、全位置MSE、S→T v参数化、区域bias只是历史候选，不是当前下一步；不扩展文本分支。新骨架方法以专用配置为准。

500轮比较必须从初始化按500日程训练；400训练后追加100轮是continuation，不等于原生500日程。LP换decoder配置也不会改变已学encoder。更多PT seed只在用户选定后考虑，当前seed1明确取消。

### 8. 踩过的坑：绝对不要再踩

1. **不git clean/reset、不删除移动artifact/cache/checkpoint、不覆盖用户改动。**未跟踪文件不是垃圾，官方隔离目录还被命令引用。用户嫌文件多，优先更新现有记录，不再每次生成重复目录/日志/备份。
2. **不拿旧计划覆盖新选择。**T17/T18/T20/T21/T22/T23/T24已完成，T19=85.89已回传；T15=85.84、T16=85.86、T25=85.44已口述。旧1/.5、旧文本旋转计划、seed1、768都不是当前已选下一步。共28条记录，T20重复LP不另编号；T21是旧no-share5层，T22/T23为原版/no-share3层配对，T24为5/5/3，T25为nscale一致性，T26为原版3层完整seed0重跑，T27为teacher v2 PT400/LP附件96轮，T28为v3 AA1从80关AMP续训PT81–399/LP90轮。新noAMP三组repeat和AA0尚未收到可确认身份的结果；T27/T28与AA风险见第0节。
3. **PT和LP必须分目录。**T14共用目录使LP0–99覆盖早期PT checkpoint0–90；399/499未受影响。T14已确认用499，不再要求“补评499”。400轮末399、500轮末499，不能混用。
4. **已有目标库拒绝覆盖是保护。**历史 `FileExistsError: Refusing to reuse target bank/current.shared.json` 时换未使用的输出目录；不要删除保护或强制覆盖。T15后来已有成绩，历史报错不能当当前训练失败证据。
5. **完整PT resume要配对快照。**同次checkpoint-X.pth与checkpoint-X-target-bank.sqlite；current.shared.json不是完整恢复快照。仅能重建无对应模型的孤立target快照，不能覆盖已有配对。LP只要模型；strict=False迁移encoder不是完整PT恢复。改hidden/share/cache/目标模式等须独立实验。
6. **已完成的caption/cache不因猜测性能而重生成。**缓存身份绑定cache_clip_motion_text.py、structured_text_cache.py、cache_clip_text.py、clip_text_cache.py的实现SHA，无关格式改动也可能破坏身份；保留JSON/metadata/sidecar，不cat拼JSON数组。
7. **sample/person映射必须原始ID。**不能用shuffle batch行号，空person0不换person1，front/side不是两人，部位顺序不可变。v3 flip=False；渲染逐帧减person0 root，没展示完整世界位移，文本无法恢复缺失信息。
8. **7句向量不是77BPE。**旧BPE/ema_remap诊断不能不适配直接解释v3/sample_target/512。bank是.9历史目标+.1成功step后remap，不是.9固定CLIP；AMP跳步/尾窗口不更新，混合后不加RMS。
9. **配置文件名和YAML默认不等于真实CLI。**当前lambda1/.1显式覆盖；旧st02实际候选1/.5，不能登记.2。basePT YAML min_lr5e-4而脚本1e-5；T14实际高floor已日志确认。核验Namespace/LR，不能以脚本计划代替实际身份。
10. **参数维度不能混同。**text_hidden_dim改remap；text_decoder_hidden_dim/depth改S→T；decoder_depth同时改native/T→S；dim_feat影响骨架全链路。none只去S→T最终LN，所有方向仍有projection，512→512不是无头。
11. **不“修复”用户确认的梯度设计。**T→S没有直接encoder梯度；S→T有，no-share仍有间接文本影响。S→T一直独立，不称三套decoder都共享。关闭两个方向的lambda还要检查lambda_text_uniformity/remap未用参数与DDP False风险，未运行验证不能承诺安全。
12. **condition dropout不是attention dropout/detach/memory全mask。**若实现，每样本一次keep供全部层，在reader输出进入modulation前乘；全行key_padding_mask=True可能NaN。首轮不同时调drop和lambda；S→T dropout也改变encoder辅助梯度频率。
13. **模型归一化不从checkpoint继承。**A是(raw-mean)/sqrt(var)，feeder normalization=False不关闭它；mean/var普通属性不在state_dict，PT/LP要匹配。B=[-.0024,-.2132,-.0446]/[.0525,.1527,.0513]；#new不证明更正确/来自NTU120。A权重只换LP B是错配诊断，不是成对重训。
14. **batch不同不能当同seed随机重复。**global=每卡batch×卡数×accum；global相同也不保证local BN/更新频率/RNG相同。DDP buffer广播和累积都不是SyncBN。原MAMP LP2×128=256、原MacDiff4×64=256、当前双卡2×64=128。
15. **末20epoch std不是seed方差。**T20同seed日志一致已用户确认，别再无证据质疑重复粘贴；也不能据此宣称跨seed/跨硬件确定性或checkpoint字节相同。少量不同配置best差不能直接归因硬件。
16. **LP lr0仍可变准确率。**head BN running stats继续更新；best@95合理。LP首轮“72多/74.65”等不是最终分数；用户已放弃checkpoint150 LP，不再催跑。dist_eval16487补1样本最多约.0061pp，解释不了大缺口。
17. **loss变低不等于encoder/LP变好。**loss权重/占比不等于梯度比例；saved energy与online variance口径不同；text_uni含对角、非跨样本；Gaussian维度瓶颈是理论期望，不是有限batch阈值。
18. **不能乱做因果归属。**T12是文本扩展不是原生复现；T13 B和T14 highfloor多因素不同；T19改LP batch；T20与T18也同时改深度/轮数/batch/源码。NTU120 FT/PKU继承问题不是NTU60 LP定因。
19. **不要偷偷改数据使复现失去身份。**预处理主要继承官方，实际NPZ待验。NTU60单人第二槽零，NTU120单人槽可能复制；有效长度用坐标求和且crop假定前缀、原人物暂缺处理有潜在问题，但未量化。先验实物，不先修再称同协议。
20. **命令无cd、Linux单行、PT&&LP。**历史cd误拼进argparse导致没跑；CalledProcessError看前面rank真实traceback。长PowerShell命令注意Windows长度限制，写文本用单引号here-string防插值。
21. **没有actual args/SHA就明确未核验。**不要声称服务器未同步、正在训练或停了；不要称AST/标准库测试为CUDA/autograd/DDP数值验证。训练由用户执行，不安装依赖/远程启停/提交推送。

### 9. 资料索引、未提交状态与历史保存

新会话优先顺序：本handoff第0节→[新骨架教师实现与命令](D:/program/MacDiff/tools/MACDIFF_SKELETON_TEACHER.md)→结果表→当前实验计划顶部→对应分析README；旧交接只用于恢复历史设计，里面“待生成/待结果/下一步”已被本文替代。

- 结果：[Stage1表](D:/program/MacDiff/STAGE1_EXPERIMENT_RESULTS.md)、[总表](D:/program/MacDiff/EXPERIMENT_RESULTS.md)。
- 当前已选命令：[Stage1计划](D:/program/MacDiff/tools/vlm_pilot/STAGE1_TEXT_EXPERIMENT_PLAN.md)。
- 最新原生T22：[Stage1最新结果](D:/program/MacDiff/STAGE1_EXPERIMENT_RESULTS.md:9)；原生5层T20：[官方双卡日志分析](D:/program/MacDiff/handoff_artifacts/official_dualgpu_results_20261013/README.md)。
- 旋转T17/T18及LP64 T19：[旋转归档](D:/program/MacDiff/handoff_artifacts/native_rotation_results_20261012/README.md)，原summary/commands.json/recommended_lp_bs64.sh/clean_default_pt_lp.sh保留。
- 官方隔离源码：[ISOLATION_README](D:/program/MacDiff/handoff_artifacts/upstream_692888f/ISOLATION_README.md)。
- 当前实现/MAMP比较：[实现审计](D:/program/MacDiff/handoff_artifacts/native_implementation_audit_20261011/README.md)。
- 原配置/论文/数据审查：[18份官方配置审计](D:/program/MacDiff/handoff_artifacts/upstream_config_audit_20261008.md)、[复现报告](D:/program/MacDiff/handoff_artifacts/macdiff_reproduction_20261008/README.md)。该目录11份候选配置是静态准备，非已运行矩阵。
- 文本完整结构与候选：[no-share审计](D:/program/MacDiff/handoff_artifacts/noshare_config_audit_20261008/README.md)、[head/LN/dropout](D:/program/MacDiff/handoff_artifacts/noshare_config_audit_20261008/HEAD_LN_DROPOUT_ABLATION.md)、[文本几何](D:/program/MacDiff/tools/vlm_pilot/STAGE1_TEXT_GEOMETRY.md)。
- T12/T13：[日志分析](D:/program/MacDiff/tools/vlm_pilot/NOSHARE_NORMB_LOG_ANALYSIS.md)、[精确统计](D:/program/MacDiff/handoff_artifacts/noshare_normb_20261007/summary.json)。T14细节位于handoff_artifacts/decoder3_20261009。
- 缓存：[链路审计](D:/program/MacDiff/tools/vlm_pilot/TEXT_CACHE_CHAIN_AUDIT.md)；生成已经完成，旧“未生成”失效。
- 核心代码：[text模型](D:/program/MacDiff/model/transformer_macdiff_text.py)、[native模型](D:/program/MacDiff/model/transformer_macdiff.py)、[PT入口](D:/program/MacDiff/main_pretrain.py)、[engine](D:/program/MacDiff/engine_pretrain.py)、[共享目标库](D:/program/MacDiff/util/shared_memory_text_target_bank.py)、[SQLite目标库](D:/program/MacDiff/util/sample_text_target_bank.py)、[structured cache](D:/program/MacDiff/util/structured_text_cache.py)。
- 有效验证：[累积策略5项测试](D:/program/MacDiff/tests/test_pretrain_accumulation_policy.py)、[旋转runner8项测试](D:/program/MacDiff/tests/test_macdiff_rotation_ablation.py)，此前13/13通过；本次纯文档交接不重新声称运行GPU测试。
- 已存在的旧交接：[10月8日前](D:/program/MacDiff/handoff_artifacts/handoff_before_20261008_session_close.md)、[10月4日前](D:/program/MacDiff/handoff_artifacts/handoff_before_20261004_session_close.md)、[10月1日前](D:/program/MacDiff/handoff_artifacts/handoff_before_20261001_refresh.md)、[VLM旧交接](D:/program/MacDiff/handoff_vlm_pilot_legacy.md)。没有为本次再新建备份。

2026-10-14写交接前git status（已有改动，不能清除）：

```text
 M EXPERIMENT_RESULTS.md
 M STAGE1_EXPERIMENT_RESULTS.md
 M engine_pretrain.py
 M handoff.md
 M tools/MACDIFF_ROTATION_ABLATION.md
 M tools/vlm_pilot/STAGE1_TEXT_EXPERIMENT_PLAN.md
?? handoff_artifacts/native_implementation_audit_20261011/
?? handoff_artifacts/native_rotation_results_20261012/
?? handoff_artifacts/official_dualgpu_results_20261013/
?? handoff_artifacts/upstream_692888f/
?? tests/test_pretrain_accumulation_policy.py
```

此前会话结束仅重写handoff。10月14日随后按用户新指示更新既有no-share decoder3 YAML（全部3层、text decoder512/remap2048、PT min_lr5e-4、accum1、seed0、新pt输出路径），给出原版/no-share两条PT&&LP；之后旧5层组回传完整结果，登记T21并更新现有结果表/no-share分析/计划/handoff。用户随后明确关闭text uniformity，现有decoder3配置权重0、模型零权重时跳过计算，输出目录/命令追加textuni0；骨架uniformity保留.02，没有加dropout。10月15日新原版3层回传完整PT400/LP100，登记T22，更新现有两份结果表/计划/handoff；10月16日最新no-share也已回传T23，当前配对均完成。没有新增配置/备份/归档目录或复制日志，没有启动训练或Git提交推送。接手后以用户新指示和实际回传证据推进，不从历史候选自动扩展任务。
