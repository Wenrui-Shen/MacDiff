# 文本关系分布蒸馏骨架：固定CLIP版本与remap历史对照

## 2026-10-28新版本：固定CLIP关系教师，不再使用T→S或remap

用户明确“换成clip固定特征就不用T到S了”，以本段为最新方案。此前“仅替换KL文本来源、保留T→S”的建议被本次定义取代；旧remap/fullkey v2及其配置保留供对照，不把旧实验改名为固定CLIP实验。新组尚未启动，没有新LP成绩或成绩编号，当前记录仍33条。

### 方法与实现

- 新模型：[transformer_macdiff_clip_relation.py](/D:/program/MacDiff/model/transformer_macdiff_clip_relation.py)，直接继承原生MacDiff；新入口：[main_pretrain_clip_relation.py](/D:/program/MacDiff/main_pretrain_clip_relation.py)；新配置：[pretrain_madiff_clip_relation_decoder3.yaml](/D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_clip_relation_decoder3.yaml)。
- 身份 `fixed_clip_full_ema_gap_shared_queue_kl_v1`。仅原生骨架diffusion decoder3，encoder8/256不变；不存在remap、T→S/S→T decoder、文本person/position embedding或可训练CLIP/projection。可训练参数应与同配置原生MacDiff完全一致，额外EMA encoder冻结。
- masked学生75个可见token的第8层GAP256为q；完整原始骨架750token经EMA encoder得到k，EMA=.999。InfoNCE当前k作正样本、历史骨架队列作负样本，温度.07/权重1e-4。
- KL文本query与入队text key都直接取现有v3缓存中person0的**全局句向量**512维，detach/L2归一化；缓存原本的逐向量RMS缩放不改变余弦方向。不是六部位平均、remap输出或逐样本平滑目标。不运行CLIP网络，无额外文本特征训练/平滑更新。
- 与骨架队列配对的固定CLIP队列仍K65536，sample IDs/槽位/指针一致；骨架queue仍由InfoNCE与KL共用。同样排除所有历史self副本，KL方向为 `KL(stopgrad(P_CLIP) || P_S)`，两侧温度均.1/权重.1。512维文本与256维骨架各自算同K列关系，不要求向量维数相等。
- 目标：`L = L_native + .02*L_uniformity + 1e-4*L_InfoNCE + .1*L_relation`。没有T→S或S→T项，入口拒绝其非零权重；命令无需写两条“权重0”。日志直接不生成对应text diffusion loss占位项。总loss去掉了旧T→S分量，不能与旧总loss按绝对数值比较。
- 保留成功完整optimizer步后一次配对gather/enqueue，再更新EMA；跳步和不完整累积尾窗不更新。冻结CLIP队列仍随样本FIFO更替，骨架key仍随EMA变化；“固定”指文本向量来源，不是停止队列入队。
- 公共cache增加 `get_global_batch`；engine只对新模型的 `text_global_only=True` 走该分支，每批不读/传输六个局部句向量。旧文本训练仍走原get_batch。启动时仍保留原完整缓存校验和mmap文件结构，不把它误称为启动时完全不访问局部文件。
- `text_target_mode=fixed_clip`，不建立逐样本文本目标库，不调用remap更新。公共engine的update_text_target仅是空操作兼容钩子。
- LP仍原 `main_linprobe.py`/`linprobe_madiff.yaml`：只用在线encoder第8层，完整骨架输入、25关节×256=6400维；无teacher/文本/fusion。A归一化/person0/PT400/总64/学习率.001→.0005/noAMP/seed0→LP100/总128/.1→0不变。
- 该实验同时更换关系教师并删除T→S，因此是用户最新定义的新方案；若之后归因“仅替换文本来源”的收益，仍需另设保留T→S的控制组。CLIP分布仍需观察熵/近邻结构，不预先保证它一定更有区分度或提升LP。

### 验证与服务器同步

新增[测试](/D:/program/MacDiff/tests/test_clip_relation.py)11项，本地8通过/3因无Torch跳过。验证包括生产forward的NumPy执行、固定全局CLIP而非传入旧target、单次在线/完整key/原生decoder调用、精确loss、person0/无效行、共享队列配对及wrap/跳步/EMA时序、全局reader与engine路由、固定目标不建target bank、resume与旧remap方法隔离。3项真实Torch检查覆盖原生同名可训练参数且无文本模块、实际反向/冻结文本/EMA、checkpoint与LP兼容，需服务器运行；本地不声称autograd/GPU/DDP通过。

相关9模块共127项：72通过/55缺Torch跳过。实际PT/LP parser配合静态YAML子集核对CLI和checkpoint-399衔接通过；不是本地PyYAML/Torch训练。新文件与修改的公共文件均通过Python3.8语法检查。未安装依赖、启动训练或改动旧训练参数。

服务器同步新模型/入口/config/test，以及修改的 `engine_pretrain.py`、`util/person_text_cache.py`。新模型复用现有 `model/transformer_macdiff_text_relation.py` 的KL/gather和 `model/transformer_macdiff_text_relation_teacher.py` 的完整key encoder/InfoNCE，所以这两份及当前公共main/util也须为此前当前版本。旧版本依赖只导入定义，不构造其remap/文本decoder。

服务器先检查：

```bash
python -m unittest tests.test_clip_relation tests.test_text_relation tests.test_text_relation_teacher tests.test_pretrain_accumulation_policy -v
```

### 固定CLIP版本PT && LP命令

从头PT使用新目录，端口10329/10330；不能直接resume旧remap/KL队列，入口会拒绝跨模型/变更方法定义。先保持teacher/student温度.1，锐化温度另做后续消融。

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10329 main_pretrain_clip_relation.py --config config/ntu60_xsub_joint/pretrain_madiff_clip_relation_decoder3.yaml --relation_bank_size 65536 --lambda_contrastive 0.0001 --ema_momentum 0.999 --contrast_temperature 0.07 --lambda_relation 0.1 --relation_temperature_text 0.1 --relation_temperature_skeleton 0.1 --disable_amp --batch_size 32 --accum_iter 1 --epochs 400 --warmup_epochs 20 --min_lr_epochs 20 --lr 0.001 --min_lr 0.0005 --mask_ratio 0.9 --motion_aware_tau -1 --seed 0 --output_dir output_dir/ntu60_xsub_cliprel_fullkey_q65536_cl1e4_kl01_gap256_dec3_noamp_seed0_20261028/pt --log_dir output_dir/ntu60_xsub_cliprel_fullkey_q65536_cl1e4_kl01_gap256_dec3_noamp_seed0_20261028/pt/tensorboard && OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10330 main_linprobe.py --config config/ntu60_xsub_joint/linprobe_madiff.yaml --finetune output_dir/ntu60_xsub_cliprel_fullkey_q65536_cl1e4_kl01_gap256_dec3_noamp_seed0_20261028/pt/checkpoint-399.pth --batch_size 64 --accum_iter 1 --epochs 100 --warmup_epochs 0 --min_lr_epochs 10 --lr 0.1 --min_lr 0 --seed 0 --dist_eval --output_dir output_dir/ntu60_xsub_cliprel_fullkey_q65536_cl1e4_kl01_gap256_dec3_noamp_seed0_20261028/lp --log_dir output_dir/ntu60_xsub_cliprel_fullkey_q65536_cl1e4_kl01_gap256_dec3_noamp_seed0_20261028/lp/tensorboard
```


## 2026-10-29额外诊断：随机两个全局描述，比较CLIP与在线remap

用户要求每次随机取两个样本，打印样本/类别ID、特征首尾数值和相似度。独立入口 [diagnose_clip_remap_pair.py](/D:/program/MacDiff/diagnose_clip_remap_pair.py)，不训练模型、不生成诊断文件、不修改缓存或checkpoint；仅构造并严格加载checkpoint中的生产ResidualTextRemap，不构造骨架encoder/decoder。当前支持v3全局句向量以及sample_target_blend/ema_remap的残差remap；固定CLIP新方案没有remap，会明确拒绝，不临时构造随机网络。

默认从有效person0描述中不放回抽两个原始训练行，不按类别筛选或强制不同类别；每次使用OS新随机种子并打印实际seed，可用 --seed N复查。输出sample_id=train_<index>、原始sample_index、class_id_0based（与feeder一致）及class_id_1based、person_id=0、全局描述正文；train_<index>是仓库的样本名约定，不伪装成原始NTU骨架文件名。

每个样本打印CLIP输入和在线remap输出的维度/L2范数/RMS，默认首8维、末8维的带符号原数值和绝对值。两样本分别比较CLIP对CLIP、remap对remap的完整向量cosine、原尺度L2、单位向量L2、mean/max绝对差，并打印remap−CLIP的cosine变化；额外打印同一样本变换前后的cosine/距离。绝对值只用于展示，余弦始终使用完整带符号向量。

默认复用训练时冻结CLIP的unit-RMS缓存，remap输入因此与训练一致；不将其误称为本次在线CLIP推理。 --clip-model PATH可现场用同一个本地预训练CLIP重编码两条描述，严格核对manifest保存的CLIP文件SHA、禁止静默截断，并核对重新编码与缓存方向一致；额外打印未RMS缩放的原始投影向量。随后仍按训练定义unit-RMS→remap，后者为checkpoint的在线text_remap，不是EMA remap、sample_target_blend目标库或queue。

checkpoint/cache身份必须匹配。默认检查保存身份、文件尺寸和头信息，并读取选中行； --full-validation额外重新计算数据/cache哈希及全缓存行检查。类别仅从同一NPZ的y_train读取，不解压大x_train数组，不用于抽样或模型输入。实时CLIP模式额外需要原有Transformers/safetensors环境，不下载模型或安装依赖。

以下使用已完成T23的记录路径；检查其他remap组时将 --checkpoint改为该组实际存在的PT checkpoint。首先同步新诊断脚本及当前text模型与依赖。服务器仓库根一行（默认使用已有CLIP缓存、CPU即可）：

```bash
python -B diagnose_clip_remap_pair.py --checkpoint output_dir/ntu60_xsub_noshare_all_dec3_h512_remap2048_textuni0_ep400_minlr5e4_pt32_lp64_seed0_20261014/pt/checkpoint-399.pth --text-cache vlm_pilot/ntu60_xsub_clip_sentence_cache_v3 --data-path ../data/MAMP/ntu/NTU60_XSub.npz --device cpu --show-dims 8
```

现场重新运行本地预训练CLIP后接同一remap（每次仍随机抽两个；需要已有CLIP运行环境）：

```bash
CUDA_VISIBLE_DEVICES=0 python -B diagnose_clip_remap_pair.py --checkpoint output_dir/ntu60_xsub_noshare_all_dec3_h512_remap2048_textuni0_ep400_minlr5e4_pt32_lp64_seed0_20261014/pt/checkpoint-399.pth --text-cache vlm_pilot/ntu60_xsub_clip_sentence_cache_v3 --data-path ../data/MAMP/ntu/NTU60_XSub.npz --clip-model /home/user9/public3/swr/models/clip-vit-base-patch32 --device cuda:0 --show-dims 8
```

新增 [测试](/D:/program/MacDiff/tests/test_clip_remap_pair.py)10项：本地9通过、1因无Torch/timm跳过，覆盖随机/可复查/不同原始ID、person0不换人、标签与描述索引、原值/绝对值首尾展示、带符号完整余弦/距离、checkpoint在线权重与缓存身份隔离；真实checkpoint/cache/生产remap/CLI只读端到端项待服务器。Python3.8语法和CLI已检查；未声称真实CLIP/GPU运行通过。只有一对样本，不能据此确认总体坍塌、类别可分性或LP收益；本次无新训练成绩/实验编号。

### 2026-10-29服务器实测：T23两样本，描述相近且与类别主动作不符

用户在 `skeleton_vlm` 环境执行上述现场CLIP命令，输出正常结束。此前 `macdiff` 环境的粘贴traceback停在Transformers导入行、缺最后异常，不能确定是缺包还是版本不兼容；本次切环境后真实CLIP→RMS→生产在线remap链路已跑通。生产remap的autocast FutureWarning只是弃用提示，本次未因它失败，不为此修改兼容旧Torch的训练代码。本次真实运行证据不等于此前缺Torch的端到端单测已经执行。

checkpoint为T23的epoch399、model.transformer_macdiff_text.Transformer；seed=258876072，person0有效40091个，cache_identity_sha256=dbd149eeab52f57b5849680488256dd62302634a29ac9cb0fe6585c947aecaa5。读取的是全局句向量与在线remap；不是关系KL旧主线A的epoch38/39 checkpoint，不将该结果归给旧KL组或新固定CLIP组。

| 样本 | 类别0起点 / 1起点 | NTU60官方类别名 | 本次全局描述内容 |
|---|---|---|---|
| train_17540 | 29 / A030 | type on a keyboard（敲键盘） | 自然向前行走、交替迈腿和摆臂 |
| train_23232 | 51 / A052 | pushing（推人） | 自然向前行走、交替迈腿和摆臂 |

类别名称核对 [NTU官方动作表](https://rose1.ntu.edu.sg/dataset/actionRecognition/)，本仓库 `data/ntu/seq_transformation.py` 按原标签减1得到y_train的0–59。服务器真实原始骨架/渲染未提供，因此只确认“类别主动作与描述明显不一致”的核查线索，不直接认定数据索引错位或渲染错误。两条描述几乎同义，CLIP本应给出高相似度，不能把异类样本的这一对高相似度直接归因于remap坍塌；VLM可能遗漏关键手部/交互动作，需核对对应原始行的GIF及六部位文本。现有renderer逐帧root居中，描述中的“向前”也不能自动解释成已保留的世界坐标位移证据。

| 指标 | CLIP unit-RMS | remap unit-RMS |
|---|---:|---:|
| 两样本cosine | .965108963 | .965999933 |
| 原尺度L2 | 5.977325708 | 5.900513876 |
| 单位向量L2 | .264162969 | .260768353 |
| mean absolute difference | .210107138 | .119695779 |
| max absolute difference | .847285926 | 2.388984621 |

cosine只提高 **.000890970**，单位向量L2下降约1.285%，这对样本的相对方向差别基本保持。不能用一对文本证明全局几何保留或坍塌。现场CLIP与训练缓存cosine均打印1.000000000，最大逐维差分别2.861e-6/3.815e-6（unit-RMS尺度），支持本次CLIP模型、描述行及缓存方向吻合；不把打印1当数学上逐bit相同。

同一样本CLIP→remap cosine分别.227769266/.276866859，说明在线remap大幅改变坐标方向；它没有要求逐维保持CLIP坐标，单凭这种跨空间cosine低不能判语义破坏。相互cosine近似不变也不排除其他样本关系变化。remap第504维约−7.0682/−7.020835，各占该样本总平方能量约9.758%/9.627%；RMS≈1只约束全向量均方，不限制单维在[−1,1]，因此不是归一化错误。共同较大坐标提示可进一步看公共分量与跨样本逐维方差，但一维或一对样本不能确定有效秩/坍塌。

本次更直接的线索是全局描述缺少类别区分信息；换成固定CLIP也会保留这两条描述的高相似度。优先检查train_17540/train_23232原始行与可视化、global/local文本对应关系，再讨论是否需要总体诊断；未重生成描述、改缓存/模型/损失或新启动训练，不新增结果ID，仍34条训练结果记录。复查同一对时在原诊断命令追加 `--seed 258876072`。

原输出：[服务器现场CLIP/remap两样本诊断](/C:/Users/97537/.codex/attachments/c519670f-0d42-4264-b69a-c554f84161bd/已粘贴的文本.txt)，SHA256=`d246d1234ecfb949372595d8dde5a321dbd5cff72960dc8fd04d1ff4f12f0a37`；不复制附件或新建结果目录。

### 2026-10-29第二对实测：描述不同，remap明显压缩该对方向差异

用户追加现场CLIP输出，仍为T23 epoch399在线remap、同一cache identity；seed=699171498。train_27667/类别7(A008坐下)的global描述为弯腰、屈膝、降低臀部；train_4325/类别27(A028打电话)描述为抬右臂与头部、腿相对静止。NTU类别见[官方动作表](https://rose1.ntu.edu.sg/dataset/actionRecognition/)。这次两条描述并不相同：前者抓住下坐的身体运动，后者与电话类的局部姿态可以相容，但没有明确手靠耳/维持姿势等更强证据，未看原始GIF不直接确认完整语义准确性。

| 配对 | CLIP cosine | remap cosine | remap−CLIP |
|---|---:|---:|---:|
| 第一对：两条global均描述自然行走 | .965108963 | .965999933 | +.000890970 |
| 第二对：降低躯干/臀部 vs 抬右臂/头部 | .908754881 | .974887726 | **+.066132845** |

第二对原尺度L2从9.666178064降至5.070993045，单位向量L2从.427188762降至.224108341，缩小 **47.5388%**。现场CLIP与缓存cosine仍打印1，最大逐维差≤2.861e-6。该对的区别进入remap前存在，经过remap后明显缩小；而且两对相似度的排序反转（原CLIP第一对更近，remap第二对更近），因此当前两对证据已不能泛化成“remap整体保持了CLIP关系”。仍不能仅凭两对证明总体坍塌、解释LP原因或归给另一个KL checkpoint。

“不同动作却相同描述”与“不同描述却相近特征”需分开：前者发生于骨架渲染→VLM文本阶段，remap不会改写文本；后者发生于CLIP/remap表示阶段。本次更直接展示后者。当前v2 prompt要求每人一句global和六句local，不输入动作标签，也没有要求类别间文本必须唯一；稀疏骨架缺少物体/手指细节、简短摘要遗漏关键阶段、VLM泛化或误述，都可能导致异类描述相近。当前global/local验证器检查字段、人物数量/索引、六部位、非空及占位符，不校验动作语义是否符合真实视频，也不按类别验证文本区分度，所以accepted不等于描述动作准确。

第一对“A030敲键盘/A052推人→两条自然行走”仍是需核对实物的较强异常线索，不能用“类别与描述本来不同”合理化。当前代码renderer用x_train[sample_index]、caption沿用sample_index、cache按sample_index写行；未发现这里把ordinal误当原始ID的直接证据，但尚未检查服务器旧GIF/JSON/NPZ，不能排除旧产物对应错误。CLIP现场与缓存一致只验证文本→特征，不证明骨架→文本语义正确。现有root居中与少量帧也可能影响可见信息，不能由本次输出定位具体哪个环节出错。

T23的remap只按去噪目标学习，没有显式保持CLIP全部余弦关系；残差MLP若产生较大共同分量，RMS归一化只固定尺度，不能阻止角度集中。这是与结果相容的机制候选；四个已见样本第504维均约−7是公共分量线索，不独立构成坍塌证据。优先核对已见异常样本的原始行/GIF及global/local，并在需要时使用更多配对的前后相似度/中心化统计量区分描述区分度与remap压缩，不自动重训或重生成全量描述。

原输出：[第二对现场CLIP/remap诊断](/C:/Users/97537/.codex/attachments/6237e2cf-5cf7-4554-9fb2-af46ad459674/已粘贴的文本.txt)，SHA256=`7b66bf4a5eb07ce61eddb77813142627e7d89977518ee0791d4b614341534f06`。复查在原命令追加 `--seed 699171498`。仍34条训练结果，无新增训练ID、代码/缓存修改或训练启动。

### 2026-10-29全量语料审查：上游文本质量问题已获全量证据

用户提供按类汇总文件并要求实际检查类内/类间错误。已完整解析40091样本/60类，检查全部47608个人物条目的七句结构；主要统计按当前训练使用的person0，未把person1混入分母。精确字符串比较，不改大小写/标点/空白：global有2521个跨类同文组、涉及32948样本（82.18%）；global＋六local七句整套有2522个跨类同文组、涉及10345样本（25.80%）。重复比例不是经视觉确认的错误率。全局泛泛运动句覆盖905样本/53类，一套自然行走七句覆盖366样本/20类，后者包括踢东西、单脚跳、跳起等类别。具体错配还包括键盘类下蹲/跳舞/行走、站起类降低身体、摇头类head全视频不动、拥抱双方镜像跳舞等。

[完整审查报告：60类表、代表性ID/原文、统计口径与原因边界](/D:/program/MacDiff/tools/vlm_pilot/NTU60_XSUB_CAPTION_AUDIT.md)。[统计与全部重复组](/D:/program/MacDiff/handoff_artifacts/caption_text_audit_20261029/summary.json)。与前两对不同，本次文本重复/泛化已是全量统计事实；但本地仍无真实骨架/GIF，不能直接给VLM语义错误率或断定渲染/索引错位。现场CLIP匹配缓存只核验文本→特征，无法核验骨架→文本是否正确。固定CLIP、读取改动和KL都不能自动恢复句子未编码的动作差异；本次没有重生成描述/修改缓存/模型或启动新训练，仍34条训练结果。

## 2026-10-28主线A训练片段：数值稳定，但文本关系分布接近均匀

用户提供[控制台原片段](/C:/Users/97537/.codex/attachments/394d9b20-3a11-4400-8356-a5e3f742624e/已粘贴的文本.txt)，SHA256=2b8b76511d52947dcd4337f0fe22991dbb88dc6b26715e0cc9fddeeb64ae9caa。共79行、17条进度记录：epoch38的step160–625（每40步及末步）和epoch39的step0/40/80/120，另有epoch38结束的Averaged stats、资源信息及随机预测打印。没有完整PT/LP、实际Namespace或服务器源码SHA，不由片段推断后续进度，不新增成绩ID。输出路径与主线A fullkey命令一致，CL/KL指标均实际出现。

### 数值及权重检查

所有显示的数值有限，未见NaN/Inf。SmoothedValue默认括号前为最近20次更新的中位数，括号内为本epoch累计均值；分项中位数不能直接相加验证总loss。以epoch38最终Averaged stats的括号均值检查：

| 项目 | 原始值 | 权重后约值 |
|---|---:|---:|
| S→S diffusion | .0207 | .0207 |
| T→S diffusion | .0259 | .0259 |
| skeleton uniformity | .0232 | .000464 |
| relation KL | .0062 | .000620 |
| InfoNCE | 9.7559 | .00097559 |
| 合计 | — | .04865959，打印total=.0487 |

差约4.04e-5，处于四位小数打印舍入范围，未见loss组合错误。CL加权值约占打印total的2.00%，KL约1.27%，这不是梯度占比。text_uniformity=0是该分支权重关闭后的占位值，不是文本特征方差/关系均匀性的实测。

满队列65536、平均非self参照65534.864、valid_fraction=1，说明该片段关系行都有足够参照；相差约1.136是按sample_id排除历史自身副本造成的正常数量差异。InfoNCE均匀logits参考约ln(65535.864)=11.09035，实际9.7559低于该值；正样本概率几何平均约exp(-9.7559)=5.80e-5，约为均匀参考1.526e-5的3.80倍，不表示检索准确率。CL约9.7不能仅凭绝对数值判异常，positive_cosine=.1636也需结合负样本分布，不能单独诊断对比学习失效。

### 主要疑点：文本teacher接近最大熵

epoch38结束平均teacher entropy=11.0899，student entropy=11.0840；参照数K≈65534.864对应lnK≈11.0903376。由打印均值估计H_T/lnK≈.9999605，显示为1.0000是四位小数舍入，不是数学上恰好均匀。平均文本分布相对均匀分布的KL约lnK−H_T≈.000438 nats（使用log平均K近似平均logK，且受四位小数精度限制）。该片段所有teacher entropy窗口中位数约11.0898–11.0900，持续接近最大熵。

这说明在当前teacher温度.1和队列下，文本相似度分布的总体区分度很弱。小的relation loss=.0062不能当作已成功学到文本语义关系；若P_T接近均匀，KL(P_T||P_S)可能主要推动骨架关系分布变平。lnK−H_S≈.00634是KL(P_S||U)，与实际目标KL(P_T||P_S)方向不同，不能直接把二者等同。熵不足以证明remap/骨架表示坍塌，也不能仅凭此判定温度、文本同质性或记忆库时滞哪个是原因。

后续优先诊断：同一批样本/共同锚点上比较原始CLIP、在线remap、逐样本平滑目标和文本queue的特征方差/余弦均值与跨度/近邻间隔，区分本身同质、remap集中、队列陈旧和温度过软；在同一份logits上离线比较teacher温度.1/.05/.04的熵及top-k概率质量，学生温度先保持.1。仅提高lambda_relation不会让teacher目标更有区分度。ReSSL的[第4.1节和温度消融](https://arxiv.org/pdf/2107.09282)支持适度锐化teacher的动机，但其图像同模态/EMA设定与本项目不同，不直接迁移其最优温度或用其坍塌结果诊断当前模型。此前两侧.1是本项目起始配置，当前日志提示需要检查该校准，不自动更改正在运行的温度/权重或启停训练。

### 其他显示值

原生forward_loss随机打印的是mask*pred、mask*target，出现整行0通常是可见位置被打印mask置零，不能据此认为输出坍塌。此处target为epsilon，打印值约正负几的量级本身正常。epoch38约187.33s/.2992s每步，峰allocated12796.9MiB/reserved13426MiB；片段max mem稳定。文本target读取.58s/更新2.05s合计约占该epoch1.4%，不是这段显示的主要耗时。epoch39首步data约2.17s拉高ETA，随后回到约.30s，符合epoch开头加载开销，不能拿首步ETA估完整轮耗时。

**2026-10-27实现边界更正：**用户追问一级为何不按原论文实现。当前B是工程上为保持EMA参数同形而采用的“细patch embedding均值合并＋随机70%”近似，不能把它称为原论文独立大patch投影已实现。真正12帧独立投影尚未实现，其与4帧学生投影的初始化/更新规则仍需明确；不是EMA原则上不允许大patch。默认A保持原生MacDiff＋完整EMA key全局InfoNCE＋T→S＋全局关系KL，见下方v2定义。此更正不修改模型、默认配置或启动训练。

## 2026-10-27可选一级消融：大时间patch＋随机mask70%，默认仍完整key v2

用户澄清实验层级：主方案是完整骨架EMA key；一级额外消融只借用“大patch＋教师随机mask70%”；运动差分输入与Motion Compass掩码作为这项消融之上的二级消融，分别评估及可选组合。**默认入口/config仍是下方完整key v2，不自动切到本消融。**

| 层级 | 教师输入 | 时间尺度 | 教师mask | 状态 |
|---|---|---|---|---|
| 主方案A | 原始骨架 | 4帧/细token | 无，750 token | 已实现，默认 |
| 一级B | 原始骨架 | 3个相邻4帧embedding合成12帧粗token | 随机70%，75 token | 本次已实现，独立配置 |
| 二级B+M | 在B上改为运动差分 | 同B | 随机70% | 候选，未实现/未启动 |
| 二级B+C | 在B上只改mask优先级 | 同B | Motion Compass-aware | 候选，未实现/未启动 |
| 二级B+M+C | 在B上同时加入运动输入与MCAM | 同B | Motion Compass-aware | 候选，未实现/未启动 |

二级不默认混入一级，不能把一级结果归给运动输入或MCAM。用户没有要求本次实现这些二级项，后续需按论文公式复核其输入/方向强度/概率采样，再独立提供配置。学生侧、T→S、global KL及队列/LP都保持当前定义。

### 论文依据与本项目适配边界

已读取本地ACA²Net原文并渲染核对PDF第4页图2/教师式5–7及第6页IV-B：论文90帧、学生时间patch3/mask90%，教师时间patch9/mask70%，两路都是75个token，教师实际输入为运动差分且使用MCAM。原文第10页解释教师70%mask是效果与计算量的折中。原文：[本地PDF](/D:/program/paper/Asymmetric_Context-Guided_Adaptive_Alignment_Network_for_Skeleton-Based_Action_Recognition.pdf)。

当前MacDiff固定120帧、学生patch4，因此仿照3倍时间尺度可得：学生120/4×25×.1=75；教师120/12×25×.3=75。**75个教师token是更粗的时间token，不是75个原网格细token。**它们名义覆盖225个细patch、约为学生可见坐标位置的3倍，但少于完整教师750细patch；不能保证压缩后的信息质量一定更高。

独立的12帧Conv/Linear投影与学生4帧投影参数形状不同，不能直接套用当前EMA。为保留同架构EMA、只改变教师视图的受控适配，本版使用：
1. 用同形EMA的4帧patch投影得到[B,30,25,D]。
2. 加原时空位置embedding；同一关节的相邻3个时间patch连同位置embedding取均值，得到[B,10,25,D]。
3. 展平250个粗token，逐样本随机保留75个，再进入8层EMA Transformer和norm/GAP/L2。
4. 该粗key作为InfoNCE正样本并进入原共享骨架queue，与配对文本同步入队。

这等价于一种由细patch权重约束的12帧线性投影，不是ACA²Net独立训练的9帧MotionEmbed；平均可能损失细节，本消融不冒称原样复现。相邻组不跨关节，位置也按组平均，无新可训练权重、无需resize EMA参数。教师mask与学生mask独立，教师保留视图不保证包含学生全部可见位置。key attention从750降到75 token，attention矩阵元素约为原教师的1/100；4帧embedding仍计算完整网格，不能把该比例当整网加速或显存实测。

### 配置、恢复与验证

沿用同一模型/入口，新增[可选配置](/D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_text_relation_teacher_coarse70_decoder3.yaml)和[专用测试](/D:/program/MacDiff/tests/test_text_relation_teacher_coarse.py)。一级身份 `relation_teacher_version=coarse3_mean_random70_ema_gap_shared_queue_kl_v3`，明确绑定 `teacher_view=coarse_mask`、`teacher_temporal_group=3`、`teacher_mask_ratio=.7`；完整key v2仍默认full/group1/mask0，旧model_args不添加冗余新字段，保持旧完整key恢复兼容。跨full/coarse的resume会被完整model_args检查拒绝，不能混装两种历史key队列。

除此之外参数不变：A/person0、encoder8/native decoder3/T→S decoder3、query可见token GAP256、global KL/配对65536队列、CL1e-4/温度.07、EMA.999、KL.1/两侧温度.1、uni.02、T→S1/七句/remap2048/no S→T。PT400/总64/floor5e-4/noAMP/seed0→原第8层LP100/总128/floor0/6400维，读本组checkpoint399。

本地一级新增7项：5通过、2因缺Torch跳过；一级+v2+v1+累积37项为28通过/9跳过。通过含实际生产粗粒度forward的NumPy执行、250→75随机选择、同关节时间分组/平均位置、默认full保持与跨视图resume拒绝。真实75query/75key的反向、EMA投影形状/更新及checkpoint2项待服务器执行；未验证GPU/DDP。相关整合回归与静态配置检查见handoff，不由NumPy或mock声称autograd/NCCL通过。

同步更新 model/transformer_macdiff_text_relation_teacher.py、main_pretrain_text_relation_teacher.py、tests/test_text_relation_teacher.py，加上本次新配置/测试；此前v2的基类关系模型/入口/公共engine/main及缓存支持文件仍须为当前版本。服务器可先运行：

```bash
python -m unittest tests.test_text_relation_teacher_coarse tests.test_text_relation_teacher tests.test_text_relation tests.test_pretrain_accumulation_policy -v
```

以下为**可选B**的独立PT&&LP命令，端口10327/10328、新目录；默认主方案A仍使用下方v2配置及10325/10326，助手未启动任一训练、不新增成绩编号：

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10327 main_pretrain_text_relation_teacher.py --config config/ntu60_xsub_joint/pretrain_madiff_text_relation_teacher_coarse70_decoder3.yaml --teacher_view coarse_mask --teacher_temporal_group 3 --teacher_mask_ratio 0.7 --relation_bank_size 65536 --lambda_contrastive 0.0001 --ema_momentum 0.999 --contrast_temperature 0.07 --lambda_relation 0.1 --relation_temperature_text 0.1 --relation_temperature_skeleton 0.1 --lambda_text_to_skeleton 1 --lambda_skeleton_to_text 0 --disable_amp --batch_size 32 --accum_iter 1 --epochs 400 --warmup_epochs 20 --min_lr_epochs 20 --lr 0.001 --min_lr 0.0005 --mask_ratio 0.9 --motion_aware_tau -1 --seed 0 --output_dir output_dir/ntu60_xsub_textrel_coarse3mean_m70_q65536_cl1e4_kl01_gap256_dec3_noamp_seed0_20261027/pt --log_dir output_dir/ntu60_xsub_textrel_coarse3mean_m70_q65536_cl1e4_kl01_gap256_dec3_noamp_seed0_20261027/pt/tensorboard && OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10328 main_linprobe.py --config config/ntu60_xsub_joint/linprobe_madiff.yaml --finetune output_dir/ntu60_xsub_textrel_coarse3mean_m70_q65536_cl1e4_kl01_gap256_dec3_noamp_seed0_20261027/pt/checkpoint-399.pth --batch_size 64 --accum_iter 1 --epochs 100 --warmup_epochs 0 --min_lr_epochs 10 --lr 0.1 --min_lr 0 --seed 0 --dist_eval --output_dir output_dir/ntu60_xsub_textrel_coarse3mean_m70_q65536_cl1e4_kl01_gap256_dec3_noamp_seed0_20261027/lp --log_dir output_dir/ntu60_xsub_textrel_coarse3mean_m70_q65536_cl1e4_kl01_gap256_dec3_noamp_seed0_20261027/lp/tensorboard
```


## 2026-10-27最新v2：完整EMA key队列由InfoNCE和全局KL共用

用户进一步明确：“原生MacDiff接key encoder做teacher对比学习，就用它的完整骨架特征queue，文本侧建立对应queue；KL仍全局，query为每批mask后10%可见token的全局池化”。本版据此实现；不加入此前讨论的六部位local KL，也不另建一个骨架关系库。旧v1 masked-online入库配置/命令仍保留在后文，不能混同实验身份。

### 计算定义

原生encoder8/256在120帧、时间patch4、25关节下有750个token。query encoder只编码随机保留的75个增强视图token；key encoder编码同一crop内全部750个干净骨架token（不是整段原始视频，不加入diffusion噪声）。两侧均第8层norm后GAP：

```text
q_i = normalize(mean_visible E_query(mask(x_aug_i)))       # 256维，有梯度
k_i = normalize(mean_all E_key(x_clean_i))                 # 256维，no_grad/EMA/eval
y_i = normalize(sample_target_blend_global_text_i)         # 512维，detach

Q_S[j] = 历史完整骨架 k_j
Q_T[j] = 与同一sample_id配对的历史文本 y_j

P_S(i,j) = softmax_j(cos(q_i,Q_S[j]) / tau_S)
P_T(i,j) = softmax_j(cos(y_i,Q_T[j]) / tau_T)
L_KL = mean_i KL(stopgrad(P_T(i,:)) || P_S(i,:))

L_CL = InfoNCE(q_i, 当前完整k_i为正样本, 同一个Q_S作为历史负样本)
L = L_S2S + 1*L_T2S + 1e-4*L_CL + .1*L_KL + .02*Uni_skeleton
```

InfoNCE和KL实际访问同一个 `relation_skeleton_keys` buffer、同一IDs；没有额外 `contrast_queue`。两个损失均排除历史同sample_id的全部副本；当前完整k_i在InfoNCE中保留为正样本。KL两边的参照列一一对应，排除self后不足2个参照的行暂不提供KL，空队列的InfoNCE只有正样本，首步为0。

入库的是(k_i,y_i,sample_id)，**不是masked query q_i**。两个FIFO容量/IDs/指针/覆盖时序统一，当前默认65536，沿用之前AA0的骨架对比学习容量，文本队列扩到同容量。容量只由 `relation_bank_size` 设置，两个损失共同使用，不再另设contrast_queue_size。

完整key encoder是同架构EMA副本，只复制骨架encoder/patch嵌入/norm/位置参数；冻结且始终eval，无投影头、AA对齐或层混合适配器。仅在成功完整optimizer步之后提交本窗口已有的完整key/配对文本，并更新一次EMA：E_key←.999*E_key+.001*E_query。AMP跳步、空提交及不完整累积尾窗均不更新队列或EMA。调用顺序为提交该forward的旧EMA key，再更新EMA权重；不是优化后额外重算一次key。

原生main在DDP参数同步后、resume载入前调用initialize_pretraining_state，将fresh key复制到同步后的online encoder；完整resume随后恢复已保存EMA/队列/teacher_updates。当前main明确broadcast_buffers=False，队列通过成功步后的变长all_gather保持跨rank配对一致，不依赖buffer广播。旧v1公共engine的supports_relation_memory提交开关继续复用，本轮不再修改engine。

### 256维与6400维

用户仍保留维度方案讨论空间。按本次明确的“可见token全局平均”，query自然为256维，key也采用同样的GAP256，配置 `relation_feature_mode=gap`。6400维需要按25关节各自池化再拼接；90% mask下还需定义不可见关节的处理，属于另一种读出，本版不擅自加入。暂不提供6400训练模式，其他feature_mode会明确拒绝，不能只换队列维数或重复/补零冒充同一种特征。

LP保持已有第8层完整输入、时间/人物平均后25关节拼接6400维和原分类头；这与训练辅助分支GAP256的读出分开记录。预训练本身仍有局部token条件，不能称原生MacDiff只训练一个256维向量。MAMP官方LP使用GAP256、finetune用关节拼接+MLP，参见[官方LP配置](https://raw.githubusercontent.com/maoyunyao/MAMP/main/config/ntu60_xsub_joint/linprobe_t120_layer8.yaml)及[读出代码](https://github.com/maoyunyao/MAMP/blob/main/model/transformer.py)。

### 参数、状态与验证

新模型身份 `model.transformer_macdiff_text_relation_teacher.Transformer`，方法版本 `relation_teacher_version=full_ema_gap_shared_queue_global_kl_v2`；底层关系KL沿用v1实现。默认EMA=.999、CL温度.07/权重1e-4，KL两侧温度.1/权重.1。保留T→S权重1、独立原生结构decoder3、remap隐藏2048、七句v3缓存和逐样本目标值平滑.1；无S→T decoder，不接local KL、OTFM、多层diffusion或固定预训练teacher蒸馏。

两个FP32特征队列分别64 MiB（骨架65536×256）和128 MiB（文本65536×512），另有约.5 MiB IDs及指针；这是buffer静态大小，不是实测峰值显存，另有EMA encoder和完整key前向开销。无新增可训练投影/decoder，EMA参数冻结；online encoder/文本T→S分支仍正常训练。现有loss/熵/队列指标保留，另记loss_contrastive、loss_contrastive_weighted、contrast_positive_cosine；teacher_updates计数随checkpoint保存。

新模型/入口/配置与v1独立，完整resume拒绝v1或旧纯骨架teacher等方法冒充；K/EMA/CL/KL温度和权重/数据缓存/训练协议均绑定。保持原逐样本文本目标库的同checkpoint快照要求。旧控制组defaults与状态格式保持，公共基类仅新增两个缺省无效的key/辅助loss扩展方法。

本地v2新增12项：9通过、3因缺Torch跳过；v2+v1+累积共30项，23通过/7跳过。通过含生产forward NumPy适配执行（masked query、完整clean key输入、只前向一次query encoder、配对入库、CL/KL总loss）、完整key处理所有token、InfoNCE共享同一队列及self排除、成功步EMA时序、恢复隔离。3项新真实Torch检查覆盖5可见token vs50完整token的小模型、真实反向/EMA更新与配对入队、完整state/旧LP兼容，待服务器运行；NumPy不验证autograd，未验证实际GPU/NCCL/DDP。所有相关回归145项为74通过/71缺Torch跳过，teacher旧测试的导入路径需按原tests发现方式或整合runner处理。本地Python3.8语法与参数/配置/命令静态检查通过，不等于真实PyYAML或训练运行。

### 同步与运行

新文件：
- model/transformer_macdiff_text_relation_teacher.py
- main_pretrain_text_relation_teacher.py
- config/ntu60_xsub_joint/pretrain_madiff_text_relation_teacher_decoder3.yaml
- tests/test_text_relation_teacher.py

同时同步本轮修改的 model/transformer_macdiff_text_relation.py、main_pretrain_text_relation.py、tests/test_text_relation.py；沿用此前当前engine_pretrain.py、main_pretrain.py、util/misc.py和文本缓存/目标库支持文件。服务器先检查：

```bash
python -m unittest tests.test_text_relation_teacher tests.test_text_relation tests.test_pretrain_accumulation_policy -v
```

新输出目录、端口10325/10326；encoder8/native decoder3/T→S decoder3，PT400/两卡总64/floor5e-4/seed0/noAMP→LP100/总128/floor0/原第8层6400读出，从本组checkpoint399启动LP：

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10325 main_pretrain_text_relation_teacher.py --config config/ntu60_xsub_joint/pretrain_madiff_text_relation_teacher_decoder3.yaml --relation_bank_size 65536 --lambda_contrastive 0.0001 --ema_momentum 0.999 --contrast_temperature 0.07 --lambda_relation 0.1 --relation_temperature_text 0.1 --relation_temperature_skeleton 0.1 --lambda_text_to_skeleton 1 --lambda_skeleton_to_text 0 --disable_amp --batch_size 32 --accum_iter 1 --epochs 400 --warmup_epochs 20 --min_lr_epochs 20 --lr 0.001 --min_lr 0.0005 --mask_ratio 0.9 --motion_aware_tau -1 --seed 0 --output_dir output_dir/ntu60_xsub_textrel_fullkey_q65536_cl1e4_kl01_gap256_dec3_noamp_seed0_20261027/pt --log_dir output_dir/ntu60_xsub_textrel_fullkey_q65536_cl1e4_kl01_gap256_dec3_noamp_seed0_20261027/pt/tensorboard && OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10326 main_linprobe.py --config config/ntu60_xsub_joint/linprobe_madiff.yaml --finetune output_dir/ntu60_xsub_textrel_fullkey_q65536_cl1e4_kl01_gap256_dec3_noamp_seed0_20261027/pt/checkpoint-399.pth --batch_size 64 --accum_iter 1 --epochs 100 --warmup_epochs 0 --min_lr_epochs 10 --lr 0.1 --min_lr 0 --seed 0 --dist_eval --output_dir output_dir/ntu60_xsub_textrel_fullkey_q65536_cl1e4_kl01_gap256_dec3_noamp_seed0_20261027/lp --log_dir output_dir/ntu60_xsub_textrel_fullkey_q65536_cl1e4_kl01_gap256_dec3_noamp_seed0_20261027/lp/tensorboard
```

助手未启动训练，新组无成绩、不预分配T33。以下是旧v1的定义、检查与原命令，供独立对照追溯。


2026-10-27：用户提出“remap后的文本与骨架各建memory bank，KL让骨架模仿文本的相似结构”，并明确选择**保留T→S扩散训练remap，只替换S→T**。已提供独立模型、入口、配置和测试。旧S→T diffusion、OTFM采样对照、后三层diffusion是各自独立实验，本实现不混接。助手未启动训练，无新成绩或实验编号。

## 监督和梯度

- 保留原生S→S diffusion及skeleton uniformity=.02。
- 保留T→S diffusion、独立的原生结构decoder3及remap512→2048→512，权重1；沿用七句v3缓存（global+六local）、concat_memory、A/person0。
- 删除S→T TextNoiseDecoder的构造和训练；关系分支没有decoder、投影头或新可训练参数。
- 首轮只比较全局关系。骨架s为第8层norm后可见token的GAP（256维），文本y为现有sample_target_blend平滑global目标（512维），相似度前各自L2归一化。七句local继续参与T→S，六个local的关系KL未加入。

两个FIFO记忆库按相同slot和sample_id保存历史配对骨架/文本特征。对于当前样本i，j遍历两侧共同有效的同一批参照样本，并排除所有sample_id=i的列：

```text
P_T(i,j) = softmax_j[cos(stopgrad(y_i), stopgrad(K_T[j])) / tau_T]
P_S(i,j) = softmax_j[cos(s_i, stopgrad(K_S[j])) / tau_S]
L_relation = mean_i sum_j P_T(i,j) * (log P_T(i,j) - log P_S(i,j))
L_total = L_S2S + 1*L_T2S + 0.1*L_relation + 0.02*Uni_skeleton
```

KL方向是KL(P_T || P_S)，只有当前骨架query接收此损失梯度，直接回到第8层GAP和encoder；文本query、两个库都detach。remap仍由T→S训练，KL不反向推动文本目标迎合骨架。两边特征维度无需一样，因为对齐的是同一K个参照样本上的概率分布。没有温度平方补偿、额外对比loss或标签监督。

文本teacher沿用旧最佳dec3的逐样本目标值平滑定义：每次成功优化步后，以新remap输出对已访问样本更新 y_i←.9*y_i+.1*remap(CLIP_i)；首次访问未初始化目标时沿用既有固定CLIP fallback。关系query和入库文本均取本次forward开始时读取的global平滑目标，不是在线r的未平滑输出，也不是remap参数EMA。这样保留旧目标稳定方式；它与新加入的两个关系FIFO库是不同的状态。

## 记忆库和训练时序

方法身份 `paired_global_fifo_text_to_skeleton_kl_v1`。默认K=4096对、tau_T=tau_S=.1、lambda_relation=.1；这些是首轮起始设置，未调参，不声称与旧S→T噪声MSE的.1等效。

- 两个库共享一组IDs和环形指针，容量4096；这是最近历史配对条目的FIFO，**不是全训练集每ID一个槽的索引库**。两侧同时入库、同时覆盖。重复访问可以留下多个同ID条目，但query会在两侧排除该ID的全部副本。
- 条目存L2归一化的、已detach的当前forward特征；先backward/optimizer成功，再提交该累积窗口的条目。库不保留计算图，不增加encoder EMA或额外骨架前向。
- 跳过AMP优化步和未完成的累积尾窗均不入库。text sample target bank仍按既有成功优化步更新，两类状态都不在forward里提交。
- DDP在成功步后按rank顺序all_gather IDs/骨架/文本，支持各rank有效样本数不同；两侧共同更新，保证相同列含义。没有实际多卡验证，不把mock gather当作NCCL验证。
- 空库和排除self后不足两个参照的行暂不提供KL；没有epoch级预热，第一次成功步入库后即可开始。默认全局batch64时约64个成功步装满（有效person0较少时稍慢）。
- 两个特征库、IDs、指针均为持久buffer，跟随设备移动并保存在model checkpoint；256/512维、K4096的FP32特征约12 MiB，另有IDs。参数量不含这些buffer，训练无需新增OT或其他库。
- 当前原生main明确broadcast_buffers=False，配对库依靠成功步all_gather保持跨rank一致；当前实现不承诺降低通信耗时或实测显存，是否加速须服务器测量。

## 日志与判读

记录loss_relation、loss_relation_weighted、relation_bank_fill、relation_anchor_count、relation_valid_fraction、relation_teacher_entropy、relation_teacher_entropy_fraction、relation_student_entropy，前面由公共main加train_。

teacher_entropy_fraction是每行文本分布熵除以该行有效参照数的log，再取均值；接近1说明文本关系较平，较小的KL本身不能证明表示变好。冷启动无有效行时指标记0，必须结合valid_fraction/bank_fill看。目标是冻结encoder后的LP，KL数值和生成重建质量不能替代LP。

本方案属于跨模态关系蒸馏。可参考[SEED](https://arxiv.org/abs/2101.04731)的相似度分布蒸馏与[ReSSL](https://arxiv.org/abs/2107.09282)的关系分布学习；本实现用remapped text目标与配对骨架库，并非这两篇方法的完整复现，也不是OPD。

## 文件、恢复和验证

服务器同步：
- model/transformer_macdiff_text_relation.py
- main_pretrain_text_relation.py
- config/ntu60_xsub_joint/pretrain_madiff_text_relation_decoder3.yaml
- tests/test_text_relation.py
- **本次engine_pretrain.py**：新特性开关传入sample_indices、收集pending配对条目，并仅在成功完整优化步后提交。旧文本/FM/teacher路径开关缺省False。
- 此前已使用的main_pretrain.py、util/misc.py、文本模型/缓存/target bank支持文件保持现有版本。

完整resume需本方法checkpoint及既有逐样本文本目标库的同checkpoint快照；关系FIFO已经在model state_dict中。入口绑定模型身份、完整model_args（包括K/温度/权重）、数据/缓存路径、日程/精度等，公共main还核对文本缓存identity及目标库元数据。旧S→T/native/FM/teacher权重不能冒充完整续训。新PT目录若已有log/checkpoint会拒绝覆盖。未保存RNG、数据/服务器源码未hash，不保证逐位复现。

本地新测试13项：9通过、4因无PyTorch跳过；通过含生产forward的NumPy适配执行、KL方向、跨维度对应、参照同序、排除全部self/空槽、FIFO环绕/配对与成功步提交、入口/恢复/目录保护。4项真实Torch覆盖梯度隔离、无S→T完整前向/encoder与remap反向、队列checkpoint/LP兼容、mock变长gather，待服务器运行。NumPy不验证autograd，mock不验证真实DDP。既有文本/一致性/FM/梯度累积回归也已检查，详情见handoff。

服务器先执行：

```bash
python -m unittest tests.test_text_relation tests.test_pretrain_accumulation_policy -v
```

PT为encoder8/native decoder3/T→S decoder3，400轮、两卡每卡32、总64、accum1、lr.001/floor5e-4、warmup20/floor20、seed0、noAMP；LP100、总128、lr.1/floor0，只读第8层原LP特征与原分类头。历史T23为AMP，精度差异仍需区分；新组未有成绩。新端口10323/10324，新输出目录如下：

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10323 main_pretrain_text_relation.py --config config/ntu60_xsub_joint/pretrain_madiff_text_relation_decoder3.yaml --lambda_relation 0.1 --relation_bank_size 4096 --relation_temperature_text 0.1 --relation_temperature_skeleton 0.1 --lambda_text_to_skeleton 1 --lambda_skeleton_to_text 0 --disable_amp --batch_size 32 --accum_iter 1 --epochs 400 --warmup_epochs 20 --min_lr_epochs 20 --lr 0.001 --min_lr 0.0005 --mask_ratio 0.9 --motion_aware_tau -1 --seed 0 --output_dir output_dir/ntu60_xsub_textrel_kl01_k4096_t01_dec3_noamp_seed0_20261027/pt --log_dir output_dir/ntu60_xsub_textrel_kl01_k4096_t01_dec3_noamp_seed0_20261027/pt/tensorboard && OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10324 main_linprobe.py --config config/ntu60_xsub_joint/linprobe_madiff.yaml --finetune output_dir/ntu60_xsub_textrel_kl01_k4096_t01_dec3_noamp_seed0_20261027/pt/checkpoint-399.pth --batch_size 64 --accum_iter 1 --epochs 100 --warmup_epochs 0 --min_lr_epochs 10 --lr 0.1 --min_lr 0 --seed 0 --dist_eval --output_dir output_dir/ntu60_xsub_textrel_kl01_k4096_t01_dec3_noamp_seed0_20261027/lp --log_dir output_dir/ntu60_xsub_textrel_kl01_k4096_t01_dec3_noamp_seed0_20261027/lp/tensorboard
```

