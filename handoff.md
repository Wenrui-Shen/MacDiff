# MacDiff 交接文档（2026-10-18，新建文本双向重建一致性方法；T24已完成）

本文供完全没有聊天上下文的新会话接手。以下最新状态优先于其他文档中的历史计划。用户在服务器训练；助手负责审查代码/论文、分析日志、维护记录和提供命令。本地没有启动训练。

## 1. 接手先看：当前选择和最后一个问题

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

## 2. 我们在做什么

主要有两条关联任务：

1. **复现原生MacDiff的NTU60 XSub joint-only线性评估。**论文主表86.4%，第三方双3090口述86.2%，我们多组结果在85.6–86.0附近。逐项核对官方代码、论文、实际lr/batch/decoder/epoch/增强/LP协议，定位差距；目前没有证据确认是硬件或某个实现bug。
2. **研究Stage1文本辅助no-share版。**在原生骨架扩散预训练上加T→S和S→T任务，使用固定CLIP句向量、可学习remap和逐样本持久目标。最新T23 best86.280932%，比配对T22原版3层best高.818777pp、末20均值高.698083pp；比T21旧5层best仅高.060650pp、末20均值低.044578pp。全decoder3/PT floor5e-4配对已完成，当前有明确正向LP结果；随后用户已选5/5/3组，见顶部当前计划；不自动扩大其他实验矩阵。

旧Stage2 OSE/ReSA工作不是当前任务，背景保留在 [Stage2旧交接](D:/program/MacDiff/handoff_stage2_legacy_20260905.md) 与 [OSE设计](D:/program/MacDiff/MACDIFF_OSE_CROSS_INSTANCE_DESIGN.md)。用户目前明确不做新增对比学习、跨样本关系蒸馏或其他跨样本方法；最新明确关闭text uniformity，保留骨架uniformity。固定CLIP目标、固定噪声条件打乱、方差/相似度诊断已经试过，不要重复当作新方向。v参数化、区域bias只讨论过，未实现。

### 环境和边界

| 项目 | 当前信息 |
|---|---|
| 本地工作区 | `D:/program/MacDiff`，Windows PowerShell |
| 当前本地HEAD | `e854ff11cc60e0c02838331cfbf59586e3b6e15c`，2026-10-14核验 |
| Git remote | `git@github.com:Wenrui-Shen/MacDiff.git` |
| 官方MacDiff比较提交 | `692888f1e2a4227216511aac6733dff31b587213` |
| 官方MAMP比较提交 | `09e6252a5c7d061c226395d25e6aa47a69e7e58d`（2023-09-22） |
| 用户服务器仓库 | `/home/user9/public3/swr/MacDiff` |
| 服务器数据路径 | `../data/MAMP/ntu/NTU60_XSub.npz`，预期train40091/test16487 |
| 历史服务器环境 | conda macdiff，Python3.8、Torch1.8.1+cu111、双RTX4090；最新实际版本待核验 |
| 访问方式 | 助手无SSH，用户自己执行Linux命令 |
| 本地验证能力 | 本会话使用 `C:/Python314/python.exe` 做标准库/AST测试；该解释器缺Torch/NumPy/PyYAML，未做CUDA/DDP/模型运行验证 |
| 本地数据 | 没有此次训练的实际NPZ/cache/checkpoint可直接验算 |
| 用户论文 | [macdiff.pdf](D:/program/paper/macdiff.pdf)；原页面渲染临时文件未保留 |

默认Windows sandbox helper曾初始化失败；本会话 exec_command 用 require_escalated 经自动审查后执行。写文档不需要安装依赖。其他解释器的依赖应现查，不要沿用旧环境结论。

用户用中文，通常双卡；训练命令要求**单行Linux、不加cd，PT和LP用&&连成一条**。没有授权助手启动/停止服务器任务、提交推送、清理数据或重生成缓存。用户嫌handoff_artifacts新文件太多：今后优先更新现有文档/结果表，不为每次检查新建目录、复制日志或增加备份。

## 3. 已完成的工作与结论

### 原生实现审计与尾窗口恢复

对照固定官方提交的92个源文件，2026-10-11审计时55份规范化后一致，37份不同（包括文档、路径和扩展）。22项原生核心AST检查通过；diffusion全部15个文件、下游模型、LR函数、原生初始化/attention/MLP/损失/采样核心、feeder旋转函数等未发现改变原生算法的已证实差异。详见 [完整实现审计](D:/program/MacDiff/handoff_artifacts/native_implementation_audit_20261011/README.md) 与 [audit.json](D:/program/MacDiff/handoff_artifacts/native_implementation_audit_20261011/audit.json)。静态相同不证明CUDA前向/梯度逐位一致。

已知剩余差异：当前PT DDP用 `find_unused_parameters=False, broadcast_buffers=False`；官方为True及默认True。**用户只要求恢复梯度尾窗口，两个DDP开关没有改回。**原生模型以LayerNorm为主，关闭OSE/text时无对应额外参数/Queue，没有据此确认分差原因；文本辅助分支则确有额外参数和RNG消耗。LP的DDP设置未改。native forward公开默认mask/tau由.8/.75改为.9/-1，但engine显式传参；整数gather索引repeat→expand数学值相同，未做实际梯度等价验证；可选encoder activation checkpoint默认关闭。

**已按用户要求把PT不足一个累积窗口的尾梯度改回官方丢弃策略。**`engine_pretrain.py` native/text及OSE路径都 loss/=accum_iter，仅完整窗口末step；尾部到下轮zero_grad时丢弃。尾部不更新text bank/EMA或OSE teacher/Queue，AMP跳步的成功step保护保留。累积1没有尾窗口；标准NTU60双卡32/accum2共626microsteps，也是整除。验证：5项实际引擎语句的标准库控制流测试+原8项旋转runner测试，13/13通过；这不是Torch autograd/DDP测试。

原生AMP：PT默认True，LP默认False；仍用GradScaler。seed默认0，每rank使用seed+rank，Torch/NumPy初始化、worker Python/NumPy seed与sampler.set_epoch逻辑保持官方；cudnn.benchmark=True，不应承诺跨环境严格确定性。seed影响初始化、shuffle/增强、mask、扩散噪声/t、dropout等；新架构额外初始化消耗RNG，所以同seed不等于原生与文本逐位同轨迹。

### 官方隔离副本与旋转脚本

为了干净官方对照，创建了 [upstream_692888f](D:/program/MacDiff/handoff_artifacts/upstream_692888f/ISOLATION_README.md)：92份官方源文件+manifest/README，只有3份例外：util/misc.py的torch._six.inf改同值math.inf；两份NTU60 PT/LP YAML仅迁移现有数据路径。官方DDP、模型、engine保持原样。**没有切分支或覆盖当前扩展代码。**从仓库根调用副本main文件，脚本目录优先导入副本模块；不要删除这个目录，当前命令还在用它。

已完成原生decoder3/PT500随机旋转两组的自动串行runner：no_rot PT→LP，再rot PT→LP，自动输出best acc/差值；唯一组间因素是PT random_rot。它校验配置、代码/数据身份和样本数，拒绝复用输出根目录，不自动resume。入口为 [script_pretrain_madiff_rotation_ablation.sh](D:/program/MacDiff/script_pretrain_madiff_rotation_ablation.sh)，实现 [run_macdiff_rotation_ablation.py](D:/program/MacDiff/tools/run_macdiff_rotation_ablation.py)，说明 [MACDIFF_ROTATION_ABLATION.md](D:/program/MacDiff/tools/MACDIFF_ROTATION_ABLATION.md)。**两组已经有结果，不再等待或重跑。**

当前 `script_pretrain_madiff.sh` 曾按双卡默认参考调整为2×32/accum1；官方原脚本是4×32。LP原脚本仍是四卡模板。实际入口/卡数/CLI决定batch和lr，不能只看文件名或YAML。

### 日志核验与资料整理

完整维护 [EXPERIMENT_RESULTS.md](D:/program/MacDiff/EXPERIMENT_RESULTS.md) 和 [STAGE1_EXPERIMENT_RESULTS.md](D:/program/MacDiff/STAGE1_EXPERIMENT_RESULTS.md)。当前24条Stage1配置/成绩记录，不代表24次均有完整实参验证；T20第二次LP不另编号，T21是独立no-share5层PT+LP，T22是新原版3层/min_lr5e-4 PT+LP，T23是全3层/remap2048/text uni0的no-share，T24是保留T23其余设置的5/5/3组。旧文档正文有过期“待结果”和实验数量，读取顶部最新日期并以本交接为准。

handoff_artifacts保存官方源码副本、精确日志/统计、实现审计、历史交接和候选配置，不是自动启动的新实验。此前盘点约159文件/8.35MB，最近多数新增来自官方94文件副本。不要因“文件多”直接删除或移动，且不要继续无必要扩充。

## 4. 已有成绩：完成、口述与待测严格分开

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

### T20：官方默认PT与同seed两次LP已完成

按用户描述和所给命令：隔离官方692888f、native/A、decoder5、PT400/每卡32/accum1/seed0/no rotation/lr.001/min1e-5；checkpoint399；LP100/每卡64/accum1/seed0/lr.1/dist_eval。两次目录分别lp与lp1。

- PT epochs0–399完整、有限；last loss=.014521730446313231，末20 loss=.01457348432352457±.00008287206650271168；最后20轮lr1e-5由日志核验。
- 两次LP epochs0–99完整：**best86.00194081283784 @95；last85.7836002144404；末20均值85.8503153964306±.07206914375114691。**可训练参数384060，符合6400→60 FC。
- 两份LP100轮所有字段、原始字节完全一致，长度20024，SHA256=`f9ff33cdba3667ff1dd6fd84435fefccb021416b12b2111d610e9d1bd8032018`。用户明确确认两次seed0、不同输出目录、算法命令相同；新上传lp1再次核验仍同SHA。**不要继续无依据怀疑粘贴重复，也不要把T20重复LP另编号；T21是另外已完成的no-share训练。**
- 这证明本次同seed LP重复无可见差异；不测跨LP/PT seed方差，也不证明checkpoint文件逐位相同。T19改变batch，不是同配置随机重复，不能据85.79→85.89说LP随机性很大。
- LP90–99轮lr=0，但head BN仍train并更新running statistics，所以best@95正常；不能说FC95轮仍在更新。
- 距第三方86.2约.198059187pp。完整日志不含完整Namespace/数据/代码/权重SHA/运行版本，不能宣布所有实际参数已独立核验。

全文归档：[T20 README](D:/program/MacDiff/handoff_artifacts/official_dualgpu_results_20261013/README.md)、[summary.json](D:/program/MacDiff/handoff_artifacts/official_dualgpu_results_20261013/summary.json)，含pt.jsonl、lp1/lp2及重上传证据。不再等待这组结果。

第三方参考：用户确认另有人双RTX3090、原MacDiff默认配置、PT每卡32（默认accum1→总64）复现86.2。LP“估计默认每卡64→总128”仍是推测，没有链接/命令/日志；不把seed/各具体实参当已核验，也不增加项目实验ID。没有证据把我们差距归因4090/3090或双卡硬件。

## 5. MAMP与MacDiff对齐分析（已完成分析，未变更实验）

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

## 6. 当前no-share方法与准确运行命令

### 结构和维度：新会话不要混淆

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

### 2026-10-14命令，10月15日原版已完成T22：全decoder3配对

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

### 10月13日旧5层计划：已完成T21，以下保留运行身份与命令

现有 [no-share YAML](D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_text_sentence_sample_target_blend_noshare.yaml) 的顶层默认lambda=.1/.1，命令**显式覆盖1/.1和batch32**。main_pretrain把顶层lambda写入model_args，只改model_args同名值可能被覆盖。

公共设置：A mean=[-.0058,-.1333,-.0246]、var=[.0206,.0805,.0218]；PT400/warmup20/min_lr_epochs20/wd.05/AdamW betas(.9,.95)，mask.9、motion_aware_tau=-1、diffusion epsilon/inverse_cosine tau1/1000steps、native condition drop.1、Gaussian joint noise[1,.005]、PT random_rot=False/source_rot=False/flip=False。LP frozen encoder/linprobe2/6400维/head dropout.3/BN+FC，100轮/warmup0/min_lr_epochs10/SGD momentum.9/wd0，训练旋转True、测试False。

PT用当前文本代码，LP用与T20相同的隔离官方入口；公共超参数对齐，**不是PT源码逐位相同**（DDP开关、辅助任务、RNG等仍不同）。相对T12，PT accum2→1把global128降到64，LP仍128；因此新组对照T20，不能把与T12差值称文本结构单因素。

以下是T21已完成组此前提供的命令，保留用于核对身份，**不要自动重复启动**。PT/LP独立输出目录，400轮载入399：

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10275 main_pretrain.py --config config/ntu60_xsub_joint/pretrain_madiff_text_sentence_sample_target_blend_noshare.yaml --lambda_text_to_skeleton 1 --lambda_skeleton_to_text 0.1 --batch_size 32 --accum_iter 1 --epochs 400 --lr 0.001 --min_lr 0.00001 --mask_ratio 0.9 --seed 0 --output_dir output_dir/ntu60_xsub_noshare_dec5_ep400_pt32_lp64_seed0_20261013/pt --log_dir output_dir/ntu60_xsub_noshare_dec5_ep400_pt32_lp64_seed0_20261013/pt/tensorboard && OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10276 handoff_artifacts/upstream_692888f/main_linprobe.py --config handoff_artifacts/upstream_692888f/config/ntu60_xsub_joint/linprobe_madiff.yaml --finetune output_dir/ntu60_xsub_noshare_dec5_ep400_pt32_lp64_seed0_20261013/pt/checkpoint-399.pth --batch_size 64 --accum_iter 1 --epochs 100 --lr 0.1 --min_lr 0 --seed 0 --dist_eval --output_dir output_dir/ntu60_xsub_noshare_dec5_ep400_pt32_lp64_seed0_20261013/lp --log_dir output_dir/ntu60_xsub_noshare_dec5_ep400_pt32_lp64_seed0_20261013/lp/tensorboard
```

同一命令已存 [实验计划顶部当前选择](D:/program/MacDiff/tools/vlm_pilot/STAGE1_TEXT_EXPERIMENT_PLAN.md)。本地检查过CLI/配置/Bash语法，未在Torch环境运行导入。需要同步隔离副本至服务器；目前没有完整服务器身份信息，不推断已经同步或未同步。

## 7. 当前卡在哪里、下一步做什么

当前主要缺**完整实际运行身份**；T22/T23新配对结果已齐，不是已定位bug：

- 原版3层T22、no-share全3层T23、5/5/3的T24均已完成；T21是旧5层，结果不能混用。decoder蒸馏分支未收到结果，不猜服务器运行状态。
- T20/T21/T22/T23/T24日志完整，但完整Namespace、服务器当前源码/未提交改动、NPZ/checkpoint SHA与软件版本缺失。旋转组只summary，T15/T16/T19只有口述；不能从库存YAML补齐实际参数。
- 论文86.4与公开默认5层/400/no rotation、表10不同decoder成绩存在口径差；第三方86.2无公开实参。现有证据不足以确定硬件、seed、DDP、person数、LP头或某个单因素解释差距。
- 本地没有实际权重/数据和Torch，无法端到端比较前向、梯度、AMP/DDP或数据身份。

接手顺序：

1. 先按用户下一条消息继续。若问当前实验，只确认已有命令对应的真实状态，**不重启/停进程、不跑seed1、不重跑T17/T18/T20/T21/T22/T23/T24**。
2. T24已完成日志核算和T23/T21比较；下一步按用户新指示。若继续核验身份，补actual args、checkpoint399、T23三套3层/T24的5/5/3、text decoder512/remap2048、PT总64/LP总128、seed0与实际LP入口，不能从当前配置补填；若收到新日志先查重复/运行身份，再决定是否新增ID。不要自动新增消融矩阵或把条件依赖假设当已证实原因。
3. 若继续定位原生复现差距，先补完整身份：Namespace、服务器git SHA+dirty diff、Torch/CUDA/cuDNN/timm、NPZ SHA/count/classes/有限值/人物空槽、checkpoint SHA/epoch/args与load_state_dict信息。优先利用现有encoder做用户选定的LP检查，避免无根据重训。
4. MAMP对齐若获选定，先明确目标是复现各自原协议，还是共同评价协议。LP256读出/head/dropout、骨架3层、min_lr5e-4、人物数等拆成可解释控制；双人物sample target须先实现/验证，不直接改False。没有选择前只保留候选，不穷举矩阵。
5. 后续结构研究可选512vs768、最终LN或独立文本condition dropout，但各自尚无新实验结果；dropout独立开关未实现，默认应保留0。全位置MSE、S→T v参数化、区域bias也未实现，不能假装已有YAML可用。

500轮比较必须从初始化按500日程训练；400训练后追加100轮是continuation，不等于原生500日程。LP换decoder配置也不会改变已学encoder。更多PT seed只在用户选定后考虑，当前seed1明确取消。

## 8. 踩过的坑：绝对不要再踩

1. **不git clean/reset、不删除移动artifact/cache/checkpoint、不覆盖用户改动。**未跟踪文件不是垃圾，官方隔离目录还被命令引用。用户嫌文件多，优先更新现有记录，不再每次生成重复目录/日志/备份。
2. **不拿旧计划覆盖新选择。**T17/T18/T20/T21/T22/T23/T24已完成，T19=85.89已回传；T15=85.84、T16=85.86已口述。旧1/.5、旧文本旋转计划、seed1、768都不是当前已选下一步。24条记录，T20重复LP不另编号；T21是旧no-share5层，T22/T23为原版/no-share3层配对，T24为5/5/3，均已完成。
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

## 9. 资料索引、未提交状态与历史保存

新会话优先顺序：本handoff→结果表→当前实验计划顶部→对应分析README；旧交接只用于恢复历史设计，里面“待生成/待结果/下一步”已被本文替代。

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
