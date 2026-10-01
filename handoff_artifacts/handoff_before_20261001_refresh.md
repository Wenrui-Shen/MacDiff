# MacDiff Stage1 文本扩散交接（2026-09-28，会话结束版）

> 2026-10-01 新增：用户已要求检查旧文本→缓存→训练链，并接入新生成的全局/六部位描述。详见 `tools/vlm_pilot/TEXT_CACHE_CHAIN_AUDIT.md`。已新增 `cache_clip_motion_text.py` 和 v3 句级 RMS 缓存校验；逐人 reader 自动支持 v2/v3，另有 sentence_fixed_rms 与 sentence_sample_target_blend_shared 两份新配置。新文本应编码为每人七个完整句向量，部位位置 1..6、结构 context_length=7；保持独立缓存和实验目录，从头训练，不使用旧 checkpoint/目标库完整 resume。此为本地代码接入，尚未生成服务器全量 v3 缓存或跑 GPU/DDP。下文“不要重新生成描述/缓存”仅指既有旧实验，不能覆盖用户后续明确授权的新文本实验。

> 2026-10-01 后续明确选择：暂时保留原有文本 uniformity；只调整日志。空人物过滤仍执行，取消普通空骨架提示和 `empty_skeleton_persons` 指标；取消 `text_target_drift_mse`；`text_batch_variance` 改为在线 remap 后 global + 有效 local 内容向量合并后的通道方差均值，不统计保存目标、结构 embedding 或 padding。模型与引擎文件需要同步。历史训练日志的方差定义不同，勿直接混用。

本文写给完全没有上下文的新会话。请先读第1节“当前状态”、第7节“新增瓶颈分析”及第6节“绝对不要再踩的坑”，再看代码和命令。用户偏好中文、可直接复制的单行 Linux 命令。历史 handoff_stage2_legacy_20260905.md、handoff_vlm_pilot_legacy.md 和旧实验脚本仅供追溯。

## 1. 当前状态：任务、完成情况、卡点

任务：在 NTU60 XSub 的 MacDiff Stage1 骨架预训练中加入由骨架描述得到的文本监督，尝试改善骨架 encoder 的动作表示和下游 linear probe（LP）。不是 Stage2/OSE，也不是文本生成。LP 只加载骨架 encoder，文本分支不进入下游模型。

用户最后明确的目的：**T→S 分支训练 remap，使文本语义与动作相关，再通过 S→T 分支更新骨架 encoder。** T→S 不直接给 encoder 梯度是有意设计，不能把这一点本身当作错误。应该验证的是“动作信息经过 remap、保存目标和 S→T 传入 encoder”的效果。

用户最后明确的新设计是：每个原始训练样本保存自己的骨架→文本去噪目标；初始目标是固定 CLIP 向量的逐向量 RMS 归一化结果。该样本经过一次成功的 optimizer step 后，用当时在线 remap 网络的输出更新：

    target_i ← 0.9 × old_target_i + 0.1 × current_online_remap(RMS(CLIP_i))

global 文本向量和有效 local token 向量都这样更新。这里固定的是更新比例，不需要对 remap 网络参数做 EMA。这个递推在数学上属于“目标值的 EMA”，但与之前的“EMA teacher 网络”不是同一个实验。alpha=0.1 时旧目标贡献约每 6.58 次该样本更新减半；初始 CLIP 不会被永久锚定。如果用户将来要永久混合 0.9 固定 CLIP + 0.1 当前 remap，那是另一个无历史状态的公式，不能擅自替换当前确认过的公式。

新模式 sample_target_blend 已在服务器开始训练：用户提供了 epoch 0～19 的日志（附件 d0c2985f-c3a5-4814-a2e2-5ba0fc9ef962/已粘贴的文本.txt），尚无 LP。早期日志本身不足以确定组别；**后来用户的 checkpoint-130 恢复命令明确选择了共享骨架 decoder 组**，不要继续笼统写最近使用的组未知。总 loss 1.265→0.08744；global 保存目标批内方差 0.13008→0.04038、能量 1.0→0.79619、相对固定 CLIP 的 drift 0→0.73816。数值训练平稳，但 global 目标明显集中，不能据此声称 LP 提升。首轮目标库 40064、第二轮 40091，与两卡每卡32、drop_last=True 和后续重排补齐相符。

同一日志：native diff MSE 1.003737→0.0288101，骨架 uniformity 0.81493→0.0459333，文本 uniformity 0.314332→0.0514129，T→S 原始 MSE 1.22462→0.0313634，S→T 原始 MSE 1.16233→0.535485。有效 local token 稳定约20.136；CUDA allocated峰值约17441～17452MiB、reserved约18514～18550MiB，未持续增长。epoch平均lr约2.492e-5→9.7492e-4，符合20轮warmup。最后一轮加权S→T项约0.05355，占总loss约61%，**loss占比不等于encoder梯度占比**。

用户反馈每个 epoch 比之前多约10分钟。2026-09-28 已本地实现性能版：目标在同机共享 RAM 中维护，仅 checkpoint 时写兼容 SQLite 快照；同时复用累积窗口内的 CLIP 输入，避免 optimizer step 后重复读缓存和上传。两份配置与 CLI 默认 text_target_bank_backend=shared_memory，可用 sqlite 回退。CPU 双进程共享、递推值与旧后端等价、旧快照恢复及新快照兼容测试已通过；本机无 PyTorch，尚无该性能版真实 Linux /dev/shm、GPU、AMP 或 DDP 测试与测速。当前下一步是同步性能改动，服务器 smoke，再从同组配对 checkpoint 续训。服务器没有本地可用 SSH。绝不能把旧 EMA 网络实验的 LP 或诊断结果说成新逐样本方案的结果。

最新恢复尝试因训练命令后直接拼接 `cd ... && LP命令`，缺少训练与cd之间的分隔符，在argparse阶段失败；模型和目标库均尚未恢复。已提供修正命令，**尚未收到修正后的成功恢复日志**。当前卡点是缺少服务器运行证据、当前模式的语义传递诊断和LP结果。

本会话后半段发现独立S→T文本decoder的256隐藏维度/512噪声输出存在约0.5的结构性MSE下限，讨论了扩宽到512；也讨论了T→S中512→256条件投影可能限制remap动作监督。见第7节。**512架构、辅助语义loss、新诊断都只讨论，未实施，未被用户最终选定。** 会话结束前用户只要求写handoff，不要把建议当作已经授权立即改模型。

## 2. 环境、数据和历史结果

本地工作区：D:/program/MacDiff，Windows PowerShell。服务器根目录：/home/user9/public3/swr/MacDiff；历史环境为 conda macdiff、Python 3.8、PyTorch 1.8.1+cu111、两张 RTX 4090 24GB。用户自己同步代码和运行。数据：../data/MAMP/ntu/NTU60_XSub.npz。现成 v2 CLIP 缓存：vlm_pilot/ntu60_xsub_clip_cache_v2。不要重新生成描述或缓存。

本地可用Python为 `C:/Users/97537/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe`，Python3.12、有NumPy、没有PyTorch。没有可工作的WSL。不要反复尝试本机GPU训练或声称执行过torch测试。未发现适用AGENTS.md；不要擅自启用子agent。

Qwen3-VL-8B 已为 NTU60 XSub train 的 40091 个样本生成逐人物英文骨架描述，历史 18 个 GIF 错误已补齐，accepted_unique=40091、missing=[]。CLIP global/local token 缓存已完成验证。当前只用 skeleton person0 配它自己的文本；空 person0 裁剪应从 loss 和目标更新排除，不得改用 person1。local token 按 batch 动态 padding，平均约 20 个有效 token，并非固定 76。双视图 GIF 的 front/side 不是两个人。

历史 LP best（均为用户提供的单次数字，按其要求先比较数值；协议差异仍需记住）：

| 方案 | LP best |
|---|---:|
| 原始 MacDiff | 历史约 85.86% |
| 第一版双向 remap、动态目标、S→T 权重 1.0 | 83.03% |
| 固定 CLIP 原尺度、S→T 权重 1.0 | 84.64% |
| 固定 CLIP 原尺度、S→T 权重 0.1 | 85.95% |
| 固定 CLIP RMS、S→T 权重 1.0 | 约 83.7% |
| 固定 CLIP RMS、S→T 权重 0.1 | 85.82%，用户暂定基线 |
| 旧网络参数 EMA 双向，不 share decoder | 85.77% |
| 旧网络参数 EMA 双向，share decoder | 85.78% |
| 当前逐样本 0.9/0.1 目标方案 | 已提供前20轮预训练日志，尚无 LP |

新旧 LP 的训练 batch 设置可能不同。旧固定 CLIP RMS 85.82% 的 LP 有效 batch 是 64，最近两组参数 EMA LP 有效 batch 是 128；两组参数 EMA 彼此可比，与 85.82% 仅作历史数值对照。LP 要比较 best-vs-best，不能混用 best 和末轮。

## 3. 旧参数 EMA 实验与历史诊断：结论和边界

旧实验配置为 config/ntu60_xsub_joint/pretrain_madiff_text_rms_ema_bidirectional.yaml 及 _shared.yaml，text_target_mode=ema_remap、text_target_momentum=0.999。它训练一个在线残差 remap，同时维护一个不反传的 EMA teacher remap；S→T 去噪目标来自 teacher 网络输出。用户指出这并非他本意。旧实验已经完成 epoch 399 及 LP，不要再当作“待运行方案”。

针对旧实验的新诊断脚本：diagnose_ema_bidirectional.py；测试：tests/test_ema_bidirectional_diagnostic.py。用户在服务器跑出的完整 JSON 在 C:/Users/97537/.codex/attachments/d027fa4a-f322-41be-9472-8ebf4c422cfa/已粘贴的文本.txt，协议 ema_bidirectional_targeted_diagnostic_v1。诊断使用 256 个固定训练 crop；条件交换的成对分析可用 254 个样本，2 个不能配对。同一比较内固定 crop、mask、噪声、t；t=100/500/900。此脚本和结果只针对旧参数 EMA checkpoint，不能强行载入当前 sample_target_blend checkpoint。

| 诊断量（epoch 399） | share | 不 share | 固定 CLIP |
|---|---:|---:|---:|
| global teacher 平均通道方差 | 0.03298 | 0.00822 | 0.13347 |
| global teacher 跨样本平均余弦 | 0.96689 | 0.99175 | 约 0.866 |
| local teacher 平均通道方差 | 0.97584 | 0.97662 | 0.53532 |
| global teacher 与对应 CLIP 的平均余弦 | 0.17223 | 0.07177 | — |
| global teacher 与 CLIP 的中心化两两余弦关系 Pearson | 0.80189 | 0.80459 | — |

解释：旧参数 EMA 的 global 目标明显集中，尤其是不 share；但 local token 方差并未一起收缩。中心化的样本间关系仍有约 0.80 的相关，不能说全部语义结构消失。share 的 global 方差约为不 share 的 4 倍，但 LP 仅高 0.01 个百分点，不能推出方差越大 LP 越好。

把正确配对条件打乱后，去噪 MSE 相对增幅：

| 方向与交换 | share：t=100/500/900 | 不 share：t=100/500/900 |
|---|---|---|
| 文本→骨架：打乱全部文本条件 | 4.19% / 6.88% / 8.33% | 4.01% / 7.44% / 9.13% |
| 文本→骨架：只打乱 global | 0.08% / 0.11% / 0.22% | 0.08% / 0.14% / 0.18% |
| 骨架→文本：打乱全部骨架条件 | 0.38% / 0.28% / 0.13% | 0.49% / 0.29% / 0.13% |

文本→骨架确实利用了配对信息，结果支持local提供较多条件信息；global与local可能冗余，不能由global单独交换小就认定global无用。**此前只按S→T总MSE百分比说骨架条件几乎不用，解释过强，需纠正。** 总MSE包含约0.5的结构性下限（第7节），稀释相对比例。例如share、t500的正确/打乱MSE为0.5137890652/0.5152139579，差约0.001425，是总MSE的0.28%，但约为`0.513789-0.5`的10.3%。后一个比例只是粗略参照，不是精确因果贡献，需要按实际输出子空间分解误差。打乱后损失上升不证明动作类别语义质量或LP改善。固定CLIP RMS权重1.0曾有较强条件效应但LP约83.7%，就是反例。旧参数EMA share/不share的85.78%/85.77%与固定RMS0.1的85.82%接近，未显示稳定增益。

用户曾贴出 share 组 0～399 epoch 训练日志：总 loss 从 1.265 降到约 0.0696，text_target_drift_mse 从 0.011 升到约 1.66，S→T 原始 MSE 后期约 0.518，T→S 约 0.021。不要把 loss 下降当作 LP 提升；两条文本扩散日志都是乘 0.1 权重之前的原始 MSE。

另有历史固定文本/原骨架几何结果 `handoff_artifacts/stage1_text_geometry_summary.json`（约379KB，勿整份dump），协议 `stage1_caption_balanced_geometry_v1`。100个固定均衡train batch，每批16类×8=128；文本为**人物聚合global CLIP**，骨架用完整双人物和实际6400维LP读出，不是当前person0/global+local协议，更不是测试LP准确率。class-macro AUC/P@1/P@5：文本0.667381/0.191414/0.159642，原MacDiff checkpoint399骨架0.774436/0.522610/0.367702。不要预设文字整体类别判别强于已有骨架；仍可能存在互补信息。说明 `tools/vlm_pilot/STAGE1_TEXT_GEOMETRY.md`；`STAGE1_READOUT_COMPARISON.md`和`TEXT_CONDITION_DIAGNOSTIC.md`在**仓库根目录**。readout文档主要是协议，不能编造已测得的读出增益。

## 4. 当前 sample_target_blend 实现：必须按此理解

代码：

- model/transformer_macdiff_text.py：新增 text_target_mode=sample_target_blend，参数 text_target_update_ratio=0.1；只有在线 remap，没有 text_target_remap teacher 网络。T→S 用在线 remap 后 global/local 文本作条件；S→T 从目标库接收 detach 的 global/local 去噪目标，反传到骨架 encoder 和文本噪声 decoder，不直接反传到 remap。
- util/sample_text_target_bank.py：保留以原始样本索引为键的 SQLite 目标库作为回退后端，以及共用的窗口输入复用更新逻辑。首次访问从固定 RMS CLIP global/local 初始化。每次成功 optimizer step 后，用更新后的在线 remap 输出按 0.9/0.1 更新当步样本目标。空 person0 / 无有效 token 的非活动行返回零目标并跳过更新。
- util/shared_memory_text_target_bank.py：默认性能后端。同一主机的两卡共享一份 float32 目标，按实际有效 token 打包，约1.6 GiB RAM；短进程锁保证整个累积窗口更新原子性和重复样本的递推次数。Linux 使用 /dev/shm 的 mmap，所有 rank attach 后立即 unlink 后备文件，映射随进程退出释放；Windows 用具名匿名 mmap。运行中不访问 SQLite；保存时才写原有 sample_text_target_bank_v1 SQLite 快照。可从旧后端快照恢复，新快照也可切回旧后端。缺快照和身份不符仍报错。同机 DDP 限制；/dev/shm 不足会提示，可用 --text_target_bank_backend sqlite 回退。
- engine_pretrain.py：在 forward 前取该样本上次目标，按累积窗口收集非空 person0 索引和已读取/上传的固定 CLIP 输入；成功 optimizer step 后重新计算当前 remap 输出并更新目标，不复用 step 前旧输出。AMP 跳过 step 时不更新。新增每轮 train_text_target_read_seconds、train_text_target_update_seconds、train_epoch_seconds，取各 rank 最大值；更新计时含 remap 和 GPU→CPU 复制。
- main_pretrain.py：准备目标库、记日志 target_bank_initialized_samples，给每份 checkpoint 建对应目标库快照。比如 checkpoint-399.pth 对应 checkpoint-399-target-bank.sqlite。resume 先载入同组模型 checkpoint，再从同名快照恢复目标库；缺快照应报错，不可偷偷重置。
- util/misc.py：resume 检查新模式、0.1 更新比例、uniformity、共享设置和缓存身份。
- 两份新配置：config/ntu60_xsub_joint/pretrain_madiff_text_rms_sample_target_blend.yaml（独立 decoder）和 pretrain_madiff_text_rms_sample_target_blend_shared.yaml（共享 decoder）。均为 RMS、T→S 0.1、S→T 0.1、文本 uniformity 0.02、400 epoch，差别仅共享开关和输出目录。

新模式仍是 512 维 RMS CLIP 输入、随机初始化的可训练残差 remap 和 512 维文本去噪目标。文本 person/position embedding 只加到 T→S 条件，不写入 S→T 干净目标。混合后的目标不再保证每个时刻 text_energy 恰好为 1，因为两个 RMS 向量的加权平均可降低能量；首次目标应约为 1，后续以实际日志为准。历史 text_target_drift_mse 是目标相对固定 RMS CLIP global 向量的 MSE，2026-10-01 已从训练输出取消。

`text_energy`统计的是**保存目标的global向量**；`text_batch_variance`于2026-10-01 改为在线 remap 后的 global 和有效 local 向量合并后的方差（历史日志则是保存目标 global 的方差）。初始CLIP目标的直接系数为`0.9**n`，n是该样本实际更新次数，不严格等于epoch；n=20时约0.1216，n=130时约1.126e-6。CLIP仍持续作为残差remap输入，不能据此宣称CLIP信息消失。当前remap为512→512→512残差MLP、FP32 RMS归一化；当前在线remap随机初始化，旧teacher的identity初始化不适用于它。

当前配置：骨架dim_feat=256、encoder8层/decoder5层，文本decoder隐藏256、5层、8heads；120帧×25关节，t_patch_size=4、patch_size=1，共750tokens，90%mask后约75可见。T→S/S→T权重0.1/0.1，骨架/文本uniformity权重0.02/0.02。训练400轮、warmup20、lr1e-3、min_lr1e-5。注意原始 `script_pretrain_madiff.sh` 也覆盖min_lr为1e-5，不能仅凭旧YAML默认值声称学习率发生变化。

性能分析证据边界：旧后端每microbatch同步读取目标；每step又读CLIP/上传、remap、D2H、SQLite `BEGIN IMMEDIATE`/`synchronous=FULL`，两rank争写锁。每rank每轮626个microbatch/313个optimizer step，全局约626次写事务。多600秒分摊到313个step边界约1.92秒/边界，数量级合理但**并非实测原因**。新read计时不含初始CLIP cache读取；update含remap与D2H/CPU整理；snapshot含保存阶段等待，不能当成纯磁盘耗时。

尚未做的进一步速度优化：打包累积窗口global/有效local为一次remap与一次D2H；固定CLIP预打包/预取；减少仅日志使用的`.item()`/all_reduce及不必要cuda同步。当前累积2时仍逐microbatch计算global/local，约4次remap forward及4次D2H。GPU常驻bank会每GPU额外占约1.6GiB且需跨rank同步历史，不是已实施方案。先用新计时找剩余开销，不承诺固定提速倍数或GPU位级一致。

SQLite 快照按 float32 存 global 加有效 local token，约 40091 个样本、平均20 local token；每个快照达到 GiB 级。40 多份快照会占用较多磁盘；服务器运行前检查空间。shared_memory 后端没有持续落盘的 current.sqlite，current.shared.json 只是会话描述，续训必须从模型 checkpoint 和配对快照恢复。旧实验 current.sqlite 保留，不用于性能版每步更新。LP 只读模型 checkpoint，不需要目标库快照。旧参数 EMA checkpoint 不能用于新模式 resume；独立与共享组也不能互相 resume。不要删除已有库或改写旧实验目录。

## 5. 下一步和准确命令

原 sample_target_blend 已启动；本次性能版需要同步6个运行文件：engine_pretrain.py、main_pretrain.py、util/sample_text_target_bank.py、util/shared_memory_text_target_bank.py、两份 pretrain_madiff_text_rms_sample_target_blend YAML。测试再同步 tests/test_shared_memory_text_target_bank.py。默认改为 shared_memory，模型架构和目标更新比例未变。先检查 df -h /dev/shm 和快照磁盘余量，再用下面独立目录做 smoke。既有训练请在保存配对 checkpoint 后结束原进程，再完整恢复历史目标；不得只恢复模型或从头初始化目标。拿到真实 traceback 再修具体问题。本地无服务器执行能力。

**下一会话顺序：** 先核对checkpoint-130完整恢复及性能版是否成功运行，用真实计时确认额外10分钟是否改善；再适配当前模式诊断remap→目标→encoder的信息传递；取得当前256模式LP，与相同LP协议的固定RMS0.1基线比较；最后根据用户选择开展512或其他独立实验。不要未经确认就中断既有训练换架构，也不要宣称讨论方案已实施。

最近恢复错误附件 `C:/Users/97537/.codex/attachments/e1e88210-5135-4e15-ba7d-4cb0cabda08a/已粘贴的文本.txt`：用户把训练命令后直接拼接 `cd /home/... && LP命令`，缺少训练与cd之间的`&&`，报 `MAE pre-training: error: unrecognized arguments: cd /home/user9/public3/swr/MacDiff`，退出2。错误发生在argparse，模型/目标库尚未恢复，不是OOM或checkpoint不兼容。usage包含新backend参数仅证明服务器main CLI识别它，不证明其他文件已同步或GPU/DDP成功。

**共享组从checkpoint-130完整恢复的正确单行命令**，从epoch131继续到399（epochs400为总轮数）：

~~~bash
cd /home/user9/public3/swr/MacDiff && OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10244 main_pretrain.py --config config/ntu60_xsub_joint/pretrain_madiff_text_rms_sample_target_blend_shared.yaml --batch_size 32 --accum_iter 2 --epochs 400 --text_target_bank_backend shared_memory --resume output_dir/ntu60_xsub_macdiff_rms_st01_sampletarget01_t2s01_shared/checkpoint-130.pth --skip_text_cache_validation
~~~

必须同时存在同组同次保存的 `output_dir/ntu60_xsub_macdiff_rms_st01_sampletarget01_t2s01_shared/checkpoint-130-target-bank.sqlite`。本地未验证服务器文件是否存在，也尚无修正命令的成功运行输出。缺快照应报错，不能重置目标库。仅从SQLite换RAM后端允许完整恢复，换文本decoder维度不允许同样完整恢复。

Smoke（独立 decoder，两卡，每卡 batch 32，累积 2 次，2 个 micro-step，独立输出目录）：

这只验证一次optimizer step等流程，不是完整epoch测速。若smoke目录已存在目标描述/快照，另用新目录，不能删除旧实验绕过保护。同一输出目录禁止并行启动两次。

~~~bash
cd /home/user9/public3/swr/MacDiff && OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10246 main_pretrain.py --config config/ntu60_xsub_joint/pretrain_madiff_text_rms_sample_target_blend.yaml --text_target_bank_backend shared_memory --batch_size 32 --accum_iter 2 --epochs 1 --warmup_epochs 0 --min_lr_epochs 0 --max_train_steps 2 --output_dir output_dir/ntu60_xsub_sampletarget01_ram_smoke --log_dir output_dir/ntu60_xsub_sampletarget01_ram_smoke/tensorboard --skip_text_cache_validation
~~~

正式预训练使用两卡、每卡 batch 32、accum_iter 2，有效 batch 128。两个任务使用同一对 GPU，应顺序运行。全新目录从头开始，不加 --resume。

独立 decoder：

~~~bash
cd /home/user9/public3/swr/MacDiff && OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10242 main_pretrain.py --config config/ntu60_xsub_joint/pretrain_madiff_text_rms_sample_target_blend.yaml --batch_size 32 --accum_iter 2 --output_dir output_dir/ntu60_xsub_macdiff_rms_st01_sampletarget01_t2s01 --log_dir output_dir/ntu60_xsub_macdiff_rms_st01_sampletarget01_t2s01/tensorboard --skip_text_cache_validation
~~~

共享 decoder：

~~~bash
cd /home/user9/public3/swr/MacDiff && OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10244 main_pretrain.py --config config/ntu60_xsub_joint/pretrain_madiff_text_rms_sample_target_blend_shared.yaml --batch_size 32 --accum_iter 2 --output_dir output_dir/ntu60_xsub_macdiff_rms_st01_sampletarget01_t2s01_shared --log_dir output_dir/ntu60_xsub_macdiff_rms_st01_sampletarget01_t2s01_shared/tensorboard --skip_text_cache_validation
~~~

两组完成 epoch 399 后，分别用相同 LP 设置：两卡每卡 batch 64、100 epochs、lr 0.1、seed 0、dist_eval。LP 不需要目标库 SQLite。

独立 decoder 的 LP：

~~~bash
cd /home/user9/public3/swr/MacDiff && OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10243 main_linprobe.py --config config/ntu60_xsub_joint/linprobe_madiff.yaml --finetune output_dir/ntu60_xsub_macdiff_rms_st01_sampletarget01_t2s01/checkpoint-399.pth --output_dir output_dir/ntu60_xsub_macdiff_rms_st01_sampletarget01_t2s01_lp_399_bs64 --log_dir output_dir/ntu60_xsub_macdiff_rms_st01_sampletarget01_t2s01_lp_399_bs64/tensorboard --batch_size 64 --accum_iter 1 --epochs 100 --lr 0.1 --seed 0 --dist_eval
~~~

共享 decoder 的 LP：

~~~bash
cd /home/user9/public3/swr/MacDiff && OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10245 main_linprobe.py --config config/ntu60_xsub_joint/linprobe_madiff.yaml --finetune output_dir/ntu60_xsub_macdiff_rms_st01_sampletarget01_t2s01_shared/checkpoint-399.pth --output_dir output_dir/ntu60_xsub_macdiff_rms_st01_sampletarget01_t2s01_shared_lp_399_bs64 --log_dir output_dir/ntu60_xsub_macdiff_rms_st01_sampletarget01_t2s01_shared_lp_399_bs64/tensorboard --batch_size 64 --accum_iter 1 --epochs 100 --lr 0.1 --seed 0 --dist_eval
~~~

用户若提供新模式训练日志，先核对 text_target_mode=sample_target_blend、text_target_update_ratio=0.1、rms、share 开关、分支权重、目标库初始化数、checkpoint 与目标库快照是否配对，再分析 loss 和 LP。若只给 torch.distributed.launch 的 CalledProcessError，要找其前面的真实 rank traceback。若做严格历史基线比较，需要用相同 LP 命令参数重跑固定 RMS 0.1 基线，因为旧 85.82% 是另一 LP batch。

用户若要训练完自动LP，必须以 `完整训练命令 && 完整LP命令` 连接，不得直接把两段无分隔拼接。不能只粘贴LP命令的`cd`到训练参数尾部。

## 6. 本地验证、工作区和绝对不要再踩的坑

本地最新验证：Codex捆绑 Python 3.12 执行 unittest tests.test_sample_text_target_bank tests.test_shared_memory_text_target_bank -v，8项中6项通过、2项因缺torch跳过。覆盖与SQLite逐值等价、重复索引、不同token长度、空人物、旧/新快照双向恢复及完整性、两个独立CPU进程共享目标且并发更新不丢失。新代码按 Python3.8语法 ast.parse 检查通过；这不是实际Python3.8执行。git diff --check通过，仅CRLF提示。尚未验证Linux tmpfs、真实模型前向、GPU/AMP/DDP；不要说生产双卡性能已验证。

本次开始时git工作区干净。当前性能版未提交改动：engine_pretrain.py、main_pretrain.py、util/sample_text_target_bank.py、两份sample_target_blend YAML、handoff.md；新增未跟踪：util/shared_memory_text_target_bank.py、tests/test_shared_memory_text_target_bank.py。不要 reset、clean、覆盖或声称已commit。没有生成zip。

会话结束这一轮仅更新handoff；模型仍是256维文本decoder，未增加辅助loss或新诊断。此前测试踩过Windows只读句柄fsync失败，已用`r+b`；SQLite上下文管理器不会自动关闭连接，测试用`contextlib.closing`。本次文档改动无需重跑GPU或模型测试。

必须记住：

1. 当前每个样本的递推目标是用户明确选定的方案。不要再把“EMA 网络”当作用户意图，也不要把旧参数 EMA teacher 的 checkpoint、诊断、LP 混入新方案。固定比例作用于保存的目标值；在线 remap 参数正常梯度训练。
2. 不从旧 EMA、第一版 remap、固定 CLIP 或另一 share 设置 resume 新模式。新模式 resume 必须同组同配置、模型 checkpoint 与 -target-bank.sqlite 配对；不得只恢复模型而重新初始化目标库。
3. 不重新生成 40091 条描述或 v2 CLIP 缓存。现有缓存可信时可用 --skip_text_cache_validation；完整 hash 和逐行扫描耗时且双 rank 都执行。
4. 骨架 person0 必须配自己的文本；空 person0 不用 person1 代替。目标库应跳过非活动空行。front/side 是同一人的不同视角。
5. 原尺度 L2 CLIP 每通道能量约 1/512，RMS 约 1。不能直接跨尺度比较 S→T MSE、text_energy 或绝对方差。text_energy≈1 只说明尺度，不能证明语义质量；新方案混合后还可能偏离 1。
6. T→S 损失下降不代表骨架 encoder 一定改善；T→S 不直接更新 encoder。S→T 打乱测试增幅不等于 LP 改善。要看实际 LP best。
7. 总 loss 包含多个加权项，不能和原始 MacDiff native loss 直接比。同口径 native 是 loss_diff + 0.02*loss_uniformity；文本两方向日志 MSE 未乘 0.1。
8. 预训练 --batch_size 是每卡值，32×累积2×两卡=有效128。最近 LP 每卡64×两卡=有效128；历史 RMS0.1 LP 有效64。共享/独立对照必须保持相同 LP 设置。
9. checkpoint-0 已完成第一轮，不是随机初始化。训练 log.txt 可能追加重启段，出现重复 epoch；分析时按重启分段或取每个 epoch 的最后一条。
10. OSE 默认参数出现在 args 不代表启用；看 enable_ose=False。torch.distributed.launch 的 CalledProcessError 多半只是父进程摘要。误按 Ctrl+Z 会暂停训练进程却占着端口，先查进程再重启。
11. 不用只支持旧 remap 的 diagnose_text_conditioning.py、只支持固定 CLIP 的 compare_fixed_clip_conditions.py，或只支持旧参数 EMA 的 diagnose_ema_bidirectional.py 强行读取新模式 checkpoint。需要先适配目标构造及 checkpoint 加载。
12. 服务器空间需容纳 SQLite 当前库和 checkpoint 快照。不要为省空间擅自删匹配快照；那会使相应模型 checkpoint 无法完整续训。
13. 不要把share理解成T→S与S→T共用decoder。实际共享是原生骨架重建与T→S骨架decoder主体；讨论256/512时必须说清模块。`text_decoder_hidden_dim`只作用于独立S→T文本decoder。
14. 不要把S→T约0.5的MSE或很小的shuffle百分比直接解读为骨架条件没用；先考虑输出秩造成的下限。它也不是LP上限，扩宽decoder未证明能改善acc。
15. T→S的512→256条件压缩没有同样的512维噪声输出下限；对remap动作监督的潜在限制需验证，不能说固定一半坐标/语义永远不训练。
16. 修改`text_decoder_hidden_dim`到512会改变参数形状，旧130不能完整resume。不能静默`strict=False`部分加载却声称完整恢复；若以后明确采用部分加载，应列出重置的模块与optimizer状态。
17. 当前“只有S→T直接给encoder文本监督梯度”是用户设计目标，不能擅自当bug修掉。验证语义传递链与实际LP，新增辅助loss仍未选定。
18. 当前energy/variance测保存global目标，不是在线remap；句内local uniformity不是跨样本类别约束。实际LP读出6400维，不能拿`feature_only=True`的global256代替。
19. `0.9**n`的n是样本实际更新次数，不严格等于epoch；初始目标系数很小不代表CLIP完全消失，因为残差remap一直读取CLIP。原训练脚本min_lr也覆盖为1e-5，不能只看旧YAML杜撰学习率差别。
20. 最近恢复失败是shell命令缺分隔导致argparse失败；尚无真实模型恢复结果。不要把main CLI出现新参数当作整个性能版运行已验证。

## 7. 本会话新增的维度与语义传递瓶颈分析（方案未实施）

### 7.1 share的确切对象与模块维度

用户最后连续追问“512是不是就不能share”“256/512不是share的那个decoder吗”“T→S也有512文本，是否同样有问题”。此前笼统称“decoder”造成歧义，已经纠正：**讨论扩宽的模块是独立S→T文本decoder，share开关是另一个骨架decoder。**

```text
骨架 encoder（256）→ 原生骨架重建 ─┐
                                  ├→ 共享骨架 decoder（256）→ 骨架噪声（每patch12维）
CLIP（512）→ remap（512）→ 512→256 ─┘                 ↑ T→S

加噪文本目标（512）→ 独立文本 decoder（隐藏256）→ 文本噪声（512）
                                 ↑
                    骨架 encoder 的 pooled/visible tokens
```

`TextConditionedSkeletonDecoder`在shared模式复用原生`decoder_embed`、`decoder_blocks`、`decoder_norm`、`decoder_pred`、位置/时间嵌入；自己的条件投影和文本reader等仍独立。共享模块只注册原生名字，forward传owner以避免重复checkpoint键。

| 模块 | 当前维度 | 共享关系 |
|---|---:|---|
| 骨架encoder | 256 | 原生编码器 |
| 原生骨架decoder | 256 | 与T→S共享主体 |
| T→S骨架decoder | 256 | `share_skeleton_decoder=True`时复用原生主体 |
| 残差text remap | 输入/隐藏/输出均512 | 在线训练；不与decoder共享 |
| S→T文本decoder | 输入512、隐藏256、输出512 | 独立；`text_decoder_hidden_dim`控制隐藏维度 |

仅改 `text_decoder_hidden_dim: 512` 可以同时保持 `share_skeleton_decoder: True`。已有 `TextNoiseDecoder.skeleton_input` 将encoder256投影到新隐藏512。若加宽共享骨架decoder，当前原生结构与dim_feat/encoder耦合，是更大改动，不能只改文本decoder参数当作做到了。

### 7.2 S→T的结构性噪声MSE下限

`model/transformer_macdiff_text.py`中 `TextNoiseDecoder` 的输入是`Linear(512,256)`，输出为`LayerNorm(256)+Linear(256,512)`，没有512维noisy-input直接绕过主体到输出的残差。S→T对每个有效token生成512维独立标准高斯噪声，预测epsilon。

最后线性层使每个token的预测落在最多256维仿射子空间。对512维单位方差各向同性噪声，固定输出子空间下期望每通道MSE至少约 `(512-256)/512=0.5`；输出LayerNorm还可能再减少有效秩，所以说“约0.5”。这是**期望/结构限制**，不要求每个有限batch精确≥0.5。输入也将加噪文本512压成256，丢失独立噪声方向；只加深256主体或换复杂末端head并不能自动恢复这些输入信息。

它解释当前S→T0.535、旧末期约0.518为何很难继续接近零；也要求修正旧shuffle百分比的解释。**不证明256维不能承载动作语义，不给出LP上限，不证明改512一定提升LP。** 更强decoder可能只是更容易利用noisy text去噪，骨架条件/encoder受益仍需测量。

最简单的单变量实验建议是另建配置与输出目录，只将独立文本decoder隐藏维度改512，保留共享骨架256、remap/历史目标公式、权重及其余训练协议。更多计算与显存需实测。旧256的checkpoint-130不能完整resume到512；公平对照要相同起点/预算。如果以后明确选择部分加载，必须公开重置项，不能称完整resume。

另一讨论方案是保留256主体，增加512维加噪输入到输出的skip，由主体预测修正。需要按timestep设计正确系数和扩散参数化，可能更易绕过骨架；尚未设计/实现，不能直接加裸identity当作解决。

### 7.3 T→S的512→256投影：潜在监督限制，不是同一噪声下限

T→S的`text_input`将512维在线remap global/local投影到256维条件，供共享骨架decoder预测骨架噪声。骨架patch是t_patch_size4×patch_size1×dim_in3=12维，最后256→12，**不存在S→T那种256隐藏预测512维各向同性噪声的约0.5下限**。

但与用户目标有关的潜在限制是：在某一步固定投影矩阵下，落在其零空间的remap输出变化，T→S decoder直接感知不到；T→S对remap输出的梯度经过这层投影，而S→T保存并学习完整512维remap目标。可能只有部分信息充分受到动作重建监督，其他部分也进入S→T目标。

投影和remap都在学习，RMS归一化、uniformity也会参与耦合；不能把它说成固定一半坐标永远不训练或一定丢掉类别关键信息。**目前没有证据证明这已限制LP。** 仅扩宽S→T到512不会同时解决这个潜在问题。扩宽共享骨架decoder或让动作监督语义空间与S→T目标空间一致都是可研究方向，但会改结构/目标定义，尚未选方案，不能擅自改用户确认的目标链。

### 7.4 其他候选限制、诊断与下一步计划

T→S训练骨架重建，不自动等价于动作类别判别；remap与decoder可协同重建，却未必提高骨架encoder的线性可分性。固定文本整体类别几何也不天然强于原骨架（见第3节），历史强S→T权重反而降低LP。

`masked_text_uniformity_loss`仅对**同一描述内local token**的平方余弦做约束，含对角线；约20token的下限约0.05，当前0.051不能证明跨样本类别结构已健康。它没有直接约束global跨样本集中。S→T将global/local所有有效token统一平均，global约占1/21，未单独加权。

完整动作描述配0.5～1随机时间crop再90%mask，部分语义细节可能无法从encoder输入观察；这是需验证的监督可见性问题。S→T非线性decoder学会去噪也不保证6400维LP特征线性可分。

**实际LP读出**在`model/transformer_downstream.py::ActionHeadLinprobe2`：保留关节，沿人物和时间平均成25×256=6400，dropout0.3，再`main_linprobe.py`中的BatchNorm1d(affine=False,eps=1e-6)+Linear。encoder冻结，head训练，SGD lr0.1/momentum0.9/wd0，100轮；完整750tokens，双人物独立编码后平均。`feature_only=True`返回global256平均，**不是实际LP输入**。预训练person0/下游双人物需记录，原MacDiff也默认one_person，不能说这是新文本方案独有错误。

下一步诊断应先适配当前sample-target checkpoint与**配对目标bank**，不能直接跑旧EMA脚本。固定crop/mask/noise/t，测matched/shuffled条件；按输出head实际列空间（考虑LayerNorm约束）分解S→T可达及正交噪声MSE；比较native与**乘0.1后S→T**对encoder的梯度范数/夹角；分别测在线remap、保存目标的global/local几何及实际LP读出。仅看loss占比、方差、shuffle总百分比不够。诊断使用训练子集/固定协议，测试集用于最终LP评价。

曾建议若确认S→T传递不足，可加小权重辅助任务：骨架encoder特征单独预测**同一历史目标库的干净global向量**，不输入noisy text，维持目标来源链。但global集中时此任务可能变简单，需监控。**辅助loss未实现、未被用户最终选定。** 建议先诊断，再做独立S→T 256/512单变量对照；根据实际encoder/LP证据逐项决定后续方案。
