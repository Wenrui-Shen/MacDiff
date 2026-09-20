# MacDiff Stage1：固定 CLIP 文本目标消融交接

更新时间：2026-09-20。面向完全没有上下文的新会话。本文当前状态优先于其他 legacy handoff 和旧交付包。先读第1、14节；第4～13节主要记录第一版双向实验及诊断历史。

## 1. 当前任务与最重要结论

**2026-09-20 最新决策：用户报告 text 第一版最终 LP 为 83.03%，整体略有下降（本轮未提供原版精确最终 LP，勿推算差值）。用户要求先去掉 T→S，再去掉 S→T 目标侧 MLP，使用固定 CLIP 特征。已在本地实现 `text_target_mode: fixed_clip` 并切换正式 text YAML/脚本默认输出；未启动服务器训练。精确结构、测试、同步文件和新实验命令见第14节。旧版模式仍可通过 `remap` 重建，供历史 checkpoint 诊断使用。EMA 不实施。**

以下为截至 2026-09-18 的第一版背景，涉及“仅讨论/尚未实现”的固定 CLIP 描述已由上述决策及第14节取代：

我们正在 NTU60 XSub 的原始 MacDiff Stage1 上加入由骨架可视化生成的文本，目标是改善骨架 encoder 的动作表示和下游识别。不是 Stage2/OSE，不是图像监督，也不是文本生成任务。

文本已由 Qwen3-VL-8B 全量生成，冻结 CLIP 的 v2 特征缓存已在服务器完成。已实现并训练逐人配对的三个任务：原始骨架去噪、文本条件骨架去噪（T→S）、骨架条件文本特征去噪（S→T）。用户已实际运行 checkpoint-0、200、350 的只读诊断；当前正确训练目录是 `output_dir/ntu60_xsub_macdiff_bidirectional_tokens`。原始基线有400轮日志。不能推断服务器此刻训练是否仍在跑或已经结束，也没有证据双卡恢复执行过。

用户报告checkpoint200第一轮LP：原版74.91%，文本67.27%，差7.64个百分点；没有完整收敛LP或350 LP结果。原始去噪目标追平不能代表分类表示追平。三个checkpoint诊断表明：全局/局部纯内容在0→200大幅收缩，200→350部分回升；正确条件的去噪收益从早期近零到后期明确为正。低方差不等于完全丢失语义，条件依赖也不等于改善动作分类。第12节保留指标和原始附件。

当前卡在机制和下游效果的因果验证，不是工程报错。用户意图：T→S训练remap保留运动相关信息，再用S→T传给骨架encoder。讨论了固定原始CLIP目标；用户进一步提出EMA动量目标，兼顾稳定和继续适应。**两种方案仅记录，尚未改训练代码、未启动新实验，EMA系数尚未选择。** EMA不能自动修复文本内容选择、梯度竞争、裁剪错配等问题。

## 2. 环境、路径和用户偏好

- 本地 Windows 项目：`D:/program/MacDiff`，PowerShell。工具只改本地；无已配置可用的服务器 SSH，用户复制文件和执行命令。
- 服务器：`ubuntu@user9`，项目 `/home/user9/public3/swr/MacDiff`，home `/home/ubuntu`。
- 训练 conda 环境 `macdiff`，日志 Python3.8、旧 PyTorch API；不要为新代码无故升级。缓存环境 `skeleton_vlm` 已有 Torch2.5.1+cu121、Transformers4.57.1。
- 两张 RTX4090 24GB。预训练脚本历史默认GPU1；本轮用户明确要求LP和诊断用GPU0。服务器诊断报告Torch1.8.1+cu111。当前进程和空闲状态未知。
- 用户中文沟通，要求完整单行 Linux 命令，不喜欢反复确认/重跑/环境折腾。
- **以后只列修改文件和用途，不生成增量 zip 包。** 已有包属于历史，不要继续打包。
- 文件同步曾通过 Windows 复制→Ubuntu 文件管理器粘贴成功；不要强迫使用 git pull/SFTP。

服务器路径（相对项目根目录）：

| 内容 | 路径 |
|---|---|
| NTU数据 | `../data/MAMP/ntu/NTU60_XSub.npz` |
| 已生成文本 | `vlm_pilot/ntu60_xsub_train_person_captions_8b_single_gpu.jsonl` |
| GIF | `vlm_pilot/ntu60_xsub_train_rendered_v3_2view_smooth_w5` |
| CLIP权重 | `/home/user9/public3/swr/models/clip-vit-base-patch32` |
| Qwen权重 | `/home/user9/public3/swr/models/Qwen3-VL-8B-Instruct` |
| 当前文本缓存 | `vlm_pilot/ntu60_xsub_clip_cache_v2` |
| 当前文本训练输出（用户确认） | `output_dir/ntu60_xsub_macdiff_bidirectional_tokens` |
| 原始基线输出 | `output_dir/ntu60_xsub_macdiff` |
| 历史几何预实验 | `vlm_pilot/stage1_text_geometry_pk128` |

## 3. 已完成的文本与缓存：不要重做

NTU60 XSub train 40091个样本，Qwen 为每个可见人物生成一个描述，无类别标签/RGB。双视图 GIF 是 front XY/side ZY，不是两个人；person0红色、person1蓝色。每人最多35英文单词，但 CLIP 子词数不等于单词数。

历史18个GIF错误已补跑，accepted_unique=40091，missing=[]。JSONL 按原始 sample_index 取最后 accepted，历史错误行仍在，不能按行数或历史失败计数认定缺失。GIF相同相邻帧被writer合并的问题已修复，不要重新生成40091条描述。

`cache_clip_text.py` + `util/clip_text_cache.py` 使用本地HF权重、eval/inference_mode。缓存协议 `macdiff_clip_token_cache_v2`：

- `person_features.npy`：FP32 `[N,2,512]`，每个人EOS句向量经冻结text_projection再L2。
- `token_features.npy`：FP16 `[N,2,77,512]`，每个最后层hidden state经同一个冻结projection、逐token L2。
- `person_valid.npy`、`token_mask.npy`、`token_ids.npy`，保留正文＋真实EOS，排除BOS/padding/空人物，无效特征0、ID=-1。
- `samples.json`、`manifest.json`：身份、SHA256、进度。token大数组约5.89GiB，训练mmap按需读取，启动校验会读取文件。
- 同一人物全局句向量与局部EOS的CLIP来源相同，FP32/FP16精度不同。CLIP projection在各token使用不意味着逐token具有同等跨模态对齐质量。
- 缓存不包含可训练remap；当前训练不加载CLIP。

**逐人逻辑在新文件 `util/person_text_cache.py` 实现，没有改旧缓存提取helper，以免破坏v2 provenance。旧 `TokenFeatureCache` 的跨人物拼接仅留在旧接口/测试中；正式入口必须使用 `load_person_token_cache`。已有v2缓存直接复用，不要因训练架构变更重编码。**

## 4. 第一版双向模型（历史 `remap` 模式；当前固定 CLIP 见第14节）

配置：`config/ntu60_xsub_joint/pretrain_madiff_text.yaml`。骨架encoder8层、特征256、8heads；骨架decoder5层；反向文本Transformer decoder5层；patch=(4帧,1关节)，120帧×25关节→750位置，随机mask90%→75可见token。

### 人物配对是明确要求

`one_person=True`，只保留骨架person0，且只读person0句向量/token。person1不作为别的样本悄悄加入，不用person1骨架/描述替代空person0。

`False`时按sample0/person0、sample0/person1、sample1/person0…展开B×2行，各自配对。**禁止k0+k1拼接，禁止两个人句向量平均，禁止第一个骨架预测第二个人文本。** 全局token现在是该人物的句向量，不是整个双人样本的聚合。

每人有效正文＋EOS长度k；K是本batch所读取人物中的最大k，K≤76。单人缓存读取 `[B,K,512]`，双人 `[2B,K,512]`，padding mask、人物编号和原位置同行。有效骨架若没有对应描述，仍明确报错，不能拿别人描述填补。

### 数据和三个分支

1. Feeder时间裁剪50%～100%，最短64帧但不超过有效长度，再resize120。encoder增强为所有关节Gaussian std0.005；扩散目标用未加该扰动的同裁剪骨架。无旋转/翻转/vel/bone。模型标准化均值[-.0058,-.1333,-.0246]、方差[.0206,.0805,.0218]，self_shift=False。
2. 骨架 `[B,3,120,25,2]` 取person0→`[B,120,25,3]`；干净骨架采t与噪声ε形成x_t。encoder只编码增强骨架一次：Conv2d3→256(kernel/stride4,1)→`[B,30,25,256]`＋时空位置编码→750→mask留下75→8层→LN，得到latent `[B,75,256]`、平均池化h `[B,1,256]`。
3. 原始骨架分支：75局部＋675复制全局，ids_restore恢复750位置条件；原生10%整份条件丢弃。x_t→decoder embedding/位置编码→5层MacDiff调制+self-attention+调制/FFN→LN→Linear256→12，输出`[B,750,12]`。目标ε patchify到相同维度；12维平均后仅675个遮挡位置平均，再batch平均；时间权重目前全1。
4. 文本remap：共享两Linear512→512→256，中间GELU。全局r=LN(MLP(e))；局部R=LN(MLP(E)+人物embedding+原位置embedding)。最终LN为FP32、elementwise_affine=False，仍传梯度，无可训练gamma/beta。无效局部清零。拼接M=[r;R]→`[B,K+1,256]`，全局mask始终有效。
5. 文本→骨架：同一人的M，复用同一份x_t/t/ε/骨架mask。每层当前带噪骨架状态LN为Q、M为K/V，8头cross-attention输出逐位置条件`[B,750,256]`，直接送MacDiff调制，不额外广播加r，不读取encoder。该分支没有原生10%整份条件丢弃。
6. 骨架→文本：**M在加噪前detach**，逐人独立采u，每人的全部token共用u，噪声逐token/channel独立；padding排除。对应人物的全局h＋75latent→骨架memory `[B,76,256]`，不detach，不跨人物汇总。文本输入Linear256→256＋独立decoder结构embedding（global/person/position），不输入干净文本内容。每层文本状态为Q、骨架memory为K/V→FeatureModulation→masked双向self-attention→FeatureModulation→FFN256→1024→256，5层→LN/Linear输出`[B,K+1,256]`噪声。
7. 反向MSE对所有有效全局/局部token和channel**一起平均**，无全局局部单独权重、无按样本长度归一化。长描述有更多有效项。这是用户选择，不要自行改为分开加权。

MacDiff调制：z Linear256→512拆zs/zb，time正弦编码64→Linear512拆ts/tb；`zs*(ts*LN(x)+tb)+zb`，逐位置逐通道。原始骨架decoder没有cross-attention；新增读取器才有。

总loss：`L_native + 0.02 L_uniformity + 1.0 L_TS + 1.0 L_ST`。uniformity是同一样本的75骨架token L2归一化、两两cosine平方均值（含对角线），不是0.2，不是文本loss。三个任务从第一步参与，没有loss curriculum；每次采一个扩散步，不完整迭代1000步。

### 共享参数和梯度

`model_args.share_skeleton_decoder: True` 当前开启，False使用独立副本。共享骨架输入projection、位置、5层blocks、norm/pred；文本cross-attention和反向文本decoder独立。共享主体只按原生名称注册一次，仍需两次条件前向，不是计算量减半。

- 原始loss＋uniformity→骨架encoder；原始loss→骨架decoder。
- L_TS→remap/正向人物位置embedding/文本读取器/共享骨架decoder，无直接encoder梯度。
- L_ST→骨架encoder/反向文本decoder，不回传remap。
- 禁用L_TS时remap和它独立参数冻结，但共享骨架主体继续训练。此时remap是冻结随机初始化，不等于原CLIP基线。
- 两辅助权重都0走原始forward。

## 5. 训练设置、实测和保存

- 原始正式脚本：默认双卡每卡32、累积1，有效64。当前文本YAML/脚本默认仍是单卡64、累积1。
- 服务器用户先以batch32/累积1完成两步测试，20.82M参数、实际lr1e-3，显存allocated约17132MiB，reserved18288MiB。不要说batch64已跑通。
- 随后建议单卡batch32/累积2维持有效64。新正式日志LR采样差异与累积2相容，但epoch日志不包含完整启动参数，不能冒充确证。恢复时保持真实原启动参数。
- 最新文本峰值allocated约17353MiB（16.95GiB），reserved18466MiB；原始基线约9127MiB（8.9GiB），是每进程/单卡口径。
- lr1e-3，min_lr1e-5，400epochs，warmup20，最后20epochs保持最低lr，AdamW betas(.9,.95)、wd.05，bias/norm不衰减；AMP+GradScaler；drop_last=True。新增参数无单独lr。
- 原始YAML的min_lr是5e-4，但原始启动脚本`--min_lr 1e-5`覆盖；必须比较实际启动值。日志base lr是换算显示，actual lr才实际使用。
- 保存条件epoch%10==0或最后轮：checkpoint-0、10…390、399；编号0已经训练完第1轮。保存model/optimizer/scaler/epoch/args，不含精确batch/RNG恢复；从epoch+1恢复。无自动best、无在线下游评估。
- resume校验cache身份、固定loss权重、共享设置、`text_person_alignment=('per_person_v1', one_person)`，旧全局/混合人物版本不兼容完整resume。空骨架bug修复不改结构，可恢复当前配对checkpoint。
- 本地只改代码，不宣称代用户跑过服务器。当前日志证明v2缓存、CUDA AMP、正式逐人训练已实际运行，旧文档“尚未CUDA训练”已过时。

历史正式单卡建议命令（默认输出仍为person_text；不是用户当前bidirectional_tokens目录的运行证明，不要直接用来开启重复训练）：

```bash
CUDA_VISIBLE_DEVICES=1 BATCH_SIZE=32 ACCUM_ITER=2 bash script_pretrain_macdiff_text.sh
```

从实际存在的同协议checkpoint恢复，例如：

```bash
CUDA_VISIBLE_DEVICES=1 BATCH_SIZE=32 ACCUM_ITER=2 bash script_pretrain_macdiff_text.sh --resume output_dir/ntu60_xsub_macdiff_person_text/checkpoint-180.pth
```

用户询问单卡转双卡，已说明可恢复、改变采样不逐步一致；GPU0可用才运行（尚未证实执行）：

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10235 main_pretrain.py --config config/ntu60_xsub_joint/pretrain_madiff_text.yaml --batch_size 32 --accum_iter 1 --resume output_dir/ntu60_xsub_macdiff_person_text/checkpoint-100.pth
```

双卡未实测，尤其要留意动态空人物过滤与DDP是否正常。不要将上面100误认为最新checkpoint。

## 6. 已修复的空骨架错误

服务器epoch3出现 `A training sample contains no active skeleton person`。原断言要求每个样本保留人物都非空，过严。现为：只要batch存在有效人物就继续；active_rows用于原始MSE、uniformity和两个文本任务；无效人的特征不进入文本损失。整batch无有效保留人物仍报错，不悄悄零loss更新。

`engine_pretrain.py` 打印 `Empty cropped skeletons excluded from losses: sample_indices=..., person_ids=...`；metric `empty_skeleton_persons` 是每iteration数量，epoch平均0.000798722≈1/1252，表示全轮1次，不是样本比例。到epoch186共84次（非84个独立样本），单轮最多2次，影响很小。

原因尚未精确定位到样本：Feeder先按双人坐标求和估计有效帧数量，在[0,count)里裁剪，之后模型只取person0。可能person1存在而person0在该段缺失，或原person0为空；有效帧计数假定非空帧连续前置，前置/内部空洞也可能出问题；坐标有符号求和不如any非零稳健。**没有原始问题样本和裁剪区间，不要断言已经证明某一种原因。** 未改裁剪协议，避免擅自改变基线增强。若排查先用终端索引检查原始人物有效帧。

修复文件：`model/transformer_macdiff_text.py`、`engine_pretrain.py`、`tests/test_macdiff_text.py`。本次写交接前git status仅这三文件已有修改，勿覆盖。数据空时保持person0配对，不自动换人；有效骨架缺描述依然报错。

## 7. 历史训练日志证据和同口径结果（到186；更新诊断见第12节）

附件可直接本地读取，不必再次索要已经提供的原始日志：

- 原始MacDiff完整400轮：`C:/Users/97537/.codex/attachments/28ce2d8f-a7ca-4d47-8cc9-c9eebdb577ae/pasted-text.txt`
- 文本最新版到epoch186：`C:/Users/97537/.codex/attachments/9cdd18e4-87a7-4393-ba46-1041134258f4/pasted-text.txt`
- 文本先前到epoch101：`C:/Users/97537/.codex/attachments/3c484967-8086-4dc4-a4c0-4eb77aed3b83/pasted-text.txt`
- 两步batch32服务器日志：`C:/Users/97537/.codex/attachments/4f597bd7-13b0-42c6-ac60-6b3c6c1954e5/pasted-text.txt`

文本最新版190条：最前3条旧运行epoch0～2；后187条新运行epoch0～186、带empty metric。按后出现的epoch记录/新协议字段分段，不能把旧前3条串入新曲线。原始日志无重启重复。

**原始仅记录train_loss，没有diff/uniformity分项。对比必须用文本的 `train_loss_diff + .02*train_loss_uniformity`，不能拿三任务train_loss比较，也不能拿纯MSE对原始总loss。**

| epoch区间（20轮均值） | 原始 | 文本版原始目标部分 | 相对高出 |
|---|---:|---:|---:|
| 0～19 | .110904 | .147405 | 32.91% |
| 20～39 | .021257 | .024843 | 16.87% |
| 40～59 | .019051 | .020212 | 6.09% |
| 60～79 | .018265 | .018921 | 3.59% |
| 90～109 | .017564 | .017905 | 1.94% |
| 140～159 | .016931 | .016994 | 0.37% |
| 167～186 | .016579 | .016606 | 0.16% |

相同epoch平均LR最大差约4e-8，基本一致；不能把前期变慢归于lr。基线epoch399 loss .01452074，不能直接对文本epoch186（训练进度不同）。epoch186文本：diff .01615906、uni .01919487、TS .02152924、ST .00903957、native合计 .01654295、总 .04711176，lr .00056319。

全局remap方差：新epoch0 .112874 → epoch20 .006682 → epoch69最低 .00098877 → epoch100 .00123037 → epoch186 .00291425（较谷底约2.95倍）。能量约1。**结论是早期强烈趋同、后期有回升，不能继续说持续单调坍缩，也不能凭方差回升证明语义改善。** ST后期从约.0084轻微上升到.0090，可能目标差异增加使任务变难，但未验证因果。

## 8. 当前疑点、讨论结论与下一步

### 已确认的解释

- `text_batch_variance` 是**remap后全局r**跨batch按维var(unbiased=False)再平均，不是CLIP缓存、不包括局部token。
- energy≈1来自无affine LN，不能防跨样本趋同。LN每次按token通道归一化，有梯度，不是冻结统计量，不是stopgrad。输出范数约sqrt256=16，与缓存L2范数1不同。
- 去掉LN允许目标尺度漂移；加affine gamma/beta增加灵活性但不抗坍缩，gamma可压缩/放大方差，beta是公共偏移。当前均未修改，仍无affine。
- 正向去噪不保证语义保留：可能主要依赖带噪骨架/先验，全局token可能冗余于局部EOS，MLP无语义保持约束。共享decoder与原始无条件训练可能降低条件依赖，均是待验证解释。
- ST因detach不直接推动remap坍缩，也无法纠正；如果目标变简单，ST loss能降低但未必学到骨架语义。
- 多token确实可能缓解全局冗余/趋同。不能只看全局方差判整个文本分支失效。局部差异也可能仅来自位置embedding，需控制结构信息。
- MacDiff .02 uniformity是样本内骨架token去相关；直接套文本只能防内部重复，允许所有样本共享同一套位置模板。跨样本方差约束或保持CLIP关系是讨论候选，**没有获准实现，也没有加任何新loss**。

### 下一步按优先级

1. 获取/完成200、350的收敛LP及原版对应轮次LP，保持配置、有效batch、head、增强、人物读取、种子和预算一致。已查明`main_linprobe.py`读取骨架encoder，辅助decoder/remap为unexpected keys被忽略，missing keys只能是head；冻结encoder只训练head。下游仍按既有双人读取、时间/人物平均和每关节拼接的linprobe2协议，不擅自改成person0-only。
2. 只读诊断已完成且用户已跑0/200/350，勿重复实现或声称尚未诊断。精确控制和局限见`TEXT_CONDITION_DIAGNOSTIC.md`。目前不能据此认定LP下降原因。
3. 若继续定位问题，测量同batch下原始任务（明确包含.02 uniformity）与S→T在encoder上的梯度范数/余弦；原始任务与T→S在共享decoder上也可测。尚未实现。loss数值不等于梯度大小，局部梯度冲突也不是最终LP因果证据。
4. 用户当前重点是EMA目标方案，见第12节。后续若要求实施，先明确目标构造器、初始化、optimizer-step更新和checkpoint恢复；独立实验只改这一项。固定原始CLIP作为另一候选。当前仅请求记录，不擅自落地这些训练改动。
5. 保留其他候选诊断：完整/随机时间裁剪下的跨模态条件作用、文本内容和结构信息的贡献。方差/关系保持损失不再默认首选；此前推荐过早，不要来回仅凭一个统计量切换结论。空骨架低频不是目前主线。

没有服务器访问，不可声称工具代用户执行过训练；诊断是用户在服务器运行后提供结果。用户结束会话只要求更新本交接文档，未要求启动新训练/任务。

## 9. 代码入口与本地验证

- `cache_clip_text.py`、`util/clip_text_cache.py`：原v2提取/验证，尽量不改身份相关代码。
- `util/person_text_cache.py`：当前逐人mmap reader。
- `model/transformer_macdiff_text.py`：三个loss、remap、共享开关、文本decoder、空人物筛选。
- `model/transformer_macdiff.py`：原生encoder/decoder/FeatureModulation、masked MSE。
- `main_pretrain.py`：模型/缓存/optimizer、epoch loop、保存。
- `engine_pretrain.py`：按原索引查文本、AMP/累积、空人物日志。
- `util/misc.py`：checkpoint与resume校验，torch._six.inf已改math.inf兼容新旧Torch。
- `config/ntu60_xsub_joint/pretrain_madiff_text.yaml`、`script_pretrain_macdiff_text.sh`：配置与启动。
- `tests/test_macdiff_text.py`、`tests/test_clip_text_cache.py`：主要回归测试。
- `diagnose_text_conditioning.py`：当前文本checkpoint只读几何/配对条件诊断；`tests/test_text_condition_diagnostic.py`：6项CPU测试（包含两次完全一致的合成数据端到端运行）；`TEXT_CONDITION_DIAGNOSTIC.md`：协议、使用和解释。
- `tools/vlm_pilot/STAGE1_TEXT_DIFFUSION.md`：详细命令，部分测试计数/服务器状态可能旧，以本handoff为准。

本地Python：`C:/Users/97537/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe`。系统python/py可能WindowsApps占位符，不用它判断成功。
临时依赖目录通过PYTHONPATH启用：`$env:TEMP/macdiff-text-test-py312`（torch2.5.1+cpu、PyYAML）和`$env:TEMP/macdiff-clip-test-py312`（Transformers4.57.1等）。未改原环境/服务器。

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONPATH="$env:TEMP/macdiff-text-test-py312;$env:TEMP/macdiff-clip-test-py312"
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
& 'C:/Users/97537/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -m unittest tests.test_clip_text_cache tests.test_stage1_text_geometry tests.test_macdiff_text tests.test_stage1_readout_torch -q
```

逐人修改时完整35项通过；随后空骨架修复新增1项，只重跑17项模型测试通过（不要说新增后完整36项已经全部重跑）。正式尺寸单人/双人CPU前向此前通过；真实HF小型随机CLIP的safe/bin缓存测试通过；服务器预训练CLIP全量cache由用户确认。

## 10. 绝对不要再踩的坑

1. 不恢复旧跨人物平均/拼接，也不改one_person=False来绕错误。原始YAML注释`#one_person: False`表示使用构造默认True，绝不是启用双人。
2. 不把空person0换成person1；整batch空与单个样本空要区分；过滤以未加增强噪声、未标准化的source判定，aug扰动会把空槽变非零。
3. Bash脚本必须LF、无BOM。曾报 `set: pipefail: invalid option name`，是CRLF。服务器修复命令 `sed -i 's/\r$//' script_pretrain_macdiff_text.sh`。本地曾改LF，Windows/git可能再转回，发送前检查。
4. 不宣称batch64显存已验证；建议32×累积2，命令行显式覆盖；不偷偷把effective batch改32。
5. 不以三任务总loss对比原始总loss；不以低去噪loss证明识别提升；不以能量1证明不坍缩；不以全局方差代表所有token。
6. 不对旧协议checkpoint强行strict=False完整恢复；更换共享/人物/权重需独立实验。恢复不是精确batch续跑，日志会append导致重复epoch。
7. 单/双卡resume可行但双卡未实测，GPU0权限与空闲未确认。不同采样/累积不等于bitwise同轨迹。
8. 不再生成zip；只给变更文件。现有用户改动和旧handoff_artifacts删除状态不能擅自恢复。以每轮git status为准。
9. 不重新生成描述或重建现有v2缓存；cache microbatch大小不等于训练batch，不等于历史关系矩阵batch。
10. 不重新引入放弃的Qwen30B-A3B AWQ/vLLM路线：曾单4090 OOM；vLLM要求Torch2.8与glibc2.27不匹配；不升级驱动/glibc、不装容器系统服务、不重启网络。HF服务器网络曾DNS/路由异常，非简单pip镜像问题。
11. 官方CLIP.pt不能改名当HF.bin；当前缓存已跑通，不换加载体系。无必要不卸载AutoAWQ等已有环境依赖。
12. 旧Stage2/OSE、旧几何预实验“尚未授权训练”的限制已过时；当前明确获准Stage1文本训练，但尚未获准额外防坍缩loss等方案变更。
13. 不将checkpoint-0当随机初始化；不要把三个离散点说成精确定位了最低谷。平均cosine、方差占比在单位能量情形紧密相关，不是多项独立因果证据。
14. 不将“shuffled损失更高”直接等同于动作类别语义改善；错误条件也可能扰动样本特有细节。跨checkpoint反向目标在变，不能直接以S→T loss升高判训练退步。
15. 不再索要已经提供的0/200/350报告，也不重跑诊断来回答已能从报告计算的问题。EMA/固定CLIP只记录了方案，不要声称已实现。

## 11. 历史预实验背景（仅帮助理解选题，不是当前待执行计划）

此前先测过文本CLIP是否比早期骨架具有更好的类别关系。协议：train的100个类别均衡batch、每batch128(K32/P4)、原始6400维关节保留readout、双人汇总；它与现在one_person训练不同。文本AUC约.667，P@1约.191，对比checkpoint0 AUC约.664/P@1约.348，最终骨架P@1约.523，未支持“文本全局关系更可靠”。这促成了现在尝试可训练remap＋条件去噪，而不是直接全局关系蒸馏。

标签仅用于这类采样/诊断，不回填类别名、不据标签删除描述。不能把原6400维readout替换为256全局仍称同协议。`handoff_stage2_legacy_20260905.md`、`handoff_vlm_pilot_legacy.md`仅历史参考。

以下保留旧实验的精确数值，避免新会话重复索要或重跑。

### 历史几何预实验数值

完整原始 summary 已保存至本仓库：**`handoff_artifacts/stage1_text_geometry_summary.json`**。无需让用户重复上传。

原始附件：`C:/Users/97537/.codex/attachments/44ef8996-6a4a-42a2-89fc-0173975a0880/pasted-text.txt`。

本地有全部逐类/逐batch指标，但没有服务器plan、原始特征、GIF或逐样本文本，无法只凭summary重建错误邻居。

| 特征 | AUC | P@1 | P@5 | same cosine | different cosine | gap |
|---|---:|---:|---:|---:|---:|---:|
| CLIP文本 | 0.667381 | 0.191414 | 0.159642 | 0.908117 | 0.866681 | 0.041435 |
| Stage1 0 | 0.664143 | 0.347901 | 0.239186 | 0.976648 | 0.969527 | 0.007121 |
| Stage1 10 | 0.785167 | 0.456910 | 0.336177 | 0.959433 | 0.928223 | 0.031210 |
| Stage1 20 | 0.752455 | 0.436217 | 0.305546 | 0.941285 | 0.907098 | 0.034188 |
| Stage1 50 | 0.744836 | 0.453236 | 0.316345 | 0.928847 | 0.888792 | 0.040055 |
| Stage1 100 | 0.737276 | 0.468434 | 0.323633 | 0.921494 | 0.879805 | 0.041690 |
| Stage1 200 | 0.741194 | 0.487118 | 0.335788 | 0.923336 | 0.884006 | 0.039330 |
| Stage1 399 | 0.774436 | 0.522610 | 0.367702 | 0.952073 | 0.927153 | 0.024920 |

进一步统计：

| checkpoint | 文本AUC更高类数/60 | 文本P@1更高类数 | 文本P@5更高类数 | 文本AUC更高batch数/100 |
|---|---:|---:|---:|---:|
| 0 | 17 | 4 | 8 | 57 |
| 10 | 6 | 1 | 1 | 0 |
| 20 | 11 | 1 | 4 | 0 |
| 50 | 14 | 0 | 4 | 2 |
| 100 | 14 | 0 | 5 | 3 |
| 200 | 11 | 0 | 5 | 3 |
| 399 | 11 | 0 | 4 | 0 |

已经告知用户的结论：

- 相对checkpoint-0，文本AUC只+0.003238，但P@1低15.65个百分点；文本P@1/P@5分别仅1个batch占优。
- 对checkpoint-10和399，全部100个batch的AUC/P@1/P@5都是骨架更高。
- 相对最终checkpoint-399，文本P@1低33.12个百分点；60类中没有任何一类文本P@1更高。
- 文本AUC>0.5、P@1>5.51%，有类别信息，不是完全无用。
- 文本gap=0.0414反而高于最终骨架0.0249，而排序/近邻更差；只报gap会误导。
- 相对最终骨架，文本AUC优势最大的0-based类ID包括57、55、53、56、52、54、50、58，这些类P@1仍弱。尚未核对类别名，不要凭记忆贴动作名称。
- 原假设在当前流水线、已测checkpoint、均衡train batch条件下不成立，不等于否定所有文本编码器/描述方式或未测的最早训练updates。
- 不要因checkpoint-10 AUC高于最终就说模型退化：最终P@1/P@5更高，指标衡量不同。

## 12. 2026-09-18 补充：诊断结果与待讨论的稳定目标方案

本节更新优先于前文的旧路径和“尚未执行诊断”状态。用户确认当前服务器文本训练目录是 `output_dir/ntu60_xsub_macdiff_bidirectional_tokens`。用户已运行新诊断并提供 checkpoint-0、200、350 结果；0 表示完成第1轮，不是随机初始化。

- 诊断脚本：`diagnose_text_conditioning.py`；说明：`TEXT_CONDITION_DIAGNOSTIC.md`；6项本地CPU测试通过，用户结果确认服务器 PyTorch1.8.1+cu111 实际运行成功。
- 三次设置相同：固定256条person0描述，batch8，重复3次，t=100/500/900；253个样本进入等长条件打乱测试，3个无配对样本排除，无空骨架。数据/cache身份、模型、feeder和样本一致。
- 原始附件（不要再次索要）：0=`C:/Users/97537/.codex/attachments/41541f90-6bae-4c9f-ae17-21727f46e12c/pasted-text.txt`；200=`C:/Users/97537/.codex/attachments/41eff282-d7be-4fd4-a68b-59a04554fc62/pasted-text.txt`；350=`C:/Users/97537/.codex/attachments/b2068d81-cad4-430f-b83f-4d45e541c8a0/pasted-text.txt`。
- 全局方差/能量 0→200→350：6.826%→0.328%→1.963%；有效秩19.78→5.74→10.64。局部纯内容方差/能量44.027%→0.446%→3.746%；有效秩43.30→18.54→46.34。内容强烈收缩后部分恢复，不能说持续单调坍缩。
- 全局中心化CLIP关系相关性 .970→.839→.871；局部 .866→.451→.502。全局描述最近邻同类率10.55%→12.50%→10.94%，仅256样本的描述性诊断，不是LP，不证明类别收益。
- t=500整份文本打乱的T→S MSE相对增幅 .110%（0的CI跨零）→.722%→1.940%；骨架条件打乱的S→T增幅约0→7.870%→21.106%。200/350所有报告条件损失差CI均为正。0的S→T损失约1，尚未学好；方差大不等于条件有用。
- S→T跨checkpoint的干净目标/带噪输入也随remap改变，不能将损失或打乱效应变化全部归因于decoder或encoder能力变化。条件打乱效应不证明分类语义收益。
- 用户仅报告过checkpoint200的第一轮LP：原版74.91%，文本67.27%；尚无收敛LP或350 LP，不能把第一轮差距当最终结果。

### 用户要求记录的候选方案（仅记录，尚未修改训练代码或启动实验）

用户设计意图：T→S让在线remap提取运动相关内容；S→T将这些内容传给骨架encoder。当前remap仅由T→S更新，S→T目标在加噪前detach，因此不会直接推动remap变化，但仍面对随在线remap变化的目标。

1. **固定原始CLIP目标对照**：T→S保持在线remap，S→T改为缓存的逐人原始512维全局/局部CLIP特征。反向输入/输出改为512维，内部仍可256维；保持人物对齐和有效token统一平均。缓存L2归一化与现有LN尺度不同，可研究固定乘sqrt(512)维持每通道约单位能量，不能直接换尺度而误判。该方案同时改变目标内容、维度和稳定性，不能将收益只归为稳定目标。
2. **用户进一步提出的动量目标方案**：保留在线remap供T→S训练，另建无梯度的EMA目标映射用于S→T：theta_target = m*theta_target + (1-m)*theta_online。目标按当前描述重新计算，不跨不同人物/样本平均文本。若S→T目标仍包含人物/位置embedding，则这些目标构造参数也必须EMA或明确固定；仅EMA MLP不能稳定整份memory。EMA初始化为在线副本，在真实optimizer更新成功后更新（不是每个累积microbatch；AMP跳步不更新），保存/恢复EMA状态。m尚未决定。它减缓漂移但不阻止长期趋同、不保证语义正确，滞后程度需诊断。反向decoder自身的结构embedding不属于干净目标生成器。
3. 之前讨论过冻结指定checkpoint的完整目标映射作为稳定目标对照，尚未选择或实现。任何新方案使用独立输出和明确加载协议，不绕过原resume校验。

### 除目标漂移之外的待验证问题

- T→S减小坐标噪声预测误差不保证筛选的是类别相关运动语义；带噪骨架本身已有信息，文本可以只提供弱补充。现有后期打乱效应为正但较小，不能称完全忽略文本。
- 原始任务与T→S共享骨架decoder，可能产生参数竞争；S→T与原始任务同时更新encoder，可能产生梯度冲突/量级失衡。尚未测量，不能以loss数值断定梯度强弱。
- 训练骨架随机截取50%～100%时段，文本缓存对应原序列描述，并未随裁剪改写；encoder再遮挡90%token。描述中的动作阶段可能不在可见条件中，影响程度未验证，不能因此擅改基线裁剪或人物协议。
- 反向目标包含所有有效正文/EOS、全局token及额外结构信息；统一平均使长描述贡献更多项，全局一个token可能占比较小。这是现有用户选择，不是实现错误；需要区分能预测的词位/常见措辞与分类有用内容，不擅自改权重。
- 优先结合收敛LP、encoder/共享decoder上的分项梯度诊断决定改法。EMA不能自动修复这些任务设计或对齐问题。

## 13. 本轮可复用命令、论文核对和工作区状态

### 单行命令（服务器项目根目录、macdiff环境、GPU0）

LP命令：改checkpoint编号时同时改输出编号。这里batch64、单卡、accum1；比较双方时都用一致参数。旧4卡脚本的全局batch不同，不要声称与旧4卡模板完全等价。YAML使用lr=.1、warmup0、linprobe2。不是直接`--eval`预训练权重，需要训练分类头。

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python main_linprobe.py --config config/ntu60_xsub_joint/linprobe_madiff.yaml --finetune output_dir/ntu60_xsub_macdiff_bidirectional_tokens/checkpoint-350.pth --output_dir output_dir/text_lp_350 --log_dir output_dir/text_lp_350 --batch_size 64 --accum_iter 1 --epochs 100 --lr 0.1 --seed 0
```

基线使用`output_dir/ntu60_xsub_macdiff/checkpoint-350.pth`并换独立输出目录；不要混入已有log。

诊断（已有0/200/350结果，无需仅为交接重跑）：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python -u diagnose_text_conditioning.py --checkpoint output_dir/ntu60_xsub_macdiff_bidirectional_tokens/checkpoint-350.pth
```

输出自动为checkpoint同目录下`checkpoint-350_text_diagnostic/`，包含`geometry.json`、`summary.json`、`paired_records.jsonl`，拒绝覆盖已有目录；复跑需指定新`--output-dir`。默认256样本/batch8/3次重复/t=100,500,900。严格读取checkpoint args和完整state_dict，验证已有cache和数据文件身份，不需要CLIP模型下载。只支持当前person0协议，不支持基线checkpoint。

诊断细节：固定seed和每个样本的一次训练裁剪，跨repeat重新采mask/noise，正确/打乱之间严格共用target/noise/t/mask；只在有效token原位置完全相同的样本间无自配对交换。反向仅交换骨架memory，不交换待预测文本。正向分别交换global/local/all；局部EOS保留在条件任务中，但几何抽样排除EOS（每样本最多4正文token）。几何排除同样本邻居，中心化关系比较的是相同token身份的前后映射，不是把不同句子的第k词当语义对齐。CI按匹配mini-batch聚类bootstrap，非独立训练种子的置信区间。FP32/eval，不加载optimizer、不训练。

### 两篇用户给定论文已读，不要只凭摘要再次推断

- `D:/program/paper/Enhancing_Skeleton-Based_Action_Recognition_With_Language_Descriptions_From_Pre-Trained_Large_Multimodal_Models.pdf`：第5–6页式15–29，先用真实动作标签交叉熵训练各模态映射/分类器，再用分类概率熵计算归一化`exp(-H)`权重融合。它不是跨样本方差正则，也不是无标签remap防坍缩算法。
- `D:/program/paper/Multi-modality_Progressive_Prompt_Learning_for_Enhancing_Skeleton-Based_Action_Recognition.pdf`：第5页式3–8，`p=softmax(f)`在特征通道上计算熵，`f_hat=exp(-H(p))*f`，然后与另一模态remap特征做InfoNCE，正文明确sim是cosine。按写出的数学公式，正标量缩放在cosine中抵消。已向用户明确说明这是一处需核实作者实现的公式问题，不能声称熵标量能直接抗坍缩，也不据此否定整篇实验。ECMA同时新增跨模态对比损失；整体有分类监督，生成描述还含类别提示，与本项目无标签骨架GIF描述不同。
- 论文相关页已用pypdf提取并用pypdfium2渲染目视核对，临时文件在`tmp/pdfs/remap_review/`。系统runtime无pymupdf，已有pypdf/pdfplumber/pypdfium2，不必为此装依赖。

### 三checkpoint的补充数据与解释

中心化全局top10保留率：83.13%→64.38%→73.40%；局部：76.65%→55.80%→66.35%。0的方差/CLIP关系较好，但尚无整份正确文本的稳定去噪收益。t=500时0的T→S预测扰动MSE=.006274，远大于350的.000307，但正确配对优势不显著：响应幅度大不代表条件利用得好。

S→T整体配对打乱MSE增幅（0/200/350）：t100约0/10.95%/24.62%；t500约0/7.87%/21.11%；t900约0/2.24%/11.50%。T→S整份文本增幅：t100约0/.38%/1.09%；t500 .11%/.72%/1.94%；t900 .09%/1.40%/2.78%。0的T→S整体三点CI跨零，个别分项有小正差；S→T有微小负差，不能误报所有CI跨零。200和350的所有报告差值CI为正。

### 本地验证与已有改动

最近重复执行`python -m unittest tests.test_text_condition_diagnostic -q`，6项通过。使用第9节同一本地Python和临时PYTHONPATH。合成端到端fixture用了p_interval=[.95]，原因是本地新版NumPy对旧feeder的`int(size-1-array)`报错；没有因此修改生产feeder。服务器旧环境真实诊断已经成功，不能将CPU合成验证说成GPU性能测试。

本轮新增诊断三个文件，训练逻辑未因诊断/EMA讨论改动。开始时已有`engine_pretrain.py`、`model/transformer_macdiff_text.py`、`tests/test_macdiff_text.py`和handoff改动，勿覆盖。工作区另有未跟踪`test_cache_v2_discovery.py`、`test_cache_v2_graph.py`及`tests/`内同名测试，非本轮创建、未审查，不删除或宣称是本轮实现。`tmp/`含PDF阅读中间产物。结束前应以git status为准，不擅自提交或回滚。

## 14. 2026-09-20：固定 CLIP 的 S→T 消融（当前默认）

用户报告第一版 text 最终 LP **83.03%**，整体略降；未给本次最终 LP 对应 checkpoint 和原版精确值，不擅自补充。用户认为一次增加太多组件，要求先去掉文本指导骨架扩散，再去掉骨架指导文本分支的目标 MLP，直接使用固定 CLIP 特征。已在本地实现，未启动服务器实验。

### 模型与目标

- `model_args.text_target_mode: fixed_clip`，`lambda_text_to_skeleton: 0.0`，`lambda_skeleton_to_text: 1.0`。此模式不创建 `text_remap`、目标人物/位置 embedding 或 `text_skeleton_decoder`，不是冻结随机 MLP。若误将 T→S 权重设为非零，初始化直接报错。
- 直接复用现有 v2 cache：句向量 `[B,512]` 与局部 token `[B,K,512]` 拼为 `[B,K+1,512]`；保持缓存的 L2 归一化数值，局部 FP16 转 FP32。不做额外 MLP、LN、缩放或目标结构 embedding。padding 清零，目标在加噪前 detach。
- S→T 仍预测噪声 ε，不改成直接回归干净特征。独立采文本时间步和高斯噪声，沿用原扩散 schedule。文本 decoder 的输入 Linear512→256、输出 Linear256→512；骨架 memory 仍为全局 h＋可见 latent，Linear256→256。5层文本 decoder、其中的 FFN、人物/位置结构提示保留；删除的是目标侧 remap MLP。结构提示仅加在加噪后的 decoder 输入上，不混入干净目标。
- 原生骨架去噪、.02 uniformity、mask90%、数据增强、逐人配对、person0-only、空人物排除都不改。文本 MSE 仍对所有有效全局/局部 token 和 channel 一起平均。
- 总 loss：`L_native + 0.02 L_uniformity + 1.0 L_ST`。辅助梯度经 S→T decoder 到骨架 encoder，不进入原生骨架 decoder；无 T→S 对共享骨架 decoder 的梯度。
- `text_energy` / `text_batch_variance` 现在测固定 CLIP 句向量，前者约 `1/512 = .001953125`，不再接近1。缓存范数1与第一版目标LN尺度不同，同一扩散schedule下信噪比也变化；新旧 S→T loss 不直接横比。未加尺度校准、EMA或新loss，以直接CLIP目标作为本轮定义。

### 兼容与启动

类构造器的默认模式仍为 `remap`，供旧 checkpoint 使用其保存的参数重建；正式 YAML 已显式切为 `fixed_clip`。旧版 `diagnose_text_conditioning.py` 用于历史双向 remap checkpoint，尚未适配固定 CLIP 模式，不直接拿它诊断新实验。

`main_pretrain.py` 保存 `args.text_target_mode`；`util/misc.py` 在完整 resume 前校验，缺此字段的旧 checkpoint 视为 `remap`。本轮从头训练，不能 resume 第一版双向权重。后续同模式可正常 resume。新默认输出目录 `output_dir/ntu60_xsub_macdiff_fixed_clip`，避免混入旧日志。已有 CLIP 缓存无需重建。

服务器项目根目录单行命令（有效 batch64，沿用预训练 GPU1；未代用户执行）：

```bash
CUDA_VISIBLE_DEVICES=1 BATCH_SIZE=32 ACCUM_ITER=2 OUTPUT_DIR=output_dir/ntu60_xsub_macdiff_fixed_clip bash script_pretrain_macdiff_text.sh
```

本轮需同步的训练文件：

- `model/transformer_macdiff_text.py`：固定目标模式及独立文本/骨架维度。
- `main_pretrain.py`、`util/misc.py`：checkpoint 目标模式记录与恢复检查。
- `config/ntu60_xsub_joint/pretrain_madiff_text.yaml`：新消融配置。
- `script_pretrain_macdiff_text.sh`：新默认输出目录，已验证 LF、无 BOM。

另更新 `tests/test_macdiff_text.py`、本 handoff。`engine_pretrain.py` 和诊断文件本轮未修改，原有工作区改动保留。未创建zip或提交git。

### 验证结果

使用第9节本地 Python/PYTHONPATH，`python -m unittest tests.test_macdiff_text tests.test_text_condition_diagnostic -q` 共 **26项通过**。新增3项验证固定512维目标与缓存逐元素一致、optimizer更新后目标不变、目标无梯度、S→T独立反传到encoder而不到骨架decoder、空person0不替换、padding不可见、骨架条件影响预测、拒绝旧目标resume及同模式strict恢复。历史双向诊断6项仍通过。

另直接读取正式 YAML，在 CPU 上完成 B=1、120帧、25关节、6个局部CLIP token 的前向和反向：预测 `[1,750,12]`，所有可训练参数都有有限梯度，可训练参数 **19,217,420**。这不是 GPU/AMP/显存测试，不能声称服务器batch32或64已验证。

### 启动缓存校验耗时与快速模式（同日补充）

用户双卡启动停在 `Validating multi-token text cache against the training dataset...`，询问是否可以跳过。旧流程每个rank都完整hash骨架NPZ、所有缓存文件，再逐块校验token数值（token特征约5.89GiB），期间无进度日志；双卡重复执行但只打印rank0日志。无法仅凭这一行判定服务器死锁。

已增加显式 `--skip_text_cache_validation` 开关，默认仍完整验证。快速模式仅检查manifest协议/complete、样本数、数据及缓存文件大小、全部五个NPY数组的shape/dtype，然后mmap读取；不计算哈希、不扫描特征值、不构造旧的跨人物聚合特征。`util/clip_text_cache.py`及缓存生成协议未改。快速模式信任manifest记录的哈希，不能检测同大小的内容更改；适用于本次已验证且未修改的现有缓存。resume身份检查仍保留，开关本身不改变训练目标或checkpoint格式。

本次训练需另同步 `main_pretrain.py`（CLI及加载完成耗时日志）和 `util/person_text_cache.py`（快速加载）。另更新 `tests/test_clip_text_cache.py`、本handoff。`python -m unittest tests.test_clip_text_cache tests.test_macdiff_text -q` **31项通过**，包括快速/完整模式逐人batch完全相同、快速模式不调用哈希或完整校验、仍拒绝样本数/文件大小/数组头不匹配。

双卡各32、累积1、有效batch64的快速启动命令：

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10235 main_pretrain.py --config config/ntu60_xsub_joint/pretrain_madiff_text.yaml --batch_size 32 --accum_iter 1 --output_dir output_dir/ntu60_xsub_macdiff_fixed_clip --log_dir output_dir/ntu60_xsub_macdiff_fixed_clip/tensorboard --skip_text_cache_validation
```

同实验续训时在末尾追加 `--resume 实际checkpoint路径`；正在运行的旧进程不能热切换此开关，需要更新文件后重启。前面的Feeder加载骨架数据仍执行，这个开关只跳过缓存完整校验。
