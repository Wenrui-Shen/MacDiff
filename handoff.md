# MacDiff 会话交接（2026-10-04）

本文写给完全没有上下文的新会话。先读第1节，再看当前命令和注意事项。用户在服务器执行训练；本地助手负责代码检查、日志分析、记录实验及提供命令。

## 1. 接手时必须知道的当前状态

- 任务：在 **NTU60 XSub** 上，用逐样本/逐人物的 global + 六部位文本辅助 MacDiff 骨架自监督预训练，最终用只输入骨架的 linear probe（LP）评价表征。
- 新版文本和 **v3 七句 CLIP cache 已生成完成**，已有多轮完整训练结果。不要机械重跑文本生成、缓存生成或把“链路尚未跑通”当作当前状态。
- 已修复 S→T 文本 decoder 的输出瓶颈：hidden256/末端LayerNorm改为 **hidden512、末端output_norm=none**，内部归一化保留。
- 512、share=True、归一化A、**T→S=1/S→T=0.1** 的400epoch预训练已完成。用户最终更正 **LP best=85.79%**，不是最初说的85.75%；新LP完整日志未提供。
- 用户尝试了该预训练 **checkpoint-150** 的LP：首轮准确率只有“72多”，而checkpoint-399的LP首轮为 **74.65%**。**用户决定不继续150的LP。** 这是未完成实验，不能记录成150最终只有72%，也不能据首轮证明它最终更差。
- **当前用户继续推进的实验：512/share=True/A，T→S=1/S→T=1。** 已提供从头预训练400epoch与LP100epoch的串联命令。用户尚未提供这组的日志、进度或结果；不要假装助手已运行服务器命令。
- **用户最新下一步：先看权重1结果，再根据结果测试归一化B。** 归一化B先于备选no-share；不要沿用之前“先0.3”“继续150 LP”“权重1之后立即no-share”的旧建议。
- 当前没有已确认的训练报错或外部阻塞；主要未解决的是 **文本去噪显著改善尚未转化成稳定LP提升**，以及512 S→T后期回升的原因。正在等待权重1结果，B实验还没创建成对配置/运行。

本轮关闭会话仅更新文档，不更改正在跑的训练配置、模型、缓存或服务器进程。旧版交接归档在 [handoff_before_20261004_session_close.md](D:/program/MacDiff/handoff_artifacts/handoff_before_20261004_session_close.md)，更早历史见 [handoff_before_20261001_refresh.md](D:/program/MacDiff/handoff_artifacts/handoff_before_20261001_refresh.md)。旧文档中的“未生成”“待跑512”“先做150 LP”等状态不再适用。

## 2. 环境、用户偏好与关键路径

| 项目 | 当前信息 |
|---|---|
| 本地仓库 | D:\program\MacDiff，Windows PowerShell |
| 服务器仓库 | /home/user9/public3/swr/MacDiff |
| 数据 | ../data/MAMP/ntu/NTU60_XSub.npz，训练40091个样本 |
| 历史训练环境 | conda macdiff，Python3.8、PyTorch1.8.1+cu111、两张RTX4090 24GB；服务器实际版本优先 |
| 本地Python | C:/Users/97537/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe |
| 本地依赖边界 | 有NumPy/PIL，缺Torch/Transformers/PyYAML；不能声称本机验证了真实autograd、CUDA、AMP、DDP |
| 服务器访问 | 本会话没有SSH；用户自己执行Linux命令 |
| 本地Git | 会话末HEAD为c9dfb87；512修复在该版本。分析/交接文档仍未提交 |
| 服务器Git | 最新日志没有SHA；不要把历史edc924a当作当前服务器版本，也不要无依据声称尚未同步 |

用户用中文交流。命令要 **单行Linux、不要cd**，两卡，预训练和LP用 **&&** 连接。固定训练每卡batch32/accum2，LP每卡batch64/accum1，两者有效batch均128；训练400epoch，LP100epoch、lr0.1、seed0、dist_eval。不要擅自改成LP每卡32。

| 用途 | 路径 |
|---|---|
| 已有双视角骨架渲染 | vlm_pilot/ntu60_xsub_train_rendered_v3_2view_smooth_w5 |
| 新版文本 | vlm_pilot/ntu60_xsub_global_local_v2/captions.json及sidecar |
| 当前CLIP cache | vlm_pilot/ntu60_xsub_clip_sentence_cache_v3 |
| 历史BPE cache | vlm_pilot/ntu60_xsub_clip_cache_v2 |
| Qwen | /home/user9/public3/swr/models/Qwen3-VL-8B-Instruct |
| CLIP | /home/user9/public3/swr/models/clip-vit-base-patch32 |
| 当前预训练YAML | config/ntu60_xsub_joint/pretrain_madiff_text_sentence_sample_target_blend_shared.yaml |
| 当前LP YAML | config/ntu60_xsub_joint/linprobe_madiff.yaml |

上表实验目录是此前提供命令的路径；服务器真实文件/args以用户实际运行记录为准。prompt v2、caption schema v2、CLIP cache v3、render v3是不同组件版本。

## 3. 我们具体在训练什么

每个原始样本中的每个人分别生成一个global和六句local。部位顺序固定：
`head, torso, left_arm, right_arm, left_leg, right_leg`。
每句简短描述对应区域的主要运动，静止区域描述维持的姿态。front/side是同一人物的两个视角；不是两个人。person0红色、person1蓝色。

当前 `one_person=True`，只保留骨架person0和它自己的文本；person0在crop中为空时过滤，不拿person1替代。feeder返回原始sample index供缓存查表，不是shuffle后的batch顺序。

v3缓存每句独立经冻结CLIP pooled projected `text_embeds`，逐向量unit RMS、FP32：
- global：`person_features.npy [N,2,512]`。
- 六句local：`token_features.npy [N,2,6,512]`，名字保留token，内容是完整句向量。
- validity：`person_valid.npy [N,2]`、`token_mask.npy [N,2,6]`。
- 文本/身份：samples.json、manifest.json；缓存约1.07GiB，person0七向量目标库约0.535GiB，另有元数据。

**训练长度是7个向量，不是77个词元。** CLIP的77只是离线每句分词上限；编码padding=True仅补到当次batch最长句，超长报错、不静默截断。当前七句训练不存在补到77的浪费。

当前三条任务：
1. 原生骨架重建：encoder读取90%mask后保留的75个骨架token，decoder预测骨架高斯噪声；维持原uniformity。
2. **T→S**：在线remap文本作为条件，预测同样本骨架噪声；直接训练remap和骨架decoder，**没有直接encoder梯度是用户确认的设计**。
3. **S→T**：保存的文本目标detach后加噪，独立文本decoder读取同一样本的骨架encoder条件并预测文本噪声；梯度进入encoder和文本decoder。

`share_skeleton_decoder=True` 共享的是原生骨架decoder与T→S骨架decoder主体（输入、时空embedding、调制blocks、norm/head）；**S→T文本decoder始终独立**。no-share减少一条共享主体影响路径，但保留T→S→remap→保存目标→S→T→encoder的间接路径；可能有效，尚未证实。

目标模式为 **sample_target_blend**，不是旧参数EMA：
```text
initial_target_i = RMS(CLIP_i)
每次成功optimizer step后：
target_i = 0.9 * old_target_i + 0.1 * current_online_remap(RMS(CLIP_i))
```
global和有效local都递推，用step后的remap，AMP跳步不更新。不要改成0.9固定CLIP+0.1remap；不要把混合后的保存目标强行重新RMS。

当前骨架/文本扩散均预测epsilon，1000步inverse_cosine、均匀t采样。骨架window120、t_patch4、joint_patch1，共750tokens、mask0.9保留75；S→T memory为75个可见token+全局池化，共76，骨架特征维度仍256。crop=[0.5,1]、flip=False、joint_noise=[1,0.005]；没有加入部位到关节的attention mask。

LP只加载骨架encoder，不使用文本/remap/decoder/bank。真正的ActionHeadLinprobe2读出是25关节×256=6400维（人物和时间平均），再BN+Linear；不是feature_only的global256。

## 4. 已完成的代码检查与修复

1. **新版文本/缓存链路接入**：逐样本逐人物、global+六部位、metadata分离、覆盖/身份/RMS/人物有效性校验、v3 reader、固定RMS和逐样本共享目标配置。用户已确认cache完成且提供真实训练日志，生成不再是待办。
2. **目标库续训残留修复**：目标快照先于模型保存，中断可能产生孤立快照；仅模型checkpoint不存在时允许原子重建孤立快照，已配对快照保护，不能覆盖。
3. **padding/读取优化**：v3六句批量gather，保留旧v2部分有效token压紧；目标库更新先过滤无效人物/local再remap，全有效v3走dense路径，不改变目标公式。
4. **LP测试CE聚合修复**：两个evaluate入口改成按样本数加权，避免最后小batch等权偏差；Top1/Top5不受该聚合修复影响。
5. **S→T512输出空间修复**：旧hidden256+末端LN→Linear512最多255维仿射输出，高斯噪声期望误差下限约257/512=0.502。新sentence配置hidden512、output_norm=none解除限制；内部norm保留，历史模型默认仍256/LN以兼容。
6. **日志整理**：取消普通empty骨架日志与text_target_drift_mse，内部空人物过滤保留；保留原uni，variance换为在线global+有效local内容的合并总体方差。

关键实现：`model/transformer_macdiff_text.py`、`util/person_text_cache.py`、`util/sample_text_target_bank.py`、`util/shared_memory_text_target_bank.py`、`engine_pretrain.py`、`engine_linprobe.py`、两份sentence YAML。
缓存生成/校验：`cache_clip_motion_text.py`、`util/structured_text_cache.py`；旧cache_clip_text.py/util/clip_text_cache.py保留。

512 decoder约2759.6万参数，旧256约724.5万；1024约1.076亿，缺少继续扩宽的收益依据。
历史回归：padding75项（45通过/30依赖跳过）；512相关81项（49通过/32依赖跳过），新增4项CPU输出空间/兼容性检查通过，2项真实Torch检查因缺依赖跳过。Python3.8 AST与diff检查通过。**这些是历史开发验证，不是此次交接重跑，更不等于助手真实GPU测试。** 用户512400epoch训练日志提供实际运行成功证据，但无完整args/SHA/逐rank/AMP诊断，不能延伸为所有细节均验证。

## 5. 已有结果、当前问题与证据边界

最新三组都是新版七句、sample_target_blend、共享骨架decoder；权重以日志核算为准，结构以用户说明/当前代码为准：

| 实验 | T→S | S→T | LP best |
|---|---:|---:|---:|
| 旧256/LN首轮 | 0.1 | 0.1 | 85.801795% |
| 旧256/LN权重1 | 1 | 0.1 | 85.892771% |
| 新512/无末端LN | 1 | 0.1 | **85.79%**（用户更正，仅口述best） |
| 新512/无末端LN，当前继续 | 1 | 1 | **尚无结果** |
| 新512的checkpoint-150 LP | 1 | 0.1 | **未完成：仅首轮72多，用户不再继续** |

checkpoint-399的LP首轮74.65也是用户口述；不能与最终85.79混为同一指标。150首轮差距不是最终表征优劣的结论。不要重新催用户把该实验跑完。

512预训练完整400条epoch0..399，全部数值有限、6个local始终有效；LR与旧256/T→S=1每epoch完全相同。总loss确认：
`native + .02*skel_uni + .02*text_uni + 1*T→S + .1*S→T`，最大误差3.55e-9。
bank epoch0=40064、epoch1起40091，首轮未覆盖/采样不是cache漏样本证据。

| loss，最后50epoch均值 | 旧256/T→S1 | 新512/T→S1 |
|---|---:|---:|
| 总loss | 0.08901706 | 0.04063979 |
| native | 0.01439415 | 0.01435490 |
| T→S | 0.01991729 | 0.01989607 |
| S→T | 0.51023321 | 0.02710436 |

512末轮S→T=0.02788974，最低在epoch142为0.00996099，后期为最低的2.80倍：
100..149均值0.010192→150..199的0.010623→200..249的0.012662→250..299的0.017566→300..349的0.023785→350..399的0.027104。
同期native/T→S持续下降，后期与旧版几乎相同。旧256也在约139epoch最低后回升，0.5结构下限掩盖了相对变化。

末轮total0.04085835、native0.01442973、T→S0.01996165；S→T加权为0.00278897，占total约6.83%。总loss较旧版下降约54%，几乎全部来自解除S→T限制，**不能据此证明encoder语义改善、梯度不足或应该按比例补权重**。512 LP85.79比同权重旧版低0.10277个百分点；单次且缺本轮LP完整日志，不能判显著退化，可以明确的是没有显示收益。

候选解释仅是：动态目标方向变化、encoder条件变化、多任务竞争、LR下降后文本分支跟随能力不足。训练日志不能区分，也不能直接叫过拟合或代码错误。
文本uni末约0.168，variance约0.936、目标global energy约1；这些不排除语义/方向问题。

指标口径：
- text_uniformity：同样本有效local两两cosine平方均值，含对角，不含global，不跨样本；六句下限1/6，可能出现固定部位原型，不能称跨样本抗坍缩。
- text_batch_variance：在线remap后的global+有效local内容向量合并，每卡microbatch通道总体方差均值，排除embedding/padding/空人物；部位差异也会抬高它。
- text_energy：保存目标global的能量，和variance不是同一批向量。旧variance曾只统计保存global，不能跨口径比。

更早结果供背景：原MacDiff约85.86%；固定CLIP原L2尺度S→T1/0.1为84.64%/85.95%；固定RMS为约83.7%/85.82%；旧参数EMA no-share/share为85.77%/85.78%。旧EMA共享比较的T→S均0.1、旧BPE local/不同目标；部分历史LP有效batch64，当前128，不能直接宣称稳定收益或分享结论。

## 6. 当前继续的权重1实验与命令

用户选择 **T→S=1、S→T=1**，此前建议0.3已被其选择替代。使用512/share=True/归一化A，从头训练、独立目录；不要将这组混称为“只有T→S=1”。

**YAML当前两个lambda默认仍是0.1，batch默认64/accum1。** 因此必须保留命令里的两个权重覆盖和batch/accum参数。仅看YAML不能判断用户实际权重。

| 当前实验路径 | 目录 |
|---|---|
| 已完成512、S→T0.1对照 | output_dir/ntu60_xsub_macdiff_sentence_sampletarget01_shared_st512_t2s1 |
| 该对照399 LP | 上述目录名后缀_lp_399_bs64 |
| 已放弃的150 LP | 上述目录名后缀_lp_150_bs64 |
| 当前S→T1预训练 | output_dir/ntu60_xsub_macdiff_sentence_sampletarget01_shared_st512_t2s1_s2t1 |
| 当前S→T1 LP | 上述目录名后缀_lp_399_bs64 |

以下是已提供用户的命令，供核对其实际运行；**用户已在继续，不默认重复启动相同目录**：

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10254 main_pretrain.py --config config/ntu60_xsub_joint/pretrain_madiff_text_sentence_sample_target_blend_shared.yaml --lambda_text_to_skeleton 1 --lambda_skeleton_to_text 1 --batch_size 32 --accum_iter 2 --epochs 400 --seed 0 --output_dir output_dir/ntu60_xsub_macdiff_sentence_sampletarget01_shared_st512_t2s1_s2t1 --log_dir output_dir/ntu60_xsub_macdiff_sentence_sampletarget01_shared_st512_t2s1_s2t1/tensorboard && OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10255 main_linprobe.py --config config/ntu60_xsub_joint/linprobe_madiff.yaml --finetune output_dir/ntu60_xsub_macdiff_sentence_sampletarget01_shared_st512_t2s1_s2t1/checkpoint-399.pth --output_dir output_dir/ntu60_xsub_macdiff_sentence_sampletarget01_shared_st512_t2s1_s2t1_lp_399_bs64 --log_dir output_dir/ntu60_xsub_macdiff_sentence_sampletarget01_shared_st512_t2s1_s2t1_lp_399_bs64/tensorboard --batch_size 64 --accum_iter 1 --epochs 100 --lr 0.1 --seed 0 --dist_eval
```

两种CLI都用实际get_args_parser AST抽取验证，无Torch导入；参数可解析、有效batch128、无resume、新LP指向新399checkpoint。未在本地验证服务器checkpoint存在或实际GPU运行。当前代码每10epoch及最后保存一次，完整预训练的末轮为399。

## 7. 下一步：权重1结果 → 归一化B

1. 新会话承接用户的权重1进度/结果。先看实际400epoch训练日志、LP结果；若用户只给best就注明来源，不编造末10轮/分类loss/完整args。
2. 比较512/S→T1与已完成的512/S→T0.1：LP是否有明确收益，S→T后期是否仍回升，native/T→S是否变化。不能只看rawloss或首轮LP。
3. **随后根据结果测试B。** 选择明确父对照：如果权重1更有意义，就用其权重；如果未改善，可保留0.1父对照。当前尚未锁定B实验的权重。其他设置固定、只变骨架input_mean/input_var，独立预训练和LP目录，从头训练。
4. 创建一对独立B YAML（预训练和LP）再给命令；不要修改正在跑的共享基线文件，也不要同时改share、crop、v或uniformity。B实验配置目前尚未创建/运行。
5. no-share、仅S→T的v参数化、六部位软区域bias仍为备选，**当前优先级在B之后**，不机械跑完全组合，不把讨论当作用户已经要求实现。

### 归一化A与B

```yaml
# A：当前全部NTU60 XSub预训练/LP实际使用
input_mean: [-0.0058, -0.1333, -0.0246]
input_var: [0.0206, 0.0805, 0.0218]

# B：pretrain/LP中注释的#new，待测
input_mean: [-0.0024, -0.2132, -0.0446]
input_var: [0.0525, 0.1527, 0.0513]
```

扫描31份YAML共8组mean/var、50处成对出现（31生效、19注释）。A在28份配置生效；B在NTU60 XSub、NTU120 XSub/XSet的pretrain/LP共6处注释，**不是NTU60独有，也不能证明原始统计来自NTU120**。初始提交已有这些值，未找到计算脚本/统计来源。
NTU60 finetune的#new是另一组C：mean=[-0.0034,-0.1322,-0.0271]、var=[0.088325,0.106987,0.065367]，**不是本次要测的B**。

实际归一化是(raw-mean)/sqrt(var)。A→B时同骨架XYZ波动幅度变为原来的0.626/0.726/0.652，中心化方差及同t有效SNR变为0.392/0.527/0.425；噪声仍单位高斯。可能使epsilon更容易预测而降低loss，不能因此认定LP更好或B更正确。

B只改骨架归一化，复用文本cache。PT/LP都必须B；这些值是Python普通属性、不在state_dict里，LP加载旧checkpoint不会覆盖YAML统计。只给旧A权重的LP换B不能验证B预训练收益。含空人物、padding、person0/两人、crop/旋转的统计口径未知，不凭#new标签定性。

## 8. 用户明确的研究约束与备选方案

- 暂不考虑对比学习及类似batch结构/关系相似性蒸馏；**不做跨样本内容**。
- 冻结CLIP作为扩散干净目标已经试过；不要当新方向重复推荐。
- 固定t/噪声后条件打乱，以及部位跨样本方差/相似度检查已经做过；不要再把重复诊断列为默认下一步。
- 保留原uniformity是用户选择，不擅自加VICReg、跨样本方差正则或额外辅助loss。
- 单样本v和区域bias仅讨论，未实现。当前文本模型要求全局diff_prediction=noise，不能直接改全局v来实现“仅S→T改v”；需要独立S→T开关。
- v=a_t*epsilon-sigma_t*text_target，改变时间步监督权重，不增加信息、没有LP保证；它不同于更换干净目标来源。原生/T→S应保持epsilon。epsilon/v的rawloss不可直接比。
- 区域bias仅作读取先验：local优先对应关节、保留全局key和空区域回退；随机mask打乱token后必须用原始joint身份，不能把75个可见token按顺序分六段。encoder全身self-attention已混合信息，不能宣传严格局部隔离。
- 全视频描述和随机[0.5,1]crop/90%mask可能信息不匹配，但未证明是当前主因；完整序列/带时间边界的文本方案未实施。

## 9. 绝对不要再踩的坑

1. **不要git clean -fd/-fdx服务器。** vlm_pilot/包含生成文本、渲染、缓存，未跟踪不代表垃圾。历史reset --hard只因当时用户明确要求，不能泛化；也不要为clean status删cache、目标库、checkpoint。
2. 当前cache已完成，不能重生成或改caption来“修”未证实的性能问题。生成文件和captions.metadata.json、诊断/分片sidecar保留；不得cat两个JSON数组当合并。
3. 缓存身份SHA绑定实现：cache_clip_motion_text.py、util/structured_text_cache.py、cache_clip_text.py、util/clip_text_cache.py。**避免与任务无关的改动（包括格式），否则可能触发已有cache身份不匹配。** 归一化B只需YAML，别改这些文件。
4. 缓存查表必须原始sample/person身份；空person0不替换person1，front/side不当两人。左右部位顺序不改；v3禁止flip=True，除非同时实现文本交换/方向转换。
5. 77是CLIP句长上限，训练是7个句向量；不要把旧BPE缓存/EOS重复问题与新六句混同。
6. **512修复必须从头训练，不能完整resume旧256权重。** 旧256骨架encoder仍可做LP；不同缓存/目标模式/share/权重实验不能用strict=False假称完整恢复。
7. sample_target完整resume需要同次保存的checkpoint-X.pth与checkpoint-X-target-bank.sqlite；current.shared.json不是完整快照。LP只要模型，不需bank。只重建没有模型对应的孤立快照，不覆盖配对数据。
8. 目标更新严格0.9历史+0.1当前step后remap，AMP跳步不更新；不改成固定CLIP混合或每forward更新，不给保存混合目标额外RMS。
9. 两个lambda默认都0.1，CLI覆盖别漏。**当前“权重1”是S→T=1，T→S也保留1**；区别历史只把T→S改1的实验。
10. 命令不加cd，PT与LP用&&。历史曾把cd误拼成argparse参数导致整次没跑；父进程CalledProcessError要找前面真正rank traceback。保持每卡32/accum2与LP64/accum1。
11. share只连接native与T→S骨架decoder，S→T独立；T→S无直接encoder梯度是设计，别当bug改变。
12. 约0.502是旧输出子空间的高斯期望下限，不是有限batch硬界，也不是LP上限。去噪loss/variance/energy更好不等于LP好；loss占比不等梯度占比。
13. text_uni含local对角、六句下限1/6；在线合并variance和保存global energy不能混比，旧variance口径也不同。删除empty日志不等于取消内部过滤。
14. **LP首轮72多 vs74.65不能替代最终结果。** 用户不继续150LP；不要记录完整失败分数，不强推重跑或把早期更好/更差作为确定结论。
15. 512基线最终是**85.79**，不要恢复成85.75。最新S→T1尚无结果，B也未跑；不得编造成绩或断言显著退化。
16. B必须PT/LP成对使用、独立目录、从头；不能只在旧权重LP换B。不同mean/var数组出现次数、#new标签不证明统计适配当前person0/crop。
17. 本地缺Torch等依赖，CPU/AST/替身测试与用户真实GPU训练证据区分；文档变更无需反复跑缺依赖大套件，不为验证文档安装Torch。
18. 旧诊断脚本适配模式不同（参数EMA、fixed_clip、旧remap），不能未经适配直接读v3/sample_target/512并解释。当前不重复shuffle/geometry。
19. 早期同步记录不能当当前状态；确认实际SHA/args时再处理，不机械重复要求用户同步已提交代码，也不擅自停止权重1训练。
20. 原始渲染每帧减去person0 root、丢失世界位移；global文本不能恢复未展示的信息。文本格式合法或RMS≈1不证明语义质量。

## 10. 日志、资料与建议阅读顺序

最新对比数据：
- 256/T→S0.1 PT：[22235368附件](C:/Users/97537/.codex/attachments/22235368-fb22-4b3a-b4bb-9a3e2de686df/已粘贴的文本.txt)；LP：[d2e7f9e1附件](C:/Users/97537/.codex/attachments/d2e7f9e1-4468-49bf-8cc1-7d9a30f822c4/已粘贴的文本.txt)。
- 256/T→S1 PT：[81d248b4附件](C:/Users/97537/.codex/attachments/81d248b4-c708-4b84-ac44-3cf027e8f732/已粘贴的文本.txt)；LP：[6cc77fa4附件](C:/Users/97537/.codex/attachments/6cc77fa4-58fa-4466-9fa3-8afb35f169f9/已粘贴的文本.txt)。此前fe23bb66/a64cb5b7重复上传内容相同，不是另一个实验。
- 512/T→S1/S→T0.1 PT：[64857d73附件](C:/Users/97537/.codex/attachments/64857d73-03e2-4bf9-97ed-e6ca4fba0355/已粘贴的文本.txt)；LP只有用户口述85.79，150/399首轮也是口述。

按需读：
1. [当前实验计划](D:/program/MacDiff/tools/vlm_pilot/STAGE1_TEXT_EXPERIMENT_PLAN.md)：最新用户路线覆盖此前建议。
2. [512日志分析](D:/program/MacDiff/tools/vlm_pilot/ST512_TRAINING_LOG_ANALYSIS.md)：完整数值、回升趋势与证据边界。
3. [归一化审计](D:/program/MacDiff/tools/vlm_pilot/INPUT_NORMALIZATION_AUDIT.md)：8组、31YAML、50处完整位置。
4. [链路审计](D:/program/MacDiff/tools/vlm_pilot/TEXT_CACHE_CHAIN_AUDIT.md)：实现与缓存定义；其早期“待生成/未跑GPU/同步后smoke”是历史状态，以本handoff为准。
5. 当前两个YAML及对应模型/engine/target bank代码；仅具体报错时读相关测试。
6. 更早日志/生成细节才读归档交接、STAGE1_TEXT_DIFFUSION.md/STAGE1_TEXT_GEOMETRY.md；这些包含旧BPE/旧EMA及旧命令，不是当前执行清单。

新会话应从“权重1进度或结果”承接，之后协助建立B的成对配置与单变量对照。**不要重新开始生成cache，不要继续催150LP，不要先跑0.3/no-share，也不要擅自实施跨样本或新目标结构。**
