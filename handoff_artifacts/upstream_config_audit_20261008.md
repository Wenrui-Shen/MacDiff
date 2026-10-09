# MacDiff 官方配置与当前仓库对照（2026-10-08）

官方来源：[LehongWu/MacDiff](https://github.com/LehongWu/MacDiff)，固定提交 [692888f1e2a4227216511aac6733dff31b587213](https://github.com/LehongWu/MacDiff/commit/692888f1e2a4227216511aac6733dff31b587213)（2025-07-06）。本地 HEAD：38016d82ef40db3f3f8e7779a6be49725ea10892。比较当前工作区，未改训练配置或模型。

## 核查范围与结论

官方共有 18 份 YAML，本地有 35 份：18 份对应基础配置，17 份新增实验配置。已直接读取官方固定提交的全部 18 份文件，逐个对照本地所有生效字段。唯一差异是 data_path：官方 data/... 改为本地 ../data/MAMP/...，共 30 个 train/val 路径字段。统一换行、替换这些路径并忽略尾部空白后，18 份文件连注释也一致。

该结论包含归一化、self_shift、人物设置、窗口、增强、protocol、学习率、batch、epoch、diffusion schedule 等全部 YAML 字段。实际运行仍需考虑命令行覆写；本地模型还增加了 OSE、文本和 Stage2 等流程，配置一致不表示所有算法分支均与原版相同。

## one_person 的实际行为

- 官方 PT 构造器默认 one_person=True：[官方 L238](https://github.com/LehongWu/MacDiff/blob/692888f1e2a4227216511aac6733dff31b587213/model/transformer_macdiff.py#L238)。六份基础 PT YAML 的 #one_person: False 都是注释；前向 [L559](https://github.com/LehongWu/MacDiff/blob/692888f1e2a4227216511aac6733dff31b587213/model/transformer_macdiff.py#L559) 执行 x[...,0:1]，固定用第一个人物槽位。本地同样默认 True，位置为 model/transformer_macdiff.py:279、754。
- 官方 LP/FT 的 downstream 构造器没有 one_person 参数；[前向 L340](https://github.com/LehongWu/MacDiff/blob/692888f1e2a4227216511aac6733dff31b587213/model/transformer_downstream.py#L340) 将人物展开为 N*M，共享编码器独立编码，再恢复人物维。标准 NPZ 有两个槽位，空槽位未单独过滤。
- 六份基础 LP YAML 均选 linprobe2：[head L95](https://github.com/LehongWu/MacDiff/blob/692888f1e2a4227216511aac6733dff31b587213/model/transformer_downstream.py#L95) 对人物与时间取均值、保留关节、展平 J*C 后分类。[FT head L115](https://github.com/LehongWu/MacDiff/blob/692888f1e2a4227216511aac6733dff31b587213/model/transformer_downstream.py#L115) 同样保留关节并对时间、人物取均值，再经过分类头。
- 本地 transformer_downstream.py 统一换行和尾部空白后与官方全文相同；基础原生 PT 的人物截断和标准化规则保持一致。

## 归一化执行方式

所有基础配置 feeder normalization=False，但模型内部仍逐 XYZ 通道执行 (x-input_mean)/sqrt(input_var)。self_shift=True 时，先对每个人的所有帧和关节求 XYZ 均值并中心化，同时把 input_mean 强制设为 [0,0,0]；input_var 保留。PT 的干净与增强分支减同一个干净分支中心。

执行依据：[PT 初始化 L322](https://github.com/LehongWu/MacDiff/blob/692888f1e2a4227216511aac6733dff31b587213/model/transformer_macdiff.py#L322)、[PT 前向 L570](https://github.com/LehongWu/MacDiff/blob/692888f1e2a4227216511aac6733dff31b587213/model/transformer_macdiff.py#L570)、[LP/FT L345](https://github.com/LehongWu/MacDiff/blob/692888f1e2a4227216511aac6733dff31b587213/model/transformer_downstream.py#L345)。这些统计量是普通 Python 属性，不在模型 state_dict 内，加载权重不会自动覆盖 YAML 的统计量。

记号沿用现有 INPUT_NORMALIZATION_AUDIT.md：

| 组 | input_mean | input_var | 用途 |
|---|---|---|---|
| A | [-0.0058,-0.1333,-0.0246] | [0.0206,0.0805,0.0218] | 大多数基础配置的旧组 |
| E | [-0.0021,-0.2198,-0.0404] | [0.0298,0.1424,0.0358] | NTU120 XSub FT |
| F | [-0.0016,-0.2247,-0.0321] | [0.0306,0.1451,0.0398] | NTU120 XSet FT |
| H | [-0.0459,-0.0680,1.3642] | [0.2356,0.2351,1.9492] | PKUv2 PT |
| B | [-0.0024,-0.2132,-0.0446] | [0.0525,0.1527,0.0513] | 官方 #new 注释候选；本地 norm_b 两文件启用 |

## 全部 18 份基础配置

PT=预训练、LP=线性评估、FT=微调。“两槽位”表示下游使用全部输入人物槽位。self_shift=True 时，组内 mean 实际覆盖为零、var 保留。全部行与官方仅路径不同。

| 协议 | 阶段 | 统计量 | self_shift | 人物处理 | train random_rot | num_classes |
|---|---|---|---|---|---|---|
| NTU60 XSub | PT | A | False | person0 | False（注释） | — |
| NTU60 XSub | LP | A | False | 两槽位 | True | 60 |
| NTU60 XSub | FT | A | False | 两槽位 | True | 60 |
| NTU60 XView | PT | A | False | person0 | True | — |
| NTU60 XView | LP | A | False | 两槽位 | True | 60 |
| NTU60 XView | FT | A | False | 两槽位 | True | 60 |
| NTU120 XSub | PT | A | False | person0 | True | — |
| NTU120 XSub | LP | A | False | 两槽位 | True | 120 |
| NTU120 XSub | FT | E | False | 两槽位 | True | 60（官方已有） |
| NTU120 XSet | PT | A | False | person0 | True | — |
| NTU120 XSet | LP | A | False | 两槽位 | True | 120 |
| NTU120 XSet | FT | F | False | 两槽位 | True | 60（官方已有） |
| PKUv1 XSub | PT | A | True | person0 | True | — |
| PKUv1 XSub | LP | A | True | 两槽位 | True | 51 |
| PKUv1 XSub | FT | A | False | 两槽位 | True | 51 |
| PKUv2 XSub | PT | H | False | person0 | True | — |
| PKUv2 XSub | LP | A | False | 两槽位 | True | 51 |
| PKUv2 XSub | FT | A | True | 两槽位 | True | 51 |

官方文件证据：

- [NTU60 XSub PT](https://github.com/LehongWu/MacDiff/blob/692888f1e2a4227216511aac6733dff31b587213/config/ntu60_xsub_joint/pretrain_madiff.yaml)
- [NTU60 XSub LP](https://github.com/LehongWu/MacDiff/blob/692888f1e2a4227216511aac6733dff31b587213/config/ntu60_xsub_joint/linprobe_madiff.yaml)
- [NTU60 XSub FT](https://github.com/LehongWu/MacDiff/blob/692888f1e2a4227216511aac6733dff31b587213/config/ntu60_xsub_joint/finetune_madiff.yaml)
- [NTU60 XView PT](https://github.com/LehongWu/MacDiff/blob/692888f1e2a4227216511aac6733dff31b587213/config/ntu60_xview_joint/pretrain_madiff.yaml)
- [NTU60 XView LP](https://github.com/LehongWu/MacDiff/blob/692888f1e2a4227216511aac6733dff31b587213/config/ntu60_xview_joint/linprobe_madiff.yaml)
- [NTU60 XView FT](https://github.com/LehongWu/MacDiff/blob/692888f1e2a4227216511aac6733dff31b587213/config/ntu60_xview_joint/finetune_madiff.yaml)
- [NTU120 XSub PT](https://github.com/LehongWu/MacDiff/blob/692888f1e2a4227216511aac6733dff31b587213/config/ntu120_xsub_joint/pretrain_madiff.yaml)
- [NTU120 XSub LP](https://github.com/LehongWu/MacDiff/blob/692888f1e2a4227216511aac6733dff31b587213/config/ntu120_xsub_joint/linprobe_madiff.yaml)
- [NTU120 XSub FT](https://github.com/LehongWu/MacDiff/blob/692888f1e2a4227216511aac6733dff31b587213/config/ntu120_xsub_joint/finetune_madiff.yaml)
- [NTU120 XSet PT](https://github.com/LehongWu/MacDiff/blob/692888f1e2a4227216511aac6733dff31b587213/config/ntu120_xset_joint/pretrain_madiff.yaml)
- [NTU120 XSet LP](https://github.com/LehongWu/MacDiff/blob/692888f1e2a4227216511aac6733dff31b587213/config/ntu120_xset_joint/linprobe_madiff.yaml)
- [NTU120 XSet FT](https://github.com/LehongWu/MacDiff/blob/692888f1e2a4227216511aac6733dff31b587213/config/ntu120_xset_joint/finetune_madiff.yaml)
- [PKUv1 PT](https://github.com/LehongWu/MacDiff/blob/692888f1e2a4227216511aac6733dff31b587213/config/pkuv1_xsub_joint/pretrain_madiff.yaml)
- [PKUv1 LP](https://github.com/LehongWu/MacDiff/blob/692888f1e2a4227216511aac6733dff31b587213/config/pkuv1_xsub_joint/linprobe_madiff.yaml)
- [PKUv1 FT](https://github.com/LehongWu/MacDiff/blob/692888f1e2a4227216511aac6733dff31b587213/config/pkuv1_xsub_joint/finetune_madiff.yaml)
- [PKUv2 PT](https://github.com/LehongWu/MacDiff/blob/692888f1e2a4227216511aac6733dff31b587213/config/pkuv2_xsub_joint/pretrain_madiff.yaml)
- [PKUv2 LP](https://github.com/LehongWu/MacDiff/blob/692888f1e2a4227216511aac6733dff31b587213/config/pkuv2_xsub_joint/linprobe_madiff.yaml)
- [PKUv2 FT](https://github.com/LehongWu/MacDiff/blob/692888f1e2a4227216511aac6733dff31b587213/config/pkuv2_xsub_joint/finetune_madiff.yaml)

## 本地新增 17 份配置

| 配置 | 数量 | 统计量 | 人物设置 | 与原版输入设置关系 |
|---|---:|---|---|---|
| pretrain_madiff_text*.yaml | 12 | A | 显式 one_person=True | 与 NTU60 XSub PT 的统计量、人物选择相同；增加文本方法 |
| pretrain_madiff_ose_peer.yaml | 1 | A | 显式 one_person=True | 同输入设置；增加 OSE 方法 |
| pretrain_madiff_stage2.yaml | 1 | A | one_person=True，self_shift=False | 同统计量、人物选择；独立 Stage2 方法 |
| pretrain_madiff_stage2_dense_ose.yaml | 1 | A | one_person=False，self_shift=False | 双人；与原版 PT 人物选择不同 |
| pretrain_madiff_norm_b.yaml | 1 | B | 默认 one_person=True | 启用原版中被注释的 B |
| linprobe_madiff_norm_b.yaml | 1 | B | 下游两槽位 | 与 norm_b PT 配套；与原版 LP 统计量不同 |

## 原版既有设置及运行时差异

1. 官方两份 NTU120 FT YAML 的 num_classes 都为 60，本地原样继承。用于完整 120 类微调时分类头与标签不匹配，需要核对实际实验是否另行覆写为 120。此次未修改配置。
2. NTU120 FT 使用 E/F，而 PT/LP 使用 A；PKUv1/PKUv2 的统计量及 self_shift 也有跨阶段差异。均来自官方，不是本地迁移引入。符合官方不代表跨阶段归一化相同。
3. 官方 #new 的 B、NTU60 FT 的另一组 C（mean=[-0.0034,-0.1322,-0.0271]、var=[0.088325,0.106987,0.065367]）、NTU60 XView 的候选 D 均为注释。注释标签不能证明统计来自哪个数据集、是否包含空人物或 padding；公开代码未找到对应统计计算脚本及明确统计范围。
4. 官方三份 .sh 都没有覆写 one_person、self_shift 或 input_mean/input_var。当前 native PT 脚本默认 2 卡，官方默认 4 卡；每卡 batch_size=32、accum_iter=1，因此默认全局 batch 为 64 对 128。本地可通过 NPROC_PER_NODE/CUDA_VISIBLE_DEVICES 改变此默认。
5. PT YAML 的 min_lr=5e-4，但官方及当前 native PT 脚本均传 --min_lr 1e-5；直接按 YAML 启动与按脚本启动需要区分。其他 native PT 脚本显式训练参数（batch_size、accum_iter、epochs、lr、min_lr、mask_ratio、model、config）相同。见 [官方脚本](https://github.com/LehongWu/MacDiff/blob/692888f1e2a4227216511aac6733dff31b587213/script_pretrain_madiff.sh)。

验证方式：源代码、全部对应 YAML 生效字段和完整文本的静态对照。未运行预处理或训练。

