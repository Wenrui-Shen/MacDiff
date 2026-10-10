# 原版 MacDiff：PT 随机旋转对照

脚本：`script_pretrain_madiff_rotation_ablation.sh`；实现：`tools/run_macdiff_rotation_ablation.py`。数据集为 NTU60 XSub、joint 单流。此实验使用原生 MacDiff，不使用文本分支。先从头训练无旋转组并完成 LP，再从头训练旋转组并完成 LP；两组唯一配置差异是 PT 的 `train_feeder_args.random_rot=False/True`。

在服务器 MacDiff 仓库根目录、已激活的 macdiff 环境里执行，无需 `cd`：

```bash
CUDA_VISIBLE_DEVICES=0,1 bash script_pretrain_madiff_rotation_ablation.sh
```

默认生成带时间戳的新实验根目录。也可明确命名一个尚不存在的目录：

```bash
CUDA_VISIBLE_DEVICES=0,1 bash script_pretrain_madiff_rotation_ablation.sh --output-dir output_dir/native_rotation_dec3_ep500_seed0
```

| 项目 | 两组固定设置 |
|---|---|
| PT 模型 | model.transformer_macdiff.Transformer；encoder 8层/256维；decoder 3层/256维；heads8、MLP ratio4 |
| PT 训练 | 500轮，从初始化开始；AdamW lr=1e-3、min_lr=1e-5、weight_decay=.05、betas(.9,.95)；warmup20、末20轮到min_lr；AMP开启 |
| PT batch | 双卡，每卡64、accum1，总128；seed0；num_workers10 |
| 骨架输入 | A：mean[-.0058,-.1333,-.0246]、var[.0206,.0805,.0218]；PT/LP一致；self_shift=False；120帧、25关节、4×1 patch |
| PT 任务 | epsilon预测；1000步、inverse_cosine tau1；mask .9；native条件drop .1；uniformity .02；无loss重加权；one_person=True（person0） |
| PT 增强 | crop p_interval[.5,1]；joint_noise[1,.005]；source_rot=False、flip=False；其他噪声0。仅公共random_rot在两组间改变 |
| LP checkpoint | 同组PT的checkpoint-499.pth；PT/LP输出目录分别保存 |
| LP 训练 | frozen encoder、linprobe2（6400维BN+Linear60）；100轮；SGD lr=.1、momentum.9、weight_decay0；warmup0、min_lr0、末10轮到0；AMP关闭、无label smoothing |
| LP batch | 双卡，每卡128、accum1，总256；与之前T14的LP组织相同；每轮评价、dist_eval；seed0 |
| LP 增强 | 两组训练random_rot=True，测试False；训练crop[.5,1]、测试[.95]；不加PT的小高斯噪声 |

公共 random_rot 对同片段所有帧/关节/人物应用同一个旋转，每轴角度约±.3弧度；先crop再旋转，之后生成带噪encoder输入。干净骨架目标与encoder输入共享旋转。它不等同于仅额外旋转encoder输入的source_rot。

脚本保留现有训练入口/模型/feeder和scheduler，不改算法。配置在运行时生成JSON快照，由现有yaml.FullLoader直接读取。启动前检查当前源码是否接受全部参数、两组是否只差旋转、GPU数、依赖和NPZ头中的train40091/test16487/60类/150坐标形状；记录数据SHA256、代码SHA256、配置SHA256、Git状态、环境及命令。每阶段开始时检查源码/配置未改变；PT结束时确认500轮日志、末端lr=1e-5、checkpoint epoch499、实参和3层decoder。

完成后终端输出两组的 **best LP acc及对应epoch**，再输出 **rot−no_rot的百分点差**。同时保存 `rotation_comparison.json`、`rotation_comparison.csv`；附带last、末20轮均值和总体标准差。准确率直接读取完整LP0–99的JSON日志，不从终端四舍五入后的Max accuracy取值。缺轮、重复epoch、非有限分数或阶段失败会报错，不生成完整对照成绩。

输出结构：

```text
<实验根目录>/
  configs/{no_rot,rot,lp}.json
  no_rot/pt/{console.log,log.txt,checkpoint-499.pth,...}
  no_rot/lp/{console.log,log.txt,...}
  rot/pt/{console.log,log.txt,checkpoint-499.pth,...}
  rot/lp/{console.log,log.txt,...}
  run_manifest.json
  rotation_comparison.json
  rotation_comparison.csv
```

脚本拒绝复用已有实验根目录，保留失败证据。当前版本不自动resume。端口默认10270，可用`--master-port`修改；数据路径可用`--data-path`修改；`--seed`和`--num-workers`统一作用于两组。只检查配置/命令、不写文件或训练：

```bash
CUDA_VISIBLE_DEVICES=0,1 bash script_pretrain_madiff_rotation_ablation.sh --dry-run
```

两组完成后可独立重新汇总：

```bash
python tools/run_macdiff_rotation_ablation.py --summarize-only --output-dir output_dir/native_rotation_dec3_ep500_seed0
```

当前双卡协议的总PT128/LP256与官方四卡总batch相同，但LP局部BN batch128不同于官方四卡的64，不能声称逐位等价；两组之间固定相同。dist_eval对16487条测试数据补1个样本，两组同口径。单seed结果只描述这次运行的差距，不代表多seed稳定收益。

本地验证：Bash语法、Python3.8语法兼容、参数/构造器AST检查、dry-run以及8项CPU测试（准确率精度与最早并列best、异常日志拒绝、JSON/CSV输出、单因素约束、子进程错误传播、四阶段完整流程、失败停止、旧目录保护）通过。本地无Torch/训练数据/CUDA，尚未执行真实PT/LP；目前没有这两组的新best acc。
