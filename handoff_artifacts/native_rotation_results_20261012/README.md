# 原版随机旋转结果（2026-10-12）

来源是用户粘贴的rotation_comparison JSON；完整PT/LP日志、run_manifest、服务器SHA、环境、数据与checkpoint实参尚未提供。本次保存完整数值与服务器路径，没有把原始summary改成已审阅完整日志。

| ID | PT旋转 | LP best | best epoch | last | 末20均值 | 末20总体std |
|---|---|---:|---:|---:|---:|---:|
| T17 | False | 85.638039799 | 64 | 85.474284367 | 85.557678338 | 0.042406356 |
| T18 | True | 85.789665239 | 88 | 85.619844774 | 85.684133938 | 0.044831925 |

旋转−不旋转：best **+0.151625439个百分点**，last **+0.145560408个百分点**，末20均值 **+0.126455600个百分点**。三个摘要口径都提高，说明不只best发生变化；但收益较小，单次比较不能证明多seed稳定收益。末20epoch标准差不等于独立seed方差，不能据此做显著性判断或置信区间。

按脚本协议，两组native decoder3/PT500、lr1e-3/min_lr1e-5、A、PT每卡64/accum1（双卡有效128）；checkpoint499 LP100、每卡128/accum1（双卡有效256）、lr.1。LP训练旋转True、测试False；组间只改PT random_rot。这些是准备脚本的协议，尚未独立核验服务器actual args。用户可在runner命令行修改seed，summary没有seed字段，因此不把默认seed0写成实测已核验值。

这次结果支持：**PT旋转单独不足以补上复现差距**。T18距用户提供的第三方86.2%仍为0.410334761个百分点；不将第三方记录当成完整可审阅结果，也不声称硬件导致该差距。

T14 min_lr5e-4的85.704755%与本次no_rot85.638040%不能直接证明降低min_lr无效；仍缺两次运行的源码、数据、初始化和完整实参身份。T12文本86.196021%与本次native使用不同LP batch和预训练结构，不能把全部差距归于文本收益。

## 建议的下一步：复用T18 encoder，仅测试LP batch64

第三方双卡3090的PT每卡32→有效64已由用户确认；LP每卡64→有效128仍是用户估计。本次脚本有效PT128/LP256与这套双卡参考有差异。先从成本较低的LP对照入手：复用T18的rot/pt/checkpoint-499.pth，保存的configs/lp.json继续控制seed、LR、epoch、归一化和增强，仅将batch128→64，accum1保持。

改变LP batch同时改变局部BN、global梯度batch、每epoch更新次数及增强/RNG的组织，不能称为纯BN消融；这个测试用于检查LP协议是否贡献当前差距。预训练encoder固定，不需要再跑500epoch。

已准备[推荐LP命令脚本](recommended_lp_bs64.sh)，用户于10月12日要求输出此LP命令及新的PT&&LP命令；均已准备；LP64后来已由用户报告完成（T19），新的官方PT&&LP尚无回传状态。脚本在仓库根目录执行，拒绝覆盖已有lp_bs64_compare目录，复用当前run的JSON配置；不会将seed或LR另行覆写：

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10272 main_linprobe.py --config output_dir/native_rotation_dec3_ep500_20261011_012631_490772/configs/lp.json --finetune output_dir/native_rotation_dec3_ep500_20261011_012631_490772/rot/pt/checkpoint-499.pth --batch_size 64 --accum_iter 1 --dist_eval --output_dir output_dir/native_rotation_dec3_ep500_20261011_012631_490772/rot/lp_bs64_compare --log_dir output_dir/native_rotation_dec3_ep500_20261011_012631_490772/rot/lp_bs64_compare/tensorboard
```

若LP64仍无明显改善，之后才考虑在同一服务器、同一NPZ上跑干净官方代码的双卡默认协议（每卡PT32、decoder5、400epoch、脚本min_lr1e-5），作为整体复现检查；这不是现有3层/500epoch旋转对照的继续，也不是已执行实验。完全匹配协议后仍有差距，才有依据进一步检查版本、AMP跳步与运行时数值设置。

原始[summary](user_summary.json)与[精确差值和证据边界](analysis.json)已保存。实验总数现为18，第三方86.2参考不占实验ID。未修改训练源码、配置或原旋转runner，未训练。

## 2026-10-12追加：两条命令已按用户要求准备

现有encoder复评命令与新的双卡默认PT&&LP精确文本见[commands.txt](commands.txt)、[commands.json](commands.json)。新组调用[隔离官方源码](../upstream_692888f/ISOLATION_README.md)，只补同值inf导入兼容及数据路径，保留官方DDP；默认decoder5/400epoch、PT32/LP64、accum1、seed0、PTmin_lr1e-5。LP载入checkpoint399；不是3层/500轮组。不切换当前分支，不覆盖现有旋转输出；先同步官方副本到服务器。新脚本clean_default_pt_lp.sh拒绝复用已有输出根目录。仅准备、未执行，当前仍18个有成绩实验。

## T19：复用旋转encoder，LP64口述85.89%

用户于2026-10-12回传LP64成绩85.89%，明确复用T18 checkpoint499、其他沿用保存配置。相对LP128的best85.789665239%约+0.10pp；best/末轮口径、epoch、last/末20、actual seed与完整日志尚未提供。新LP配置不是同配置的随机重复，不能据此估计seed波动或确认batch收益稳定。此项只新增LP评价、无新PT；当前共19条数值记录。原始[口述身份与边界](lp_bs64_user_report.json)已保存。新官方默认PT&&LP尚无回传结果；不推断服务器是否已运行。

## 2026-10-13：新官方默认PT&&LP已回传T20

默认双卡PT400与两次LP100已提供完整日志；两次同seed0、目录lp/lp1，逐字节完全一致，best86.001940813%@95。完整分析见[官方双卡结果](../official_dualgpu_results_20261013/README.md)。该PT不再待结果；后续候选PT seed1/LP seed0尚未选定。
