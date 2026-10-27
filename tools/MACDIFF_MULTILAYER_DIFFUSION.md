# MacDiff后三层完全共享norm和decoder的diffusion监督，第8层LP

2026-10-27：用户进一步指出辅助LayerNorm也应共享，当前v3让encoder第6/7/8层复用**同一个原生最终LayerNorm（含weight/bias）和同一套MacDiff decoder**，做三次前向、分别计算噪声损失，LP只读第8层。v1独立decoder、v2共享decoder但额外保留两层辅助norm均为此前设计，不作为v3的完整resume。当前没有收到本方法训练结果，不预登记新成绩ID。固定预训练encoder蒸馏等待用户参考文献。

## 模型与损失

encoder保持8层、256维、8heads；只有一套native decoder，3层、256维、FFN1024。三次调用完全共享decoder骨架patch embedding、时空embedding、blocks/调制、norm和噪声输出head；代码均调用decode_skeleton(self, ...)，没有auxiliary_decoders或复制decoder。

encoder一次前向输出第6/7/8层，三层均调用原有self.norm，包括共享其可学习缩放和偏置。LayerNorm对各次输入分别计算token内的均值/方差，没有BatchNorm式运行统计，因此不必独立维护归一化模块。没有新增可训练参数或模块，参数量及state_dict键/形状与原生MacDiff完全一致。每层按原生规则使用可见local token，并用该层token GAP填充masked位置；共享ids_restore恢复全部750个时空位置。120帧、时间patch4、25关节；mask.9时各层75个可见token，每次decoder输出750×12的epsilon预测。

三路共用同一原始骨架、增强裁剪、mask、离散t、Gaussian epsilon、x_t和一次抽样的每样本条件dropout（.1）；扩散仍1000步/inverse_cosine tau1。三路都只在encoder-masked位置计算原生epsilon MSE。没有给特征本身加噪，也没有先融合三层作为单次decoder条件。

默认目标：

```text
L = L8 + 0.1 * (L6 + L7)/2 + 0.02 * uniformity8
```

L8系数1；L6/L7各.05；骨架token uniformity仅在第8层计算一次。--lambda_aux_diffusion覆盖辅助平均项权重并保存进model_args。设为0仍计算三次decoder，辅助项有效梯度为0，最终损失应与原生一致，不作为节省计算的实现。

三路梯度同时累积到同一decoder。encoder第6层及之前收到三路监督，第7层收到第7/8层两路，第8层只收到最后一路；中间特征不detach。共享约束会让不同层学习可由同一去噪器读取的条件，也可能引入梯度竞争，不保证LP提高。

encoder只跑一次，decoder仍跑三次。共享可节省相对v1的decoder参数、梯度和优化器状态，但三路反向需要保存相应计算图；不声称decoder计算或激活显存变成一次。

## LP与已有权重兼容

根encoder参数名与原生相同（joints_embed、pos_embed、temp_embed、blocks、norm）；norm由三路共同训练，LP仍读取同一个norm并作用于第8层。不存在auxiliary_norms或auxiliary_decoders。

LP用原有main_linprobe.py和linprobe_madiff.yaml，载入全部encoder键并忽略decoder；冻结encoder，只训练现有分类head，保持linprobe2/6400维关节读出/head dropout.3/BN+FC/训练旋转与测试无旋转。LP不融合第6/7层，不修改LP文件。若以后单独测试三层平均融合，可对同一PT checkpoint重新训练对应LP头，本次不加入。

## 配置、精度及恢复

- 方法版本：last3_shared_norm_decoder_final_layer_lp_v3。旧v1独立decoder、v2共享decoder但保留辅助norm的完整checkpoint不作为本版resume；不能仅凭参数形状相近忽略方法/目标版本。
- PT数据：NTU60 XSub joint，A归一化，person0，PT不旋转；无文本/cache/EMA teacher/InfoNCE/AA/FM/OSE。
- PT400、双卡每卡32、accum1、总64；seed0；AdamW/betas(.9,.95)/wd.05；lr.001、warmup20、min_lr5e-4、最后20轮floor。
- LP100、双卡每卡64、accum1、总128；seed0；lr.1、warmup0、min_lr0、最后10轮floor；载入本组checkpoint399。
- 默认PT关闭autocast/GradScaler。decoder与epsilon loss显式FP32；若CLI打开AMP，encoder可使用AMP，但decoder/loss仍FP32。
- 历史T26为AMP协议，本版默认noAMP，与历史结果比较时保留精度边界。
- 新PT目录若已有log.txt或checkpoint拒绝从头覆盖。完整resume须匹配方法、model_args、数据配置、mask与训练日程/精度等；之后由公共入口严格恢复模型/优化器/scaler/epoch。
- 恢复仅核对保存的数据路径/配置，不将其当作数据文件SHA验证。

## 运行

同步当前main_pretrain_multilayer.py、model/transformer_macdiff_multilayer.py、config/ntu60_xsub_joint/pretrain_madiff_multilayer.yaml。公共main_pretrain.py/engine_pretrain.py/util/misc.py保持当前版本（包含通用pretrain metrics、--disable_amp及可关闭scaler支持），LP仍用原入口。

完全共享版新目录带sharednormdec3，端口10319/10320仅为本次选择；完整一行PT成功后自动LP命令如下：

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10319 main_pretrain_multilayer.py --config config/ntu60_xsub_joint/pretrain_madiff_multilayer.yaml --disable_amp --lambda_aux_diffusion 0.1 --batch_size 32 --accum_iter 1 --epochs 400 --warmup_epochs 20 --min_lr_epochs 20 --lr 0.001 --min_lr 0.0005 --mask_ratio 0.9 --motion_aware_tau -1 --seed 0 --output_dir output_dir/ntu60_xsub_multilayer678_sharednormdec3_aux01_lp8_noamp_seed0_20261027/pt --log_dir output_dir/ntu60_xsub_multilayer678_sharednormdec3_aux01_lp8_noamp_seed0_20261027/pt/tensorboard && OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10320 main_linprobe.py --config config/ntu60_xsub_joint/linprobe_madiff.yaml --finetune output_dir/ntu60_xsub_multilayer678_sharednormdec3_aux01_lp8_noamp_seed0_20261027/pt/checkpoint-399.pth --batch_size 64 --accum_iter 1 --epochs 100 --warmup_epochs 0 --min_lr_epochs 10 --lr 0.1 --min_lr 0 --seed 0 --dist_eval --output_dir output_dir/ntu60_xsub_multilayer678_sharednormdec3_aux01_lp8_noamp_seed0_20261027/lp --log_dir output_dir/ntu60_xsub_multilayer678_sharednormdec3_aux01_lp8_noamp_seed0_20261027/lp/tensorboard
```

## 日志与验证

日志分别记录loss_diffusion_layer6/7/8、loss_native（L8+.02*uni8）、loss_auxiliary_mean、loss_auxiliary_weighted、loss_uniformity；TensorBoard前缀multilayer_diffusion/，复用CUDA峰值指标。核算total=native+auxiliary_weighted，LP收益仍须实测。

本机没有Torch/PyYAML/timm。已运行：

```bash
python -m unittest tests.test_multilayer_diffusion tests.test_pretrain_accumulation_policy -v
```

共23项，16通过、7因缺Torch跳过。通过项执行真实生产损失组合、decoder算术与原生一致、encoder提层控制流、三路forward与原生masked loss（NumPy tensor适配），验证三次调用的decoder对象相同、共享噪声/t/mask/dropout/person0与输入不被修改；另含CLI覆盖、恢复/覆盖保护和公共累积策略。NumPy适配不验证autograd/CUDA/DDP。

7项Torch检查已写，待有Torch环境执行：原生参数量、参数键及初始化完全一致；所有参数有限梯度；第6层支路梯度不进入第7/8层；辅助权重0时与原生兼容；真实decoder共享dropout/扩散状态；共享norm/decoder参数梯度符合三路加权组合（norm另含最后层uniformity）；已有LP完整载入encoder且忽略decoder。服务器可先运行上述测试，再执行PT&&LP。没有启动训练，实际GPU显存/耗时和LP尚无结果。
