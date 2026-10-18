# 隔离官方源码双卡默认基线

官方提交692888f1e2a4227216511aac6733dff31b587213的92份原始文件副本。共3份有明确改动：util/misc.py仅将torch._six.inf导入改为同值math.inf以兼容当前Torch；NTU60 XSub两份配置仅将数据路径改成现有../data/MAMP/ntu/NTU60_XSub.npz。模型、engine、PT DDP、LP算法、随机采样、LR scheduler等保持官方源码。每文件SHA与例外见snapshot_manifest.json。

从当前MacDiff仓库根目录调用此目录里的main_pretrain.py/main_linprobe.py。Python worker的脚本目录在模块搜索路径前端，model/feeder/util/engine/guided_diffusion解析到此副本。不要从副本目录运行，数据/output路径以当前仓库根为基准。

PT双卡32/accum1→有效64、decoder5、400epoch、seed0、lr1e-3/min_lr1e-5、PT不旋转；LP checkpoint399、双卡64/accum1→有效128、100epoch、lr.1、seed0、训练旋转True/测试False。这样对齐用户提供的双卡默认参考；第三方LP batch仍为用户估计。

原当前分支不切换、不覆盖；本次不启动训练。此副本已完成文件哈希、Python3.8 AST和CLI字段验证；没有本地Torch/GPU，因此没有实际导入或训练验证。

PT/LP命令保存在../native_rotation_results_20261012/commands.json与clean_default_pt_lp.sh。先同步此完整目录和命令到服务器；直接粘贴命令时也使用新输出目录。
