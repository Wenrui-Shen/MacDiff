# 历史记录：MacDiff 骨架视频 → VLM 样本级文本实验交接

更新时间：2026-08-18

## 1. 任务目标

我们正在为 MacDiff 探索一条不使用动作类别标签、也不使用原始 RGB 视频的样本级文本生成路线：

1. 从已有的骨架 XYZ 序列生成只包含骨架的可视化视频。
2. 将骨架视频输入开源视觉语言模型（VLM），为每个样本生成简洁的动作文本。
3. 后续再研究如何把这些文本作为条件或跨模态知识接入 MacDiff 的扩散过程。

研究动机：骨架领域现有视觉语言方法经常把类别标签直接写进提示词再扩写文本，在无监督设定中有标签泄漏争议；另一类方法从 RGB 视频生成文本，又会引入背景、物体、衣着和光照等与骨架动作无关的信息。本方案只从 XYZ 骨架构造视觉输入，希望得到无标签、样本级、主要描述人体运动的文本。

当前只完成了“骨架可视化 + 单样本 VLM 推理”的 pilot；文本接入 MacDiff 扩散尚未开始。

## 2. 已阅读/参考的论文

用户最初提供了以下本地论文：

- `D:/program/paper/2511.10091v1.pdf`
- `D:/program/paper/Chen 等 - 2025 - Vision-Language Meets the Skeleton Progressively Distillation With Cross-Modal Knowledge for 3D Act.pdf`
- `D:/program/paper/Do和Kim - Bridging the Skeleton-Text Modality Gap Diffusion-Powered Modality Alignment for Zero-shot Skeleton.pdf`

如果后续继续论文设计，应重新打开这些 PDF 核对具体方法；当前 pilot 的重点是先证明“XYZ → 骨架视频 → VLM 文本”能跑通。

## 3. 已经确定的可视化方案

### 3.1 使用哪种骨架坐标

最终选择：使用 root-centered 的三维骨架，并保留三个同步视图，不使用单独的 root-centered XY 平面图，也暂时不保留 world-view 视频。

这里容易混淆的三个概念：

- `world`：保留全局位置和根节点轨迹，可以观察行走位移，但不同相机位置、朝向和采集空间会带来额外变化。
- `root-centered`：每帧减去根节点坐标，保留身体内部三维相对姿态，但移除了人物在世界坐标中的平移轨迹。
- `fixed-oblique-3D`：是对三维骨架选择一个固定斜视角进行显示，本质是“怎么拍”，不是另一种坐标数据。

当前三视图的目的是减少单一投影视角造成的遮挡和深度丢失。三块画面是同一时刻、同一个人的不同视角，绝不能让模型把三个视图当成三个人。

root-centered 的明确代价：模型无法可靠判断全局行走方向、移动距离和根节点轨迹。提示词/文本不应凭空描述全局 locomotion。如果未来确实需要全局位移，应额外保存 root trajectory 或加一个 world-view 分支，而不是让 VLM 猜。

### 3.2 最终渲染风格

- 每帧显示 root-centered 三视图。
- 第一个有效人物统一使用红色骨架。
- 第二个有效人物统一使用蓝色骨架。
- 不再按身体部位使用不同颜色。
- 不使用关节描边。
- 不突出根节点。
- 关节点已经调小。
- 关节点与骨骼连线需要有可分辨的粗细差异，不要重新改成完全一样。
- 只有实际非零的第二个人物槽位才画蓝色；全零槽位不算第二个人。

用户已经认为当前可视化“差不多 OK”，下一阶段不要无理由大改渲染风格。

## 4. 当前代码和文件

当前核心目录：

```text
tools/vlm_pilot/
```

已知核心文件：

- `tools/vlm_pilot/caption_qwen3vl_sample.py`
  - 当前单样本 Qwen3-VL 推理脚本。
  - 直接使用 Hugging Face Transformers，不是 vLLM/SGLang 服务。
  - 读取排序后的 PNG 帧，把图像列表按视频处理，调用 `model.generate()`。
  - 从渲染 metadata（优先）或颜色检测得到人物数，并动态写入提示词。
  - 解析模型 JSON，保存完整响应及最终 `text`。
- `tools/vlm_pilot/skeleton_motion_prompt_v0.txt`
  - 当前简化后的英文提示词。

由于本次写交接时本地命令执行器出现 `helper_unknown_error: setup refresh had errors`，没有成功重新列出目录。新会话第一步请运行：

```bash
rg --files tools/vlm_pilot vlm_pilot
```

确认渲染脚本的准确文件名、所有参数和当前未提交改动，不要凭本交接猜文件名。

测试产生过的目录/样本包括：

```text
vlm_pilot/sample_000000/
vlm_pilot/sample_000002/
vlm_pilot/compare_sample_000002_32/
```

其中 `sample_000002` 是已知“喝水”动作，只用于人工检查 VLM 描述是否合理；动作类别标签没有传给 VLM。

## 5. 当前提示词设计

提示词已经从复杂的全身详细描述，缩减为只抓最主要运动部位和时序。原因是复杂模板会增加空字段、幻觉和不稳定输出。

### 5.1 人物数

人物数不再交给 VLM 自由判断：

- 只出现红色骨架：1 人。
- 同时出现红色和蓝色骨架：2 人。
- 三个视图只是同一动作的同步视角，不能据此增加人数。
- 推理脚本从渲染 metadata 得到 `actor_count_from_color`，并记录 `actor_count_source: render_metadata`。

这是一个非常重要的设计决定：人物槽位是否为零是结构化信息，程序判断比 VLM 看颜色更可靠。

### 5.2 身体部位

只允许六个粗粒度部位：

```text
head, torso, left_arm, right_arm, left_leg, right_leg
```

模型需要：

1. 选择第一个/最主要运动的部位。
2. 描述这个部位的主要运动形式。
3. 分别描述 beginning / middle / end。
4. 描述该部位与其他身体部位的交互关系。
5. 最后合并为一条简洁英文 `text`。

“交互”绝不只指身体接触，还包括：靠近、远离、相对位置变化、同步/协调运动和保持接近。例如喝水动作中手向头部下方靠近，即使没有精确识别为接触嘴部，也应算有意义的交互。

### 5.3 时序约束

模型曾把单次抬手动作错误描述成“抬起—放下—再次抬起”。提示词后来增加了这些约束：

- 输入帧按时间顺序排列。
- 不要自动假设动作循环或重复。
- beginning、middle、end 必须分别根据相应阶段的可见姿态判断。
- end 必须只看最后阶段，不能由常见动作先验补全。
- 只有最终姿态确实回到起始姿态时，才能使用 `returns` / `returns to`。

### 5.4 当前预期输出结构

```json
{
  "actors": 1,
  "main_actor": "red",
  "main_part": "left_arm",
  "motion": "...",
  "beginning": "...",
  "middle": "...",
  "end": "...",
  "interaction": "...",
  "text": "..."
}
```

请先读取 `skeleton_motion_prompt_v0.txt` 获取当前精确文本，不要根据上面摘要重新发明一份更复杂的提示词。

## 6. 当前推理方式和已经跑通的命令

Ubuntu 项目路径：

```text
/home/user9/public3/swr/MacDiff
```

当前模型路径：

```text
/home/user9/public3/swr/models/Qwen3-VL-2B-Instruct
/home/user9/public3/swr/models/Qwen3-VL-8B-Instruct
```

当前已经成功运行的 8B 示例：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 python tools/vlm_pilot/caption_qwen3vl_sample.py --frames_dir vlm_pilot/compare_sample_000002_32/frames --prompt_path tools/vlm_pilot/skeleton_motion_prompt_v0.txt --model /home/user9/public3/swr/models/Qwen3-VL-8B-Instruct --sample_fps 8 --max_new_tokens 512 --output_path vlm_pilot/compare_sample_000002_32/caption_8b_compact.json
```

这个脚本每执行一次都会重新加载完整模型然后退出。适合少量 pilot，不适合全数据集批量生成。

### 6.1 `num_frames` 和 `sample_fps` 的含义

这是已经踩过的关键概念坑：

- 16 / 32 / 64 帧：实际传给模型的图像数量，决定模型看到多少个离散时间点。
- `--sample_fps 8`：告诉模型这些图片对应每秒 8 帧的时间尺度；不会把 32 张图再次抽成 8 张。
- 例如 32 张图配 `sample_fps=8`，表示大约 4 秒的序列。

不要再把 `sample_fps=8` 解释成“只输入 8 帧”。

当前较合理的基线是：32 个实际帧 + `sample_fps=8`。16 帧可能漏掉关键阶段，64 帧会明显增加视觉 token、显存和模型理解负担。

## 7. 已经观察到的实验结果

### 7.1 显存

服务器 GPU：

```text
2 × NVIDIA GeForce RTX 4090
每张 24564 MiB
驱动 535.54.03
```

实测大致峰值：

- Qwen3-VL-2B-Instruct：约 6GB。
- Qwen3-VL-8B-Instruct：约 21GB。

当前命令用了 `CUDA_VISIBLE_DEVICES=0`，所以只在一张卡上运行。

### 7.2 2B 结果

2B 能跑通，但对骨架动作描述明显较弱。它曾把样本描述成四肢向前伸展，不能可靠识别喝水相关的手臂—头部关系。2B 只适合验证流程和排查代码，不应作为最终伪文本教师。

### 7.3 8B 结果

8B 对 `sample_000002`（喝水）的最终一版输出大意是：

```text
Red skeleton's left arm raises and rotates upward, coordinating with torso, ending near shoulder level.
```

它没有识别出“喝水”语义，但这不是当前必须目标。无标签文本首先要忠实描述可观察的骨架运动。模型识别为手臂抬起、末端靠近肩/头部下方，整体可接受。

左右手在不同视图/相机方向下容易混淆；用户明确表示当前左右识别不是核心问题。除非下游明确依赖解剖左右，不要把精力首先花在左右手修正上。

模型早期曾错误输出“手臂抬起、放下、再抬起”，但渲染视频实际没有重复，最终手仍靠近头部。经提示词增加非循环和末帧约束后，结果改善。

### 7.4 空模板问题

8B 在 `sample_000000`、64 帧、`sample_fps=4` 的一次测试中返回了字段齐全但内容几乎全空的 JSON，脚本却标记为：

```json
"status": "accepted"
```

这暴露的是校验器缺陷：当前校验大概率只检查 JSON/键是否存在，没有检查文本字段是否为空。不能把这个问题简单归因于“8B 参数不够”；64 帧输入负担、复杂提示词、生成设置等都可能参与，但没有证据确定单一原因。

下一步必须加强验证：至少拒绝空 `text`、空 `motion`、空 beginning/middle/end、非法 `main_part`，并允许有限次数重试。绝不能让空模板进入训练文本库。

### 7.5 generation flags 警告

日志中出现过：

```text
The following generation flags are not valid and may be ignored: ['temperature', 'top_p', 'top_k']
```

当前脚本使用 `do_sample=False`，这些采样参数被忽略是正常警告，不是推理失败原因。要么清理 generation config 中无效字段，要么保留警告；不要因为这个警告盲目重装环境。

## 8. Ubuntu 环境现状与依赖坑

当前独立 conda 环境：

```text
skeleton_vlm
Python 3.11.13
```

已使用的主要包版本：

```text
transformers==4.57.1
qwen-vl-utils==0.0.14
accelerate
safetensors
Pillow
numpy
```

当前环境已经成功跑通 2B/8B，不要为了升级 Qwen3.8 直接破坏它。

已经踩过的安装坑：

1. 最初的 `macdiff` 环境 Python 版本太老，无法安装 `transformers==4.57.1`；后来新建了 Python 3.11 的 `skeleton_vlm`。
2. 新环境起初没有 PyTorch，`ModuleNotFoundError: No module named 'torch'`。新环境必须先按服务器 CUDA/驱动安装 PyTorch，再装上层 VLM 包。
3. 测试版本时曾把 `torch.__version__` 从富文本复制成 `torch.**version**`，导致命令语法错误。正确的是双下划线。
4. 安装 `qwen-vl-utils==0.0.14` 时，pip 尝试从源码构建 `av==18.0.0`，报 `pkg-config is required for building PyAV`。不要反复执行同一条失败命令。优先使用有预编译 wheel 的兼容 PyAV、conda-forge 包，或在确有权限时安装 FFmpeg/pkg-config 开发依赖。
5. 模型下载和推理已经拆成两步。模型放在统一的 `/home/user9/public3/swr/models/`，推理时使用绝对本地路径和 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`，避免每次联网。

在对环境做任何升级前，先记录：

```bash
python --version
python -c "import torch, transformers; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), transformers.__version__)"
nvidia-smi
python -m pip freeze
```

## 9. 关于 vLLM、SGLang 和 Qwen3.8 的最新结论

### 9.1 当前脚本与服务框架的区别

当前 `caption_qwen3vl_sample.py` 是单进程 Transformers 推理：每个命令加载一次模型、处理一个样本、保存后退出。

vLLM 和 SGLang 是模型服务/推理引擎：让模型常驻显存，通过 OpenAI 兼容 API 连续接收请求，提供连续批处理、更好的 KV cache 和多 GPU 管理。它们能提高吞吐和部署便利性，但不会自动修复骨架时序幻觉；同一权重、同一输入和同一生成设置下，模型能力主要由权重和预处理决定。

本任务优先考虑 vLLM，原因是 Qwen3.8 官方模型卡目前明确给出了 vLLM 的视频 FPS/抽帧请求参数；SGLang 也支持 Qwen3.8 原生视觉和视频，但精确视频采样控制不是当前首选。

### 9.2 Qwen3.8 是真实的官方模型

不要再误判 Qwen3.8 不存在。官方链接：

- https://github.com/QwenLM/Qwen3.8
- https://huggingface.co/Qwen/Qwen3.8-27B
- https://huggingface.co/Qwen/Qwen3.8-27B-FP8
- https://recipes.vllm.ai/Qwen/Qwen3.8-27B
- https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B

Qwen3.8-27B 于 2026-08-14 发布，是基于 Qwen3.5 架构的 27B dense 原生视觉语言模型，支持图像和视频。官方也提供了 block-wise FP8 权重，并称其性能与原模型基本一致。

### 9.3 两张 4090 的可行性

- BF16 权重约 54GB，双 24GB 4090 也不适合。
- 官方 FP8 权重约 28.5GB，单张 24GB 4090 放不下；“FP8 单卡部署”通常指 32GB 5090、48GB 或更大显存卡，不是 24GB 4090。
- 两张 4090 使用 tensor parallel size 2 时，预计每张约 14.3GB 权重，剩余显存用于视觉编码、GDN 状态和 KV cache。对于本项目的 32 帧短输入，理论上可行。
- 官方 vLLM 配方验证的是 2 × RTX 5090 TP2，不是 2 × RTX 4090，因此“两张4090可跑”是有硬件依据的合理推断，仍需要实测。
- 官方示例中的 TP4 + 262K context 是为了超长上下文/大 KV cache，不代表本任务必须四卡。我们的 pilot 应把 `max-model-len` 限到 32768 左右，从 TP2 开始。

RTX 4090 的 Ada Tensor Core 支持 FP8，但当前服务器驱动只有 535.54.03。最新 vLLM、PyTorch、CUDA wheel 和新模型内核的组合可能对驱动更敏感；先在新环境验证，遇到明确的 driver/runtime 错误再考虑升级驱动，绝不要未经确认直接动服务器现有驱动。

### 9.4 不能直接替换现有模型路径

当前环境是 `transformers==4.57.1`。Qwen3.8 的官方 vLLM recipe 要求 `transformers >= 5.8.0`，模型类/processor 路径也与当前 Qwen3-VL 脚本不同。因此不能只把：

```text
--model /.../Qwen3-VL-8B-Instruct
```

替换成 Qwen3.8 路径就认为能工作。

必须新建环境，例如 `qwen38_vllm`，不要升级或覆盖已经能跑的 `skeleton_vlm`。

### 9.5 是否值得部署

结论：值得作为更强教师模型做小规模 A/B 测试，但现在没有理由直接全量替换 8B。

Qwen3.8 官方展示了更强的视觉推理、指令遵循和原生视频能力，但没有骨架动作/细粒度时序描述 benchmark。骨架三视图还是一种特殊的分布外视觉输入。27B 可能改善时序和关系描述，也可能仍受可视化、抽帧和提示词限制。必须用真实骨架样本验证，而不是凭参数量决定。

## 10. 当前卡在哪里

没有代码层面的死锁；当前处在以下决策/实现节点：

1. root-centered 三视图渲染基本完成。
2. Qwen3-VL 2B/8B 单样本链路已经跑通。
3. 简化提示词已经能对喝水样本给出基本合理的“手臂抬起并靠近上身/头部”描述。
4. 尚未做多样本系统评估，尚不知道提示词在不同动作/多人动作上的稳定性。
5. 空模板仍可能被错误标记为 accepted，验证器需要加强。
6. Qwen3.8-27B-FP8 尚未下载、尚未建立 vLLM 环境、尚未在双 4090 上启动。
7. 当前 Transformers 单样本脚本尚未改造成常驻模型批处理或 vLLM API 客户端。
8. 文本如何接入 MacDiff 扩散完全尚未实现，应等伪文本质量得到验证后再开始。

## 11. 建议的下一步计划

### 阶段 A：冻结并评估 8B 基线

1. 用 `rg --files tools/vlm_pilot vlm_pilot` 和 `git status --short` 核对当前文件及未提交改动。
2. 备份/记录当前渲染参数、提示词和环境版本。
3. 加强 `caption_qwen3vl_sample.py` 的结果校验：
   - `text` 非空。
   - `motion` 非空。
   - beginning/middle/end 均非空。
   - `main_part` 必须来自六个允许值。
   - `actors` 必须等于程序注入的人物数。
   - 不接受只有模板没有内容的 JSON。
   - 失败时保存 raw response，并最多重试 1–2 次，不能无限重试。
4. 选 10–20 个覆盖不同模式的样本：静态/上肢/下肢/全身/单人/双人/容易遮挡。
5. 固定 32 个实际输入帧和 `sample_fps=8`，先用 Qwen3-VL-8B 生成基线。
6. 人工记录：主部位正确性、begin/middle/end 忠实度、末端姿态、靠近/接触/协调关系、空字段、JSON 有效率、推理时间、峰值显存。

### 阶段 B：建立独立的 Qwen3.8 vLLM pilot

1. 新建独立环境 `qwen38_vllm`，Python 3.11；不要修改 `skeleton_vlm`。
2. 根据 Qwen3.8/vLLM 最新官方 recipe 安装兼容的 vLLM、PyTorch 和 `transformers>=5.8.0`。
3. 首先验证驱动、CUDA、两张 GPU 和 FP8 kernel；不要一上来下载/启动 262K context。
4. 将官方 `Qwen/Qwen3.8-27B-FP8` 单独下载到：

   ```text
   /home/user9/public3/swr/models/Qwen3.8-27B-FP8
   ```

5. 从 TP2、`max-model-len=32768`、FP8 KV cache 和单并发开始。
6. Qwen3.8 默认 thinking 开启。对于简洁结构化骨架描述，第一轮建议显式关闭 thinking，减少冗余和 JSON 解析风险；随后可用少量样本对比 `reasoning_effort=low` 是否改善时序。
7. 不要默认使用 vLLM 视频的 `fps=2`。我们需要保证模型看到与基线相同的 32 帧：
   - 如果把 PNG 编成 MP4，明确编码 FPS；
   - 明确设置/关闭二次抽帧；
   - 输出中同时记录原始帧数、编码 FPS 和模型实际采样帧数；
   - 避免“先抽 32 帧，再被服务端按 2 FPS 抽一次”。
8. 改写/新增 API 客户端脚本，不要破坏当前 Transformers baseline 脚本。

### 阶段 C：A/B 决策

对同一批样本比较：

- Qwen3-VL-8B，当前 Transformers pipeline。
- Qwen3.8-27B-FP8，vLLM TP2。

除了文字“看起来更好”，至少统计：

- 主运动部位正确率。
- 末帧/结束姿态正确率。
- 是否虚构重复或返回起点。
- 部位间接近、接触、协调关系的忠实度。
- 人数一致率。
- 非空合法 JSON 比例。
- 每样本耗时、峰值显存、吞吐。

只有 Qwen3.8 在这些指标上有稳定收益，才值得将它作为批量教师模型。否则继续使用 8B，并把精力放在骨架几何辅助、时序分段和后处理校验上。

### 阶段 D：批量文本库

开始全量生成前，每条记录至少保存：

```text
sample_id
source split（不得包含作为提示的动作标签）
render configuration/version
actual input frame count
video/render fps
model id and revision
precision/quantization
prompt version or hash
raw response
parsed fields
final text
validation status
retry count
runtime and peak VRAM（抽样记录也可以）
```

必须可追溯，否则后面无法分析文本质量或重生成。

### 阶段 E：接入 MacDiff（暂缓）

在伪文本质量稳定后再讨论：

- 使用哪个 text encoder。
- 文本条件注入扩散模型的哪一层/哪一阶段。
- 作为条件、蒸馏目标还是跨模态对齐损失。
- 训练时是否冻结文本编码器。
- 如何避免伪文本错误反过来污染骨架表示。

不要在伪文本还大量空字段/时序错误时提前修改 MacDiff 主训练代码。

## 12. 绝对不要再踩的坑

1. **不要把动作类别标签、类别名或由标签扩写的句子放进 VLM 提示词。** 这是整个无标签方案的核心约束。
2. **不要把 RGB 视频混入当前 pilot。** 当前研究点就是只由骨架生成样本级文本；RGB 可以以后作为对照实验，但不能悄悄进入主流程。
3. **不要让 VLM 自己数三视图中的“人”。** 三个视图不是三个人；人物数由红/蓝有效骨架和 metadata 决定。
4. **不要把全零第二槽位算成第二个人。** 只有蓝色有效骨架存在才是双人。
5. **不要把 root-centered 数据描述成保留了全局移动。** 根节点平移已经删除。
6. **不要把 `sample_fps` 当成输入帧数。** 32 帧和 FPS=8 是两个不同参数。
7. **不要使用 vLLM 默认 FPS 后又对 32 帧二次抽样而不记录。** 这会破坏与 8B baseline 的公平对比。
8. **不要因为常识动作先验写出不可见语义。** 喝水样本可以描述“手靠近头部”，但没有 RGB 物体就不要强迫模型生成“holding a cup”。
9. **不要自动假设动作循环。** 最终阶段只能依据末尾帧；没有回到起点就不能写 `returns`。
10. **不要把字段齐全的空 JSON 标为 accepted。** 强化内容验证是下一步第一优先级之一。
11. **不要重新把提示词堆得非常复杂。** 复杂提示词已经导致空模板/冗余；保持单一主运动部位、三阶段、交互、最终一句话。
12. **不要把左右手误差当作当前最高优先级。** 先保证部位类别、时序和接近关系正确。
13. **不要用 2B 的能跑通等价于文本质量够用。** 2B 只是 smoke test。
14. **不要因为 generation flags 警告就重装环境。** `do_sample=False` 时 temperature/top-p 被忽略是正常的。
15. **不要直接升级已经能运行的 `skeleton_vlm`。** Qwen3.8 必须新建环境。
16. **不要认为 Qwen3.8 FP8 能单张 24GB 4090运行。** 官方 FP8 权重本身约 28.5GB；本机要测试的是双 4090 TP2。
17. **不要照抄官方 TP4/262K 服务命令。** 本项目先 TP2/32K，超长上下文只会浪费显存。
18. **不要认为换成 vLLM/SGLang会自动提高描述质量。** 它们首先解决服务吞吐、缓存和多卡部署，模型质量仍需 A/B 实测。
19. **不要直接假定 27B 一定比 8B 更懂骨架。** 官方没有骨架动作 benchmark；必须用同样本同帧数验证。
20. **不要在伪文本质量未通过小规模检查前全量生成或接入扩散训练。** 错误文本一旦大规模写入，再排查成本很高。
21. **不要忽略文件名/路径中的潜在标签泄漏。** 即使 prompt 不写标签，也要确认发送给模型的文本内容、URL、metadata 和文件名不会暴露类别名。
22. **不要用 Grounding DINO 替代动作描述 VLM。** Grounding DINO主要做开放词汇检测/定位，不是视频动作 caption 模型。
23. **不要把闭源 GPT-4V API 与本地开源模型的成本/隐私/可复现性混为一谈。** 可以作为少量上限对照，但当前主路线是可本地部署的开源 VLM。

## 13. 给新会话的最短启动指令

先不要改代码，依次检查：

```bash
cd /home/user9/public3/swr/MacDiff
git status --short
rg --files tools/vlm_pilot vlm_pilot
sed -n '1,260p' tools/vlm_pilot/caption_qwen3vl_sample.py
sed -n '1,220p' tools/vlm_pilot/skeleton_motion_prompt_v0.txt
find vlm_pilot/compare_sample_000002_32 -maxdepth 2 -type f | sort
```

然后用已有 8B 命令复现 `sample_000002`，确认 baseline 仍工作。复现后先修空模板校验，再开始建立独立的 Qwen3.8 vLLM 环境。

新会话如果只能记住一句话：**当前要做的不是立刻改 MacDiff，而是先把“固定的 root-centered 三视图 32 帧 → 无标签、非空、时序忠实的样本级文本”做成可验证、可追溯的稳定数据生成管线。**
