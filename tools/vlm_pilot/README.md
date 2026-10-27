# One-sample skeleton-to-text pilot

This pilot deliberately separates deterministic visualization from VLM inference.
The renderer reads only skeleton arrays from the processed MAMP NPZ and never
loads labels. It also bypasses the random crop, flip, rotation, and noise in the
training feeder.

## 1. Render one sample

```bash
python tools/vlm_pilot/render_skeleton_sample.py \
  --data_path ../data/MAMP/ntu/NTU60_XSub.npz \
  --split train \
  --sample_index 0 \
  --num_frames 16 \
  --output_dir vlm_pilot/sample_000000
```

Inspect `preview.gif` and `preview_middle.png`. The two panels are per-frame
root-centered X-Y and Z-Y projections. Together they retain all three XYZ axes
while deliberately removing the primary actor's global translation. For a
two-person sample, both actors are translated by person 1's root, so their
relative position remains visible. Person 1 is uniformly red and person 2 is
uniformly blue. Joint circles are deliberately larger than the thinner bone
lines so that the VLM can distinguish joints from connections. Joints have no
outline, and the root joint uses the same marker size as every other joint.
The renderer counts nonempty tracks over the full sequence before smoothing or
sampling. Exact copies in both NTU120 person slots count as one actor and only
the red copy is drawn. Two distinct tracks count as two actors. The captioner
reads this structural information from `render_metadata.json`. Blue pixels
alone cannot establish two people; ambiguous blue-only images require metadata.

The rendering code can be checked without a dataset:

```bash
python tools/vlm_pilot/render_skeleton_sample.py \
  --demo \
  --output_dir vlm_pilot/demo
```

## 2. Install Qwen3-VL dependencies on Ubuntu

Use the CUDA-compatible PyTorch build already selected for the server, then:

```bash
pip install "transformers>=4.57.0" accelerate qwen-vl-utils==0.0.14 Pillow
```

The default pilot model is `Qwen/Qwen3-VL-2B-Instruct`. It minimizes the cost of
debugging the data path and prompt. Switch to 4B or 8B only after the pipeline
works and the rendered motion is readable.

## 3. Generate the sample-level description

```bash
python tools/vlm_pilot/caption_qwen3vl_sample.py \
  --frames_dir vlm_pilot/sample_000000/frames \
  --output_path vlm_pilot/sample_000000/caption.json
```

For a larger model:

```bash
python tools/vlm_pilot/caption_qwen3vl_sample.py \
  --frames_dir vlm_pilot/sample_000000/frames \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --attn_implementation flash_attention_2 \
  --output_path vlm_pilot/sample_000000/caption_qwen3vl_8b.json
```

The captioner passes an ordered PNG list as one video input, so FFmpeg and video
codec behavior do not affect this first experiment. The current schema returns
one non-empty `persons` entry for each visible skeleton. `person_index: 0` is
red and `person_index: 1` is blue; a single-person sample contains no synthetic
second-person or empty-string target.

The default prompt is `skeleton_motion_prompt_v2.txt`. Each person has one brief
`global` description and six fixed `local` descriptions: `head`, `torso`,
`left_arm`, `right_arm`, `left_leg`, and `right_leg`. The Chinese translation is
`skeleton_motion_prompt_v2_zh.txt`. The v0/v1 prompts remain available explicitly
through `--prompt_path`; their original response schemas are still validated.
The single-sample script saves only `persons` in its output JSON. Model, device,
prompt and settings are saved in a `.metadata.json` sidecar; the raw response and
validation result are saved in `.diagnostics.jsonl`.

## 4. Complete train split as two independent stages

The recommended full pipeline persists every visualization before any model is
loaded. Stage 1 reads only `x_train` and creates one `preview.gif` plus one
`render_metadata.json` under each `train_<index>` directory. Stage 2 reads only
those files; it never reopens the NPZ. This makes rendering independently
inspectable and reusable across prompt or model experiments.

Render all train samples once:

```bash
python tools/vlm_pilot/render_qwen3vl_train.py \
  --data_path ../data/MAMP/ntu/NTU60_XSub.npz \
  --output_root vlm_pilot/ntu60_xsub_train_rendered_v3_2view_smooth_w5 \
  --num_frames 32 \
  --sample_fps 8 \
  --temporal_smooth savgol \
  --median_window 3 \
  --smooth_window 5 \
  --smooth_polyorder 2 \
  --max_interp_gap 2 \
  --resume
```

Temporal smoothing is applied independently to every person's full joint
trajectories before the 32 output frames are sampled.  A 3-frame median removes
isolated spikes, short internal gaps of at most 2 frames are interpolated, and
a dependency-free 5-frame quadratic Savitzky-Golay filter reduces jitter
without causal lag.  Pass `--temporal_smooth none` for an unsmoothed comparison.
Use a new output root when comparing with renders made before smoothing.

The GIF representation avoids writing 32 separate PNG files per sample. For
NTU60 XSub, allow roughly 10-20 GB depending on the motion content and
filesystem. `--resume` validates both the GIF and matching render configuration
before skipping a sample.

Verify the persisted inputs and expanded prompt without loading Qwen:

```bash
python tools/vlm_pilot/caption_qwen3vl_rendered_train_transformers.py \
  --rendered_root vlm_pilot/ntu60_xsub_train_rendered_v3_2view_smooth_w5 \
  --output_path vlm_pilot/rendered_dry_run.jsonl \
  --num_shards 2 --shard_id 0 --dry_run
```

The recommended dual-GPU entry point starts both workers, automatically resumes
saved samples, and merges accepted results into one sorted JSON array:

```bash
python tools/vlm_pilot/run_caption_dual_gpu.py --rendered_root vlm_pilot/ntu60_xsub_train_rendered_v3_2view_smooth_w5 --model /home/user9/public3/swr/models/Qwen3-VL-8B-Instruct --output_dir vlm_pilot/ntu60_xsub_global_local_v2 --gpus 0,1 --expected_samples 40091
```

Use the Qwen inference environment, not the older MacDiff training environment.
The existing GIFs are reused. GPU 0 handles even sample indices and GPU 1 handles
odd indices, with one independent model per GPU. The launcher sets each worker's
`CUDA_VISIBLE_DEVICES` and uses a 1024-token generation ceiling for the new
global/local JSON, including two-person samples.

Run the same command after interruption: it checks model/prompt identity, skips
accepted samples, and retries unsuccessful samples. Only an incomplete final
JSONL line is backed up and removed automatically. Completed malformed lines
and metadata mismatches cause errors. Ctrl+C stops the workers and saves a
sorted partial result; an OS-released lock prevents simultaneous launchers from
writing this output directory. Missing indices are listed in the merged metadata
and produce a nonzero exit code, rather than reporting a complete run.

Outputs under `--output_dir`:
- `captions.json`: text-only JSON array, one sample per line, sorted by `sample_index`.
- `captions.metadata.json`: merged run information and completeness/missing-index summary.
- `shard0.jsonl` / `shard1.jsonl`: resumable per-GPU text results.
- `shard0.metadata.json` / `shard1.metadata.json`: per-worker run metadata.
- `shard0.diagnostics.jsonl` / `shard1.diagnostics.jsonl`: raw responses, errors and retries.
- `shard0.log` / `shard1.log`: worker terminal output.

Keep the directory intact for resume. To test the first four samples, add
`--max_samples 4`; remove that argument to continue the full run in the same
directory. Add `--merge_only` to merge existing results without loading models,
or `--dry_run` to inspect pending counts and worker commands without starting
generation. `--expected_samples 40091` also checks that the rendered input itself
contains the complete NTU60 XSub train index range.

Alternatively, run the workers in two terminals. Each GPU loads one model; GPU 0
handles even indices and GPU 1 handles odd indices:

```bash
OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
python tools/vlm_pilot/caption_qwen3vl_rendered_train_transformers.py \
  --rendered_root vlm_pilot/ntu60_xsub_train_rendered_v3_2view_smooth_w5 \
  --model /home/user9/public3/swr/models/Qwen3-VL-8B-Instruct \
  --prompt_path tools/vlm_pilot/skeleton_motion_prompt_v2.txt \
  --output_path vlm_pilot/ntu60_xsub_global_local_shard0.jsonl \
  --num_shards 2 --shard_id 0 --resume
```

```bash
OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 \
python tools/vlm_pilot/caption_qwen3vl_rendered_train_transformers.py \
  --rendered_root vlm_pilot/ntu60_xsub_train_rendered_v3_2view_smooth_w5 \
  --model /home/user9/public3/swr/models/Qwen3-VL-8B-Instruct \
  --prompt_path tools/vlm_pilot/skeleton_motion_prompt_v2.txt \
  --output_path vlm_pilot/ntu60_xsub_global_local_shard1.jsonl \
  --num_shards 2 --shard_id 1 --resume
```

Caption resume reads the separate metadata file, checks the model, requested
revision and prompt SHA-256, then skips sample indices in the text-only output.
Invalid and failed samples are retried and recorded in the diagnostic sidecar.
Use a new output path for legacy manifests and the new global/local experiment.
The older `caption_qwen3vl_train_transformers.py` remains available as the
single-stage in-memory path, but it is no longer the recommended full run.

Batch output contains only a sample index and its generated person descriptions.
With `.jsonl`, each sample is a complete JSON object on its own line. With
`.json`, output is a valid JSON array with each sample on its own line. Only
accepted descriptions enter the result file. For example:

```json
{
  "sample_index": 42,
  "persons": [
    {
      "person_index": 0,
      "global": "The person raises the right hand toward the face, then lowers it.",
      "local": {
        "head": "The head remains upright.",
        "torso": "The torso remains upright.",
        "left_arm": "The left arm stays beside the torso.",
        "right_arm": "The right elbow bends as the hand rises toward the face, then lowers.",
        "left_leg": "The left leg remains straight.",
        "right_leg": "The right leg remains straight."
      }
    }
  ]
}
```

For `captions.jsonl` or `captions.json`, `captions.metadata.json` stores the model,
hardware, runtime versions, prompt and run settings, once per run.
`captions.diagnostics.jsonl` stores sample-specific raw responses, retries,
errors and timings. Keep the metadata with the result file for `--resume`.
The inspection utility accepts both formats and still reads legacy manifests.
The existing v2 CLIP token-cache builder remains the legacy word-token pipeline;
use `cache_clip_motion_text.py` for the new global/local sentence cache (v3).
It encodes seven complete sentences per person with fixed body-region slots,
stores unit-RMS FP32 features and is accepted by the existing training reader.
See [TEXT_CACHE_CHAIN_AUDIT.md](TEXT_CACHE_CHAIN_AUDIT.md) for the legacy-chain
audit, separate training configurations, resume behavior and server commands.

Neither stage reads `y_train`. The model prompt contains neither a sample
filename nor an action label. Concatenate the two caption shard files only after
both processes finish; use `sample_index` when ordered records are required.
Only JSONL shards can be concatenated directly; merge JSON arrays as JSON.

### 4.1 NTU60 XView、NTU120 与 PKU 的独立入口（2026-10-17）

以下入口仅读取各自划分的 **x_train**，沿用上面的渲染/描述设置：32 帧、8 fps、
root-centered 正面/侧面双视图、median3 + Savitzky–Golay5/poly2、
`skeleton_motion_prompt_v2.txt`、每卡一个常驻 Qwen3-VL-8B、1024 token、
最多两次修复。原三个入口默认 `--stage all` 完成渲染和描述，自动断点续跑；
独立XSet入口默认只做render，便于先完成全部骨架渲染。
CLIP 编码是独立的第三步，见4.2。按用户要求分阶段执行时也使用4.2的命令。
在仓库根目录、已有的 Qwen 推理环境中分别执行：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python tools/vlm_pilot/run_caption_ntu60_xview.py
```

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python tools/vlm_pilot/run_caption_ntu120.py
```

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python tools/vlm_pilot/run_caption_pku.py
```

模型默认 `/home/user9/public3/swr/models/Qwen3-VL-8B-Instruct`，卡号默认
`0,1`，可用 `--model` / `--gpus` 覆盖。这些入口共用
`dataset_caption_pipeline.py` 和已有渲染/描述模块；XView还复用
`run_caption_ntu120.py` 的映射逻辑。上传时同步这些入口、公共模块及
`run_caption_dual_gpu.py`（本次补上无新增结果时保持合并输出稳定的修正）。

**NTU60 XView 需要补生成 10,970 个样本。**预处理将相机2的所有样本放在前，
然后放相机3的所有样本；其训练集共37,646行，测试集为相机1的18,932行。
与已有XSub训练集40,091行相交26,676行，新增10,970行来自XSub测试受试者的
相机2/3，在XView中属于训练集。NTU120 XSub也不包含这些旧测试受试者，
因此不能用NTU120 XSub的新增描述补齐。
按完整ID（包含相机号）和实际坐标核验复用，不跨相机猜测文本。
XView默认 `--data_path ../data/MAMP/ntu/NTU60_XView.npz`；
复用源仍为下面同一份NTU60 XSub NPZ、完整描述及 `data/ntu/statistics`。
输出按XView相机优先顺序重新映射行号，XSub训练/测试划分与cache均保持独立。

NTU120 默认 `--benchmark XSub`，读取：

- `../data/MAMP/ntu120/NTU120_XSub.npz`（`--data_path`）。
- `../data/MAMP/ntu/NTU60_XSub.npz`（`--ntu60_data_path`）。
- 已完成的 `vlm_pilot/ntu60_xsub_global_local_v2/captions.json` 及同名
  `.metadata.json`（`--ntu60_captions`）。源描述必须完整，且模型、prompt SHA、revision 一致。
- `data/ntu/statistics` 和 `data/ntu120/statistics` 中的原始骨架名、performer；
  可通过 `--ntu60_statistics_dir` / `--ntu120_statistics_dir` 指定与实际 NPZ 匹配的统计目录。

脚本复现预处理的 **subject-major 行顺序**，按完整原始骨架 ID 映射，
逐个核对重合样本的完整坐标 SHA256，而非沿用 NTU60 行号、假设 NTU60 是
NTU120 的连续前缀，或只根据动作编号去重。只忽略全零 padding 和完全相同的
第二人物复制槽。当前仓库统计为 NTU120 63,026 行，其中 40,091 行复用
NTU60，22,935 行需要渲染/推理。实际 NPZ 数量或重合坐标不匹配会报错，
不会默默重新调用模型。两个大 NPZ 顺序加载，不同时驻留内存。

NTU120 单人预处理可能把 person0 完整复制到 person1：复用时只将原 person0
描述复制到该人物槽，不调用 VLM，映射保存在 `dataset_manifest.json`。
原 NPZ、原 NTU60 描述均保持原样。新生成的复制槽样本只画person0红色骨架、
只推理一份描述；数据集入口最终合并时复制回person0/person1两个原始槽，
使CLIP cache仍与原NPZ对齐。真实双人仍分别渲染、描述。

NTU120 cross-setup在项目中记为 **XSet**（即XSetup）。按“全部先渲染，再生成描述”
流程使用独立入口 `run_caption_ntu120_xset.py`，默认数据为
`../data/MAMP/ntu120/NTU120_XSet.npz`，输出改为 `vlm_pilot/ntu120_xset/`。
其训练是偶数setup 2,4,...,32，按setup号依次拼接，测试为奇数setup。
复用优先级为NTU60 XSub → NTU60 XView → NTU120 XSub。
**渲染阶段只读取骨架，不需要XView或NTU120 XSub的描述已经存在**；
按ID与坐标核验去重，并在 `dataset_manifest.json` 保存来源/目标行映射及原始人物槽数量。
描述阶段才要求三份完整源文本，检查模型/prompt/revision、覆盖和人物槽，
将文本及metadata的哈希绑定到 `reuse_caption_sources.json` 并写入最终描述metadata。
开始描述生成后，源文本变化会被拒绝，防止续跑混用不同描述。
其中XSub/XView的单人零槽与NTU120的复制槽按实际坐标适配。
当前统计54,468个训练样本，可合计复用31,965个，只新增22,503个；
额外源默认读取 `--output_root` 下的 `ntu60_xview/captions.json`、
`ntu120_xsub/captions.json`，NPZ分别为上述NTU60 XView和NTU120 XSub默认路径；
可用同名 `--ntu60_xview_captions` / `--ntu120_xsub_captions` 及
`--ntu60_xview_data_path` / `--ntu120_xsub_data_path` 覆盖。
新入口默认stage=render；其渲染与描述都使用同一独立入口。
旧 `run_caption_ntu120.py --benchmark XSet` 仍保留给此前已经创建的旧计划，
该旧入口的渲染前置仍包含完整源描述；不要混用两种入口的XSet计划。
已渲染但尚未生成描述的旧计划，先按4.5执行人数修复，再继续原来的描述命令。

PKU 默认分别读取 `../data/MAMP/pku_v1/PKUv1_XSub.npz` 和
`../data/MAMP/pku_v2/PKUv2_XSub.npz`，可用 `--pkuv1_data_path` /
`--pkuv2_data_path` 覆盖。默认依次处理 v1、v2；只处理一个版本时添加
`--versions v1` 或 `--versions v2`。

默认保存位置如下，`--output_root` 可统一更换 `vlm_pilot` 这一层：

| 数据集 | 最终完整描述 | 仅新生成的描述与 GPU 分片 | GIF 与渲染元数据 |
|---|---|---|---|
| NTU60 XView | `vlm_pilot/ntu60_xview/captions.json` | `vlm_pilot/ntu60_xview/generated/` | `vlm_pilot/ntu60_xview/rendered/` |
| NTU120 XSub | `vlm_pilot/ntu120_xsub/captions.json` | `vlm_pilot/ntu120_xsub/generated/` | `vlm_pilot/ntu120_xsub/rendered/` |
| NTU120 XSet | `vlm_pilot/ntu120_xset/captions.json` | `vlm_pilot/ntu120_xset/generated/` | `vlm_pilot/ntu120_xset/rendered/` |
| PKU v1 | `vlm_pilot/pku_v1_xsub/captions.json` | `vlm_pilot/pku_v1_xsub/generated/` | `vlm_pilot/pku_v1_xsub/rendered/` |
| PKU v2 | `vlm_pilot/pku_v2_xsub/captions.json` | `vlm_pilot/pku_v2_xsub/generated/` | `vlm_pilot/pku_v2_xsub/rendered/` |

最终描述保留各自 NPZ 的原始 `sample_index`，按行号排序；NTU120、XView最终文件
包含“复用＋新生成”，其 `generated/captions.json` 只含新增行。
均带 `.metadata.json`，沿用 `macdiff.caption_text.v2/global_local`，可供
`cache_clip_motion_text.py` 读取。v1/v2 的 manifest、分片、日志和索引空间分别保存。

可先添加 `--dry_run`，校验输入并查看复用/新增数量，不写文件、不加载 VLM。
添加 `--stage render` 只做渲染，随后用同一命令的 `--stage caption` 做推理；
caption 阶段只读取已保存的计划、GIF 和描述，不重新加载 NPZ 数组。
`--merge_only` 只合并已有结果。正式运行中断后重跑原命令即可。
`--start_index` / `--max_samples` 限定原 NPZ 行号范围，不会重编号；
小范围输出的 metadata 中 `dataset_complete=false`，全量完成才为 true。
缺失描述会保存部分结果并返回非零状态。

### 4.2 按三个阶段执行：保存骨架 → 保存描述 → 保存 CLIP cache

所有命令均在仓库根目录执行。NTU60 XSub的描述和v3 cache已有，不重做。
下面的NTU60 XView、NTU120 XSub/XSet和PKU均仅处理训练集；PKU v1/v2每个阶段都分别保存。

**第一步：渲染新增骨架并持久化。**不加载Qwen或CLIP。依次执行：

```bash
python tools/vlm_pilot/run_caption_ntu60_xview.py --stage render
python tools/vlm_pilot/run_caption_ntu120.py --stage render
python tools/vlm_pilot/run_caption_pku.py --stage render
python tools/vlm_pilot/run_caption_ntu120_xset.py --stage render
```

每个新样本保存 `rendered/train_<原行号>/preview.gif` 和 `render_metadata.json`，
根目录保存 `dataset_manifest.json`。XView仅渲染10,970行，NTU120 XSub仅渲染22,935行，
XSet仅渲染22,503行。XSet此时不要求其他新划分已有文本；
重合样本的描述已存在于NTU60 XSub，不再次渲染/推理。
原始ID与完整坐标验证在此阶段完成。续跑会核验GIF和渲染配置，再跳过完整样本。
可以先加 `--dry_run` 核对数量，不产生文件。

**第二步：只读取保存的GIF，生成并保存描述。**在Qwen推理环境依次执行：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python tools/vlm_pilot/run_caption_ntu60_xview.py --stage caption
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python tools/vlm_pilot/run_caption_ntu120.py --stage caption
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python tools/vlm_pilot/run_caption_ntu120_xset.py --stage caption
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python tools/vlm_pilot/run_caption_pku.py --stage caption
```

此阶段不重新加载NPZ数组，也不进行渲染。先核对请求范围的GIF是否齐全；
解码时检查帧数。每卡一个常驻模型，分片、失败重试和日志在 `generated/` 中。
结构检查通过的描述与复用描述合并到数据集根目录的 `captions.json`，
其 `.metadata.json` 中 `summary.dataset_complete=true` 才表示全量完成。
XView应为37,646行，NTU120 XSub应为63,026行，XSet应为54,468行；PKU按各自实际NPZ训练行数检查。
每个人保留 `global` 和固定顺序的六个 `local` 部位句子。

描述阶段先完成XView和NTU120 XSub，再运行XSet的 `--stage caption`，以复用完整文本。
XSet的GIF在第一步已经保存，不需要重新渲染；此阶段会校验源文本并绑定其身份。

**第三步：对完整描述使用冻结的CLIP text encoder，保存v3句向量cache。**
在此前可提取CLIP的环境中依次执行以下五条命令。
每条对应的NPZ和描述必须属于同一划分；使用完整合并后的 `captions.json`，
不要把只含新增行的 `generated/captions.json` 传给cache脚本。

```bash
OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 python -u cache_clip_motion_text.py --data-path ../data/MAMP/ntu/NTU60_XView.npz --captions vlm_pilot/ntu60_xview/captions.json --clip-model /home/user9/public3/swr/models/clip-vit-base-patch32 --output-dir vlm_pilot/ntu60_xview/clip_sentence_cache_v3 --batch-size 16 --resume
OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 python -u cache_clip_motion_text.py --data-path ../data/MAMP/ntu120/NTU120_XSub.npz --captions vlm_pilot/ntu120_xsub/captions.json --clip-model /home/user9/public3/swr/models/clip-vit-base-patch32 --output-dir vlm_pilot/ntu120_xsub/clip_sentence_cache_v3 --batch-size 16 --resume
OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 python -u cache_clip_motion_text.py --data-path ../data/MAMP/ntu120/NTU120_XSet.npz --captions vlm_pilot/ntu120_xset/captions.json --clip-model /home/user9/public3/swr/models/clip-vit-base-patch32 --output-dir vlm_pilot/ntu120_xset/clip_sentence_cache_v3 --batch-size 16 --resume
OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 python -u cache_clip_motion_text.py --data-path ../data/MAMP/pku_v1/PKUv1_XSub.npz --captions vlm_pilot/pku_v1_xsub/captions.json --clip-model /home/user9/public3/swr/models/clip-vit-base-patch32 --output-dir vlm_pilot/pku_v1_xsub/clip_sentence_cache_v3 --batch-size 16 --resume
OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 python -u cache_clip_motion_text.py --data-path ../data/MAMP/pku_v2/PKUv2_XSub.npz --captions vlm_pilot/pku_v2_xsub/captions.json --clip-model /home/user9/public3/swr/models/clip-vit-base-patch32 --output-dir vlm_pilot/pku_v2_xsub/clip_sentence_cache_v3 --batch-size 16 --resume
```

cache脚本在加载CLIP前检查完整训练行覆盖、重复行、七句非空以及人物槽与
实际 `x_train` 是否一致，均不读取类别标签。CLIP处于eval/冻结状态，
每句话独立编码、取投影后的句向量、转FP32并做unit RMS；超过77 BPE token会报错，
不会截断。当前512维CLIP对应如下文件：

| 文件 | 形状/含义 |
|---|---|
| `person_features.npy` | `[N,2,512]`，global句向量 |
| `token_features.npy` | `[N,2,6,512]`，六个完整local句向量 |
| `person_valid.npy` | `[N,2]`，原NPZ人物槽是否存在 |
| `token_mask.npy` | `[N,2,6]`，有效部位句向量mask |
| `samples.json` | 按原行号排列的描述副本 |
| `manifest.json` | NPZ/描述/模型/实现SHA、进度、文件校验和；`complete=true`才可训练 |

每个划分独立构建cache，因为训练行顺序、N和NPZ身份不同，不能直接共用XSub的
旧cache。此处复用的是已有文本；第三步对目标划分的全部文本编码。
训练配置的 `text_cache` 应指向对应的 `clip_sentence_cache_v3`。
三个阶段均可使用原命令续跑；完整描述和metadata在无新增结果时保持原样，
避免只刷新合并时间戳就使已有CLIP cache失效。cache实现及已有NTU60 cache未改。

本地检查：真实统计划分和重叠数量、真实GIF渲染、模拟VLM与CLIP的三阶段衔接、
完整覆盖/人物槽/七向量顺序、失败续跑和完整cache复用。尚未用服务器NPZ或真实GPU模型全量生成。

### 4.3 五种协议的检查范围与服务器只读检查

范围固定为NTU60 XSub/XView、NTU120 XSub/XSet、PKU XSub；PKU分v1/v2，
因此实际是六份独立NPZ、描述和cache。下表NTU数量来自当前仓库statistics，
不是对服务器实际文件已完成状态的声明。新增数量按已有NTU60 XSub → XView →
NTU120 XSub → XSet的复用顺序计算。

| 划分 | 训练行 | 测试行 | 可复用已有描述 | 需要新增描述 |
|---|---:|---:|---:|---:|
| NTU60 XSub | 40,091 | 16,487 | 已有完整描述 | 0 |
| NTU60 XView | 37,646 | 18,932 | 26,676 | 10,970 |
| NTU120 XSub | 63,026 | 50,919 | 40,091 | 22,935 |
| NTU120 XSet | 54,468 | 59,477 | 31,965 | 22,503 |
| PKU v1 XSub | 18,840（服务器回传） | 2,704 | 与NTU不混用 | 18,840 |
| PKU v2 XSub | 5,299（服务器回传） | 1,613 | 与v1分开 | 5,299 |

在含NumPy/Pillow的环境中检查全部六份数据与三个阶段产物，不加载Qwen/CLIP、不写文件：

```bash
python tools/vlm_pilot/check_text_pipeline.py
```

检查NPZ中 `x_train/x_test` 头信息与NTU统计数量、原始行号的渲染/复用完整分区、
GIF和metadata存在/来源/参数匹配、完整global+6local描述与实际原始人物槽一致，
以及cache的NPZ/特征文件SHA、形状、mask、unit RMS、当前描述/metadata身份。
缺失、错误和不完整分别报告，任一项未通过返回非零；缺少文件不会当作已完成。
尚未生成 `dataset_manifest.json` 的新划分报告 `render.status=not_planned`，
`expected_new_renders=null`，不会再将全部训练行误标为“新增渲染/缺失渲染”。
`estimated_new_after_prerequisites`单独给出前置描述源齐全后的预计新增数；
NTU的实际复用行仍须在第一步核对ID/坐标，已有计划才报告确定的新增与缺失数量。
原NTU60 XSub没有新版manifest，仍按既有全量GIF检查，标记 `plan_kind=legacy_full_train`。
PKU的原始split文件未包含在仓库，当前不能从本地静态代码证明服务器NPZ的原始
受试者划分正确；检查器能验证其实际行数及各阶段与该NPZ的一致性。

默认cache校验需要读取NPZ/特征文件做完整hash；只看数据数量可执行
`python tools/vlm_pilot/check_text_pipeline.py --stages data`。
`--quick`仅将cache检查降为size/header模式，报告会明确未做全量hash；
`--check_gifs`额外解码每个新增GIF核验帧数。可用 `--datasets`、`--stages`
限定范围；`--data_root`、`--output_root` 可覆盖公共根目录。
既有NTU60 XSub的旧保存位置分别用 `--ntu60_xsub_rendered`、
`--ntu60_xsub_captions`、`--ntu60_xsub_cache` 覆盖。
训练缓存仍按各自NPZ顺序构建，不能把一个协议的测试行文本用于该协议训练。

**2026-10-17服务器检查回传：**六份NPZ头信息均读到，NTU各训练/测试数量与上表相同；
PKU v1实际为train18,840/test2,704，v2为train5,299/test1,613。
NTU60 XSub的40,091份渲染文件/metadata、40,091条描述（47,608个人物槽）和
512维/7句v3 cache通过检查，NPZ与特征全量hash通过；此轮未解码GIF帧，
不等于动作描述语义质量已重新抽检。其余五份的描述/cache在默认目录缺失，
旧版无计划时的全量“expected_new_renders”只是检查器回退计数，不是实际新增任务量。
已据此修正上面的未规划报告；已有NTU60 XSub无需重做。

### 4.4 与原NTU60 XSub渲染的一致性及速度检查

新入口仍直接调用原 `render_sample_frames`、`save_gif`、`make_metadata`：
默认32帧/8fps、960×360、font18、正面XY/侧面ZY、逐帧person0 root居中，
median3、Savgol5/poly2、插值gap2，GIF optimize=False/disposal2均一致。
新增回归测试对同环境的单人、复制人物槽、双人输入，核对新旧入口GIF逐字节、
config和render元数据一致。不同数据集的有效帧数/人物槽不同，计算量仍可能不同。
该一致性测试比较的是两个入口调用当前同一渲染器；4.5的人数修复已使复制槽只画一次。

本地300帧合成样本CPU对照（不是服务器速度）：两轮各3例，旧入口1.372/1.434秒，
新入口1.437/1.412秒。分阶段样本计时中，绘制/平滑约0.078–0.106秒，
GIF编码约0.346–0.404秒；profile主要耗时是Pillow逐帧调色板量化。
该小规模测试不提供服务器吞吐或总任务ETA。

新旧运行路径的差别：

- 新NTU入口首次运行会额外读取复用源NPZ/描述、逐样本核对坐标hash，再加载目标NPZ。
  这是首次计划构建开销，已保存且身份匹配的计划会跳过这部分。
- 新入口目前每100个新样本打印一次，原入口每个样本打印并给ETA。
  所以两行日志之间的等待包含100个样本；判断单样本速度可读取下述已保存计时。
- 新渲染是CPU单进程，`--gpus 0,1`只影响后续Qwen推理；原渲染默认也单进程，
  但原脚本支持用户手动 `--num_shards` 多进程。若旧任务实际用了多进程，不能直接对比总吞吐。
- 两边读取NPZ的 `x_train` 都会将完整数组读入内存；旧入口的mmap_mode参数不使NPZ成员变为memmap。
  新入口的坐标核验、顺序加载多个NPZ和共享磁盘I/O会增加启动等待。

以XView为例，只读最近20个已完成样本保存的 `render_seconds`（包括绘制和平滑/GIF写入，
不包括首次NPZ加载、去重扫描、JSON写入及后续推理）：

```bash
python -c 'import json,pathlib,statistics; fs=sorted(pathlib.Path("vlm_pilot/ntu60_xview/rendered").glob("train_*/render_metadata.json"))[-20:]; rs=[json.loads(p.read_text()) for p in fs]; print({"samples":len(rs),"mean_seconds":statistics.mean(r["render_seconds"] for r in rs) if rs else None,"median_seconds":statistics.median(r["render_seconds"] for r in rs) if rs else None})'
```

以上速度数值来自人数修复前的检查。更新至4.5后，旧计划先修复再续跑。

### 4.5 修复NTU120复制人物槽被误判为双人

人数现在按完整有效序列的非空人物槽判断；只有person0和person1的**全部坐标完全相同**
才合并为一个可见人物。在平滑和32帧采样前比较，避免把姿势接近、只在采样帧重合、
或有部分缺失帧的真实双人误合并。复制槽只画红色person0，metadata记录
`actor_layout_version=distinct_full_sequence_v2`、`duplicate_person1=true`、
`raw_active_person_indices=[0,1]`、`rendered_person_indices=[0]`，人数为1。
`active_rendered_frames_per_person`仍记录采样后两个原始槽的有效帧数，不是可见独立人数。

VLM对该样本只输出一条人物描述。**请使用数据集根目录的最终 `captions.json` 编码CLIP**：
数据集入口会把该描述复制回原NPZ的两个槽位；`generated/captions.json`只保存VLM直接输出，
复制槽在该中间文件中只有一条描述，不能直接用来为原双槽NPZ建cache。
已完成的NTU60 XSub描述和cache不需要重做，模型的`one_person`和训练输入没有修改。

服务器同步本次修改的`tools/vlm_pilot`文件后，对“只有GIF、没有描述/cache”的旧输出先执行：

```bash
python tools/vlm_pilot/repair_render_actor_counts.py --dry_run
python tools/vlm_pilot/repair_render_actor_counts.py
```

默认处理`vlm_pilot`下已有计划的NTU60 XView、NTU120 XSub/XSet、PKU v1/v2；
尚未建立计划的划分跳过。也可指定单个目录：

```bash
python tools/vlm_pilot/repair_render_actor_counts.py vlm_pilot/ntu120_xsub
```

修复仅加载目标NPZ的`x_train`，不要求其他划分已经有描述，不加载Qwen/CLIP。
先核对旧生产代码的已知SHA、原NPZ身份和全部已保存metadata，再只重渲染复制槽等人数变化样本；
其他GIF/metadata不改。日志分别给出`rerender`、缺失渲染数和真正只有person1的样本数。
会备份原计划为`dataset_manifest.before_actor_layout_v2.json`，保存`actor_layout_upgrade.json`，
全部修复成功后才更新计划的实现SHA。中断后可重复同一命令，已修复样本跳过。
若有缺失GIF，修复后用原数据集入口的`--stage render`补齐，再按4.2顺序生成描述。
有已有描述/cache、未知旧代码SHA或NPZ变化的目录会拒绝迁移，需另设输出目录。

真正“person0全空、仅person1非空”的样本人数会正确记为1，保留原person1索引和蓝色；
现有文本/cache协议要求person0，因此描述阶段会明确报错，不偷偷把person1重编号为person0。
这类情况与两个槽完全重复不同，需要根据实际NPZ排查来源。

本地检查覆盖重复槽的颜色/人数、真实双人、采样外坐标差异、仅person1、旧GIF修复与中断续跑、
新描述复制回原槽及CLIP计划校验：

```bash
python -m unittest tests.test_render_actor_counts tests.test_dataset_captions tests.test_caption_output tests.test_caption_dual_gpu tests.test_rendered_gif_recovery tests.test_structured_text_cache
```

## 5. Randomly inspect accepted captions

### Single-GPU AWQ with offline vLLM over existing GIFs

The rendered-GIF entry point also accepts `--engine vllm`. It loads one engine
with tensor parallel size 1 and one active request. It preserves the GIF frames,
FPS metadata, prompt and validation; preprocessing is not repeated by vLLM.
This is sequential full-dataset generation, not a concurrent throughput benchmark.
AWQ is detected from checkpoint configuration. Engine/preprocessing errors are
written to JSONL and then stop the vLLM run; invalid captions receive bounded
validation retries. Use a fresh output file when changing engines or settings.

In an independent, compatible Linux vLLM environment, test train index 2:

```bash
OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
python tools/vlm_pilot/caption_qwen3vl_rendered_train_transformers.py \
  --engine vllm \
  --rendered_root vlm_pilot/ntu60_xsub_train_rendered_v3_2view_smooth_w5 \
  --model /home/user9/public3/swr/models/Qwen3-VL-30B-A3B-Instruct-AWQ \
  --prompt_path tools/vlm_pilot/skeleton_motion_prompt_v1.txt \
  --output_path vlm_pilot/30b_a3b_awq_test_2.jsonl \
  --start_index 2 --max_samples 1 --max_retries 0 \
  --num_shards 1 --shard_id 0 --max_model_len 16384 \
  --gpu_memory_utilization 0.90 --enforce_eager
```

Use `python -m json.tool vlm_pilot/30b_a3b_awq_test_2.jsonl` to inspect the raw
response as well as validation errors. For a full run remove `--start_index`
and `--max_samples`, set `--max_retries 2`, select a new output path, and add
`--resume`. Keep that full-run output path on subsequent restarts. Add
`--dry_run` to check one GIF and its expanded prompt without loading the model.

The 24GB/32-frame AWQ run still requires server verification. CPU-only adapter
checks: `python -m unittest tests.test_rendered_vllm`.

Read both shard files, deduplicate accepted results by `sample_index`, and print
five random samples with all person-level fields:

```bash
python tools/vlm_pilot/inspect_random_captions.py \
  --input_paths vlm_pilot/ntu60_xsub_train_person_captions_v1_shard0.jsonl \
                vlm_pilot/ntu60_xsub_train_person_captions_v1_shard1.jsonl \
  --num_samples 5
```

Add `--compact` to print only the final `text` fields, or `--seed 42` to make
the random selection reproducible.

If a visualization already exists, do not load the NPZ again. Read its
`render_metadata.json` and look up the matching accepted caption directly:

```bash
python tools/vlm_pilot/inspect_random_captions.py \
  --input_paths vlm_pilot/ntu60_xsub_train_person_captions_v1_shard0.jsonl \
                vlm_pilot/ntu60_xsub_train_person_captions_v1_shard1.jsonl \
  --visualization_dirs vlm_pilot/sample_000000 \
  --compact
```

Use `--sample_indices 0 42 108` when the desired train indices are already
known. Both lookup modes scan only the caption JSONL files and do not read
`x_train` or `y_train`.

To export labels and skeleton animations into a standalone review folder, add
the aligned NTU60 NPZ and a new output directory:

```bash
python tools/vlm_pilot/inspect_random_captions.py \
  --input_paths vlm_pilot/ntu60_xsub_train_person_captions_v1_shard0.jsonl \
                vlm_pilot/ntu60_xsub_train_person_captions_v1_shard1.jsonl \
  --data_path ../data/MAMP/ntu/NTU60_XSub.npz \
  --output_dir vlm_pilot/random_caption_review_seed42 \
  --num_samples 10 \
  --seed 42
```

The root `index.html` shows every selected GIF beside its `y_train` class and
person captions. Each sample subdirectory also contains `preview.gif`,
`preview_middle.png`, `summary.txt`, and the complete `review.json`. The output
directory must not already exist, which prevents accidental review overwrites.
