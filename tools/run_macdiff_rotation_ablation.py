#!/usr/bin/env python3
"""Sequential native NTU60 XSub PT/LP rotation ablation; Python 3.8 compatible."""
import argparse
import ast
import copy
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shlex
import signal
import statistics
import subprocess
import sys
from datetime import datetime
import zipfile

VARIANTS = ("no_rot", "rot")
PT_EPOCHS = 500
LP_EPOCHS = 100


def make_configs(data_path, seed=0, num_workers=10):
    feeder = {
        "data_path": str(data_path), "split": "train", "debug": False,
        "window_size": 120, "p_interval": [0.5, 1], "normalization": False,
        "random_choose": False, "random_shift": False, "random_move": False,
        "random_rot": False, "source_rot": False, "flip": False,
        "motion_noise": 0, "impulse_noise": 0, "joint_noise": [1, 0.005],
        "frame_noise": [0, 0], "vel": False, "bone": False,
    }
    model = {
        "dim_in": 3, "dim_feat": 256, "depth": 8, "num_heads": 8,
        "mlp_ratio": 4, "num_frames": 120, "num_joints": 25,
        "patch_size": 1, "t_patch_size": 4, "qkv_bias": True,
        "qk_scale": None, "drop_rate": 0.0, "attn_drop_rate": 0.0,
        "drop_path_rate": 0.0, "self_shift": False,
        "input_mean": [-0.0058, -0.1333, -0.0246],
        "input_var": [0.0206, 0.0805, 0.0218],
    }
    pt_model = dict(model, dim_t_embed=64, decoder_depth=3,
                    diff_prediction="noise", diff_steps=1000,
                    diff_noise_schedule=["inverse_cosine", 1],
                    lambda_loss_uni=0.02, layer_mask_ratio=0.0,
                    uncond_ratio=0.1, one_person=True, loss_reweight=None)
    no_rot = {
        "feeder": "feeder.feeder_ntu.Feeder", "train_feeder_args": feeder,
        "model": "model.transformer_macdiff.Transformer", "model_args": pt_model,
        "epochs": PT_EPOCHS, "warmup_epochs": 20, "min_lr_epochs": 20,
        "batch_size": 64, "accum_iter": 1, "lr": 1e-3, "min_lr": 1e-5,
        "weight_decay": 0.05, "mask_ratio": 0.9, "motion_aware_tau": -1,
        "enable_amp": True, "enable_ose": False, "text_cache": "",
        "max_train_steps": 0, "seed": seed, "num_workers": num_workers,
        "resume": "", "start_epoch": 0,
    }
    rot = copy.deepcopy(no_rot)
    rot["train_feeder_args"]["random_rot"] = True
    lp_train = copy.deepcopy(feeder)
    lp_train.update(random_rot=True, joint_noise=[0, 0])
    lp_val = copy.deepcopy(lp_train)
    lp_val.update(split="test", p_interval=[0.95], random_rot=False)
    lp = {
        "feeder": "feeder.feeder_ntu.Feeder", "train_feeder_args": lp_train,
        "val_feeder_args": lp_val, "model": "model.transformer_downstream.Transformer",
        "model_args": dict(model, num_classes=60, protocol="linprobe2", act_layer="nn.GELU"),
        "epochs": LP_EPOCHS, "warmup_epochs": 0, "min_lr_epochs": 10,
        "batch_size": 128, "accum_iter": 1, "lr": 0.1, "min_lr": 0.0,
        "weight_decay": 0.0, "smoothing": 0.0, "motion_aware_tau": -1,
        "enable_amp": False, "dist_eval": True, "eval_epoch": 1,
        "seed": seed, "num_workers": num_workers, "resume": "", "start_epoch": 0,
    }
    return {"no_rot": no_rot, "rot": rot, "lp": lp}


def validate_pair(configs):
    control = copy.deepcopy(configs["no_rot"])
    treatment = copy.deepcopy(configs["rot"])
    if control["train_feeder_args"].pop("random_rot") is not False:
        raise ValueError("no_rot must disable PT rotation")
    if treatment["train_feeder_args"].pop("random_rot") is not True:
        raise ValueError("rot must enable PT rotation")
    if control != treatment:
        raise ValueError("PT configs differ beyond train_feeder_args.random_rot")
    for name in VARIANTS:
        cfg = configs[name]
        if (cfg["epochs"], cfg["lr"], cfg["min_lr"], cfg["model_args"]["decoder_depth"]) != (500, 1e-3, 1e-5, 3):
            raise ValueError("Unexpected PT protocol")
        for field in ("input_mean", "input_var", "self_shift"):
            if cfg["model_args"][field] != configs["lp"]["model_args"][field]:
                raise ValueError("PT/LP normalization mismatch: " + field)


def validate_source(repo, configs):
    # Check accepted arguments without importing training modules/Torch.
    for stage, names in (("pretrain", VARIANTS), ("linprobe", ("lp",))):
        tree = ast.parse((repo / ("main_" + stage + ".py")).read_text(encoding="utf-8"))
        accepted = set()
        for call in ast.walk(tree):
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == "add_argument":
                dest = next((kw.value.value for kw in call.keywords if kw.arg == "dest" and isinstance(kw.value, ast.Constant)), None)
                for arg in call.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("--"):
                        accepted.add(dest or arg.value[2:].replace("-", "_"))
        for name in names:
            unknown = set(configs[name]) - accepted
            if unknown:
                raise ValueError("Unsupported {} config keys: {}".format(stage, sorted(unknown)))
    for filename, classname, kwargs in (
        ("model/transformer_macdiff.py", "Transformer", configs["no_rot"]["model_args"]),
        ("model/transformer_downstream.py", "Transformer", configs["lp"]["model_args"]),
        ("feeder/feeder_ntu.py", "Feeder", configs["no_rot"]["train_feeder_args"]),
        ("feeder/feeder_ntu.py", "Feeder", configs["lp"]["train_feeder_args"]),
        ("feeder/feeder_ntu.py", "Feeder", configs["lp"]["val_feeder_args"]),
    ):
        tree = ast.parse((repo / filename).read_text(encoding="utf-8"))
        cls = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == classname][-1]
        init = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
        accepted = {arg.arg for arg in init.args.args + init.args.kwonlyargs}
        unknown = set(kwargs) - accepted
        if unknown:
            raise ValueError("Unsupported {} constructor keys: {}".format(filename, sorted(unknown)))


def build_commands(root, port, python=sys.executable):
    stages = []
    for variant in VARIANTS:
        pt = root / variant / "pt"
        lp = root / variant / "lp"
        launcher = [python, "-m", "torch.distributed.launch", "--nproc_per_node=2", "--master_port=" + str(port)]
        stages.append((variant, "pt", launcher + [
            "main_pretrain.py", "--config", str(root / "configs" / (variant + ".json")),
            "--output_dir", str(pt), "--log_dir", str(pt / "tensorboard"),
            "--epochs", "500", "--lr", "0.001", "--min_lr", "0.00001",
            "--batch_size", "64", "--accum_iter", "1",
        ]))
        stages.append((variant, "lp", launcher + [
            "main_linprobe.py", "--config", str(root / "configs" / "lp.json"),
            "--finetune", str(pt / "checkpoint-499.pth"),
            "--output_dir", str(lp), "--log_dir", str(lp / "tensorboard"),
            "--epochs", "100", "--lr", "0.1", "--min_lr", "0",
            "--batch_size", "128", "--accum_iter", "1", "--dist_eval",
        ]))
    return stages


def read_epoch_log(path, expected_epochs, metric):
    rows = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            epoch = row.get("epoch")
            value = row.get(metric)
            if type(epoch) is not int or epoch in rows or epoch not in range(expected_epochs):
                raise ValueError("Invalid/duplicate epoch at {}:{}".format(path, line_number))
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("Missing/nonfinite {} at {}:{}".format(metric, path, line_number))
            if metric == "test_acc1" and not 0 <= value <= 100:
                raise ValueError("Accuracy outside [0, 100] at {}:{}".format(path, line_number))
            rows[epoch] = row
    missing = set(range(expected_epochs)) - set(rows)
    if missing:
        raise ValueError("Incomplete {}: missing epochs {}".format(path, sorted(missing)))
    return [rows[epoch] for epoch in range(expected_epochs)]


def lp_stats(path):
    rows = read_epoch_log(path, LP_EPOCHS, "test_acc1")
    best = max(rows, key=lambda row: row["test_acc1"])
    tail = [row["test_acc1"] for row in rows[-20:]]
    return {
        "best_lp_acc": best["test_acc1"], "best_epoch": best["epoch"],
        "last_lp_acc": rows[-1]["test_acc1"], "last20_mean": statistics.mean(tail),
        "last20_population_std": statistics.pstdev(tail), "lp_log": str(Path(path).resolve()),
    }


def write_summary(root):
    results = {name: lp_stats(root / name / "lp" / "log.txt") for name in VARIANTS}
    delta = results["rot"]["best_lp_acc"] - results["no_rot"]["best_lp_acc"]
    payload = {"protocol": "native_ntu60_xsub_decoder3_pt500_rotation_ablation_v1",
               "accuracy_unit": "percent", "results": results,
               "rot_minus_no_rot_best_pp": delta}
    write_json(root / "rotation_comparison.json", payload)
    with (root / "rotation_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["variant"] + list(results["no_rot"]))
        writer.writeheader()
        for name in VARIANTS:
            writer.writerow(dict(variant=name, **results[name]))
    print("\n=== Best LP accuracy (checkpoint-499, LP100) ===", flush=True)
    for name in VARIANTS:
        result = results[name]
        print("{}: {:.6f}% @ epoch {}".format(name, result["best_lp_acc"], result["best_epoch"]), flush=True)
    print("rot - no_rot: {:+.6f} percentage points".format(delta), flush=True)
    print("Saved: {}".format(root / "rotation_comparison.json"), flush=True)
    print("Saved: {}".format(root / "rotation_comparison.csv"), flush=True)
    return payload


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def preflight(data_path):
    for module in ("torch", "numpy", "yaml", "timm", "tensorboard"):
        if importlib.util.find_spec(module) is None:
            raise RuntimeError("Missing dependency: {}. Use the macdiff training environment.".format(module))
    import numpy as np
    import torch
    import yaml
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("This protocol requires exactly two visible CUDA GPUs")
    if not torch.distributed.is_available() or not torch.distributed.is_nccl_available():
        raise RuntimeError("NCCL distributed training is unavailable")
    if not data_path.is_file():
        raise FileNotFoundError("NTU60 XSub dataset not found: " + str(data_path))
    shapes = {}
    # Read NPY headers inside NPZ without loading the large arrays a second time.
    with zipfile.ZipFile(data_path) as archive:
        for name, count in (("x_train", 40091), ("y_train", 40091), ("x_test", 16487), ("y_test", 16487)):
            with archive.open(name + ".npy") as handle:
                version = np.lib.format.read_magic(handle)
                if version == (1, 0):
                    shape, _, _ = np.lib.format.read_array_header_1_0(handle)
                elif version == (2, 0):
                    shape, _, _ = np.lib.format.read_array_header_2_0(handle)
                else:
                    raise ValueError("Unsupported NPY header version: " + str(version))
            if shape[0] != count:
                raise ValueError("Unexpected {} sample count: {} != {}".format(name, shape[0], count))
            if name.startswith("x_") and (len(shape) != 3 or shape[-1] != 150):
                raise ValueError("Expected flattened NTU joint data [N,T,150]: " + str(shape))
            if name.startswith("y_") and (len(shape) != 2 or shape[-1] != 60):
                raise ValueError("Expected NTU60 labels [N,60]: " + str(shape))
            shapes[name] = list(shape)
    print("Preflight passed; recording dataset SHA256...", flush=True)
    return {"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
            "numpy": np.__version__, "yaml": yaml.__version__,
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(2)],
            "data_path": str(data_path), "data_shapes": shapes,
            "data_sha256": sha256_file(data_path)}


def verify_checkpoint(path, cfg):
    import torch
    checkpoint = torch.load(str(path), map_location="cpu")
    if checkpoint.get("epoch") != PT_EPOCHS - 1:
        raise ValueError("Expected PT checkpoint epoch 499")
    saved = checkpoint.get("args")
    saved = saved if isinstance(saved, dict) else vars(saved)
    for key in ("model", "model_args", "train_feeder_args", "epochs", "lr", "min_lr",
                "batch_size", "accum_iter", "seed", "enable_ose", "text_cache"):
        if saved.get(key) != cfg[key]:
            raise ValueError("Checkpoint args mismatch for {}: {} != {}".format(key, saved.get(key), cfg[key]))
    decoder_ids = {int(key.split(".")[1]) for key in checkpoint["model"] if key.startswith("decoder_blocks.")}
    if decoder_ids != {0, 1, 2}:
        raise ValueError("Checkpoint does not contain exactly 3 native decoder blocks")
    return {"checkpoint": str(path), "epoch": 499, "decoder_depth": 3, "args_verified": True}


def run_stage(command, console_path, env):
    console_path.parent.mkdir(parents=True, exist_ok=False)
    print("\n$ " + shlex.join(command), flush=True)
    with console_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace", env=env,
                                   start_new_session=(os.name == "posix"))
        try:
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
            code = process.wait()
        except BaseException:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.wait()
            raise
        finally:
            process.stdout.close()
        if code:
            raise subprocess.CalledProcessError(code, command)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0,1"))
    parser.add_argument("--master-port", type=int, default=10270)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--data-path", default="../data/MAMP/ntu/NTU60_XSub.npz")
    parser.add_argument("--output-dir", help="A new directory; never reused or overwritten")
    parser.add_argument("--dry-run", action="store_true", help="Print configs/commands without writing or training")
    parser.add_argument("--summarize-only", action="store_true", help="Recompute best LP accuracies in --output-dir")
    args = parser.parse_args(argv)
    if args.summarize_only:
        if not args.output_dir or args.dry_run:
            parser.error("--summarize-only requires --output-dir and cannot use --dry-run")
        write_summary(Path(args.output_dir).resolve())
        return
    repo = Path(__file__).resolve().parents[1]
    if Path.cwd().resolve() != repo:
        parser.error("Run this script from the MacDiff repository root (no automatic cd)")
    gpus = [gpu.strip() for gpu in args.gpus.split(",")]
    if len(gpus) != 2 or any(not gpu for gpu in gpus) or len(set(gpus)) != 2:
        parser.error("--gpus must name two distinct GPUs, e.g. 0,1")
    if args.num_workers < 1:
        parser.error("--num-workers must be >=1 to retain the existing worker seed initialization")
    if not 1 <= args.master_port <= 65535:
        parser.error("--master-port must be in 1..65535")
    root = Path(args.output_dir or ("output_dir/native_rotation_dec3_ep500_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))).resolve()
    data_path = Path(args.data_path).resolve()
    configs = make_configs(data_path, args.seed, args.num_workers)
    validate_pair(configs)
    validate_source(repo, configs)
    stages = build_commands(root, args.master_port)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(gpus),
               OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "1"), PYTHONUNBUFFERED="1")
    print("Output root: " + str(root), flush=True)
    print("PT: native decoder3, 500 epochs, AdamW lr=1e-3/min_lr=1e-5, warmup20, batch=2*64=128", flush=True)
    print("LP: checkpoint499, 100 epochs, SGD lr=.1/min_lr=0, warmup0, batch=2*128=256", flush=True)
    if args.dry_run:
        print(json.dumps(configs, indent=2), flush=True)
        for variant, stage, command in stages:
            print("[{} {}] {}".format(variant, stage, shlex.join(command)), flush=True)
        print("Dry run: no files written, no training launched.", flush=True)
        return
    if root.exists():
        raise FileExistsError("Refusing to reuse an existing output directory: " + str(root))
    os.environ["CUDA_VISIBLE_DEVICES"] = env["CUDA_VISIBLE_DEVICES"]
    runtime = preflight(data_path)
    root.mkdir(parents=True, exist_ok=False)
    (root / "configs").mkdir()
    for name, cfg in configs.items():
        # JSON is accepted by the existing yaml.FullLoader training entry points.
        write_json(root / "configs" / (name + ".json"), cfg)
    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    git_status = subprocess.check_output(["git", "status", "--short"], text=True)
    source_files = ["main_pretrain.py", "main_linprobe.py", "engine_pretrain.py", "engine_linprobe.py",
                    "model/transformer_macdiff.py", "model/transformer_downstream.py", "model/util.py",
                    "feeder/feeder_ntu.py", "feeder/tools.py", "util/lr_sched.py", "util/misc.py",
                    "tools/run_macdiff_rotation_ablation.py"]
    source_hashes = {name: sha256_file(repo / name) for name in source_files}
    config_hashes = {name: sha256_file(root / 'configs' / (name + '.json')) for name in configs}
    manifest = {"status": "running", "repo": str(repo), "git_head": git_head,
                "git_status": git_status, "runtime": runtime, "configs": configs,
                "cuda_visible_devices": env["CUDA_VISIBLE_DEVICES"], "seed": args.seed,
                "source_sha256": source_hashes, "config_sha256": config_hashes, "completed_stages": [],
                "commands": [{"variant": v, "stage": s, "argv": cmd} for v, s, cmd in stages],
                "caveat": "Two-GPU LP local BN batch128 differs from official four-GPU batch64. Both variants use identical evaluation. A single seed measures this run's difference, not seed stability."}
    manifest_path = root / "run_manifest.json"
    write_json(manifest_path, manifest)
    try:
        for variant, stage, command in stages:
            if any(sha256_file(repo / name) != digest for name, digest in source_hashes.items()):
                raise RuntimeError("Source changed during the experiment; refusing to mix code versions")
            if any(sha256_file(root / 'configs' / (name + '.json')) != digest for name, digest in config_hashes.items()):
                raise RuntimeError('Config snapshot changed during the experiment')
            stage_dir = root / variant / stage
            run_stage(command, stage_dir / "console.log", env)
            if stage == "pt":
                rows = read_epoch_log(stage_dir / "log.txt", PT_EPOCHS, "train_loss")
                if not math.isclose(rows[-1]["train_lr"], 1e-5, rel_tol=0, abs_tol=1e-9):
                    raise ValueError("PT did not finish at min_lr=1e-5")
                record = verify_checkpoint(stage_dir / "checkpoint-499.pth", configs[variant])
            else:
                record = lp_stats(stage_dir / "log.txt")
                print("[{}] best LP acc: {:.6f}% @ epoch {}".format(variant, record["best_lp_acc"], record["best_epoch"]), flush=True)
            manifest["completed_stages"].append(dict(variant=variant, stage=stage, **record))
            write_json(manifest_path, manifest)
        write_summary(root)
        manifest["status"] = "complete"
        write_json(manifest_path, manifest)
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = str(exc)
        write_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print("ERROR: " + str(exc), file=sys.stderr, flush=True)
        sys.exit(1)
