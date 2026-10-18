#!/usr/bin/env python3
"""Generate NTU60 XView train captions, reusing completed NTU60 XSub train text."""
import argparse
from pathlib import Path

from dataset_caption_pipeline import (PROJECT_ROOT, add_common_arguments, cli_main, pipeline_identity,
                                      run_dataset, sha256_file, validate_common_arguments)
from run_caption_ntu120 import NTU60_SUBJECTS, build_plan, source_identity, train_names


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--data_path", type=Path, default=Path("../data/MAMP/ntu/NTU60_XView.npz"))
    parser.add_argument("--ntu60_data_path", type=Path, default=Path("../data/MAMP/ntu/NTU60_XSub.npz"))
    parser.add_argument("--ntu60_captions", type=Path,
                        default=Path("vlm_pilot/ntu60_xsub_global_local_v2/captions.json"))
    parser.add_argument("--ntu60_statistics_dir", type=Path, default=PROJECT_ROOT / "data/ntu/statistics")
    args = parser.parse_args(argv)
    validate_common_arguments(parser, args)
    return args


def main(argv=None):
    args = parse_args(argv)
    names60 = train_names(args.ntu60_statistics_dir, NTU60_SUBJECTS)
    # seq_transformation.py concatenates all camera-2 rows, then camera-3 rows.
    names_view = train_names(args.ntu60_statistics_dir, (2, 3), group="camera")
    identity = pipeline_identity(args, "ntu60_xview", args.data_path)
    identity["ntu60"] = source_identity(args, names60, names_view, "ntu60_xview")
    identity["entrypoint_sha256"] = sha256_file(Path(__file__))
    identity["reuse_implementation_sha256"] = sha256_file(Path(__file__).with_name("run_caption_ntu120.py"))
    return run_dataset(args, "ntu60_xview", args.data_path, identity,
                       lambda: build_plan(args, names60, names_view, target_label="NTU60 XView"))


if __name__ == "__main__":
    raise SystemExit(cli_main(main))
