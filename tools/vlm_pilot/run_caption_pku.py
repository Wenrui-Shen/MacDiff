#!/usr/bin/env python3
"""Generate PKU-MMD v1/v2 XSub train captions into separate dataset directories."""
import argparse
from pathlib import Path

from dataset_caption_pipeline import (add_common_arguments, cli_main, load_train, pipeline_identity,
                                      run_dataset, sha256_file, validate_common_arguments)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--versions", choices=("v1", "v2"), nargs="+", default=["v1", "v2"])
    parser.add_argument("--pkuv1_data_path", type=Path, default=Path("../data/MAMP/pku_v1/PKUv1_XSub.npz"))
    parser.add_argument("--pkuv2_data_path", type=Path, default=Path("../data/MAMP/pku_v2/PKUv2_XSub.npz"))
    args = parser.parse_args(argv)
    validate_common_arguments(parser, args)
    if len(set(args.versions)) != len(args.versions):
        parser.error("--versions must not contain duplicates")
    return args


def build_plan(path):
    data = load_train(path)
    return {"sample_count": len(data), "generate_indices": list(range(len(data))), "reuse": []}, data


def main(argv=None):
    args = parse_args(argv)
    # Validate both paths before starting expensive work for either phase.
    paths = {version: getattr(args, "pku%s_data_path" % version) for version in args.versions}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    result = 0
    for version, path in paths.items():
        dataset = "pku_%s_xsub" % version
        identity = pipeline_identity(args, dataset, path)
        identity["entrypoint_sha256"] = sha256_file(Path(__file__))
        code = run_dataset(args, dataset, path, identity, lambda: build_plan(path))
        if code == 130:
            return code
        result = result or code
    return result


if __name__ == "__main__":
    raise SystemExit(cli_main(main))
