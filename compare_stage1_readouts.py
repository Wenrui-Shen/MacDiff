"""Compare frozen MacDiff readouts using the actual linprobe2 forward."""
from pathlib import Path

from stage1_readout import build_parser, compare_cache, extract_cache, fingerprint


def load_encoder(model, path):
    import torch
    try:
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location='cpu')
    if not isinstance(checkpoint, dict):
        raise ValueError('Expected a native MacDiff Stage1 checkpoint dictionary')
    if any(key in checkpoint for key in ('training_engine', 'stage2_protocol', 'mask_protocol')):
        raise ValueError('Stage2 checkpoint is not a valid Stage1 source')
    source = checkpoint.get('model', checkpoint)
    source = {key.removeprefix('module.'): value for key, value in source.items()
              if torch.is_tensor(value)} if hasattr(str, 'removeprefix') else {
                  (key[7:] if key.startswith('module.') else key): value
                  for key, value in source.items() if torch.is_tensor(value)}
    selected = {}
    for name, target in model.state_dict().items():
        if name not in source or source[name].shape != target.shape:
            raise ValueError('Missing/shape-mismatched Stage1 encoder tensor: ' + name)
        selected[name] = source[name]
    model.load_state_dict(selected, strict=True)
    return {'encoder_tensors': len(selected), 'ignored_tensors': len(source) - len(selected)}


def prepare_extraction(args):
    import random
    import numpy as np
    import torch
    import yaml
    from feeder.feeder_ntu import Feeder
    from model.transformer_downstream import Transformer

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    config = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    model_args = dict(config['model_args'])
    if (model_args.get('protocol') != 'linprobe2'
            or model_args.get('num_frames') != 120
            or model_args.get('num_joints') != 25
            or model_args.get('dim_feat') != 256
            or model_args.get('t_patch_size') != 4
            or model_args.get('patch_size') != 1):
        raise ValueError('Expected NTU60 linprobe2: 120 frames, 25 joints, 256 channels, 4x1 patches')
    model = Transformer(**model_args)
    # Retain the original LP head's exact person/time pooling and joint ordering.
    # eval() disables its dropout. No supervised fc or LP running BN is loaded.
    model.head.fc = torch.nn.Identity()
    transfer = load_encoder(model, args.checkpoint)
    path = args.data_path or config['val_feeder_args']['data_path']
    datasets = [Feeder(data_path=path, split=split, p_interval=[0.95],
                       window_size=120, random_rot=False, source_rot=False,
                       normalization=False, bone=False, vel=False, use_mmap=True)
                for split in ('train', 'test')]
    if not np.array_equal(np.unique(datasets[0].label), np.arange(60)):
        raise ValueError('This experiment is fixed to NTU60 XSub, classes 0..59')
    provenance = {
        'config': fingerprint(args.config), 'data_files': [fingerprint(path)],
        'model_args': model_args, 'strict_transfer': transfer,
        'input': 'native valid_crop_resize, center 95%, resize 120, full 750 tokens/person, both persons',
        'features': {'J': 6400}, 'pooling': 'actual ActionHeadLinprobe2, fc=Identity',
        'classifier': 'raw features BEFORE LP BatchNorm and Linear; no projector',
        'train_preprocessing': 'uses deterministic LP evaluation crop, not stochastic LP classifier training views',
    }
    return model, datasets, provenance


def forward_features(model, samples):
    return {'J': model(samples)}


def main():
    parser, extract = build_parser(
        'MacDiff', './output_dir/ntu60_xsub_macdiff/checkpoint-399.pth', 4)
    extract.add_argument('--config', default='./config/ntu60_xsub_joint/linprobe_madiff.yaml')
    extract.add_argument('--data-path', default='')
    args = parser.parse_args()
    if args.command == 'compare':
        compare_cache(args, 'MacDiff')
    else:
        if Path(args.cache_dir).exists():
            raise FileExistsError('Use a NEW cache directory; existing files are never overwritten')
        model, datasets, provenance = prepare_extraction(args)
        extract_cache(args, 'MacDiff', model, datasets, forward_features, provenance)


if __name__ == '__main__':
    main()
