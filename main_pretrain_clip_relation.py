"""Frozen CLIP global-relation teacher; no text reconstruction/remap branch."""
import importlib
from pathlib import Path

import main_pretrain_text_relation as relation_entry


MODEL_NAME = 'model.transformer_macdiff_clip_relation.Transformer'
DEFAULT_CONFIG = 'config/ntu60_xsub_joint/pretrain_madiff_clip_relation_decoder3.yaml'
DEFAULTS = dict(clip_relation_version='fixed_clip_full_ema_gap_shared_queue_kl_v1',
    relation_bank_size=65536, lambda_relation=.1, relation_temperature_text=.1,
    relation_temperature_skeleton=.1, ema_momentum=.999, lambda_contrastive=1.e-4,
    contrast_temperature=.07, relation_feature_mode='gap', one_person=True,
    text_target_mode='fixed_clip', text_target_norm='rms',
    lambda_text_to_skeleton=0., lambda_skeleton_to_text=0.)


def get_args_parser():
    parser = relation_entry.get_args_parser()
    parser.set_defaults(config=DEFAULT_CONFIG, model=MODEL_NAME)
    parser.add_argument('--ema_momentum', type=float, default=None)
    parser.add_argument('--lambda_contrastive', type=float, default=None)
    parser.add_argument('--contrast_temperature', type=float, default=None)
    return parser


def prepare_args(args):
    if args.model != MODEL_NAME or args.enable_ose or not args.text_cache or args.task != 'recognition':
        raise ValueError('Use the fixed-CLIP relation model/cache for recognition, without OSE')
    options = dict(args.model_args)
    for name, default in DEFAULTS.items():
        options.setdefault(name, default)
        value = getattr(args, name, None)
        if value is not None:
            options[name] = value
    module = importlib.import_module('model.transformer_macdiff_clip_relation')
    module.validate_clip_relation_options(**{name: options[name] for name in DEFAULTS})
    args.model_args = options
    if args.resume:
        import torch
        if args.resume.startswith(('https://', 'http://')):
            raise ValueError('Use a local full fixed-CLIP relation checkpoint for --resume')
        checkpoint = torch.load(args.resume, map_location='cpu')
        if {'model', 'args', 'optimizer', 'scaler', 'epoch'} - set(checkpoint):
            raise ValueError('Resume needs a complete fixed-CLIP relation checkpoint')
        relation_entry._validate_resume_definition(checkpoint['args'], args, model_name=MODEL_NAME)
        del checkpoint
    elif args.output_dir:
        output = Path(args.output_dir)
        if (output / 'log.txt').exists() or any(output.glob('checkpoint-*.pth')):
            raise FileExistsError('Choose a fresh fixed-CLIP relation PT directory or explicitly resume')
    return args


def parse_args(argv=None):
    return relation_entry.parse_args(argv, parser=get_args_parser())


def main(args):
    from main_pretrain import main as native_main
    args = prepare_args(args)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    native_main(args)


if __name__ == '__main__':
    main(parse_args())
