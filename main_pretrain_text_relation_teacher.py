"""Full-key default/optional coarse EMA view, with one paired InfoNCE/KL queue."""
import importlib
from pathlib import Path

import main_pretrain_text_relation as relation_entry


MODEL_NAME = 'model.transformer_macdiff_text_relation_teacher.Transformer'
DEFAULT_CONFIG = 'config/ntu60_xsub_joint/pretrain_madiff_text_relation_teacher_decoder3.yaml'
TEACHER_DEFAULTS = dict(relation_teacher_version='full_ema_gap_shared_queue_global_kl_v2',
    ema_momentum=.999, lambda_contrastive=1.e-4, contrast_temperature=.07, relation_feature_mode='gap')


def get_args_parser():
    parser = relation_entry.get_args_parser()
    parser.set_defaults(config=DEFAULT_CONFIG, model=MODEL_NAME)
    parser.add_argument('--ema_momentum', type=float, default=None)
    parser.add_argument('--lambda_contrastive', type=float, default=None)
    parser.add_argument('--contrast_temperature', type=float, default=None)
    parser.add_argument('--teacher_view', choices=('full', 'coarse_mask'), default=None)
    parser.add_argument('--teacher_temporal_group', type=int, default=None)
    parser.add_argument('--teacher_mask_ratio', type=float, default=None)
    return parser


def prepare_args(args):
    options = dict(args.model_args)
    options.setdefault('relation_bank_size', 65536)
    for name, default in TEACHER_DEFAULTS.items():
        options.setdefault(name, default)
        value = getattr(args, name, None)
        if value is not None:
            options[name] = value
    # Do not add redundant fields to old full-view model_args: its saved definition
    # remains compatible. The new coarse-view YAML explicitly binds all view fields.
    for name in ('teacher_view', 'teacher_temporal_group', 'teacher_mask_ratio'):
        value = getattr(args, name, None)
        if value is not None:
            options[name] = value
    module = importlib.import_module('model.transformer_macdiff_text_relation_teacher')
    module.validate_relation_teacher_options(**{name: options[name] for name in TEACHER_DEFAULTS})
    module.validate_teacher_view_options(options['relation_teacher_version'],
        options.get('teacher_view', 'full'), options.get('teacher_temporal_group', 1),
        options.get('teacher_mask_ratio', 0.))
    args.model_args = options
    return relation_entry.prepare_args(args, model_name=MODEL_NAME)


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
