"""Native MacDiff + full-context EMA teacher, InfoNCE and positional layer AA."""
import importlib
from pathlib import Path


MODEL_NAME = 'model.transformer_macdiff_skeleton_teacher.Transformer'
DEFAULT_CONFIG = 'config/ntu60_xsub_joint/pretrain_madiff_skeleton_teacher.yaml'
TEACHER_DEFAULTS = dict(ema_momentum=.999, contrast_temperature=.07,
                        contrast_queue_size=65536, alignment_teacher_layers=5,
                        lambda_contrastive=1.e-4, lambda_alignment=1.,
                        teacher_method_version='raw_full_position_student_adapter_aa1_v3')


def get_args_parser():
    from main_pretrain import get_args_parser as native_parser
    parser = native_parser()
    parser.set_defaults(config=DEFAULT_CONFIG, model=MODEL_NAME, enable_amp=False)
    for name in ('lambda_contrastive', 'lambda_alignment', 'ema_momentum', 'contrast_temperature'):
        parser.add_argument('--' + name, type=float, default=None)
    for name in ('contrast_queue_size', 'alignment_teacher_layers'):
        parser.add_argument('--' + name, type=int, default=None)
    return parser


def _argument(value, name, default=None):
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def _canonical(value):
    if isinstance(value, dict):
        return {key: _canonical(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    return value


def _validate_resume_definition(saved_args, model_args):
    if _argument(saved_args, 'model') != MODEL_NAME:
        raise ValueError('--resume requires this skeleton-teacher method, not native/text/OSE weights')
    saved = _argument(saved_args, 'model_args', {})
    if _canonical(saved) != _canonical(model_args):
        raise ValueError('Skeleton teacher resume definition mismatch: model_args')


def prepare_args(args):
    if args.model != MODEL_NAME or args.enable_ose or args.text_cache:
        raise ValueError('Use the skeleton-teacher model with OSE and text caches disabled')
    options = dict(args.model_args)
    for name, default in TEACHER_DEFAULTS.items():
        options.setdefault(name, default)
        override = getattr(args, name, None)
        if override is not None:
            options[name] = override
    if options['teacher_method_version'] != TEACHER_DEFAULTS['teacher_method_version']:
        raise ValueError('Unsupported teacher_method_version; use the current skeleton-teacher config')
    module = importlib.import_module('model.transformer_macdiff_skeleton_teacher')
    module.validate_teacher_options(
        options['ema_momentum'], options['contrast_temperature'], options['contrast_queue_size'],
        options['alignment_teacher_layers'], options.get('depth', 5),
        options['lambda_contrastive'], options['lambda_alignment'])
    if args.resume:
        import torch
        checkpoint = torch.load(args.resume, map_location='cpu')
        missing = {'model', 'args', 'optimizer', 'scaler', 'epoch'} - set(checkpoint)
        if missing:
            raise ValueError('Full skeleton teacher resume checkpoint is missing: %s' % sorted(missing))
        _validate_resume_definition(checkpoint['args'], options)
        del checkpoint
    args.model_args = options
    args.skeleton_teacher_numerics_version = (
        'native_decoder_loss_fp32_v1' if getattr(args, 'enable_amp', False)
        else 'full_fp32_no_amp_v1')
    return args


def parse_args(argv=None):
    import yaml
    parser = get_args_parser()
    preliminary = parser.parse_args(argv)
    with open(preliminary.config, 'r', encoding='utf-8') as stream:
        defaults = yaml.safe_load(stream)
    if not isinstance(defaults, dict):
        raise ValueError('Config must contain a mapping')
    unknown = set(defaults) - {action.dest for action in parser._actions}
    if unknown:
        raise ValueError('Unknown config arguments: %s' % sorted(unknown))
    parser.set_defaults(**defaults)
    return parser.parse_args(argv)


def main(args):
    from main_pretrain import main as native_main
    args = prepare_args(args)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    native_main(args)


if __name__ == '__main__':
    main(parse_args())
