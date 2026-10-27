"""Independent skeleton OT conditional FM entry with native PT/LP weights."""
import importlib
from pathlib import Path


MODEL_NAME = 'model.transformer_macdiff_flow_matching.Transformer'
DEFAULT_CONFIG = 'config/ntu60_xsub_joint/pretrain_madiff_flow_matching.yaml'
FLOW_DEFAULTS = dict(flow_time_scale=999., flow_sigma_min=1.e-4,
                     flow_matching_version='conditional_ot_noise_to_data_v2',
                     flow_time_sampling='uniform')
RESUME_FIELDS = (
    'feeder', 'train_feeder_args', 'mask_ratio', 'motion_stride', 'motion_aware_tau',
    'batch_size', 'accum_iter', 'epochs', 'warmup_epochs', 'min_lr_epochs',
    'lr', 'min_lr', 'weight_decay', 'seed', 'enable_amp',
)


def get_args_parser():
    from main_pretrain import get_args_parser as native_parser
    parser = native_parser()
    parser.set_defaults(config=DEFAULT_CONFIG, model=MODEL_NAME, enable_amp=False)
    parser.add_argument('--flow_time_scale', default=None, type=float,
                        help='Override continuous FM time embedding scale (default: 999)')
    parser.add_argument('--flow_sigma_min', default=None, type=float,
                        help='Override OT conditional terminal noise std (default: 0.0001)')
    parser.add_argument('--flow_time_sampling', default=None,
                        choices=('uniform', 'inverse_cosine_snr_v1'),
                        help='OT time density; inverse_cosine_snr_v1 matches native 1000-step SNRs')
    return parser


def _argument(value, name, default=None):
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def _canonical(value):
    if isinstance(value, dict):
        return {key: _canonical(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    return value


def _protocol_value(name, value):
    if name == 'mask_ratio' and isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    return _canonical(value)


def _validate_resume_definition(saved_args, args):
    if _argument(saved_args, 'model') != MODEL_NAME:
        raise ValueError('FM resume requires an FM checkpoint, not native diffusion/text/teacher weights')
    saved_options = dict(_argument(saved_args, 'model_args', {}))
    # Old v2 checkpoints predate this switch and always used uniform sampling.
    saved_options.setdefault('flow_time_sampling', 'uniform')
    if _canonical(saved_options) != _canonical(args.model_args):
        raise ValueError('FM resume definition mismatch: model_args')
    for name in RESUME_FIELDS:
        if _protocol_value(name, _argument(saved_args, name)) != _protocol_value(
                name, getattr(args, name, None)):
            raise ValueError('FM resume protocol mismatch: ' + name)


def prepare_args(args):
    if args.model != MODEL_NAME or args.enable_ose or args.text_cache:
        raise ValueError('Use the native FM model with text caches and OSE disabled')
    if args.task != 'recognition':
        raise ValueError('The native FM baseline requires task=recognition')
    if args.lambda_text_to_skeleton is not None or args.lambda_skeleton_to_text is not None:
        raise ValueError('Text loss switches do not apply to native FM')
    options = dict(args.model_args)
    for name, value in FLOW_DEFAULTS.items():
        options.setdefault(name, value)
    for name in ('flow_time_scale', 'flow_sigma_min', 'flow_time_sampling'):
        value = getattr(args, name)
        if value is not None:
            options[name] = value
    forbidden = {'diff_prediction', 'diff_steps', 'diff_noise_schedule', 'loss_reweight'} & set(options)
    if forbidden:
        raise ValueError('Remove diffusion-only model arguments for FM: %s' % sorted(forbidden))
    module = importlib.import_module('model.transformer_macdiff_flow_matching')
    module.validate_flow_options(options['flow_time_scale'], options['flow_matching_version'],
                                 options.get('one_person', True), options.get('lambda_loss_uni', 0.),
                                 flow_sigma_min=options['flow_sigma_min'],
                                 flow_time_sampling=options['flow_time_sampling'])
    args.model_args = options
    if args.resume:
        import torch
        if args.resume.startswith(('https://', 'http://')):
            raise ValueError('Use a local full FM checkpoint for --resume')
        checkpoint = torch.load(args.resume, map_location='cpu')
        missing = {'model', 'args', 'optimizer', 'scaler', 'epoch'} - set(checkpoint)
        if missing:
            raise ValueError('Full FM resume checkpoint is missing: %s' % sorted(missing))
        _validate_resume_definition(checkpoint['args'], args)
        del checkpoint
    elif args.output_dir:
        output = Path(args.output_dir)
        if (output / 'log.txt').exists() or any(output.glob('checkpoint-*.pth')):
            raise FileExistsError('Choose a fresh PT directory or explicitly resume this same method')
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
