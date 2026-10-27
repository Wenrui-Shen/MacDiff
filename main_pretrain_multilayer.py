"""Last-three-layer PT with a shared native decoder; LP keeps the final layer."""
import importlib
from pathlib import Path


MODEL_NAME = 'model.transformer_macdiff_multilayer.Transformer'
DEFAULT_CONFIG = 'config/ntu60_xsub_joint/pretrain_madiff_multilayer.yaml'
MULTILAYER_VERSION = 'last3_shared_norm_decoder_final_layer_lp_v3'
RESUME_FIELDS = (
    'feeder', 'train_feeder_args', 'mask_ratio', 'motion_stride', 'motion_aware_tau',
    'batch_size', 'accum_iter', 'epochs', 'warmup_epochs', 'min_lr_epochs',
    'lr', 'min_lr', 'weight_decay', 'seed', 'enable_amp',
)


def get_args_parser():
    from main_pretrain import get_args_parser as native_parser
    parser = native_parser()
    parser.set_defaults(config=DEFAULT_CONFIG, model=MODEL_NAME, enable_amp=False)
    parser.add_argument('--lambda_aux_diffusion', default=None, type=float,
                        help='Weight of the MEAN of the two auxiliary diffusion losses (default: .1)')
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
        raise ValueError('Resume requires this full multilayer checkpoint')
    if _canonical(_argument(saved_args, 'model_args', {})) != _canonical(args.model_args):
        raise ValueError('Multilayer resume definition mismatch: model_args')
    for name in RESUME_FIELDS:
        if _protocol_value(name, _argument(saved_args, name)) != _protocol_value(
                name, getattr(args, name, None)):
            raise ValueError('Multilayer resume protocol mismatch: ' + name)


def prepare_args(args):
    if args.model != MODEL_NAME or args.enable_ose or args.text_cache or args.task != 'recognition':
        raise ValueError('Use the pure-skeleton multilayer recognition model')
    if args.lambda_text_to_skeleton is not None or args.lambda_skeleton_to_text is not None:
        raise ValueError('Text loss switches do not apply to multilayer diffusion')
    options = dict(args.model_args)
    options.setdefault('depth', 8)
    options.setdefault('decoder_depth', 3)
    options.setdefault('lambda_aux_diffusion', .1)
    options.setdefault('multilayer_version', MULTILAYER_VERSION)
    if args.lambda_aux_diffusion is not None:
        options['lambda_aux_diffusion'] = args.lambda_aux_diffusion
    module = importlib.import_module('model.transformer_macdiff_multilayer')
    module.validate_multilayer_options(
        options['depth'], options['lambda_aux_diffusion'],
        one_person=options.get('one_person', True),
        diff_prediction=options.get('diff_prediction', 'noise'),
        lambda_loss_uni=options.get('lambda_loss_uni', 0.),
        layer_mask_ratio=options.get('layer_mask_ratio', 0.),
        uncond_ratio=options.get('uncond_ratio', .1),
        multilayer_version=options['multilayer_version'])
    args.model_args = options
    if args.resume:
        import torch
        if args.resume.startswith(('https://', 'http://')):
            raise ValueError('Use a local full multilayer checkpoint for --resume')
        checkpoint = torch.load(args.resume, map_location='cpu')
        missing = {'model', 'args', 'optimizer', 'scaler', 'epoch'} - set(checkpoint)
        if missing:
            raise ValueError('Full multilayer resume checkpoint is missing: %s' % sorted(missing))
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
