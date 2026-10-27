"""Independent entry for paired text-to-skeleton relation distillation."""
import importlib
from pathlib import Path


MODEL_NAME = 'model.transformer_macdiff_text_relation.Transformer'
DEFAULT_CONFIG = 'config/ntu60_xsub_joint/pretrain_madiff_text_relation_decoder3.yaml'
RELATION_DEFAULTS = dict(relation_version='paired_global_fifo_text_to_skeleton_kl_v1',
    relation_bank_size=4096, lambda_relation=.1,
    relation_temperature_text=.1, relation_temperature_skeleton=.1)
RESUME_FIELDS = ('feeder', 'train_feeder_args', 'text_cache', 'mask_ratio', 'motion_stride',
    'motion_aware_tau', 'batch_size', 'accum_iter', 'epochs', 'warmup_epochs', 'min_lr_epochs',
    'lr', 'min_lr', 'weight_decay', 'seed', 'enable_amp')


def get_args_parser():
    from main_pretrain import get_args_parser as native_parser
    parser = native_parser()
    parser.set_defaults(config=DEFAULT_CONFIG, model=MODEL_NAME, enable_amp=False)
    parser.add_argument('--lambda_relation', type=float, default=None)
    parser.add_argument('--relation_bank_size', type=int, default=None)
    parser.add_argument('--relation_temperature_text', type=float, default=None)
    parser.add_argument('--relation_temperature_skeleton', type=float, default=None)
    return parser


def _argument(value, name, default=None):
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def _canonical(value):
    if isinstance(value, dict):
        return {name: _canonical(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    return value


def _validate_resume_definition(saved, args, model_name=MODEL_NAME):
    if _argument(saved, 'model') != model_name:
        raise ValueError('Resume requires this relation method, not old S->T/native/teacher/FM weights')
    if _canonical(_argument(saved, 'model_args', {})) != _canonical(args.model_args):
        raise ValueError('Relation resume definition mismatch: model_args')
    for name in RESUME_FIELDS:
        before, after = _argument(saved, name), getattr(args, name, None)
        if name == 'mask_ratio':
            before = before[0] if isinstance(before, (list, tuple)) and len(before) == 1 else before
            after = after[0] if isinstance(after, (list, tuple)) and len(after) == 1 else after
        if _canonical(before) != _canonical(after):
            raise ValueError('Relation resume protocol mismatch: ' + name)


def prepare_args(args, model_name=MODEL_NAME):
    if args.model != model_name or args.enable_ose or not args.text_cache or args.task != 'recognition':
        raise ValueError('Use this relation model with the sentence cache, recognition, and OSE disabled')
    options = dict(args.model_args)
    for name, default in RELATION_DEFAULTS.items():
        options.setdefault(name, default)
        value = getattr(args, name, None)
        if value is not None:
            options[name] = value
    for name, default in (('lambda_text_to_skeleton', 1.), ('lambda_skeleton_to_text', 0.)):
        options.setdefault(name, default)
        if getattr(args, name) is not None:
            options[name] = getattr(args, name)
    module = importlib.import_module(model_name.rsplit('.', 1)[0])
    module.validate_relation_options(**{name: options[name] for name in RELATION_DEFAULTS},
        lambda_text_to_skeleton=options['lambda_text_to_skeleton'],
        lambda_skeleton_to_text=options['lambda_skeleton_to_text'],
        one_person=options.get('one_person', True),
        text_target_mode=options.get('text_target_mode', 'sample_target_blend'),
        text_target_norm=options.get('text_target_norm', 'rms'))
    args.model_args = options
    if args.resume:
        import torch
        if args.resume.startswith(('https://', 'http://')):
            raise ValueError('Use a local full relation checkpoint for --resume')
        checkpoint = torch.load(args.resume, map_location='cpu')
        required = {'model', 'args', 'optimizer', 'scaler', 'epoch'}
        if required - set(checkpoint):
            raise ValueError('Resume needs a complete relation checkpoint')
        _validate_resume_definition(checkpoint['args'], args, model_name=model_name)
        del checkpoint
    elif args.output_dir:
        output = Path(args.output_dir)
        if (output / 'log.txt').exists() or any(output.glob('checkpoint-*.pth')):
            raise FileExistsError('Choose a fresh relation PT directory or explicitly resume')
    return args


def parse_args(argv=None, parser=None):
    import yaml
    parser = get_args_parser() if parser is None else parser
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
