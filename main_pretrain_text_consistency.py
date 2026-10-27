"""Independent entry for native + T->S bidirectional reconstruction consistency."""
import importlib
from pathlib import Path


MODEL_NAME = 'model.transformer_macdiff_text_consistency.Transformer'
DEFAULT_CONFIG = 'config/ntu60_xsub_joint/pretrain_madiff_text_consistency.yaml'


def get_args_parser():
    from main_pretrain import get_args_parser as native_parser
    parser = native_parser()
    parser.set_defaults(config=DEFAULT_CONFIG, model=MODEL_NAME)
    parser.add_argument('--lambda_consistency', default=None, type=float,
                        help='Override noise-scale-normalized bidirectional x0 MSE weight (default: 1)')
    return parser


def _argument(value, name, default=None):
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def _canonical(value):
    if isinstance(value, dict):
        return {key: _canonical(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    return value


def _validate_resume_definition(saved_args, model_args):
    if _argument(saved_args, 'model') != MODEL_NAME:
        raise ValueError('--resume must be this consistency method, not old S->T/native training')
    saved_options = _argument(saved_args, 'model_args', {})
    if _canonical(saved_options) != _canonical(model_args):
        raise ValueError('Consistency resume definition mismatch: model_args '
                         '(including lambda_consistency and decoder depth)')


def prepare_args(args):
    if args.model != MODEL_NAME or args.enable_ose or not args.text_cache:
        raise ValueError('Use the text-consistency model with a sentence cache and OSE disabled')
    options = dict(args.model_args)
    if args.lambda_consistency is not None:
        options['lambda_consistency'] = args.lambda_consistency
    if args.lambda_text_to_skeleton is not None:
        options['lambda_text_to_skeleton'] = args.lambda_text_to_skeleton
    if args.lambda_skeleton_to_text is not None:
        options['lambda_skeleton_to_text'] = args.lambda_skeleton_to_text
    if options.get('lambda_consistency') is None:
        options['lambda_consistency'] = 1.
    module = importlib.import_module('model.transformer_macdiff_text_consistency')
    # Saved in model_args: raw-MSE checkpoints must not silently resume under
    # a different objective, including when the user omits the YAML field.
    options.setdefault('consistency_normalization', module.CONSISTENCY_NORMALIZATION)
    module.validate_consistency_definition(
        options['lambda_consistency'], options.get('lambda_text_to_skeleton', 1.),
        options.get('one_person', True), options.get('share_skeleton_decoder', False),
        options.get('text_target_mode', 'reconstruction_consistency'),
        options.get('text_target_norm', 'rms'), options.get('lambda_skeleton_to_text', 0.),
        options.get('lambda_text_uniformity', 0.), options['consistency_normalization'])
    if args.resume:
        import torch
        if args.resume.startswith(('https://', 'http://')):
            raise ValueError('Use a local full consistency checkpoint for --resume')
        checkpoint = torch.load(args.resume, map_location='cpu')
        required = {'model', 'args', 'optimizer', 'scaler', 'epoch'}
        if required - set(checkpoint):
            raise ValueError('Resume needs a complete consistency training checkpoint')
        _validate_resume_definition(checkpoint['args'], options)
        del checkpoint
    args.model_args = options
    return args


def parse_args(argv=None):
    import yaml
    parser = get_args_parser()
    preliminary = parser.parse_args(argv)
    if preliminary.config is not None:
        with open(preliminary.config, 'r', encoding='utf-8') as stream:
            defaults = yaml.safe_load(stream)
        if not isinstance(defaults, dict):
            raise ValueError('Config must contain a mapping')
        unknown = set(defaults) - set(vars(preliminary))
        if unknown:
            raise ValueError('Unknown config arguments: %s' % sorted(unknown))
        parser.set_defaults(**defaults)
    return prepare_args(parser.parse_args(argv))


if __name__ == '__main__':
    from main_pretrain import main
    arguments = parse_args()
    if arguments.output_dir:
        Path(arguments.output_dir).mkdir(parents=True, exist_ok=True)
    main(arguments)
