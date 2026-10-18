"""Independent native-decoder distillation entry point; existing runners stay unchanged."""
import importlib
from pathlib import Path


MODEL_NAME = 'model.transformer_macdiff_decoder_distill.Transformer'
DEFAULT_CONFIG = 'config/ntu60_xsub_joint/pretrain_madiff_decoder_distill.yaml'


def get_args_parser():
    from main_pretrain import get_args_parser as native_parser
    parser = native_parser()
    parser.set_defaults(config=DEFAULT_CONFIG, model=MODEL_NAME)
    parser.add_argument('--decoder_checkpoint', default=None, type=str,
                        help='Native PT checkpoint supplying only the frozen decoder body')
    return parser


def _validate_resume_definition(saved_args, model_args):
    from model.transformer_macdiff_decoder_distill import _argument, _canonical
    if _argument(saved_args, 'model') != MODEL_NAME:
        raise ValueError('--resume must be this method\'s checkpoint; '
                         'use --decoder_checkpoint for native teacher initialization')
    saved_options = _argument(saved_args, 'model_args', {})
    defaults = dict(distill_t_min=300, distill_t_max=700,
                    diff_prediction='noise2joint', lambda_loss_uni=0.,
                    uncond_ratio=0., layer_mask_ratio=0., one_person=True,
                    self_shift=False, loss_reweight=None)
    for name, default in defaults.items():
        if _canonical(saved_options.get(name, default)) != _canonical(model_args.get(name, default)):
            raise ValueError('Distillation resume definition mismatch: ' + name)


def prepare_args(args):
    # Explicit import registers the submodule on the model package for the
    # existing import_class(). No modification to model/__init__.py is needed.
    module = importlib.import_module('model.transformer_macdiff_decoder_distill')
    if args.model != MODEL_NAME or args.enable_ose or args.text_cache:
        raise ValueError('This entry point requires the decoder-distillation model, '
                         'with OSE and text caches disabled')
    options = dict(args.model_args)
    if args.resume:
        if args.decoder_checkpoint:
            raise ValueError('Choose --resume or --decoder_checkpoint, not both')
        checkpoint = module.load_checkpoint_file(args.resume)
        missing = {'model', 'args', 'optimizer', 'scaler', 'epoch'} - set(checkpoint)
        if missing:
            raise ValueError('Full distillation resume checkpoint is missing: %s' % sorted(missing))
        saved_args = checkpoint.get('args')
        _validate_resume_definition(saved_args, options)
        # Saved model includes the frozen body. Its construction no longer
        # depends on retaining the original teacher file; main() then restores
        # the complete student/head/optimizer/scaler through the native resume.
        options['decoder_checkpoint'] = args.resume
        args.decoder_source_checkpoint = module._argument(
            saved_args, 'decoder_source_checkpoint')
        del checkpoint
    else:
        path = args.decoder_checkpoint or options.get('decoder_checkpoint')
        if not path:
            raise ValueError('Pass --decoder_checkpoint /path/to/native/PT/checkpoint-399.pth')
        if not Path(path).expanduser().is_file():
            raise FileNotFoundError('Decoder checkpoint not found: %s' % path)
        options['decoder_checkpoint'] = path
        args.decoder_source_checkpoint = str(path)
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
