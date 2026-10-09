"""Read actual checkpoint args and block indices on the training server."""
import argparse
import hashlib
import json
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoints', nargs='+')
    parser.add_argument('--sha256', action='store_true', help='Also stream the checkpoint SHA256')
    args = parser.parse_args()
    import torch
    for name in args.checkpoints:
        path = Path(name)
        checkpoint = torch.load(str(path), map_location='cpu')
        saved_args = checkpoint.get('args', {})
        saved_args = saved_args if isinstance(saved_args, dict) else vars(saved_args)
        state = checkpoint['model']
        keys = [key.removeprefix('module.') if hasattr(key, 'removeprefix')
                else key[7:] if key.startswith('module.') else key for key in state]
        depths = {}
        for prefix in ['blocks', 'decoder_blocks', 'text_skeleton_decoder.decoder_blocks',
                       'text_noise_decoder.blocks']:
            indices = sorted({int(match.group(1)) for key in keys
                              for match in [re.match(re.escape(prefix)+r'\.(\d+)\.', key)] if match})
            depths[prefix] = {'indices': indices, 'count': len(indices)}
        output = {'checkpoint': str(path), 'epoch': checkpoint.get('epoch'),
                  'bytes': path.stat().st_size, 'torch_version': torch.__version__,
                  'state_dict_depths': depths,
                  'saved_args': {key: saved_args.get(key) for key in
                                 ['model', 'model_args', 'config', 'epochs', 'batch_size',
                                  'accum_iter', 'world_size', 'warmup_epochs', 'min_lr_epochs',
                                  'lr', 'min_lr', 'seed', 'train_feeder_args',
                                  'lambda_text_to_skeleton', 'lambda_skeleton_to_text',
                                  'finetune', 'output_dir']}}
        if args.sha256:
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024*1024), b''):
                    digest.update(chunk)
            output['sha256'] = digest.hexdigest()
        print(json.dumps(output, ensure_ascii=False, indent=2, default=str))


if __name__ == '__main__':
    main()