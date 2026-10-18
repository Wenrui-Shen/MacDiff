"""Reproducible static comparison of the working tree to official MacDiff.
No training modules are imported and no runtime numerical parity is asserted.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
BASE = '692888f1e2a4227216511aac6733dff31b587213'
OUT = Path(__file__).resolve().parent


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def norm(raw):
    return raw.replace(b'\r\n', b'\n')


def upstream(path):
    return norm(git('show', BASE + ':' + path))


def current(path):
    return norm((ROOT / path).read_bytes())


def tree(path, original=False):
    return ast.parse((upstream(path) if original else current(path)).decode('utf-8-sig'))


def definitions(module, prefix=''):
    result = {}
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = prefix + node.name
            result[name] = node
            if isinstance(node, ast.ClassDef):
                result.update(definitions(node, name + '.'))
    return result


def dump(node):
    return ast.dump(node, include_attributes=False)


def equal(a, b):
    return dump(a) == dump(b)


def strip_docstring(node):
    node = copy.deepcopy(node)
    if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
        node.body = node.body[1:]
    return node


paths = git('ls-tree', '-r', '--name-only', BASE).decode().splitlines()
inventory = []
for path in paths:
    original = upstream(path)
    local_path = ROOT / path
    local = current(path) if local_path.exists() else None
    inventory.append({
        'path': path,
        'status': 'missing' if local is None else ('identical_lf' if original == local else 'changed'),
        'upstream_sha256_lf': hashlib.sha256(original).hexdigest(),
        'current_sha256_lf': None if local is None else hashlib.sha256(local).hexdigest(),
    })

model_path = 'model/transformer_macdiff.py'
old = definitions(tree(model_path, True))
new = definitions(tree(model_path))
checks = []


def check(name, left, right, note):
    passed = equal(left, right)
    checks.append({'name': name, 'passed': passed, 'kind': 'AST comparison after stated normalization', 'note': note})
    return passed


for name in ('MLP', 'Attention', 'Block', 'FeatureModulation', 'DecoderBlock', 'SkeleEmbed',
             'Transformer._init_weights', 'Transformer.random_masking',
             'Transformer.motion_aware_random_masking', 'Transformer.patchify',
             'Transformer.forward_loss', 'Transformer.update_diffusion_sampler'):
    check(name, old[name], new[name], 'Exact AST equality; comments and source locations are ignored.')

init = copy.deepcopy(new['Transformer.__init__'])
removed_fields = []
kept = []
for node in init.body:
    if (isinstance(node, ast.Assign) and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Attribute)
            and isinstance(node.targets[0].value, ast.Name)
            and node.targets[0].value.id == 'self'
            and isinstance(node.value, ast.Constant) and node.value.value is None
            and node.targets[0].attr in {
                'ose_memory', 'ose_momentum_encoder', 'ose_momentum', 'ose_start_epoch',
                'ose_exemplar_checkpoint', 'lambda_ose', 'self_prob_start',
                'self_prob_end', 'peer_prob_start', 'peer_prob_end'}):
        removed_fields.append(node.targets[0].attr)
    else:
        kept.append(node)
init.body = kept
check('Transformer.__init__ excluding inactive OSE None fields', old['Transformer.__init__'], init,
      'Removed only assignments to None: ' + ', '.join(removed_fields))

encoder_old = copy.deepcopy(old['Transformer.forward_encoder'])
encoder_new = copy.deepcopy(new['Transformer.forward_encoder'])
encoder_new.args.kwonlyargs = [arg for arg in encoder_new.args.kwonlyargs if arg.arg != 'checkpoint_blocks']
encoder_new.args.kw_defaults = encoder_new.args.kw_defaults[:-1]
for node in encoder_old.body:
    if isinstance(node, ast.For) and isinstance(node.target, ast.Tuple):
        node.target = ast.Name(id='blk', ctx=ast.Store())
        node.iter = ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='blocks', ctx=ast.Load())
for node in encoder_new.body:
    if isinstance(node, ast.For):
        assert len(node.body) == 1 and isinstance(node.body[0], ast.If)
        node.body = node.body[0].orelse
check('Transformer.forward_encoder with checkpoint_blocks=False', encoder_old, encoder_new,
      'Select default non-checkpoint branch and discard unused enumerate index.')

decoder_old = copy.deepcopy(old['Transformer.forward_decoder'])
decoder_new = copy.deepcopy(new['Transformer.forward_decoder'])
decoder_old.args = copy.deepcopy(decoder_new.args)
decoder_old.body = [node for node in decoder_old.body if not (
    isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
    and (node.targets[0].id == 'L' or (node.targets[0].id == 'z' and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute) and node.value.func.attr in ('cat', 'gather'))))]
check('Transformer.forward_decoder after moving condition assembly', decoder_old, decoder_new,
      'Condition cat/gather moved into build_global_local_condition; remaining decoder AST is exact.')

feeder_old = definitions(tree('feeder/feeder_ntu.py', True))
feeder_new = definitions(tree('feeder/feeder_ntu.py'))
for name in ('Feeder.__init__', 'Feeder.load_data', 'Feeder.__len__', 'Feeder.__iter__', 'Feeder.add_source_aug'):
    check(name, feeder_old[name], feeder_new[name], 'Exact AST equality.')
inline = copy.deepcopy(feeder_new['Feeder.__getitem__'])
helper = feeder_new['Feeder._load_processed']
assert isinstance(helper.body[-1], ast.Return)
inline.body = copy.deepcopy(helper.body[:-1] + inline.body[1:])
check('Feeder.__getitem__ after inlining _load_processed', feeder_old['Feeder.__getitem__'], inline,
      'Inline crop/flip/rotation helper, leaving synthetic noise and return order intact.')

rename = {'N': 'batch_size', 'C': 'channels', 'T': 'frames', 'V': 'joints', 'M': 'people',
          'mu': 'center', 'i': 'channel', 'loss_uni': 'loss_uniformity', 'v': 'velocity'}


class NativeForwardNormalizer(ast.NodeTransformer):
    def visit_Name(self, node):
        node.id = rename.get(node.id, node.id)
        return node

    def visit_If(self, node):
        node = self.generic_visit(node)
        test = ast.unparse(node.test)
        if 'np.random.rand' in test or 'torch.any(torch.isnan(pred))' in test:
            node.body = [ast.Pass()]
        if isinstance(node.test, ast.Compare) and isinstance(node.test.left, ast.Attribute) and node.test.left.attr == 'diff_prediction':
            if node.orelse and not isinstance(node.orelse[0], ast.If):
                node.orelse = [ast.Pass()]
        return node


forward_old = copy.deepcopy(old['Transformer.forward'])
forward_new = strip_docstring(new['Transformer.forward_macdiff'])
forward_old.name = forward_new.name
forward_old.args.kwarg = None
body = []
for node in forward_old.body:
    if (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == 'pred'):
        body.extend(ast.parse('z = self.build_global_local_condition(latent, cls_token, ids_restore)\npred = self.forward_decoder(x, z=z, t=t)').body)
    else:
        body.append(node)
forward_old.body = body
forward_old = NativeForwardNormalizer().visit(forward_old)
forward_new = NativeForwardNormalizer().visit(forward_new)
check('Transformer native forward after refactor normalization', forward_old, forward_new,
      'Rename local variables; express moved condition call; ignore diagnostic print bodies and invalid-target error body. Diagnostic RNG conditions are retained. The removed **kwargs is unused upstream.')

preprocessing_functions = []
for path in [row['path'] for row in inventory if row['path'].startswith('data/') and row['path'].endswith('.py')]:
    old_defs = definitions(tree(path, True))
    new_defs = definitions(tree(path))
    preprocessing_functions.append({
        'path': path, 'function_count': len(old_defs),
        'changed_functions': [name for name in old_defs if name in new_defs and not equal(old_defs[name], new_defs[name])],
    })
import_time_calls = {}
class TopLevelCalls(ast.NodeVisitor):
    def __init__(self):
        self.calls = []
    def visit_FunctionDef(self, node):
        for expression in node.decorator_list + node.args.defaults + [value for value in node.args.kw_defaults if value is not None]:
            self.visit(expression)
    def visit_Call(self, node):
        self.calls.append(ast.unparse(node))
        self.generic_visit(node)
for path in ('model/transformer_stage2.py', 'model/transformer_macdiff_text.py', 'model/ose_memory.py', 'feeder/feeder_stage2.py'):
    visitor = TopLevelCalls()
    visitor.visit(tree(path))
    import_time_calls[path] = visitor.calls

parser_defaults = {}
for file in ('main_pretrain.py', 'main_linprobe.py'):
    parser_defaults[file] = {}
    for original in (True, False):
        defs = {}
        for call in ast.walk(tree(file, original)):
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == 'add_argument':
                options = [arg.value for arg in call.args if isinstance(arg, ast.Constant) and isinstance(arg.value, str)]
                keywords = {kw.arg: ast.unparse(kw.value) for kw in call.keywords}
                for option in options:
                    defs[option] = keywords
        parser_defaults[file]['official' if original else 'current'] = defs
    official = parser_defaults[file]['official']
    local = parser_defaults[file]['current']
    parser_defaults[file]['changed_common_options'] = {key: {'official': official[key], 'current': local.get(key)} for key in official if official[key] != local.get(key)}
    parser_defaults[file]['added_options'] = sorted(set(local) - set(official))

counts = {}
for row in inventory:
    counts[row['status']] = counts.get(row['status'], 0) + 1
runtime_available = {}
import importlib.util
for name in ('torch', 'numpy', 'yaml'):
    runtime_available[name] = importlib.util.find_spec(name) is not None

report = {
    'baseline_commit': BASE,
    'baseline_url': 'https://github.com/LehongWu/MacDiff/tree/' + BASE,
    'current_head': git('rev-parse', 'HEAD').decode().strip(),
    'comparison': 'actual working-tree files; CRLF is normalized to LF for file equality',
    'original_file_count': len(inventory), 'original_file_status_counts': counts,
    'original_files': inventory, 'static_checks': checks,
    'all_static_checks_passed': all(row['passed'] for row in checks),
    'parser_defaults': parser_defaults,
    'expected_ntu60_xsub_tail_windows': {
        'sample_count_assumed': 40091,
        'profiles': [{'gpus': w, 'per_gpu_batch': b, 'accum_iter': a,
                      'loader_steps': ((40091 + w - 1) // w) // b,
                      'incomplete_window_steps': (((40091 + w - 1) // w) // b) % a}
                     for w,b,a in ((4,32,1),(2,32,1),(2,32,2),(2,64,1))]},
    'preprocessing_function_comparison': preprocessing_functions,
    'extension_import_time_calls': import_time_calls,
    'local_runtime_modules_available': runtime_available,
    'numerical_or_training_parity_executed': False,
    'limitations': ['No Torch runtime, GPU/DDP, data archive, or server checkout/checkpoint was available for this audit.',
                    'AST normalization is evidence about selected source paths, not proof of floating-point/gradient equivalence.',
                    'Condition gather uses expand instead of repeat for the integer index; tensor values and shapes match mathematically, runtime parity remains untested.'],
}
(OUT / 'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'baseline': BASE, 'current_head': report['current_head'], 'file_counts': counts,
                  'checks': [{'name': row['name'], 'passed': row['passed']} for row in checks],
                  'changed_common_parser_options': {key: value['changed_common_options'] for key,value in parser_defaults.items()},
                  'runtime_available': runtime_available}, ensure_ascii=False, indent=2))