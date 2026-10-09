"""Analyze the combined PT500/LP100 attachment; produce reproducible SVG/PNG plots."""
import hashlib
import json
import math
import statistics
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent
ATTACHMENTS = Path('C:/Users/97537/.codex/attachments')
SOURCE_ID = 'f6643024-0636-4677-ab2b-60b92d4ec967'
HISTORY = {'T12': 'cc1abd5f-c4d1-481b-8670-badf284c8140',
           'T13': '4cb61fd1-2294-42bf-82dd-a231581f3661'}


def load(identifier):
    path = next((ATTACHMENTS / identifier).glob('*.txt'))
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.decode('utf-8-sig').splitlines() if line.strip()]
    assert all(isinstance(row, dict) for row in rows), path
    assert all(math.isfinite(value) for row in rows for value in row.values()
               if isinstance(value, (int, float))), path
    return rows, {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'rows': len(rows)}


def window(rows, key, count):
    values = [row[key] for row in rows[-count:]]
    return {'start_epoch': rows[-count]['epoch'], 'end_epoch': rows[-1]['epoch'],
            'mean': statistics.mean(values), 'population_std': statistics.pstdev(values),
            'min': min(values), 'max': max(values)}


def lp_summary(rows):
    return {'best': max(rows, key=lambda row: row['test_acc1']),
            'first': rows[0], 'last': rows[-1],
            'last20': window(rows, 'test_acc1', 20),
            'last10': window(rows, 'test_acc1', 10),
            'zero_lr_epochs': [row['epoch'] for row in rows if row['train_lr'] == 0],
            'n_parameters': sorted({row['n_parameters'] for row in rows})}


def main():
    rows, identity = load(SOURCE_ID)
    pt = [row for row in rows if 'test_acc1' not in row]
    lp = [row for row in rows if 'test_acc1' in row]
    assert [row['epoch'] for row in pt] == list(range(500))
    assert [row['epoch'] for row in lp] == list(range(100))
    assert identity['rows'] == 600
    history = {}; historical_rows = {}
    for name, identifier in HISTORY.items():
        old, old_identity = load(identifier)
        assert [row['epoch'] for row in old] == list(range(100))
        history[name] = {'identity': old_identity, 'lp': lp_summary(old)}
        historical_rows[name] = old
    result = {'date': '2026-10-09', 'id': 'T14', 'source': identity,
              'integrity': {'pt_epochs': 500, 'lp_epochs': 100, 'ordered_contiguous': True,
                            'nonfinite_values': []},
              'user_reported': {'normalization': 'A', 'decoder_depth': 3},
              'command_confirmed': {'world_size': 2, 'pt_batch_per_gpu': 64, 'pt_accum_iter': 1,
                                   'pt_epochs': 500, 'lp_batch_per_gpu': 128, 'lp_accum_iter': 1,
                                   'lp_epochs': 100, 'seed': 0, 'lp_lr': 0.1,
                                   'lp_finetune': 'output_dir/ntu60_xsub_macdiff_decoder3/checkpoint-499.pth',
                                   'pt_lp_shared_output_directory': True},
              'not_yet_verified': ['server checkpoint actual depth/args', 'server code SHA',
                                   'NPZ identity', 'checkpoint actual args/identity'],
              'pt': {'first': pt[0], 'last': pt[-1], 'epoch399': pt[399],
                     'minimum_loss': min(pt, key=lambda row: row['train_loss']),
                     'last20_loss': window(pt, 'train_loss', 20),
                     'last50_loss': window(pt, 'train_loss', 50),
                     'last100_loss': window(pt, 'train_loss', 100),
                     'late_min_lr_observed': pt[-1]['train_lr'],
                     'peak_lr_observed': max(row['train_lr'] for row in pt),
                     'loss_drop_399_to_499_percent': 100*(pt[399]['train_loss']-pt[-1]['train_loss'])/pt[399]['train_loss']},
              'lp': lp_summary(lp), 'history': history,
              'statistics_note': 'Epoch SD is within-run fluctuation, not a multi-seed confidence interval.'}
    result['comparison_pp'] = {'paper_86_4_minus_T14_best': 86.4-result['lp']['best']['test_acc1'],
                               'T12_best_minus_T14_best': history['T12']['lp']['best']['test_acc1']-result['lp']['best']['test_acc1'],
                               'T12_tail20_minus_T14_tail20': history['T12']['lp']['last20']['mean']-result['lp']['last20']['mean']}
    OUTPUT.joinpath('summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    render(pt, lp, historical_rows)
    print(json.dumps({'counts': result['integrity'], 'best': result['lp']['best'],
                      'last20': result['lp']['last20'], 'differences_pp': result['comparison_pp'],
                      'pt_late_loss_drop_percent': result['pt']['loss_drop_399_to_499_percent']}, indent=2))


def render(pt, lp, history):
    from reportlab.graphics import renderSVG, renderPDF
    from reportlab.graphics.charts.lineplots import LinePlot
    from reportlab.graphics.shapes import Drawing, Rect, String, Line
    from reportlab.lib.colors import HexColor
    blue, orange, green = map(HexColor, ['#2563EB', '#EA580C', '#16A34A'])
    ink, muted, grid = map(HexColor, ['#172033', '#5B6474', '#E5EAF0'])
    d = Drawing(1200, 875)
    d.add(Rect(0, 0, 1200, 875, fillColor=HexColor('#F7F9FC'), strokeColor=None))
    def label(x, y, value, size=11, bold=False, color=ink):
        d.add(String(x, y, value, fontName='Helvetica-Bold' if bold else 'Helvetica', fontSize=size, fillColor=color))
    def panel(x, y, title, subtitle, series, colors, xr, xt, yr, yt, fmt):
        d.add(Rect(x, y, 550, 320, rx=8, ry=8, fillColor=HexColor('#FFFFFF'), strokeColor=grid))
        label(x+18, y+294, title, 14, True)
        label(x+18, y+274, subtitle, 10, color=muted)
        plot = LinePlot(); plot.x, plot.y, plot.width, plot.height = x+62, y+56, 465, 197
        plot.data = series; plot.joinedLines = True
        plot.xValueAxis.valueMin, plot.xValueAxis.valueMax, plot.xValueAxis.valueSteps = xr[0], xr[1], xt
        plot.yValueAxis.valueMin, plot.yValueAxis.valueMax, plot.yValueAxis.valueSteps = yr[0], yr[1], yt
        plot.yValueAxis.labelTextFormat = fmt
        for axis in [plot.xValueAxis, plot.yValueAxis]:
            axis.labels.fontName = 'Helvetica'; axis.labels.fontSize = 9
            axis.labels.fillColor = muted; axis.strokeColor = grid
            axis.visibleGrid = True; axis.gridStrokeColor = grid; axis.gridStrokeWidth = .5
        for i, color in enumerate(colors):
            plot.lines[i].strokeColor = color; plot.lines[i].strokeWidth = 1.2 if i < 3 else .8
        d.add(plot); label(x+275, y+21, 'Epoch (zero based)', 10, color=muted)
    def points(rows, key, start=0):
        return [(r['epoch'], r[key]) for r in rows if r['epoch'] >= start]
    label(30, 835, 'MacDiff: PT500 run, linear probe of checkpoint-499', 23, True)
    label(30, 809, 'T14: A normalization, user-reported decoder3 | LP499 best 85.704755% at epoch94', 12, color=muted)
    for x, name, color in [(30, 'T14 native / A (new)', blue), (300, 'T12 no-share / A', green), (570, 'T13 native / B', orange)]:
        d.add(Line(x, 782, x+22, 782, strokeColor=color, strokeWidth=2))
        label(x+30, 778, name, 11)
    panel(30, 426, 'A. Native pretraining loss', 'Epochs 20-499; total loss includes skeleton uniformity',
          [points(pt, 'train_loss', 20)], [blue], (20, 499), [20,100,200,300,399,499],
          (.014, .029), [.015,.018,.021,.024,.027], '%.3f')
    panel(620, 426, 'B. Pretraining learning rate', 'Observed floor 5e-4; orange reference: same schedule with floor 1e-5',
          [points(pt, 'train_lr'), [(e, lr_reference(e+.5)) for e in range(500)]], [blue, orange],
          (0,499), [0,100,200,300,399,499], (0,.00105), [0,.00025,.0005,.00075,.001], '%.5f')
    # The reference is an illustrative counterfactual schedule, never a trained result.
    # Different batches/BN/depth/normalization mean the history curves are not matched controls.
    panel(30, 78, 'C. Linear probe top-1 accuracy (%)', 'T14 LP batch 256; T12 batch 128; T13 batch 256',
          [points(lp,'test_acc1'),points(history['T12'],'test_acc1'),points(history['T13'],'test_acc1'),
           [(0,86.4),(99,86.4)]], [blue,green,orange,muted], (0,99), [0,20,40,60,80,99],
          (74,87), [75,78,81,84,87], '%.1f')
    panel(620, 78, 'D. Linear probe late top-1 (%)', 'Epochs 60-99; LR=0 for 90-99, BN running statistics still update',
          [points(lp,'test_acc1',60),points(history['T12'],'test_acc1',60),points(history['T13'],'test_acc1',60),
           [(60,86.4),(99,86.4)]], [blue,green,orange,muted], (60,99), [60,70,80,90,99],
          (84.8,86.5), [84.8,85.2,85.6,86,86.4], '%.1f')
    label(30, 48, 'Historical curves have different protocols and are descriptive comparisons. Horizontal gray line: paper 86.4%. No smoothing.', 10, color=muted)
    label(30, 29, 'PT499 is evaluated. Reference LR in panel B is illustrative; it is not a training run.', 10, color=muted)
    renderSVG.drawToFile(d, str(OUTPUT/'comparison.svg'))
    import pypdfium2
    document = pypdfium2.PdfDocument(renderPDF.drawToString(d))
    page = document[0]
    bitmap = page.render(scale=120/72)
    bitmap.to_pil().save(OUTPUT/'comparison.png')
    bitmap.close(); page.close(); document.close()


def lr_reference(epoch):
    if epoch < 20:
        return .001*epoch/20
    if epoch > 480:
        return .00001
    return .00001 + (.001-.00001)*.5*(1+math.cos(math.pi*(epoch-20)/460))


if __name__ == '__main__':
    main()