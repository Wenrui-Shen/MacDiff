"""Render the uploaded MacDiff logs with bundled ReportLab; no extra dependencies."""
from __future__ import annotations

import json
from pathlib import Path

from reportlab.graphics import renderSVG
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.shapes import Drawing, Group, Line, Rect, String
from reportlab.lib.colors import HexColor, white


OUTPUT = Path(__file__).resolve().parent
ATTACHMENTS = Path(r"C:\Users\97537\.codex\attachments")
SOURCE_IDS = {
    "noshare_pt": "02d00a0e-de94-42cb-98f4-6ad1cc73af47",
    "noshare_lp": "cc1abd5f-c4d1-481b-8670-badf284c8140",
    "normb_lp": "4cb61fd1-2294-42bf-82dd-a231581f3661",
    "shared_pt": "64857d73-03e2-4bf9-97ed-e6ca4fba0355",
}
BLUE = HexColor("#2563EB")
ORANGE = HexColor("#E87924")
GRAY = HexColor("#7B8494")
INK = HexColor("#172033")
MUTED = HexColor("#566176")
GRID = HexColor("#E8ECF2")
OUTLINE = HexColor("#D8DFE9")
HISTORICAL_BEST = 85.79


def load_rows(source_id: str) -> list[dict]:
    source = next((ATTACHMENTS / source_id).glob("*.txt"))
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    assert len({row["epoch"] for row in rows}) == len(rows), source
    return sorted(rows, key=lambda row: row["epoch"])


def points(rows: list[dict], key: str, first: int = 0) -> list[tuple[int, float]]:
    return [(row["epoch"], row[key]) for row in rows if row["epoch"] >= first]


def text(drawing, x, y, value, size=10, color=INK, anchor="start", bold=False):
    drawing.add(String(x, y, value, fontName="Helvetica-Bold" if bold else "Helvetica",
                       fontSize=size, fillColor=color, textAnchor=anchor))


def panel(drawing, x, y, title, subtitle, series, colors, x_range, x_ticks,
          y_range, y_ticks, y_label, y_format, reference=False):
    drawing.add(Rect(x, y, 550, 330, rx=9, ry=9, fillColor=white,
                     strokeColor=OUTLINE, strokeWidth=0.8))
    text(drawing, x + 20, y + 306, title, size=14, bold=True)
    text(drawing, x + 20, y + 285, subtitle, size=10, color=MUTED)
    plot = LinePlot()
    plot.x, plot.y = x + 65, y + 64
    plot.width, plot.height = 460, 208
    plot.data = series
    plot.joinedLines = True
    plot.xValueAxis.valueMin, plot.xValueAxis.valueMax = x_range
    plot.xValueAxis.valueSteps = x_ticks
    plot.yValueAxis.valueMin, plot.yValueAxis.valueMax = y_range
    plot.yValueAxis.valueSteps = y_ticks
    plot.yValueAxis.labelTextFormat = y_format
    for axis in (plot.xValueAxis, plot.yValueAxis):
        axis.strokeColor = OUTLINE
        axis.strokeWidth = 0.7
        axis.labels.fontName = "Helvetica"
        axis.labels.fontSize = 9
        axis.labels.fillColor = MUTED
        axis.visibleGrid = True
        axis.gridStrokeColor = GRID
        axis.gridStrokeWidth = 0.6
    plot.xValueAxis.labels.dy = -9
    plot.yValueAxis.labels.dx = -7
    for i, color in enumerate(colors):
        plot.lines[i].strokeColor = color
        plot.lines[i].strokeWidth = 1.55
    if reference:
        plot.lines[len(series) - 1].strokeWidth = 1.1
        plot.lines[len(series) - 1].strokeDashArray = [5, 3]
    drawing.add(plot)
    text(drawing, x + 295, y + 22, "Epoch (zero-based)", size=10, color=MUTED, anchor="middle")
    label_group = Group()
    label_group.translate(x + 18, y + 165)
    label_group.rotate(90)
    label_group.add(String(0, 0, y_label, fontName="Helvetica", fontSize=10,
                          fillColor=MUTED, textAnchor="middle"))
    drawing.add(label_group)


def legend_item(drawing, x, y, label, color, dashed=False):
    sample = Line(x, y + 3, x + 24, y + 3, strokeColor=color, strokeWidth=2)
    if dashed:
        sample.strokeDashArray = [5, 3]
    drawing.add(sample)
    text(drawing, x + 32, y, label, size=10, color=MUTED)


def main():
    rows = {key: load_rows(source_id) for key, source_id in SOURCE_IDS.items()}
    ns_lp, b_lp = rows["noshare_lp"], rows["normb_lp"]
    ns_best = max(ns_lp, key=lambda row: row["test_acc1"])
    b_best = max(b_lp, key=lambda row: row["test_acc1"])
    d = Drawing(1180, 920)
    d.add(Rect(0, 0, 1180, 920, fillColor=HexColor("#F7F9FC"), strokeColor=None))
    text(d, 30, 887, "MacDiff: no-share and new normalization", size=23, bold=True)
    text(d, 30, 862, "Different protocols: no-share A effective batch 128 vs original B effective batch 256 (both PT and LP)",
         size=11, color=MUTED)
    legend_item(d, 30, 837, "no-share / A", BLUE)
    legend_item(d, 240, 837, "original / B (new)", ORANGE)
    legend_item(d, 495, 837, "shared-512 PT", GRAY)
    legend_item(d, 728, 837, "shared 85.79%: historical single best", GRAY, dashed=True)

    subtitle = f"Best: no-share {ns_best['test_acc1']:.2f}% (e{ns_best['epoch']}) | B {b_best['test_acc1']:.2f}% (e{b_best['epoch']})"
    panel(d, 30, 477, "A. Linear probe accuracy", subtitle,
          [points(ns_lp, "test_acc1"), points(b_lp, "test_acc1"), [(0, HISTORICAL_BEST), (99, HISTORICAL_BEST)]],
          [BLUE, ORANGE, GRAY], (0, 99), [0, 20, 40, 60, 80, 99],
          (74, 87), [74, 76, 78, 80, 82, 84, 86], "Top-1 accuracy (%)", "%.0f", reference=True)
    panel(d, 600, 477, "B. Linear probe accuracy: late-stage view", "Epochs 40-99; same data, expanded vertical scale",
          [points(ns_lp, "test_acc1", 40), points(b_lp, "test_acc1", 40), [(40, HISTORICAL_BEST), (99, HISTORICAL_BEST)]],
          [BLUE, ORANGE, GRAY], (40, 99), [40, 50, 60, 70, 80, 90, 99],
          (83.5, 86.5), [83.5, 84, 84.5, 85, 85.5, 86, 86.5], "Top-1 accuracy (%)", "%.1f", reference=True)
    panel(d, 30, 87, "C. Pretraining: skeleton-to-text loss", "No-share vs shared-512; unweighted loss, epochs 100-399",
          [points(rows["noshare_pt"], "train_loss_skeleton_to_text", 100),
           points(rows["shared_pt"], "train_loss_skeleton_to_text", 100)],
          [BLUE, GRAY], (100, 399), [100, 150, 200, 250, 300, 350, 399],
          (0, 0.03), [0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03], "S -> T loss (dimensionless)", "%.3f")
    panel(d, 600, 87, "D. Linear probe test cross-entropy", "Raw test CE; accuracy and confidence measure different properties",
          [points(ns_lp, "test_loss"), points(b_lp, "test_loss")],
          [BLUE, ORANGE], (0, 99), [0, 20, 40, 60, 80, 99],
          (0, 7), [0, 1, 2, 3, 4, 5, 6, 7], "Cross-entropy (nats)", "%.0f")

    text(d, 30, 55, "A / B are different input mean-variance settings. Batch values above are effective across two GPUs, including accumulation.",
         size=10, color=MUTED)
    text(d, 30, 37, "Gray dashed line is a historical best, not a matched LP curve. No smoothing. PT: 400 epochs. LP: 100 epochs.",
         size=10, color=MUTED)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    svg = OUTPUT / "comparison.svg"
    renderSVG.drawToFile(d, str(svg))
    print(f"SVG: {svg}")
    try:
        from reportlab.graphics import renderPM
        png = OUTPUT / "comparison.png"
        renderPM.drawToFile(d, str(png), fmt="PNG", dpi=140)
        print(f"PNG: {png}")
    except Exception as exc:
        print(f"PNG unavailable: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
