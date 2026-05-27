from __future__ import annotations

from pathlib import Path

import pandas as pd

METRICS = ["mse", "mae", "rmse", "mape", "smape"]


def _fmt(value: float, bold: bool, fmt: str) -> str:
    text = f"{value:.4f}"
    if not bold:
        return text
    return f"**{text}**" if fmt == "markdown" else f"\\textbf{{{text}}}"


def build_table(input_csv: str | Path, fmt: str = "markdown") -> str:
    df = pd.read_csv(input_csv)
    horizons = sorted(df["pred_len"].unique()) if "pred_len" in df else [None]
    cols = ["model"]
    for h in horizons:
        for m in METRICS:
            cols.append(f"{m}@{h}" if h is not None else m)
    rows = []
    for model in sorted(df["model"].unique()):
        row = {"model": model}
        for h in horizons:
            sub_h = df[df["pred_len"] == h] if h is not None else df
            for m in METRICS:
                model_value = float(sub_h[sub_h["model"] == model][m].mean())
                best = float(sub_h[m].min())
                row[f"{m}@{h}" if h is not None else m] = _fmt(model_value, abs(model_value - best) < 1e-12, fmt)
        rows.append(row)
    if fmt == "markdown":
        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |"
        body = ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
        return "\n".join([header, sep, *body]) + "\n"
    if fmt == "latex":
        nl = " " + '\\\\'
        lines = ["\\begin{tabular}{" + "l" + "r" * (len(cols) - 1) + "}", " \\toprule", " & ".join(cols) + nl, " \\midrule"]
        lines += [" & ".join(str(r[c]) for c in cols) + nl for r in rows]
        lines += [" \\bottomrule", "\\end{tabular}"]
        return "\n".join(lines) + "\n"
    raise ValueError("format must be markdown or latex")


def write_table(input_csv: str | Path, output: str | Path, fmt: str) -> Path:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_table(input_csv, fmt), encoding="utf-8")
    return out
