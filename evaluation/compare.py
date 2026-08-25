"""Compara los runs guardados de una lista y genera tabla (CSV/Markdown) y gráficas PNG.

  python -m evaluation.compare --list Productos

Lee evaluation/results/<lista>/*.json y deja en esa misma carpeta comparison.csv,
comparison.md y charts/{metrics,latency,ranks}.png (pensados para la memoria del TFG).
"""

import argparse
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .common import list_results_dir, load_runs  # noqa: E402

METRIC_COLUMNS = ["top1_accuracy", "recall@3", "recall@5", "recall@10", "mrr"]
RANK_BUCKETS = [("1", lambda r: r == 1), ("2-3", lambda r: r is not None and 2 <= r <= 3),
                ("4-10", lambda r: r is not None and 4 <= r <= 10),
                (">10 / no", lambda r: r is None or r > 10)]


def run_label(run: dict) -> str:
    return run.get("label") or run.get("model") or run["timestamp"]


def build_rows(runs: list[dict]) -> list[dict]:
    rows = []
    for run in runs:
        metrics = run["metrics"]
        rows.append({
            "run": run_label(run),
            "timestamp": run["timestamp"],
            "training_id": run.get("training_id") or "",
            "queries": metrics["queries"],
            **{col: metrics.get(col) for col in METRIC_COLUMNS},
            "mean_found_rank": metrics.get("mean_found_rank"),
            "not_found": metrics.get("not_found_count"),
            "avg_ms": metrics.get("avg_duration_ms"),
            "p95_ms": metrics.get("p95_duration_ms"),
        })
    return rows


def fmt(value, decimals=3):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{decimals}f}"
    return str(value)


def write_tables(rows: list[dict], directory) -> None:
    columns = list(rows[0].keys())
    csv_path = directory / "comparison.csv"
    with csv_path.open("w", encoding="utf-8") as fh:
        fh.write(",".join(columns) + "\n")
        for row in rows:
            fh.write(",".join(fmt(row[c]) for c in columns) + "\n")

    md_path = directory / "comparison.md"
    with md_path.open("w", encoding="utf-8") as fh:
        fh.write("| " + " | ".join(columns) + " |\n")
        fh.write("|" + "---|" * len(columns) + "\n")
        for row in rows:
            fh.write("| " + " | ".join(fmt(row[c]) for c in columns) + " |\n")
    print(f"Escritos {csv_path.name} y {md_path.name}")


def chart_metrics(rows: list[dict], charts_dir) -> None:
    labels = [r["run"] for r in rows]
    x = range(len(METRIC_COLUMNS))
    width = 0.8 / max(1, len(rows))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, row in enumerate(rows):
        offsets = [xi + i * width for xi in x]
        values = [row[c] or 0.0 for c in METRIC_COLUMNS]
        bars = ax.bar(offsets, values, width=width, label=labels[i])
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=1)
    ax.set_xticks([xi + width * (len(rows) - 1) / 2 for xi in x])
    ax.set_xticklabels(METRIC_COLUMNS)
    ax.set_ylim(0, 1.28)
    ax.set_ylabel("valor (0-1)")
    ax.set_title("Métricas de precisión por run")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(charts_dir / "metrics.png", dpi=150)
    plt.close(fig)


def chart_latency(rows: list[dict], charts_dir) -> None:
    labels = [r["run"] for r in rows]
    x = range(len(rows))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([xi - 0.2 for xi in x], [r["avg_ms"] or 0 for r in rows], width=0.4, label="media")
    ax.bar([xi + 0.2 for xi in x], [r["p95_ms"] or 0 for r in rows], width=0.4, label="p95")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("latencia del servicio (ms)")
    ax.set_title("Latencia de búsqueda por run")
    ax.legend()
    fig.tight_layout()
    fig.savefig(charts_dir / "latency.png", dpi=150)
    plt.close(fig)


def chart_ranks(runs: list[dict], charts_dir) -> None:
    labels = [run_label(r) for r in runs]
    bucket_names = [name for name, _ in RANK_BUCKETS]
    x = range(len(bucket_names))
    width = 0.8 / max(1, len(runs))
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, run in enumerate(runs):
        ranks = [q["rank"] for q in run.get("queries", [])]
        counts = [sum(1 for r in ranks if match(r)) for _, match in RANK_BUCKETS]
        ax.bar([xi + i * width for xi in x], counts, width=width, label=labels[i])
    ax.set_xticks([xi + width * (len(runs) - 1) / 2 for xi in x])
    ax.set_xticklabels(bucket_names)
    ax.set_ylabel("nº de consultas")
    ax.set_xlabel("posición del resultado esperado")
    ax.set_title("Distribución del rango del esperado")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(charts_dir / "ranks.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m evaluation.compare",
                                     description=__doc__.splitlines()[0])
    parser.add_argument("--list", required=True, dest="list_name")
    args = parser.parse_args()

    runs = load_runs(args.list_name)
    if not runs:
        sys.exit(f"No hay runs en {list_results_dir(args.list_name)} — lanza antes "
                 f"python -m evaluation.evaluate --list '{args.list_name}' ...")
    print(f"{len(runs)} runs de {args.list_name!r}: " + ", ".join(run_label(r) for r in runs))

    directory = list_results_dir(args.list_name)
    charts_dir = directory / "charts"
    charts_dir.mkdir(exist_ok=True)

    rows = build_rows(runs)
    write_tables(rows, directory)
    chart_metrics(rows, charts_dir)
    chart_latency(rows, charts_dir)
    chart_ranks(runs, charts_dir)
    print(f"Gráficas en {charts_dir}/(metrics|latency|ranks).png")


if __name__ == "__main__":
    main()
