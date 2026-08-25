"""Evalúa la precisión de búsqueda de una lista contra un despliegue del search-service.

Lanza las consultas del dataset (query -> elemento esperado) contra POST /api/v1/search y
calcula métricas de ranking (top-1 accuracy, recall@k, MRR...). Guarda el run en
evaluation/results/<lista>/ para poder compararlo después con `python -m evaluation.compare`.

Ejemplo contra producción:
  python -m evaluation.evaluate --list Productos --api-key xeye_... \
      --search-url https://search.xeye.es --label bge-m3

El rate limit del servicio es 60 req/min por API key (ventana fija): el script espacia las
peticiones (--qpm) y reintenta los 429 esperando a la ventana siguiente.
"""

import argparse
import asyncio
import getpass
import json
import os
import sys
import time
from pathlib import Path

import httpx

from .backend import BackendClient, BackendError
from .common import dataset_path, list_results_dir, slugify
from .metrics import normalize, rank_of_expected, summarize

MAX_RETRIES_429 = 3
TOP_N_SAVED = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.evaluate", description=__doc__.splitlines()[0]
    )
    parser.add_argument("--list", required=True, dest="list_name",
                        help="nombre EXACTO de la lista (sensible a mayúsculas)")
    parser.add_argument("--api-key", default=os.environ.get("XEYE_API_KEY"),
                        help="API key xeye_... (o variable de entorno XEYE_API_KEY)")
    parser.add_argument("--search-url", default="http://localhost:8002",
                        help="base del search-service (producción: https://search.xeye.es)")
    parser.add_argument("--dataset", type=Path, default=None,
                        help="fichero de consultas (por defecto evaluation/datasets/<lista>.json)")
    parser.add_argument("--limit", type=int, default=50, help="resultados pedidos por consulta")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--qpm", type=int, default=55,
                        help="consultas/minuto máx. (el servicio corta a 60 por key)")
    parser.add_argument("--allow-private", action="store_true",
                        help="permite buscar en listas privadas propias")
    parser.add_argument("--label", default=None,
                        help="etiqueta del run (p.ej. el modelo de embedding) si no usas --email")
    parser.add_argument("--backend-url", default="http://localhost:8000")
    parser.add_argument("--email", default=None,
                        help="credenciales del backend para etiquetar el run con el modelo en uso")
    parser.add_argument("--password", default=None)
    return parser.parse_args()


def load_dataset(path: Path) -> list[dict]:
    if not path.is_file():
        sys.exit(f"No existe el dataset {path}")
    with path.open(encoding="utf-8") as fh:
        entries = json.load(fh)
    problems = [e for e in entries if not e.get("query") or not e.get("expected")]
    if problems:
        sys.exit(f"{len(problems)} entradas del dataset sin 'query' o 'expected'")
    identical = [e["query"] for e in entries if normalize(e["query"]) == normalize(e["expected"])]
    if identical:
        print(f"AVISO: {len(identical)} consultas idénticas (normalizadas) a su esperado — el "
              f"servicio les da score 1.0 por match exacto y no miden nada: {identical}")
    return entries


def resolve_run_label(args: argparse.Namespace) -> tuple[str | None, int | None, str | None]:
    """(model, training_id, error) consultando el backend si hay credenciales."""
    if not args.email:
        return None, None, None
    password = args.password or getpass.getpass(f"Contraseña de {args.email}: ")
    try:
        client = BackendClient(args.backend_url, args.email, password)
        try:
            list_id = client.find_list_id(args.list_name)
            in_use = client.in_use_training(list_id)
        finally:
            client.close()
        if in_use is None:
            return None, None, "la lista no tiene ningún training en uso"
        return in_use[1], in_use[0], None
    except (BackendError, httpx.HTTPError) as exc:
        return None, None, str(exc)


class Pacer:
    """Espacia los inicios de petición para no pasar de qpm (ventana fija en el servidor)."""

    def __init__(self, qpm: int):
        self._interval = 60.0 / max(1, qpm)
        self._lock = asyncio.Lock()
        self._next_start = 0.0

    async def wait(self) -> None:
        async with self._lock:
            now = time.monotonic()
            start = max(now, self._next_start)
            self._next_start = start + self._interval
        delay = start - now
        if delay > 0:
            await asyncio.sleep(delay)


async def run_query(client: httpx.AsyncClient, args: argparse.Namespace,
                    pacer: Pacer, entry: dict) -> dict:
    payload = {
        "list_name": args.list_name,
        "search_term": entry["query"],
        "limit": args.limit,
        "include_score_breakdown": True,
        "register_log": False,
        "allow_private": args.allow_private,
    }
    for attempt in range(MAX_RETRIES_429 + 1):
        await pacer.wait()
        started = time.monotonic()
        response = await client.post("/api/v1/search", json=payload)
        client_ms = int((time.monotonic() - started) * 1000)
        if response.status_code == 429:
            wait = 61 - (time.time() % 60)
            print(f"  429 rate limit ({entry['query'][:40]!r}); espero {wait:.0f}s "
                  f"(si el servidor es tuyo, sube RATE_LIMIT_PER_MINUTE)")
            if attempt < MAX_RETRIES_429:
                await asyncio.sleep(wait)
                continue
        if response.status_code != 200:
            raise RuntimeError(
                f"HTTP {response.status_code} en {entry['query']!r}: {response.text[:300]}")
        break
    body = response.json()
    results = body.get("results", [])
    rank = rank_of_expected([r["item"] for r in results], entry["expected"])
    expected_result = results[rank - 1] if rank is not None else None
    semantic_won = None
    if expected_result is not None and expected_result.get("semantic_score") is not None:
        semantic_won = expected_result["semantic_score"] >= (expected_result.get("text_score") or 0.0)
    return {
        "query": entry["query"],
        "expected": entry["expected"],
        "category": entry.get("category"),
        "rank": rank,
        "score": expected_result["score"] if expected_result else None,
        "semantic_won": semantic_won,
        "duration_ms": body.get("duration_ms"),
        "client_ms": client_ms,
        "total_results": body.get("total_results"),
        "top": [
            {k: r.get(k) for k in ("item", "score", "text_score", "semantic_score")}
            for r in results[:TOP_N_SAVED]
        ],
    }


async def run_all(args: argparse.Namespace, entries: list[dict]) -> list[dict]:
    pacer = Pacer(args.qpm)
    semaphore = asyncio.Semaphore(args.concurrency)
    done = 0

    async with httpx.AsyncClient(
        base_url=args.search_url.rstrip("/"),
        headers={"X-API-Key": args.api_key},
        timeout=30.0,
    ) as client:
        async def worker(entry: dict) -> dict:
            nonlocal done
            async with semaphore:
                result = await run_query(client, args, pacer, entry)
            done += 1
            mark = f"#{result['rank']}" if result["rank"] else "MISS"
            print(f"  [{done}/{len(entries)}] {mark:>5}  {entry['query']}")
            return result

        return list(await asyncio.gather(*(worker(e) for e in entries)))


def print_summary(metrics: dict, queries: list[dict]) -> None:
    print("\n=== Métricas ===")
    for key in ("queries", "top1_accuracy", "recall@3", "recall@5", "recall@10", "mrr",
                "mean_found_rank", "not_found_count", "avg_duration_ms", "p95_duration_ms",
                "semantic_contribution"):
        value = metrics.get(key)
        if isinstance(value, float):
            value = f"{value:.3f}"
        print(f"  {key:22} {value}")
    if metrics.get("by_category"):
        print("\n=== Por categoría de consulta ===")
        for category, sub in metrics["by_category"].items():
            print(f"  {category:12} n={sub['queries']:<3} top1={sub['top1_accuracy']:.2f} "
                  f"recall@3={sub['recall@3']:.2f} mrr={sub['mrr']:.2f}")
    misses = [q for q in queries if q["rank"] is None or q["rank"] > 3]
    if misses:
        print("\n=== Fallos (fuera del top-3) ===")
        for q in misses:
            rank = q["rank"] or "no encontrado"
            best = q["top"][0]["item"] if q["top"] else "-"
            print(f"  [{rank}] {q['query']!r} -> esperado {q['expected']!r}; 1º: {best!r}")


def save_run(run: dict, list_name: str) -> Path:
    directory = list_results_dir(list_name)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    label = slugify(run.get("model") or run.get("label") or "run")
    path = directory / f"{stamp}_{label}.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(run, fh, ensure_ascii=False, indent=2)

    csv_path = directory / "runs.csv"
    metrics = run["metrics"]
    header = ("timestamp,label,model,training_id,queries,top1_accuracy,recall@3,recall@5,"
              "recall@10,mrr,mean_found_rank,not_found,avg_duration_ms,p95_duration_ms\n")
    if not csv_path.exists():
        csv_path.write_text(header, encoding="utf-8")
    def fmt(value):  # noqa: E306
        return "" if value is None else (f"{value:.4f}" if isinstance(value, float) else str(value))
    row = [run["timestamp"], run.get("label") or "", run.get("model") or "",
           run.get("training_id") or "", metrics["queries"], metrics["top1_accuracy"],
           metrics["recall@3"], metrics["recall@5"], metrics["recall@10"], metrics["mrr"],
           metrics["mean_found_rank"], metrics["not_found_count"], metrics["avg_duration_ms"],
           metrics["p95_duration_ms"]]
    with csv_path.open("a", encoding="utf-8") as fh:
        fh.write(",".join(fmt(v) for v in row) + "\n")
    return path


def main() -> None:
    args = parse_args()
    if not args.api_key:
        sys.exit("Falta la API key: --api-key o variable XEYE_API_KEY")
    dataset_file = args.dataset or dataset_path(args.list_name)
    entries = load_dataset(dataset_file)

    model, training_id, backend_error = resolve_run_label(args)
    if backend_error:
        print(f"AVISO: no pude etiquetar el run desde el backend: {backend_error}")
    label = args.label or model
    if not label:
        print("AVISO: run sin etiqueta (--label o --email); compare lo mostrará por timestamp")

    print(f"Evaluando {len(entries)} consultas de {dataset_file.name} contra "
          f"{args.search_url} (lista {args.list_name!r}, modelo {model or label or '?'})\n")
    started = time.time()
    queries = asyncio.run(run_all(args, entries))
    metrics = summarize(queries)
    print_summary(metrics, queries)

    run = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "list_name": args.list_name,
        "search_url": args.search_url,
        "dataset": dataset_file.name,
        "label": label,
        "model": model,
        "training_id": training_id,
        "limit": args.limit,
        "wall_seconds": round(time.time() - started, 1),
        "metrics": metrics,
        "queries": queries,
    }
    path = save_run(run, args.list_name)
    print(f"\nRun guardado en {path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path}")
    print(f"Compara runs con: python -m evaluation.compare --list '{args.list_name}'")


if __name__ == "__main__":
    main()
