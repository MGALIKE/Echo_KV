"""Text-model benchmarks for echokv (run locally, log JSON, emit PNG+PDF figures).

Produces, per model: per-layer triviality bars, the quality(ppl-gap)-vs-saving
Pareto (echo vs random vs anti-oracle), KV-saving-vs-context, retrieval-needle
survival, and chunked-prefill peak memory.  Every number is measured here and
written to results/<model>_<bench>.json; figures go to figures/.

    python run_text_benchmarks.py [--models gpt2,Qwen/Qwen2.5-0.5B-Instruct] [--quick]

Designed for a 4 GB GPU.  bf16 for modern models, fp32 for GPT-2.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from echokv import calibrate  # noqa: E402
from echokv.benchmarks import (bench_chunked_peak, bench_needle_survival,  # noqa: E402
                               bench_ppl_pareto, bench_saving_vs_context,
                               bench_triviality, hardware_tag, load_model,
                               _plot_pareto, _plot_triviality)

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)


def slug(name):
    return name.split("/")[-1].replace(".", "_")


def wikitext_paragraphs(n=300, min_chars=200):
    """Held-out calibration corpus (WikiText-2), matching the X9/X10/X11 protocol.
    Falls back to None (package default calibration) if the dataset is unavailable."""
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        return [t for t in ds["text"] if len(t) > min_chars][:n]
    except Exception as e:
        print(f"  [warn] WikiText unavailable ({e}); using package default calibration")
        return None


def dump(obj, name):
    with open(os.path.join(RES, name + ".json"), "w") as f:
        json.dump(obj, f, indent=2)
    print(f"  -> results/{name}.json")


def plot_saving_vs_context(res, path, title):
    pts = res["points"]
    fig, ax = plt.subplots(figsize=(6, 4))
    x = [p["final_len"] for p in pts]
    ax.plot(x, [100 * p["kv_saving"] for p in pts], "o-", c="#2e86ab",
            label="measured")
    ax.plot(x, [100 * p["projected"] for p in pts], "s--", c="#a0a0a0",
            label="projected", alpha=0.7)
    ax.axhline(100 * res["asymptote"], ls=":", c="#d1495b",
               label=f"asymptote n_local/L = {100*res['asymptote']:.0f}%")
    ax.set_xlabel("context length (tokens)")
    ax.set_ylabel("KV-cache saving (%)")
    ax.set_title(title); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=140)
    plt.close(fig)


def plot_needle(res, path, title):
    pts = res["points"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ks = [p["k"] for p in pts]
    ax.plot(ks, [p["echo"] for p in pts], "o-", c="#2e86ab", label="echo")
    rm = [p["random_mean"] for p in pts]
    ax.plot(ks, rm, "^:", c="gray", label="random (mean)")
    ax.fill_between(ks, [p["random_min"] for p in pts],
                    [p["random_max"] for p in pts], color="gray", alpha=0.2,
                    label="random (min-max)")
    ax.plot(ks, [p["anti"] for p in pts], "s--", c="#d1495b", label="anti-oracle")
    ax.set_xlabel("number of layers localized (K)")
    ax.set_ylabel("fraction of needle retrievals preserved")
    ax.set_title(title + f"\n({res['retrievable']} retrievable needles)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=140)
    plt.close(fig)


def plot_peak(res_list, path):
    """Bar chart: full vs echo peak memory, across the contexts that ran."""
    fig, ax = plt.subplots(figsize=(6, 4))
    labels, full, echo = [], [], []
    for r in res_list:
        if "full_peak_mb" not in r:
            continue
        labels.append(f"T={r['context']}")
        full.append(r["full_peak_mb"]); echo.append(r["echo_peak_mb"])
    if not labels:
        plt.close(fig); return False
    x = range(len(labels)); w = 0.38
    ax.bar([i - w/2 for i in x], full, w, label="full (single-shot prefill)",
           color="#a0a0a0")
    ax.bar([i + w/2 for i in x], echo, w, label="echo (chunked prefill)",
           color="#2e86ab")
    for i, (f, e) in enumerate(zip(full, echo)):
        ax.text(i, max(f, e), f"-{100*(1-e/f):.0f}%", ha="center", va="bottom",
                fontsize=8)
    ax.set_xticks(list(x)); ax.set_xticklabels(labels)
    ax.set_ylabel("peak CUDA memory (MB)")
    ax.set_title("Chunked-prefill peak memory: full vs echo")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=140)
    plt.close(fig)
    return True


WINDOW = 48        # X9/X10/X11 protocol; deployable tool default is 64 (more conservative)
SINK = 4


def run_model(name, calib, quick=False):
    print(f"\n=== {name} ({hardware_tag()}) ===")
    dtype = torch.float32 if "gpt2" in name.lower() else torch.bfloat16
    model, tok = load_model(name, dtype=dtype)
    s = slug(name)
    meta = {"model": name, "hardware": hardware_tag(), "dtype": str(dtype),
            "calibration": "wikitext-2" if calib else "default", "window": WINDOW}

    # 1. triviality
    r = bench_triviality(model, tok, window=WINDOW, sink=SINK, calib_texts=calib)
    r["_meta"] = meta
    dump(r, f"{s}_triviality")
    _plot_triviality(r, os.path.join(FIG, f"{s}_triviality"))

    # 2. ppl-vs-saving Pareto (echo / random / anti)
    ks = [0, 2, 4, 6] if quick else None
    r = bench_ppl_pareto(model, tok, ks=ks, window=WINDOW, sink=SINK, calib_texts=calib)
    r["_meta"] = meta
    dump(r, f"{s}_pareto")
    _plot_pareto(r, os.path.join(FIG, f"{s}_pareto"))

    # 3. saving vs context (measured) -- Qwen only (modern GQA model)
    if "qwen" in name.lower():
        sch = calibrate(model, tok, target_saving=0.4, window=WINDOW, sink=SINK,
                        texts=calib)
        lengths = (256, 512, 1024) if quick else (256, 512, 1024, 2048)
        r = bench_saving_vs_context(model, tok, sch, lengths=lengths)
        r["_meta"] = meta; r["local_layers"] = sorted(sch.local_layers)
        dump(r, f"{s}_saving_vs_context")
        plot_saving_vs_context(r, os.path.join(FIG, f"{s}_saving_vs_context"),
                               f"KV saving vs context ({name})")

        # 4. needle survival
        r = bench_needle_survival(model, tok, window=WINDOW, sink=SINK,
                                  calib_texts=calib,
                                  ks=[1, 2, 3, 4] if quick else [1, 2, 3, 4, 6, 8])
        r["_meta"] = meta
        dump(r, f"{s}_needle")
        plot_needle(r, os.path.join(FIG, f"{s}_needle"),
                    f"Retrieval-needle survival under localization ({name})")

        # 5. chunked-prefill peak memory (CUDA)
        if torch.cuda.is_available():
            peaks = []
            for ctx in ([1024, 2048] if quick else [2048, 4096]):
                try:
                    pr = bench_chunked_peak(model, tok, sch, context=ctx,
                                            prefill_chunk=256)
                    pr["_meta"] = meta; peaks.append(pr)
                    print(f"  peak T={ctx}: full {pr.get('full_peak_mb',0):.0f}MB "
                          f"echo {pr.get('echo_peak_mb',0):.0f}MB "
                          f"(-{100*pr.get('peak_reduction',0):.0f}%)")
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    peaks.append({"context": ctx, "oom_full_baseline": True,
                                  "_meta": meta})
                    print(f"  peak T={ctx}: full baseline OOM (echo would run)")
            dump({"runs": peaks}, f"{s}_chunked_peak")
            plot_peak(peaks, os.path.join(FIG, f"{s}_chunked_peak"))

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="gpt2,Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    print("Loading WikiText-2 calibration corpus (X9/X10/X11 protocol)...")
    calib = wikitext_paragraphs()
    for name in args.models.split(","):
        run_model(name.strip(), calib, quick=args.quick)
    print("\nDone. JSON in results/, figures in figures/.")


if __name__ == "__main__":
    main()
