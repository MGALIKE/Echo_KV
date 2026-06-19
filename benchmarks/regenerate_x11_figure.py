"""Regenerate the X11 long-context scaling figure from its logged run.

X11 (experiments/results/x11_locality_scaling.json) is a LOGGED local-4GB run: calibrate
the echo layer ranking once at T=256, freeze a K-local schedule, and re-evaluate
perplexity at growing context without recalibrating. It is the cleanest evidence that the
echo CLASSIFICATION matters more at long context -- a frozen echo schedule's gap to full
stays bounded while an equal-size RANDOM schedule diverges. Window 48 (matches the
benchmark protocol). We reuse the logged JSON and redraw the figure here for the paper.

    python regenerate_x11_figure.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "..", "experiments", "results", "x11_locality_scaling.json")
RES = os.path.join(HERE, "results"); FIG = os.path.join(HERE, "figures")
os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)


def main():
    d = json.load(open(SRC))
    # copy the logged source into the benchmark results for traceability
    d["_source"] = "experiments/results/x11_locality_scaling.json (logged local 4GB run)"
    with open(os.path.join(RES, "x11_scaling.json"), "w") as f:
        json.dump(d, f, indent=2)

    fig, axes = plt.subplots(1, len(d["runs"]), figsize=(5.2 * len(d["runs"]), 4),
                             squeeze=False)
    for ax, run in zip(axes[0], d["runs"]):
        name = run["model"].split("/")[-1]
        Ks = run["Ks"] if "Ks" in run else sorted(run["rows"][0]["K"].keys(), key=int)
        K = str(max(int(k) for k in run["rows"][0]["K"]))   # the aggressive schedule
        Ts = [r["T"] for r in run["rows"]]
        echo_gap = [r["K"][K]["echo_gap"] for r in run["rows"]]
        rand_gap = [r["K"][K]["random_gap"] for r in run["rows"]]
        ax.plot(Ts, echo_gap, "o-", c="#2e86ab", label=f"echo (K={K})")
        ax.plot(Ts, rand_gap, "s--", c="#d1495b", label=f"random (K={K})")
        ax.set_xlabel("context length T"); ax.set_ylabel("perplexity gap vs full")
        ax.set_title(f"{name}: frozen schedule, growing context")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        print(f"{name}  K={K}: "
              + ", ".join(f"T={r['T']} echo+{r['K'][K]['echo_gap']:.1f} "
                          f"random+{r['K'][K]['random_gap']:.1f}" for r in run["rows"]))
    fig.suptitle("X11: a frozen echo schedule holds as context grows; "
                 "an equal-size random schedule diverges (local 4 GB)", fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"x11_scaling.{ext}"), dpi=140)
    plt.close(fig)
    print("  -> figures/x11_scaling.{png,pdf}, results/x11_scaling.json")


if __name__ == "__main__":
    main()
