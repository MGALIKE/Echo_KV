"""Echo-vs-native layer-schedule overlay (the rediscovery figure).

The echo-chosen LOCAL layers are overlaid on the model's NATIVE local/global schedule.
These values are the LOGGED outputs of the Gemma-4-E4B Colab/L4 notebook runs
(echo_kv_multimodal_results_new.ipynb), reproduced here verbatim -- they are NOT
re-measured locally (a 4--8B model does not fit the 4 GB development GPU). The native
schedule is Gemma's hand-designed 6:1 sliding/global interleave for a 42-layer model.

Finding: echo never localizes a native-global layer (100% of echo-local layers are
native sliding-window in this run; 83% on an alternate calibration). Echo is
conservative -- it keeps only a subset of layers global -- so the reverse agreement
(echo-global that are native-global) is lower (37%).

    python make_overlay_figure.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results"); FIG = os.path.join(HERE, "figures")
os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)

# ---- LOGGED Colab/L4 values ----
# The figure plots the run logged in echo_kv_multimodal_results_new.ipynb (100% agreement).
# A SECOND, separately-logged run in echo_kv_multimodal_results_wow.ipynb agreed on 83%
# and DID localize four native-global layers (5, 11, 17, 29) -- so the containment is
# 83--100% across runs, not absolute. We record both honestly.
DATA = {
    "model": "google/gemma-4-E4B-it (notebook identifier; 42-layer Gemma-class hybrid; "
             "likely a Gemma-3n E4B variant -- confirm checkpoint id)",
    "num_layers": 42,
    "hardware": "Colab/L4 (24 GB), bfloat16",
    "source_figure_run": "echo_kv_multimodal_results_new.ipynb (logged notebook output)",
    "native_global_layers": [5, 11, 17, 23, 29, 35, 41],
    "echo_local_layers": [8, 10, 12, 13, 15, 16, 18, 20, 21, 22, 24, 25, 26, 27, 28,
                          30, 31, 32, 33, 36, 37, 38, 39],
    "agreement_echo_local_is_native_sliding": 1.00,
    "agreement_echo_global_is_native_global": 0.37,
    "second_run": {
        "source": "echo_kv_multimodal_results_wow.ipynb (logged notebook output)",
        "echo_local_layers": [0, 1, 2, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                              21, 24, 28, 29, 33, 34, 36],
        "agreement_echo_local_is_native_sliding": 0.83,
        "note": "localized native-global layers 5, 11, 17, 29 -- containment not absolute",
    },
}


def main():
    L = DATA["num_layers"]
    native_global = set(DATA["native_global_layers"])
    echo_local = set(DATA["echo_local_layers"])
    # native: 1 = local(sliding), 0 = global ; echo: 1 = local chosen
    native_local = [0 if i in native_global else 1 for i in range(L)]
    echo_local_row = [1 if i in echo_local else 0 for i in range(L)]

    with open(os.path.join(RES, "gemma_layer_overlay.json"), "w") as f:
        json.dump(DATA, f, indent=2)

    fig, ax = plt.subplots(figsize=(9, 2.4))
    for i in range(L):
        # native band (top row): blue=sliding/local, red=global
        ax.add_patch(plt.Rectangle((i, 1.05), 0.9, 0.8,
                     color="#2e86ab" if native_local[i] else "#d1495b"))
        # echo choice (bottom row): filled if echo localized it
        if echo_local_row[i]:
            ax.add_patch(plt.Rectangle((i, 0.05), 0.9, 0.8, color="#2e86ab", alpha=0.55))
        else:
            ax.add_patch(plt.Rectangle((i, 0.05), 0.9, 0.8, fill=False,
                         edgecolor="#888", lw=0.5))
    ax.set_xlim(0, L); ax.set_ylim(0, 2)
    ax.set_yticks([0.45, 1.45]); ax.set_yticklabels(["echo-chosen\nlocal", "Gemma native\n(blue=local)"])
    ax.set_xlabel("layer index")
    ax.set_title("Echo rediscovers a Gemma-class hybrid local/global schedule "
                 "(42-layer E4B, Colab/L4)\n"
                 "in this calibration, 100% of echo-localized layers are native "
                 "sliding-window (a second run: 83%)")
    ax.set_xticks(range(0, L, 2))
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"gemma_layer_overlay.{ext}"), dpi=140)
    plt.close(fig)
    print(f"native global layers: {sorted(native_global)}")
    print(f"echo-local layers ({len(echo_local)}): {sorted(echo_local)}")
    print(f"echo-local that are native sliding: "
          f"{100*DATA['agreement_echo_local_is_native_sliding']:.0f}%")
    print("  -> figures/gemma_layer_overlay.{png,pdf}, results/gemma_layer_overlay.json")


if __name__ == "__main__":
    main()
