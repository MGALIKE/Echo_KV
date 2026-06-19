"""Cross-check the manuscript's headline numbers against the logged JSON results.

This is the automated half of the integrity pass: every assertion below ties a number
that appears in the paper / README / BENCHMARKS.md to the exact field in a results JSON.
Run it after any change to the numbers. Exits non-zero on the first mismatch.

    python check_claims.py
"""
import json
import os

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
ok = []


def check(name, got, want, tol=0.01):
    good = abs(got - want) <= tol
    ok.append(good)
    print(f"  [{'OK ' if good else 'XX '}] {name}: got {got}, claim {want}")


def load(f):
    return json.load(open(os.path.join(R, f)))

# --- saving vs context: 37% -> 45%, asymptote 46%, 11/24 local ---
s = load("Qwen2_5-0_5B-Instruct_saving_vs_context.json")
check("saving@256 = 37%", round(100 * s["points"][0]["kv_saving"]), 37, tol=1)
check("saving@2048 = 45%", round(100 * s["points"][-1]["kv_saving"]), 45, tol=1)
check("asymptote = 46%", round(100 * s["asymptote"]), 46, tol=1)
check("n_local = 11", s["n_local"], 11, tol=0)
check("num_layers = 24", s["num_layers"], 24, tol=0)

# --- needle: echo 0.92-0.97 through K=4, random mean 0.41 / min 0.07, anti 0.66 @K=4 ---
n = load("Qwen2_5-0_5B-Instruct_needle.json")
by_k = {p["k"]: p for p in n["points"]}
check("needle retrievable = 120", n["retrievable"], 120, tol=0)
check("needle echo@K1 ~0.92", by_k[1]["echo"], 0.92, tol=0.02)
check("needle echo@K4 ~0.97", by_k[4]["echo"], 0.97, tol=0.02)
check("needle random_mean@K4 ~0.41", by_k[4]["random_mean"], 0.41, tol=0.03)
check("needle random_min@K4 ~0.07", by_k[4]["random_min"], 0.07, tol=0.02)

# --- GPT-2 pareto: echo<random<anti at 24% (k=3) and 87% (k=11) ---
g = load("gpt2_pareto.json")
gp = {p["k"]: p for p in g["points"]}
full = g["full_ppl"]
check("GPT-2 echo gap@k3 ~0.4", gp[3]["ppl_gap"], 0.4, tol=0.15)
check("GPT-2 random gap@k3 ~2.5", gp[3]["random_ppl"] - full, 2.5, tol=0.3)
check("GPT-2 anti gap@k3 ~9.6", gp[3]["anti_ppl"] - full, 9.6, tol=0.5)
check("GPT-2 echo gap@k11 ~16", gp[11]["ppl_gap"], 16.4, tol=1.0)
check("GPT-2 random gap@k11 ~28", gp[11]["random_ppl"] - full, 27.9, tol=1.5)

# --- Qwen pareto: echo +1.8 @24% (k=6); echo < anti at k=6 ---
qp = {p["k"]: p for p in load("Qwen2_5-0_5B-Instruct_pareto.json")["points"]}
check("Qwen echo gap@k6 ~1.8", qp[6]["ppl_gap"], 1.75, tol=0.2)
check("Qwen echo < anti @k6", 1 if qp[6]["echo_ppl"] < qp[6]["anti_ppl"] else 0, 1, tol=0)

# --- X11 scaling: GPT-2 echo+5.1 vs random+67 @1024 (K=5) ---
x = load("x11_scaling.json")
gpt = [r for r in x["runs"] if "gpt2" in r["model"]][0]
row1024 = [r for r in gpt["rows"] if r["T"] == 1024][0]["K"]["5"]
check("X11 GPT-2 echo gap@1024 ~5.1", row1024["echo_gap"], 5.1, tol=0.3)
check("X11 GPT-2 random gap@1024 ~67", row1024["random_gap"], 67.4, tol=2.0)

# --- chunked peak: 33%@2k, 65%@4k (3466->1199) ---
cp = {r["context"]: r for r in load("Qwen2_5-0_5B-Instruct_chunked_peak.json")["runs"]
      if "full_peak_mb" in r}
check("peak drop@2k = 33%", round(100 * cp[2048]["peak_reduction"]), 33, tol=1)
check("peak drop@4k = 65%", round(100 * cp[4096]["peak_reduction"]), 65, tol=1)
check("peak full@4k ~3466MB", round(cp[4096]["full_peak_mb"]), 3466, tol=5)
check("peak echo@4k ~1199MB", round(cp[4096]["echo_peak_mb"]), 1199, tol=5)

# --- multimodal SmolVLM: full 100, recency 25, uniform->100@b8, scored@32, digit 20 ---
m = load("smolvlm_multimodal.json")
col = m["needles"]["colour"]
check("MM colour full = 100%", round(100 * col["full_acc"]), 100, tol=0)
check("MM colour all-local recency = 25%", round(100 * col["alllocal_recency_acc"]), 25, tol=0)
check("MM colour deploy recency = 75%", round(100 * col["deploy_recency_acc"]), 75, tol=0)
b8 = [s for s in col["anchor_sweep"] if s["budget"] == 8][0]
check("MM uniform@b8 = 100%", round(100 * b8["uniform_acc"]), 100, tol=0)
check("MM scored@b8 = 50% (< uniform)", round(100 * b8["score_acc"]), 50, tol=0)
b32 = [s for s in col["anchor_sweep"] if s["budget"] == 32][0]
check("MM scored reaches 100 @b32", round(100 * b32["score_acc"]), 100, tol=0)
check("MM digit ceiling = 20% (INVALID)", round(100 * m["needles"]["digit"]["full_acc"]), 20, tol=0)

# --- Gemma overlay: native global, echo-local 100% / second-run 83% ---
ov = load("gemma_layer_overlay.json")
check("Gemma echo-local->native-sliding = 100%",
      round(100 * ov["agreement_echo_local_is_native_sliding"]), 100, tol=0)
check("Gemma reverse (echo-global) = 37%",
      round(100 * ov["agreement_echo_global_is_native_global"]), 37, tol=0)
check("Gemma second-run agreement = 83%",
      round(100 * ov["second_run"]["agreement_echo_local_is_native_sliding"]), 83, tol=0)

print(f"\n{sum(ok)}/{len(ok)} claims verified against logged JSON.")
raise SystemExit(0 if all(ok) else 1)
