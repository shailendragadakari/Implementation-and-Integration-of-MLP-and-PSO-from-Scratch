# Experiment helpers which is used by experiments.ipynb and is also runnable as a script.
# 20% of the data is a TEST set that is only used for final reporting.
# The other 80% is split again into FIT (60% of total) and VALIDATION (20% of total).
# Hyperparameters train PSO on FIT and are compared on VALIDATION, so that no tuning decision
# is made by looking at the test set.
# Every setting is repeated over several PSO seeds.
# Preprocessing (z-scoring) is fitted on the data PSO trains on and nothing else.

import os
import time
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path

import numpy as np

from training import train_pso_mlp
from utils.data import Standardiser, load_dataset, train_test_split
from utils.metrics import classification_metrics, probabilities_to_labels

DATA_PATH = Path(__file__).parent / "data" / "data_banknote_authentication.txt"

DEFAULT_CONFIG = {
    "hidden": [8], "activations": ["tanh"],
    "strategy": "nearest", "k": 3,
    "particles": 30, "iterations": 100,
    "alpha": 0.7, "beta": 1.5, "delta": 1.5, "gamma": 0.0, "epsilon": 1.0,
}


# --------------------------------------------------------------------------- data
def make_data(split_seed=42):
    # Return a dict with FIT/VAL arrays (for tuning) and TRAIN/TEST arrays (for final).
    X, y = load_dataset(DATA_PATH)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, 0.2, split_seed)
    X_fit, X_val, y_fit, y_val = train_test_split(X_tr, y_tr, 0.25, split_seed + 1)
    s_fit = Standardiser().fit(X_fit)
    s_tr = Standardiser().fit(X_tr)
    return {
        "fit": (s_fit.transform(X_fit), y_fit), "val": (s_fit.transform(X_val), y_val),
        "train": (s_tr.transform(X_tr), y_tr), "test": (s_tr.transform(X_te), y_te),
    }


# --------------------------------------------------------------------------- one run
def run_trial(config, seed, train, evaluate):
    # Train an MLP with PSO on `train`, report metrics on `evaluate`. Returns a flat dict.
    X_tr, y_tr = train
    X_ev, y_ev = evaluate
    t0 = time.perf_counter()

    # Same build-and-train path as main.py / the GUI (adds optional extensions via config keys).
    network, loss, pop = train_pso_mlp(X_tr, y_tr, {**config, "seed": seed})
    best_fit = pop.global_best_fitness

    probs = network.predict(X_ev)
    m = classification_metrics(y_ev, probabilities_to_labels(probs))
    hist = np.array(pop.history["global_best"])
    within5 = np.where(hist <= hist[-1] * 1.05 + 1e-12)[0]
    return {
        "seed": seed, "dimension": pop.dimension,
        "activations_used": "|".join(type(l.activation).__name__.lower() for l in network.layers[:-1]),
        "train_loss": best_fit,
        "eval_loss": loss.compute(y_ev, probs),
        "eval_accuracy": m["accuracy"], "eval_precision": m["precision"],
        "eval_recall": m["recall"], "eval_f1": m["f1"],
        "iters_to_within_5pct": int(within5[0]),
        "evaluations": pop.n_evaluations, "seconds": time.perf_counter() - t0,
        "history": hist, "mean_history": np.array(pop.history["mean_fitness"]),
    }


def _worker(args):
    return run_trial(*args)


# --------------------------------------------------------------------------- experiments
def make_configs(param, values, base=None):
    # One config per value of a single hyperparameter. Returns {label: config}.
    base = base or DEFAULT_CONFIG
    out = {}
    for v in values:
        c = deepcopy(base)
        c[param] = v
        # keep activations consistent when the hidden-layer list changes
        if param == "hidden":
            c["activations"] = [base["activations"][0]] * len(v)
        out[str(v)] = c
    return out


def run_experiment(configs, seeds, data, mode="val", n_jobs=None):
    # Run each labelled config for each seed.
    #
    #     mode='val'  : train on FIT, evaluate on VALIDATION (use for tuning)
    #     mode='test' : train on full TRAIN, evaluate on TEST (use once, for final numbers)
    #     Returns a list of result dicts (each includes 'label' and the config values).
    train, ev = (data["fit"], data["val"]) if mode == "val" else (data["train"], data["test"])
    jobs, labels = [], []
    for label, cfg in configs.items():
        for s in seeds:
            jobs.append((cfg, s, train, ev))
            labels.append((label, cfg))
    n_jobs = n_jobs or max(1, (os.cpu_count() or 2) - 1)
    if n_jobs == 1:
        results = [_worker(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            results = list(ex.map(_worker, jobs, chunksize=4))
    for r, (label, cfg) in zip(results, labels):
        r["label"] = label
        r.update({f"cfg_{k}": (str(v) if isinstance(v, list) else v) for k, v in cfg.items()})
    return results


# --------------------------------------------------------------------------- extra experiments
def make_budget_configs(budget=500, particle_counts=(10, 20, 25, 50, 100), base=None):
    # Same evaluation budget, different swarm size / iteration split.
    #     evaluations = particles * (iterations + 1)  (the +1 is the initial evaluation),
    #     so iterations = budget / particles - 1 keeps every config at the same total.
    base = base or DEFAULT_CONFIG
    out = {}
    for n in particle_counts:
        if budget % n:
            continue
        c = deepcopy(base)
        c["particles"], c["iterations"] = n, budget // n - 1
        c["k"] = min(c["k"], n - 1)
        out[f"{n}x{c['iterations']}"] = c
    return out


def make_grid_configs(param_a, values_a, param_b, values_b, base=None):
    # One config for every combination of two hyperparameters. Labels look like "a=0.5,b=1.0".
    base = base or DEFAULT_CONFIG
    out = {}
    for a in values_a:
        for b in values_b:
            c = deepcopy(base)
            c[param_a], c[param_b] = a, b
            out[f"{param_a}={a},{param_b}={b}"] = c
    return out


def compare_configs(name_a, name_b, results, metric="eval_accuracy"):
    # Paired sign-flip permutation test between two labelled configs in `results` (same seeds).
    #     Returns a dict row: mean(a) - mean(b) and the p-value.
    df = to_frame(results)
    a = df[df.label == name_a].sort_values("seed")[metric].values
    b = df[df.label == name_b].sort_values("seed")[metric].values
    diff, p = paired_permutation_test(a, b)
    return {"a": name_a, "b": name_b, "metric": metric, "mean_a": float(a.mean()),
            "mean_b": float(b.mean()), "mean_diff_a_minus_b": diff, "p_value": p}


def extra_experiment_configs():
    # All the extra experiments, as {name: {label: config}}. The names are also the CSV file names.
    #     budget_split_*     : same number of evaluations, different swarm size / iterations
    #     pso_beta_delta_grid: cognitive (beta) x informant (delta) balance
    #     pso_boundary       : boundary-handling modes (extension)
    #     activation_genes   : fixed tanh / sigmoid / relu vs PSO choosing the activations (extension)
    #     pso_inertia_schedule: constant vs linearly decreasing inertia (extension)
    #     tuned_candidates   : best single-sweep settings combined, vs the default
    #     informants_k3      : random vs nearest-neighbour at k=3 (for the paired test)
    out = {
        "budget_split_500": make_budget_configs(500),
        "budget_split_3000": make_budget_configs(3000, (10, 20, 30, 50, 100)),
        "pso_beta_delta_grid": make_grid_configs("beta", [0.5, 1.0, 1.5, 2.0], "delta", [0.5, 1.0, 1.5, 2.0]),
    }
    boundary = {"none": dict(DEFAULT_CONFIG)}
    for B in (2.0, 5.0):
        for mode in ("clamp", "reflect", "random"):
            c = deepcopy(DEFAULT_CONFIG); c["boundary"], c["bound"] = mode, B
            boundary[f"{mode}-B{B:g}"] = c
    out["pso_boundary"] = boundary

    acts = {}
    for name in ("tanh", "sigmoid", "relu"):
        c = deepcopy(DEFAULT_CONFIG); c["activations"] = [name]; acts[name] = c
    c = deepcopy(DEFAULT_CONFIG); c["evolve_activations"] = True
    acts["evolved"] = c
    out["activation_genes"] = acts

    sched = {"constant-0.7": dict(DEFAULT_CONFIG)}
    for a0, a1 in ((0.9, 0.4), (0.8, 0.5), (0.9, 0.2)):
        c = deepcopy(DEFAULT_CONFIG)
        c["alpha"], c["inertia_schedule"], c["inertia_end"] = a0, "linear", a1
        sched[f"linear-{a0}to{a1}"] = c
    out["pso_inertia_schedule"] = sched

    tuned_same = deepcopy(DEFAULT_CONFIG)
    tuned_same.update(beta=1.0, delta=1.0, k=10)
    tuned_big = deepcopy(tuned_same); tuned_big.update(particles=50, iterations=200)
    out["tuned_candidates"] = {"default": dict(DEFAULT_CONFIG),
                               "tuned-same-budget": tuned_same,
                               "tuned-50x200": tuned_big}
    ks = {}
    for strat in ("random", "nearest"):
        c = deepcopy(DEFAULT_CONFIG); c["strategy"] = strat; c["k"] = 3
        ks[f"{strat}-k3"] = c
    out["informants_k3"] = ks
    return out


def significance_table(all_results):
    # Paired sign-flip tests for the key comparisons. `all_results` is {name: results list}
    #     as produced by running extra_experiment_configs(). Each test pairs runs by seed.
    pairs = [
        ("informants_k3", "nearest-k3", "random-k3"),
        ("budget_split_500", "50x9", "10x49"),
        ("budget_split_3000", "100x29", "10x299"),
        ("pso_boundary", "clamp-B5", "none"), ("pso_boundary", "reflect-B5", "none"),
        ("pso_boundary", "random-B5", "none"), ("pso_boundary", "clamp-B2", "none"),
        ("pso_boundary", "reflect-B2", "none"), ("pso_boundary", "random-B2", "none"),
        ("activation_genes", "relu", "tanh"), ("activation_genes", "sigmoid", "tanh"),
        ("activation_genes", "evolved", "tanh"),
        ("pso_inertia_schedule", "linear-0.9to0.4", "constant-0.7"),
        ("pso_inertia_schedule", "linear-0.8to0.5", "constant-0.7"),
        ("pso_inertia_schedule", "linear-0.9to0.2", "constant-0.7"),
        ("tuned_candidates", "tuned-same-budget", "default"),
        ("tuned_candidates", "tuned-50x200", "default"),
    ]
    rows = []
    for name, a, b in pairs:
        if name not in all_results:
            continue
        labels = {r["label"] for r in all_results[name]}
        if a not in labels or b not in labels:
            continue
        for metric in ("eval_accuracy", "eval_loss"):
            row = compare_configs(a, b, all_results[name], metric)
            row["experiment"] = name
            rows.append(row)
    import pandas as pd
    return pd.DataFrame(rows)[["experiment", "a", "b", "metric", "mean_a", "mean_b",
                               "mean_diff_a_minus_b", "p_value"]]


# --------------------------------------------------------------------------- analysis
def to_frame(results):
    import pandas as pd
    return pd.DataFrame([{k: v for k, v in r.items() if k not in ("history", "mean_history")}
                         for r in results])


def summarise(results, order=None):
    df = to_frame(results)
    cols = ["train_loss", "eval_loss", "eval_accuracy", "eval_f1", "iters_to_within_5pct", "seconds"]
    g = df.groupby("label", sort=False)[cols].agg(["mean", "std"])
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    return g.loc[order] if order else g


def histories_by_label(results, key="history"):
    # {label: array (n_seeds, n_iters+1)} — all seeds for a label share a length.
    out = {}
    for r in results:
        out.setdefault(r["label"], []).append(r[key])
    return {k: np.stack(v) for k, v in out.items()}


def paired_permutation_test(a, b, n_perm=20000, seed=0):
    # Two-sided sign-flip permutation test on paired differences a-b (same seeds).
    #     Returns (mean difference, p-value). NumPy only, no distributional assumptions.
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    obs = abs(d.mean())
    signs = rng.choice([-1.0, 1.0], size=(n_perm, d.size))
    perm = np.abs((signs * d).mean(axis=1))
    return float(d.mean()), float((np.sum(perm >= obs - 1e-15) + 1) / (n_perm + 1))


# --------------------------------------------------------------------------- overfitting checks
def train_vs_test(config, seeds, data):
    # Overfitting check: train the SAME network on TRAIN and score it on TRAIN and on TEST.
    # Overfitting would show as high train accuracy but clearly lower test accuracy.
    # Runs are repeatable, so training twice with one seed gives the same weights both times.
    rows = []
    for s in seeds:
        on_test = run_trial(config, s, data["train"], data["test"])
        on_train = run_trial(config, s, data["train"], data["train"])
        rows.append({"seed": s,
                     "train_accuracy": on_train["eval_accuracy"], "test_accuracy": on_test["eval_accuracy"],
                     "train_loss": on_test["train_loss"], "test_loss": on_test["eval_loss"]})
    return rows


def duplicate_overlap(split_seed=42):
    # Count test rows whose 4 feature values also appear exactly in the training rows.
    # The public banknote file contains duplicate rows, so this can make test accuracy a bit optimistic.
    X, y = load_dataset(DATA_PATH)
    X_tr, X_te, _, _ = train_test_split(X, y, 0.2, split_seed)
    train_rows = {tuple(r) for r in X_tr}
    n_dup = sum(tuple(r) in train_rows for r in X_te)
    return {"duplicate_rows_in_file": int(len(X) - len(np.unique(X, axis=0))),
            "test_rows_also_in_train": int(n_dup), "test_rows": int(len(X_te))}


def split_robustness(config, split_seeds, seeds):
    # Repeat the final train/test experiment on several different random splits.
    # This shows how much the test accuracy depends on which rows landed in the test set.
    rows = []
    for ss in split_seeds:
        d = make_data(ss)
        for s in seeds:
            r = run_trial(config, s, d["train"], d["test"])
            rows.append({"split_seed": ss, "seed": s, "test_accuracy": r["eval_accuracy"],
                         "test_loss": r["eval_loss"]})
    return rows


# --------------------------------------------------------------------------- plotting
def plot_convergence(histories, title="PSO convergence", logy=True, ax=None):
    import matplotlib.pyplot as plt
    ax = ax or plt.subplots(figsize=(7, 4.2))[1]
    for label, h in histories.items():
        mean, std = h.mean(axis=0), h.std(axis=0)
        x = np.arange(len(mean))
        ax.plot(x, mean, label=label)
        ax.fill_between(x, np.maximum(mean - std, 1e-9), mean + std, alpha=0.15)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Global-best training loss (BCE)")
    ax.set_title(title); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    return ax


def plot_metric_vs_param(results, metric="eval_accuracy", xlabel="", title="", ax=None):
    import matplotlib.pyplot as plt
    df = to_frame(results)
    labels = list(dict.fromkeys(df["label"]))
    data = [df.loc[df["label"] == l, metric].values for l in labels]
    ax = ax or plt.subplots(figsize=(7, 4))[1]
    ax.boxplot(data, tick_labels=labels, showmeans=True)
    ax.set_xlabel(xlabel); ax.set_ylabel(metric); ax.set_title(title); ax.grid(alpha=0.3)
    return ax


if __name__ == "__main__":
    d = make_data()
    t = time.perf_counter()
    res = run_experiment({"default": DEFAULT_CONFIG}, seeds=range(5), data=d)
    print(summarise(res).T)
    print(f"{time.perf_counter() - t:.1f}s")
