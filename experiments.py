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
from adapter import MLPFitnessAdapter
from mlp import MLPBuilder
from pso import NearestNeighborInformantSelection, Population, RandomInformantSelection
from utilities.data import Standardiser, load_dataset, train_test_split
from utilities.metrics import classification_metrics, probabilities_to_labels

DATA_PATH = Path(__file__).parent / "data" / "data_banknote_authentication.txt"

DEFAULT_CONFIG = {
    "hidden": [8], "activations": ["tanh"],
    "strategy": "nearest", "k": 3,
    "particles": 30, "iterations": 100,
    "alpha": 0.7, "beta": 1.5, "delta": 1.5, "gamma": 0.0, "epsilon": 1.0,
}

# data
def make_data(split_seed=42):
    # Return a dictionary with FIT/VAL arrays (for tuning) and TRAIN/TEST arrays.
    X, y = load_dataset(DATA_PATH)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, 0.2, split_seed)
    X_fit, X_val, y_fit, y_val = train_test_split(X_tr, y_tr, 0.25, split_seed + 1)
    s_fit = Standardiser().fit(X_fit)
    s_tr = Standardiser().fit(X_tr)
    return {
        "fit": (s_fit.transform(X_fit), y_fit), "val": (s_fit.transform(X_val), y_val),
        "train": (s_tr.transform(X_tr), y_tr), "test": (s_tr.transform(X_te), y_te),
    }
    
# one run
def run_trail(config, seed, train, evaluate):
    # Train an MLP with PSO on 'train', report metrics to 'evaluate'. Returns a flat dictionary.
    X_tr, y_tr = train
    X_ev, y_ev = evaluate
    t0 = time.perf_counter()

    builder = MLPBuilder(seed=seed).set_dataset(X_tr, y_tr)
    builder.set_configuration({"hidden_layers": config["hidden"],
                               "activations": config["activations"],
                               "task": "binary_classification"})
    network, loss = builder.build()
    fitness = MLPFitnessAdapter(network, loss, X_tr, y_tr)
    strategy = (NearestNeighborInformantSelection() if config["strategy"] == "nearest"
                else RandomInformantSelection(seed=seed))
    pop = Population(
        fitness_function=fitness, informant_strategy=strategy,
        num_particles=config["particles"], num_informants=config["k"],
        num_iterations=config["iterations"], dimension=network.count_parameters(),
        inertia_weight=config["alpha"], cognitive_coefficient=config["beta"],
        social_coefficient=config["delta"], global_coefficient=config["gamma"],
        step_size=config["epsilon"], seed=seed,
    )
    best_pos, best_fit = pop.optimise()
    network.set_parameters(best_pos)

    probs = network.predict(X_ev)
    m = classification_metrics(y_ev, probabilities_to_labels(probs))
    hist = np.array(pop.history["global_best"])
    within5 = np.where(hist <= hist[-1] * 1.05 + 1e-12)[0]
    return {
        "seed": seed, "dimension": network.count_parameters(),
        "train_loss": best_fit,
        "eval_loss": loss.compute(y_ev, probs),
        "eval_accuracy": m["accuracy"], "eval_precision": m["precision"],
        "eval_recall": m["recall"], "eval_f1": m["f1"],
        "iters_to_within_5pct": int(within5[0]),
        "evaluations": pop.n_evaluations, "seconds": time.perf_counter() - t0,
        "history": hist, "mean_history": np.array(pop.history["mean_fitness"]),
    }
    
def _worker(args):
    return run_trail(*args)

# experiments
def make_configs(param, values, base=None):
    # One config per value of a single hyperparameter.
    base = base or DEFAULT_CONFIG
    out = {}
    for v in values:
        c = deepcopy(base)
        c[param] = v
        # keep activations consistent when the hidden layer list changes
        if param == "hidden":
            c["activations"] = [base["activations"][0]] * len(v)
        out[str(v)] = c
    return out

def run_experiment(configs, seeds, data, mode="val", n_jobs=None):
    # Run each labelled config for each seed.
    # mode='val'    : train on FIT, evaluate on VALIDATION (use for tuning)
    # mode='test    : train on full TRAIN, evaluate on TEST (use once, for final numbers)
    # Returns a list of result dictionaries (each includes 'label' and the config values)
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

# analysis
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
    # {label: array (n_seeds, n_iters+1)} - all seeds for a label share a length
    out = {}
    for r in results:
        out.setdefault(r["label"], []).append(r[key])
    return {k: np.stack(v) for k, v in out.items()}

def paired_permutation_test(a, b, n_perm=20000, seed=0):
    # Two-sided sign flip permutation test on paired differences a-b (same seeds).
    # Returns (mean difference, p-value). NumPy only, no distributional assumptions.
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    obs = abs(d.mean())
    signs = rng.choice([-1.0, 1.0], size=(n_perm, d.size))
    perm = np.abs((signs * d).mean(axis=1))
    return float(d.mean()), float((np.sum(perm >= obs - 1e-15) + 1) / (n_perm + 1))

# plotting
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
