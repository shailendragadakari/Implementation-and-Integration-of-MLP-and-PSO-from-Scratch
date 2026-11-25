# Helpers for the GUI: linking particles x iterations to a fixed number of solution
# evaluations, and recording runs (parameters + results) in a CSV file.
# Kept free of Streamlit so they can be tested on their own.
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

# Column order of the run log. One row = one trained network.
COLUMNS = [
    "timestamp", "split_seed", "hidden_layers", "activations", "n_parameters",
    "informant_strategy", "k_informants", "particles", "iterations",
    "target_evaluations", "fitness_calls", "alpha", "beta", "delta", "gamma", "epsilon",
    "pso_seed", "best_training_loss", "test_loss", "accuracy", "precision", "recall", "f1",
    "tp", "tn", "fp", "fn",
]


# ------------------------------------------------------------- evaluations <-> particles x iterations
def valid_splits(evaluations, min_particles=5, max_particles=200, min_iterations=2, max_iterations=1000):
    # All (particles, iterations) pairs with particles * iterations == evaluations exactly,
    # inside the allowed ranges, sorted by particles. Only exact divisors are offered so the
    # product always equals the chosen number of evaluations.
    n = int(evaluations)
    out = []
    for p in range(min_particles, min(max_particles, n) + 1):
        if n % p == 0:
            it = n // p
            if min_iterations <= it <= max_iterations:
                out.append((p, it))
    return out


def suggest_evaluations(evaluations, count=4, min_choices=3, search=2000):
    # Nearby evaluation numbers that do have at least `min_choices` exact splits,
    # closest first. Used when the typed number has too few divisors (for example a prime).
    n = int(evaluations)
    found = []
    for d in range(1, search + 1):
        for c in (n - d, n + d):
            if c >= 1 and len(valid_splits(c)) >= min_choices:
                found.append(c)
        if len(found) >= count:
            break
    return sorted(found[:count])


# ------------------------------------------------------------- run records
def make_record(config, metrics, best_loss, test_loss, fitness_calls, split_seed, n_parameters,
                timestamp=None):
    # Flatten one finished run (the parameters it was trained with and its results) into a dict
    # with exactly the COLUMNS above.
    return {
        "timestamp": timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "split_seed": int(split_seed),
        "hidden_layers": "|".join(str(int(h)) for h in config["hidden"]),
        "activations": "|".join(config["activations"]),
        "n_parameters": int(n_parameters),
        "informant_strategy": config["strategy"],
        "k_informants": int(config["k"]),
        "particles": int(config["particles"]),
        "iterations": int(config["iterations"]),
        "target_evaluations": int(config["particles"]) * int(config["iterations"]),
        "fitness_calls": int(fitness_calls),
        "alpha": config["alpha"], "beta": config["beta"], "delta": config["delta"],
        "gamma": config["gamma"], "epsilon": config["epsilon"],
        "pso_seed": int(config["seed"]),
        "best_training_loss": float(best_loss), "test_loss": float(test_loss),
        "accuracy": float(metrics["accuracy"]), "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]), "f1": float(metrics["f1"]),
        "tp": int(metrics["tp"]), "tn": int(metrics["tn"]),
        "fp": int(metrics["fp"]), "fn": int(metrics["fn"]),
    }


def append_record(path, record):
    # Add one row to the CSV at `path` (created with a header if missing). If the file already
    # has other columns, they are kept and the new columns are added. Returns the row count.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([record], columns=COLUMNS)
    if path.exists() and path.stat().st_size > 0:
        old = pd.read_csv(path)
        df = pd.concat([old, new], ignore_index=True)
    else:
        df = new
    df.to_csv(path, index=False)
    return len(df)


def read_log(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(path)


# ------------------------------------------------------------- file names
def clean_csv_filename(name, default="gui_runs.csv"):
    # Make a safe download file name: no folders, no odd characters, always ends in .csv.
    name = (name or "").strip().replace("\\", "/").split("/")[-1]
    name = re.sub(r"[^\w\-. ]", "_", name).strip(" .")
    if not name:
        name = default
    if not name.lower().endswith(".csv"):
        name += ".csv"
    return name
