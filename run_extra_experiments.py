# Runs the extra experiments (budget split, beta x delta grid, boundary handling, activation
# genes, inertia schedule, tuned candidates, paired significance tests) on the VALIDATION set
# and writes CSVs (and PNG figures) into results/.
#
#     python run_extra_experiments.py            # 10 seeds
#     N_SEEDS=3 python run_extra_experiments.py  # quick dry run
#
# The test set is never used here. The notebook (experiments.ipynb, section 5) runs the same
# experiments; this script is the command-line version.
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments import (extra_experiment_configs, histories_by_label, make_data, plot_convergence,
                         plot_metric_vs_param, run_experiment, significance_table, summarise, to_frame)

N_SEEDS = int(os.environ.get("N_SEEDS", 10))
SEEDS = range(N_SEEDS)


def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    data = make_data()
    all_results = {}
    for name, configs in extra_experiment_configs().items():
        t0 = time.perf_counter()
        res = run_experiment(configs, SEEDS, data, mode="val")
        all_results[name] = res
        to_frame(res).to_csv(f"results/{name}.csv", index=False)
        print(f"\n== {name} ({time.perf_counter() - t0:.1f}s)")
        s = summarise(res)
        print(s[["train_loss_mean", "eval_accuracy_mean", "eval_accuracy_std",
                 "eval_f1_mean", "iters_to_within_5pct_mean"]].to_string(float_format=lambda v: f"{v:.4f}"))
        if name == "pso_beta_delta_grid":
            df = to_frame(res)
            betas = sorted(df.cfg_beta.unique()); deltas = sorted(df.cfg_delta.unique())
            M = np.array([[df[(df.cfg_beta == b) & (df.cfg_delta == d)].eval_accuracy.mean()
                           for d in deltas] for b in betas])
            fig, ax = plt.subplots(figsize=(5.5, 4.2))
            im = ax.imshow(M, cmap="viridis", origin="lower")
            ax.set_xticks(range(len(deltas)), deltas); ax.set_yticks(range(len(betas)), betas)
            ax.set_xlabel("informant delta"); ax.set_ylabel("cognitive beta")
            for i in range(len(betas)):
                for j in range(len(deltas)):
                    ax.text(j, i, f"{M[i, j]*100:.1f}", ha="center", va="center", color="w", fontsize=8)
            fig.colorbar(im, label="validation accuracy"); ax.set_title("Mean validation accuracy")
            fig.tight_layout(); fig.savefig(f"results/figures/{name}.png", dpi=150); plt.close(fig)
        else:
            fig, axs = plt.subplots(1, 2, figsize=(14, 4.2))
            plot_metric_vs_param(res, "eval_accuracy", "", f"{name}: validation accuracy", axs[0])
            axs[0].tick_params(axis="x", rotation=45)
            plot_convergence(histories_by_label(res), f"{name}: convergence", ax=axs[1])
            fig.tight_layout(); fig.savefig(f"results/figures/{name}.png", dpi=150); plt.close(fig)

    sig = significance_table(all_results)
    sig.to_csv("results/significance.csv", index=False)
    print("\n== significance (paired sign-flip permutation tests, validation set)")
    print(sig.to_string(float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
