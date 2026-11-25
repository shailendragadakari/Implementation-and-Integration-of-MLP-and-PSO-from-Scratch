# Implementation and Integration of MLP and PSO from Scratch

The aim of this project is to implement Artificial Neural Networks (ANNs) and Particle Swarm Optimisation (PSO), and experimentally investigate how PSO can be used to optimise an ANN to carry out a specified task.

Both algorithms are written from scratch in Python with NumPy. No existing ANN or PSO library is used. The network has no back-propagation: PSO searches for the network's weights and biases directly.

## Headline result

The task is the UCI banknote authentication dataset (1,372 notes, 4 features, 2 classes). With the final configuration (one hidden layer of 8 tanh neurons, 49 parameters, 30 particles, 100 iterations, nearest-neighbour informants with k = 10, cognitive β = 1.0, informant δ = 1.0), the PSO-trained network reaches a **mean test accuracy of 100.00% (± 0.00) over 10 seeds**: every seed classifies all 274 test notes correctly (mean test loss 0.0027 ± 0.0037). This configuration was chosen on the validation set by combining the best single-sweep values (see "Combined best settings" below) and the test set was run once with it. The earlier default configuration (k = 3, β = 1.5, δ = 1.5) reached 99.49% ± 0.43 on the same test set; its result files are kept in `results/previous_default_final/`. Because 274 test notes is a small test set and the banknote data is close to separable, 100% on every seed should be read as "at the ceiling of what this test set can show", not as a perfect classifier. See the overfitting checks at the end of this section.

Each run takes well under a second on the machine that ran the notebook.

## Where the results are stored

Everything the notebook produces is in the `results/` folder.

| File | What it contains |
|---|---|
| `results/final_test.csv` | The final configuration run on 10 seeds and scored on the **test set**. This is where the test accuracy figures come from. |
| `results/run_info.json` | The final configuration and the number of seeds used. |
| `results/best_run.json` | Snapshot of the best of the 10 final runs (seed 4, chosen by the lowest test loss because every seed reached 100%), in the same style as `run_info.json`, plus a summary of all 10 seeds. |
| `results/previous_default_final/` | The final-test, run-info, best-run, overfitting and split-robustness files from the earlier run with the default configuration (k = 3, β = 1.5, δ = 1.5), kept for comparison. |
| `results/overfitting_check.csv`, `results/split_robustness.csv` | Train-versus-test accuracy and repeated random splits for the final configuration. |
| `results/pso_alpha.csv`, `pso_beta.csv`, `pso_delta.csv`, `pso_gamma.csv`, `pso_epsilon.csv` | PSO coefficient sweeps (validation set). |
| `results/pso_iterations.csv`, `pso_particles.csv` | Iteration-count and swarm-size sweeps (validation set). |
| `results/mlp_width.csv`, `mlp_depth.csv`, `mlp_activation.csv` | Network-size and activation sweeps (validation set). |
| `results/informants.csv` | Informant strategy (random vs nearest) against the number of informants k (validation set). |
| `results/budget_split_500.csv`, `budget_split_3000.csv`, `pso_beta_delta_grid.csv`, `pso_boundary.csv`, `activation_genes.csv`, `pso_inertia_schedule.csv`, `tuned_candidates.csv`, `informants_k3.csv` | Extra experiments (validation set), see "Extensions and extra experiments". |
| `results/significance.csv` | Paired permutation tests for the key comparisons. |
| `results/figures/` | PNG plots for the extra experiments. |

Each sweep CSV has one row per seed and setting. Useful columns are `eval_accuracy`, `eval_loss`, `eval_f1`, `train_loss`, `iters_to_within_5pct` (how quickly the run converged), `evaluations`, `seconds`, `activations_used` (the hidden activation of each layer actually used) and the `cfg_*` columns that record the configuration used. Empty `cfg_*` values mean that option was not set and the default applied.

### Best-run snapshot

`results/best_run.json` holds the snapshot of the best run: seed 4, which classified all 274 test notes correctly (test loss 0.0001, F1 1.0), took 0.38 s and 3,030 fitness evaluations, and used the final configuration.

"Best" means the highest test accuracy, with ties broken by the lowest test loss. All ten seeds reached 100%, so here the choice is made by test loss alone and the snapshot carries little information. Quote the 10-seed mean above as the headline figure. Timings depend on the machine and are not a meaningful ranking.

### Final test results, per seed

| Seed | Test accuracy | Test loss | F1 | Seconds |
|---|---|---|---|---|
| 0 | 100.00% | 0.0004 | 1.0000 | 0.26 |
| 1 | 100.00% | 0.0006 | 1.0000 | 0.27 |
| 2 | 100.00% | 0.0010 | 1.0000 | 0.27 |
| 3 | 100.00% | 0.0051 | 1.0000 | 0.32 |
| 4 | 100.00% | 0.0001 | 1.0000 | 0.38 |
| 5 | 100.00% | 0.0009 | 1.0000 | 0.26 |
| 6 | 100.00% | 0.0118 | 1.0000 | 0.26 |
| 7 | 100.00% | 0.0052 | 1.0000 | 0.29 |
| 8 | 100.00% | 0.0006 | 1.0000 | 0.28 |
| 9 | 100.00% | 0.0017 | 1.0000 | 0.25 |

**Overfitting checks** (`results/overfitting_check.csv`, `results/split_robustness.csv`, notebook section 5):

- Train accuracy and test accuracy are both 100.00% (mean over the 10 seeds), so there is no train-to-test gap.
- 24 rows of the dataset file are exact duplicates, and 9 of the 274 test rows also appear in the training rows. This can make test accuracy a little optimistic.
- On 5 other random train/test splits (5 seeds each) the mean test accuracy was 99.94%. Splits 1, 2 and 4 gave 100% on every seed, and splits 3 and 5 gave 99.85% (one misclassified note in some runs).
- For comparison, the earlier default configuration scored 99.49% ± 0.43 on the main split.

## How the experiments were run

- **Data split.** A stratified 20% test set (274 notes) is held out and used only for the final evaluation. The other 80% is split again into a fit set (60% of the data) and a validation set (20%).
- **Sweeps** train on the fit set and are compared on the validation set, so no tuning decision looks at the test set.
- **Final run** trains on the full 80% and is scored on the test set.
- **Preprocessing.** Features are standardised (z-score) using statistics from the training data only.
- **Seeds.** Every setting is repeated over 10 PSO seeds.
- **One parameter at a time.** Each sweep changes a single hyperparameter and keeps the others at the default (marked in the tables below). Interactions between hyperparameters were not explored.
- **Loss and fitness.** A particle is a full vector of network weights and biases. Its fitness is the binary cross-entropy on the training data, and PSO minimises it. PSO never sees the test data.
- **Evaluations.** Each run makes particles × (iterations + 1) fitness evaluations, which is 30 × 101 = 3,030 for the default.

## Findings

All figures in this section are validation accuracy (mean over 10 seeds) from the sweep CSVs, unless stated otherwise. Differences smaller than roughly one standard deviation (about 0.5 percentage points near the top of these tables) should not be treated as real. The explanations after each table are plausible readings of the numbers, not things the experiments tested directly.

### PSO coefficients

Inertia α: moderate values work best (0.5 and 0.7, about 99.8%). Low inertia (0.3) converges early to a worse solution, and values near 1.0 let particles overshoot.

| Inertia α | Validation accuracy (mean ± sd) | Validation loss | Iterations to within 5% of final loss |
|---|---|---|---|
| `0.3` | 96.35% ± 1.54 | 0.0962 | 29.5 |
| `0.5` | 99.78% ± 0.35 | 0.0155 | 79.8 |
| `0.7` (default) | 99.74% ± 0.46 | 0.0158 | 88.6 |
| `0.9` | 95.51% ± 2.12 | 0.1448 | 13.4 |
| `1.0` | 95.04% ± 2.54 | 0.1484 | 9.0 |

Cognitive β: 0.5 to 1.5 all perform well. β = 0 (no pull towards the particle's own best) and β = 2.5 are clearly worse.

| Cognitive β | Validation accuracy (mean ± sd) | Validation loss | Iterations to within 5% of final loss |
|---|---|---|---|
| `0.0` | 98.10% ± 1.50 | 0.0717 | 50.4 |
| `0.5` | 99.93% ± 0.15 | 0.0065 | 90.3 |
| `1.0` | 99.96% ± 0.12 | 0.0048 | 95.1 |
| `1.5` (default) | 99.74% ± 0.46 | 0.0158 | 88.6 |
| `2.0` | 98.39% ± 1.16 | 0.0433 | 51.9 |
| `2.5` | 94.38% ± 3.17 | 0.1481 | 13.4 |

Informant δ: this one matters most. With δ = 0 accuracy falls to 83%, so the pull towards the informants' best is what drives the search. δ of 1.0 to 1.5 is best.

| Informant δ | Validation accuracy (mean ± sd) | Validation loss | Iterations to within 5% of final loss |
|---|---|---|---|
| `0.0` | 83.25% ± 6.17 | 0.4224 | 1.3 |
| `0.5` | 96.31% ± 0.56 | 0.1156 | 48.0 |
| `1.0` | 99.78% ± 0.35 | 0.0181 | 83.5 |
| `1.5` (default) | 99.74% ± 0.46 | 0.0158 | 88.6 |
| `2.0` | 96.17% ± 2.31 | 0.1036 | 28.0 |
| `2.5` | 91.90% ± 4.93 | 0.2351 | 6.4 |

Global γ: the default of 0 (no global-best pull) is best. Adding it reduced accuracy in every setting tried.

| Global γ | Validation accuracy (mean ± sd) | Validation loss | Iterations to within 5% of final loss |
|---|---|---|---|
| `0.0` (default) | 99.74% ± 0.46 | 0.0158 | 88.6 |
| `0.5` | 98.18% ± 1.22 | 0.0523 | 41.5 |
| `1.0` | 98.32% ± 0.91 | 0.0473 | 51.0 |
| `1.5` | 97.01% ± 1.84 | 0.0968 | 28.9 |

Step size ε: ε = 1.0 is best, and both smaller and larger steps are worse.

| Step size ε | Validation accuracy (mean ± sd) | Validation loss | Iterations to within 5% of final loss |
|---|---|---|---|
| `0.25` | 94.01% ± 2.79 | 0.1991 | 30.2 |
| `0.5` | 98.76% ± 1.09 | 0.0566 | 55.3 |
| `0.75` | 99.56% ± 0.66 | 0.0196 | 87.1 |
| `1.0` (default) | 99.74% ± 0.46 | 0.0158 | 88.6 |
| `1.25` | 95.69% ± 3.00 | 0.1262 | 21.8 |

### Iterations and swarm size

More iterations keep helping: 400 iterations reached 100% validation accuracy on all 10 seeds, at a cost of about 4 times the evaluations of the default. More particles helped up to 50.

| Iterations | Validation accuracy (mean ± sd) | Validation loss | Iterations to within 5% of final loss |
|---|---|---|---|
| `25` | 97.88% ± 0.77 | 0.0598 | 20.8 |
| `50` | 98.83% ± 0.99 | 0.0309 | 41.1 |
| `100` (default) | 99.74% ± 0.46 | 0.0158 | 88.6 |
| `200` | 99.89% ± 0.25 | 0.0057 | 192.6 |
| `400` | 100.00% ± 0.00 | 0.0017 | 377.8 |

| Particles | Validation accuracy (mean ± sd) | Validation loss | Iterations to within 5% of final loss |
|---|---|---|---|
| `10` | 98.69% ± 1.52 | 0.0292 | 86.6 |
| `20` | 99.74% ± 0.30 | 0.0139 | 85.2 |
| `30` (default) | 99.74% ± 0.46 | 0.0158 | 88.6 |
| `50` | 100.00% ± 0.00 | 0.0055 | 91.7 |
| `100` | 99.96% ± 0.12 | 0.0041 | 94.4 |

### Network design

Smaller networks did better. A hidden layer of 2 neurons (13 parameters) scored highest and 32 neurons (193 parameters) lowest, and adding hidden layers lowered accuracy more (98.72% with two hidden layers of 8, 96.82% with three). This is consistent with PSO finding it harder to search a larger parameter space with the same swarm and budget. The dataset is also easy, so a very small network is enough.

| Hidden layer | Validation accuracy (mean ± sd) | Validation loss | Iterations to within 5% of final loss |
|---|---|---|---|
| `[2]` | 99.93% ± 0.15 | 0.0087 | 93.2 |
| `[4]` | 99.85% ± 0.31 | 0.0085 | 91.6 |
| `[8]` (default) | 99.74% ± 0.46 | 0.0158 | 88.6 |
| `[16]` | 99.34% ± 1.07 | 0.0260 | 88.1 |
| `[32]` | 99.01% ± 0.57 | 0.0268 | 79.2 |

| Hidden layers | Validation accuracy (mean ± sd) | Validation loss | Iterations to within 5% of final loss |
|---|---|---|---|
| `[8]` (default) | 99.74% ± 0.46 | 0.0158 | 88.6 |
| `[8, 8]` | 98.72% ± 1.89 | 0.0317 | 72.8 |
| `[8, 8, 8]` | 96.82% ± 1.41 | 0.1110 | 68.2 |

Tanh was slightly ahead of sigmoid, but the gap is within the seed-to-seed spread.

| Activation | Validation accuracy (mean ± sd) | Validation loss | Iterations to within 5% of final loss |
|---|---|---|---|
| `['tanh']` (default) | 99.74% ± 0.46 | 0.0158 | 88.6 |
| `['sigmoid']` | 99.56% ± 0.68 | 0.0189 | 87.5 |

### Informant strategy

Nearest-neighbour informants were much better than random ones when k is small: nearest with k = 1 gave 99.20% against about 96% for random with k = 1 to 5. As k approaches the swarm size (30 particles, so k = 29 means everyone), the two strategies converge to the same accuracy, which is expected because each particle then listens to the whole swarm.

| Strategy and k | Validation accuracy (mean ± sd) | Validation loss | Iterations to within 5% of final loss |
|---|---|---|---|
| `random-k1` | 96.02% ± 2.02 | 0.1267 | 43.5 |
| `random-k2` | 96.13% ± 1.47 | 0.1251 | 32.9 |
| `random-k3` | 96.02% ± 1.64 | 0.1132 | 25.4 |
| `random-k5` | 96.50% ± 1.55 | 0.1101 | 19.8 |
| `random-k10` | 98.18% ± 1.14 | 0.0722 | 53.4 |
| `random-k20` | 99.89% ± 0.18 | 0.0046 | 86.5 |
| `random-k29` | 99.96% ± 0.12 | 0.0018 | 97.8 |
| `nearest-k1` | 99.20% ± 0.82 | 0.0331 | 87.5 |
| `nearest-k2` | 99.74% ± 0.35 | 0.0117 | 86.9 |
| `nearest-k3` (default) | 99.74% ± 0.46 | 0.0158 | 88.6 |
| `nearest-k5` | 99.85% ± 0.26 | 0.0069 | 91.4 |
| `nearest-k10` | 99.96% ± 0.12 | 0.0029 | 95.1 |
| `nearest-k20` | 99.93% ± 0.15 | 0.0023 | 97.6 |
| `nearest-k29` | 99.96% ± 0.12 | 0.0018 | 97.8 |

## Extensions and extra experiments

These go beyond the minimum specification. All of them are switched off by default, so the default behaviour is exactly the plain algorithm and the results above are unchanged (checked: seeds 0 and 6 of the final run reproduce the same test accuracy and loss). They use the same protocol as the sweeps (train on the fit set, compare on the validation set, 10 seeds), and the numbers below are validation accuracy (mean ± sd) from the CSVs named in each heading. Run them with `python run_extra_experiments.py` or from section 3b of the notebook.

**Added to the code**

- **ReLU activation** (`mlp/activations.py`), alongside sigmoid and tanh, usable as `"relu"` in any configuration.
- **Boundary handling** (`pso/population.py`): `boundary="none" | "clamp" | "reflect" | "random"` with a box half-width `bound`. Clamp puts the coordinate on the wall and zeroes that velocity component, reflect bounces it back and reverses the velocity component, random re-draws the coordinate inside the box.
- **Linearly decreasing inertia** (`pso/population.py`): `inertia_schedule="linear"` moves alpha from `inertia_weight` to `inertia_end` over the run.
- **PSO chooses the hidden activations** (`adapter.py`, `MLPFitnessAdapterWithActivations`): the particle vector gets one extra gene per hidden layer. Each gene is squashed with tanh and cut into equal bins (sigmoid, tanh, relu), so any real value decodes to a valid choice. PSO is unchanged and still sees only a longer vector and a scalar fitness.
- **Algorithm 39 line references** in the comments of `pso/population.py` and `pso/particle.py`, checked against the algorithm in *Essentials of Metaheuristics* (Section 3.5). Two deliberate differences from the book's wording follow the coursework design document: (1) the names of the two social coefficients are swapped, since the book uses gamma for the informants' best and delta for the global best while this code uses delta for the informants' best and gamma for the global best; (2) a particle is never its own informant, whereas the book's line 18 includes it. The mapping is repeated in the comments.
- `training.py` and `experiments.py` accept the new options through optional config keys (`boundary`, `bound`, `inertia_schedule`, `inertia_end`, `evolve_activations`).

**Fixed budget** (`results/budget_split_500.csv`, `results/budget_split_3000.csv`). Evaluations are particles × (iterations + 1), so each row spends the same total.

| Particles x iterations (budget 500) | Validation accuracy | Particles x iterations (budget 3000) | Validation accuracy |
|---|---|---|---|
| `10x49` | 97.63% ± 1.51 | `10x299` | 99.82% ± 0.35 |
| `20x24` | 96.79% ± 1.53 | `20x149` | 99.96% ± 0.12 |
| `25x19` | 96.72% ± 1.37 | `30x99` | 99.74% ± 0.46 |
| `50x9` | 95.18% ± 1.67 | `50x59` | 99.71% ± 0.48 |
| `100x4` | 93.47% ± 2.28 | `100x29` | 98.69% ± 0.71 |

With a small budget, fewer particles and more iterations did better. At 3000 the middle (20 particles) was best and a large swarm with few iterations was worst. The difference between the smallest and largest swarm is significant in both budgets (paired test, p < 0.05, see `results/significance.csv`).

**Cognitive beta and informant delta** (`results/pso_beta_delta_grid.csv`). Delta matters far more than beta: with delta = 0.5 or 2.0 every beta gives only about 95 to 97%, while delta = 1.5 gives 99.7 to 99.96% for beta up to 1.5. The best cell was beta = 1.0 with delta = 1.5 (99.96% ± 0.12).

**Boundary handling** (`results/pso_boundary.csv`)

| Mode | Validation accuracy |
|---|---|
| `none` (default) | 99.74% ± 0.46 |
| `clamp`, B = 2 | 99.85% ± 0.19 |
| `clamp`, B = 5 | 99.82% ± 0.31 |
| `reflect`, B = 5 | 99.49% ± 0.43 |
| `random`, B = 5 | 99.16% ± 1.00 |
| `reflect`, B = 2 | 98.91% ± 1.32 |
| `random`, B = 2 | 97.85% ± 1.67 |

Clamping matched or slightly beat no boundary. Reflect and especially random re-initialisation were worse, most clearly with the tight box (B = 2). Clamp versus none was not a significant difference.

**Activation genes** (`results/activation_genes.csv`)

| Hidden activation | Validation accuracy |
|---|---|
| `evolved` (PSO chooses) | 99.96% ± 0.12 |
| `tanh` | 99.74% ± 0.46 |
| `sigmoid` | 99.56% ± 0.68 |
| `relu` | 99.16% ± 1.00 |

Letting PSO choose scored highest, but the difference from fixed tanh is within the seed-to-seed spread (p = 0.38 for accuracy, 0.06 for loss), so treat it as suggestive, not proven. The evolved run also has one extra dimension per hidden layer.

**Inertia schedule** (`results/pso_inertia_schedule.csv`)

| Inertia | Validation accuracy |
|---|---|
| constant 0.7 (default) | 99.74% ± 0.46 |
| linear 0.9 to 0.4 | 99.56% ± 0.94 |
| linear 0.8 to 0.5 | 99.93% ± 0.15 |
| linear 0.9 to 0.2 | 100.00% ± 0.00 |

The two schedules that end lower looked better, but none of the differences is significant on accuracy (p >= 0.25).

**Combined best settings** (`results/tuned_candidates.csv`). Combining the best single-sweep values (beta = 1.0, delta = 1.0, nearest-neighbour k = 10) at the same budget (30 particles, 100 iterations) gave 100.00% ± 0.00 against 99.74% ± 0.46 for the default. The loss is significantly lower (p = 0.015) but accuracy is not (p = 0.25). A larger swarm and more iterations (50 x 200, about 3.3 times the evaluations) gave 99.93% ± 0.15. The comparison was made on the validation set only. This combined setting (30 particles, 100 iterations, nearest-neighbour k = 10, β = 1.0, δ = 1.0, all other values at the default) was then adopted as the final configuration and run once on the test set. The single-sweep values were each picked from a one-at-a-time sweep and the combination was tested together only in this comparison, so it is a modest amount of tuning, not an exhaustive search.

**Significance tests** (`results/significance.csv`). Paired sign-flip permutation tests (same seeds on both sides) for the key comparisons. With 10 seeds the smallest possible p-value is about 0.002, and about 20 comparisons were run without multiple-comparison correction, so p-values just under 0.05 are weak evidence. Random versus nearest-neighbour informants at k = 3 is the clearest result (p = 0.002).

### Limitations

- The final configuration combines three single-sweep winners (β = 1.0, δ = 1.0, k = 10) and keeps the default budget (30 particles, 100 iterations). Larger budgets (for example 400 iterations or 50 particles) also scored well on validation but were not adopted, so the comparison stays at an equal number of evaluations.
- Every seed reached 100% test accuracy, so the test set can no longer separate this configuration from an even better one, and differences between settings at this level are within what one misclassified note would change.
- The test set holds 274 notes, so one misclassified note changes accuracy by about 0.36 percentage points.
- Runtime figures depend on the machine and on what else was running.
- Most sweeps vary one parameter at a time. Only the β × δ grid, the budget split and the combined-settings comparison look at more than one parameter together.

## Project structure

```
data/           banknote dataset (data_banknote_authentication.txt)
mlp/            MLP from scratch: activations, losses, layer, network, builder
pso/            PSO from scratch: fitness interface, particle, informant strategies, population (with boundary handling and inertia schedule)
utils/          dataset loading, stratified split, standardiser, classification metrics
adapter.py      couples the MLP to PSO (the only place that knows both)
training.py     shared helper that builds the network and trains it with PSO
main.py         command-line run (one training run)
app.py          Streamlit GUI (train and inspect one run)
experiments.py  experiment helpers (data protocol, sweeps, statistics, plots, extra experiments)
run_extra_experiments.py  command-line runner for the extra experiments
experiments.ipynb  the experiment notebook that writes results/
tests/          automatic checks (pytest)
results/        CSV and JSON outputs of the experiments
report/         final project report
```

Design: the MLP builder follows the Builder pattern, informant selection follows the Strategy pattern, and the adapter follows the Adapter pattern. The `mlp` and `pso` packages do not import each other.

## Replicating this project

This section is for anyone who wants to run the code and reproduce the results on their own machine. Everything runs on a normal laptop with no GPU, and the dataset is included in the repository.

### 1. What you need

| Requirement | Details |
|---|---|
| Python | Python 3 (the project's virtual environment was created with Python 3.14, and it has not been tested on older versions). Check with `python3 --version`. |
| Packages | `numpy`, `pandas`, `matplotlib`, `ipykernel`, `pytest`, `streamlit` (listed in `requirements.txt`, installed in step 3). |
| Editor for the notebook | VS Code with the Python and Jupyter extensions, or any Jupyter front end. |
| Disk and network | A few hundred MB for the packages. Internet access is only needed for `pip install` and `git clone`. |

On macOS, install Python with Homebrew if you don't have it: `brew install python`.

### 2. Get the code

```bash
git clone https://github.com/shailendragadakari/Implementation-and-Integration-of-MLP-and-PSO-from-Scratch.git
cd Implementation-and-Integration-of-MLP-and-PSO-from-Scratch
```

If you received the project as a zip file instead, unzip it and `cd` into the folder. Check that `data/data_banknote_authentication.txt` is there, because every script reads it from that path.

### 3. Create a virtual environment and install the packages

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Your prompt should now start with `(.venv)`. Run every command below from the project's root folder with the environment active. The project has no non-Python dependencies.

### 4. Check that everything works

```bash
pytest tests -q
```

All tests should pass (35 at the time of writing). They cover the activation functions (including ReLU), the boundary modes, the inertia schedule, the activation-gene adapter, losses, layers, network, builder, PSO components, the data utilities, and a check that `mlp/` and `pso/` do not import each other.

### 5. Run one training run

```bash
python main.py
```

This loads the dataset, splits it, builds the MLP, trains it with PSO and prints the training loss and test metrics. With the default settings you should see test accuracy of roughly 99 to 100%, and it finishes in about a second. The command-line defaults are the final configuration (k = 10, β = 1.0, δ = 1.0). Options can be listed with `python main.py --help`, for example:

```bash
python main.py --hidden 8 --informants nearest --k 3 --particles 30 --iterations 100 --seed 42
python main.py --hidden 4 4 --activations tanh tanh --informants random --k 5
```

Using the same `--seed` gives the same result on the same machine.

### 6. Launch the GUI

```bash
streamlit run app.py
```

Streamlit prints a local address (normally http://localhost:8501) and opens it in your browser. The sidebar sets the network and PSO hyperparameters, and the tabs let you train a single run and inspect the convergence curve, the results, individual predictions and the data. Stop the app with Ctrl+C in the terminal.

### 7. Reproduce the experiments and the results files

1. Open `experiments.ipynb` in VS Code and choose the `.venv` environment as the kernel (top-right "Select Kernel").
2. For a quick dry run, change the line `N_SEEDS = int(os.environ.get("N_SEEDS", 10))` in the first code cell to `N_SEEDS = 3` (or set the `N_SEEDS` environment variable before starting the notebook server). Leave it as it is for the full 10 seeds used in this README.
3. Run all cells. The notebook runs the baseline, every sweep, the informant comparison, the extra experiments (section 3b) and the final test evaluation, and in its last cell saves the CSV files and `run_info.json` into `results/`.

Notes:

- Every setting is run over many seeds in a process pool, so the full run takes longer than a single `main.py` run. As a rough guide, a 3-seed dry run took about 77 seconds on a 2-core Linux machine during development. The full 10-seed run was not timed there, so expect several minutes, depending on your CPU.
- Running the notebook **overwrites** the files in `results/`. Copy the folder first if you want to keep the originals for comparison.
- The notebook does not create `results/best_run.json`. That file was derived from `results/final_test.csv`: take the row with the highest `eval_accuracy`, breaking ties with the lowest `eval_loss`.
- Seeds, the data split and the PSO settings are fixed in the code, so your numbers should match `results/` very closely. Small differences are possible if your NumPy version or CPU differs from the one used here, so compare means and spreads rather than expecting identical digits.

To confirm you have reproduced the main result, check that `results/final_test.csv` has 10 rows with test accuracy at or very close to 100% (the mean should be 100% or within one misclassified note of it).

### 8. Make your own changes

- **Different hyperparameters in the notebook:** edit `FINAL_CONFIG` in the final-evaluation cell, or the sweep values passed to `make_configs`.
- **Different network or PSO settings from the command line:** use the `main.py` flags above.
- **A different dataset:** `mlp/builder.py` derives the input size, output size, output activation and loss from the data and a task name (`binary_classification` or `multiclass_classification`). The metrics in `utils/metrics.py` are binary-only, so multi-class data would need extra metric code.

### Troubleshooting

| Problem | Likely fix |
|---|---|
| `ModuleNotFoundError: No module named 'numpy'` (or `streamlit`, `pandas`) | The environment is not active or the install did not run. Activate `.venv` and run `pip install -r requirements.txt` again. |
| `ModuleNotFoundError: No module named 'utils'` / `mlp` / `pso` | You are not in the project's root folder. `cd` to it first. In the notebook, make sure it was opened from that folder. |
| `FileNotFoundError` for the dataset | `data/data_banknote_authentication.txt` is missing or the folder was moved. Restore it to that path. |
| `command not found: streamlit` | The environment is not active. Run `source .venv/bin/activate`, or use `python -m streamlit run app.py`. |
| The notebook cannot find the kernel | In VS Code choose "Select Kernel", then "Python Environments", then `.venv`. Install the Jupyter extension if the option is missing. |
| `pip install` is slow or fails | Update pip with `python -m pip install --upgrade pip` and retry. Check your network or proxy. |
| Port 8501 is already in use | Run `streamlit run app.py --server.port 8502`. |
| The notebook takes very long | Use `N_SEEDS = 3` for a dry run, or close other heavy applications. |
