# Streamlit GUI for the PSO-trained MLP.   Run:  streamlit run app.py
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from training import train_pso_mlp
from utils.data import Standardiser, load_dataset, train_test_split
from utils.metrics import classification_metrics, probabilities_to_labels

DATA = Path(__file__).parent / "data" / "data_banknote_authentication.txt"
FEATURES = ["Variance", "Skewness", "Curtosis", "Entropy"]

st.set_page_config(page_title="PSO-trained MLP", layout="wide")


@st.cache_data
def get_data(split_seed: int):
    X, y = load_dataset(DATA)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, 0.2, split_seed)
    scaler = Standardiser().fit(X_tr)  # fitted on training data only
    return X, y, X_tr, X_te, y_tr, y_te, scaler


def convergence_fig(best, mean=None):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(best, label="global best")
    if mean is not None:
        ax.plot(mean, label="swarm mean (current)", alpha=0.6)
    ax.set_yscale("log"); ax.set_xlabel("Iteration"); ax.set_ylabel("Training loss (BCE)")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ sidebar
sb = st.sidebar
sb.header("Data")
split_seed = sb.number_input("Train/test split seed", 0, 10_000, 42)

sb.header("MLP")
n_layers = sb.slider("Hidden layers", 1, 3, 1)
hidden, acts = [], []
for i in range(n_layers):
    c1, c2 = sb.columns(2)
    hidden.append(c1.number_input(f"Units L{i+1}", 1, 64, 8, key=f"u{i}"))
    acts.append(c2.selectbox(f"Act. L{i+1}", ["tanh", "sigmoid"], key=f"a{i}"))

sb.header("PSO")
particles = sb.slider("Particles", 5, 200, 30)
iterations = sb.slider("Iterations", 10, 500, 100, step=10)
strategy = sb.radio("Informant selection", ["nearest", "random"],
                    format_func=lambda s: "Nearest neighbor (Euclidean)" if s == "nearest" else "Random",
                    horizontal=False)
k = sb.slider("Informants (k)", 1, particles - 1, min(3, particles - 1))
alpha = sb.slider("Inertia α", 0.0, 1.2, 0.7, 0.05)
beta = sb.slider("Cognitive β", 0.0, 3.0, 1.5, 0.1)
delta = sb.slider("Informants δ", 0.0, 3.0, 1.5, 0.1)
gamma = sb.slider("Global best γ", 0.0, 3.0, 0.0, 0.1)
epsilon = sb.slider("Step size ε", 0.05, 1.5, 1.0, 0.05)
seed = sb.number_input("PSO seed", 0, 10_000, 42)

config = dict(hidden=hidden, activations=acts, strategy=strategy, k=k, particles=particles,
              iterations=iterations, alpha=alpha, beta=beta, delta=delta, gamma=gamma,
              epsilon=epsilon, seed=int(seed))

X, y, X_tr, X_te, y_tr, y_te, scaler = get_data(int(split_seed))
Xs_tr, Xs_te = scaler.transform(X_tr), scaler.transform(X_te)
dim = sum(a * b + b for a, b in zip([4] + hidden, hidden + [1]))

st.title("Banknote authentication: MLP trained with PSO")
st.caption(f"Architecture 4 → {' → '.join(str(h) for h in hidden)} → 1 · "
           f"{dim} parameters = PSO search-space dimension · "
           f"{particles * (iterations + 1):,} fitness evaluations")

tab_train, tab_results, tab_predict, tab_data = st.tabs(["Train", "Results", "Predict", "Data"])

# ------------------------------------------------------------------ train
with tab_train:
    if st.button("Train network with PSO", type="primary"):
        bar = st.progress(0.0, text="Starting…")
        status = st.empty()
        chart = st.empty()
        every = max(1, iterations // 40)

        def on_iter(it, pop):
            if it % every == 0 or it == iterations:
                bar.progress(it / iterations, text=f"Iteration {it}/{iterations}")
                status.metric("Best training loss", f"{pop.global_best_fitness:.5f}")
                fig = convergence_fig(pop.history["global_best"], pop.history["mean_fitness"])
                chart.pyplot(fig); plt.close(fig)

        network, loss, pop = train_pso_mlp(Xs_tr, y_tr, config, callback=on_iter)
        probs = network.predict(Xs_te)
        m = classification_metrics(y_te, probabilities_to_labels(probs))
        st.session_state["run"] = dict(
            network=network, config=config, scaler=scaler, metrics=m,
            best=list(pop.history["global_best"]), mean=list(pop.history["mean_fitness"]),
            best_loss=pop.global_best_fitness, test_loss=loss.compute(y_te, probs),
            evals=pop.n_evaluations)
        bar.progress(1.0, text="Done")
        status.empty()
        st.success("Training finished. See the Results and Predict tabs.")
    else:
        st.info("Choose hyperparameters in the sidebar, then press **Train network with PSO**. "
                "PSO only sees the 80 % training split; the 20 % test split is used afterwards.")

# ------------------------------------------------------------------ results
with tab_results:
    run = st.session_state.get("run")
    if not run:
        st.info("Train a network first.")
    else:
        m = run["metrics"]
        c = st.columns(6)
        c[0].metric("Best training loss", f"{run['best_loss']:.5f}")
        c[1].metric("Test loss", f"{run['test_loss']:.5f}")
        c[2].metric("Accuracy", f"{m['accuracy']:.4f}")
        c[3].metric("Precision", f"{m['precision']:.4f}")
        c[4].metric("Recall", f"{m['recall']:.4f}")
        c[5].metric("F1", f"{m['f1']:.4f}")
        left, right = st.columns(2)
        with left:
            st.subheader("PSO convergence")
            fig = convergence_fig(run["best"], run["mean"]); st.pyplot(fig); plt.close(fig)
        with right:
            st.subheader("Confusion matrix (test set)")
            cm = pd.DataFrame([[m["tn"], m["fp"]], [m["fn"], m["tp"]]],
                              index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])
            st.dataframe(cm, width="stretch")
            st.caption(f"{run['evals']:,} fitness evaluations · test set size {len(y_te)} "
                       "· class 1 is the positive class for precision/recall.")
            st.download_button("Download convergence (CSV)",
                               pd.DataFrame({"global_best": run["best"], "swarm_mean": run["mean"]})
                               .to_csv(index_label="iteration"), "convergence.csv", "text/csv")
        with st.expander("Hyperparameters used for this run"):
            st.json({k_: v for k_, v in run["config"].items()})

# ------------------------------------------------------------------ predict
with tab_predict:
    run = st.session_state.get("run")
    if not run:
        st.info("Train a network first.")
    else:
        st.write("Enter wavelet-transform features of a banknote image (defaults = dataset means).")
        cols = st.columns(4)
        vals = [cols[i].number_input(f, value=float(round(X[:, i].mean(), 3)), format="%.4f", key=f"p{i}")
                for i, f in enumerate(FEATURES)]
        x = run["scaler"].transform(np.array([vals]))
        p1 = float(run["network"].predict(x)[0, 0])
        st.metric("P(class 1)", f"{p1:.4f}")
        st.write(f"Predicted class: **{int(p1 >= 0.5)}**  (threshold 0.5)")
        st.progress(min(max(p1, 0.0), 1.0))

# ------------------------------------------------------------------ data
with tab_data:
    df = pd.DataFrame(X, columns=FEATURES); df["class"] = y
    c = st.columns(4)
    c[0].metric("Samples", len(df)); c[1].metric("Class 0", int((y == 0).sum()))
    c[2].metric("Class 1", int((y == 1).sum())); c[3].metric("Duplicate rows", int(df.duplicated().sum()))
    st.dataframe(df.describe().T, width="stretch")
    st.dataframe(df.head(50), width="stretch")
