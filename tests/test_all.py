import ast
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adapter import MLPFitnessAdapter
from mlp import (BinaryCrossEntropy, CategoricalCrossEntropy, Layer, MLPBuilder, MSE,
                 Network, Sigmoid, Softmax, Tanh)
from pso import (FitnessFunction, NearestNeighborInformantSelection, Particle, Population,
                 RandomInformantSelection)
from utilities.data import Standardiser, load_dataset, train_test_split
from utilities.metrics import classification_metrics

# MLP
def build(hidden=(5, 3), acts=("tanh", "sigmoid"), task="binary_classification", n_feat=4, n_cls=2):
    X = np.random.default_rng(0).normal(size=(29, n_feat))
    y = np.arange(20) % n_cls
    b = MLPBuilder(seed=1).set_dataset(X, y)
    b.set_configuration({"hidden_layers": list(hidden), "activations": list(acts), "task": task})
    return (*b.build(), X, y)

def test_parameter_count_matches_spec_formula():
    net, _, _, _ = build()
    sizes = [4, 5, 3, 1] # input + hidden + output
    expected = sum(sizes[i - 1] * sizes[i] + sizes[i] for i in range(1, len(sizes)))
    assert net.count_parameters() == expected == len(net.get_parameters())
    
def test_parameter_roundtrip_and_order():
    net, _, X, _ = build()
    p = np.random.default_rng(3).normal(size=net.count_parameters())
    net.set_parameters(p)
    assert np.array_equal(net.get_parameters(), p)
    # documented order: layer-1 weights, layer-1 biases, layer-2 weights and so on...
    l1 = net.layers[0]
    assert np.array_equal(l1.weights.ravel(), p[: l1.n_in * l1.n_out])
    assert np.array_equal(l1.biases, p[l1.n_in * l1.n_out: l1.count_parameters()])
    
def test_set_parameters_wrong_size_rejected():
    net, _, _, _ = build()
    with pytest.raises(ValueError):
        net.set_parameters(np.zeros(net.count_parameters() + 1))
        
def test_set_parameters_copies_input():
    net, _, _, _ = build()
    p = np.zeros(net.count_parameters())
    net.set_parameters(p)
    p[:] = 9.0  # mutating PSO's vector must not change the network's parameters
    assert not np.all(net.get_parameters() == 9.0)
    
def test_forward_matches_manual_computation():
    net, _, X, _ = build()
    h = np.tanh(X @ net.layers[0].weights + net.layers[0].biases)
    h2 = Sigmoid().forward(h @ net.layers[1].weights + net.layers[1].biases)
    out = Sigmoid().forward(h2 @ net.layers[2].weights + net.layers[2].biases)
    assert np.allclose(net.forward(X), out) and out.shape == (20, 1)
    
def test_activations():
    z = np.array([[-1000.0, 0.0, 1000.0]])
    s = Sigmoid().forward(z)
    assert np.all(np.isfinite(s)) and s[0, 0] == 0.0 and s[0, 1] == 0.5 and s[0, 2] == 1.0
    assert np.all(np.abs(Tanh().forward(z)) <= 1)
    sm = Softmax().forward(np.array([[1000.0, 1001.0, 999.0], [0.0, 0.0, 0.0]]))
    assert np.allclose(sm.sum(axis=1), 1) and np.all(np.isfinite(sm))
    
def test_losses_known_values():
    assert MSE().compute(np.array([1.0, 0.0]), np.array([0.5, 0.5])) == pytest.approx(0.25)
    bce = BinaryCrossEntropy().compute(np.array([1, 0]), np.array([[0.9], [0.1]]))
    assert bce == pytest.approx(-np.log(0.9))
    cce = CategoricalCrossEntropy().compute(np.array([[1, 0], [0, 1]]),
                                            np.array([[0.8, 0.2], [0.3, 0.7]]))
    assert cce == pytest.approx(-(np.log(0.8) + np.log(0.7)) / 2)
    
def test_bce_is_finite_at_extremes_and_no_broadcast_bug():
    y = np.array([1, 0, 1])
    assert np.isfinite(BinaryCrossEntropy().compute(y, np.array([[1.0], [0.0], [0.0]])))
    # (n,) labels vs (n,1) predictions must equal the (n,1) vs (n,1) result
    p = np.array([[0.9], [0.2], [0.6]])
    assert BinaryCrossEntropy().compute(y, p) == pytest.approx(
        BinaryCrossEntropy().compute(y.reshape(-1, 1), p))
    
def test_builder_validation_and_multiclass():
    X, y = np.zeros((9, 6)), np.arange(9) % 3
    b = MLPBuilder().set_dataset(X, y)
    with pytest.raises(ValueError):
        b.set_configuration({"hidden_layers": [4, 4], "activations": ["tanh"], "task": "multiclass_classification"})
    with pytest.raises(ValueError):
        b.set_configuration({"hidden_layers": [4], "activations": ["relu6"], "task": "multiclass_classification"})
    b.set_configuration({"hidden_layers": [8, 5], "activations": ["tanh", "sigmoid"],
                         "task": "multiclass_classification"})
    net, loss = b.build()  # the spec's worked example, with 6 inputs and 3 classes
    assert [(l.n_in, l.n_out) for l in net.layers] == [(6, 8), (8, 5), (5, 3)]
    assert isinstance(net.layers[-1].activation, Softmax) and isinstance(loss, CategoricalCrossEntropy)
    assert np.allclose(net.forward(X).sum(axis=1), 1)
    
def test_binary_output_config():
    net, loss, _, _ = build()
    assert net.layers[-1].n_out == 1 and isinstance(net.layers[-1].activation, Sigmoid)
    assert isinstance(loss, BinaryCrossEntropy)
    
# PSO
class Sphere(FitnessFunction):
    def evaluate(self, position):
        return float(np.sum(position ** 2))
    
def make_particles(n=8, d=3, seed=0):
    rng = np.random.default_rng(seed)
    return [Particle(rng.normal(size=d), np.zeros(d)) for _ in range(n)]

@pytest.mark.parametrize("strategy", [RandomInformantSelection(seed=0), 
                                      NearestNeighborInformantSelection()])
def test_informants_exclude_self_distinct_and_k_bounds(strategy):
    ps = make_particles()
    for k in (1, 3, 7):
        for p in ps:
            inf = strategy.select_informants(p, ps, k)
            assert len(inf) == k and p not in inf and len({id(q) for q in inf}) == k
    for bad in (0, 8):
        with pytest.raises(ValueError):
            strategy.select_informants(ps[0], ps, bad)
            
def test_nearest_neighbor_matches_bruteforce():
    ps = make_particles(n=12, d=5, seed=4)
    target = ps[0]
    got = NearestNeighborInformantSelection().select_informants(target, ps, 4)
    dist = sorted((np.linalg.norm(p.position - target.position), id(p)) for p in ps[1:])
    assert {id(p) for p in got} == {i for _, i in dist[:4]}
    
def test_velocity_update_formula():
    p = Particle(np.array([1.0, 1.0]), np.array([2.0, -2.0]), rng=np.random.default_rng(0))
    p.best_position = np.array([0.0, 0.0])
    r = np.random.default_rng(0)
    r1, r2, r3 = r.random(2), r.random(2), r.random(2)
    ib, gb = np.array([3.0, 3.0]), np.array([-1.0, -1.0])
    expected = (0.5 * p.velocity + 1.4 * r1 * (p.best_position - p.position)
                + 1.1 * r2 * (ib - p.position) + 0.9 * r3 * (gb - p.position))
    p.update_velocity(ib, gb, alpha=0.5, beta=1.4, gamma=0.9, delta=1.1)
    assert np.allclose(p.velocity, expected)
    old = p.position.copy()
    p.update_position(epsilon=0.5)
    assert np.allclose(p.position, old + 0.5 * expected)
    
def test_personal_best_only_improves():
    p = Particle(np.array([1.0]), np.zeros(1))
    p.fitness = 5.0; p.update_personal_best()
    p.position = np.array([2.0]); p.fitness = 9.0; p.update_personal_best()
    assert p.best_fitness == 5.0 and p.best_position[0] == 1.0
    
@pytest.mark.parametrize("strategy_cls", [RandomInformantSelection, 
                                          NearestNeighborInformantSelection])
def test_pso_minimizes_sphere_and_global_best_monotone(strategy_cls):
    pop = Population(Sphere(), strategy_cls(), num_particles=20, num_informants=3,
                     num_iterations=150, dimension=5, inertia_weight=0.7,
                     cognitive_coefficient=1.5, social_coefficient=1.5, seed=1)
    _, best = pop.optimize()
    h = pop.history["global_best"]
    assert best < 1e-3 and all(b <= a for a, b in zip(h, h[1:]))
    assert len(h) == 151 and pop.n_evaluations == 20 * 151
    
def test_pso_is_reproducible_with_seed():
    def run():
        pop = Population(Sphere(), RandomInformantSelection(seed=7), 10, 2, 30, 4, seed=7)
        return pop.optimize()[1]
    assert run() == run()
    
def test_population_rejects_invalid_k():
    with pytest.raises(ValueError):
        Population(Sphere(), RandomInformantSelection(), 5, 5, 10, 2)
        
def test_pso_package_is_independent_of_mlp():
    # Spec: PSO must not contain any MLP specific code and it must not import it.
    for f in (ROOT / "pso").glob("*.py"):
        for node in ast.walk(ast.parse(f.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            assert not any(n == "mlp" or n.startswith("mlp.") or n == "adapter" for n in names), f
            
def test_mlp_package_is_numpy_only_and_independent_of_pso():
    for f in (ROOT / "mlp").glob("*.py"):
        mods = []
        for node in ast.walk(ast.parse(f.read_text())):
            if isinstance(node, ast.Import):
                mods += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                mods.append(node.module or "")
        top = {m.split(".")[0] for m in mods}
        assert top <= {"numpy", "mlp", "abc"}, (f.name, top)
        
# Adapter
def test_adapter_returns_scalar_installs_params_and_matches_manual_loss():
    net, loss, X, y = build()
    ad = MLPFitnessAdapter(net, loss, X, y)
    assert isinstance(ad, FitnessFunction)
    pos = np.random.default_rng(5).normal(size=net.count_parameters())
    f = ad.evaluate(pos)
    assert isinstance(f, float)
    assert np.array_equal(net.get_parameters(), pos)
    assert f == pytest.approx(loss.compute(y, net.forward(X)))
    
# Data and metrics
def test_split_is_stratified_disjoint_and_deterministic():
    X, y = load_dataset(ROOT / "data" / "data_banknote_authentication.txt")
    assert X.shape == (1372, 4) and np.bincount(y).tolist() == [762, 610]
    Xtr, Xte, ytr, yte = train_test_split(X, y, 0.2, 42)
    assert len(ytr) + len(yte) == 1372
    assert abs(yte.mean() - y.mean()) < 0.01
    again = train_test_split(X, y, 0.2, 42)
    assert np.array_equal(again[1], Xte)
    # row-level disjointness (ignoring genuine duplicate rows that exist in the raw data)
    tr_rows = {tuple(r) for r in Xtr}
    dup_rows = len(Xte) - len({tuple(r) for r in Xte} - tr_rows)
    assert dup_rows <= 24  # only the 24 duplicates in the source file can straddle the split
    
def test_standardiser_uses_training_statistics_only():
    tr, te = np.array([[0.0], [2.0]]), np.array([[100.0]])
    s = Standardiser().fit(tr)
    assert s.mean_[0] == 1.0 and s.transform(te)[0, 0] == 99.0
    
def test_metrics():
    m = classification_metrics([1, 1, 0, 0, 1], [1, 0, 0, 1, 1])
    assert (m["tp"], m["tn"], m["fp"], m["fn"]) == (2, 1, 1, 1)
    assert m["accuracy"] == pytest.approx(0.6) and m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3) and m["f1"] == pytest.approx(2 / 3)
    assert classification_metrics([0, 0], [0, 0])["f1"] == 0.0  # no division by zero
    
def test_optimize_callback_called_each_iteration():
    calls = []
    pop = Population(Sphere(), RandomInformantSelection(seed=0), 6, 2, 10, 3, seed=0)
    pop.optimize(callback=lambda it, p: calls.append((it, p.global_best_fitness)))
    assert [c[0] for c in calls] == list(range(11))
    assert [c[1] for c in calls] == pop.history["global_best"]