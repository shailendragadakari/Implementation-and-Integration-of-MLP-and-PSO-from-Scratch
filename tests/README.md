# tests

Run from the project root:

```bash
pytest tests -q
```

`test_all.py` checks the activations, losses, layers, network, builder, informant strategies, PSO on a simple test function, the adapter, the data and metrics helpers, the extensions (boundary modes, inertia schedule, activation genes), and that `mlp/` and `pso/` do not import each other.
