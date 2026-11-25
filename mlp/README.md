# mlp: multilayer perceptron from scratch (NumPy only)

No neural-network library and no training algorithm live here. The network only computes outputs; PSO (in `pso/`) searches for its weights and biases.

| File | Contents |
|---|---|
| `activations.py` | `ActivationFunction` interface with `Sigmoid`, `Tanh`, `ReLU` and `Softmax`; `get_activation(name)` looks one up by name |
| `losses.py` | `Loss` interface with `MSE`, `BinaryCrossEntropy` and `CategoricalCrossEntropy` (clipped to avoid `log(0)`) |
| `layer.py` | `Layer`: weights, biases and an activation object; flat `get_parameters` / `set_parameters` |
| `network.py` | `Network`: ordered layers, `forward`, `predict`, and the whole parameter vector as one flat array |
| `builder.py` | `MLPBuilder` (Builder pattern): builds the network and loss from the data, a task name and the hidden layers and activations |

Parameter order in the flat vector: for each layer, its weights (row-major) and then its biases. The vector length is the sum over layers of `n_in * n_out + n_out`.
