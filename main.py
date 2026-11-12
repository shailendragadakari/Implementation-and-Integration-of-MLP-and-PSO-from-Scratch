# Main application: load -> split -> build MLP -> adapter -> PSO -> test.
# python main.py --informants nearest --k 3 --particles 30 --iterations 100

import argparse
from pathlib import Path
from adapter import MLPFitnessAdapter
from mlp import MLPBuilder
from pso import NearestNeighborInformantSelection, Population, RandomInformantSelection
from utilities.data import Standardiser, load_dataset, train_test_split
from utilities.metrics import classification_metrics, probabilities_to_labels

DATA = Path(__file__).parent / "data" / "data_banknote_authentication.txt"

def parse_args():
    a = argparse.ArgumentParser()
    a.add_argument("--hidden", type=int, nargs="+", default=[8])
    a.add_argument("--activations", nargs="+", default=None, help="default: tanh per hidden layer")
    a.add_argument("--informants", choices=["random", "nearest"], default="nearest")
    a.add_argument("--k", type=int, default=3, help="number of informants")
    a.add_argument("--particles", type=int, default=30)
    a.add_argument("--iterations", type=int, default=100)
    a.add_argument("--alpha", type=float, default=0.7, help="inertia")
    a.add_argument("--beta", type=float, default=1.5, help="cognitive")
    a.add_argument("--delta", type=float, default=1.5, help="informant (social)")
    a.add_argument("--gamma", type=float, default=0.0, help="global-best")
    a.add_argument("--epsilon", type=float, default=1.0, help="step size")
    a.add_argument("--seed", type=int, default=42)
    return a.parse_args()

def main():
    args = parse_args()
    
    # 1. Load and prepare the dataset (split first, then fit scaling on train only)
    X, y = load_dataset(DATA)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = Standardiser().fit(X_train)
    X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)
    
    # 2. Configure and build the MLP
    activations = args.activations or ["tanh"] * len(args.hidden)
    builder = MLPBuilder(seed=args.seed)
    builder.set_dataset(X_train, y_train)
    builder.set_configuration({
        "hidden_layers": args.hidden,
        "activations": activations,
        "task": "binary_classification"
    })
    network, loss = builder.build()
    print(f"MLP: {X_train.shape[1]} -> {args.hidden} -> 1 | parameters (PSO dimension): "
          f"{network.count_parameters()}")
    
    # 3. Adapter (training data only)
    fitness = MLPFitnessAdapter(network=network, loss=loss, X_train=X_train, y_train=y_train)
    
    # 4. Informant strategy
    strategy = (NearestNeighborInformantSelection() if args.informants == "nearest"
                else RandomInformantSelection(seed=args.seed))
    
    # 5. PSO Population
    population = Population(
        fitness_function=fitness, informant_strategy=strategy,
        num_particles=args.particles, num_informants=args.k,
        num_iterations=args.iterations, dimension=network.count_parameters(),
        inertia_weight=args.alpha, cognitive_coefficient=args.beta,
        social_coefficient=args.delta, global_coefficient=args.gamma,
        step_size=args.epsilon, seed=args.seed,
    )
    
    # 6. Optimize
    best_position, best_fitness = population.optimize()
    print(f"Best training loss: {best_fitness:.5f} ({population.n_evaluations} fitness evaluations)")
    
    # 7. Install the best parameters
    network.set_parameters(best_position)
    
    # 8. Evaluate on unseen test data
    labels = probabilities_to_labels(network.predict(X_test))
    m = classification_metrics(y_test, labels)
    print(f"Test accuracy : {m['accuracy']:.4f}")
    print(f"Test precision: {m['precision']:.4f}")
    print(f"Test recall   : {m['recall']:.4f}")
    print(f"Test F1-score : {m['f1']:.4f}")
    print(f"Confusion (tp, tn, fp, fn): {m['tp']}, {m['tn']}, {m['fp']}, {m['fn']}")
    
if __name__ == "__main__":
    main()