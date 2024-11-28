# hyperparam_tuning.py
import random
from itertools import product
import torch
from my_graphgps.network.model import GPSNetwork

def hyperparameter_tuning(curr_data, num_features, num_classes, param_grid, num_samples=30, random_seed=42):
    """
    Perform hyperparameter tuning for a GPSNetwork model using random sampling.

    Parameters
    ----------
    curr_data : torch_geometric.data.Data
        The PyG Data object containing the graph data.
    num_features : int
        Number of input features for the model.
    num_classes : int
        Number of output classes for classification.
    param_grid : dict
        Dictionary defining the parameter grid for hyperparameter tuning.
        Keys should be parameter names and values should be lists of possible values.
    num_samples : int, optional (default=30)
        Number of random parameter combinations to sample from the grid.
    random_seed : int, optional (default=42)
        Seed for random sampling to ensure reproducibility.

    Returns
    -------
    tuple
        Best parameters and their corresponding test accuracy as (best_params, best_acc).
    """
    # Generate all possible combinations of parameters
    param_combinations = list(product(*param_grid.values()))

    # Select a random subset of parameter combinations to evaluate
    random.seed(random_seed)
    random_combinations = random.sample(param_combinations, min(num_samples, len(param_combinations)))

    # Function to train and evaluate the model for a given set of parameters
    def train_and_evaluate(params):
        hidden_channels, heads, dropout, learning_rate, epochs = params

        # Initialize the model with the current parameters
        model = GPSNetwork(
            in_channels=num_features,
            hidden_channels=hidden_channels,
            out_channels=num_classes,
            heads=heads,
            dropout=dropout
        )

        # Optimizer and loss function
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()

        # Training function
        def train():
            model.train()
            optimizer.zero_grad()
            out = model(curr_data.x, curr_data.edge_index)
            loss = criterion(out[curr_data.train_mask], curr_data.y[curr_data.train_mask])
            loss.backward()
            optimizer.step()
            return loss.item()

        # Testing function
        def test(mask):
            model.eval()
            with torch.no_grad():
                out = model(curr_data.x, curr_data.edge_index)
                pred = out.argmax(dim=1)
                correct = pred[mask] == curr_data.y[mask]
                acc = int(correct.sum()) / int(mask.sum())
            return acc

        # Train the model
        for epoch in range(epochs):
            loss = train()
            if epoch % 10 == 0:
                train_acc = test(curr_data.train_mask)
                val_acc = test(curr_data.val_mask)
                print(f"Epoch {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        # Test the final model
        test_acc = test(curr_data.test_mask)
        print(f"Test Accuracy: {test_acc:.4f}")
        return test_acc

    # Evaluate random parameter combinations
    results = []
    for params in random_combinations:
        print(f"Evaluating parameters: {params}")
        test_acc = train_and_evaluate(params)
        results.append((params, test_acc))

    # Find the best parameter combination
    best_params, best_acc = max(results, key=lambda x: x[1])
    print(f"Best Parameters: {best_params}, Best Test Accuracy: {best_acc:.4f}")

    return best_params, best_acc
