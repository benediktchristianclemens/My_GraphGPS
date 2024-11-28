# test_best_model.py
import torch
import matplotlib.pyplot as plt
from my_graphgps.network.model import GPSNetwork
import os  # Added to handle directory operations

def train_and_evaluate_best_model(curr_data, num_features, num_classes, best_params, data_name=""):
    """
    Train and evaluate the best model with optimal parameters.

    Parameters
    ----------
    curr_data : torch_geometric.data.Data
        The PyG Data object containing the graph data.
    num_features : int
        Number of input features for the model.
    num_classes : int
        Number of output classes for classification.
    best_params : dict
        Best hyperparameters as a dictionary.
    data_name : str, optional
        Name of the dataset variant for plot naming.

    Returns
    -------
    best_model : torch.nn.Module
        The trained model.
    """
    best_hidden_channels, best_heads, best_dropout, best_learning_rate, best_epochs = best_params

    # Initialize the best model
    best_model = GPSNetwork(
        in_channels=num_features,
        hidden_channels=best_hidden_channels,
        out_channels=num_classes,
        heads=best_heads,
        dropout=best_dropout
    )

    # Optimizer and loss function
    optimizer = torch.optim.Adam(best_model.parameters(), lr=best_learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    # Lists to store training and validation losses
    train_losses = []
    val_losses = []

    def train_best():
        """Perform a single training step."""
        best_model.train()
        optimizer.zero_grad()
        out = best_model(curr_data.x, curr_data.edge_index)
        loss = criterion(out[curr_data.train_mask], curr_data.y[curr_data.train_mask])
        loss.backward()
        optimizer.step()
        return loss.item()

    def calculate_val_loss():
        """Calculate validation loss."""
        best_model.eval()
        with torch.no_grad():
            out = best_model(curr_data.x, curr_data.edge_index)
            val_loss = criterion(out[curr_data.val_mask], curr_data.y[curr_data.val_mask])
        return val_loss.item()

    # Train the model and track losses
    for epoch in range(1, best_epochs + 1):
        train_loss = train_best()
        train_losses.append(train_loss)  # Track training loss

        val_loss = calculate_val_loss()  # Calculate and track validation loss
        val_losses.append(val_loss)

        # Print progress every 10 epochs or at the last epoch
        if epoch % 10 == 0 or epoch == best_epochs:
            print(f"Epoch {epoch:03d}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    # Plot training and validation loss curves
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, best_epochs + 1), train_losses, label="Train Loss", color="blue")
    plt.plot(range(1, best_epochs + 1), val_losses, label="Validation Loss", color="orange")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Train and Validation Loss Curve for {data_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Ensure the 'results' directory exists
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Save the plot to the 'results' folder
    plot_path = os.path.join(results_dir, f"train_val_loss_{data_name}.png")
    plt.savefig(plot_path)
    plt.show()

    return best_model
