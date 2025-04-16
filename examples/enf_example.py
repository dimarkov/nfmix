"""
Example usage of Equivariant Normalizing Flows.

This script demonstrates how to use the equivariant normalizing flows
implemented in the nfmix package.
"""

import os

# Import the nfmix package
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import make_moons

sys.path.append(os.path.abspath(".."))
from nfmix.equivariant import E3EquivariantFlow, EquivariantNormalizingFlow
from nfmix.utils.utils import apply_rotation, random_rotation_matrix

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def generate_2d_data(n_samples=1000, noise=0.1):
    """Generate 2D data for testing equivariant flows."""
    # Generate two moons dataset
    X, y = make_moons(n_samples=n_samples, noise=noise)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    return X, y


def generate_3d_point_cloud(n_samples=100, n_points=10):
    """Generate 3D point cloud data for testing E(3) equivariant flows."""
    # Generate random point clouds (e.g., simple geometric shapes)
    point_clouds = []

    for _ in range(n_samples):
        # Generate a simple shape (e.g., a cube with random perturbations)
        points = torch.rand(n_points, 3) * 2 - 1  # Points in [-1, 1]^3

        # Apply random rotation for data augmentation
        rotation = random_rotation_matrix(dim=3)
        points = apply_rotation(points.unsqueeze(0), rotation).squeeze(0)

        point_clouds.append(points)

    # Stack into a batch
    point_clouds = torch.stack(point_clouds)

    return point_clouds


def train_equivariant_flow_2d():
    """Train an equivariant normalizing flow on 2D data."""
    print("Training equivariant normalizing flow on 2D data...")

    # Generate data
    X, y = generate_2d_data(n_samples=1000, noise=0.1)

    # Create dataloader
    dataset = torch.utils.data.TensorDataset(X)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=True)

    # Create equivariant flow
    flow = EquivariantNormalizingFlow(
        dim=2, hidden_features=[64, 64], num_layers=3, solver="dopri5"
    ).to(device)

    # Create optimizer
    optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3)

    # Train the flow
    n_epochs = 100
    losses = []

    for epoch in range(n_epochs):
        epoch_loss = 0.0

        for batch in dataloader:
            x = batch[0].to(device)

            # Compute loss
            loss = -flow.log_prob(x).mean()

            # Update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Record average loss
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.4f}")

    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Negative Log Likelihood")
    plt.title("Training Loss")
    plt.savefig("enf_2d_loss.png")

    # Visualize the learned distribution
    visualize_2d_flow(flow, X)

    return flow


def train_e3_equivariant_flow():
    """Train an E(3) equivariant normalizing flow on 3D point cloud data."""
    print("Training E(3) equivariant normalizing flow on 3D point cloud data...")

    # Generate data
    point_clouds = generate_3d_point_cloud(n_samples=1000, n_points=10)

    # Flatten point clouds for the flow
    batch_size, n_points, dim = point_clouds.shape
    flat_point_clouds = point_clouds.reshape(batch_size, -1)

    # Create dataloader
    dataset = torch.utils.data.TensorDataset(flat_point_clouds)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=True)

    # Create E(3) equivariant flow
    flow = E3EquivariantFlow(
        n_points=n_points, point_dim=dim, hidden_dim=64, num_layers=3, solver="dopri5"
    ).to(device)

    # Create optimizer
    optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3)

    # Train the flow
    n_epochs = 50
    losses = []

    for epoch in range(n_epochs):
        epoch_loss = 0.0

        for batch in dataloader:
            x = batch[0].to(device)

            # Compute loss
            loss = -flow.log_prob(x).mean()

            # Update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Record average loss
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.4f}")

    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Negative Log Likelihood")
    plt.title("Training Loss")
    plt.savefig("enf_3d_loss.png")

    # Visualize samples from the flow
    visualize_3d_flow(flow)

    return flow


def visualize_2d_flow(flow, data):
    """Visualize the learned 2D distribution."""
    # Generate samples from the flow
    with torch.no_grad():
        samples = flow.sample(1000).cpu().numpy()

    # Convert data to numpy
    data = data.cpu().numpy()

    # Plot the data and samples
    plt.figure(figsize=(12, 5))

    # Plot original data
    plt.subplot(1, 2, 1)
    plt.scatter(data[:, 0], data[:, 1], alpha=0.5, label="Data")
    plt.title("Original Data")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()

    # Plot samples from the flow
    plt.subplot(1, 2, 2)
    plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5, label="Samples")
    plt.title("Samples from Flow")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()

    plt.tight_layout()
    plt.savefig("enf_2d_samples.png")

    # Test equivariance by applying a rotation to the data
    test_equivariance_2d(flow)


def test_equivariance_2d(flow):
    """Test the equivariance property of the flow by applying rotations."""
    # Generate a batch of points
    x = torch.randn(100, 2).to(device)

    # Compute log probability of x
    log_prob_x = flow.log_prob(x).cpu().numpy()

    # Apply a rotation to x
    angle = np.pi / 4  # 45 degrees
    c, s = np.cos(angle), np.sin(angle)
    rotation = torch.tensor([[c, -s], [s, c]], dtype=torch.float32).to(device)
    x_rotated = torch.matmul(x, rotation.T)

    # Compute log probability of rotated x
    log_prob_x_rotated = flow.log_prob(x_rotated).cpu().numpy()

    # Check if log probabilities are similar (they should be for an equivariant flow)
    diff = np.abs(log_prob_x - log_prob_x_rotated).mean()

    print(f"Mean absolute difference in log probabilities after rotation: {diff:.6f}")
    print("This should be close to zero for a perfectly equivariant flow.")

    # Plot the results
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(
        x.cpu().numpy()[:, 0], x.cpu().numpy()[:, 1], c=log_prob_x, cmap="viridis"
    )
    plt.colorbar(label="Log Probability")
    plt.title("Log Probability of Original Points")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.subplot(1, 2, 2)
    plt.scatter(
        x_rotated.cpu().numpy()[:, 0],
        x_rotated.cpu().numpy()[:, 1],
        c=log_prob_x_rotated,
        cmap="viridis",
    )
    plt.colorbar(label="Log Probability")
    plt.title("Log Probability of Rotated Points")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.tight_layout()
    plt.savefig("enf_2d_equivariance.png")


def visualize_3d_flow(flow):
    """Visualize samples from the 3D point cloud flow."""
    # Generate samples from the flow
    with torch.no_grad():
        flat_samples = flow.sample(5).cpu()

    # Reshape to point clouds
    point_clouds = flow.reshape_points(flat_samples)

    # Plot the point clouds
    fig = plt.figure(figsize=(15, 10))

    for i in range(min(5, len(point_clouds))):
        ax = fig.add_subplot(1, 5, i + 1, projection="3d")
        points = point_clouds[i].numpy()
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c=range(len(points)),
            cmap="viridis",
        )
        ax.set_title(f"Sample {i+1}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_zlim(-2, 2)

    plt.tight_layout()
    plt.savefig("enf_3d_samples.png")

    # Test equivariance by applying a rotation to the data
    test_equivariance_3d(flow)


def test_equivariance_3d(flow):
    """Test the equivariance property of the E(3) flow by applying rotations and translations."""
    # Generate a batch of point clouds
    batch_size = 5
    n_points = flow.n_points
    point_dim = flow.point_dim

    # Create random point clouds
    point_clouds = torch.randn(batch_size, n_points, point_dim).to(device)
    flat_point_clouds = point_clouds.reshape(batch_size, -1)

    # Compute log probability of original point clouds
    log_prob_original = flow.log_prob(flat_point_clouds).cpu().numpy()

    # Apply a random rotation to each point cloud
    rotated_point_clouds = []
    for i in range(batch_size):
        rotation = random_rotation_matrix(dim=3).to(device)
        rotated = apply_rotation(point_clouds[i : i + 1], rotation)
        rotated_point_clouds.append(rotated.squeeze(0))

    rotated_point_clouds = torch.stack(rotated_point_clouds)
    flat_rotated_point_clouds = rotated_point_clouds.reshape(batch_size, -1)

    # Compute log probability of rotated point clouds
    log_prob_rotated = flow.log_prob(flat_rotated_point_clouds).cpu().numpy()

    # Apply a random translation to each point cloud
    translated_point_clouds = []
    for i in range(batch_size):
        translation = torch.randn(3).to(device)
        translated = point_clouds[i] + translation.unsqueeze(0)
        translated_point_clouds.append(translated)

    translated_point_clouds = torch.stack(translated_point_clouds)
    flat_translated_point_clouds = translated_point_clouds.reshape(batch_size, -1)

    # Compute log probability of translated point clouds
    log_prob_translated = flow.log_prob(flat_translated_point_clouds).cpu().numpy()

    # Check if log probabilities are similar
    diff_rotation = np.abs(log_prob_original - log_prob_rotated).mean()
    diff_translation = np.abs(log_prob_original - log_prob_translated).mean()

    print(
        f"Mean absolute difference in log probabilities after rotation: {diff_rotation:.6f}"
    )
    print(
        f"Mean absolute difference in log probabilities after translation: {diff_translation:.6f}"
    )
    print("These should be close to zero for a perfectly E(3) equivariant flow.")


if __name__ == "__main__":
    # Train and evaluate 2D equivariant flow
    flow_2d = train_equivariant_flow_2d()

    # Train and evaluate 3D equivariant flow
    flow_3d = train_e3_equivariant_flow()
