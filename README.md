# NFMix: Mixtures of Normalizing Flows

Exploration of mixtures of normalizing flows

## Installation

This project uses `uv` for environment and dependency management.

1.  **Create and activate a virtual environment:**
    ```bash
    # Create the environment (in .venv directory)
    uv venv

    # Activate the environment (syntax depends on your shell)
    # For bash/zsh:
    source .venv/bin/activate
    # For fish:
    source .venv/bin/activate.fish
    # For Powershell:
    .venv\Scripts\Activate.ps1
    ```

2.  **Install dependencies:**
    Once the environment is activated, install the required packages. You need to choose between CPU or CUDA versions of PyTorch and specify the corresponding index URL.

    *   **For CPU:**
        ```bash
        uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        uv pip install .
        ```

    *   **For CUDA (cu126):**
        *Note: Replace `cu126` with your specific CUDA version if different.*
        ```bash
        uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
        uv pip install .
        ```

    This installs the core dependencies (`zuko`, `seaborn`, `pyro-ppl`) plus the selected PyTorch variant.

## Development Installation

If you plan to contribute to the project, install the development dependencies, including pre-commit hooks. Choose the appropriate command based on your PyTorch version (CPU or CUDA).

1.  **Install Development Dependencies:**
    Make sure your virtual environment is activated.

        ```bash
        uv pip install .[dev]
        ```

2.  **Install Git hooks:**
    Set up the pre-commit hooks defined in `.pre-commit-config.yaml`.
    ```bash
    pre-commit install
    ```
    Now, the hooks will run automatically before each commit.

## Examples

- `examples/testing_nfmixes_zuko.ipynb`: Some examples of mixtures of normalising flows on two moons dataset.
- ... (add other examples if needed)
