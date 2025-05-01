## Setup Instructions

1. Create a Conda Environment

    Use the provided environment.yml file to set up the environment:

    ```
    conda env create -f environment.yml
    ```

1. Activate the Environment

    Once created, activate the environment using:

    ```
    conda activate mnist-vit
    ```

1. Install Optional Dependencies

    If you need matplotlib, install it via pip:

    ```
    pip install matplotlib
    ```

1. Running the Code

    Ensure you have Python installed and the environment activated.

    Run the script using:

    ```
    python mnist-vit.py
    ```

1. Notes

    The script uses the standard MNIST dataset from torchvision.
    For reproducibility, set a random seed (e.g., torch.manual_seed(42)).
    Modify the train_loader if you want to use a custom dataset.
    Environment Requirements

    PyTorch and TorchVision for deep learning.
    NumPy for numerical operations.
    Optional: matplotlib for visualization (install via pip).
    License
    This project is open-source and available under the MIT License.

    Contributing
    If you'd like to contribute, feel free to open an issue or submit a pull request. Please ensure your code follows the project's style and includes tests where applicable.

    Contact
    For questions or feedback, reach out via GitHub Issues.