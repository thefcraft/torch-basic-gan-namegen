**GAN for Generating Names**

This project implements a Generative Adversarial Network (GAN) using PyTorch to generate Indian female names. The GAN consists of a generator and a discriminator trained on a dataset of Indian female names.

### Dataset
The dataset used for training is sourced from the "Indian-Female-Names.csv" file, containing a list of Indian female names. Names of a specific length (5 characters in this implementation) are selected for training.

### Model Architecture
- **Generator:** The generator is a simple feedforward neural network consisting of two linear layers with leaky ReLU activation functions. It takes random noise as input and generates synthetic Indian female names.
- **Discriminator:** The discriminator is also a feedforward neural network consisting of two linear layers with leaky ReLU activation functions. It takes a name (real or generated) as input and predicts whether it is a real Indian female name or not.

### Training
The GAN is trained using the adversarial training approach:
1. The discriminator is trained to distinguish between real and generated names.
2. The generator is trained to generate names that are classified as real by the discriminator.

### Hyperparameters
- Learning Rate: 3e-4
- Latent Dimension: 32
- Input Dimension: 5 * 26 (5 characters, each represented as a one-hot encoded vector of length 26)
- Batch Size: 32
- Number of Epochs: 64

### Results
After training, the generator is capable of generating Indian female names. The generated names are evaluated using the discriminator, and the top-ranked names with their corresponding scores are saved to an output file ("output.txt").

### Files
- `gen_model.pt`: Saved model weights and optimizer state for the generator.
- `disc_model.pt`: Saved model weights and optimizer state for the discriminator.
- `output.txt`: Text file containing the top-ranked generated names with their scores.

### Dependencies
- PyTorch
- Pandas
- NumPy
- scikit-learn

### Usage
1. Clone the repository.
2. Install the required dependencies.
3. Run the script to train the GAN and generate Indian female names.
4. Check the output file for the generated names.

### Notes
- This is a basic implementation and may require further tuning for better performance.
- Additional preprocessing or post-processing steps may be needed depending on the dataset and desired output.
