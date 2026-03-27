
# 3D Voxel Classification: Cube vs Sphere

A deep learning project for binary classification of 3D voxel objects (cubes vs spheres) using synthetic data augmentation and a custom 3D Convolutional Neural Network.

## Overview

This project generates synthetic 3D voxel data (cubes and spheres) with various deformations, fractures, rotations, and erosions to simulate realistic fossil-like structures. A custom 3D CNN model is then trained to classify these objects into two categories.

## Project Structure

```
3dClassification/
├── database/                    # Generated data storage
│   ├── cubes.npy               # Augmented cube dataset
│   └── spheres.npy             # Augmented sphere dataset
├── models/                      # Trained models
│   └── 3dClassificationTest/   # Model checkpoints and logs
├── 0_datageneration_paleovox.py # Data generation script
└── 1_testmodel.py              # Model training and evaluation script
```

## Dependencies

- Python 3.x
- TensorFlow / Keras
- NumPy
- Open3D
- PaleoVox
- Seaborn
- Matplotlib
- python-telegram-bot

## Scripts Documentation

### 1. Data Generation Script (`0_datageneration_paleovox.py`)

This script generates synthetic 3D voxel data for cubes and spheres with realistic deformations.

#### Key Functions

##### Basic Shape Generation
- `create_sphere(voxel_size=32, radius=12, center=None)`: Creates a binary sphere in a 3D voxel grid
- `create_cube(voxel_size=32, cube_size=16, center=None)`: Creates a binary cube in a 3D voxel grid

##### Data Augmentation Functions

**Deformation:**
```python
deformation(voxel_array, compaction_factor=0.6, compaction_axis=0)
```
Compresses the voxel object along a specified axis.

**Propagator Fracture:**
```python
propagator_fracture(voxel_grid, max_position=10, return_both=False, pr=False)
```
Generates synthetic fracture patterns within a 3D voxel volume using stochastic propagation. This simulates natural-looking cracks and fractures.

**Rotation:**
```python
rotate_voxel(voxel_array, x_angle, y_angle, z_angle)
```
Rotates the voxel object by specified angles in radians.

**Erosion:**
```python
erotion_general(voxel, axis_idx, increment_min=0.75, pr=False)
```
Randomly erodes surface voxels to create natural wear patterns.

##### Data Pipeline
```python
pipeline(voxel)
```
Applies a sequence of augmentations:
1. Random deformation (compaction factor: 0.6-0.95)
2. First fracture
3. Random rotation (0-360° in all axes)
4. Second fracture
5. Random rotation
6. Third fracture
7. General erosion

##### Data Generation
The script generates 10,000 augmented samples for both cubes and spheres, saving them as:
- `database/cubes.npy` (10,000 × 32 × 32 × 32)
- `database/spheres.npy` (10,000 × 32 × 32 × 32)

### 2. Model Training Script (`1_testmodel.py`)

This script loads the generated data, defines a 3D CNN architecture, trains the model, and evaluates its performance.

#### Data Loading and Preparation

```python
# Load data
cubes = np.load('database/cubes.npy')      # Shape: (10000, 32, 32, 32)
spheres = np.load('database/spheres.npy')  # Shape: (10000, 32, 32, 32)

# Create labels
cubes_labels = np.zeros(len(cubes))        # Label: 0 for cubes
spheres_labels = np.ones(len(spheres))     # Label: 1 for spheres

# Combine and shuffle
x_data = np.expand_dims(np.concatenate((cubes, spheres)), axis=-1)  # Add channel dimension
y_data = np.expand_dims(np.concatenate((cubes_labels, spheres_labels)), axis=-1)

# Train/validation/test split (80/10/10)
x_train, y_train = x_data[:0.8], y_data[:0.8]
x_val, y_val = x_data[0.8:0.9], y_data[0.8:0.9]
x_test, y_test = x_data[0.9:], y_data[0.9:]
```

#### Model Architecture

The `CNN_3D` function builds a custom 3D convolutional neural network:

```python
CNN_3D(
    inputs = Input((32, 32, 32, 1)),
    filters = [25, 50, 100],           # Filters per convolutional block
    kernel = [(8,8,8), (4,4,4), (2,2,2)],  # Kernel sizes
    pad_type = 'valid',                 # Padding type
    pool = (2,2,2),                    # Pooling window size
    stride = (2,2,2),                  # Pooling stride
    nodes = [50, 25],                  # Dense layer nodes
    DP = 5,                            # Dropout percentage
    n_final = 1,                       # Output neurons
    final_act_func = 'sigmoid',        # Binary classification output
    L1 = 1e-6, L2 = 1e-6              # L1/L2 regularization
)
```

**Architecture Summary:**
- 3 Convolutional blocks with decreasing kernel sizes
- AveragePooling3D for downsampling
- Flatten layer
- 2 Dense layers with Dropout
- Sigmoid output for binary classification

#### Training Configuration

```python
model_CNN.compile(
    optimizer = keras.optimizers.Adam(learning_rate=0.001),
    loss = "binary_crossentropy",
    metrics = []  # Add custom metrics as needed
)

# Training parameters
batch_size = 64
num_epochs = 30

# Callbacks
callbacks = standard_callbacks(
    folder_name='3dClassificationTest',
    folder_path='models',
    patiences=[1000, 1000],          # Early stopping & ReduceLR patience
    monitor=['val_loss'],            # Monitor validation loss
    flow_direction=['min']           # Minimize loss
)
```

#### Evaluation

The script evaluates the trained model on a test set and provides:
- Loss value
- Predictions with visualization of random test samples
- Training history plots (loss curves)
- Model architecture visualization

## Usage

### 1. Generate Data
```python
# Run the data generation script
python 0_datageneration_paleovox.py

# This will create:
# - database/cubes.npy
# - database/spheres.npy
```

### 2. Train the Model
```python
# Run the model training script
python 1_testmodel.py

# Output will be saved to:
# - models/3dClassificationTest/model.h5 (full model)
# - models/3dClassificationTest/val_loss_min.keras (best model)
# - models/3dClassificationTest/images/ (training plots)
```

## Visualization

The project includes visualization capabilities:
- `plot_voxels()`: Renders 3D voxel objects
- Training history plots showing loss curves
- Model architecture diagrams
- Random sample predictions with visualizations

## Expected Results

With the current configuration:
- **Training time**: ~30 epochs (adjustable)
- **Test accuracy**: High binary classification performance (target > 99%)
- **Loss**: Binary cross-entropy minimized to < 0.01

## Notes

- The project uses GPU acceleration when available (configured via `get_gpu(0)`)
- All results are saved with timestamps for reproducibility
- Random seeds are set for consistent training
- Model checkpoints save the best performing model based on validation loss


