---
layout: post
title:  "convolution"
date:   2025-09-06 23:57:10 +0800
categories: computer-vision
---
## Convolutional Layer

### Convolution Forward and Back Propagation

Given an input image $X$ and a filter $F$, one forward pass of convolution is $O = X \otimes F$.
The result $O$ is termed *feature map*.

$$
\begin{bmatrix}
    O_{11} & O_{12} \\
    O_{21} & O_{22}
\end{bmatrix} =
\begin{bmatrix}
    X_{11} & X_{12} & X_{13} \\
    X_{21} & X_{22} & X_{23} \\
    X_{31} & X_{32} & X_{33}
\end{bmatrix} \otimes
\begin{bmatrix}
    F_{11} & F_{12} \\
    F_{21} & F_{22}
\end{bmatrix}
$$

unfold the $\otimes$ operator, there are

$$
\begin{align*}
O_{11} &= X_{11} F_{11} + X_{12} F_{12} + X_{21} F_{21} + X_{22} F_{22} \\
O_{12} &= X_{12} F_{11} + X_{13} F_{12} + X_{22} F_{21} + X_{23} F_{22} \\
O_{21} &= X_{21} F_{11} + X_{22} F_{12} + X_{31} F_{21} + X_{32} F_{22} \\
O_{22} &= X_{22} F_{11} + X_{23} F_{12} + X_{32} F_{21} + X_{33} F_{22} \\
\end{align*}
$$

Express convolution to element-wise multiplication, assumed filter size $K_M \times K_N$, for a spatial point at $(i,j)$, there is

$$
O_{i,j} = \sum_{m}^{K_M} \sum_{n}^{K_N} X_{i+m, j+n} \cdot F_{m,n}
$$

The back propagation of $F_{11}$ given loss $\mathcal{L}$ is

$$
\frac{\partial \mathcal{L}}{\partial F_{11}} =
\frac{\partial \mathcal{L}}{\partial O_{11}} X_{11} +
\frac{\partial \mathcal{L}}{\partial O_{12}} X_{12} +
\frac{\partial \mathcal{L}}{\partial O_{21}} X_{21} +
\frac{\partial \mathcal{L}}{\partial O_{22}} X_{22}
$$

### Other Setups in A Convolutional Layer

#### Channel

Assume filter size $K_M \times K_N$; there are $C_{in}$ input channels, for a spatial point at $(i,j)$, there is

$$
O_{i,j} = \sum_{m}^{K_M} \sum_{n}^{K_N} \sum_{c}^{C_{in}} X_{i+m, j+n, c} \cdot F_{m,n}
$$

This means that one output/feature map needs $K_M \times K_N \times C_{in}$ CNN parameters.
Assume there are $C_{out}$ output channels (also termed the num of filters), total parameters required are $K_M \times K_N \times C_{in} \times C_{out}$ for $C_{out}$ feature maps.

#### Stride

Skip a number of $s$ pixels then do next convolution.

It can break spatial info as feature points from adjacent convolutions are likely together contribute the same semantic visual feature.

Use large stride when image resolution is high; small when low.

#### Padding

Insert zeros to the surroundings of input so that the output remains the same size as the input's.
For example for $O = X \otimes F$, having done $X$ padding by zeros to $X_{\text{padding}} \in \mathbb{R}^{5 \times 5}$, there is convolution result $O \in \mathbb{R}^{3 \times 3}$ same as input $X \in \mathbb{R}^{3 \times 3}$.

$$
\begin{bmatrix}
    O_{11} & O_{12} & O_{13} \\
    O_{21} & O_{22} & O_{23} \\
    O_{31} & O_{32} & O_{33}
\end{bmatrix} =
\begin{bmatrix}
    0 & 0 & 0 & 0 & 0\\
    0 & X_{11} & X_{12} & X_{13} & 0\\
    0 & X_{21} & X_{22} & X_{23} & 0\\
    0 & X_{31} & X_{32} & X_{33} & 0\\
    0 & 0 & 0 & 0 & 0\\
\end{bmatrix} \otimes
\begin{bmatrix}
    F_{11} & F_{12} \\
    F_{21} & F_{22}
\end{bmatrix}
$$

#### Pooling

### Typical Computation Cost of A Convolutional Layer

* Filter kernel size: $K \times K$
* Image size $I_M \times I_N$ divided by stride $s$: $\frac{I_M \times I_N}{s \times s}$
* The number of filters/channels: $C_{in}$ and $C_{out}$

Total: $(K \times K) \times \frac{I_M \times I_N}{s \times s} \times C_{in} \times C_{out}$

### Calculation Example

Given an image $224 \times 224 \times 3$, consider $11 \times 11$ kernel with stride by $4$.
There are $64$ filters.

* Num convolutions over a row/col: $56=224/4$

## Up-Convolution

Up-convolution typically surrounds entries with zeros and apply a typical convolution operation.

Below is an example.

$$
A = \begin{bmatrix}
1 & 2 \\
3 & 4
\end{bmatrix}
\quad\xRightarrow[\text{zero insertion and padding}]{}
P = \begin{bmatrix}
0 & 0 & 0 & 0 & 0 \\
0 & 1 & 0 & 2 & 0 \\
0 & 0 & 0 & 0 & 0 \\
0 & 3 & 0 & 4 & 0 \\
0 & 0 & 0 & 0 & 0
\end{bmatrix}
$$

Denote $\otimes$ as a convolution operator.
Below $P$ is convolved by kernel $K$.

$$
K\otimes P = \begin{bmatrix}
1 & 0 & 1 \\
0 & 1 & 0 \\
1 & 0 & 1
\end{bmatrix} \otimes
\begin{bmatrix}
0 & 0 & 0 & 0 & 0 \\
0 & 1 & 0 & 2 & 0 \\
0 & 0 & 0 & 0 & 0 \\
0 & 3 & 0 & 4 & 0 \\
0 & 0 & 0 & 0 & 0
\end{bmatrix} =
\begin{bmatrix}
1 & 0 & 2 \\
0 & 10 & 0 \\
3 & 0 & 4 \\
\end{bmatrix}
$$

## Convolutional Neural Network (CNN) vs Fully Convolutional Network (FCN)

CNN includes fully connected layers that result in loss of image spatial info but that is preserved in FCN.

||CNN|FCN|
|-|-|-|
|Arch|Combines convolutional and pooling layers, **followed by fully connected layers**|Consists only of convolutional, pooling, and upsampling layers.|
|Tasks|Image classification, object detection (with post-processing like bounding boxes).|Semantic segmentation, dense prediction, depth estimation, super-resolution tasks.|

## AlexNet

<div style="display: flex; justify-content: center;">
      <img src="{{ site.baseurl }}/assets/imgs/alexnet.png" width="30%" height="60%" alt="alexnet" />
</div>
</br>

### Why convolution then pooling, and why three convolutions then pooling

Intuitively,

* First and second convolution then pooling: contained large kernels $11 \times 11$ and $5 \times 5$, useful to quickly locate local image features and reduce image resolution; large kernels are good to capture semantic features.
* Multi-convolution then pooling: used small kernel of $3 \times 3$ for 3 times consecutively that enables learning highly abstract features; pooling discards spatial details, so delaying pooling allows the network to retain more fine-grained information for complex abstractions.

### Why two dense layers before the output dense layer

A dense layer is a fully connected layer $\bold{y}=\sigma(W\bold{x}+\bold{b})$.

Intuitively,

* First Dense Layer: Reshapes/flattens low-level features $256 \times 6 \times 6$ to a vector of size $9216$ higher-level abstractions. Activations primarily capture feature combinations (e.g., patterns like "edges forming an object").
* Second Dense Layer: Refines these abstractions into class-specific patterns, applying non-linear transformations to aggregate and focus the receptive fields.
* Third Dense Layer: output transform

Mathematical Insight:

* First Dense Layer: Local receptive field combinations (flattened convolutional outputs by $W_1 \in \mathbb{R}^{4096 \times 9216}$).
* Second Dense Layer: Global object representations (class scores by $W_2 \in \mathbb{R}^{4096 \times 4096}$).
* Third Dense Layer: output transform by $W_3 \in \mathbb{R}^{1000 \times 4096}$ with activation by softmax

Empirical study:

More dense layers can see $\text{ReLU}$ saturation that many activation values tend to get very large or zero, signaled redundancy in containing too many neurons (two layers are enough).

### Indicators of Saturation

#### Empirical Observations

* Training and validation accuracy/loss plateau even with extended training
* Larger/more convolution kernels do not yield better results
* Getting deeper/more layers does not give better results
* Getting wider/larger weight matrix/more neuron per layer does not give better results
* Having small/even no stride does not give better results

#### Theoretical Indicators

* If weight matrices $W$ have **small eigenvalues**, the weight matrix may not be effectively transforming the input space.
* If weight matrices $W$ have **large eigenvalues**, the transformations may be overly redundant or lead to gradient instability.
* A large fraction of neurons consistently output zero (dead neurons in ReLU layers), indicating wasted capacity.

Recall linear algebra that $W\bold{x}=\lambda\bold{x}$ means transforming input $\bold{x}$ by $W$ is same as getting scaled by $\lambda$.

If $\lambda \gg 0$, it leads to excessive amplification of inputs $\bold{x}$ along certain directions.

* Gradient Instability: Large eigenvalues propagate large gradients during back-propagation, which can destabilize training.
* Redundancy: Over-amplifying features may result in over-fitting or redundant transformations.

If $\lambda \approx 0$, it means low rank, not utilizing all degrees of freedom in the input.

* Some features are being ignored or not contributing to the output.

#### Human Evaluation

* Receptive Field Analysis: for low-level features are semantic to human understanding, one can manually review the convolution results of the first layer; if objects have too many filters repeatedly focusing on the same areas extracting similar features, it signals saturation.

## Convolution Example

```py
import numpy as np
import matplotlib.pyplot as plt

# Define pooling parameters
pool_size = (3, 3)  # Max pooling window size
stride = (3, 3)     # Stride size

# Helper function to add noise and dim pixels
def apply_noise_and_dim(arr, dim_level=30):
    dimmed_arr = arr.copy()
    for i in range(dimmed_arr.shape[0]):
        for j in range(dimmed_arr.shape[1]):
            if dimmed_arr[i][j] > 255:
                dimmed_arr[i][j] -= dim_level
    return dimmed_arr

# Define ReLU function
def relu(matrix):
    return np.maximum(0, matrix)

# Function to perform max pooling
def max_pooling(matrix, pool_size, stride):
    output_height = (matrix.shape[0] - pool_size[0]) // stride[0] + 1
    output_width = (matrix.shape[1] - pool_size[1]) // stride[1] + 1
    pooled_output = np.zeros((output_height, output_width))
    
    for i in range(0, matrix.shape[0] - pool_size[0] + 1, stride[0]):
        for j in range(0, matrix.shape[1] - pool_size[1] + 1, stride[1]):
            pooled_output[i // stride[0], j // stride[1]] = np.max(
                matrix[i:i + pool_size[0], j:j + pool_size[1]]
            )
    return pooled_output


# Function to normalize matrix
def normalize(matrix, target_min=0, target_max=255):
    """
    Normalize the input matrix to a specified range.

    Parameters:
        matrix (np.ndarray): Input matrix (e.g., convolution output).
        target_min (float): Minimum value of the target range.
        target_max (float): Maximum value of the target range.

    Returns:
        np.ndarray: Normalized matrix.
    """
    matrix_min = np.min(matrix)
    matrix_max = np.max(matrix)

    # Avoid division by zero in case all values are the same
    if matrix_max == matrix_min:
        return np.full_like(matrix, target_min)

    # Apply min-max normalization
    normalized_matrix = (matrix - matrix_min) / (matrix_max - matrix_min)  # Scale to [0, 1]
    normalized_matrix = normalized_matrix * (target_max - target_min) + target_min  # Scale to [target_min, target_max]
    return normalized_matrix

# Step 1: Create synthetic MNIST-like images of 6 and 9
def create_synthetic_6():
    image = np.array([
        [0]*28 for _ in range(28)
    ])

    # Draw the digit "6" within the 28x28 grid
    image[1:26, 7:24] = [
        [0,   0,   0,   0,   0,   0,   0,   0,   78,  111, 89,  0,   0,   0,   0,   0,   0,   ],
        [0,   0,   0,   0,   0,   0,   0,   101, 199, 199, 66,  0,   0,   0,   0,   0,   0,   ],
        [0,   0,   0,   0,   0,   0,   111, 255, 255, 87,  11,  0,   0,   0,   0,   0,   0,   ],
        [0,   0,   0,   0,   0,   123, 255, 255, 87,  0,   0,   0,   0,   0,   0,   0,   0,   ],
        [0,   0,   0,   0,   144, 255, 255, 87,  0,   0,   0,   0,   0,   0,   0,   0,   0,   ],
        [0,   0,   0,   123, 255, 255, 78,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   ],
        [0,   0,   113, 189, 255, 82,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   ],
        [0,   0,   189, 255, 255, 82,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   ],
        [0,   67,  213, 255, 82,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   ],
        [62,  233, 255, 255, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   ],
        [82,  255, 255, 82,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   ],
        [82,  255, 255, 82,  77,  77,  77,  77,  77,  77,  77,  0,   0,   0,   0,   0,   0,   ],
        [82,  255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 82,  0,   0,   0,   0,   0,   ],
        [82,  255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 82,  0,   0,   0,   0,   ],
        [82,  255, 255, 77,  43,  0,   0,   0,   0,   177, 255, 255, 82,  82,  0,   0,   0,   ],
        [82,  255, 255, 44,  0,   0,   0,   0,   0,   0,   153, 255, 255, 82,  82,  0,   0,   ],
        [82,  255, 255, 32,  0,   0,   0,   0,   0,   0,   0,   132, 255, 255, 82,  0,   0,   ],
        [82,  255, 255, 11,  0,   0,   0,   0,   0,   0,   0,   92,  255, 255, 82,  0,   0,   ],
        [82,  255, 255, 31,  0,   0,   0,   0,   0,   0,   0,   132, 255, 255, 82,  0,   0,   ],
        [72,  255, 255, 132, 0,   0,   0,   0,   0,   0,   0,   255, 255, 231, 82,  0,   0,   ],
        [75,  255, 255, 211, 112, 0,   0,   0,   0,   0,   201, 255, 255, 82,  0,   0,   0,   ],
        [78,  188, 255, 255, 131, 99,  0,   0,   0,   156, 255, 255, 178, 0,   0,   0,   0,   ],
        [0,   77,  167, 255, 132, 123, 77,  55,  65,  255, 255, 82,  0,   0,   0,   0,   0,   ],
        [0,   0,   82,  255, 255, 255, 255, 255, 255, 255, 82,  0,   0,   0,   0,   0,   0,   ],
        [0,   0,   0,   77,  255, 255, 255, 255, 255, 82,  0,   0,   0,   0,   0,   0,   0,   ],
    ]

    # Add some noise to make it look more realistic
    image = apply_noise_and_dim(image)
    return image

def create_synthetic_9():
    # MNIST-like representation of the number 9 as a 180-degree rotation of digit 6
    digit_6 = create_synthetic_6()
    image = np.rot90(digit_6, 2)
    return image

# Step 2: Define convolutional kernels
def create_diagonal_edge_kernel():
    kernel = np.array([[ 3,  1,  0, -1, -3],
                    [ 1,  3,  1,  0, -1],
                    [ 0,  1,  3,  1,  0],
                    [-1,  0,  1,  3,  1],
                    [-3, -1,  0,  1,  3]]
                )
    kernel = 1/np.sum(kernel) * kernel
    return kernel

def create_rot_diagonal_edge_kernel():
    kernel = np.array([[ 3,  1,  0, -1, -3],
                    [ 1,  3,  1,  0, -1],
                    [ 0,  1,  3,  1,  0],
                    [-1,  0,  1,  3,  1],
                    [-3, -1,  0,  1,  3]]
                )
    kernel = np.rot90(kernel)
    kernel = 1/np.sum(kernel) * kernel
    return kernel

# Step 3: Apply convolution
def convolve(image, kernel):
    kernel_height, kernel_width = kernel.shape
    image_height, image_width = image.shape
    output_height = image_height - kernel_height + 1
    output_width = image_width - kernel_width + 1
    output = np.zeros((output_height, output_width))
    
    for i in range(output_height):
        for j in range(output_width):
            output[i, j] = np.sum(image[i:i+kernel_height, j:j+kernel_width] * kernel)
    
    return output

# Create synthetic images
image_6 = create_synthetic_6()

# Create kernels
diagonal_kernel = create_diagonal_edge_kernel()
rot_diagonal_kernel = create_rot_diagonal_edge_kernel()

# Apply convolution
output_6_diagonal = convolve(image_6, diagonal_kernel)
output_6_rot_diagonal = convolve(image_6, rot_diagonal_kernel)
output_6_diagonal = normalize(output_6_diagonal)
output_6_rot_diagonal = normalize(output_6_rot_diagonal)

# Apply relu
output_6_relu_diagonal = relu(output_6_diagonal)
output_6_relu_rot_diagonal = relu(output_6_rot_diagonal)

# Apply max pooling
output_6_pooled_diagonal = max_pooling(output_6_relu_diagonal, pool_size, stride)
output_6_pooled_rot_diagonal = max_pooling(output_6_relu_rot_diagonal, pool_size, stride)

# Plot the results
fig = plt.figure(figsize=(10, 8))

plt.subplot(3, 3, 4)
plt.title("'6' Image")
plt.imshow(image_6, cmap='gray')

ax1 = fig.add_subplot(3, 3, 1)
ax1.set_title("Diagonal Edge Kernel")
ax1.imshow(diagonal_kernel, cmap='gray', interpolation='nearest')
ax1.axis('off')
# Scale both x and y axes and center the image
scale_factor = 3  # Adjust to shrink or enlarge
image_size = diagonal_kernel.shape[0] / 2  # Half the image size to calculate limits
ax1.set_xlim(-image_size * scale_factor+image_size, image_size * scale_factor+image_size)  # Scale x-axis
ax1.set_ylim(image_size * scale_factor+image_size, -image_size * scale_factor+image_size)  # Scale y-axis (reverse to maintain orientation)


plt.subplot(3, 3, 2)
plt.title("Conv (6)")
plt.imshow(output_6_diagonal, cmap='gray')

plt.subplot(3, 3, 3)
plt.title("Max Pooled (6)")
plt.imshow(output_6_pooled_diagonal, cmap='gray')


ax2 = fig.add_subplot(3, 3, 7)
ax2.set_title("Rotated Diagonal Edge Kernel")
ax2.imshow(rot_diagonal_kernel, cmap='gray', interpolation='nearest')
ax2.axis('off')
# Scale both x and y axes and center the image
scale_factor = 3  # Adjust to shrink or enlarge
image_size = rot_diagonal_kernel.shape[0] / 2  # Half the image size to calculate limits
ax2.set_xlim(-image_size * scale_factor+image_size, image_size * scale_factor+image_size)  # Scale x-axis
ax2.set_ylim(image_size * scale_factor+image_size, -image_size * scale_factor+image_size)  # Scale y-axis (reverse to maintain orientation)


plt.subplot(3, 3, 8)
plt.title("Conv (6)")
plt.imshow(output_6_rot_diagonal, cmap='gray')

plt.subplot(3, 3, 9)
plt.title("Max Pooled (6)")
plt.imshow(output_6_pooled_rot_diagonal, cmap='gray')

plt.tight_layout()
plt.show()

```