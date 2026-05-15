# Neural Style Transfer

## Description
Implementation of Neural Style Transfer (NST) using TensorFlow and the VGG19 convolutional neural network. NST is an algorithm that combines the content of one image with the artistic style of another to generate a new stylized image.

## Requirements
- Python 3.5
- NumPy 1.15
- TensorFlow 1.12

## Files

| File | Description |
|------|-------------|
| `0-neural_style.py` | NST class initialization and image scaling |
| `1-neural_style.py` | Load VGG19 model with average pooling layers |
| `2-neural_style.py` | Gram matrix calculation |
| `3-neural_style.py` | Extract style and content features |
| `4-neural_style.py` | Single layer style cost |
| `5-neural_style.py` | Total style cost across all layers |
| `6-neural_style.py` | Content cost |
| `7-neural_style.py` | Total cost |
| `8-neural_style.py` | Gradient computation using GradientTape |
| `9-neural_style.py` | Image generation with Adam optimizer |
| `10-neural_style.py` | Variational cost for smoother output |

## How It Works

### Model
VGG19 is used as the base model with MaxPooling layers replaced by AveragePooling for better gradient flow. The model is frozen (non-trainable) and used only to extract feature maps.

### Style Layers
```python
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
```

### Content Layer
```python
content_layer = 'block5_conv2'
```

### Cost Functions

**Style Cost** — measures difference in texture/style using gram matrices:

$$L_{style} = \sum_{l} w_l E_l$$

**Content Cost** — measures difference in content/structure:

$$L_{content} = \frac{1}{H_l W_l C_l} \sum_{i,j,k} (F_{ijk}^l - P_{ijk}^l)^2$$

**Total Cost:**

$$L_{total} = \alpha L_{content} + \beta L_{style}$$

**Variational Cost** (Task 10) — reduces noise and improves smoothness:

$$L_{total} = \alpha L_{content} + \beta L_{style} + var \cdot L_{var}$$

## Usage

```python
import matplotlib.image as mpimg
from 10-neural_style import NST

style_image = mpimg.imread("starry_night.jpg")
content_image = mpimg.imread("golden_gate.jpg")

nst = NST(style_image, content_image)
generated_image, cost = nst.generate_image(iterations=2000, step=100, lr=0.002)
```

## Example Output

| Content Image | Style Image | Generated Image |
|---|---|---|
| Golden Gate Bridge | Starry Night | Starry Gate |

## Author
Holberton School — Machine Learning Track
