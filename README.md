# FGSC: Fine-Grained Style Control in Text-to-Image Generation

Enhanced CSGO with semantic-aware style adaptation for fine-grained control over style attributes.

## Features

🎨 **Multi-Attribute Style Control**: Separate control over color, texture, brushstrokes, composition, and lighting

🧠 **Content-Aware Adaptation**: Intelligent style application based on semantic understanding

⚡ **Memory Efficient**: Optimized for RTX 4090 (24GB) with ~155M trainable parameters

🔧 **Easy Integration**: Extends existing CSGO with minimal code changes

## Quick Start

```bash
# Setup the project
cd ~/FGSC
chmod +x setup_fgsc.sh
./setup_fgsc.sh

# Install the package
pip install -e .

# Run tests
python tests/test_style_disentangler.py

# Try example usage
python examples/basic_usage.py
```

## Usage

```python
from fgsc import create_enhanced_csgo
from PIL import Image

# Create Enhanced CSGO
enhanced_csgo = create_enhanced_csgo(
    pipe=pipe,
    image_encoder_path="./base_models/IP-Adapter/models/image_encoder",
    csgo_ckpt="./CSGO/csgo.bin",
    device=device
)

# Generate with fine-grained control
images, analysis = enhanced_csgo.generate_with_attribute_control(
    pil_content_image=content_image,
    pil_style_image=style_image,
    prompt="A beautiful landscape painting",
    attribute_weights={
        'color': 1.5,      # Emphasize colors
        'texture': 0.5,    # Reduce texture
        'brush': 1.0,      # Normal brushstrokes
        'composition': 1.2, # Emphasize composition
        'lighting': 0.8    # Subtle lighting
    }
)
```

## Architecture

- **Block 1**: Text & Style Encoders (CLIP-based)
- **Block 2**: Style Disentangler (5 attribute branches)
- **Block 3**: Content Analyzer (semantic understanding)
- **Block 4**: Multi-Attribute Controller (adaptive fusion)
- **Block 5**: Spatial Adapter (region-aware styling)
- **Block 6**: CSGO Generation (SDXL backbone)

## Memory Requirements

- Base CSGO: ~12-15GB
- FGSC Components: ~3-5GB  
- Total: ~18-20GB (fits RTX 4090)

## Project Structure

```
FGSC/
├── fgsc/                    # Main package
│   ├── models/             # Model implementations
│   ├── utils/              # Utility functions
│   └── config/             # Configuration files
├── examples/               # Usage examples
├── tests/                  # Unit tests
├── scripts/                # Training scripts
└── outputs/                # Generated results
```
