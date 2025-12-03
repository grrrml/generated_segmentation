# Image Segmentation Pipeline

A Python pipeline for segmenting objects in contextual and generated images based on text prompts. Uses **SAM3** (Segment Anything 3) for segmentation and **Qwen3-VL-8B** for entity extraction from text.

## Features

- **Text-guided segmentation**: Extract entities from natural language prompts and segment them in images
- **Multi-image support**: Process multiple contextual images alongside a generated image
- **Automatic entity extraction**: Uses Qwen3-VL-8B to identify all segmentable objects from text prompts
- **Instance segmentation**: Returns masks, bounding boxes (XYXY format), and confidence scores
- **Visualization support**: Optional annotated image output with segmentation overlays
- **Configurable**: YAML-based configuration for model settings and thresholds

## Requirements

- Python 3.8+
- CUDA-capable GPU with ~18GB VRAM (recommended: A100 40GB)
- PyTorch 2.0+

## Installation

1. Clone the repository:
```bash
git clone https://github.com/grrrml/generated_segmentation.git
cd generated_segmentation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python main.py --prompt "A dog sitting on a red sofa in a garden" \
               --contextual images/context1.jpg images/context2.jpg \
               --generated images/generated.jpg \
               --output results/output.json
```

### With Configuration File

```bash
python main.py --config config/config.yaml \
               --prompt "A cat and dog on a sofa" \
               --contextual ctx1.jpg ctx2.jpg \
               --generated output.jpg
```

### With Visualization

```bash
python main.py --prompt "A red car in the city" \
               --contextual car.jpg city.jpg \
               --generated result.jpg \
               --output results/segmentation.json \
               --save-viz --viz-dir results/visualizations
```

## Command Line Arguments

| Argument | Short | Required | Description |
|----------|-------|----------|-------------|
| `--prompt` | `-p` | Yes | Text prompt describing the desired image |
| `--contextual` | `-c` | Yes | Path(s) to contextual image(s) |
| `--generated` | `-g` | Yes | Path to the generated image |
| `--output` | `-o` | No | Path to save JSON output (default: `segmentation_output.json`) |
| `--config` | | No | Path to YAML config file |
| `--text-model` | | No | HuggingFace model for text parsing (default: `Qwen/Qwen3-VL-8B-Instruct`) |
| `--save-viz` | | No | Save visualization images |
| `--viz-dir` | | No | Directory for saving visualizations |

## Project Structure

```
generated_segmentation/
├── main.py                    # Main entry point
├── requirements.txt           # Python dependencies
├── config/
│   └── config.yaml           # Pipeline configuration
├── models/
│   ├── __init__.py
│   ├── segmentation.py       # SAM3 segmentation module
│   └── text_parser.py        # Qwen3-VL text parsing module
├── pipeline/
│   ├── __init__.py
│   └── segment_pipeline.py   # Main segmentation pipeline
└── utils/
    ├── __init__.py
    ├── mask_utils.py         # Mask encoding utilities
    └── visualization.py      # Visualization utilities
```

## Configuration

The `config/config.yaml` file allows customization of:

- **Model settings**: Model names, data types, and device placement
- **Segmentation thresholds**: Detection and mask confidence thresholds
- **Output settings**: Bounding box format, coordinate units, visualization options
- **Device settings**: CUDA device selection and Flash Attention

Example configuration:
```yaml
models:
  text_parser:
    model_name: "Qwen/Qwen3-VL-8B-Instruct"
    dtype: "bfloat16"
    device_map: "auto"
  segmentation:
    model_name: "facebook/sam3"
    dtype: "bfloat16"
    device: "cuda"

segmentation:
  threshold: 0.5
  mask_threshold: 0.5
```

## Output Format

The pipeline outputs a JSON file containing segmentation results:

```json
{
  "prompt": "A dog sitting on a red sofa",
  "entities": ["dog", "red sofa"],
  "contextual_images": [
    {
      "path": "context.jpg",
      "objects": [
        {
          "object_id": "dog_0",
          "label": "dog",
          "bbox": [100, 150, 300, 400],
          "score": 0.95,
          "mask_rle": "..."
        }
      ]
    }
  ],
  "generated_image": {
    "path": "generated.jpg",
    "objects": [...]
  }
}
```

## Models Used

- **[Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)**: Vision-language model for extracting entities from text prompts
- **[SAM3](https://huggingface.co/facebook/sam3)**: Meta's Segment Anything 3 model for instance segmentation
