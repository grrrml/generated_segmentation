# Image Segmentation Pipeline

Segment objects in contextual and generated images using text prompts. Uses SAM3 for segmentation, Qwen3-VL-8B for entity extraction, and DINOv3 for object embeddings and pairing.

## Quick usage

```bash
python main.py --prompt "A dog on a sofa" --contextual ctx1.jpg ctx2.jpg --generated gen.jpg --output results/output.json
```

## Configuration

Edit `config.yaml` to set model names, thresholds, and device options.

## Output

- `output.json`: segmentation results (prompt, entities, objects, masks, bboxes, scores, pairings)
- `output_embeddings.json`: DINOv3 embeddings for all objects

## Pairing Logic

Objects in generated and contextual images are paired by cosine similarity of DINOv3 embeddings, but only if their labels match.

## Models

- `Qwen3-VL-8B-Instruct` (entity extraction)
- `facebook/sam3` (segmentation)
- `facebook/dinov3-vitb16-pretrain-lvd1689m` (embeddings).
