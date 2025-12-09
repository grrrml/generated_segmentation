#!/usr/bin/env python3
"""
Image Segmentation Pipeline - Main Entry Point

Segments objects in contextual and generated images based on text prompts.
Uses SAM3 for segmentation and Qwen3-VL-8B for entity extraction.

Usage:
    python main.py --prompt "A dog sitting on a red sofa in a garden" \
                   --contextual images/context1.jpg images/context2.jpg \
                   --generated images/generated.jpg \
                   --output results/output.json

    python main.py --config config/config.yaml \
                   --prompt "..." \
                   --contextual ... \
                   --generated ...
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import yaml


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Segment objects in images based on text prompts using SAM3 and Qwen3-VL.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python main.py --prompt "A cat and dog on a sofa" \\
                 --contextual ctx1.jpg ctx2.jpg \\
                 --generated output.jpg

  # With visualization and custom output
  python main.py --prompt "A red car in the city" \\
                 --contextual car.jpg city.jpg \\
                 --generated result.jpg \\
                 --output results/segmentation.json \\
                 --save-viz --viz-dir results/visualizations

  # With config file
  python main.py --config config/config.yaml \\
                 --prompt "..." --contextual ... --generated ...
        """,
    )
    
    # Required arguments
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        required=True,
        help="Text prompt describing the desired image",
    )
    parser.add_argument(
        "--contextual", "-c",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to contextual image(s) containing objects to transfer",
    )
    parser.add_argument(
        "--generated", "-g",
        type=str,
        required=True,
        help="Path to the generated image",
    )
    
    # Optional arguments
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="segmentation_output.json",
        help="Path to save JSON output (default: segmentation_output.json)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file",
    )
    
    # Model options
    parser.add_argument(
        "--text-model",
        type=str,
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="HuggingFace model for text parsing (default: Qwen/Qwen3-VL-8B-Instruct)",
    )
    parser.add_argument(
        "--seg-model",
        type=str,
        default="facebook/sam3",
        help="HuggingFace model for segmentation (default: facebook/sam3)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Model precision (default: bfloat16)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference (default: cuda)",
    )
    
    # Threshold options
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Detection confidence threshold (default: from config or 0.5)",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=None,
        help="Mask binarization threshold (default: from config or 0.5)",
    )
    
    # Visualization options
    parser.add_argument(
        "--save-viz",
        action="store_true",
        help="Save annotated visualization images",
    )
    parser.add_argument(
        "--viz-dir",
        type=str,
        default="outputs/visualizations",
        help="Directory for saving visualizations (default: outputs/visualizations)",
    )
    
    # Utility options
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Include object comparison in output",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output",
    )
    
    return parser.parse_args()


def validate_paths(args) -> bool:
    """Validate that all input paths exist."""
    errors = []
    
    for path in args.contextual:
        if not Path(path).exists():
            errors.append(f"Contextual image not found: {path}")
    
    if not Path(args.generated).exists():
        errors.append(f"Generated image not found: {args.generated}")
    
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return False
    
    return True


def main():
    """Main entry point."""
    args = parse_args()
    
    # Load config if provided
    config = {}
    if args.config and Path(args.config).exists():
        config = load_config(args.config)
        if not args.quiet:
            print(f"Loaded config from: {args.config}")
    
    # Override with command line args
    text_model = args.text_model or config.get("models", {}).get("text_parser", {}).get("model_name", "Qwen/Qwen3-VL-8B-Instruct")
    seg_model = args.seg_model or config.get("models", {}).get("segmentation", {}).get("model_name", "facebook/sam3")
    dtype = args.dtype or config.get("models", {}).get("text_parser", {}).get("dtype", "bfloat16")
    device = args.device or config.get("models", {}).get("segmentation", {}).get("device", "cuda")
    threshold = args.threshold if args.threshold is not None else config.get("segmentation", {}).get("threshold", 0.5)
    mask_threshold = args.mask_threshold if args.mask_threshold is not None else config.get("segmentation", {}).get("mask_threshold", 0.5)
    pairing_threshold = config.get("pairing", {}).get("threshold", 0.8)
    
    
    # Validate input paths
    if not validate_paths(args):
        sys.exit(1)
    
    # Import pipeline (delayed to avoid loading models before validation)
    from pipeline.segment_pipeline import SegmentationPipeline
    
    # Initialize pipeline
    if not args.quiet:
        print("\n" + "=" * 60)
        print("Image Segmentation Pipeline")
        print("=" * 60)
        print(f"Text Model: {text_model}")
        print(f"Segmentation Model: {seg_model}")
        print(f"Device: {device} | Dtype: {dtype}")
        print(f"Thresholds: detection={threshold}, mask={mask_threshold}")
        print("=" * 60)
    
    dino_model = config.get("models", {}).get("dino", {}).get("model_name", "facebook/dinov3-vitb16-pretrain-lvd1689m")
    pipeline = SegmentationPipeline(
        text_parser_model=text_model,
        segmentation_model=seg_model,
        dino_model=dino_model,
        dtype=dtype,
        device=device,
        threshold=threshold,
        mask_threshold=mask_threshold,
    )
    
    # Run segmentation
    if not args.quiet:
        print(f"\nPrompt: {args.prompt}")
        print(f"Contextual images: {len(args.contextual)}")
        print(f"Generated image: {args.generated}")
    
    output = pipeline.run(
        text_prompt=args.prompt,
        contextual_image_paths=args.contextual,
        generated_image_path=args.generated,
        save_visualizations=args.save_viz,
        visualization_dir=args.viz_dir if args.save_viz else None,
        pair_objects=True,
        pairing_threshold=pairing_threshold,
    )
    
    # Add comparison if requested
    if args.compare:
        comparison = pipeline.compare_objects(
            contextual_results=output["contextual_images"],
            generated_result=output["generated_image"],
        )
        output["comparison"] = comparison
    
    # Save JSON output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    json_output = pipeline.to_json(output, str(output_path))
    
    if not args.quiet:
        print(f"\n" + "=" * 60)
        print("Results Summary")
        print("=" * 60)
        print(f"Entities detected: {len(output['detected_entities'])}")
        print(f"  -> {output['detected_entities']}")
        
        for i, ctx in enumerate(output["contextual_images"]):
            print(f"Contextual image {i+1}: {len(ctx['objects'])} objects")
        
        print(f"Generated image: {len(output['generated_image']['objects'])} objects")
        print(f"\nOutput saved to: {output_path.absolute()}")
        
        if args.save_viz:
            print(f"Visualizations saved to: {Path(args.viz_dir).absolute()}")
    
    # Print JSON to stdout if quiet mode
    if args.quiet:
        print(json_output)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
