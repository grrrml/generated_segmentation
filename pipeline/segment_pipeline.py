"""
Segmentation Pipeline Module

Orchestrates the complete segmentation workflow:
1. Parse text prompt to extract all entities using Qwen3-VL
2. Segment contextual images for matching entities
3. Segment generated image for ALL entities from prompt
4. Return structured output with object IDs for comparison
"""

from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import json
import numpy as np
from numpy.linalg import norm
from PIL import Image
from models.text_parser import TextParser
from models.segmentation import SAM3Segmenter
from models.dino_embedder import DINOv3Embedder
from utils.mask_utils import encode_rle, decode_rle, masks_iou, bbox_iou
from utils.visualization import draw_results, save_visualization


class SegmentationPipeline:
    """
    Complete segmentation pipeline using SAM3 and Qwen3-VL.
    
    Both models are loaded simultaneously (~18GB VRAM total on A100 40GB).
    """
    
    def __init__(
        self,
        text_parser_model: str = "Qwen/Qwen3-VL-8B-Instruct",
        segmentation_model: str = "facebook/sam3",
        dino_model: str = "facebook/dinov3-base-224",
        dtype: str = "bfloat16",
        device: str = "cuda",
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
    ):
        """
        Initialize the segmentation pipeline.
        
        Args:
            text_parser_model: HuggingFace model for text parsing
            segmentation_model: HuggingFace model for segmentation
            dtype: Model precision
            device: Device for inference
            threshold: Detection confidence threshold
            mask_threshold: Mask binarization threshold
        """
        self.threshold = threshold
        self.mask_threshold = mask_threshold
        
        # Load both models simultaneously
        print("Initializing Segmentation Pipeline...")
        print("=" * 50)
        
        self.text_parser = TextParser(
            model_name=text_parser_model,
            dtype=dtype,
            device_map="auto",
        )
        
        self.segmenter = SAM3Segmenter(
            model_name=segmentation_model,
            dtype=dtype,
            device=device,
        )
        
        self.dino_embedder = DINOv3Embedder(model_name=dino_model, device=device)
        print("=" * 50)
        print("Pipeline ready!")
    
    def run(
        self,
        text_prompt: str,
        contextual_image_paths: List[str],
        generated_image_path: str,
        save_visualizations: bool = False,
        visualization_dir: Optional[str] = None,
        pair_objects: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the complete segmentation pipeline.
        
        Args:
            text_prompt: Text prompt describing the desired image
            contextual_image_paths: List of paths to contextual images
            generated_image_path: Path to the generated image
            save_visualizations: Whether to save annotated images
            visualization_dir: Directory for saving visualizations
            
        Returns:
            Structured output dict with segmentation results
        """
        # Step 1: Extract all entities from the text prompt
        print(f"\n[1/3] Extracting entities from prompt...")
        entities = self.text_parser.extract_entities(text_prompt)
        print(f"      Found {len(entities)} entities: {entities}")
        
        # Step 2: Segment contextual images
        print(f"\n[2/3] Segmenting {len(contextual_image_paths)} contextual image(s)...")
        contextual_results = []
        
        for i, img_path in enumerate(contextual_image_paths):
            print(f"      Processing contextual image {i+1}/{len(contextual_image_paths)}...")
            result = self._segment_image(
                image_path=img_path,
                labels=entities,
                image_type="contextual",
            )
            contextual_results.append(result)
            
            if save_visualizations and visualization_dir:
                self._save_image_visualization(
                    result=result,
                    output_dir=visualization_dir,
                    prefix=f"contextual_{i}",
                )
        
        # Step 3: Segment generated image for ALL entities
        print(f"\n[3/3] Segmenting generated image for all entities...")
        generated_result = self._segment_image(
            image_path=generated_image_path,
            labels=entities,
            image_type="generated",
        )
        
        if save_visualizations and visualization_dir:
            self._save_image_visualization(
                result=generated_result,
                output_dir=visualization_dir,
                prefix="generated",
            )
        
        # Step 4: Compute DINO embeddings for all objects
        print("\n[4/4] Computing DINO v3 embeddings for all objects...")
        for result in contextual_results:
            image = Image.open(result["path"]).convert("RGB")
            for obj in result["objects"]:
                mask = self._decode_mask(obj["mask"])
                obj["dino_embedding"] = self.dino_embedder.get_embedding(image, mask).tolist()
        image = Image.open(generated_result["path"]).convert("RGB")
        for obj in generated_result["objects"]:
            mask = self._decode_mask(obj["mask"])
            obj["dino_embedding"] = self.dino_embedder.get_embedding(image, mask).tolist()

        # Step 5: Pair objects by cosine similarity
        pairings = None
        if pair_objects:
            pairings = self.pair_objects_by_embedding(contextual_results, generated_result)

        # Build output structure
        output = {
            "text_prompt": text_prompt,
            "detected_entities": entities,
            "contextual_images": contextual_results,
            "generated_image": generated_result,
            "box_format": "xyxy",
            "coordinate_unit": "pixels",
            "object_pairings": pairings,
        }

        print(f"\nPipeline complete!")
        print(f"  - Entities detected: {len(entities)}")
        print(f"  - Contextual images: {len(contextual_results)}")
        print(f"  - Total objects in generated image: {len(generated_result['objects'])}")

        return output

    def _decode_mask(self, mask_rle):
        from utils.mask_utils import decode_rle
        return decode_rle(mask_rle)

    def pair_objects_by_embedding(self, contextual_results, generated_result, threshold=0.8):
        import numpy as np
        from numpy.linalg import norm
        def cosine_similarity(a, b):
            a = np.array(a)
            b = np.array(b)
            return float(np.dot(a, b) / (norm(a) * norm(b) + 1e-8))

        pairs = []
        # Flatten contextual objects
        ctx_objs = []
        for ctx in contextual_results:
            for obj in ctx["objects"]:
                ctx_objs.append({**obj, "source_image": ctx["path"]})
        # Pair each generated object to the most similar contextual object
        for gen_obj in generated_result["objects"]:
            best_sim = -1
            best_ctx = None
            for ctx_obj in ctx_objs:
                sim = cosine_similarity(gen_obj["dino_embedding"], ctx_obj["dino_embedding"])
                if sim > best_sim:
                    best_sim = sim
                    best_ctx = ctx_obj
            pair_info = {
                "generated_object_id": gen_obj["object_id"],
                "generated_label": gen_obj["label"],
                "best_contextual_object_id": best_ctx["object_id"] if best_ctx else None,
                "best_contextual_label": best_ctx["label"] if best_ctx else None,
                "source_image": best_ctx["source_image"] if best_ctx else None,
                "cosine_similarity": best_sim,
                "threshold": threshold,
                "paired": best_sim >= threshold if best_ctx else False,
            }
            pairs.append(pair_info)
        return pairs
    
    def _segment_image(
        self,
        image_path: str,
        labels: List[str],
        image_type: str,
    ) -> Dict[str, Any]:
        """
        Segment a single image and format results.
        
        Args:
            image_path: Path to the image
            labels: Entity labels to segment
            image_type: Type of image (contextual/generated/background)
            
        Returns:
            Dict with image info and detected objects
        """
        # Load image
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        
        # Run segmentation
        raw_results = self.segmenter.segment(
            image=image,
            labels=labels,
            threshold=self.threshold,
            mask_threshold=self.mask_threshold,
        )
        
        # Convert to output format with RLE-encoded masks
        objects = []
        for obj in raw_results:
            obj_output = {
                "object_id": obj["object_id"],
                "label": obj["label"],
                "bbox": obj["bbox"],  # XYXY, pixels
                "mask": encode_rle(obj["mask"]),
                "score": obj["score"],
            }
            objects.append(obj_output)
        
        return {
            "path": str(Path(image_path).absolute()),
            "image_dimensions": {
                "width": width,
                "height": height,
            },
            "objects": objects,
        }
    
    def _save_image_visualization(
        self,
        result: Dict[str, Any],
        output_dir: str,
        prefix: str,
    ) -> str:
        """
        Save visualization for a segmented image.
        
        Args:
            result: Segmentation result dict
            output_dir: Directory to save visualization
            prefix: Filename prefix
            
        Returns:
            Path to saved visualization
        """
        from utils.mask_utils import decode_rle
        
        # Load original image
        image = Image.open(result["path"]).convert("RGB")
        
        # Decode masks for visualization
        objects_for_viz = []
        for obj in result["objects"]:
            obj_viz = obj.copy()
            obj_viz["mask"] = decode_rle(obj["mask"])
            objects_for_viz.append(obj_viz)
        
        # Draw results
        annotated = draw_results(image, objects_for_viz)
        
        # Save
        output_path = Path(output_dir) / f"{prefix}_annotated.jpg"
        return save_visualization(annotated, str(output_path))
    
    def segment_single_image(
        self,
        image_path: str,
        labels: List[str],
    ) -> Dict[str, Any]:
        """
        Segment a single image with given labels (utility method).
        
        Args:
            image_path: Path to image
            labels: Entity labels to segment
            
        Returns:
            Segmentation result dict
        """
        return self._segment_image(
            image_path=image_path,
            labels=labels,
            image_type="single",
        )
    
    def compare_objects(
        self,
        contextual_results: List[Dict[str, Any]],
        generated_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Compare objects between contextual and generated images.
        
        Matches objects by label and provides comparison metrics.
        Useful for downstream quality evaluation.
        
        Args:
            contextual_results: List of contextual image results
            generated_result: Generated image result
            
        Returns:
            Comparison dict with matched objects and metrics
        """
        from utils.mask_utils import decode_rle, masks_iou, bbox_iou
        
        comparisons = []
        
        # Group contextual objects by label
        contextual_by_label: Dict[str, List[Dict]] = {}
        for ctx_result in contextual_results:
            for obj in ctx_result["objects"]:
                label = obj["label"]
                if label not in contextual_by_label:
                    contextual_by_label[label] = []
                contextual_by_label[label].append({
                    **obj,
                    "source_image": ctx_result["path"],
                })
        
        # Match with generated objects
        for gen_obj in generated_result["objects"]:
            label = gen_obj["label"]
            
            comparison = {
                "label": label,
                "generated_object_id": gen_obj["object_id"],
                "generated_bbox": gen_obj["bbox"],
                "generated_score": gen_obj["score"],
                "matched_contextual_objects": [],
            }
            
            if label in contextual_by_label:
                gen_mask = decode_rle(gen_obj["mask"])
                
                for ctx_obj in contextual_by_label[label]:
                    ctx_mask = decode_rle(ctx_obj["mask"])
                    
                    # Note: IoU comparison is meaningful only if images are aligned
                    # For now, we just indicate the match exists
                    match_info = {
                        "contextual_object_id": ctx_obj["object_id"],
                        "source_image": ctx_obj["source_image"],
                        "contextual_bbox": ctx_obj["bbox"],
                        "contextual_score": ctx_obj["score"],
                    }
                    comparison["matched_contextual_objects"].append(match_info)
            
            comparisons.append(comparison)
        
        return {
            "comparisons": comparisons,
            "summary": {
                "total_generated_objects": len(generated_result["objects"]),
                "total_contextual_objects": sum(
                    len(r["objects"]) for r in contextual_results
                ),
                "matched_labels": len([
                    c for c in comparisons 
                    if c["matched_contextual_objects"]
                ]),
            },
        }
    
    def to_json(
        self,
        output: Dict[str, Any],
        output_path: Optional[str] = None,
        indent: int = 2,
    ) -> str:
        """
        Convert pipeline output to JSON.
        
        Args:
            output: Pipeline output dict
            output_path: Optional path to save JSON file
            indent: JSON indentation level
            
        Returns:
            JSON string
        """
        json_str = json.dumps(output, indent=indent, ensure_ascii=False)
        
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(json_str)
        
        return json_str
