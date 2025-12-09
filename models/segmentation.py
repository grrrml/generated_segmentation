"""
SAM3 Segmentation Module

Uses SAM3 (Segment Anything 3) from Meta for text-prompted instance segmentation.
Returns masks and bounding boxes (XYXY format, pixel coordinates).
"""

from typing import List, Dict, Any, Optional, Union
from collections import defaultdict

import torch
import numpy as np
from PIL import Image
from transformers import Sam3Processor, Sam3Model


class SAM3Segmenter:
    """
    Segments objects in images using SAM3 with text prompts.
    
    Uses post_process_instance_segmentation() which returns both
    masks and tight bounding boxes derived from the masks.
    """
    
    def __init__(
        self,
        model_name: str = "facebook/sam3",
        dtype: str = "bfloat16",
        device: str = "cuda",
    ):
        """
        Initialize SAM3 segmenter.
        
        Args:
            model_name: HuggingFace model identifier for SAM3
            dtype: Model precision (bfloat16 recommended for A100)
            device: Device to run inference on
        """
        self.model_name = model_name
        self.device = device
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        
        print(f"Loading {model_name}...")
        self.processor = Sam3Processor.from_pretrained(model_name)
        self.model = Sam3Model.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
        ).to(device)
        self.model.eval()
        print(f"SAM3 loaded successfully on {device}")
    
    def segment(
        self,
        image: Union[str, Image.Image],
        labels: List[str],
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Segment objects in an image using text labels.
        
        Args:
            image: PIL Image or path to image file
            labels: List of entity labels to segment (e.g., ["dog", "red car"])
            threshold: Detection confidence threshold
            mask_threshold: Mask binarization threshold
            
        Returns:
            List of dicts containing:
                - object_id: Unique identifier (label_index)
                - label: Entity label
                - bbox: Bounding box [x1, y1, x2, y2] in XYXY pixel format
                - mask: Binary mask as numpy array
                - score: Confidence score
        """
        # Load image if path provided
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        
        image_width, image_height = image.size
        results = []
        label_counters = defaultdict(int)
        
        # Segment each label separately for better control
        for label in labels:
            label_results = self._segment_single_label(
                image=image,
                label=label,
                threshold=threshold,
                mask_threshold=mask_threshold,
                image_width=image_width,
                image_height=image_height,
            )
            
            # Assign object IDs based on label and detection order
            for obj in label_results:
                idx = label_counters[label]
                obj["object_id"] = f"{label}_{idx}"
                label_counters[label] += 1
                results.append(obj)
        
        return results
    
    def _segment_single_label(
        self,
        image: Image.Image,
        label: str,
        threshold: float,
        mask_threshold: float,
        image_width: int,
        image_height: int,
    ) -> List[Dict[str, Any]]:
        """
        Segment a single label in the image.
        
        Args:
            image: PIL Image
            label: Single entity label to segment
            threshold: Detection confidence threshold
            mask_threshold: Mask binarization threshold
            image_width: Image width in pixels
            image_height: Image height in pixels
            
        Returns:
            List of detected objects for this label
        """
        # Prepare inputs with text prompt
        inputs = self.processor(
            images=image,
            text=label,
            return_tensors="pt",
        ).to(self.device)
        
        # Convert pixel_values to model's dtype to avoid dtype mismatch
        if "pixel_values" in inputs and inputs["pixel_values"].dtype != self.dtype:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=self.dtype)
        
        # Store original sizes for post-processing
        original_sizes = [[image_height, image_width]]
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process to get instance segmentation results
        # This returns masks, boxes, and scores
        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=mask_threshold,
            target_sizes=original_sizes,
        )[0]  # Get first (and only) image results
        
        # Convert to our output format
        detected_objects = []
        
        masks = results.get("masks", [])
        boxes = results.get("boxes", [])
        scores = results.get("scores", [])
        
        for i, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
            # Convert tensors to numpy/list
            mask_np = mask.cpu().numpy() if torch.is_tensor(mask) else np.array(mask)
            box_list = box.cpu().tolist() if torch.is_tensor(box) else list(box)
            score_val = score.item() if torch.is_tensor(score) else float(score)
            
            # Ensure bbox is in XYXY integer format
            bbox = [int(coord) for coord in box_list[:4]]
            
            detected_objects.append({
                "label": label,
                "bbox": bbox,  # XYXY format, pixels
                "mask": mask_np.astype(bool),
                "score": score_val,
            })
        
        return detected_objects
    
    def segment_with_boxes(
        self,
        image: Union[str, Image.Image],
        boxes: List[List[int]],
        labels: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Segment objects using bounding box prompts instead of text.
        
        Useful when using Qwen3-VL grounding results as input.
        
        Args:
            image: PIL Image or path to image file
            boxes: List of bounding boxes [[x1, y1, x2, y2], ...] in pixel coords
            labels: Optional labels for each box
            
        Returns:
            List of segmentation results with masks and boxes
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        
        if labels is None:
            labels = [f"object_{i}" for i in range(len(boxes))]
        
        image_width, image_height = image.size
        results = []
        
        for i, (box, label) in enumerate(zip(boxes, labels)):
            # Prepare inputs with box prompt
            inputs = self.processor(
                images=image,
                input_boxes=[[[box]]],  # Nested list for batch processing
                return_tensors="pt",
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Get mask from outputs
            pred_masks = outputs.pred_masks
            if pred_masks is not None and len(pred_masks) > 0:
                mask = pred_masks[0, 0].cpu().numpy()
                mask = (mask > 0.5).astype(bool)
                
                # Resize mask to original image size if needed
                if mask.shape != (image_height, image_width):
                    from PIL import Image as PILImage
                    mask_img = PILImage.fromarray(mask.astype(np.uint8) * 255)
                    mask_img = mask_img.resize((image_width, image_height))
                    mask = np.array(mask_img) > 127
                
                results.append({
                    "object_id": f"{label}_{i}",
                    "label": label,
                    "bbox": box,
                    "mask": mask,
                    "score": 1.0,  # Box-prompted doesn't have confidence
                })
        
        return results
    
    def segment_batch(
        self,
        images: List[Union[str, Image.Image]],
        labels: List[str],
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
    ) -> List[List[Dict[str, Any]]]:
        """
        Segment multiple images with the same labels.
        
        Args:
            images: List of PIL Images or paths
            labels: List of entity labels to segment in all images
            threshold: Detection confidence threshold
            mask_threshold: Mask binarization threshold
            
        Returns:
            List of results for each image
        """
        all_results = []
        for image in images:
            results = self.segment(
                image=image,
                labels=labels,
                threshold=threshold,
                mask_threshold=mask_threshold,
            )
            all_results.append(results)
        return all_results
