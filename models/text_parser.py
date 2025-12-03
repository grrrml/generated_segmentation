"""
Text Parser Module

Uses Qwen3-VL-8B-Instruct to extract all objects/entities mentioned in a text prompt.
These entities are then used to guide SAM3 segmentation.
"""

import json
import re
from typing import List, Optional

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


class TextParser:
    """
    Extracts objects/entities from text prompts using Qwen3-VL-8B-Instruct.
    
    The model identifies all segmentable objects/entities mentioned in the prompt,
    including people, animals, objects, furniture, vehicles, etc.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        dtype: str = "bfloat16",
        device_map: str = "auto",
    ):
        """
        Initialize the TextParser with Qwen3-VL model.
        
        Args:
            model_name: HuggingFace model identifier
            dtype: Model precision (bfloat16 recommended for A100)
            device_map: Device placement strategy
        """
        self.model_name = model_name
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        
        print(f"Loading {model_name}...")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            device_map=device_map,
        )
        self.model.eval()
        print(f"Model loaded successfully on {self.model.device}")
    
    def extract_entities(self, prompt: str) -> List[str]:
        """
        Extract all objects/entities mentioned in the text prompt.
        
        Args:
            prompt: Text prompt describing the desired image
            
        Returns:
            List of entity names (e.g., ["dog", "woman", "red sofa", "garden"])
        """
        # System prompt for entity extraction
        extraction_prompt = f"""Analyze the following image generation prompt and extract ALL objects, entities, people, animals, or things that could be visually segmented in the resulting image.

Return ONLY a JSON array of strings, where each string is a segmentable entity.
Include:
- People (e.g., "woman", "man", "child")
- Animals (e.g., "dog", "cat", "bird")
- Objects (e.g., "chair", "table", "car")
- Nature elements (e.g., "tree", "flower", "mountain")
- Buildings/structures (e.g., "house", "bridge")
- Any other visually distinct entities

Be specific with attributes when mentioned (e.g., "red car" not just "car").

Prompt: "{prompt}"

Return only the JSON array, no explanation:"""

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": extraction_prompt}
                ]
            }
        ]
        
        # Process input
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[text],
            return_tensors="pt",
        ).to(self.model.device)
        
        # Generate response
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
            )
        
        # Decode output
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
        )[0]
        
        # Parse JSON response
        entities = self._parse_entities(output_text)
        return entities
    
    def _parse_entities(self, response: str) -> List[str]:
        """
        Parse the model's response to extract entity list.
        
        Args:
            response: Raw model output text
            
        Returns:
            List of entity strings
        """
        # Try to find JSON array in response
        response = response.strip()
        
        # Remove any markdown code block markers
        response = re.sub(r'^```json\s*', '', response)
        response = re.sub(r'^```\s*', '', response)
        response = re.sub(r'\s*```$', '', response)
        
        # Try direct JSON parsing
        try:
            entities = json.loads(response)
            if isinstance(entities, list):
                return [str(e).strip() for e in entities if e]
        except json.JSONDecodeError:
            pass
        
        # Fallback: try to extract array from response
        match = re.search(r'\[([^\]]+)\]', response)
        if match:
            try:
                entities = json.loads(f"[{match.group(1)}]")
                if isinstance(entities, list):
                    return [str(e).strip() for e in entities if e]
            except json.JSONDecodeError:
                pass
        
        # Last resort: split by common delimiters
        entities = re.split(r'[,\n]', response)
        entities = [
            e.strip().strip('"\'[]') 
            for e in entities 
            if e.strip() and not e.strip().startswith('{')
        ]
        
        return entities if entities else []
    
    def extract_entities_with_grounding(
        self,
        prompt: str,
        image_path: Optional[str] = None,
    ) -> List[dict]:
        """
        Extract entities with optional grounding (bounding boxes) if an image is provided.
        
        Args:
            prompt: Text prompt describing the desired image
            image_path: Optional path to image for visual grounding
            
        Returns:
            List of dicts with 'label' and optionally 'bbox' keys
        """
        if image_path is None:
            # Text-only extraction
            entities = self.extract_entities(prompt)
            return [{"label": e} for e in entities]
        
        # Visual grounding mode
        from PIL import Image
        image = Image.open(image_path).convert("RGB")
        
        grounding_prompt = f"""Given this image and the prompt: "{prompt}"

Locate and identify all objects/entities from the prompt that are visible in the image.
Return a JSON array where each element has:
- "label": the entity name
- "bbox_2d": bounding box as [x1, y1, x2, y2] in coordinates from 0-1000

Return only the JSON array:"""

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": grounding_prompt}
                ]
            }
        ]
        
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
        ).to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
        )[0]
        
        return self._parse_grounding_response(output_text)
    
    def _parse_grounding_response(self, response: str) -> List[dict]:
        """
        Parse grounding response with bounding boxes.
        
        Args:
            response: Raw model output with JSON grounding data
            
        Returns:
            List of dicts with 'label' and 'bbox' keys
        """
        response = response.strip()
        response = re.sub(r'^```json\s*', '', response)
        response = re.sub(r'^```\s*', '', response)
        response = re.sub(r'\s*```$', '', response)
        
        try:
            data = json.loads(response)
            if isinstance(data, list):
                results = []
                for item in data:
                    if isinstance(item, dict) and "label" in item:
                        result = {"label": item["label"]}
                        if "bbox_2d" in item:
                            result["bbox"] = item["bbox_2d"]
                        results.append(result)
                return results
        except json.JSONDecodeError:
            pass
        
        return []
    
    def convert_qwen_bbox_to_pixels(
        self,
        bbox: List[int],
        image_width: int,
        image_height: int,
    ) -> List[int]:
        """
        Convert Qwen3-VL bounding box (0-1000 scale) to pixel coordinates.
        
        Args:
            bbox: Bounding box in Qwen format [x1, y1, x2, y2] (0-1000)
            image_width: Image width in pixels
            image_height: Image height in pixels
            
        Returns:
            Bounding box in pixel coordinates [x1, y1, x2, y2]
        """
        x1, y1, x2, y2 = bbox
        return [
            int(x1 / 1000 * image_width),
            int(y1 / 1000 * image_height),
            int(x2 / 1000 * image_width),
            int(y2 / 1000 * image_height),
        ]
