"""
DINO v3 Embedding Utility

Extracts DINO v3 embeddings for image regions (object masks).
"""

from typing import Union
import torch
import numpy as np
from PIL import Image
from transformers import AutoModel, AutoImageProcessor

class DINOv3Embedder:
    def __init__(self, model_name: str = "facebook/dinov3-base-224", device: str = "cuda"):
        self.device = device
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self.processor = AutoImageProcessor.from_pretrained(model_name)

    def get_embedding(self, image: Union[str, Image.Image], mask: np.ndarray) -> np.ndarray:
        """
        Extract DINO v3 embedding for the masked region of the image.
        Args:
            image: PIL Image or path
            mask: binary numpy array (H, W)
        Returns:
            embedding: numpy array (feature vector)
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        # Apply mask: set background to black
        arr = np.array(image)
        arr[mask == 0] = 0
        masked_img = Image.fromarray(arr)
        inputs = self.processor(images=masked_img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use CLS token or mean pooling
            if hasattr(outputs, "last_hidden_state"):
                emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy().squeeze()
            else:
                emb = outputs.pooler_output.cpu().numpy().squeeze()
        return emb
