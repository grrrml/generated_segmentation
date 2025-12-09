"""
DINO v3 Embedding Utility

Extracts DINO v3 embeddings for image regions (object masks).
"""

from typing import Union
import torch
import numpy as np
from PIL import Image
from transformers import AutoModel, AutoImageProcessor
from PIL import Image as PILImage

class DINOv3Embedder:
    def __init__(self, model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m", device: str = "cuda"):
        self.device = device
        print(f"Loading {model_name}...")
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        print(f"DINO v3 loaded successfully on {device}")

    def get_embedding(self, image: Union[str, Image.Image], mask: np.ndarray) -> np.ndarray:
        """
        Extract DINO v3 embedding for the masked region of the image using Masked Global Average Pooling.
        Args:
            image: PIL Image or path
            mask: binary numpy array (H, W)
        Returns:
            embedding: numpy array (feature vector)
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        arr = np.array(image)
        # DINOv3 expects 224x224, resize image and mask
        image_resized = PILImage.fromarray(arr).resize((224, 224), PILImage.BICUBIC)
        mask_resized = PILImage.fromarray(mask.astype(np.uint8)*255).resize((224, 224), PILImage.NEAREST)
        mask_resized = np.array(mask_resized) > 127
        inputs = self.processor(images=image_resized, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            # DINOv3: CLS (1), REG (4), PATCH (196) tokens
            all_tokens = outputs.last_hidden_state.squeeze(0)  # (201, dim)
            patch_tokens = all_tokens[5:201]  # indices 5-200, total 196
        # Map mask to patch tokens
        # For 224x224 image, 196 patches (14x14)
        patch_size = 16
        h, w = mask_resized.shape
        mask_patches = mask_resized.reshape(h//patch_size, patch_size, w//patch_size, patch_size)
        mask_patches = mask_patches.transpose(0,2,1,3).reshape(h//patch_size, w//patch_size, patch_size*patch_size)
        mask_patch_map = mask_patches.mean(axis=2) > 0.5  # (14, 14)
        mask_patch_flat = mask_patch_map.flatten()  # (196,)
        # Select only patch tokens on the object
        selected_tokens = patch_tokens[mask_patch_flat]
        if selected_tokens.shape[0] == 0:
            # fallback: use mean of all patch tokens
            emb = patch_tokens.mean(dim=0).cpu().numpy()
        else:
            emb = selected_tokens.mean(dim=0).cpu().numpy()
        return emb
