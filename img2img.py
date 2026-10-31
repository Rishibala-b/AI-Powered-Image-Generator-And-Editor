import hf_cache  # (must be first: sets the model cache location)
from hf_cache import with_retries
import io
import os

import torch

from gpu_utils import use_offload
from PIL import Image, ImageOps
from diffusers import (
    EulerAncestralDiscreteScheduler,
    StableDiffusionInstructPix2PixPipeline,
)

# https://huggingface.co/timbrooks/instruct-pix2pix


class Image2Image:
    """InstructPix2Pix: edits an image from a short instruction ("make it snowy")."""

    def __init__(self, model_id: str = "timbrooks/instruct-pix2pix", device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device.lower()
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.pipe = with_retries(
            StableDiffusionInstructPix2PixPipeline.from_pretrained,
            model_id,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.pipe.scheduler.config
        )

        if use_offload(self.device):
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe = self.pipe.to(self.device)

    @staticmethod
    def prepare_image(img, max_side: int = 512) -> Image.Image:
        """Accept a PIL image, bytes, or a Streamlit UploadedFile / file object."""
        if isinstance(img, Image.Image):
            pil = img
        else:
            try:
                if isinstance(img, (bytes, bytearray)):
                    data = bytes(img)
                elif hasattr(img, "getvalue"):          # Streamlit UploadedFile
                    data = img.getvalue()               # safe even if already read
                else:
                    img.seek(0)
                    data = img.read()
                pil = Image.open(io.BytesIO(data))
            except Exception as e:
                raise ValueError("Failed to open the uploaded image.") from e

        pil = ImageOps.exif_transpose(pil).convert("RGB")
        pil.thumbnail((max_side, max_side))             # keeps VRAM use sane
        w, h = (pil.width // 8) * 8, (pil.height // 8) * 8   # model needs multiples of 8
        return pil.resize((max(w, 8), max(h, 8)))

    def generate(self, image, prompt: str):
        pil_image = self.prepare_image(image)
        images = self.pipe(
            prompt,
            image=pil_image,
            num_inference_steps=10,
            image_guidance_scale=1,
        ).images
        return images[0]
