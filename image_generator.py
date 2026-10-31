import hf_cache  # (must be first: sets the model cache location)
from hf_cache import with_retries
import os

import torch

from gpu_utils import use_offload
from diffusers import DiffusionPipeline, LCMScheduler

# Light default that fits a 4 GB GPU / 8 GB RAM: SD 1.5 family + LCM-LoRA (4 steps).
# For a big GPU you can switch to SDXL with IMAGE_MODEL=stabilityai/stable-diffusion-xl-base-1.0
DEFAULT_MODEL = "Lykon/dreamshaper-7"


class ImageGenerator:
    def __init__(self, model_name: str | None = None, device: str | None = None):
        model_name = model_name or os.getenv("IMAGE_MODEL") or DEFAULT_MODEL
        self.is_xl = "xl" in model_name.lower()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device.lower()

        use_fp16 = self.device == "cuda"
        load_options = {"torch_dtype": torch.float16 if use_fp16 else torch.float32}
        if use_fp16:
            load_options["variant"] = "fp16"

        self.pipe = with_retries(
            DiffusionPipeline.from_pretrained, model_name, **load_options
        )
        self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)

        lcm_id = (
            "latent-consistency/lcm-lora-sdxl"
            if self.is_xl
            else "latent-consistency/lcm-lora-sdv1-5"
        )
        try:  # LoRA adapters need the `peft` package
            with_retries(self.pipe.load_lora_weights, lcm_id, adapter_name="lcm")
            if self.is_xl:  # Papercut style exists only for SDXL
                with_retries(
                    self.pipe.load_lora_weights,
                    "TheLastBen/Papercut_SDXL",
                    weight_name="papercut.safetensors",
                    adapter_name="papercut",
                )
        except (ValueError, ImportError) as e:
            if "peft" in str(e).lower():
                raise RuntimeError(
                    "The `peft` package is required. Run: pip install peft==0.10.0"
                ) from e
            raise

        if use_offload(self.device):
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe = self.pipe.to(self.device)

    def generate(self, prompt: str, papercut: bool = True):
        if self.is_xl and papercut:
            self.pipe.set_adapters(["lcm", "papercut"], adapter_weights=[1.0, 0.8])
        else:
            self.pipe.set_adapters(["lcm"], adapter_weights=[1.0])
        return self.pipe(prompt, num_inference_steps=4, guidance_scale=1.0).images[0]
