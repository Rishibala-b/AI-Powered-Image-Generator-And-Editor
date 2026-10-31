# AI Image Generator and Editor

AI Image Generator and Editor is an AI-powered application that allows users to generate images from text descriptions and edit existing images using natural language instructions. It uses Google Gemini to enhance user prompts and locally running diffusion models to generate or modify images. The application provides a simple and interactive interface using Streamlit, with support for GPU memory optimization and faster image generation using LCM-LoRA.

# Tech Stack

## Frontend / User Interface

Streamlit

## Backend / Programming

Python

## AI & Machine Learning

Google Gemini API – Prompt Enhancement

Stable Diffusion (DreamShaper 7) – Text-to-Image Generation

LCM-LoRA – Faster Image Generation

InstructPix2Pix – Instruction-Based Image Editing

PyTorch – Deep Learning Computation

Hugging Face Diffusers – Model Loading and Inference

## Model Hosting & Storage

Hugging Face Model Hub

Local Model Cache

## Build / Tools

VS Code

Git & GitHub

CUDA (GPU Acceleration, when available)

# Core Features & Use Cases

## Text-to-Image Generation

Enter a text description to generate a new image.

Gemini enhances the user's input into a detailed image-generation prompt.

Uses Stable Diffusion with DreamShaper 7 to generate images.

LCM-LoRA enables image generation using fewer inference steps (4 steps in the configured pipeline).

## Image-to-Image Editing

Upload an existing image in PNG, JPG, or JPEG format.

Enter a natural language instruction describing the desired modification.

Gemini converts the input into a concise editing instruction.

InstructPix2Pix edits the uploaded image based on the instruction.

## Gemini Prompt Enhancement

Uses the Google Gemini API through the Google GenAI SDK.

Separate prompt templates for image generation and image editing.

Configurable Gemini model selection with fallback for model-not-found errors.

Handles API errors such as authentication failures, quota limits, and server errors.

Falls back to the original user prompt if Gemini is unavailable.

## Papercut Style (Optional)

Supports an optional Papercut LoRA adapter for compatible SDXL models.

Combines the LCM and Papercut adapters to influence the generated image style.

This option is available only when an SDXL model is configured.

## GPU Memory Optimization

Automatically detects GPU availability.

Supports FP16 precision on CUDA devices and FP32 on CPU.

Uses CPU offloading for GPUs with less than 6 GB VRAM by default.

Supports manual memory configuration through the `LOW_VRAM` environment variable.

Keeps only one heavy image-generation model loaded at a time to reduce memory usage.

## Image Output

Displays generated or edited images directly in Streamlit.

Shows the final prompt used for generation or editing.

Allows users to download the output image in PNG format.

# Repository Structure

```text
AI-Image-Generator/
│
├── app.py                  # Main Streamlit application
├── prompt_generator.py     # Gemini-based prompt enhancement
├── image_generator.py      # Text-to-image generation using Stable Diffusion
├── img2img.py              # Image editing using InstructPix2Pix
├── gpu_utils.py            # GPU memory and CPU offloading management
├── hf_cache.py             # Hugging Face cache configuration and retry logic
├── check_gemini.py         # Gemini API connection testing
│
├── requirements.txt        # Python dependencies
├── .env                    # API keys and model configuration (not committed)
├── .env.example            # Example environment configuration
├── .gitignore              # Files excluded from Git
└── README.md               # Project documentation
```

# Installation & Setup

## 1. Clone the Repository

```bash
git clone https://github.com/Rishibala-b/AI-Powered-Image-Generator-And-Editor.git
```

Navigate into the project directory:

```bash
cd AI-Powered-Image-Generator-And-Editor
```

## 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Configure Environment Variables

Create a `.env` file in the project root directory.

Add the following configuration:

```env
GEMINI_API_KEY=YOUR_GEMINI_API_KEY

# Optional Gemini model configuration
GEMINI_MODEL=gemini-flash-latest

# Optional image model configuration
IMAGE_MODEL=Lykon/dreamshaper-7

# Optional GPU memory configuration
LOW_VRAM=1

# Optional Hugging Face cache location
HF_HOME=D:\huggingface_cache
```

Replace `YOUR_GEMINI_API_KEY` with your API key from Google AI Studio.

The Hugging Face cache location can be customized to store downloaded models on a drive with sufficient storage space.

## 5. Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application opens in your browser.

# Model Configuration

## Text-to-Image Model

**DreamShaper 7**

Hugging Face: `Lykon/dreamshaper-7`

Uses a pretrained Stable Diffusion model with LCM-LoRA to generate images from enhanced text prompts.

Configured for 4 inference steps.

## Image Editing Model

**InstructPix2Pix**

Hugging Face: `timbrooks/instruct-pix2pix`

Uses an existing image and a text instruction to produce an edited image.

Configured for 10 inference steps.

## Acceleration Adapter

**LCM-LoRA**

Hugging Face: `latent-consistency/lcm-lora-sdv1-5`

Used with the default DreamShaper 7 configuration to support faster image generation.

## Optional Style Adapter

**Papercut SDXL LoRA**

Hugging Face: `TheLastBen/Papercut_SDXL`

Provides a layered paper-cut illustration style when a compatible SDXL model is configured.

# GPU & Memory Configuration

The application supports both CPU and GPU execution.

* CUDA is used when a compatible GPU is available.
* FP16 precision is used for GPU execution.
* FP32 precision is used for CPU execution.
* CPU offloading helps reduce GPU memory consumption.
* `LOW_VRAM=1` enables CPU offloading.
* `LOW_VRAM=0` disables automatic CPU offloading.

The first execution may take longer because the pretrained models need to be downloaded from Hugging Face.

# Important Notes

* A working Gemini API key is required for prompt enhancement, but image generation can fall back to the original prompt if Gemini fails.
* Pretrained image models are downloaded from Hugging Face and stored in the configured local cache.
* Model weights and cache files are not included in the GitHub repository.
* Sufficient RAM, disk space, and GPU memory (if using CUDA) are required to run the models.
* Image generation and editing performance depend on the hardware and selected model.

# License

This project is for educational and learning purposes.
