import os

import hf_cache  # noqa: F401  (sets HF cache + loads .env; must come first)

import gc
import traceback
from io import BytesIO

import streamlit as st
import torch

from image_generator import ImageGenerator
from img2img import Image2Image
from prompt_generator import PromptGenerator

st.set_page_config(page_title="AI powered Image generator and Editor", layout="wide")
st.title("AI powered Image generator and Editor")


# Only ONE heavy model is kept in memory at a time. On a 4 GB GPU, holding the
# text-to-image and image-editing models together causes "CUDA out of memory".
@st.cache_resource
def _model_slot() -> dict:
    return {}


def free_models():
    _model_slot().clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_model(kind: str):
    slot = _model_slot()
    if slot.get("kind") != kind:
        free_models()                      # drop the previous model first
        with st.spinner("Loading the model (first run downloads it, ~2.5-4 GB)..."):
            slot["model"] = ImageGenerator() if kind == "text2img" else Image2Image()
        slot["kind"] = kind
    return slot["model"]


# Session state
st.session_state.setdefault("llm_prompt", None)
st.session_state.setdefault("generated_image", None)

# Sidebar
with st.sidebar:
    uploaded_image = st.file_uploader(
        label="Upload an image",
        label_visibility="collapsed",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=False,
        help="Upload an image to use the image-to-image model.",
    )
    papercut = True
    if "xl" in os.getenv("IMAGE_MODEL", "").lower():   # style only exists for SDXL
        papercut = st.checkbox(
            "Papercut style", value=True, disabled=uploaded_image is not None
        )
    st.divider()
    if st.button("Test Gemini connection"):
        tester = PromptGenerator()
        with st.spinner("Calling Gemini..."):
            ok = tester.get_response("a red apple")
        if ok:
            st.success(f"Gemini works (model: {tester.model_used})")
        else:
            st.error(tester.last_error)

col1, col2 = st.columns(2, gap="large")

# Left column
with col1:
    if uploaded_image:
        st.write("Image uploaded")
        # No width argument on purpose: works on every Streamlit version.
        st.image(uploaded_image, caption=uploaded_image.name)

    user_prompt = st.text_input(
        label="Your prompt idea here",
        placeholder="Example: A cat sitting on the moon"
        if not uploaded_image else "Example: make it look like winter",
    )

    if st.button("Generate", type="primary"):
        if not user_prompt.strip():
            st.warning("Please enter your image idea.")
        else:
            st.session_state["llm_prompt"] = None
            st.session_state["generated_image"] = None

            # Step 1: let Gemini improve the prompt (an edit instruction for img2img)
            prompt_generator = PromptGenerator()
            with st.spinner("Preparing your prompt..."):
                llm_prompt = prompt_generator.get_response(
                    user_prompt.strip(),
                    mode="edit" if uploaded_image else "generate",
                )

            # Step 2: fall back to the user's own text, and say WHY
            if llm_prompt:
                final_prompt = llm_prompt
                st.success(f"Enhanced prompt generated ({prompt_generator.model_used}).")
            else:
                final_prompt = user_prompt.strip()
                st.warning(
                    f"Gemini failed, so your original prompt is used.\n\n"
                    f"**Reason:** {prompt_generator.last_error}"
                )
            st.session_state["llm_prompt"] = final_prompt

            # Step 3: generate the image
            try:
                if uploaded_image:
                    model = get_model("img2img")
                    with st.spinner("Editing your image..."):
                        image = model.generate(image=uploaded_image, prompt=final_prompt)
                else:
                    model = get_model("text2img")
                    with st.spinner("Generating your image..."):
                        image = model.generate(prompt=final_prompt, papercut=papercut)

                buffer = BytesIO()
                image.save(buffer, format="PNG")
                st.session_state["generated_image"] = buffer.getvalue()

            except Exception as e:
                msg = str(e)
                if "out of memory" in msg.lower():
                    free_models()          # release whatever was half-loaded
                    st.error(
                        "GPU ran out of memory. Add LOW_VRAM=1 to your .env, "
                        "close other GPU programs (games, browsers with "
                        "hardware acceleration), restart the app and try again."
                    )
                else:
                    st.error(f"Image generation failed: {e}")
                with st.expander("Technical details"):
                    st.code(traceback.format_exc())

# Right column
with col2:
    if st.session_state["llm_prompt"]:
        st.subheader("Final Image Prompt")
        st.write(st.session_state["llm_prompt"])

    if st.session_state["generated_image"]:
        st.subheader("Generated Image")
        st.image(st.session_state["generated_image"], width=500)
        st.download_button(
            label="Download Image",
            data=st.session_state["generated_image"],
            file_name="generated_image.png",
            mime="image/png",
        )
