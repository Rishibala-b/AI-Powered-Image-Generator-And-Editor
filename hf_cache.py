"""Import this BEFORE torch/diffusers so every entry point (streamlit, python -c, ...)
uses the same Hugging Face cache. Keeps models on D: if it exists (override with HF_HOME in .env)."""
import os

from dotenv import load_dotenv

load_dotenv()

if "HF_HOME" not in os.environ and os.name == "nt" and os.path.isdir("D:\\"):
    os.environ["HF_HOME"] = r"D:\huggingface_cache"
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Slow / unstable connection: wait longer before giving up on a download.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")

import time


def with_retries(fn, *args, tries: int = 5, **kwargs):
    """Run a Hugging Face download/load, retrying when the connection drops.
    Partial files are resumed, so each retry continues where the last one stopped."""
    for attempt in range(1, tries + 1):
        try:
            return fn(*args, **kwargs)
        except (OSError, ConnectionError) as e:
            if attempt == tries:
                raise
            print(f"Download interrupted ({type(e).__name__}); retry {attempt}/{tries - 1}...")
            time.sleep(3 * attempt)
