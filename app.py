import streamlit as st
import numpy as np
from PIL import Image
import io
import base64
from model import generate
import time
import torch
import gc

# --- Website Setup --- #
st.set_page_config(page_title="Anime Face Generator", layout="wide")

st.markdown("<h1 style='text-align: center;'>Anime Face Generator</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center'>Click the button to generate random anime faces!</p>", unsafe_allow_html=True)

# use monotonic clock for reliable elapsed-time checks
if "last_click" not in st.session_state:
    st.session_state.last_click = 0.0

cooldown = 2.0  # seconds
arr = None

# To display the initial image
if "img_b64" not in st.session_state:
    arr = generate()
    # Generate image
    img = Image.fromarray(arr)
    del arr
    torch.cuda.empty_cache()
    gc.collect()
    # Turn image into proper encoding
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    st.session_state.img_b64 = base64.b64encode(buf.getvalue()).decode()
    st.session_state.img_bytes = buf.getvalue()
    st.session_state.last_click = time.monotonic()

# --- UI Setup --- #
col1, col2, col3 = st.columns([5, 2, 5])
with col2:
    if st.button("Generate Image", use_container_width=True):
        now = time.monotonic()
        # To check the cooldown
        if now - st.session_state.last_click < cooldown:
            remaining = round(cooldown - (now - st.session_state.last_click), 1)
            st.warning(f"Please wait {remaining} seconds before generating again!")

        else:
            # run generation first, update last_click AFTER generation finishes
            with st.spinner("Generating..."):
                arr = generate()
                # update timestamp after generation completes (measures cooldown from end)
                st.session_state.last_click = time.monotonic()
                print("Clicked")

            img = Image.fromarray(arr)
            del arr
            torch.cuda.empty_cache()
            gc.collect()

            # Save image to bytes
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            st.session_state.img_b64 = base64.b64encode(buf.getvalue()).decode()
            st.session_state.img_bytes = buf.getvalue()


# --- Display image if available --- #
if "img_b64" in st.session_state:
    st.markdown(
        f"""
        <div style="text-align: center;">
            <img src="data:image/png;base64,{st.session_state.img_b64}"
                 style="border-radius: 5%; border: 4px solid #888;" width="256">
        </div>
        """,
        unsafe_allow_html=True
    )
    # Extra spacing
    st.write("\n\n\n")
    # Center the download button
    col1, col2, col3 = st.columns([5, 2, 5])
    with col2:
        st.download_button(
            label="Download Image",
            data=st.session_state.img_bytes,
            file_name="generated.png",
            mime="image/png",
            use_container_width=True
        )