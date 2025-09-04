import streamlit as st
import numpy as np
from PIL import Image
import io
import base64
from model import generate
import time

if "last_click" not in st.session_state:
    st.session_state.last_click = 0

cooldown = 1  # cooldown in second

# --- Website Setup --- #
st.set_page_config(page_title="Anime Face Generator", layout="wide")

st.markdown("<h1 style='text-align: center;'>Anime Face Generator</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center'>Click the button to generate random anime faces!</p>", unsafe_allow_html=True)
# --- UI Setup --- #
col1, col2, col3 = st.columns([5, 2, 5])
with col2:
    if st.button("Generate Image", use_container_width=True):
        now = time.time()
        # Show cooldown timer
        if now - st.session_state.last_click < cooldown:
            remaining = round(cooldown - (now - st.session_state.last_click), 1)
            st.warning(f"Please wait {remaining} seconds before generating again!")
        else:
            st.session_state.last_click = now  # update timer
            # Spinner
            with st.spinner("Generating..."):
                arr = generate()
            img = Image.fromarray(arr)

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
                 style="border-radius: 5%; border: 4px solid #888;" width="200">
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