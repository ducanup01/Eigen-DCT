import streamlit as st
from functions import *
import numpy as np

# img = np.array([
#     [[12, 200, 45], [255, 30, 90], [60, 180, 240], [10, 10, 10], [123, 222, 111]],
#     [[80, 90, 100], [200, 10, 50], [33, 144, 255], [90, 60, 30], [5, 250, 200]],
#     [[170, 80, 40], [20, 200, 220], [140, 140, 140], [255, 120, 0], [75, 25, 190]],
#     [[60, 10, 255], [180, 180, 20], [0, 0, 0], [100, 200, 50], [240, 240, 240]],
#     [[130, 60, 200], [45, 255, 120], [220, 30, 30], [80, 80, 255], [15, 160, 90]]
# ], dtype=np.uint8)

def singular_component(img, k):
    img = img.astype(float)

    channels = []

    for c in range(3):

        A = img[:, :, c]

        U, S, Vt = np.linalg.svd(A, full_matrices=False)

        component = S[k] * np.outer(U[:, k], Vt[k, :])

        channels.append(component)

    result = np.stack(channels, axis=2)

    result = np.clip(result, 0, 255)

    return result.astype(np.uint8)

img = np.array(Image.open("test_cat.png"), dtype=np.uint8)

uploaded_file = st.file_uploader("Upload image")
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")

    img = np.array(img, dtype=np.uint8)

# Use one channel just to define k range
R = img[:, :, 0]
k = st.slider("Number of singular values", 1, min(img.shape[0], img.shape[1]), 1)

# Compress entire image at once
compressed_img = compress_channel(img, k)

m, n = R.shape
num_values_per_channel = k * (m + n + 1)
total_values = 3 * num_values_per_channel

st.write(f"Size difference: {img.size - total_values:,} integers")

# Display images
# pil_img = Image.fromarray(img).resize((200, 200), Image.NEAREST)
# rgb_img = Image.fromarray(compressed_img).resize((200, 200), Image.NEAREST)
pil_img = Image.fromarray(img).resize((400, 400), Image.LANCZOS)
rgb_img = Image.fromarray(compressed_img).resize((400, 400), Image.LANCZOS)

component_img = singular_component(img, k - 1)

component_pil = Image.fromarray(component_img).resize(
    (400, 400),
    Image.LANCZOS
)

with st.expander("View Original and Compressed Images", expanded=True):

    col0, col1 = st.columns(2)

    with col0:
        st.write(f"Numbers to represent: {img.size:,} integers")
        st.image(pil_img, caption="Original image", use_container_width=True)

        st.image(component_pil, caption=f"Singular component k = {k}", use_container_width=True)


    with col1:
        st.write(f"Numbers to represent: {total_values:,} integers")
        st.image(rgb_img, caption="Compressed image", use_container_width=True)


# Compression Ratio
compression_ratio = img.size / total_values

# Mean Squared Error
mse = np.mean((img.astype(np.float64) - compressed_img.astype(np.float64)) ** 2)

# Peak Signal-to-Noise Ratio
if mse == 0:
    psnr = float("inf")
else:
    psnr = 20 * np.log10(255 / np.sqrt(mse))

st.subheader("Compression Metrics")

metric_col1, metric_col2, metric_col3 = st.columns(3)

with metric_col1:
    st.metric("Compression Ratio (CR)", f"{compression_ratio:.2f}:1")

with metric_col2:
    st.metric("Mean Squared Error (MSE)", f"{mse:.2f}")

with metric_col3:
    st.metric("PSNR", f"{psnr:.2f} dB")