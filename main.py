import streamlit as st
from functions import *
import numpy as np

img = np.array([
    [[12, 200, 45], [255, 30, 90], [60, 180, 240], [10, 10, 10], [123, 222, 111]],
    [[80, 90, 100], [200, 10, 50], [33, 144, 255], [90, 60, 30], [5, 250, 200]],
    [[170, 80, 40], [20, 200, 220], [140, 140, 140], [255, 120, 0], [75, 25, 190]],
    [[60, 10, 255], [180, 180, 20], [0, 0, 0], [100, 200, 50], [240, 240, 240]],
    [[130, 60, 200], [45, 255, 120], [220, 30, 30], [80, 80, 255], [15, 160, 90]]
], dtype=np.uint8)

uploaded_file = st.file_uploader("Upload image")
if uploaded_file is not None:
    img = Image.open(uploaded_file)

    img = np.array(img, dtype=np.uint8)

R = img[:, :, 0]
G = img[:, :, 1]
B = img[:, :, 2]

k = st.slider("Number of singular values", 1, len(R), 1)

R_k = compress_channel(R, k)
G_k = compress_channel(G, k)
B_k = compress_channel(B, k)


m, n = R.shape

num_values_per_channel = k * (m + n + 1)
total_values = 3 * num_values_per_channel



st.write(f"Size difference: {img.size - total_values} integers")

R = Image.fromarray(R_k).resize((200, 200), Image.NEAREST)
G = Image.fromarray(G_k).resize((200, 200), Image.NEAREST)
B = Image.fromarray(B_k).resize((200, 200), Image.NEAREST)

pil_img = Image.fromarray(img).resize((200, 200), Image.NEAREST)
rgb_img = Image.merge("RGB", (R, G, B)).resize((200, 200), Image.NEAREST)

col0, col1 = st.columns(2)

with col0:
    st.write(f"Numbers to represent: {img.size} integers")
    st.image(pil_img, caption="Manually constructed image", use_container_width = True)

with col1:
    st.write(f"Numbers to represent: {total_values} integers")
    st.image(rgb_img, caption="Reconstructed image", use_container_width = True)

#     st.image(R, caption="Red channel", use_container_width = True)

# with col2:
#     st.image(G, caption="Green channel", use_container_width = True)

# with col3:
#     st.image(B, caption="Blue channel", use_container_width = True)
