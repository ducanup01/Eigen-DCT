import streamlit as st
from functions import *

file_uploaded = Image.open("sample_image.png")
if not file_uploaded:
    file_uploaded = st.file_uploader("Upload image here", type = "image/*")

YCbCr_image, YCbCr_array = convert_to_YCbCr(file_uploaded)

st.image(YCbCr_image)

Y  = YCbCr_array[:, :, 0]
Cb = YCbCr_array[:, :, 1]
Cr = YCbCr_array[:, :, 2]

Y_ds, Cb_ds, Cr_ds = down_sampling(Y, Cb, Cr)

Y_pad, Cb_pad, Cr_pad = pad_channels(Y_ds, Cb_ds, Cr_ds)

Y_dct  = apply_dct_fast(Y_pad)
Cb_dct = apply_dct_fast(Cb_pad)
Cr_dct = apply_dct_fast(Cr_pad)

import numpy as np

# Number of blocks
num_blocks_h = Y_pad.shape[0] // 8
num_blocks_w = Y_pad.shape[1] // 8

# Sliders to pick block
block_row = st.slider("Block Row", 0, num_blocks_h - 1)
block_col = st.slider("Block Col", 0, num_blocks_w - 1)

# Extract 8x8 blocks
y_block = Y_pad[block_row*8:(block_row+1)*8,
                block_col*8:(block_col+1)*8]

dct_block = Y_dct[block_row*8:(block_row+1)*8,
                  block_col*8:(block_col+1)*8]


col1, col2 = st.columns(2)

with col1:
    st.write("Original 8*8 Block (Y)")
    st.dataframe(y_block)

with col2:
    st.write("DCT Coefficients")
    st.dataframe(np.round(dct_block, 2))


pixel_y = block_row * 8
pixel_x = block_col * 8

import numpy as np
import cv2
import streamlit as st

# ----------------------------
# RGB block (handle RGBA safely)
# ----------------------------
rgb_array = np.array(file_uploaded)

if rgb_array.shape[-1] == 4:
    rgb_array = rgb_array[:, :, :3]

rgb_block = rgb_array[pixel_y:pixel_y+8, pixel_x:pixel_x+8]

# ----------------------------
# YCbCr blocks
# ----------------------------
y_block  = Y[pixel_y:pixel_y+8, pixel_x:pixel_x+8]
cb_block = Cb[pixel_y:pixel_y+8, pixel_x:pixel_x+8]
cr_block = Cr[pixel_y:pixel_y+8, pixel_x:pixel_x+8]

# ----------------------------
# DCT block (Y channel)
# ----------------------------
dct_block = Y_dct[pixel_y:pixel_y+8, pixel_x:pixel_x+8]


# ----------------------------
# Upscale function for display
# ----------------------------
def upscale(block, scale=20):
    return cv2.resize(
        block,
        (8 * scale, 8 * scale),
        interpolation=cv2.INTER_NEAREST
    )


# ----------------------------
# Display RGB + YCbCr
# ----------------------------
col1, col2 = st.columns(2)

with col1:
    st.write("RGB 8×8 Block")
    st.image(upscale(rgb_block))

with col2:
    st.write("Y 8×8 Block")
    st.image(upscale(y_block), clamp=True)


col3, col4 = st.columns(2)

with col3:
    st.write("Cb Block")
    st.image(upscale(cb_block), clamp=True)

with col4:
    st.write("Cr Block")
    st.image(upscale(cr_block), clamp=True)


# ----------------------------
# DCT visualization
# ----------------------------
st.write("DCT Coefficients (log scale)")

dct_vis = np.log(np.abs(dct_block) + 1)
st.image(upscale(dct_vis / dct_vis.max()))