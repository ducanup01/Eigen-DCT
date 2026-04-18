import streamlit as st
from PIL import Image
import numpy as np
from scipy.fft import dct

def convert_to_YCbCr(file):
    # If it's already a PIL Image, use it directly
    if isinstance(file, Image.Image):
        img = file
    else:
        img = Image.open(file)

    # Convert to numpy array
    img_array = np.array(img)

    # Handle grayscale (no channel dimension)
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array]*3, axis=-1)

    # Handle RGBA
    elif img_array.shape[-1] == 4:
        rgb = img_array[:, :, :3].copy()
        alpha = img_array[:, :, 3]

        # Threshold: treat low alpha as transparent
        threshold = 75
        mask = (alpha < threshold)

        # Set those pixels to white
        rgb[mask] = 255

        img_array = rgb

    # Convert back to RGB image
    img = Image.fromarray(img_array.astype("uint8"), "RGB")

    # Convert to YCbCr
    ycbcr_img = img.convert("YCbCr")
    ycbcr_array = np.array(ycbcr_img)

    return ycbcr_img, ycbcr_array

def down_sampling(Y, Cb, Cr):
    # Helper: 2x2 average downsampling
    def downsample_2x2(channel):
        h, w = channel.shape

        # Make dimensions even (important!)
        h_even = h - (h % 2)
        w_even = w - (w % 2)

        channel = channel[:h_even, :w_even]

        return (
            channel[0::2, 0::2] +
            channel[0::2, 1::2] +
            channel[1::2, 0::2] +
            channel[1::2, 1::2]
        ) // 4

    # Y stays the same
    Y_ds = Y

    # Downsample chroma channels
    Cb_ds = downsample_2x2(Cb)
    Cr_ds = downsample_2x2(Cr)

    return Y_ds, Cb_ds, Cr_ds

def pad_channels(Y, Cb, Cr):

    def pad_channel(channel, block_size=8):
        h, w = channel.shape

        pad_h = (block_size - h % block_size) % block_size
        pad_w = (block_size - w % block_size) % block_size

        padded = np.pad(
            channel,
            ((0, pad_h), (0, pad_w)),
            mode='edge'
        )

        return padded
    
    Y_pad  = pad_channel(Y)
    Cb_pad = pad_channel(Cb)
    Cr_pad = pad_channel(Cr)

    return Y_pad, Cb_pad, Cr_pad

def dct_2d_fast(block):
    block = block.astype(float) - 128
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def apply_dct_fast(channel):
    h, w = channel.shape
    dct_coeffs = np.zeros((h, w))

    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = channel[i:i+8, j:j+8]
            dct_block = dct_2d_fast(block)
            dct_coeffs[i:i+8, j:j+8] = dct_block

    return dct_coeffs