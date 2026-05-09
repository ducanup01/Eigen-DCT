from PIL import Image
import numpy as np
from scipy.fft import dct

def performSVD(A):
    A = A.astype(float)

    AtA = A.T @ A
    # AAt = A @ A.T

    # eigenvalues_L, V_L = np.linalg.eig(AAt)
    eigenvalues_R, V_R = np.linalg.eig(AtA)

    # sort the eigen values and eigen vectors
    idx = np.argsort(eigenvalues_R)[::-1]
    eigenvalues_R = eigenvalues_R[idx]
    V_R = V_R[:, idx]

    S = np.sqrt(eigenvalues_R)

    U = A @ V_R
    for i in range(len(S)):
        if S[i] > 1e-9:
            U[:, i] /= S[i]

    return U, np.diag(S), V_R.T


# def compress_channel(A, k):
#     U, S, Vt = performSVD(A)

#     U_k = U[:, :k]
#     S_k = S[:k, :k]
#     Vt_k = Vt[:k, :]

#     A_k = U_k @ S_k @ Vt_k

#     return np.clip(A_k, 0, 255).astype(np.uint8)

# version 2
# def compress_channel(A, k):
#     A = A.astype(float)

#     # Fast, stable SVD
#     U, S, Vt = np.linalg.svd(A, full_matrices=False)

#     # Truncate to rank k
#     U_k = U[:, :k]
#     S_k = S[:k]
#     Vt_k = Vt[:k, :]

#     # Efficient reconstruction (no np.diag needed)
#     A_k = (U_k * S_k) @ Vt_k

#     return np.clip(A_k, 0, 255).astype(np.uint8)

# version 3
def compress_channel(A, k):
    A = A.astype(float)

    # If grayscale (2D), expand to 3D
    if A.ndim == 2:
        A = A[np.newaxis, :, :]
        squeeze_back = True
    else:
        # RGB: (H, W, 3) → (3, H, W)
        A = np.transpose(A, (2, 0, 1))
        squeeze_back = False

    # Batched SVD
    U, S, Vt = np.linalg.svd(A, full_matrices=False)

    # Truncate
    U_k = U[:, :, :k]
    S_k = S[:, :k]
    Vt_k = Vt[:, :k, :]

    # Reconstruct
    A_k = np.matmul(U_k * S_k[:, np.newaxis, :], Vt_k)

    if squeeze_back:
        A_k = A_k[0]
    else:
        A_k = np.transpose(A_k, (1, 2, 0))

    return np.clip(A_k, 0, 255).astype(np.uint8)

img = np.array([
    [[12, 200, 45], [255, 30, 90], [60, 180, 240], [10, 10, 10], [123, 222, 111]],
    [[80, 90, 100], [200, 10, 50], [33, 144, 255], [90, 60, 30], [5, 250, 200]],
    [[170, 80, 40], [20, 200, 220], [140, 140, 140], [255, 120, 0], [75, 25, 190]],
    [[60, 10, 255], [180, 180, 20], [0, 0, 0], [100, 200, 50], [240, 240, 240]],
    [[130, 60, 200], [45, 255, 120], [220, 30, 30], [80, 80, 255], [15, 160, 90]]
], dtype=np.uint8)

R = img[:, :, 0]
G = img[:, :, 1]
B = img[:, :, 2]

U, S, Vt = performSVD(R)

A_reconstructed = U @ S @ Vt

print("Reconstruction error:", np.linalg.norm(R - A_reconstructed))

