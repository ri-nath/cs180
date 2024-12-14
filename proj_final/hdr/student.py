"""
HDR stencil code - student.py
CS 1290 Computational Photography, Brown U.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt


# ========================================================================
# RADIANCE MAP RECONSTRUCTION
# ========================================================================


def solve_g(Z, B, l, w):
    """
    Given a set of pixel values observed for several pixels in several
    images with different exposure times, this function returns the
    imaging system's response function g as well as the log film irradiance
    values for the observed pixels.

    Args:
        Z[i,j]: the pixel values of pixel location number i in image j.
        B[i,j]: the log delta t, or log shutter speed, for image j at pixel i
                (will be the same value for each i within the same j).
        l       lamdba, the constant that determines the amount of
                smoothness.
        w[z]:   the weighting function value for pixel value z (where z is between 0 - 255).

    Returns:
        g[z]:   the log exposure corresponding to pixel value z (where z is between 0 - 255).
        lE[i]:  the log film irradiance at pixel location i.

    """
    n = 256
    N, P = Z.shape[0], Z.shape[1]
    A = np.zeros((N * P + n + 1, N + n), np.float64)
    b = np.zeros((A.shape[0], 1), np.float64)

    for i in range(N):
        for j in range(P):
            k = i * P + j
            wij = w[Z[i, j]]
            A[k, Z[i, j]] = wij; A[k, n + i] = -wij; b[k, 0] = wij * B[i, j]

    A[N * P, 128] = 1

    for i in range(n - 1):
        k = N * P + i + 1
        wi = w[i]
        A[k, i] = l * wi; A[k, i + 1] = -2 * l * wi; A[k, i + 2] = l * wi

    x, _, _, _ = np.linalg.lstsq(A, b)

    return x[:n, 0], x[n:, 0]


def hdr(file_names, g_red, g_green, g_blue, w, exposure_matrix, nr_exposures, scale=5., gamma=0.5):
    """
    Given the imaging system's response function g (per channel), a weighting function
    for pixel intensity values, and an exposure matrix containing the log shutter
    speed for each image, reconstruct the HDR radiance map in accordance to section
    2.2 of Debevec and Malik 1997.

    Args:
        file_names:           exposure stack image filenames
        g_red:                response function g for the red channel.
        g_green:              response function g for the green channel.
        g_blue:               response function g for the blue channel.
        w[z]:                 the weighting function value for pixel value z
                              (where z is between 0 - 255).
        exposure_matrix[i,j]: the log delta t, or log shutter speed, for image j at pixel i
                              (will be the same value for each i within the same j).
        nr_exposures:         number of images / exposures

    Returns:
        hdr:                  the hdr radiance map.
    """
    ims = [cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB) for file in file_names]
    H, W, _ = ims[0].shape
    P, N, C = len(ims), len(ims[0][..., 0].flatten()), 3
    # shape of Z: N, C, P
    Z = np.array([[im[..., c].flatten() for c in range(3)] for im in ims]).transpose(2, 1, 0)

    logt = exposure_matrix[0, :]

    E = np.zeros((N, C))
    for c, g in zip(range(C), (g_red, g_green, g_blue)):
        for i in range(N):
            wi = w[Z[i, c]]
            E[i, c] = wi @ (g[Z[i, c]] - logt) / np.sum(wi)

    return np.array([E[..., c].reshape(H, W) for c in range(3)]).transpose(1, 2, 0)

# ========================================================================
# TONE MAPPING
# ========================================================================


def tm_global_simple(hdr_radiance_map):
    """
    Simple global tone mapping function (Reinhard et al.)

    Equation:
        E_display = E_world / (1 + E_world)

    Args:
        hdr_radiance_map (np.array): HDR radiance map of the image
                                     with shape (H, W, 3)
    Returns:
        np.array of image with values in range [0.0, 1.0]
    """

    return np.clip(hdr_radiance_map / (1 + hdr_radiance_map), 0, 1)


def tm_durand(hdr_radiance_map, dR=4., gamma=0.5):
    """
    Your implementation of:
    http://people.csail.mit.edu/fredo/PUBLI/Siggraph2002/DurandBilateral.pdf

    Args:
        hdr_radiance_map (np.array): HDR radiance map of the image
                                     with shape (H, W, 3)
    Returns:
        np.array of image with values in range [0.0, 1.0]
    """
    r, g, b = [hdr_radiance_map[..., n] for n in range(3)]
    intensity = (r + g + b) / 3
    L = np.log2(np.where(intensity > 1e-5, intensity, 1e-5))
    B = cv2.bilateralFilter(L.astype(np.float32), 10, 75, 75)
    D = L - B
    scale = dR / (np.max(B) - np.min(B))
    B = (B - np.max(B)) * scale
    O = 2 ** (B + D)

    out = np.dstack([O * c / intensity for c in (r, g, b)])

    return np.clip(out, 0, 1)


