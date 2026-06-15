#!/usr/bin/env python3
"""
compute_tm.py

Compute the Transmission Matrix (TM) using the
Gerchberg-Saxton algorithm for binary amplitude modulation.

Author: ChatGPT
"""

import os
import glob
import argparse

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm


# ==========================================================
# Parameters
# ==========================================================

DMD_GRID = (32, 32)

CAM_GRID = (32, 32)

CAM_REGION_W = 4
CAM_REGION_H = 4

CAM_ORIGIN_X = 1220
CAM_ORIGIN_Y = 1017

N_ITER = 200


# ==========================================================
# Utilities
# ==========================================================

def load_folder(folder):

    files = sorted(glob.glob(os.path.join(folder, "*.npy")))

    if len(files) == 0:
        raise RuntimeError(f"No npy files found in {folder}")

    return files


# ==========================================================
# DMD image -> binary vector
# ==========================================================

def dmd_image_to_vector(img):

    img = img.astype(np.float32)

    h, w = img.shape

    bh = h // DMD_GRID[0]
    bw = w // DMD_GRID[1]

    vec = []

    for r in range(DMD_GRID[0]):
        for c in range(DMD_GRID[1]):

            block = img[
                r*bh:(r+1)*bh,
                c*bw:(c+1)*bw
            ]

            vec.append(float(block.mean() > 127))

    return np.asarray(vec, dtype=np.float32)


# ==========================================================
# Camera image -> region averages
# ==========================================================

def camera_image_to_vector(img):

    if img.ndim == 3:
        img = img.mean(axis=2)

    roi = img[
        CAM_ORIGIN_Y:CAM_ORIGIN_Y + CAM_GRID[0]*CAM_REGION_H,
        CAM_ORIGIN_X:CAM_ORIGIN_X + CAM_GRID[1]*CAM_REGION_W
    ]

    vec = []

    for r in range(CAM_GRID[0]):
        for c in range(CAM_GRID[1]):

            block = roi[
                r*CAM_REGION_H:(r+1)*CAM_REGION_H,
                c*CAM_REGION_W:(c+1)*CAM_REGION_W
            ]

            vec.append(block.mean())

    return np.asarray(vec, dtype=np.float32)


# ==========================================================
# Load training dataset
# ==========================================================

def load_dataset(input_folder,
                 output_folder):

    input_files = load_folder(input_folder)

    output_files = load_folder(output_folder)

    assert len(input_files) == len(output_files)

    X = []

    Y = []

    print("Loading dataset...")

    for xin, yout in tqdm(list(zip(input_files,
                                   output_files))):

        x = np.load(xin)

        y = np.load(yout)

        X.append(
            dmd_image_to_vector(x)
        )

        Y.append(
            camera_image_to_vector(y)
        )

    X = np.asarray(X,
                   dtype=np.float32)

    Y = np.asarray(Y,
                   dtype=np.float32)

    return X, Y


# ==========================================================
# Gerchberg Saxton
# ==========================================================

def ggs_dmd(X,
            Y,
            n_iter=200,
            T_aug=None):

    K, N = X.shape

    M = Y.shape[1]

    if K < 4*(N+1):

        print(
            "WARNING:"
            " too few measurements."
        )

    A_out = np.sqrt(
        np.clip(Y,
                0,
                None)
    ).astype(np.complex64)

    X_aug = np.hstack(
        [
            X,
            np.ones((K,1),
            dtype=np.float32)
        ]
    )

    X_pinv = np.linalg.pinv(X_aug)

    if T_aug is None:

        T_aug = (
            X_pinv @
            Y.astype(np.float32)
        ).astype(np.complex64)

    residuals = []

    print("Phase 1")

    for _ in tqdm(range(n_iter//2)):

        Y_pred = X_aug @ T_aug

        E = (
            A_out**2
        ) * np.exp(
            1j*np.angle(Y_pred)
        )

        T_aug = X_pinv @ E

        residual = np.mean(
            (
                np.abs(Y_pred)**2
                - Y
            )**2
        )

        residuals.append(
            float(residual)
        )

    print("Phase 2")

    for _ in tqdm(range(n_iter//2)):

        Y_pred = X_aug @ T_aug

        E = (
            A_out
        ) * np.exp(
            1j*np.angle(Y_pred)
        )

        T_aug = X_pinv @ E

        residual = np.mean(
            (
                np.abs(Y_pred)**2
                - Y
            )**2
        )

        residuals.append(
            float(residual)
        )

    TM = T_aug[:-1].T

    bias = T_aug[-1]

    return TM, bias, residuals



# ==========================================================
# Main
# ==========================================================

def main():

    parser = argparse.ArgumentParser(
        description="Compute Transmission Matrix using Gerchberg-Saxton"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Training input folder (*.npy)"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Training output folder (*.npy)"
    )

    parser.add_argument(
        "--save",
        default="results",
        help="Directory to save TM"
    )

    parser.add_argument(
        "--iterations",
        default=N_ITER,
        type=int,
        help="Number of GS iterations"
    )

    args = parser.parse_args()

    os.makedirs(args.save, exist_ok=True)

    print("=" * 60)
    print("Loading training dataset...")
    print("=" * 60)

    X, Y = load_dataset(
        args.input,
        args.output
    )

    print()
    print("Dataset summary")
    print("-------------------------")
    print(f"Samples      : {len(X)}")
    print(f"Input modes  : {X.shape[1]}")
    print(f"Output modes : {Y.shape[1]}")
    print(f"Mean output  : {Y.mean():.3f}")
    print(f"Max output   : {Y.max():.3f}")
    print()

    print("=" * 60)
    print("Running Gerchberg-Saxton...")
    print("=" * 60)

    TM, bias, residuals = ggs_dmd(
        X,
        Y,
        n_iter=args.iterations
    )

    print()
    print("=" * 60)
    print("Evaluating training reconstruction")
    print("=" * 60)

    # Predict complex field
    field = X @ TM.T + bias

    # Predicted intensity
    Y_pred = np.abs(field) ** 2

    # Metrics
    correlation = np.corrcoef(
        Y_pred.ravel(),
        Y.ravel()
    )[0, 1]

    mse = np.mean(
        (Y_pred - Y) ** 2
    )

    rmse = np.sqrt(mse)

    mae = np.mean(
        np.abs(Y_pred - Y)
    )

    print(f"Pearson r : {correlation:.4f}")
    print(f"MSE       : {mse:.6f}")
    print(f"RMSE      : {rmse:.6f}")
    print(f"MAE       : {mae:.6f}")

    # Save
    np.save(
        os.path.join(args.save, "TM.npy"),
        TM
    )

    np.save(
        os.path.join(args.save, "bias.npy"),
        bias
    )

    np.save(
        os.path.join(args.save, "residuals.npy"),
        np.asarray(residuals)
    )

    # Save training prediction (optional)
    np.save(
        os.path.join(args.save, "train_prediction.npy"),
        Y_pred
    )

    # Residual plot
    plt.figure(figsize=(7,4))
    plt.semilogy(residuals)
    plt.xlabel("Iteration")
    plt.ylabel("Residual")
    plt.title("Gerchberg-Saxton Convergence")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(args.save, "convergence.png"),
        dpi=200
    )

    # TM magnitude
    plt.figure(figsize=(6,5))
    plt.imshow(
        np.abs(TM),
        cmap="viridis",
        aspect="auto"
    )
    plt.colorbar(label="|TM|")
    plt.xlabel("Input Mode")
    plt.ylabel("Output Mode")
    plt.title("Recovered Transmission Matrix")
    plt.tight_layout()
    plt.savefig(
        os.path.join(args.save, "TM_magnitude.png"),
        dpi=200
    )

    # TM phase
    plt.figure(figsize=(6,5))
    plt.imshow(
        np.angle(TM),
        cmap="hsv",
        aspect="auto",
        vmin=-np.pi,
        vmax=np.pi
    )
    plt.colorbar(label="Phase (rad)")
    plt.xlabel("Input Mode")
    plt.ylabel("Output Mode")
    plt.title("Recovered TM Phase")
    plt.tight_layout()
    plt.savefig(
        os.path.join(args.save, "TM_phase.png"),
        dpi=200
    )

    print()
    print("=" * 60)
    print("Finished.")
    print("=" * 60)
    print(f"Results saved to: {args.save}")
    print("Generated files:")
    print("  TM.npy")
    print("  bias.npy")
    print("  residuals.npy")
    print("  train_prediction.npy")
    print("  convergence.png")
    print("  TM_magnitude.png")
    print("  TM_phase.png")


if __name__ == "__main__":
    main()