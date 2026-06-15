import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Same parameters as training
DMD_GRID = (32,32)

CAM_GRID = (32,32)
CAM_REGION_W = 4
CAM_REGION_H = 4

CAM_ORIGIN_X = 1220
CAM_ORIGIN_Y = 1017



def load_folder(folder):

    files = sorted(glob.glob(os.path.join(folder,"*.npy")))

    if len(files)==0:
        raise RuntimeError(folder)

    return files


def dmd_image_to_vector(img):

    img = img.astype(np.float32)

    h,w = img.shape

    bh = h//DMD_GRID[0]
    bw = w//DMD_GRID[1]

    x=[]

    for r in range(DMD_GRID[0]):
        for c in range(DMD_GRID[1]):

            block = img[
                r*bh:(r+1)*bh,
                c*bw:(c+1)*bw
            ]

            x.append(float(block.mean()>127))

    return np.asarray(x,dtype=np.float32)


def camera_image_to_vector(img):

    if img.ndim==3:
        img = img.mean(axis=2)

    roi = img[
        CAM_ORIGIN_Y:CAM_ORIGIN_Y+CAM_GRID[0]*CAM_REGION_H,
        CAM_ORIGIN_X:CAM_ORIGIN_X+CAM_GRID[1]*CAM_REGION_W
    ]

    y=[]

    for r in range(CAM_GRID[0]):
        for c in range(CAM_GRID[1]):

            block = roi[
                r*CAM_REGION_H:(r+1)*CAM_REGION_H,
                c*CAM_REGION_W:(c+1)*CAM_REGION_W
            ]

            y.append(block.mean())

    return np.asarray(y,dtype=np.float32)



def load_dataset(input_folder,output_folder):

    in_files = load_folder(input_folder)
    out_files = load_folder(output_folder)

    X=[]
    Y=[]

    for xf,yf in tqdm(zip(in_files,out_files),
                      total=len(in_files)):

        X.append(
            dmd_image_to_vector(np.load(xf))
        )

        Y.append(
            camera_image_to_vector(np.load(yf))
        )

    return np.asarray(X),np.asarray(Y)



def predict(TM,bias,X):

    field = X @ TM.T + bias

    intensity = np.abs(field)**2

    return intensity



def evaluate(pred,gt):

    r=[]

    mse=[]

    mae=[]

    for p,g in zip(pred,gt):

        r.append(
            pearsonr(
                p,
                g
            )[0]
        )

        mse.append(
            mean_squared_error(g,p)
        )

        mae.append(
            mean_absolute_error(g,p)
        )

    rmse=np.sqrt(mse)

    print()

    print("========== TEST RESULTS ==========")

    print(f"Pearson : {np.mean(r):.4f}")

    print(f"MSE     : {np.mean(mse):.4f}")

    print(f"RMSE    : {np.mean(rmse):.4f}")

    print(f"MAE     : {np.mean(mae):.4f}")

    return r,mse,rmse,mae



def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--tm",required=True)

    parser.add_argument("--bias",required=True)

    parser.add_argument("--input",required=True)

    parser.add_argument("--output",required=True)

    args = parser.parse_args()

    TM=np.load(args.tm)

    bias=np.load(args.bias)

    X,Y=load_dataset(
        args.input,
        args.output
    )

    pred = predict(
        TM,
        bias,
        X
    )

    evaluate(
        pred,
        Y
    )

    plt.figure(figsize=(10,4))

    plt.subplot(121)
    plt.imshow(
        Y[:20],
        aspect="auto",
        cmap="viridis"
    )
    plt.title("Ground Truth")

    plt.subplot(122)
    plt.imshow(
        pred[:20],
        aspect="auto",
        cmap="viridis"
    )
    plt.title("Prediction")

    plt.tight_layout()

    plt.show()


if __name__=="__main__":

    main()