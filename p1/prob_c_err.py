import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    u_64  = np.load("./p1/u_64.npy")
    u_128 = np.load("./p1/u_128.npy")[0::2, 0::2]
    u_256 = np.load("./p1/u_256.npy")[0::4, 0::4]
    # u_gt =  np.load("./p1/u_512.npy")[0::8, 0::8]
    u_gt = np.load("./p1/u_64_b.npy")

    N_list = [64, 128, 256, 512]
    err_list = [
        np.linalg.norm(u_64 - u_gt, ord="fro"),
        np.linalg.norm(u_128 - u_gt, ord="fro"),
        np.linalg.norm(u_256 - u_gt, ord="fro"),
        0
    ]

    fig, ax = plt.subplots(1, 1, figsize=(6, 4), facecolor="white", dpi=120)
    ax.plot(N_list, err_list, 'ro-')
    ax.set_xlabel("Number of cells (N)")
    ax.set_ylabel("L2 norm")
    ax.set_title("N vs Accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig("p1_c_err.png", dpi=120)
    plt.close(fig)