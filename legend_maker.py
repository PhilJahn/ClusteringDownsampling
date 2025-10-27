import os

import matplotlib.pyplot as plt

# makes the legend for figures
if __name__ == "__main__":
    if not os.path.exists("grid_figures"):
        os.makedirs("grid_figures")
    plt.rcParams.update({'font.size': 8})
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]

    plt.figure(figsize=(3.5, 0.7))
    content = [[[-0.2], [0], [0.2], [0.4], [0.6], [0.8], [1.0], [1.2], [1.4], [1.6], [1.8], [2.0]]]
    plt.imshow(content, vmin=0, vmax=2, aspect="auto")
    plt.xticks([0,1,2,3,4,5,6,7,8,9,10, 11],["-∞", 0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200])
    plt.yticks([])
    #plt.title("Legend")
    plt.xlabel("Mean Supervised Score")
    plt.tight_layout()
    plt.savefig("grid_figures/legend.pdf", bbox_inches="tight")
    plt.show()