import os
import matplotlib.pyplot as plt
import numpy as np

def plot_spectra_grid():
    base_dir = os.path.expanduser("~/Desktop/Summer Work 2025/MS_Data")

    files = {
        "Alanine_D$_2$SO$_4$": "Alanine_D2SO4_1Day_300.txt",
        "Alanine_H$_2$O": "Alanine_H2O_1Day_300.txt",
        "Glycine_D$_2$SO$_4$": "Glycine_D2SO4_1Day_300.txt",
        "Glycine_H$_2$O": "Glycine_H2O_1Day_300.txt"
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for idx, (label, filename) in enumerate(files.items()):
        path = os.path.join(base_dir, filename)
        try:
            data = np.loadtxt(path)
            mz, intensity = data[:, 0], data[:, 1]

            ax = axes[idx]
            ax.plot(mz, intensity)
            ax.set_title(label)
            ax.set_xlabel("m/z")
            ax.set_ylabel("Intensity")
            ax.grid(True)
        except Exception as e:
            print(f"Error reading {path}: {e}")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_spectra_grid()
