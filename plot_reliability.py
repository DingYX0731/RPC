"""
Plot reliability diagrams (confidence vs accuracy).
Two subplots: (a) SC, (b) RPC. Edit RELIABILITY_DATA below to change input.
"""
import matplotlib.pyplot as plt
import numpy as np

N_BINS = 10

# Example input: method -> list of 10 bin dicts.
# Each bin: bin_low, bin_high, avg_confidence, accuracy_pct, gap_pct, count.
# Replace with your own data or parse from results.txt.
RELIABILITY_DATA = {
    "SC": [
        {"bin_low": 0.0, "bin_high": 0.1, "avg_confidence": 0.0570, "accuracy_pct": 5.00, "gap_pct": 0.70, "count": 20.0},
        {"bin_low": 0.1, "bin_high": 0.2, "avg_confidence": 0.1481, "accuracy_pct": 3.92, "gap_pct": 10.89, "count": 51.0},
        {"bin_low": 0.2, "bin_high": 0.3, "avg_confidence": 0.2511, "accuracy_pct": 16.95, "gap_pct": 8.16, "count": 59.0},
        {"bin_low": 0.3, "bin_high": 0.4, "avg_confidence": 0.3514, "accuracy_pct": 19.48, "gap_pct": 15.66, "count": 77.0},
        {"bin_low": 0.4, "bin_high": 0.5, "avg_confidence": 0.4509, "accuracy_pct": 35.48, "gap_pct": 9.60, "count": 62.0},
        {"bin_low": 0.5, "bin_high": 0.6, "avg_confidence": 0.5589, "accuracy_pct": 44.44, "gap_pct": 11.44, "count": 45.0},
        {"bin_low": 0.6, "bin_high": 0.7, "avg_confidence": 0.6446, "accuracy_pct": 56.10, "gap_pct": 8.37, "count": 41.0},
        {"bin_low": 0.7, "bin_high": 0.8, "avg_confidence": 0.7311, "accuracy_pct": 57.89, "gap_pct": 15.21, "count": 19.0},
        {"bin_low": 0.8, "bin_high": 0.9, "avg_confidence": 0.8549, "accuracy_pct": 57.14, "gap_pct": 28.35, "count": 7.0},
        {"bin_low": 0.9, "bin_high": 1.0, "avg_confidence": 0.9707, "accuracy_pct": 75.00, "gap_pct": 22.07, "count": 8.0},
    ],
    "RPC": [
        {"bin_low": 0.0, "bin_high": 0.1, "avg_confidence": 0.05, "accuracy_pct": 5.0, "gap_pct": 5.0, "count": 20.0},
        {"bin_low": 0.1, "bin_high": 0.2, "avg_confidence": 0.18, "accuracy_pct": 18.0, "gap_pct": 2.0, "count": 50.0},
        {"bin_low": 0.2, "bin_high": 0.3, "avg_confidence": 0.28, "accuracy_pct": 28.0, "gap_pct": 2.0, "count": 55.0},
        {"bin_low": 0.3, "bin_high": 0.4, "avg_confidence": 0.35, "accuracy_pct": 35.0, "gap_pct": 5.0, "count": 70.0},
        {"bin_low": 0.4, "bin_high": 0.5, "avg_confidence": 0.43, "accuracy_pct": 43.0, "gap_pct": 7.0, "count": 60.0},
        {"bin_low": 0.5, "bin_high": 0.6, "avg_confidence": 0.54, "accuracy_pct": 54.0, "gap_pct": 6.0, "count": 48.0},
        {"bin_low": 0.6, "bin_high": 0.7, "avg_confidence": 0.64, "accuracy_pct": 64.0, "gap_pct": 6.0, "count": 42.0},
        {"bin_low": 0.7, "bin_high": 0.8, "avg_confidence": 0.68, "accuracy_pct": 68.0, "gap_pct": 12.0, "count": 22.0},
        {"bin_low": 0.8, "bin_high": 0.9, "avg_confidence": 0.69, "accuracy_pct": 69.0, "gap_pct": 21.0, "count": 10.0},
        {"bin_low": 0.9, "bin_high": 1.0, "avg_confidence": 0.83, "accuracy_pct": 83.0, "gap_pct": 17.0, "count": 9.0},
    ],
}


def plot_reliability_ax(ax, bins, title):
    """Draw one reliability diagram: stacked bar (Predict=blue, Gap=red) + diagonal."""
    if not bins or len(bins) != N_BINS:
        ax.set_title(title + " (no data)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 100)
        return

    x_centers = np.array([(b["bin_low"] + b["bin_high"]) / 2 for b in bins])
    width = 1.0 / N_BINS * 0.85
    accuracy = np.array([b["accuracy_pct"] for b in bins])
    gap = np.array([b["gap_pct"] for b in bins])

    ax.bar(x_centers, accuracy, width=width, color="steelblue", label="Predict", edgecolor="white", linewidth=0.5)
    ax.bar(x_centers, gap, width=width, bottom=accuracy, color="coral", label="Gap", edgecolor="white", linewidth=0.5)

    ax.plot([0, 1], [0, 100], "k--", linewidth=1.5, label="Perfect calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)


def main():
    data = RELIABILITY_DATA
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    plot_reliability_ax(axes[0], data.get("SC"), "SC")
    plot_reliability_ax(axes[1], data.get("RPC"), "RPC")

    plt.tight_layout()
    plt.savefig("reliability_diagram.png", dpi=150, bbox_inches="tight")
    plt.savefig("reliability_diagram.pdf", bbox_inches="tight")
    print("Saved reliability_diagram.png and reliability_diagram.pdf")
    plt.show()


if __name__ == "__main__":
    main()
