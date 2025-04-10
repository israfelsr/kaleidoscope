import matplotlib.pyplot as plt
import numpy as np

# Example data
model_sizes = [
    "Qwen-VL-3B",
    "Qwen-VL-7B",
    "Qwen-VL-32B",
    "Qwen-VL-72B",
]  # X-axis labels
x_axis = [3, 7, 32, 72]  # Actual x-axis values
overall = [35.63, 39.60, 48.64, 52.95]  # Overall performance
mm = [33.79, 36.88, 45.05, 48.41]  # Multimodal performance
text = [38.53, 43.96, 54.60, 60.01]  # Text-only performance

# Create the plot
plt.figure(figsize=(10, 4))

# Line plot with x_axis values
plt.plot(x_axis, mm, linestyle=":", marker="o", label="Multimodal")
plt.plot(x_axis, text, linestyle=":", marker="o", label="Text-only")
plt.plot(x_axis, overall, linestyle=":", marker="o", label="Overall")

# Add labels and title
plt.ylabel("Accuracy (%)", fontweight="bold", fontsize=8)
plt.legend(
    loc="upper left",  # Force top-left position
    bbox_to_anchor=(0, 1),  # Fine-tune placement (x,y)
    frameon=True,  # Show background
    edgecolor="grey",  # Border color
    fancybox=True,  # Enable rounded corners
    borderpad=0.5,  # Inner padding
    borderaxespad=0.7,  # Space between axes and legend
    framealpha=0.8,  # Slight transparency
)
plt.grid(True, linestyle="--", alpha=0.6)

plt.xscale("log")
# Set x-axis ticks and labels
# plt.xticks(x_axis, model_sizes)  # Use x_axis for positions and model_sizes for labels
plt.xticks(x_axis, model_sizes, ha="right", fontweight="bold", fontsize=8)

# Adjust layout to prevent overlap
plt.tight_layout()
plt.xlim(2, 100)
plt.ylim(30, 70)
for spine in plt.gca().spines.values():
    spine.set_visible(False)

# Show the plot
plt.savefig("scaling_results.svg", format="svg", bbox_inches="tight")
plt.show()
