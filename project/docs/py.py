import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style

# Set a professional plot style
style.use('seaborn-v0_8-talk')

# --- Data Extracted from the Report (with PointNet data now included) ---
# The data is structured in a dictionary where keys are the method names
# and values are dictionaries of their performance metrics.
# For methods with multiple results (e.g., with and without bagging), the best result was chosen.
performance_data = {
    "LBP + SVM (Indirect)": {
        "Accuracy": 0.794,
        "Precision": 0.77, # From non-bagging table, as bagging table is missing
        "Recall": 0.76,    # From non-bagging table
        "F1-score": 0.77   # From non-bagging table
    },
    "Pretrained CNN + SVM": {
        "Accuracy": 0.718,
        "Precision": 0.72,
        "Recall": 0.72,
        "F1-score": 0.71
    },
    "Transfer Learning": {
        "Accuracy": 0.806,
        "Precision": 0.82,
        "Recall": 0.81,
        "F1-score": 0.81
    },
    "LBP + CNN (Fusion)": {
        "Accuracy": 0.837,
        "Precision": 0.84,
        "Recall": 0.84,
        "F1-score": 0.84
    },
    "Pretrained CNN (Fine-tuned)": {
        "Accuracy": 0.863,
        "Precision": 0.87,
        "Recall": 0.86,
        "F1-score": 0.86
    },
    "FPFH + SVM (Quasi-Direct)": {
        "Accuracy": 0.873, # Best result (with Bagging)
        "Precision": 0.84, # From non-bagging table
        "Recall": 0.83,    # From non-bagging table
        "F1-score": 0.83   # From non-bagging table
    },
    "DGCNN (Direct)": {
        "Accuracy": 0.821,
        "Precision": 0.84,
        "Recall": 0.82,
        "F1-score": 0.81
    },
    "Multi-view CNN (Scratch)": {
        "Accuracy": 0.911,
        "Precision": 0.91,
        "Recall": 0.91,
        "F1-score": 0.91
    },
    "PointNet (Direct)": {
        "Accuracy": 0.7015,
        "Precision": 0.75, # Weighted Avg
        "Recall": 0.70,    # Weighted Avg
        "F1-score": 0.71   # Weighted Avg
    }
}

# Sort the methods by accuracy for a cleaner visualization
sorted_methods = sorted(performance_data.items(), key=lambda item: item[1]['Accuracy'], reverse=True)
method_names = [item[0] for item in sorted_methods]
metrics_data = {metric: [item[1][metric] for item in sorted_methods] for metric in performance_data["PointNet (Direct)"].keys()}

# --- Plotting Logic ---
metrics = list(metrics_data.keys())
num_methods = len(method_names)
num_metrics = len(metrics)

# Set the positions and width for the bars
positions = np.arange(num_methods)
bar_width = 0.20
spacing = 0.02

# Create the figure and axes
fig, ax = plt.subplots(figsize=(18, 10))

# Create bars for each metric
for i, metric in enumerate(metrics):
    # Calculate the offset for each bar in the group
    offset = bar_width * (i - (num_metrics - 1) / 2)
    bars = ax.bar(positions + offset, metrics_data[metric], bar_width, label=metric)
    
    # Add a data label on top of each bar
    ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=10, fontweight='bold')

# --- Customize the Plot ---
ax.set_title('Performance Comparison of Tree Classification Methods', fontsize=20, fontweight='bold', pad=20)
ax.set_ylabel('Score', fontsize=16, fontweight='bold')
ax.set_xticks(positions)
ax.set_xticklabels(method_names, rotation=45, ha="right", fontsize=14)
ax.set_ylim(0, 1.15) # Set y-axis limit to give space for labels
ax.legend(title='Metrics', fontsize=14, title_fontsize=16)
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Add a horizontal line at 1.0 for reference
ax.axhline(1.0, color='gray', linestyle=':', linewidth=1)

# Ensure the layout is tight and clean
fig.tight_layout()

plt.savefig('comparaison.png', dpi=300)
# Display the plot
plt.show()