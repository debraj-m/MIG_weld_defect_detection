import pandas as pd

# Load CSVs
default_df = pd.read_csv("filtered_detections.csv")
tuned_df = pd.read_csv("threshold_detections.csv")

# Count detections per image
default_counts = default_df["image"].value_counts().sort_index()
tuned_counts = tuned_df["image"].value_counts().sort_index()

# Merge into single DataFrame
comparison = pd.DataFrame({
    "Default": default_counts,
    "Tuned": tuned_counts
}).fillna(0).astype(int)

# Calculate % change
comparison["Change (%)"] = ((comparison["Tuned"] - comparison["Default"]) / comparison["Default"].replace(0, 1) * 100).round(2)

# Save
comparison.to_csv("imagewise_detection_comparison.csv")
print("✅ Saved image-wise detection comparison to 'imagewise_detection_comparison.csv'")

# Print summary
print("\n📊 Detection Summary:")
print(f"🔹 Default total detections: {default_df.shape[0]}")
print(f"🔸 Tuned total detections:   {tuned_df.shape[0]}")
print(f"🔻 Difference: {default_df.shape[0] - tuned_df.shape[0]} ({(1 - tuned_df.shape[0]/default_df.shape[0])*100:.2f}%)")
