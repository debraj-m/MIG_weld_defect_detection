import pandas as pd

# Load CSVs
raw_df = pd.read_csv("raw_detections.csv")
filtered_df = pd.read_csv("filtered_detections.csv")

# Compare total detections
print("📊 TOTAL DETECTIONS:")
print(f"🔹 Raw: {len(raw_df)}")
print(f"🔹 Filtered: {len(filtered_df)}")
print(f"🔻 Reduction: {len(raw_df) - len(filtered_df)} detections ({(1 - len(filtered_df)/len(raw_df)) * 100:.2f}%)\n")

# Class-wise comparison
raw_class_counts = raw_df["class_name"].value_counts().sort_index()
filtered_class_counts = filtered_df["class_name"].value_counts().sort_index()

print("📊 CLASS-WISE COMPARISON:")
comparison = pd.DataFrame({
    "Raw": raw_class_counts,
    "Filtered": filtered_class_counts
}).fillna(0).astype(int)
comparison["Reduced"] = comparison["Raw"] - comparison["Filtered"]
comparison["% Reduced"] = (comparison["Reduced"] / comparison["Raw"] * 100).round(2)
print(comparison)

# Save results
comparison.to_csv("classwise_comparison.csv")
print("\n✅ Class-wise comparison saved to classwise_comparison.csv")
