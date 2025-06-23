import pandas as pd

# Load the results.csv file
csv_path = r"C:\Users\debra\Desktop\results.csv" 
df = pd.read_csv(csv_path)

# Extract the final (last row) metrics
final_metrics = df.iloc[-1]

print("📊 Final Epoch Metrics:")
for col, val in final_metrics.items():
    print(f"{col}: {val:.4f}" if isinstance(val, (int, float)) else f"{col}: {val}")
