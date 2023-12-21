import pandas as pd

filename = "logs.csv"
df = pd.read_csv(filename)
mean_best_score = df["best_score"].mean()

print(f"Mean of 'best_score': {mean_best_score:.4f}")
