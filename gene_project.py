import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

print("Data load ho raha hai...")
df = pd.read_csv('human_liver.tsv', sep='\t', index_col=0)

# Sirf top 500 genes — fast hoga
df = df.sample(n=500, random_state=42)
df = df.fillna(0)
X = df.T
y = X.mean(axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Model train ho raha hai...")
model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2  = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"R2 Score : {r2:.4f}")
print(f"MSE      : {mse:.4f}")

# Graph 2
print("Graph 2 ban raha hai...")
importances = pd.Series(
    model.feature_importances_, index=X.columns
).sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 5))
sns.barplot(x=importances.values, y=importances.index, palette='viridis')
plt.xlabel("Importance Score")
plt.title("Top 10 Most Important Genes")
plt.tight_layout()
plt.savefig("top_genes.png")
plt.show()
print("Graph 2 save ho gaya!")
print("Project complete! ✅")