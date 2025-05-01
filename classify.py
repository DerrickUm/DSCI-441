import pandas as pd
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score


# 1. Load your merged data
df = pd.read_csv("data/Master_DatasetValidation.csv")

# 2. Define predictor columns
diet_vars = [
    "Diet.Inv","Diet.Vend","Diet.Vect","Diet.Vfish","Diet.Vunk",
    "Diet.Scav","Diet.Fruit","Diet.Nect","Diet.Seed","Diet.PlantO"
]
combined_vars = [
    "Diet.VertTerrestrial","Diet.VertAll","Diet.AnimalsAll",
    "Diet.PlantHighSugar","Diet.PlantLowSugar","Diet.PlantAll"
]
features = diet_vars + combined_vars

# 3. Target column
target = "DerekDietClassification90InsVertivoreSorting"
df = df.dropna(subset=[target])

X = df[features]
y = df[target]

# 4. Train a fully-grown decision tree (overfit to capture every rule)
dt = DecisionTreeClassifier(
    max_depth=None,          # no depth limit
    min_samples_split=2,     # split until pure
    min_samples_leaf=1,      # allow leaves of size 1
    random_state=0
)
dt.fit(X, y)

# 5. Export the learned rules
rules = export_text(dt, feature_names=features)
print(rules)

# 2. Predict on the training set
y_pred = dt.predict(X)

# 3. Compare to the actual ins-vert column
actual = df["DerekDietClassification90InsVertivoreSorting"].values

# 4. Create a small DataFrame to inspect mismatches
comp = pd.DataFrame({
    "actual": actual,
    "pred":   y_pred
})
comp["match"] = comp["actual"] == comp["pred"]

# 5. Overall accuracy
print("Perfect‐fit accuracy:", accuracy_score(actual, y_pred))

# 6. See where it failed (if any)
fails = comp[~comp["match"]]
print(f"\nNumber of mismatches: {len(fails)}")
print(fails.head(10))

import pandas as pd
from pathlib import Path

# 1. Load the full dataset
master_all = pd.read_csv("data/Master_DatasetAll.csv")

# 2. Recompute the combined columns exactly as you did in validation
master_all["Diet.VertTerrestrial"] = (
    master_all["Diet.Vend"]
  + master_all["Diet.Vect"]
  + master_all["Diet.Vunk"]
)
master_all["Diet.VertAll"]    = master_all["Diet.VertTerrestrial"] + master_all["Diet.Vfish"]
master_all["Diet.AnimalsAll"] = master_all["Diet.VertAll"] + master_all["Diet.Inv"]
master_all["Diet.PlantHighSugar"] = master_all["Diet.Fruit"] + master_all["Diet.Nect"]
master_all["Diet.PlantLowSugar"]  = master_all["Diet.Seed"] + master_all["Diet.PlantO"]
master_all["Diet.PlantAll"]       = (
    master_all["Diet.PlantHighSugar"] + master_all["Diet.PlantLowSugar"]
)

# 3. Build the feature matrix
diet_vars = [
    "Diet.Inv","Diet.Vend","Diet.Vect","Diet.Vfish","Diet.Vunk",
    "Diet.Scav","Diet.Fruit","Diet.Nect","Diet.Seed","Diet.PlantO"
]
combined_vars = [
    "Diet.VertTerrestrial","Diet.VertAll","Diet.AnimalsAll",
    "Diet.PlantHighSugar","Diet.PlantLowSugar","Diet.PlantAll"
]
features = diet_vars + combined_vars

X_all = master_all[features]

master_all["Predicted_Diet_Class"] = dt.predict(X_all)

# 5. Print counts in each predicted class
print("Predicted class distribution:")
print(master_all["Predicted_Diet_Class"].value_counts(), "\n")

# 6. If true labels exist, evaluate and print their distribution too
if "DerekDietClassification90InsVertivoreSorting" in master_all:
    actual = master_all["DerekDietClassification90InsVertivoreSorting"].dropna()
    mask = master_all["DerekDietClassification90InsVertivoreSorting"].notna()
    acc = accuracy_score(actual, master_all.loc[mask, "Predicted_Diet_Class"])
    print("Actual class distribution:")
    print(actual.value_counts(), "\n")
    print(f"Accuracy on labeled subset: {acc:.3f}\n")

# 7. Save out
output_path = Path("data") / "Master_DatasetAll_with_predictions.csv"
master_all.to_csv(output_path, index=False)
print(f"Saved augmented dataset to {output_path}")