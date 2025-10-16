import pandas as pd


data = {
    'Student': [1, 2, 3, 4, 5],
    'Study Hours': [2, 4, 6, 8, 10],
    'Pass': [0, 0, 1, 1, 1]
}
df = pd.DataFrame(data)


def gini_impurity(labels):
    total = len(labels)
    if total == 0:
        return 0.0
    p1 = sum(labels)/total
    p0 = 1 - p1
    return 1 - p1**2 - p0**2

# Find possible splits 
study_hours = sorted(df['Study Hours'].unique())
splits = [(study_hours[i] + study_hours[i+1]) / 2 for i in range(len(study_hours) - 1)]


step_results = []
for split in splits:
    left = df[df['Study Hours'] <= split]['Pass'].tolist()
    right = df[df['Study Hours'] > split]['Pass'].tolist()
    gini_left = gini_impurity(left)
    gini_right = gini_impurity(right)
    weighted_gini = (len(left)/len(df)) * gini_left + (len(right)/len(df)) * gini_right
    step_results.append({
        'split': split,
        'left_group': left,
        'right_group': right,
        'gini_left': gini_left,
        'gini_right': gini_right,
        'weighted_gini': weighted_gini
    })


for res in step_results:
    print(f"Split at {res['split']}:")
    print(f" Left group: {res['left_group']}, Gini: {res['gini_left']:.3f}")
    print(f" Right group: {res['right_group']}, Gini: {res['gini_right']:.3f}")
    print(f" Weighted Gini: {res['weighted_gini']:.3f}\n")


best_split = min(step_results, key=lambda x: x['weighted_gini'])
print(f"Best split is at Study Hours = {best_split['split']}")
print("Decision tree:")
print(f"  If Study Hours <= {best_split['split']}, Predict:", 
      "Pass" if all(x == 1 for x in best_split['left_group']) else "Fail")
print(f"  If Study Hours > {best_split['split']}, Predict:",
      "Pass" if all(x == 1 for x in best_split['right_group']) else "Fail")
