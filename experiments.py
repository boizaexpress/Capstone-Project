import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns

WORKDIR = Path(__file__).parent
CSV = WORKDIR / "sample_iris.csv"
if CSV.exists():
    df = pd.read_csv(CSV)
else:
    from sklearn.datasets import load_iris
    iris = load_iris(as_frame=True)
    df = iris.frame
    df.columns = list(iris.feature_names) + ['target']

X = df.drop(columns=[c for c in df.columns if c == 'target'])
y = df['target']

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
models = {
    'LogisticRegression': Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=2000))]),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVC': Pipeline([('scaler', StandardScaler()), ('clf', SVC(probability=True))])
}

results = {}
for name, model in models.items():
    print(f'Running CV for {name}...')
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    results[name] = {
        'fold_scores': scores.tolist(),
        'mean_accuracy': float(scores.mean()),
        'std_accuracy': float(scores.std())
    }

# Save results
with open(WORKDIR / 'experiments_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2)

# Markdown summary
with open(WORKDIR / 'experiments_results.md', 'w', encoding='utf-8') as f:
    f.write('# Experiments: 5-fold CV comparison\n\n')
    for name, r in results.items():
        f.write(f'## {name}\n')
        f.write(f'- Fold scores: {r["fold_scores"]}\n')
        f.write(f'- Mean accuracy: {r["mean_accuracy"]:.4f}\n')
        f.write(f'- Std accuracy: {r["std_accuracy"]:.4f}\n\n')

# Boxplot
plt.figure(figsize=(6,4))
data = [results[name]['fold_scores'] for name in models.keys()]
plt.boxplot(data, labels=list(models.keys()))
plt.title('CV accuracy comparison')
plt.ylabel('Accuracy')
plt.grid(axis='y')
plt.savefig(WORKDIR / 'experiments_cv_boxplot.png', bbox_inches='tight')
plt.close()

print('Experiments complete: experiments_results.json, experiments_results.md, experiments_cv_boxplot.png')
