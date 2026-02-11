import json
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

models = {
    'logreg': (Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=2000))]),
               {'clf__C': [0.01, 0.1, 1, 10], 'clf__penalty': ['l2']}),
    'rf': (Pipeline([('scaler', StandardScaler()), ('clf', RandomForestClassifier())]),
           {'clf__n_estimators': [50, 100, 200], 'clf__max_depth': [None, 3, 5, 10]}),
    'svc': (Pipeline([('scaler', StandardScaler()), ('clf', SVC(probability=True))]),
            {'clf__C': [0.1, 1, 10], 'clf__gamma': ['scale', 'auto']})
}

results = {}

for name, (pipe, param_dist) in models.items():
    print(f"Tuning {name}...")
    rs = RandomizedSearchCV(pipe, param_distributions=param_dist, n_iter=6, cv=3, scoring='accuracy', n_jobs=-1, random_state=42, verbose=0)
    rs.fit(X_train, y_train)
    best = rs.best_estimator_
    pred = best.predict(X_test)
    acc = accuracy_score(y_test, pred)
    creport = classification_report(y_test, pred, digits=4)
    cm = confusion_matrix(y_test, pred)

    # save model
    model_fp = WORKDIR / f'best_model_{name}.pkl'
    joblib.dump(best, model_fp)

    # save confusion matrix plot
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name} confusion matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    out_img = WORKDIR / f'tuning_confusion_{name}.png'
    plt.savefig(out_img, bbox_inches='tight')
    plt.close()

    results[name] = {
        'best_cv_score': float(rs.best_score_),
        'test_accuracy': float(acc),
        'best_params': rs.best_params_,
        'model_file': str(model_fp),
        'confusion_image': str(out_img),
        'classification_report': creport
    }

# Write summary files
with open(WORKDIR / 'tune_more_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2)

with open(WORKDIR / 'tune_more_results.md', 'w', encoding='utf-8') as f:
    f.write('# Multi-model Tuning Results\n\n')
    for name, r in results.items():
        f.write(f'## {name}\n')
        f.write(f'- Best CV score: {r["best_cv_score"]:.4f}\n')
        f.write(f'- Test accuracy: {r["test_accuracy"]:.4f}\n')
        f.write('```\n')
        f.write(json.dumps(r['best_params'], indent=2))
        f.write('\n```\n\n')
        f.write('### Classification report\n')
        f.write('```\n')
        f.write(r['classification_report'])
        f.write('\n```\n')
        f.write(f'![Confusion matrix]({r["confusion_image"]})\n\n')

print('Multi-model tuning finished. Artifacts: tune_more_results.json, tune_more_results.md, best_model_<name>.pkl, tuning_confusion_<name>.png')
