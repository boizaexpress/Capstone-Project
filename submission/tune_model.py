import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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
    df.columns = list(iris.feature_names) + ["target"]

X = df.drop(columns=[c for c in df.columns if c == 'target'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, multi_class='multinomial', solver='lbfgs'))

param_grid = {
    'logisticregression__C': [0.01, 0.1, 1.0, 10.0, 100.0]
}

gs = GridSearchCV(pipe, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
print("Starting GridSearchCV...")
gs.fit(X_train, y_train)
print("GridSearchCV done. Best score:", gs.best_score_)

best = gs.best_estimator_
joblib.dump(best, WORKDIR / 'best_model.pkl')

pred = best.predict(X_test)
acc = accuracy_score(y_test, pred)
clf_report = classification_report(y_test, pred, digits=4)
cm = confusion_matrix(y_test, pred)

# save confusion matrix plot
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Tuning Confusion Matrix')
plt.savefig(WORKDIR / 'tuning_confusion_matrix.png', bbox_inches='tight')
plt.close()

# write results
results_md = WORKDIR / 'tuning_results.md'
results_json = WORKDIR / 'tuning_results.json'
with open(results_md, 'w', encoding='utf-8') as f:
    f.write('# Tuning Results\n\n')
    f.write(f'- Best CV score: {gs.best_score_:.4f}\n')
    f.write(f'- Test accuracy: {acc:.4f}\n\n')
    f.write('## Best parameters\n')
    f.write('```\n')
    f.write(json.dumps(gs.best_params_, indent=2))
    f.write('\n```\n\n')
    f.write('## Classification report\n')
    f.write('```\n')
    f.write(clf_report)
    f.write('\n```\n')
    f.write('![Confusion matrix](tuning_confusion_matrix.png)\n')

with open(results_json, 'w', encoding='utf-8') as f:
    json.dump({
        'best_score': gs.best_score_,
        'test_accuracy': acc,
        'best_params': gs.best_params_
    }, f, indent=2)

print('Tuning finished. Artifacts written: best_model.pkl, tuning_results.md, tuning_results.json, tuning_confusion_matrix.png')
