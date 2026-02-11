import os
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

out_dir = os.getcwd()
print('Output directory:', out_dir)

# Load dataset
csv_path = os.path.join(out_dir, 'sample_iris.csv')
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
else:
    try:
        from sklearn.datasets import load_iris
        ir = load_iris(as_frame=True)
        df = ir.frame
    except Exception as e:
        raise SystemExit('No data available and sklearn import failed: ' + str(e))

# Load model
model_path = os.path.join(out_dir, 'baseline_model.pkl')
if not os.path.exists(model_path):
    raise SystemExit(f'Model file not found: {model_path}')
model = joblib.load(model_path)

# Prepare X, y
if 'target' in df.columns:
    y = df['target']
    X = df.drop(columns=['target'])
else:
    y = df.iloc[:, -1]
    X = df.iloc[:, :-1]

X_numeric = X.select_dtypes(include=[np.number]).copy()
if X_numeric.shape[1] != X.shape[1]:
    dropped = set(X.columns) - set(X_numeric.columns)
    print('Dropping non-numeric columns for evaluation:', dropped)

# Split (same random_state/stratify as notebook)
stratify = y if len(np.unique(y)) > 1 else None
X_train, X_test, y_train, y_test = train_test_split(X_numeric, y, test_size=0.2, random_state=42, stratify=stratify)

# Predict and evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
crep = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Save confusion matrix figure
cm_path = os.path.join(out_dir, 'confusion_matrix.png')
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig(cm_path)
plt.close()
print('Saved confusion matrix to', cm_path)

# Write Markdown report
md_path = os.path.join(out_dir, 'results_report.md')
with open(md_path, 'w', encoding='utf-8') as f:
    f.write('# Modeling Results Report\n\n')
    f.write('**Dataset:** sample_iris.csv\n\n')
    f.write(f'**Dataset shape:** {df.shape[0]} rows x {df.shape[1]} columns\n\n')
    f.write('---\n\n')
    f.write('## Baseline Model\n\n')
    f.write('- Model file: `baseline_model.pkl`\n')
    f.write(f'- Accuracy: **{acc:.4f}**\n\n')
    f.write('### Classification report\n\n')
    f.write('```\n')
    f.write(crep)
    f.write('\n```\n')
    f.write('\n### Confusion matrix\n\n')
    f.write('![Confusion matrix](confusion_matrix.png)\n')

print('Wrote Markdown report to', md_path)

# Write a simple HTML report
html_path = os.path.join(out_dir, 'results_report.html')
html_content = f'''<!doctype html>
<html>
<head><meta charset="utf-8"><title>Modeling Results Report</title></head>
<body>
<h1>Modeling Results Report</h1>
<p><strong>Dataset:</strong> sample_iris.csv<br>
<strong>Dataset shape:</strong> {df.shape[0]} rows x {df.shape[1]} columns</p>
<hr>
<h2>Baseline Model</h2>
<ul>
<li>Model file: baseline_model.pkl</li>
<li>Accuracy: <strong>{acc:.4f}</strong></li>
</ul>
<h3>Classification report</h3>
<pre>{crep}</pre>
<h3>Confusion matrix</h3>
<p><img src="confusion_matrix.png" alt="Confusion matrix" style="max-width:600px"></p>
</body>
</html>'''
with open(html_path, 'w', encoding='utf-8') as f:
    f.write(html_content)
print('Wrote HTML report to', html_path)

print('\nDone.')
