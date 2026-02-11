import shutil
from pathlib import Path

root = Path(r"C:\Users\user\Downloads\Capstone project")
sub = root / 'submission'
sub.mkdir(exist_ok=True)
files = [
    'QCTO---Workplace-Module-Notebook-Template-4571 (1).ipynb',
    'QCTO---Workplace-Module-Notebook-Template-4571.ipynb',
    'export_results.py',
    'tune_model.py',
    'tune_more.py',
    'experiments.py',
    'results_report.md',
    'results_report.html',
    'tuning_results.md',
    'tuning_results.json',
    'tune_more_results.md',
    'tune_more_results.json',
    'experiments_results.md',
    'experiments_results.json',
    'baseline_model.pkl',
    'best_model.pkl',
    'best_model_logreg.pkl',
    'best_model_rf.pkl',
    'best_model_svc.pkl',
    'confusion_matrix.png',
    'tuning_confusion_logreg.png',
    'tuning_confusion_rf.png',
    'tuning_confusion_svc.png',
    'experiments_cv_boxplot.png',
    'README.md',
    'requirements.txt',
    'PR_DESCRIPTION_pr_add-readme-tuning.md'
]

copied = []
for f in files:
    p = root / f
    if p.exists():
        shutil.copy2(p, sub / p.name)
        copied.append(p.name)

zip_path = root / 'Capstone_submission.zip'
if zip_path.exists():
    zip_path.unlink()
shutil.make_archive(str(zip_path.with_suffix('')), 'zip', root_dir=sub)

print('Copied files:')
for c in copied:
    print(' -', c)
print('\nCreated zip:', zip_path)
