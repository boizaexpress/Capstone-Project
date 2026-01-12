Capstone Project - Automated run

This repo contains a notebook template and a small synthetic dataset used to demonstrate the workflow.

Files added by assistant:
- `data/your_file.csv` : small synthetic dataset
- `scripts/train_and_report.py` : script to train a Logistic Regression baseline and save the model
- `models/final_model.joblib` : (created after running the script)
- `results.txt` : classification report and accuracy

To reproduce locally:

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
python scripts/train_and_report.py
```
