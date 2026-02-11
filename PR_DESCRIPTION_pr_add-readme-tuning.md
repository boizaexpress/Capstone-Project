PR: Add README, tuning scripts, and experiment artifacts

Summary

This PR adds:
- `README.md` describing the project and how to run tuning scripts
- `requirements.txt` listing minimal dependencies
- `tune_more.py` for multi-model randomized tuning
- `experiments.py` for 5-fold CV comparisons
- Generated artifacts: best models and result files

Notes for reviewer

- The notebook was tidied: corrupted EDA cell removed and replaced with a clean EDA cell.
- Large binary artifacts (models and images) were included for reproducibility; consider moving them to `artifacts/` or adding a `.gitattributes`/LFS if desired.

Requested reviewers: Yves Matanga

Suggested checklist

- [ ] Confirm README accuracy
- [ ] Spot-check `tune_more.py` and `experiments.py` for reproducibility
- [ ] Approve inclusion of model artifacts
