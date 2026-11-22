# Credit Risk AI Analysis — ANN Frontend

A Streamlit frontend for the PyTorch ANN trained in `Credit_Risk_Prediction_ANN.ipynb`.

## Required model artifacts

Place these three files in the same project folder as `app.py`:

- `credit_risk_ann.pth`
- `credit_risk_scaler.joblib`
- `credit_risk_feature_names.joblib`

They are produced by running the save-model cell in the training notebook.

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this folder and the three model artifacts to a GitHub repository.
2. Sign in at https://share.streamlit.io/ using GitHub.
3. Select the repository, branch, and `app.py` as the main file.
4. Deploy.

## Important implementation note

The frontend recreates the same one-row pandas One-Hot Encoding, aligns columns to the saved training feature order, applies the saved scaler, and then loads the PyTorch model. Keep the preprocessing and model artifacts from the exact same training run.

This is an academic demonstration, not a production credit decision system. The model uses a historical educational dataset and its output should not be used to approve or reject real applicants.
