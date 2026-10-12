# TVFM Dashboard Prototype

**Trustworthy Vision Foundation Model (TVFM) — chest CT research dashboard**

This is a local research dashboard prototype. It does not include a trained lung cancer model and does not generate diagnostic predictions. It deliberately displays model outputs as unavailable until a real, tested checkpoint and its preprocessing pipeline are integrated.

## Included features
- Dark research dashboard and navigation
- Upload and visual inspection of PNG/JPG/TIFF/BMP 2D images
- Pseudonymous study ID, SHA-256 file hash, session-only study manifest export
- Evaluation lab for a CSV containing real held-out `y_true` and `y_score` values
- AUROC, sensitivity, specificity, precision, F1, Brier score, confusion matrix, score distribution
- Trustworthiness checklist export
- Research protocol and reference links
- JSON export for evaluation and research session metadata

## Important limitations
- No trained checkpoint is included.
- No diagnostic inference, malignancy probability, segmentation, or heatmap is generated.
- The viewer currently accepts 2D common image formats, not complete DICOM series or NIfTI volumes.
- Study uploads are stored in Streamlit session memory only and are not persisted to a database.
- This prototype is not for clinical use. Do not upload identifiable patient data.
- Metrics are only as valid as the uploaded labels and predictions. The app cannot verify patient-level split integrity or reference-standard quality.

## Run on Windows

1. Install Python 3.10 or newer.
2. Open Command Prompt in the extracted `TVFM_dashboard_prototype` folder.
3. Create a virtual environment:

   ```bat
   py -m venv .venv
   .venv\Scripts\activate
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   streamlit run app.py
   ```

4. Open the local URL Streamlit prints (usually `http://localhost:8501`).

## Run on Google Colab

The dashboard itself is best run as a web app, while GPU training can run in Colab. A minimal Colab cell sequence:

```python
!pip -q install streamlit pyngrok pandas numpy pillow plotly scikit-learn
```

Upload `app.py` into the Colab runtime, then launch Streamlit. For a temporary tunnel, configure your own authenticated tunnel provider and follow its current official instructions. Do not embed tunnel tokens in notebooks or share public links containing patient data.

For GPU research, install a compatible PyTorch build and MONAI following their official install guides:
- https://pytorch.org/get-started/locally/
- https://monai.readthedocs.io/

Do not assume that a Colab GPU automatically makes a trained model available to this dashboard. A checkpoint, model definition, preprocessing configuration, and tested inference API are separate implementation steps.

## Evaluation CSV format

```csv
study_id,y_true,y_score,split,model_version
case_001,0,0.13,test,baseline-001
case_002,1,0.84,test,baseline-001
```

`y_true` must be a binary reference label (0/1) for a clearly defined target; `y_score` must be a real model score/probability between 0 and 1. These two rows are format examples only, not evidence or a benchmark dataset. Do not use example rows to claim performance. Use a sufficiently sized held-out dataset and report uncertainty intervals.

## Publication-oriented next steps
1. Define the intended endpoint: nodule detection, nodule characterization, or malignancy prediction.
2. Select and license datasets; document reference labels and inclusion/exclusion criteria.
3. Split by patient, not image slice, and reserve an external dataset.
4. Establish a baseline and predefine metrics and model-selection rules.
5. Implement 3D CT preprocessing and model training in Colab using PyTorch/MONAI.
6. Add inference only after a compatible checkpoint and end-to-end tests exist.
7. Report methods and results using CLAIM 2024, including limitations and uncertainty.

## References
1. Tejani AS, Klontzas ME, Gatti AA, et al. Checklist for Artificial Intelligence in Medical Imaging (CLAIM): 2024 Update. *Radiology: Artificial Intelligence*. 2024;6(4):e240300. https://doi.org/10.1148/ryai.240300
2. Project MONAI documentation. https://monai.readthedocs.io/
3. Armato SG III, McLennan G, Bidaut L, et al. The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI): A completed reference database of lung nodules on CT scans. *Medical Physics*. 2011;38(2):915–931. https://doi.org/10.1118/1.3528204
