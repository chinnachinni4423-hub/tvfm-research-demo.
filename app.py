import io
import json
import hashlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from PIL import Image, ImageOps

st.set_page_config(page_title="TVFM | Trustworthy Vision Research Lab", page_icon="🫁", layout="wide")

# ---- Theme and lightweight session state ----
st.markdown("""
<style>
:root { color-scheme: dark; }
.stApp { background: #0b1220; color: #e7eef8; }
[data-testid="stSidebar"] { background: #101a2b; }
.block-container { padding-top: 1.5rem; max-width: 1500px; }
.tvfm-card { background: linear-gradient(135deg,#15243a,#101a2b); border: 1px solid #263a55;
  padding: 18px; border-radius: 16px; min-height: 120px; }
.tvfm-muted { color: #9fb2ca; font-size: 0.9rem; }
.tvfm-pill { display:inline-block; border:1px solid #8c6b2e; color:#f4cb75; background:#352a17;
  padding:4px 10px; border-radius:999px; font-size:0.78rem; }
div[data-testid="stMetric"] { background:#121f32; padding:14px; border:1px solid #263a55; border-radius:12px; }
div.stButton > button { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

if "studies" not in st.session_state:
    st.session_state.studies = []
if "last_upload" not in st.session_state:
    st.session_state.last_upload = None

def add_study(study):
    # Avoid accidental duplicates by file hash.
    if not any(x["sha256"] == study["sha256"] for x in st.session_state.studies):
        st.session_state.studies.append(study)

def safe_name(name):
    return "".join(c for c in name if c.isalnum() or c in "._-")[:100] or "uploaded_scan"

def metric_card(label, value, note):
    st.markdown(f'<div class="tvfm-card"><div class="tvfm-muted">{label}</div><h2 style="margin:8px 0;color:#f1f6ff">{value}</h2><div class="tvfm-muted">{note}</div></div>', unsafe_allow_html=True)

st.sidebar.markdown("## 🫁 TVFM")
st.sidebar.caption("Trustworthy Vision Foundation Model")
st.sidebar.markdown('<span class="tvfm-pill">RESEARCH MODE · MODEL NOT CONFIGURED</span>', unsafe_allow_html=True)
page = st.sidebar.radio("Navigation", [
    "Overview", "CT Study Workspace", "Evaluation Lab", "Trust Center", "Research Protocol"
])
st.sidebar.divider()
st.sidebar.caption("Prototype build 0.1.0")
st.sidebar.caption("No clinical diagnosis is produced.")

if page == "Overview":
    st.title("Trustworthy Vision Research Lab")
    st.markdown("A research dashboard for chest CT study management, model evaluation, and trustworthy-AI reporting.")
    st.warning("**Model status: unavailable.** No trained TVFM checkpoint is connected. The dashboard will not fabricate predictions or diagnostic probabilities.")
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("Studies in this session", len(st.session_state.studies))
    with c2: st.metric("Trained model", "Unavailable")
    with c3: st.metric("Test AUROC", "—", help="Displayed only after valid held-out predictions are uploaded.")
    with c4: st.metric("Calibration", "—", help="No evaluation results have been loaded.")
    st.subheader("Research workflow")
    cols = st.columns(3)
    items = [
        ("01 · Import", "Upload a supported image for visual inspection. Files are processed in this session only."),
        ("02 · Evaluate", "Import a CSV of held-out labels and model scores to calculate classification metrics."),
        ("03 · Document", "Review trustworthiness checks and export a transparent research-session report.")
    ]
    for col,(title,desc) in zip(cols,items):
        with col:
            st.markdown(f'<div class="tvfm-card"><h3>{title}</h3><p class="tvfm-muted">{desc}</p></div>', unsafe_allow_html=True)
    st.subheader("Recent studies")
    if st.session_state.studies:
        view = pd.DataFrame([{k:v for k,v in s.items() if k not in ("bytes","sha256")} for s in st.session_state.studies])
        st.dataframe(view, use_container_width=True, hide_index=True)
    else:
        st.info("No studies uploaded yet. Open **CT Study Workspace** to begin.")
    st.caption("Research prototype only. Do not upload identifiable patient data to a shared or unapproved environment.")

elif page == "CT Study Workspace":
    st.title("CT Study Workspace")
    st.markdown("Upload an image file for visual inspection. This prototype currently supports common 2D image formats; it does **not** yet parse full DICOM series or NIfTI volumes.")
    st.info("Supported formats: PNG, JPG/JPEG, TIFF, BMP. No model inference is available in this prototype.")
    with st.form("study_form"):
        a,b = st.columns([1,2])
        with a:
            study_id = st.text_input("Pseudonymous study ID", placeholder="e.g. TVFM-001")
            modality = st.selectbox("Image type", ["Chest CT slice", "Other / unknown"])
        with b:
            uploaded = st.file_uploader("Choose an image", type=["png","jpg","jpeg","tif","tiff","bmp"])
        consent = st.checkbox("I confirm this file is authorized for research use and does not contain directly identifying information.")
        submit = st.form_submit_button("Load study", type="primary")
    if submit:
        if not uploaded:
            st.error("Choose an image file first.")
        elif not study_id.strip():
            st.error("Enter a pseudonymous study ID.")
        elif not consent:
            st.error("Confirm authorized research use before loading the image.")
        else:
            raw = uploaded.getvalue()
            digest = hashlib.sha256(raw).hexdigest()
            try:
                image = Image.open(io.BytesIO(raw))
                image.verify()
                record = {
                    "study_id": study_id.strip(),
                    "file_name": safe_name(uploaded.name),
                    "modality": modality,
                    "uploaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "sha256": digest,
                    "size_bytes": len(raw),
                    "status": "Image loaded; model unavailable",
                    "bytes": raw
                }
                add_study(record)
                st.session_state.last_upload = digest
                st.success("Image loaded for visual inspection. No diagnostic inference was run.")
            except Exception as e:
                st.error(f"Could not read this image: {e}")
    if st.session_state.studies:
        st.subheader("Loaded study viewer")
        options = {f'{s["study_id"]} · {s["file_name"]}': s for s in st.session_state.studies}
        chosen_label = st.selectbox("Select a loaded image", list(options.keys()))
        study = options[chosen_label]
        try:
            im = Image.open(io.BytesIO(study["bytes"]))
            im = ImageOps.exif_transpose(im)
            left,right = st.columns([1.5,1])
            with left:
                st.image(im, caption=f'{study["study_id"]} · {study["file_name"]}', use_container_width=True)
            with right:
                st.markdown("### Study metadata")
                st.write("**Study ID:**", study["study_id"])
                st.write("**Image type:**", study["modality"])
                st.write("**File size:**", f'{study["size_bytes"]:,} bytes')
                st.write("**SHA-256:**")
                st.code(study["sha256"])
                st.write("**Inference status:** Model unavailable")
                st.warning("No malignancy score, nodule localization, or heatmap is generated because no trained model is connected.")
        except Exception:
            st.error("The stored image could not be displayed.")
        metadata = pd.DataFrame([{k:v for k,v in s.items() if k not in ("bytes",)} for s in st.session_state.studies])
        st.download_button("Export study manifest (CSV)", metadata.to_csv(index=False).encode("utf-8"), "tvfm_study_manifest.csv", "text/csv")
        if st.button("Clear studies from this session"):
            st.session_state.studies = []
            st.session_state.last_upload = None
            st.rerun()

elif page == "Evaluation Lab":
    st.title("Evaluation Lab")
    st.markdown("Calculate metrics from **real, held-out model predictions**. Do not upload training predictions as test results.")
    st.info("Expected CSV columns: `y_true` (0/1 reference labels), `y_score` (model probability in [0,1]). Optional: `study_id`, `split`, `model_version`.")
    eval_file = st.file_uploader("Upload evaluation predictions CSV", type=["csv"], key="eval_csv")
    if eval_file:
        try:
            df = pd.read_csv(eval_file)
            needed = {"y_true","y_score"}
            if not needed.issubset(df.columns):
                st.error("Missing required columns: " + ", ".join(sorted(needed - set(df.columns))))
            else:
                df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")
                df["y_score"] = pd.to_numeric(df["y_score"], errors="coerce")
                df = df.dropna(subset=["y_true","y_score"])
                valid = df["y_true"].isin([0,1]) & df["y_score"].between(0,1)
                df = df[valid]
                if len(df) == 0:
                    st.error("No valid rows. y_true must be 0/1 and y_score must be between 0 and 1.")
                else:
                    st.write(f"Valid rows: **{len(df)}**")
                    threshold = st.slider("Classification threshold", 0.05, 0.95, 0.50, 0.05)
                    y = df["y_true"].astype(int).to_numpy()
                    score = df["y_score"].to_numpy()
                    pred = (score >= threshold).astype(int)
                    tp = int(((y==1)&(pred==1)).sum()); tn = int(((y==0)&(pred==0)).sum())
                    fp = int(((y==0)&(pred==1)).sum()); fn = int(((y==1)&(pred==0)).sum())
                    sens = tp/(tp+fn) if tp+fn else np.nan
                    spec = tn/(tn+fp) if tn+fp else np.nan
                    prec = tp/(tp+fp) if tp+fp else np.nan
                    f1 = 2*prec*sens/(prec+sens) if np.isfinite(prec) and np.isfinite(sens) and prec+sens else np.nan
                    brier = float(np.mean((score-y)**2))
                    try:
                        from sklearn.metrics import roc_auc_score
                        auc = roc_auc_score(y,score) if len(np.unique(y)) == 2 else np.nan
                    except Exception:
                        auc = np.nan
                    m1,m2,m3,m4,m5 = st.columns(5)
                    m1.metric("AUROC", f"{auc:.3f}" if np.isfinite(auc) else "N/A")
                    m2.metric("Sensitivity", f"{sens:.3f}" if np.isfinite(sens) else "N/A")
                    m3.metric("Specificity", f"{spec:.3f}" if np.isfinite(spec) else "N/A")
                    m4.metric("F1 score", f"{f1:.3f}" if np.isfinite(f1) else "N/A")
                    m5.metric("Brier score", f"{brier:.3f}")
                    st.caption("These metrics describe only the uploaded rows. They are not evidence of clinical validity. AUROC is unavailable if the CSV contains only one class.")
                    cm = pd.DataFrame([[tn,fp],[fn,tp]], index=["Actual 0","Actual 1"], columns=["Predicted 0","Predicted 1"])
                    left,right = st.columns(2)
                    with left:
                        st.subheader("Confusion matrix")
                        st.dataframe(cm, use_container_width=True)
                    with right:
                        st.subheader("Score distribution")
                        plot_df = df.copy()
                        plot_df["reference_label"] = plot_df["y_true"].astype(int).astype(str)
                        fig = px.histogram(plot_df, x="y_score", color="reference_label", nbins=20,
                                           labels={"y_score":"Model score","reference_label":"Reference label"},
                                           title="Scores by reference label")
                        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                        st.plotly_chart(fig, use_container_width=True)
                    report = {
                        "generated_utc": datetime.now(timezone.utc).isoformat(),
                        "source_file": eval_file.name,
                        "n_valid_rows": int(len(df)),
                        "threshold": threshold,
                        "auroc": None if not np.isfinite(auc) else float(auc),
                        "sensitivity": None if not np.isfinite(sens) else float(sens),
                        "specificity": None if not np.isfinite(spec) else float(spec),
                        "precision": None if not np.isfinite(prec) else float(prec),
                        "f1": None if not np.isfinite(f1) else float(f1),
                        "brier_score": brier,
                        "confusion_matrix": {"tn":tn,"fp":fp,"fn":fn,"tp":tp},
                        "note": "Metrics are derived from user-supplied predictions; split integrity and labels were not independently verified."
                    }
                    st.download_button("Download evaluation report (JSON)", json.dumps(report,indent=2).encode(), "tvfm_evaluation_report.json", "application/json")
                    st.download_button("Download valid rows (CSV)", df.to_csv(index=False).encode(), "tvfm_evaluation_rows.csv", "text/csv")
        except Exception as e:
            st.error(f"Could not read the CSV: {e}")
    else:
        st.markdown("### Evaluation status")
        st.metric("Available measured metrics", "0")
        st.caption("Metrics remain blank until a real prediction CSV is supplied.")

elif page == "Trust Center":
    st.title("Trust Center")
    st.markdown("A checklist for responsible research; checks are not automatically passed just because they appear here.")
    checks = [
        ("Patient-level split verified", "Confirm that no patient appears across training, internal testing, and external testing."),
        ("External testing performed", "Use an independent dataset from a separate source or institution where feasible."),
        ("Reference labels documented", "Describe who created labels, annotation protocol, adjudication, and limitations."),
        ("Calibration assessed", "Report calibration plots and a proper scoring rule such as the Brier score."),
        ("Uncertainty and OOD behavior assessed", "Define what happens when an input is unfamiliar or quality is inadequate."),
        ("Subgroup performance reviewed", "Report supported subgroup analyses and uncertainty intervals."),
        ("Reproducibility recorded", "Save code version, model checkpoint hash, configuration, seeds, and data manifest."),
        ("Privacy and licensing reviewed", "Confirm permission to use each dataset and avoid identifiable patient data.")
    ]
    st.warning("Model unavailable. Trustworthiness cannot be inferred from dashboard appearance or an explanation map alone.")
    statuses = {}
    for title, desc in checks:
        statuses[title] = st.checkbox(f"{title} — {desc}", key="trust_"+title)
    if st.button("Export trust checklist"):
        checklist = {"exported_utc": datetime.now(timezone.utc).isoformat(),
                     "items": [{"item":k,"marked_complete_by_user":v} for k,v in statuses.items()],
                     "warning":"User-entered checklist; completion has not been independently verified."}
        st.download_button("Download checklist JSON", json.dumps(checklist,indent=2).encode(), "tvfm_trust_checklist.json", "application/json")

elif page == "Research Protocol":
    st.title("Research Protocol & Publication Readiness")
    st.markdown("### Proposed primary research question")
    st.write("Does a trust-aware CT representation-learning pipeline improve the reliability, calibration, and external generalization of pulmonary nodule analysis compared with an appropriate baseline?")
    st.markdown("### Planned protocol")
    protocol = [
        "Define the target task and reference standard before training.",
        "Document dataset inclusion/exclusion criteria and licensing.",
        "Split at patient level; keep external data untouched until final evaluation.",
        "Specify preprocessing, augmentation, architecture, optimizer, seeds, and model-selection rules.",
        "Compare against a clearly defined baseline and perform ablation studies.",
        "Report discrimination, threshold-based performance, calibration, uncertainty, and confidence intervals.",
        "Document failed experiments and deviations; do not report planned results as observed results.",
        "Complete the CLAIM 2024 checklist before manuscript submission."
    ]
    for item in protocol:
        st.markdown(f"- {item}")
    st.markdown("### Suggested core references")
    st.markdown("""
1. Tejani AS, Klontzas ME, Gatti AA, et al. **Checklist for Artificial Intelligence in Medical Imaging (CLAIM): 2024 Update.** *Radiology: Artificial Intelligence*. 2024;6(4):e240300. [https://doi.org/10.1148/ryai.240300](https://doi.org/10.1148/ryai.240300)
2. Project MONAI. **MONAI documentation and medical-imaging AI toolkit.** [https://monai.readthedocs.io/](https://monai.readthedocs.io/)
3. Armato SG III, McLennan G, Bidaut L, et al. **The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI): A completed reference database of lung nodules on CT scans.** *Medical Physics*. 2011;38(2):915–931. [https://doi.org/10.1118/1.3528204](https://doi.org/10.1118/1.3528204)
""")
    st.markdown("### Export session report")
    report = {
        "project":"TVFM",
        "build":"0.1.0",
        "generated_utc":datetime.now(timezone.utc).isoformat(),
        "studies_loaded_this_session":len(st.session_state.studies),
        "model_status":"unavailable",
        "experimental_results":"not supplied",
        "research_question":"Does trust-aware CT representation learning improve reliability, calibration, and external generalization?",
        "limitations":["No trained model checkpoint is connected","Image workspace currently supports 2D common image formats only","No clinical diagnosis is generated"],
        "references":[
            {"citation":"Tejani et al. CLAIM 2024 Update","doi":"10.1148/ryai.240300"},
            {"citation":"Project MONAI documentation","url":"https://monai.readthedocs.io/"},
            {"citation":"Armato et al. LIDC-IDRI database","doi":"10.1118/1.3528204"}
        ]
    }
    st.download_button("Download research session report (JSON)", json.dumps(report,indent=2).encode(), "tvfm_research_report.json", "application/json")

st.divider()
st.caption("TVFM is an experimental research prototype, not a medical device. It must not be used to diagnose, rule out, or treat lung cancer.")
