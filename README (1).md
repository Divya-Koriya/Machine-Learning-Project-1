# Evaluating Stability of XAI Methods in Sentiment Analysis under Noisy Social Media Text

**A Comparative Study of LIME and SHAP for Explaining Transformer-based Models on Noisy Twitter Data**

- **Author:** Nikita Lakhani
- **Programme:** M.Sc. Big Data & AI, SRH University Heidelberg (Leipzig Campus)
- **Supervisors:** Dr. Prof. Ghanbarnejad, Fakhteh & Mrs. Shamli
- **Main file:** `Nikita_100001995.ipynb`

---

## 1. What this project does

This notebook fine-tunes a BERT sentiment classifier on Twitter data and then tests how
**stable** two popular explainability methods — **LIME** and **SHAP** — are when the input
tweets contain realistic social-media noise (typos, emojis, synonym swaps, and deleted words).

It answers three research questions:

- **RQ1** — How consistent is each method across repeated runs on the same tweet?
- **RQ2** — How do the four noise types affect explanation stability?
- **RQ3** — Which method is more stable under noise, and for which noise types?

Stability is measured with four metrics: **Jaccard Similarity, Sign Consistency,
L2 Distance, and Spearman Rank Correlation**.

---

## 2. Requirements

- **Platform:** Google Colab (recommended) with a **GPU runtime** (T4 is enough).
- **Google Drive:** used to store the dataset, trained model, results, and figures.
- **Python libraries:** all are installed inside the notebook automatically
  (`transformers`, `datasets`, `lime`, `shap`, `nlpaug`, `nltk`, `plotly`,
  `matplotlib`, `seaborn`, `scikit-learn`, `wordcloud`, `kaleido`).

No manual installation is needed — the notebook installs what it needs as it runs.

---

## 3. One-time setup

1. Open the notebook in **Google Colab**.
2. Set the runtime to GPU: **Runtime → Change runtime type → Hardware accelerator → GPU**.
3. The notebook uses this folder in your Google Drive (created automatically on first run):

```
MyDrive/thesis_xai_stability/
├── data/            # dataset cache
├── models/          # trained BERT model
├── checkpoints/     # progress files for long runs (resumable)
├── results/         # CSV result files
└── figures/
    ├── eda/         # EDA charts
    ├── noise/       # noise examples
    ├── results/     # final publication charts
    └── xai/         # LIME/SHAP figures
```

You do **not** need to create these by hand — the Session Starter cell makes them.

---

## 4. How to run — the two cells you always need first

Google Colab sessions expire after a few hours. **Every time you open or reconnect the
notebook, run these two cells first:**

| Cell | Name | What it does | When to run |
|------|------|--------------|-------------|
| **Section A** | Session Starter | Mounts Drive, defines paths, imports libraries, loads the dataset and the saved BERT model | **Every session** |
| **Section B** | Setup LIME & SHAP | Defines the `predict_proba` function and the LIME/SHAP explainers | **Every session, before Sections 5–8** |

> **Note:** Section B is only required before any LIME/SHAP work (Sections 5, 6, 7, 8).
> It is *not* needed for BERT training (Section 3) or EDA (Sections 1–2).
> Always run Section A first — it reloads the saved model so nothing is lost after a session expires.

Wait for **"Model loaded: ✅ Yes"** after Section A before continuing.

---

## 5. Notebook sections in order

| Section | Title | Run when | Approx. time |
|---------|-------|----------|--------------|
| **A** | Session Starter | Every session | ~1 min |
| **B** | Setup LIME & SHAP | Every session (before 5–8) | ~1 min |
| **1** | Exploratory Data Analysis (EDA) | Once | ~3 min |
| **2** | EDA Statistics (mean, SD, skewness) | Once | ~1 min |
| **3** | BERT Fine-Tuning | Once | ~18 min (GPU) |
| **4** | Sample Selection (200 balanced tweets) | Once | ~1 min |
| **5** | Noise Injection + LIME/SHAP (main experiment) | Once | ~6 hrs (resumable) |
| **6** | Final Analysis (4 metrics + Wilcoxon test) | Once | ~1 min |
| **7** | RQ1 Run-to-Run Consistency | Once | ~14 hrs (resumable) |
| **8** | Final Publication Charts | Anytime after Section 6 | ~2 min |

The notebook ends with a **Results Summary & Conclusions** cell.

---

## 6. Important: long runs are resumable (checkpointing)

Sections 5 and 7 take several hours. They are **checkpoint-safe**:

- They **save progress after every tweet** to a file in `checkpoints/`.
- If Colab disconnects, simply **re-run the same cell** — it skips tweets already done
  and continues from where it stopped.
- A full run may span several sessions. This is normal — just keep re-running the cell.

You can leave these running overnight with the Colab tab open.

---

## 7. Typical first-time run order

If you are running the whole project from scratch:

```
1. Section A  (Session Starter)
2. Section 1  (EDA)               — run cell 1.0 imports first, then 1.1–1.6
3. Section 2  (EDA statistics)
4. Section 3  (BERT fine-tuning)  — ~18 min, saves model to Drive
5. Section 4  (pick 200 tweets)
6. Section B  (Setup LIME & SHAP)
7. Section 5  (noise + LIME/SHAP) — long, resumable
8. Section 6  (final analysis + significance test)
9. Section 7  (RQ1 run-to-run)    — long, resumable
10. Section 8 (final charts)
```

If you come back later (new session) and the model is already trained, you only need:

```
1. Section A
2. Section B
3. whichever analysis/chart section you want
```

You can skip Section 3 forever after training once — Section A reloads the saved model.

---

## 8. Key output files (in `results/`)

| File | Contents |
|------|----------|
| `bert_results.csv` | BERT test-set performance (Macro F1 etc.) |
| `eda_statistics.csv` | mean, SD, skewness of tweet lengths |
| `stability_results_200_allmetrics_FINAL.csv` | all 4 metrics per tweet/noise pair |
| `final_4metrics_summary.csv` | overall LIME vs SHAP + Wilcoxon p-values |
| `rq1_run_variance_200.csv` | run-to-run consistency results |

Charts are saved as both interactive `.html` and 300 DPI `.png` in `figures/results/`.

---

## 9. Summary of main findings

- **BERT test performance:** Macro F1 = 69.09% (n = 12,284).
- **RQ1:** SHAP is perfectly reproducible across runs (1.000); LIME varies (0.775).
- **RQ2:** Typos cause the most explanation instability; emojis the least.
- **RQ3:** LIME is significantly more stable than SHAP on all four metrics
  (Wilcoxon signed-rank test, p < 0.05), winning three of four noise types;
  SHAP is more stable only under emoji noise.
- **Overall:** Neither method is universally better — SHAP offers reproducibility,
  LIME offers robustness to noise.

---

## 10. Notes for the reader

- The dataset used is **TweetEval Sentiment** (`cardiffnlp/tweet_eval`), loaded automatically.
- Noise is generated randomly, so exact counts can vary slightly between runs; the
  overall patterns and statistical conclusions are stable.
- All figures use a consistent design system (LIME = solid fill, SHAP = diagonal hatch)
  so they remain readable when printed in black and white.
