# ⚙️ Physics-Informed Machine Learning (PiML) for Bearing Fault Diagnosis

🚀 **TLS-DMD + PINN Features + Random Forest on CWRU Dataset**
📊 Achieving **98.89% Accuracy** with interpretable, CPU-efficient model

---

## 📌 Overview

This project presents a **Physics-Informed Machine Learning (PiML)** framework for **bearing fault diagnosis**, integrating:

* 🔹 **TLS-DMD (Total Least Squares Dynamic Mode Decomposition)**
* 🔹 **PINN-based physics constraints**
* 🔹 **Statistical signal features**
* 🔹 **Random Forest classifier**

Unlike traditional ML or deep learning approaches, this method:

✔ Uses **domain physics (fault frequencies + decay)**
✔ Requires **no GPU**
✔ Is **interpretable + lightweight**
✔ Achieves near state-of-the-art performance

---

## 🎯 Key Results

| Metric                     | Value                |
| -------------------------- | -------------------- |
| ✅ Accuracy (GroupKFold CV) | **98.89% ± 0.57%**   |
| 📈 AUC                     | ≥ 0.998              |
| ⚡ Speed                    | **7.1 ms / segment** |
| 🧠 Features                | 17-D PiML Vector     |
| 💻 Hardware                | CPU-only             |

---

## 🧠 Problem Statement

* ⚙️ **40–50% of motor failures** are due to bearing faults
* 💸 ~$50B annual industrial loss
* ❌ Traditional ML → ignores physics
* ❌ Deep learning → needs large data + GPU

👉 Solution: **Physics-Informed ML (PiML)**

---

## 🏗️ Proposed Architecture

```
Raw Signal → MRDMD → Hankel Embedding → TLS-DMD → Physics Features → 17-D Vector → Random Forest
```

---

## 🔬 Feature Engineering (17-D PiML Vector)

### 📊 1. Statistical Features (5)

* RMS
* Kurtosis
* Skewness
* Peak-to-Peak
* Crest Factor

### 🔷 2. TLS-DMD Features (8)

* Real & Imaginary parts of top 4 eigenvalues

### ⚙️ 3. Physics-Informed Features (4)

* BPFI Energy (Inner race fault)
* BPFO Energy (Outer race fault)
* BSF Energy (Ball fault)
* PINN-based decay rate (α̂)

---

## 📂 Dataset

* 📁 **CWRU Bearing Dataset**
* ⚙️ Sampling: **12 kHz**
* 🔁 Segment size: **2048 samples**
* 🔄 Overlap: **50%**
* 🧪 Classes:

  * Normal
  * Inner Race Fault
  * Ball Fault
  * Outer Race Fault

---

## ⚙️ Pipeline Implementation

Main pipeline file:
👉 

### Steps:

1. Load `.mat` vibration signals
2. Segment signals
3. Extract:

   * Statistical features
   * TLS-DMD features
   * Physics features
4. Normalize features
5. Train models
6. Evaluate using:

   * Accuracy
   * F1-score
   * ROC
   * Confusion Matrix

---

## 🤖 Models Compared

| Model                 | Accuracy   |
| --------------------- | ---------- |
| 🥇 PiML-RF (Proposed) | **~98.9%** |
| Gradient Boosting     | ~97–98%    |
| SVM (RBF)             | ~96–97%    |
| MLP                   | ~95–97%    |
| KNN                   | ~94–96%    |
| Logistic Regression   | ~85–90%    |

---

## 📊 Cross Validation

* 🔁 **5-Fold GroupKFold**
* Prevents data leakage
* Ensures generalization

---

## 📉 Ablation Study

| Feature Set      | Accuracy   |
| ---------------- | ---------- |
| Statistical only | 91.3%      |
| + TLS-DMD        | 95.7%      |
| + PiML           | 94.2%      |
| Full (17-D)      | **98.89%** |

👉 Physics + DMD = HUGE improvement

---

## ⚡ Computational Efficiency

* 🚀 24× real-time speed
* 🧠 < 5 MB memory
* 💻 Runs on:

  * CPU
  * Raspberry Pi
  * Edge devices

---

## 📊 Visual Results

Project report with plots:
👉 

Includes:

* Confusion Matrix
* ROC Curves
* Feature Importance
* Signal Analysis
* DMD Eigenvalue plots

---

## 📁 Project Structure

```
PiML-Fault-Diagnosis/
│
├── dataset/                # CWRU dataset (.mat files)
├── results/                # Generated plots
├── main_pipeline.py        # Main implementation
├── Project_Report.html     # Full report
├── presentation.pptx       # Slides
└── README.md
```

---

## ▶️ How to Run

### 1️⃣ Install dependencies

```bash
pip install numpy scipy matplotlib seaborn scikit-learn pandas
```

### 2️⃣ Set dataset path

Edit:

```python
DATA_DIR = 'path_to_dataset'
```

### 3️⃣ Run pipeline

```bash
python main_pipeline.py
```

---

## 🧪 Output

* Accuracy & metrics printed
* Plots saved in `/results`
* Confusion matrix
* ROC curves

---

## 🔍 Key Insights

* 🔥 Physics features (BPFI, decay) are **highly important**
* 🔷 TLS-DMD captures **dynamic behavior**
* ⚙️ Model is **interpretable + robust**
* 📉 Deep learning is **not always necessary**

---

## 🚀 Future Work

* Variable speed conditions
* Multi-dataset validation (IMS, Paderborn)
* Online streaming (IoT)
* Federated PiML
* Compound fault detection

---

## 📜 Citation

If you use this work, please cite:

```
Physics-Informed Machine Learning for Bearing Fault Diagnosis
Using TLS-DMD & PINN Features (2026)
```

---

## ❤️ Acknowledgements

* Case Western Reserve University (CWRU Dataset)
* Signal processing & DMD research community

---

## ⭐ GitHub Tips

If you like this project:

⭐ Star the repo
🍴 Fork it
📢 Share it

---

> “Physics + ML = Better Generalization, Better Engineering.” ⚙️🔥
