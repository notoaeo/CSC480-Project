"""
CSC 480 Final Presentation: new experiments since mid-project.
1. Ablation study (stop words, sublinear_tf, min_df, C)
2. Error analysis by length bucket and negation
3. Skyline context table
"""
import re, json, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

RNG = 42
np.random.seed(RNG)

# ---------- Load (same as mid-project) ----------
df = pd.read_csv("imdb_repo/IMDB-Dataset.csv")
def clean(text):
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

df["review"] = df["review"].astype(str).map(clean)
df["label"] = (df["sentiment"] == "positive").astype(int)
texts = np.array(df["review"].tolist(), dtype=object)
labels = df["label"].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.5, stratify=labels, random_state=RNG
)

# ============================================================
# 1. ABLATION STUDY
# ============================================================
print("=" * 60)
print("ABLATION STUDY")
print("=" * 60)

# Baseline config: TF-IDF uni+bi, sublinear_tf=True, no stop words, min_df=5, C=1.0
ablation_results = []

def run_ablation(name, vec_kwargs, model_cls, model_kwargs):
    vec = TfidfVectorizer(**vec_kwargs)
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)
    m = model_cls(**model_kwargs)
    m.fit(Xtr, y_train)
    pred = m.predict(Xte)
    acc = accuracy_score(y_test, pred)
    p, r, f1, _ = precision_recall_fscore_support(y_test, pred, average="binary")
    vocab = len(vec.vocabulary_)
    ablation_results.append({
        "ablation": name, "acc": acc, "f1": f1, "vocab": vocab
    })
    print(f"  {name:45s}  acc={acc:.4f}  f1={f1:.4f}  vocab={vocab}")
    return pred

base_vec = dict(min_df=5, max_df=0.9, ngram_range=(1,2), sublinear_tf=True)
base_svm = dict(C=1.0)

# Baseline
print("\n--- Baseline (best mid-project config) ---")
best_pred = run_ablation("Baseline: TF-IDF uni+bi, sublinear, C=1",
                         base_vec, LinearSVC, base_svm)

# A: Toggle stop words
print("\n--- Stop words ---")
run_ablation("+ English stop words removed",
             {**base_vec, "stop_words": "english"}, LinearSVC, base_svm)

# B: Toggle sublinear_tf
print("\n--- Sublinear TF ---")
run_ablation("sublinear_tf=False (raw counts)",
             {**base_vec, "sublinear_tf": False}, LinearSVC, base_svm)

# C: Vary min_df
print("\n--- min_df ---")
for mdf in [1, 2, 10, 25, 50]:
    run_ablation(f"min_df={mdf}",
                 {**base_vec, "min_df": mdf}, LinearSVC, base_svm)

# D: Vary C for SVM
print("\n--- C (SVM regularization) ---")
for c in [0.01, 0.1, 0.5, 2.0, 5.0]:
    run_ablation(f"C={c}",
                 base_vec, LinearSVC, {"C": c})

# E: Unigrams only (to isolate bigram contribution)
print("\n--- Unigrams only ---")
run_ablation("Unigrams only (no bigrams)",
             {**base_vec, "ngram_range": (1,1)}, LinearSVC, base_svm)

# F: Same ablations for LogReg to compare
print("\n--- LogReg baseline ---")
lr_pred = run_ablation("LogReg baseline: TF-IDF uni+bi, sublinear, C=1",
                       base_vec, LogisticRegression,
                       {"max_iter": 1000, "C": 1.0, "solver": "liblinear"})

abl_df = pd.DataFrame(ablation_results)
abl_df.to_csv("ablation_results.csv", index=False)
print("\nSaved ablation_results.csv")
print(abl_df.to_string(index=False))

# ============================================================
# 2. DEEPER ERROR ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("ERROR ANALYSIS BY LENGTH BUCKET")
print("=" * 60)

word_lens = np.array([len(r.split()) for r in X_test])
correct = (best_pred == y_test)

# Length buckets
buckets = [
    ("Short (< 100 words)", word_lens < 100),
    ("Medium (100-250)",    (word_lens >= 100) & (word_lens < 250)),
    ("Long (250-500)",      (word_lens >= 250) & (word_lens < 500)),
    ("Very long (500+)",    word_lens >= 500),
]

bucket_results = []
for bname, mask in buckets:
    n = mask.sum()
    if n == 0:
        continue
    bacc = correct[mask].mean()
    bucket_results.append({"bucket": bname, "n": int(n), "accuracy": bacc})
    print(f"  {bname:25s}  n={n:5d}  acc={bacc:.4f}")

bucket_df = pd.DataFrame(bucket_results)
bucket_df.to_csv("error_by_length.csv", index=False)

# ============================================================
# NEGATION ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("NEGATION ANALYSIS")
print("=" * 60)

neg_words = {"not", "no", "never", "neither", "nobody", "nothing",
             "nowhere", "nor", "cannot", "can't", "don't", "doesn't",
             "didn't", "won't", "wouldn't", "shouldn't", "couldn't",
             "isn't", "aren't", "wasn't", "weren't", "hasn't", "haven't"}

has_negation = np.array([
    any(w in neg_words for w in r.split()) for r in X_test
])

neg_acc = correct[has_negation].mean()
no_neg_acc = correct[~has_negation].mean()
print(f"  Reviews WITH negation:    n={has_negation.sum():5d}  acc={neg_acc:.4f}")
print(f"  Reviews WITHOUT negation: n={(~has_negation).sum():5d}  acc={no_neg_acc:.4f}")

# Error type breakdown for misclassified reviews
errors_idx = np.where(~correct)[0]
err_has_neg = has_negation[errors_idx].sum()
err_no_neg = (~has_negation[errors_idx]).sum()
print(f"\n  Of {len(errors_idx)} errors:")
print(f"    {err_has_neg} ({100*err_has_neg/len(errors_idx):.1f}%) contain negation words")
print(f"    {err_no_neg} ({100*err_no_neg/len(errors_idx):.1f}%) do not")

# ============================================================
# 3. PLOTS
# ============================================================

# Plot A: Ablation bar chart (key ablations only)
print("\n" + "=" * 60)
print("GENERATING PLOTS")
print("=" * 60)

key_ablations = abl_df[abl_df["ablation"].isin([
    "Baseline: TF-IDF uni+bi, sublinear, C=1",
    "+ English stop words removed",
    "sublinear_tf=False (raw counts)",
    "Unigrams only (no bigrams)",
    "min_df=1",
    "min_df=50",
    "C=0.1",
    "C=5.0",
])].copy()
key_ablations["short"] = [
    "Baseline",
    "Remove stop words",
    "No sublinear TF",
    "Unigrams only",
    "min_df=1",
    "min_df=50",
    "C=0.1",
    "C=5.0",
]

fig, ax = plt.subplots(figsize=(8, 4))
colors_bar = ["#7A0019" if "Baseline" in n else "#6c8ebf" for n in key_ablations["short"]]
bars = ax.barh(range(len(key_ablations)), key_ablations["acc"].values,
               color=colors_bar, edgecolor="black", height=0.65)
ax.set_yticks(range(len(key_ablations)))
ax.set_yticklabels(key_ablations["short"].values, fontsize=10)
ax.set_xlim(0.87, 0.92)
ax.set_xlabel("Test accuracy")
ax.set_title("Ablation study: Linear SVM + TF-IDF (uni+bi)")
ax.invert_yaxis()
# Add value labels
for i, (v, name) in enumerate(zip(key_ablations["acc"].values, key_ablations["short"])):
    ax.text(v + 0.001, i, f"{v:.4f}", va="center", fontsize=9,
            fontweight="bold" if "Baseline" in name else "normal")
# Baseline reference line
baseline_acc = key_ablations[key_ablations["short"] == "Baseline"]["acc"].values[0]
ax.axvline(baseline_acc, color="#7A0019", linestyle="--", alpha=0.5, linewidth=1)
plt.tight_layout()
plt.savefig("plot_ablation.png", dpi=180)
plt.close()
print("Saved plot_ablation.png")

# Plot B: Accuracy by length bucket
fig, ax = plt.subplots(figsize=(6, 3.5))
bnames = bucket_df["bucket"].values
baccs = bucket_df["accuracy"].values
bcolors = ["#82b366", "#6c8ebf", "#d79b00", "#c0504d"]
bars = ax.bar(range(len(bnames)), baccs, color=bcolors[:len(bnames)],
              edgecolor="black", width=0.6)
ax.set_xticks(range(len(bnames)))
ax.set_xticklabels(bnames, fontsize=9)
ax.set_ylim(0.85, 0.95)
ax.set_ylabel("Test accuracy")
ax.set_title("Accuracy by review length (Linear SVM + TF-IDF uni+bi)")
for i, v in enumerate(baccs):
    ax.text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
plt.tight_layout()
plt.savefig("plot_length_buckets.png", dpi=180)
plt.close()
print("Saved plot_length_buckets.png")

# Plot C: Skyline context
fig, ax = plt.subplots(figsize=(7, 3.2))
skyline_data = [
    ("Maas et al. 2011\n(word vectors)", 0.889, "#aaaaaa"),
    ("This project\n(Linear SVM + TF-IDF bi)", 0.9112, "#7A0019"),
    ("Fine-tuned BERT\n(Devlin et al. 2019)", 0.935, "#6c8ebf"),
    ("RoBERTa + ensemble\n(CFA, 2025)", 0.971, "#4A0010"),
]
names = [d[0] for d in skyline_data]
accs = [d[1] for d in skyline_data]
cols = [d[2] for d in skyline_data]
bars = ax.barh(range(len(names)), accs, color=cols, edgecolor="black", height=0.55)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=10)
ax.set_xlim(0.85, 1.0)
ax.set_xlabel("Test accuracy on IMDb 25k")
ax.set_title("Where does this project sit?")
ax.invert_yaxis()
for i, v in enumerate(accs):
    ax.text(v + 0.003, i, f"{v*100:.1f}%", va="center", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig("plot_skyline.png", dpi=180)
plt.close()
print("Saved plot_skyline.png")

# Save all analysis
analysis = {
    "ablations": ablation_results,
    "length_buckets": bucket_results,
    "negation": {
        "with_negation": {"n": int(has_negation.sum()), "acc": float(neg_acc)},
        "without_negation": {"n": int((~has_negation).sum()), "acc": float(no_neg_acc)},
        "errors_with_negation_pct": float(100 * err_has_neg / len(errors_idx)),
    },
}
with open("final_analysis.json", "w") as f:
    json.dump(analysis, f, indent=2)
print("\nSaved final_analysis.json")
print("\nDONE.")
