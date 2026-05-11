"""
CSC 480 Project: Sentiment Analysis of Movie Reviews
Mid-project experiments: NB / LogReg / SVM x BoW / TF-IDF on IMDb 50k.
"""
import re
import time
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report)

RNG = 42
np.random.seed(RNG)

# ---------- Load ----------
print("Loading IMDb dataset...")
df = pd.read_csv("imdb_repo/IMDB-Dataset.csv")
print(f"  total reviews: {len(df)}")
print(f"  label balance:\n{df['sentiment'].value_counts()}")

# Light cleaning: remove HTML <br /> tags and lowercase
def clean(text):
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

df["review"] = df["review"].astype(str).map(clean)
df["label"] = (df["sentiment"] == "positive").astype(int)

texts = np.array(df["review"].tolist(), dtype=object)
labels = df["label"].to_numpy()

# Match the Maas et al. split: 25k train / 25k test, stratified
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.5, stratify=labels, random_state=RNG
)
print(f"  train: {len(X_train)}  test: {len(X_test)}")

# ---------- Length stats for slide ----------
lengths = df["review"].str.split().map(len)
print(f"  word-length: mean={lengths.mean():.0f}, median={lengths.median():.0f}, "
      f"p95={lengths.quantile(0.95):.0f}")

# ---------- Vectorizers ----------
vectorizers = {
    "BoW (uni)":     CountVectorizer(min_df=5, max_df=0.9, ngram_range=(1, 1)),
    "TF-IDF (uni)":  TfidfVectorizer(min_df=5, max_df=0.9, ngram_range=(1, 1),
                                     sublinear_tf=True),
    "TF-IDF (uni+bi)": TfidfVectorizer(min_df=5, max_df=0.9, ngram_range=(1, 2),
                                       sublinear_tf=True),
}

# ---------- Models ----------
def make_models():
    return {
        "Naive Bayes":         MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0,
                                                  solver="liblinear"),
        "Linear SVM":          LinearSVC(C=1.0),
    }

# ---------- Run ----------
results = []
fitted = {}  # cache for error analysis

for vname, vec in vectorizers.items():
    print(f"\n=== Vectorizer: {vname} ===")
    t0 = time.time()
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)
    vocab = len(vec.vocabulary_)
    print(f"  vocab size: {vocab}  ({time.time()-t0:.1f}s)")

    for mname, model in make_models().items():
        t0 = time.time()
        model.fit(Xtr, y_train)
        train_t = time.time() - t0
        pred = model.predict(Xte)
        acc = accuracy_score(y_test, pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, pred, average="binary"
        )
        # 5-fold CV on training set
        cv = cross_val_score(model, Xtr, y_train, cv=5, scoring="accuracy",
                             n_jobs=-1)
        results.append({
            "vectorizer": vname, "model": mname, "vocab": vocab,
            "test_acc": acc, "precision": prec, "recall": rec, "f1": f1,
            "cv_mean": cv.mean(), "cv_std": cv.std(), "train_s": train_t,
        })
        print(f"  {mname:22s}  acc={acc:.4f}  f1={f1:.4f}  "
              f"cv={cv.mean():.4f}±{cv.std():.4f}  ({train_t:.1f}s)")
        fitted[(vname, mname)] = (model, vec, pred)

res_df = pd.DataFrame(results)
res_df.to_csv("results.csv", index=False)
print("\nSaved results.csv")
print(res_df.to_string(index=False))

# ---------- Best model: full report ----------
best = res_df.sort_values("test_acc", ascending=False).iloc[0]
bkey = (best["vectorizer"], best["model"])
bmodel, bvec, bpred = fitted[bkey]
print(f"\nBest: {bkey}  acc={best['test_acc']:.4f}")
print(classification_report(y_test, bpred, target_names=["neg", "pos"]))
cm = confusion_matrix(y_test, bpred)
print("Confusion matrix:\n", cm)

# ---------- Error analysis: misclassified examples ----------
errors_idx = np.where(bpred != y_test)[0]
print(f"\nTotal errors: {len(errors_idx)} / {len(y_test)}")

# Length distribution: errors vs correct
err_lens = np.array([len(X_test[i].split()) for i in errors_idx])
cor_lens = np.array([len(X_test[i].split()) for i in range(len(y_test))
                     if i not in set(errors_idx)])
print(f"  err mean len: {err_lens.mean():.0f}  correct mean len: {cor_lens.mean():.0f}")

# Save a few sample errors for the slides
sample_errors = []
for i in errors_idx[:6]:
    sample_errors.append({
        "true": "pos" if y_test[i] == 1 else "neg",
        "pred": "pos" if bpred[i] == 1 else "neg",
        "text": X_test[i][:280] + ("..." if len(X_test[i]) > 280 else ""),
    })
with open("sample_errors.json", "w") as f:
    json.dump(sample_errors, f, indent=2)

# ---------- Most informative features (LogReg w/ TF-IDF uni+bi) ----------
key = ("TF-IDF (uni+bi)", "Logistic Regression")
if key in fitted:
    lr, lvec, _ = fitted[key]
    feats = np.array(lvec.get_feature_names_out())
    coefs = lr.coef_[0]
    top_pos = feats[np.argsort(coefs)[-15:][::-1]]
    top_neg = feats[np.argsort(coefs)[:15]]
    print("\nTop POSITIVE features (LogReg, TF-IDF uni+bi):")
    print(", ".join(top_pos))
    print("Top NEGATIVE features:")
    print(", ".join(top_neg))
    with open("top_features.json", "w") as f:
        json.dump({"pos": top_pos.tolist(), "neg": top_neg.tolist()}, f, indent=2)

# ---------- Plots ----------
# 1. Accuracy bar chart
fig, ax = plt.subplots(figsize=(8, 4.2))
pivot = res_df.pivot(index="model", columns="vectorizer", values="test_acc")
pivot = pivot[["BoW (uni)", "TF-IDF (uni)", "TF-IDF (uni+bi)"]]
pivot = pivot.loc[["Naive Bayes", "Logistic Regression", "Linear SVM"]]
pivot.plot(kind="bar", ax=ax, width=0.78,
           color=["#6c8ebf", "#82b366", "#d79b00"], edgecolor="black")
ax.set_ylabel("Test accuracy")
ax.set_ylim(0.80, 0.92)
ax.set_xlabel("")
ax.set_title("Test accuracy by classifier and feature representation")
ax.legend(title="Features", fontsize=8, loc="lower right")
plt.xticks(rotation=0)
for c in ax.containers:
    ax.bar_label(c, fmt="%.3f", fontsize=7, padding=2)
plt.tight_layout()
plt.savefig("plot_accuracy.png", dpi=180)
plt.close()
print("\nSaved plot_accuracy.png")

# 2. Confusion matrix for best model
fig, ax = plt.subplots(figsize=(4.4, 4))
im = ax.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                fontsize=14, color="black" if cm[i, j] < cm.max()/2 else "white")
ax.set_xticks([0, 1]); ax.set_xticklabels(["neg", "pos"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["neg", "pos"])
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title(f"Confusion matrix\n{best['model']} + {best['vectorizer']}")
plt.tight_layout()
plt.savefig("plot_confusion.png", dpi=180)
plt.close()
print("Saved plot_confusion.png")

# 3. Top features visualization
if key in fitted:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    pos_coefs = sorted(coefs)[-15:]
    neg_coefs = sorted(coefs)[:15]
    axes[0].barh(range(15), pos_coefs, color="#82b366", edgecolor="black")
    axes[0].set_yticks(range(15))
    axes[0].set_yticklabels(top_pos[::-1], fontsize=8)
    axes[0].set_title("Top POSITIVE features")
    axes[0].invert_yaxis()
    axes[1].barh(range(15), neg_coefs, color="#c0504d", edgecolor="black")
    axes[1].set_yticks(range(15))
    axes[1].set_yticklabels(top_neg, fontsize=8)
    axes[1].set_title("Top NEGATIVE features")
    plt.suptitle("Logistic Regression coefficients (TF-IDF uni+bi)", fontsize=11)
    plt.tight_layout()
    plt.savefig("plot_features.png", dpi=180)
    plt.close()
    print("Saved plot_features.png")

# Save final summary
summary = {
    "n_train": len(X_train), "n_test": len(X_test),
    "best": {"model": best["model"], "vectorizer": best["vectorizer"],
             "acc": float(best["test_acc"]), "f1": float(best["f1"])},
    "results": results,
}
with open("summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\nDONE.")
