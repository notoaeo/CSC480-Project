# Speaker Notes — CSC 480 Mid-Project Presentation

**Target length:** 6 minutes (class window is 5–10, aim for the low end so you
land comfortably and leave room for one or two questions).
**Total slides:** 6.
**Pacing budget:** roughly 1 minute per slide, with slide 4 getting the most
time.

A few delivery notes before the slide-by-slide script:

- Say the headline number (**91.1%**) out loud at least twice. Once on slide 1,
  once on slide 4. It is the single thing you want people to remember.
- The "I" vs "we": pick one and stick with it. This is a solo project so "I"
  is natural, but if you are more comfortable with "we" that also reads as
  modest and is fine.
- Do not read the slides. The slides are scaffolding; your voice is the content.
- If you get nervous, slow down on transitions between slides. That is the
  moment the audience is catching up, and slowing down reads as confidence.

---

## Slide 1 — Title (~30 seconds)

> "Hi everyone. My project is on sentiment analysis of movie reviews using
> classical machine learning. The one-line version: I wanted to see how far
> plain scikit-learn pipelines can push on this task before anyone reaches for
> word vectors or neural models. Short answer, about 91 percent, which I'll
> come back to in a minute."

**Cue to click:** after you say "come back to in a minute," move to slide 2.

---

## Slide 2 — The question (~50 seconds)

> "The reason this is interesting is that movie reviews are messy. Two people
> can use the same words and mean opposite things, sarcasm is everywhere, and
> writing style varies a lot. So the underlying question is whether a
> classifier that only sees word counts can reliably pick up on human tone."

> "I'm comparing three classical models: Multinomial Naive Bayes as a
> probabilistic baseline, Logistic Regression as a linear workhorse, and a
> Linear SVM, which is known to be strong on high-dimensional text."

> "On the features side, I originally proposed unigrams only, with bigrams as
> a fallback. After seeing early results I pulled bigrams into the main
> comparison, because negation patterns like 'not good' matter a lot for this
> task and bigrams are the cheapest way to capture them."

**Cue to click:** right after "cheapest way to capture them."

---

## Slide 3 — Data and pipeline (~50 seconds)

> "The dataset is the IMDb Large Movie Review Dataset from Maas et al. 2011.
> 50,000 labeled reviews, perfectly balanced between positive and negative,
> and I use their original 25k/25k train/test split so my results compare
> apples to apples with their paper."

> "The pipeline is straightforward: strip HTML tags, lowercase, vectorize with
> either bag of words or TF-IDF, train, and evaluate with five-fold
> cross-validation on the training set plus accuracy, precision, recall, and
> F1 on the held-out test set."

> "One number worth noticing: adding bigrams blows the vocabulary up from
> about 27,000 features to 152,000, which is what lets the model finally see
> phrases like 'the worst' and 'the best' as single signals."

**Cue to click:** after "single signals."

---

## Slide 4 — Results (~90 seconds, the centerpiece)

> "This is the main result. Rows are the three classifiers, columns are the
> three feature representations, and the table on the right has the exact
> numbers."

*(Point at the chart.)*

> "Two things jump out. First, bigrams help every single model. Naive Bayes
> gets the biggest jump, almost three points, which is consistent with the
> negation story: NB is bag-of-words at its purest, and pairs like 'not good'
> are exactly the kind of signal unigram NB cannot see."

> "Second, Linear SVM with TF-IDF bigrams lands at 91.12 percent on the test
> set. The original Maas et al. paper reported 88.9 percent on this same split
> using learned word vectors. So a straight classical pipeline, no embeddings,
> beats their result by more than two points. That's the headline."

> "Cross-validation agrees with the held-out test numbers to within about
> half a point, so this isn't an overfit to the test set."

**Cue to click:** after "overfit to the test set."

---

## Slide 5 — Inside the model (~50 seconds)

> "I did a quick sanity check and error analysis. On the left are the top
> fifteen positive and negative features from the Logistic Regression model.
> The positive side is 'great, excellent, perfect, amazing, hilarious.' The
> negative side is 'worst, awful, boring, waste, terrible.' Bigrams like 'the
> best' and 'the worst' show up in both lists. That all looks right."

> "On the right is the confusion matrix for the best SVM. Errors are roughly
> balanced between false positives and false negatives, about 11 hundred each,
> so the model isn't biased toward one class."

> "The interesting thing in the error analysis is that misclassified reviews
> are noticeably longer on average, 235 words versus 229 for correct ones.
> My working hypothesis is that long reviews are where mixed tones and
> sarcasm live, and that's the next thing I want to dig into."

**Cue to click:** after "next thing I want to dig into."

---

## Slide 6 — What's next (~50 seconds)

> "Remaining work. Between now and the final report I'm focusing on three
> things. One, a deeper error taxonomy, broken out by review length and by
> whether the review contains negation. Two, ablations over vocabulary size,
> stop words, and sublinear TF, which should help me separate Logistic
> Regression from SVM. They're within a point right now and I'd like to know
> if the gap is real."

> "On the risk side, the biggest one is that my own 91 percent is hard to
> beat without moving to word vectors. If extensions stall, I plan to shift
> some of the writeup toward interpretability and calibration rather than
> chasing another accuracy point."

> "That's my update. Thanks, happy to take questions."

**Cue to stop:** after "happy to take questions." Stand still, smile, wait.

---

## Likely questions and short answers

Have these ready. They are the questions most likely to come up.

**Q: Why not include a neural baseline?**
A: Intentional for the mid-project. The proposal scope was classical
   pipelines only. The final report will include a short discussion of where
   word-vector and transformer models pick up the remaining headroom, but I
   didn't want to dilute the comparison for this checkpoint.

**Q: What happens if you tune C for the SVM?**
A: I used C=1.0 for all the numbers on slide 4. Quick exploration suggested
   the SVM is not very sensitive around that value, but a grid over C is one
   of the ablations I plan to run before the final report.

**Q: Could the 2 point gain over Maas et al. just be from the bigger
   vectorizer, not the SVM?**
A: Partly. TF-IDF unigrams alone already hit 89.3 percent with LogReg, so
   about half the gain comes from the features and the other half from the
   SVM plus bigrams together. I'll separate those contributions in the
   ablation section.

**Q: Why stop at bigrams?**
A: Tried, memory and vocab size explode fast and trigrams did not help in
   preliminary runs. I'll include the trigram row in the ablation table.

**Q: Did you handle negation explicitly?**
A: Not yet. Right now bigrams are my proxy for negation. Explicit negation
   tagging (flipping the sign of words inside a "not ... [punctuation]"
   window) is on the list for the next two weeks.
