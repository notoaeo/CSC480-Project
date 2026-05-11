# Speaker Notes - CSC 480 Final Presentation

**Target length:** 3 minutes (budget ~35 seconds per slide)
**Total slides:** 5

---

## Slide 1 - Title (~15 seconds)

> "My project is sentiment analysis of movie reviews using
> classical machine learning. Quick reminder: the goal was to see
> how far simple scikit-learn pipelines can push on IMDb binary
> sentiment without any neural models. After ablation tuning, the
> best result is 91.3%."

Click to slide 2.

---

## Slide 2 - Recap (~30 seconds)

> "Quick recap from the mid-project. I compared Naive Bayes,
> Logistic Regression, and Linear SVM across three feature
> representations: bag of words, TF-IDF, and TF-IDF with bigrams.
> Linear SVM with TF-IDF bigrams came out on top at 91.1%, beating
> the Maas et al. 2011 baseline of 88.9% on the same dataset. What
> I promised to do next was ablations and deeper error analysis.
> That's what the rest of this talk covers."

Click to slide 3.

---

## Slide 3 - Ablation study (~45 seconds)

> "I ran ablations over stop words, sublinear TF, vocabulary size,
> and the SVM regularization parameter C. Two findings stand out."

> "First, removing stop words hurts. It drops accuracy 1.3 points.
> That was surprising until I looked at what gets removed: words
> like 'not,' 'no,' and 'never' are on the standard English stop
> word list, and those carry exactly the negation signal the model
> depends on. So for sentiment analysis, you should keep your stop
> words."

> "Second, bigrams are the single most impactful feature choice.
> Going from unigrams to bigrams adds 2 full points. Everything
> else, sublinear TF, min_df cutoff, the value of C, those all
> matter much less. The model is actually quite robust to
> hyperparameters once you have the right features."

Click to slide 4.

---

## Slide 4 - Error analysis (~45 seconds)

> "I also dug into where the model fails. On the left, accuracy by
> review length: short reviews are easiest at 92.5% because they
> tend to be blunt. Longer reviews are harder because they mix
> positive and negative language."

> "But the big finding is on the right. I tagged every test review
> for whether it contains negation words like 'not,' 'never,' or
> 'don't.' 89% of all errors, 1,976 out of 2,219, contain
> negation. Bigrams help with local negation like 'not good,' but
> they cannot catch long-range patterns like 'I thought this would
> be great but it wasn't.' That is the dominant failure mode."

Click to slide 5.

---

## Slide 5 - Skyline and conclusions (~45 seconds)

> "To put this in context: the chart on the left shows where my
> result sits. Maas et al. in 2011 got 88.9% using learned word
> vectors. I'm at 91.1% with classical methods. Fine-tuned BERT
> gets around 93.5%, and the current state of the art with
> RoBERTa ensembles is about 97%."

> "Three takeaways. One, classical ML captures most of the signal
> in this task. Two, how you represent the text matters more than
> which classifier you pick. And three, the wall you hit is
> negation, which is exactly where contextual models like BERT
> pick up the remaining headroom."

> "Thank you. Happy to take questions."

---

## Likely questions

**Q: Why does removing stop words hurt?**
A: Stop words include "not," "no," "never," "don't." These are
critical for sentiment. The standard advice to remove stop words
comes from information retrieval, not classification, and it
doesn't transfer.

**Q: Did you try tuning min_df and C together?**
A: Not a full grid search, but min_df=2 with C=0.5 gave 91.3%,
which is the best I found. The gains over the default are small.

**Q: What about the skyline? (Prof. Zhang's question)**
A: Fine-tuned BERT hits ~93.5%, RoBERTa ensembles reach ~97%.
My 91.1% closes about 27% of the gap between the 2011 baseline
and BERT, with no neural components.

**Q: Why not try BERT yourself?**
A: Scope was intentionally classical methods only. The point
is understanding what features and classifiers contribute,
which a transformer would obscure.

**Q: What would you do differently?**
A: I'd add explicit negation tagging (flip word polarity inside
"not ... punctuation" windows) as a cheap feature engineering
trick before jumping to neural models.
