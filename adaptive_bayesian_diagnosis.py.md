---
description: >-
  This version not only loads disease/symptom likelihoods from a CSV but also
  updates priors automatically each time you run it, based on the frequency of
  diagnoses
---

# adaptive\_bayesian\_diagnosis.py

{% include ".gitbook/includes/adaptive-bayesian-diagnos....md" %}

#### 🧩 What’s New

✅ **Loads disease/symptom data from CSV**\
✅ **Asks user about symptoms interactively**\
✅ **Shows posterior probabilities + chart**\
✅ **Asks if the diagnosis was correct**\
✅ **Updates priors automatically** (learns over time)\
✅ **Stores experience in `diagnosis_stats.csv`**

#### 🧾 Example Workflow

1. Run `python adaptive_bayesian_diagnosis.py`.
2. Answer “Y/N” for symptoms.
3. The program shows the diagnosis and bar chart.
4. You confirm or correct it — the priors update automatically.
5. The next time you run it, it uses improved priors.
