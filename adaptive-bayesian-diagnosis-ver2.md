---
description: >-
  Store full patient case histories (date, symptoms, diagnosis, and confidence
  levels) in a separate CSV for later analysis or visualization
---

# Adaptive Bayesian Diagnosis Ver2

This version adds **case history logging**: every diagnostic session (with symptoms, computed probabilities, chosen diagnosis, and user confirmation) is saved in a `case_history.csv` file for later review or analysis.

{% code title="adaptive_bayesian_diagnosis_history.py" overflow="wrap" lineNumbers="true" %}
```python
// 
# Adaptive Bayesian Diagnostic Assistant with Case History Logging
# Learns from diagnosis outcomes and stores full patient cases.
# Author: ChatGPT

import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

DATA_FILE = "disease_data.csv"
STATS_FILE = "diagnosis_stats.csv"
HISTORY_FILE = "case_history.csv"


# -------------------------------
# Utility Functions
# -------------------------------

def load_data(csv_path):
    """Load disease priors and likelihoods from a CSV."""
    df = pd.read_csv(csv_path)
    priors = {}
    likelihoods = {}

    for _, row in df.iterrows():
        disease = row["disease"].lower()
        priors[disease] = float(row["prior"])
        likelihoods[disease] = row.drop(["disease", "prior"]).astype(float).to_dict()

    return priors, likelihoods


def get_symptoms(likelihoods):
    """Return list of all symptoms in dataset."""
    return list(next(iter(likelihoods.values())).keys())


def load_or_init_stats(priors):
    """Load or initialize diagnosis statistics."""
    if os.path.exists(STATS_FILE):
        df = pd.read_csv(STATS_FILE, index_col=0)
        return df["count"].to_dict()
    else:
        stats = {disease: 1 for disease in priors}  # Initialize with 1 for stability
        save_stats(stats)
        return stats


def save_stats(stats):
    """Save diagnosis statistics."""
    df = pd.DataFrame({"disease": list(stats.keys()), "count": list(stats.values())})
    df.to_csv(STATS_FILE, index=False)


def update_priors(priors, stats):
    """Recalculate priors based on observed counts."""
    total = sum(stats.values())
    for disease in priors:
        priors[disease] = stats[disease] / total
    return priors


def log_case(patient_symptoms, posteriors, likely, correct=None):
    """Append full diagnostic record to case history file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    record = {
        "timestamp": timestamp,
        "predicted_diagnosis": likely,
        "confirmed_correct": correct,
    }

    # Add symptom responses
    record.update({f"symptom_{k}": v for k, v in patient_symptoms.items()})

    # Add posterior probabilities
    record.update({f"prob_{k}": round(v, 3) for k, v in posteriors.items()})

    df_record = pd.DataFrame([record])

    # Append or create file
    if os.path.exists(HISTORY_FILE):
        df_record.to_csv(HISTORY_FILE, mode="a", header=False, index=False)
    else:
        df_record.to_csv(HISTORY_FILE, mode="w", header=True, index=False)


# -------------------------------
# Core Diagnosis Function
# -------------------------------

def bayesian_diagnosis(priors, likelihoods, stats):
    symptoms = get_symptoms(likelihoods)
    patient_symptoms = {}

    print("=== Adaptive Bayesian Diagnostic Assistant ===")
    print("Please answer Y/N for the following symptoms:\n")

    for s in symptoms:
        while True:
            ans = input(f"Do you have {s.replace('_', ' ')}? (Y/N): ").strip().lower()
            if ans in ["y", "n"]:
                patient_symptoms[s] = ans == "y"
                break
            else:
                print("Please enter 'Y' or 'N'.")

    # Compute posterior probabilities
    posteriors = {}
    for disease, prior in priors.items():
        p_disease = prior
        for symptom, present in patient_symptoms.items():
            likelihood = float(likelihoods[disease][symptom])
            p_disease *= likelihood if present else (1 - likelihood)
        posteriors[disease] = p_disease

    # Normalize
    total = sum(posteriors.values())
    for disease in posteriors:
        posteriors[disease] /= total

    print("\n=== Diagnostic Probabilities ===")
    for disease, prob in sorted(posteriors.items(), key=lambda x: -x[1]):
        print(f"{disease.capitalize():<20}: {prob:.3f}")

    likely = max(posteriors, key=posteriors.get)
    print(f"\n→ Most likely condition: {likely.capitalize()}")

    # Visualization
    plt.bar(posteriors.keys(), posteriors.values(), color=['orange', 'skyblue', 'lightgreen'])
    plt.title("Posterior Probabilities by Condition")
    plt.xlabel("Condition")
    plt.ylabel("Probability")
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.show()

    # User feedback for adaptive learning
    correct = None
    while True:
        confirm = input(f"\nWas the diagnosis ({likely.capitalize()}) correct? (Y/N): ").strip().lower()
        if confirm == "y":
            stats[likely] += 1
            correct = True
            break
        elif confirm == "n":
            print("Please enter the correct diagnosis from the list:")
            for d in priors.keys():
                print(" -", d)
            correct_diag = input("Enter correct diagnosis: ").strip().lower()
            if correct_diag in stats:
                stats[correct_diag] += 1
                correct = False
                likely = correct_diag
            break
        else:
            print("Please answer Y or N.")

    # Log this diagnostic case
    log_case(patient_symptoms, posteriors, likely, correct)

    # Update priors
    save_stats(stats)
    priors = update_priors(priors, stats)

    # Update priors in source CSV for persistence
    df = pd.read_csv(DATA_FILE)
    for d in priors:
        df.loc[df["disease"].str.lower() == d, "prior"] = priors[d]
    df.to_csv(DATA_FILE, index=False)

    print("\nUpdated priors based on observed outcomes:")
    for d, p in priors.items():
        print(f"{d.capitalize():<20}: {p:.3f}")

    print("\nCase logged successfully to case_history.csv")


# -------------------------------
# Entry Point
# -------------------------------

if __name__ == "__main__":
    try:
        priors, likelihoods = load_data(DATA_FILE)
        stats = load_or_init_stats(priors)
        priors = update_priors(priors, stats)
        bayesian_diagnosis(priors, likelihoods, stats)
    except FileNotFoundError:
        print(f"Error: '{DATA_FILE}' not found. Please create it using the proper format.")

```
{% endcode %}

#### 🧩 What This Version Adds

✅ **Adaptive learning:** updates priors automatically with each confirmed diagnosis\
✅ **Case history logging:** saves

* date/time
* user symptom responses
* posterior probabilities
* predicted + confirmed diagnosis

✅ **Persistent priors:** updates saved in both `diagnosis_stats.csv` and the main `disease_data.csv`\
✅ **Data analysis ready:** you can later load `case_history.csv` into Excel, Pandas, or a BI tool for trend tracking

🧾 **Example Output in** `case_history.csv`

| timestamp           | predicted\_diagnosis | confirmed\_correct | symptom\_fever | symptom\_cough | symptom\_nausea | prob\_flu | prob\_cold | prob\_gastrointestinal |
| ------------------- | -------------------- | ------------------ | -------------- | -------------- | --------------- | --------- | ---------- | ---------------------- |
| 2025-10-06 14:23:45 | gastrointestinal     | True               | True           | False          | True            | 0.12      | 0.08       | 0.80                   |
