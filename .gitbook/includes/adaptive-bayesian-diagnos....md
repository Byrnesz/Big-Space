---
title: '# Adaptive Bayesian Diagnos...'
---

{% code title="adaptive_bayesian_diagnosis.py" overflow="wrap" lineNumbers="true" %}
```python
# Adaptive Bayesian Diagnostic Assistant
# Learns from diagnosis outcomes using Bayesian updating.
# Author: ChatGPT

import pandas as pd
import matplotlib.pyplot as plt
import os

DATA_FILE = "disease_data.csv"
STATS_FILE = "diagnosis_stats.csv"


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
    """Load prior update stats (counts of diagnoses), or initialize if not present."""
    if os.path.exists(STATS_FILE):
        df = pd.read_csv(STATS_FILE, index_col=0)
        return df["count"].to_dict()
    else:
        stats = {disease: 1 for disease in priors}  # Start with 1 to avoid zero division
        save_stats(stats)
        return stats


def save_stats(stats):
    """Save diagnosis counts to CSV."""
    df = pd.DataFrame({"disease": list(stats.keys()), "count": list(stats.values())})
    df.to_csv(STATS_FILE, index=False)


def update_priors(priors, stats):
    """Update priors based on observed frequencies."""
    total = sum(stats.values())
    for disease in priors:
        priors[disease] = stats[disease] / total
    return priors


def bayesian_diagnosis(priors, likelihoods, stats):
    """Interactive diagnosis session."""
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

    # Update learning stats
    while True:
        confirm = input(f"\nWas the diagnosis ({likely.capitalize()}) correct? (Y/N): ").strip().lower()
        if confirm == "y":
            stats[likely] += 1
            save_stats(stats)
            break
        elif confirm == "n":
            print("Please enter the correct diagnosis from the list:")
            for d in priors.keys():
                print(" -", d)
            correct = input("Enter correct diagnosis: ").strip().lower()
            if correct in stats:
                stats[correct] += 1
                save_stats(stats)
                break
            else:
                print("Unknown disease, skipping update.")
                break
        else:
            print("Please answer Y or N.")

    # Update priors based on new stats
    priors = update_priors(priors, stats)
    print("\nUpdated priors based on observed outcomes:")
    for d, p in priors.items():
        print(f"{d.capitalize():<20}: {p:.3f}")

    # Save updated priors back to CSV for persistence
    df = pd.read_csv(DATA_FILE)
    for d in priors:
        df.loc[df["disease"].str.lower() == d, "prior"] = priors[d]
    df.to_csv(DATA_FILE, index=False)


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
