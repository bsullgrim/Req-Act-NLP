import pandas as pd
import spacy
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load spaCy medium model (ensure: python -m spacy download en_core_web_md)
nlp = spacy.load("en_core_web_md")

# Load CSV exports from MagicDraw
requirements_df = pd.read_csv("requirements.csv")  # Columns: Requirement Name, Requirement Text
activities_df = pd.read_csv("activities.csv")      # Columns: Activity Name

# Clean up missing values
requirements_df["Requirement Text"] = requirements_df["Requirement Text"].fillna("")
activities_df["Activity Name"] = activities_df["Activity Name"].fillna("")

def extract_verb_noun(doc):
    verb = None
    nouns = []
    # Simple heuristic: first VERB lemma, plus all NOUN/PROPN lemmas
    for token in doc:
        if verb is None and token.pos_ == "VERB":
            verb = token.lemma_
        if token.pos_ in ("NOUN", "PROPN"):
            nouns.append(token.lemma_)
    noun_phrase = " ".join(nouns)
    return verb, noun_phrase

# Preprocess activities: extract verb, noun phrase, and vector of noun phrase
activity_data = []
activity_docs = list(nlp.pipe(activities_df["Activity Name"]))
for i, doc in enumerate(activity_docs):
    verb, noun_phrase = extract_verb_noun(doc)
    # Vectorize noun phrase (fallback to zero vector if empty)
    vec = nlp(noun_phrase).vector if noun_phrase else np.zeros((nlp.vocab.vectors_length,))
    activity_data.append({
        "Activity Name": activities_df.iloc[i]["Activity Name"],
        "Verb": verb,
        "Noun Phrase": noun_phrase,
        "Vector": vec
    })

TOP_N = 3  # Number of top matches per requirement

matches = []
requirement_docs = list(nlp.pipe(requirements_df["Requirement Text"]))

for i, r_doc in enumerate(requirement_docs):
    r_verb, r_noun_phrase = extract_verb_noun(r_doc)
    if not r_verb:
        # Skip requirements with no verb (optional)
        continue
    r_vec = nlp(r_noun_phrase).vector if r_noun_phrase else np.zeros((nlp.vocab.vectors_length,))

    # Filter activities to those with matching verb
    candidates = [a for a in activity_data if a["Verb"] == r_verb]

    if not candidates:
        # No verb match, fallback to all activities or skip
        candidates = activity_data

    # Calculate cosine similarity of noun phrases
    sims = []
    for a in candidates:
        if np.linalg.norm(r_vec) > 0 and np.linalg.norm(a["Vector"]) > 0:
            sim = cosine_similarity(r_vec.reshape(1, -1), a["Vector"].reshape(1, -1))[0][0]
        else:
            sim = 0.0
        sims.append((a["Activity Name"], sim))

    sims = sorted(sims, key=lambda x: x[1], reverse=True)[:TOP_N]

    for act_name, sim_score in sims:
        matches.append({
            "Requirement Name": requirements_df.iloc[i]["Requirement Name"],
            "Requirement Text": requirements_df.iloc[i]["Requirement Text"],
            "Requirement Verb": r_verb,
            "Requirement Noun Phrase": r_noun_phrase,
            "Activity Name": act_name,
            "Similarity Score": round(sim_score, 3)
        })

matches_df = pd.DataFrame(matches)

matches_df.to_csv("req_to_activity_matches.csv", index=False, encoding="utf-8")
matches_df.to_excel("req_to_activity_matches.xlsx", index=False, engine="openpyxl")

print("Done! See req_to_activity_matches.csv and req_to_activity_matches.xlsx for your results.")
