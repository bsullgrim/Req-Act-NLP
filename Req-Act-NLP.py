import pandas as pd
import spacy
import numpy as np

# Configuration
VERB_NOUN_WEIGHT = 0.8  # 60% weight to verb/noun presence
SEMANTIC_WEIGHT = 0.2   # 40% weight to semantic similarity
MIN_SIMILARITY = 0.4    # Minimum combined score to consider
TOP_N = 5               # Max matches per requirement

# Load spaCy transformer model
nlp = spacy.load("en_core_web_trf")

def trf_similarity(doc1, doc2):
    # Get the last hidden layer state and convert to numpy
    vec1 = doc1._.trf_data.last_hidden_layer_state.data.mean(axis=0)
    vec2 = doc2._.trf_data.last_hidden_layer_state.data.mean(axis=0)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


# Load data
requirements_df = pd.read_csv("requirements.csv").fillna({"Requirement Text": ""})
activities_df = pd.read_csv("activities.csv").fillna({"Activity Name": ""})

# Extract verbs/nouns from activities
def extract_action_parts(text):
    doc = nlp(text)
    verbs = [token.lemma_.lower() for token in doc if token.pos_ == "VERB"]
    nouns = [token.lemma_.lower() for token in doc if token.pos_ in ("NOUN", "PROPN")]
    return verbs, nouns

activities_df[["Verbs", "Nouns"]] = activities_df["Activity Name"].apply(
    lambda x: pd.Series(extract_action_parts(x))
)

# Precompute activity docs
activity_docs = list(nlp.pipe(activities_df["Activity Name"]))

matches = []
for req_idx, req_row in requirements_df.iterrows():
    req_text = req_row["Requirement Text"]
    req_doc = nlp(req_text)
    req_lemmas = {token.lemma_.lower() for token in req_doc}
    scores = []
    for act_idx, (act_name, act_verbs, act_nouns) in enumerate(zip(
        activities_df["Activity Name"],
        activities_df["Verbs"],
        activities_df["Nouns"]
    )):
        # Verb/Noun Matching Score (0-1)
        verb_score = sum(1 for v in act_verbs if v in req_lemmas) / max(1, len(act_verbs))
        noun_score = sum(1 for n in act_nouns if n in req_lemmas) / max(1, len(act_nouns))
        vn_score = (verb_score + noun_score) / 2

        # Semantic Similarity Score (0-1)
        sim_score = trf_similarity(req_doc, activity_docs[act_idx])

        # Combined Score
        combined_score = (vn_score * VERB_NOUN_WEIGHT) + (sim_score * SEMANTIC_WEIGHT)

        if combined_score >= MIN_SIMILARITY:
            scores.append((combined_score, act_idx, vn_score, sim_score))

    # Sort and keep top N
    for score, act_idx, vn_score, sim_score in sorted(scores, reverse=True)[:TOP_N]:
        matches.append({
            "Requirement ID": req_row["ID"],
            "Requirement Name": req_row["Requirement Name"],
            "Requirement Text": req_text,
            "Activity Name": activities_df.iloc[act_idx]["Activity Name"],
            "Activity Verbs": ", ".join(activities_df.iloc[act_idx]["Verbs"]),
            "Activity Nouns": ", ".join(activities_df.iloc[act_idx]["Nouns"]),
            "VerbNoun Score": round(vn_score, 3),
            "Similarity Score": round(sim_score, 3),
            "Combined Score": round(score, 3)
        })

# Save results
matches_df = pd.DataFrame(matches)
matches_df.to_csv("hybrid_matches_trf.csv", index=False)
matches_df.to_excel("hybrid_matches_trf.xlsx", index=False, engine="openpyxl")

print(f"Generated {len(matches)} matches. Results saved to hybrid_matches_trf.csv and .xlsx")
