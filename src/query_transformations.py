# src/query_transformations.py


# ======================================================================================
# LLM-based Transformations
# These require an external LLM call. The prompts are defined here.
# ======================================================================================

SEMANTICS_PRESERVING_PARAPHRASE_PROMPT = """
You are given a search query.

Rewrite the query so that its meaning is preserved EXACTLY, but the surface form is slightly altered. 
Return a similar length query with all the details.
ONLY RETURN the rewritten query without any additional text.

You MUST:
- Paraphrase lightly (do not rephrase aggressively)
- Substitute some words with close synonyms
- Slightly reorder words where natural
- Remove or change punctuation

You MUST NOT:
- Add new information
- Remove important concepts
- Change the intent or assumptions of the query
- Add explanations or commentary

The rewritten query should still retrieve the same relevant documents.

Return ONLY the rewritten query, nothing else. NOT in quotation marks and DO NOT say "Here's the rewritten query:" or any paraphrased version of that.
Keep the query realistic and natural; DO NOT exaggerate informality. Maintain the same level of formality as the original query.

Original query:
"{QUERY}"
"""

SEMANTICS_NEUTRAL_NOISE_PROMPT = """
You are given a search query.

Rewrite the query so that the INTENDED MEANING is preserved, but the query becomes much noisier and less well-formed.
Return a similar length query with the same details.
ONLY RETURN the rewritten query without any additional text.

You MUST apply the following:
- Introduce typos throughout the entire query. You must use character swaps, deletions, and misspellings
- Insert irrelevant fluff words all throughout the query to add lots of noise

You MUST NOT:
- Change the underlying intent
- Add or remove core concepts
- Introduce incorrect assumptions

The query should sound less precise, with irrelevant noise, but still express the same information need.

Return ONLY the rewritten query, nothing else. NOT in quotation marks and DO NOT say "Here's the rewritten query:" or any paraphrased version of that.
Avoid exaggerated slang, character voices, or parody-like phrasing.
Keep the query realistic and natural; DO NOT exaggerate informality. Maintain the same level of formality as the original query.

Original query:
"{QUERY}"
"""

SEMANTICS_BREAKING_INCORRECT_PREMISE_PROMPT = """
You are given a search query.

Rewrite the query so that its meaning is ALTERED and SLIGHTLY but CLEARLY DISTORTED by introducing incorrect or misleading premises.
Return a similar length query with the same details.
ONLY RETURN the rewritten query without any additional text.

You SHOULD:
- Negate or invert a key causal or factual assumption
- Remove or weaken some critical keywords
- Make the query subtly but meaningfully incorrect

You MUST:
- Keep the query coherent and plausible
- Ensure it still resembles a reasonable user query
- Avoid making the query completely nonsensical

You MUST NOT:
- Turn it into a joke or absurd query
- Completely change the topic
- Add explanations or commentary

The rewritten query should look realistic but express a flawed assumption.

Return ONLY the rewritten query, nothing else. NOT in quotation marks and DO NOT say "Here's the rewritten query:" or any paraphrased version of that.
Keep the query realistic and natural; DO NOT exaggerate informality. Maintain the same level of formality as the original query.
DO NOT wrap the query in quotation marks.

Original query:
"{QUERY}"
"""


# These are the official transformation names for Phase 15
TRANSFORMATION_NAMES = [
    "semantics_preserving_paraphrase",
    "semantics_neutral_noise",
    "semantics_breaking_incorrect_premise",
]
