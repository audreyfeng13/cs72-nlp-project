"""Category prediction with Claude Haiku, using the project's own category definitions.

Shared by the Streamlit demo (app.py) and the evaluation script (llm_categories.py).
One category per comment. The definitions come from the annotators' sheet (typos fixed).
"""
CATEGORY_MODEL = "claude-haiku-5-5"

# Order matters only for display; the model sees all five definitions.
CATEGORY_NAMES = ["Threats/Intimidation", "Sexualisation/Objectification", "Stereotypes/Exclusion",
                  "Personal Insult", "Safe"]
DEFINITIONS = {
    "Threats/Intimidation": "threats, wishing injury on someone, doxxing, or encouraging self-harm",
    "Sexualisation/Objectification": "unsolicited sexual remarks, body-part fixation, or lewd or romantic propositions",
    "Stereotypes/Exclusion": "gender or identity generalisations, gatekeeping, or dismissing women's sports as inferior",
    "Personal Insult": "direct name-calling, mockery, non-sexual appearance policing, or targeted character attacks",
    "Safe": "none of the above",
}
PROMPT = (
    "You are classifying a social media comment left under an athlete's post. "
    "Choose exactly one category for the comment.\n\n"
    + "\n".join(f"- {name}: {DEFINITIONS[name]}" for name in CATEGORY_NAMES)
    + "\n\nAnswer with only the category name, written exactly as above.\n\nComment: {text}"
)
# Accepted spellings in a reply (checked case-insensitively; the earliest match in the reply wins).
_ALIASES = {
    "Threats/Intimidation": ["threats/intimidation", "threats", "intimidation"],
    "Sexualisation/Objectification": ["sexualisation/objectification", "sexualization/objectification",
                                      "sexualisation", "sexualization", "objectification"],
    "Stereotypes/Exclusion": ["stereotypes/exclusion", "stereotypes", "exclusion"],
    "Personal Insult": ["personal insult", "insult"],
    "Safe": ["safe"],
}


def parse_category(reply: str):
    """Return the category named in `reply`, or None if no category is recognised."""
    r = reply.lower()
    hits = [(r.find(a), name) for name, aliases in _ALIASES.items() for a in aliases if a in r]
    return min(hits)[1] if hits else None


def predict_category(client, text: str, model: str = CATEGORY_MODEL, max_tokens: int = 1024):
    """Ask Claude for the comment's category. Returns (category or None, raw reply).
    max_tokens leaves headroom because the model may emit a reasoning block before its answer."""
    r = client.messages.create(model=model, max_tokens=max_tokens,
                               messages=[{"role": "user", "content": PROMPT.format(text=text)}])
    reply = "".join(b.text for b in r.content if b.type == "text")
    return parse_category(reply), reply
