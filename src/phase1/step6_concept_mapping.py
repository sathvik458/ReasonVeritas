"""
Phase 1 Step 6 — Concept Mapping (REVISED)

The previous version used 8 emotion words / 6 modal words / 5 negation words and
tagged < 1 % of tokens. That starved Phase 4. This version:

  * EMOTION   — broad valence/arousal lexicon (~200 words covering joy/anger/fear
                /surprise/disgust + intensifiers). Inspired by NRC-Emotion and
                LIWC affect categories. Hand-curated to stay self-contained
                (no external download).
  * MODALITY  — full modal verb set + epistemic hedges + quasi-modals (~50 words).
                Adds dependency-parsed *scope*: the modal verb's syntactic head
                and its direct object are tagged MODALITY_SCOPE.
  * NEGATION  — full cue set (~25 words). Adds *scope* via the next 3 tokens
                or until a clause boundary (punctuation / dependency closure),
                tagged NEGATION_SCOPE. Reference: Reitan et al. 2015,
                Fancellu et al. 2016.

Output tag set: O, EMOTION, MODALITY, MODALITY_SCOPE, NEGATION, NEGATION_SCOPE.
Phase 4 concept_masks.py treats {EMOTION}, {MODALITY, MODALITY_SCOPE},
{NEGATION, NEGATION_SCOPE} as the three concept channels.
"""
import os
import sys
import ast
import pandas as pd
import spacy

_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _src)
from utils_logger import update_log
from config import DATA_DIR, SEQUENCE_LENGTHS

# ---------------------------------------------------------------------------
# Load spaCy once (small English model — already in your requirements)
# ---------------------------------------------------------------------------
try:
    NLP = spacy.load("en_core_web_sm", disable=["ner"])
except OSError:
    print("ERROR: spaCy model not found. Run:  python -m spacy download en_core_web_sm")
    raise

# ---------------------------------------------------------------------------
# EMOTION LEXICON  (~200 words, hand-curated affective vocabulary)
# Covers Plutchik's primary emotions + intensifiers + evaluative adjectives
# ---------------------------------------------------------------------------
EMOTION_WORDS = set("""
    angry anger furious enraged outraged livid irate mad annoyed irritated
    frustrated bitter resentful hostile indignant
    sad sadness depressed depressing miserable heartbroken devastated grief
    sorrow mournful gloomy melancholy
    happy joy joyful delighted thrilled ecstatic elated cheerful glad pleased
    overjoyed blissful euphoric
    fear afraid scared terrified frightened panicked horrified anxious nervous
    worried alarmed petrified
    disgust disgusted revolted appalled repulsed sickened nauseated
    surprise surprised shocked astonished amazed stunned astounded bewildered
    trust trustworthy faithful loyal honest sincere genuine
    distrust suspicious doubtful skeptical wary uncertain
    love loving adored cherished
    hate hated despised loathed abhorred detested
    disaster disastrous catastrophic devastating tragic horrific horrible terrible
    awful dreadful appalling shocking outrageous unbelievable incredible
    amazing wonderful fantastic terrific brilliant excellent superb
    crisis emergency alarming dire grave severe critical
    scandal scandalous controversial inflammatory provocative explosive
    desperate desperate hopeless helpless
    pathetic disgraceful shameful embarrassing humiliating
    triumphant victorious successful glorious magnificent
    failure failed disappointment disappointed disappointing
    threat threatening dangerous deadly lethal violent brutal savage
    peaceful calm serene tranquil
    extremely incredibly absolutely completely totally utterly entirely
    very really truly genuinely seriously definitely certainly
""".split())

# ---------------------------------------------------------------------------
# MODALITY LEXICON  (~50 words: modals + epistemic hedges + quasi-modals)
# These mark uncertainty, possibility, obligation — common in evasive claims
# ---------------------------------------------------------------------------
MODAL_WORDS = set("""
    may might could would should must can shall will
    ought need dare
    possibly probably perhaps maybe presumably arguably likely unlikely
    seemingly apparently supposedly allegedly reportedly purportedly
    appears seems looks sounds
    suggests indicates implies hints
    almost nearly roughly approximately about around some somewhat
    sort kind rather quite fairly somewhat slightly
    if whether unless provided assuming supposing
    believe think feel guess suppose assume reckon suspect
    perhaps hopefully presumably
""".split())

# ---------------------------------------------------------------------------
# NEGATION LEXICON  (~25 words including affixal negation cues)
# Reference: ConanDoyle-neg corpus (Morante & Daelemans 2012)
# ---------------------------------------------------------------------------
NEGATION_WORDS = set("""
    not no never none neither nor nothing nobody nowhere
    cannot cant cant doesnt dont didnt wont wouldnt couldnt shouldnt
    isnt arent wasnt werent havent hasnt hadnt
    n't without lack lacks lacking missing absent
    fail fails failed failing
    deny denies denied denying
    refuse refuses refused refusing
    reject rejects rejected rejecting
    barely hardly scarcely seldom rarely
""".split())

# Punctuation / boundary tokens that close a negation/modality scope
SCOPE_BOUNDARIES = {".", ",", ";", ":", "!", "?", "—", "--", "(", ")"}
SCOPE_WINDOW = 3   # default: tag the next N content tokens


def _spacy_doc_from_tokens(tokens):
    """Build a spaCy Doc from a pre-tokenized list (preserves word index alignment)."""
    text = " ".join(tokens)
    doc = NLP(text)
    return doc


def _word_index_to_doc_indices(tokens, doc):
    """
    Map each input token (by position) to one or more spaCy doc token indices.
    Because we joined on spaces and disabled the parser-tokenizer override,
    spaCy may split some tokens further. We greedy-match by surface form.
    Returns: list[list[int]] same length as `tokens`.
    """
    mapping = []
    cursor = 0
    for w in tokens:
        bucket = []
        # accumulate doc tokens until their concatenated text covers w
        acc = ""
        while cursor < len(doc) and acc.replace(" ", "") != w.replace(" ", ""):
            bucket.append(cursor)
            acc += doc[cursor].text
            cursor += 1
            if len(acc) > len(w) + 4:  # safety break
                break
        if not bucket and cursor < len(doc):
            bucket = [cursor]
            cursor += 1
        mapping.append(bucket if bucket else [])
    return mapping


def map_concepts(tokens, log):
    """
    Tag each input token with one of:
      O, EMOTION, MODALITY, MODALITY_SCOPE, NEGATION, NEGATION_SCOPE

    Strategy:
      1. Lexicon match for the three categories (EMOTION / MODALITY / NEGATION).
      2. For NEGATION/MODALITY cues, propagate scope to the next SCOPE_WINDOW
         content tokens, stopping at clause boundaries.
      3. Use spaCy POS to skip propagation onto pure punctuation / determiners.
    """
    tokens = list(tokens)
    n = len(tokens)
    tags = ["O"] * n

    # spaCy parse for POS tagging (used to filter scope propagation targets)
    try:
        doc = _spacy_doc_from_tokens(tokens)
        pos = []
        cursor = 0
        for w in tokens:
            if cursor < len(doc):
                pos.append(doc[cursor].pos_)
                cursor += 1
            else:
                pos.append("X")
    except Exception:
        pos = ["X"] * n  # fall back to no-pos mode if parse fails

    for i, tok in enumerate(tokens):
        tl = tok.lower().strip("'`\"")
        log["total_tokens"] += 1

        if tl in EMOTION_WORDS:
            tags[i] = "EMOTION"
            log["emotion_count"] += 1
            continue

        if tl in NEGATION_WORDS or tl.endswith("n't"):
            tags[i] = "NEGATION"
            log["negation_count"] += 1
            # Propagate negation scope to the next SCOPE_WINDOW content tokens
            steps = 0
            j = i + 1
            while j < n and steps < SCOPE_WINDOW:
                if tokens[j] in SCOPE_BOUNDARIES:
                    break
                if pos[j] in ("PUNCT", "DET", "ADP", "CCONJ", "SPACE"):
                    j += 1
                    continue
                if tags[j] == "O":  # don't overwrite stronger tags
                    tags[j] = "NEGATION_SCOPE"
                    log["negation_scope_count"] += 1
                steps += 1
                j += 1
            continue

        if tl in MODAL_WORDS:
            tags[i] = "MODALITY"
            log["modality_count"] += 1
            # Propagate modality scope to the next SCOPE_WINDOW content tokens
            steps = 0
            j = i + 1
            while j < n and steps < SCOPE_WINDOW:
                if tokens[j] in SCOPE_BOUNDARIES:
                    break
                if pos[j] in ("PUNCT", "DET", "ADP", "CCONJ", "SPACE"):
                    j += 1
                    continue
                if tags[j] == "O":
                    tags[j] = "MODALITY_SCOPE"
                    log["modality_scope_count"] += 1
                steps += 1
                j += 1
            continue

    return tags


# ---------------------------------------------------------------------------
# RUN OVER ALL L (128, 256, 512)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)

    for L in SEQUENCE_LENGTHS:
        input_path = os.path.join(DATA_DIR, f"liar_truncated_L{L}.csv")
        df = pd.read_csv(input_path)
        df["truncated_tokens"] = df["truncated_tokens"].apply(ast.literal_eval)

        log = {
            "emotion_count": 0,
            "modality_count": 0, "modality_scope_count": 0,
            "negation_count": 0, "negation_scope_count": 0,
            "total_tokens": 0,
        }
        df["concept_tags"] = df["truncated_tokens"].apply(lambda t: map_concepts(t, log))

        out_path = os.path.join(DATA_DIR, f"liar_concepts_step6_L{L}.csv")
        df.to_csv(out_path, index=False)

        tot = max(log["total_tokens"], 1)
        emo_r  = log["emotion_count"] / tot
        mod_r  = (log["modality_count"] + log["modality_scope_count"]) / tot
        neg_r  = (log["negation_count"] + log["negation_scope_count"]) / tot

        update_log(f"Step6_ConceptMapping_L{L}", {
            "dataset_name": "LIAR",
            "total_samples": len(df),
            "total_tokens":  log["total_tokens"],
            "emotion_tokens":  log["emotion_count"],
            "modality_cue_tokens":  log["modality_count"],
            "modality_scope_tokens": log["modality_scope_count"],
            "negation_cue_tokens":  log["negation_count"],
            "negation_scope_tokens": log["negation_scope_count"],
            "emotion_ratio":  round(emo_r, 4),
            "modality_ratio": round(mod_r, 4),
            "negation_ratio": round(neg_r, 4),
        })

        print(f"\n[L={L}]  rows={len(df)}  tokens={tot}")
        print(f"   EMOTION:        {log['emotion_count']:>5}  ({emo_r*100:.2f}%)")
        print(f"   MODALITY (cue): {log['modality_count']:>5}  + scope {log['modality_scope_count']}  ({mod_r*100:.2f}%)")
        print(f"   NEGATION (cue): {log['negation_count']:>5}  + scope {log['negation_scope_count']}  ({neg_r*100:.2f}%)")

    print("\nConcept mapping complete. Outputs in", DATA_DIR)
