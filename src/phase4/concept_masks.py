"""
phase4/concept_masks.py

Reads concept_tags from the Phase 1 CSV and builds binary masks
for each concept type: EMOTION, MODALITY, NEGATION.

Each mask is a list of 0s and 1s, same length as the token list.
A 1 means that token is tagged with that concept.

Example:
  tokens       = ["the", "president", "never", "signed", "crisis"]
  concept_tags = ["O",   "O",         "NEGATION", "O",   "EMOTION"]

  emotion_mask  = [0, 0, 0, 0, 1]
  modality_mask = [0, 0, 0, 0, 0]
  negation_mask = [0, 0, 1, 0, 0]
"""

import ast
import torch
import pandas as pd


def parse_tags(tag_str):
    """
    Converts the concept_tags string from CSV back into a Python list.
    The CSV stores it as a string like "['O', 'NEGATION', 'O']"
    ast.literal_eval safely converts it back to a real list.
    """
    if isinstance(tag_str, list):
        return tag_str
    return ast.literal_eval(tag_str)


def build_masks_from_tags(concept_tags):
    """
    Given a list of concept tag strings, returns three binary lists:
    one for EMOTION, one for MODALITY, one for NEGATION.

    Args:
        concept_tags: list of strings, e.g. ["O", "EMOTION", "NEGATION", "O"]

    Returns:
        emotion_mask  : list of int (0 or 1), length = len(concept_tags)
        modality_mask : list of int (0 or 1)
        negation_mask : list of int (0 or 1)
    """
    emotion_mask  = [1 if t == "EMOTION"  else 0 for t in concept_tags]
    modality_mask = [1 if t == "MODALITY" else 0 for t in concept_tags]
    negation_mask = [1 if t == "NEGATION" else 0 for t in concept_tags]
    return emotion_mask, modality_mask, negation_mask


def masks_to_tensors(emotion_mask, modality_mask, negation_mask, device):
    """
    Converts the three binary lists into PyTorch float tensors.
    Shape: (sequence_length,)

    We use float32 because these masks are multiplied with attention
    weights (which are floats) inside the model.
    """
    e = torch.tensor(emotion_mask,  dtype=torch.float32).to(device)
    m = torch.tensor(modality_mask, dtype=torch.float32).to(device)
    n = torch.tensor(negation_mask, dtype=torch.float32).to(device)
    return e, m, n


def load_masks_for_split(csv_path, max_len):
    """
    Loads ALL concept masks from a split CSV (e.g. train_split_L128.csv
    merged with concept tags from liar_concepts_step6_L128.csv).

    Actually: the concept_tags column is already in liar_concepts_step6_L{L}.csv
    and step7 splits that file. So the split CSVs already have concept_tags.

    Returns:
        emotion_masks  : list of lists  (one per row)
        modality_masks : list of lists
        negation_masks : list of lists
    """
    df = pd.read_csv(csv_path)
    df["concept_tags"] = df["concept_tags"].apply(parse_tags)

    emotion_masks, modality_masks, negation_masks = [], [], []

    for tags in df["concept_tags"]:
        # Pad or truncate to max_len so all masks are the same length
        tags = tags[:max_len]
        pad_len = max_len - len(tags)
        tags = tags + ["O"] * pad_len  # pad with "O" (no concept)

        e, m, n = build_masks_from_tags(tags)
        emotion_masks.append(e)
        modality_masks.append(m)
        negation_masks.append(n)

    return emotion_masks, modality_masks, negation_masks


def masks_to_batch_tensor(mask_list, device):
    """
    Converts a list of mask lists into a single 2D tensor.
    Shape: (batch_size, sequence_length)

    This is what gets passed into the model's forward() call.
    """
    return torch.tensor(mask_list, dtype=torch.float32).to(device)
