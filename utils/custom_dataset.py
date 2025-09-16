import json
import os
from io import BytesIO
from typing import Dict, List, Tuple, Optional

from datasets import Dataset, DatasetDict


def _scene_to_paths(data_root: str, scene_id: str, prefer_idx: str, non_prefer_idx: str) -> Tuple[str, str]:
    """
    Given a scene_id like "D1-00001" and style indices as strings, return absolute
    filepaths to preferred and non-preferred images following:
      {data_root}/images/{folder_id}/{scene_id}_{style_id}.jpg
    """
    folder_id = scene_id.split("-")[0]
    pref = os.path.join(data_root, "images", folder_id, f"{scene_id}_{prefer_idx}.jpg")
    nonpref = os.path.join(data_root, "images", folder_id, f"{scene_id}_{non_prefer_idx}.jpg")
    return pref, nonpref


def _read_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def _build_examples(
    data_root: str,
    user_json_path: str,
    caption_default: str = "",
) -> List[Dict]:
    """
    Build a list of dict examples with keys: 'jpg_0', 'jpg_1', 'label_0', 'caption'.
    We always order jpg_0 as preferred and jpg_1 as non-preferred, setting label_0 = 1.
    """
    with open(user_json_path, "r") as f:
        responses = json.load(f)

    examples: List[Dict] = []
    for scene_id, choice in responses.items():
        prefer_idx = str(choice.get("prefer"))
        non_prefer_idx = str(choice.get("non_prefer"))
        if prefer_idx is None or non_prefer_idx is None:
            # skip malformed entries
            continue
        pref_path, nonpref_path = _scene_to_paths(data_root, scene_id, prefer_idx, non_prefer_idx)
        if not (os.path.isfile(pref_path) and os.path.isfile(nonpref_path)):
            # skip missing files
            continue
        try:
            jpg_0 = _read_bytes(pref_path)
            jpg_1 = _read_bytes(nonpref_path)
        except Exception:
            # Skip unreadable files
            continue

        examples.append(
            {
                "jpg_0": jpg_0,  # preferred
                "jpg_1": jpg_1,  # non-preferred
                "label_0": 1,    # first image is the preferred one
                "caption": caption_default,
            }
        )

    return examples


def build_user_pairwise_dataset(
    data_root: str,
    user_json_path: str,
    n_train: int = 60,
    seed: Optional[int] = 42,
    caption_default: str = "",
) -> DatasetDict:
    """
    Create a HuggingFace DatasetDict with 'train' and 'test' from a user's preference JSON.

    - data_root: path to dataset root containing 'images/' and 'responses/'
    - user_json_path: path to a specific user's responses JSON file
    - n_train: number of samples to include in the train split (default 60)
    - seed: shuffle seed for reproducibility
    - caption_default: default caption string to use for all items
    """
    all_examples = _build_examples(data_root, user_json_path, caption_default=caption_default)
    if not all_examples:
        raise ValueError(
            f"No valid examples found. data_root={data_root}, user_json_path={user_json_path}"
        )

    # Deterministic shuffle then split
    if seed is not None:
        import random

        rnd = random.Random(seed)
        rnd.shuffle(all_examples)

    n_train = max(0, min(n_train, len(all_examples)))
    train_list = all_examples[:n_train]
    test_list = all_examples[n_train:]

    dsd = {}
    dsd["train"] = Dataset.from_list(train_list)
    # If test_list empty, still create split for compatibility
    dsd["test"] = Dataset.from_list(test_list) if test_list else Dataset.from_list([])
    return DatasetDict(dsd)

