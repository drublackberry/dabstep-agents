"""
DABstep utilities - merged from utils.py and scorer.py
Contains data processing, evaluation, and scoring utilities for the DABstep project.
"""

import json
import re
import math
import threading
from typing import Union
from difflib import SequenceMatcher
from tqdm import tqdm
import logging
from huggingface_hub import hf_hub_download
from constants import REPO_ID 
from pathlib import Path
import pandas as pd
import os
from dotenv import load_dotenv


# Threading locks for file operations
append_answer_lock = threading.Lock()
append_console_output_lock = threading.Lock()


# =============================================================================
# FILE AND DATA UTILITIES (from utils.py)
# =============================================================================

def read_only_open(*a, **kw):
    """Restricted open function that only allows read mode."""
    if (len(a) > 1 and isinstance(a[1], str) and a[1] != 'r') or kw.get('mode', 'r') != 'r':
        raise Exception("Only mode='r' allowed for the function open")
    return open(*a, **kw)


def download_context(base_dir: str, hf_token: str = None) -> str:
    """Download context files from HuggingFace dataset."""
    ctx_files = [
        "data/context/acquirer_countries.csv",
        "data/context/payments.csv",
        "data/context/merchant_category_codes.csv",
        "data/context/fees.json",
        "data/context/merchant_data.json",
        "data/context/manual.md",
        "data/context/payments-readme.md"
    ]
    for f in ctx_files:
        hf_hub_download(REPO_ID, repo_type="dataset", filename=f, local_dir=base_dir, token=hf_token)

    ctx_dir = Path(ctx_files[0]).parent
    return str(ctx_dir)


def get_tasks_to_run(data, total: int, base_filename: Path, tasks_ids: list[int]):
    """Get tasks that haven't been completed yet."""
    import json
    f = base_filename.parent / f"{base_filename.stem}_answers.jsonl"
    done = set()
    if f.exists():
        with open(f, encoding="utf-8") as fh:
            done = {json.loads(line)["task_id"] for line in fh if line.strip()}

    tasks = []
    for i in range(total):
        task_id = int(data[i]["task_id"])
        if task_id not in done:
            if tasks_ids is not None:
                if task_id in tasks_ids:
                    tasks.append(data[i])
            else:
                tasks.append(data[i])
    return tasks


def append_answer(entry: dict, jsonl_file: Path) -> None:
    """Thread-safe append of answer to JSONL file."""
    jsonl_file.parent.mkdir(parents=True, exist_ok=True)
    with append_answer_lock, open(jsonl_file, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(entry) + "\n")


def append_console_output(captured_text: str, txt_file: Path) -> None:
    """Thread-safe append of console output to text file."""
    txt_file.parent.mkdir(parents=True, exist_ok=True)
    with append_console_output_lock, open(txt_file, "a", encoding="utf-8") as fp:
        fp.write(captured_text + "\n")


def evaluate(agent_answers: pd.DataFrame, tasks_with_gt: pd.DataFrame, submission_id: str = ""):
    """Evaluate agent answers against ground truth using question scorer."""
    task_scores = []
    for index, row in tasks_with_gt.iterrows():
          correct_answer = row["answer"]
          level = str(row["level"])
          task_id = str(row["task_id"])

          if task_id not in agent_answers["task_id"].values:
              raise KeyError(f"Task ID: {task_id} not found. Are you sure you submitted the correct file?")

          agent_answer = agent_answers.loc[agent_answers.task_id == task_id, "agent_answer"].values[0]
          score = question_scorer(agent_answer, correct_answer)

          task_scores.append(
              {
                  "submission_id": submission_id,
                  "task_id": task_id,
                  "score": score,
                  "level": level,
                  "agent_answer": agent_answer,
              }
          )

    return task_scores


# =============================================================================
# SCORING UTILITIES (from scorer.py)
# =============================================================================

def is_numeric_with_commas(value: str) -> bool:
    """Check if the string is a number with comma separators."""
    return bool(re.match(r'^\$?(\d{1,3}(,\d{3})*(\.\d+)?|\.\d+)$', value.strip()))


def question_scorer(input1: str, input2: str) -> bool:
    """
    Main scoring function that compares two inputs and returns True if they match.
    Handles numeric values, lists, and string comparisons.
    """
    # Remove leading/trailing whitespace and convert to lowercase
    input1 = input1.strip().lower()
    input2 = input2.strip().lower()

    # Check if inputs are numeric with commas
    if is_numeric_with_commas(input1) or is_numeric_with_commas(input2):
        num1 = extract_numeric(input1)
        num2 = extract_numeric(input2)
        return compare_numeric(num1, num2) if num1 is not None and num2 is not None else False

    # Check for list match
    if ';' in input1 or ';' in input2 or ',' in input1 or ',' in input2:
        return compare_lists(input1, input2)

    # Extract numeric values if present
    num1 = extract_numeric(input1)
    num2 = extract_numeric(input2)

    # If both inputs have numeric values, compare them
    if num1 is not None and num2 is not None:
        return compare_numeric(num1, num2)

    # Check for string match or subset
    return compare_strings(input1, input2)


def extract_numeric(value: str) -> Union[float, None]:
    """Extract numeric value from string, handling commas and currency symbols."""
    # Remove commas and currency symbols from the value string
    value = value.replace(',', '').replace('$', '')
    
    # Extract the first occurrence of a numeric value (including percentages and leading decimal point)
    match = re.search(r'(\d*\.\d+|\d+\.?\d*)%?', value)
    if match:
        num_str = match.group(1)
        try:
            return float(num_str)
        except ValueError:
            return None
    return None


def compare_numeric(num1: float, num2: float) -> bool:
    """Compare two numeric values with appropriate tolerance."""
    # Check for exact equality first
    if num1 == num2:
        return True

    # For percentages and small numbers, use a more lenient comparison
    if num1 < 1 and num2 < 1:
        return math.isclose(num1, num2, rel_tol=1e-2, abs_tol=1e-4)

    # For larger numbers, use the original comparison method
    dec_places1 = len(str(num1).split('.')[-1]) if '.' in str(num1) else 0
    dec_places2 = len(str(num2).split('.')[-1]) if '.' in str(num2) else 0
    round_to = min(dec_places1, dec_places2)
    rounded1 = round(num1, round_to)
    rounded2 = round(num2, round_to)

    if rounded1 == rounded2:
        return True

    return math.isclose(num1, num2, rel_tol=1e-2, abs_tol=1e-2)


def compare_strings(str1: str, str2: str) -> bool:
    """Compare two strings with fuzzy matching and subset logic."""
    # Remove all whitespace and punctuation
    clean1 = re.sub(r'[^\w]', '', str1)
    clean2 = re.sub(r'[^\w]', '', str2)
    
    if clean1 == clean2:
        return True

    words1 = re.findall(r'\b\w+\b', str1.lower())
    words2 = re.findall(r'\b\w+\b', str2.lower())

    # Only do subset comparison if neither list is empty
    if (len(words1) == 1 or len(words2) == 1) and words1 and words2:
        return set(words1).issubset(set(words2)) or set(words2).issubset(set(words1))

    # Use similarity score for fuzzy matching
    similarity = SequenceMatcher(None, str1, str2).ratio()
    return similarity > 0.95


def compare_lists(list1: str, list2: str) -> bool:
    """Compare two list-like strings, handling different separators and ordering."""
    # Normalize list representations by removing brackets
    list1 = re.sub(r'^\[|\]$', '', list1.strip())
    list2 = re.sub(r'^\[|\]$', '', list2.strip())

    # Split the lists and remove whitespace
    items1 = [item.strip() for item in re.split(r'[,;]', list1) if item.strip()]
    items2 = [item.strip() for item in re.split(r'[,;]', list2) if item.strip()]

    # Sort the items to handle different order
    items1.sort()
    items2.sort()

    # Check if the lists are identical
    if items1 == items2:
        return True

    # If lists are not identical, compare each item
    if len(items1) != len(items2):
        return False

    for item1, item2 in zip(items1, items2):
        if not question_scorer(item1, item2):
            return False

    return True
