#!/usr/bin/env python3
"""
Script to analyze scores from the latest run.
Usage: python scripts/show-scores.py [run_directory]
"""

import json
import sys
from pathlib import Path
from typing import List, Dict


def find_latest_run(runs_dir: Path) -> Path:
    """Find the most recent run directory containing answers.jsonl"""
    answers_files = list(runs_dir.rglob("answers.jsonl"))
    
    if not answers_files:
        raise FileNotFoundError(f"No answers.jsonl files found in {runs_dir}")
    
    # Sort by modification time and get the most recent
    latest_file = max(answers_files, key=lambda p: p.stat().st_mtime)
    return latest_file.parent


def load_answers(answers_file: Path) -> List[Dict]:
    """Load answers from JSONL file"""
    answers = []
    with open(answers_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                answers.append(json.loads(line))
    return answers


def analyze_scores(answers: List[Dict]) -> None:
    """Analyze and print scores"""
    if not answers:
        print("❌ No answers found!")
        return
    
    # Filter answers that have scores (dev split)
    scored_answers = [a for a in answers if 'score' in a]
    
    if not scored_answers:
        print("❌ No scores found. This may not be a dev split run.")
        print(f"Found {len(answers)} answers without scores.")
        return
    
    # Calculate statistics
    scores = [a['score'] for a in scored_answers]
    avg_score = sum(scores) / len(scores)
    
    # Count by level
    easy_tasks = [a for a in scored_answers if a.get('level') == 'easy']
    hard_tasks = [a for a in scored_answers if a.get('level') == 'hard']
    
    easy_correct = sum(1 for a in easy_tasks if a['score'] == 1)
    hard_correct = sum(1 for a in hard_tasks if a['score'] == 1)
    
    # Print summary
    print("=" * 60)
    print("📊 SCORE ANALYSIS")
    print("=" * 60)
    print(f"\n✨ Average Score: {avg_score:.2%} ({sum(scores)}/{len(scores)} correct)")
    print(f"\n📝 Total Tasks: {len(scored_answers)}")
    
    if easy_tasks:
        easy_avg = sum(a['score'] for a in easy_tasks) / len(easy_tasks)
        print(f"  • Easy: {easy_correct}/{len(easy_tasks)} ({easy_avg:.2%})")
    
    if hard_tasks:
        hard_avg = sum(a['score'] for a in hard_tasks) / len(hard_tasks)
        print(f"  • Hard: {hard_correct}/{len(hard_tasks)} ({hard_avg:.2%})")
    
    # Print individual results
    print("\n" + "=" * 60)
    print("📋 INDIVIDUAL TASK SCORES")
    print("=" * 60)
    
    for i, answer in enumerate(scored_answers, 1):
        task_id = answer.get('task_id', 'unknown')
        score = answer['score']
        level = answer.get('level', 'unknown').upper()
        status = "✅" if score == 1 else "❌"
        
        print(f"{i:2d}. {status} Task {task_id:4s} | {level:4s} | Score: {score}")
    
    print("=" * 60)


def main():
    # Determine run directory
    if len(sys.argv) > 1:
        run_dir = Path(sys.argv[1])
        if not run_dir.exists():
            print(f"❌ Error: Directory not found: {run_dir}")
            sys.exit(1)
    else:
        # Find latest run automatically
        runs_base = Path(__file__).parent.parent / "runs"
        if not runs_base.exists():
            print(f"❌ Error: Runs directory not found: {runs_base}")
            sys.exit(1)
        
        try:
            run_dir = find_latest_run(runs_base)
            print(f"📁 Latest run found: {run_dir.relative_to(runs_base.parent)}")
        except FileNotFoundError as e:
            print(f"❌ Error: {e}")
            sys.exit(1)
    
    # Load and analyze answers
    answers_file = run_dir / "answers.jsonl"
    if not answers_file.exists():
        print(f"❌ Error: answers.jsonl not found in {run_dir}")
        sys.exit(1)
    
    answers = load_answers(answers_file)
    analyze_scores(answers)


if __name__ == "__main__":
    main()
