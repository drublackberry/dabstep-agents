#!/usr/bin/env python3
"""
Script to view tasks from the dataset.
Usage: python scripts/show-tasks.py [--split dev|default] [--limit 5]
"""

import argparse
import datasets
from constants import REPO_ID
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main():
    parser = argparse.ArgumentParser(description="View tasks from the DABStep dataset")
    parser.add_argument("--split", type=str, default="dev", choices=["dev", "default"],
                       help="Dataset split to view")
    parser.add_argument("--limit", type=int, default=5,
                       help="Number of tasks to show")
    parser.add_argument("--task-id", type=int, default=None,
                       help="Show specific task by ID")
    args = parser.parse_args()
    
    print(f"📚 Loading {args.split} split from {REPO_ID}...")
    data = datasets.load_dataset(REPO_ID, name="tasks", split=args.split, 
                                download_mode='reuse_dataset_if_exists')
    
    print(f"✅ Loaded {len(data)} tasks")
    print("=" * 80)
    
    if args.task_id is not None:
        # Find specific task
        for task in data:
            if int(task['task_id']) == args.task_id:
                print_task(task, detailed=True)
                return
        print(f"❌ Task {args.task_id} not found")
        return
    
    # Show first N tasks
    for i, task in enumerate(data):
        if i >= args.limit:
            break
        print_task(task, detailed=False)
        print("-" * 80)
    
    print(f"\n💡 Total tasks in {args.split} split: {len(data)}")
    print(f"💡 Use --task-id <ID> to see full details of a specific task")


def print_task(task, detailed=False):
    """Print task information"""
    task_id = task.get('task_id', 'unknown')
    level = task.get('level', 'unknown')
    question = task.get('question', 'N/A')
    guidelines = task.get('guidelines', 'N/A')
    answer = task.get('answer', 'N/A')
    
    print(f"\n📝 Task ID: {task_id} | Level: {level.upper()}")
    print(f"Question: {question}")
    
    if detailed:
        print(f"\nGuidelines:\n{guidelines}")
        print(f"\nCorrect Answer: {answer}")
    else:
        # Show truncated versions
        guidelines_short = guidelines[:100] + "..." if len(guidelines) > 100 else guidelines
        print(f"Guidelines: {guidelines_short}")
        if 'answer' in task:
            print(f"Answer: {answer}")


if __name__ == "__main__":
    main()
