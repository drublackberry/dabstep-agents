#!/usr/bin/env python3

import yaml
from utils.tracing import setup_smolagents_tracing

import argparse
import logging
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import datasets
import pandas as pd
# from smolagents.utils import console
from constants import REPO_ID
from agents.prompts import (
    reasoning_llm_system_prompt,
    reasoning_llm_task_prompt,
    chat_llm_task_prompt,
    chat_llm_system_prompt
)
from utils.dabstep_utils import (
    get_tasks_to_run,
    append_answer,
    append_console_output,
    download_context, 
    evaluate
)
from utils.execution import TqdmLoggingHandler, get_env, validate_reasoning_model_compatibility
from agents.code_agents import ReasoningCodeAgent, ChatCodeAgent

logging.basicConfig(level=logging.WARNING, handlers=[TqdmLoggingHandler()])
logger = logging.getLogger(__name__)


def parse_args():
    # Load environment configuration first
    try:
        env_config = get_env()
    except ValueError as e:
        logger.warning(f"Environment configuration error: {e}")
        env_config = {}
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--model-id", type=str, default=env_config.get("MODEL", "openai/o3-mini"))
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--max-tasks", type=int, default=-1)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--tasks-ids", type=int, nargs="+", default=None)
    parser.add_argument("--api-base", type=str, default=env_config.get("BASE_URL"))
    parser.add_argument("--api-key", type=str, default=env_config.get("API_KEY"))
    parser.add_argument("--hf_token", type=str, default=env_config.get("HF_TOKEN"))
    parser.add_argument("--llm-gateway", type=str, default=env_config.get("LLM_GATEWAY"))
    parser.add_argument("--otlp-endpoint", type=str, default=env_config.get("OTLP_ENDPOINT"))
    parser.add_argument("--split", type=str, default="dev", choices=["default", "dev"])
    parser.add_argument("--timestamp", type=str, default=None)
    parser.add_argument("--use-reasoning", action="store_true", help="Use reasoning mode for the agent", default=False)

    return parser.parse_args()


def run_single_task(
        task: dict,
        model_id: str,
        api_base: str,
        api_key: str,
        ctx_path: str,
        base_filename: Path,
        is_dev_data: bool,
        max_steps: int,
        use_reasoning: bool
):
    # Validate model compatibility with use_reasoning parameter
    validate_reasoning_model_compatibility(model_id, use_reasoning)
    
    # Choose agent based on use_reasoning parameter
    if use_reasoning:
        agent = ReasoningCodeAgent(
            model_id=model_id,
            api_base=api_base,
            api_key=api_key,
            max_steps=max_steps,
            ctx_path=ctx_path
        )
        prompt = reasoning_llm_task_prompt.format(
            question=task["question"],
            guidelines=task["guidelines"]
        )
    else:
        agent = ChatCodeAgent(
            model_id=model_id,
            api_base=api_base,
            api_key=api_key,
            max_steps=max_steps,
            ctx_path=ctx_path
        )
        prompt = chat_llm_task_prompt.format(
            ctx_path=ctx_path,
            question=task["question"],
            guidelines=task["guidelines"]
        )

    # with console.capture() as capture:
    answer = agent.run(prompt)

    logger.warning(f"Task id: {task['task_id']}\tQuestion: {task['question']} Answer: {answer}\n{'=' * 50}")

    answer_dict = {"task_id": str(task["task_id"]), "agent_answer": str(answer)}
    answers_file = base_filename / "answers.jsonl"
    logs_file = base_filename / "logs.txt"

    if is_dev_data:
        scores = evaluate(agent_answers=pd.DataFrame([answer_dict]), tasks_with_gt=pd.DataFrame([task]))
        entry = {**answer_dict, "answer": task["answer"], "score": scores[0]["score"], "level": scores[0]["level"]}
        append_answer(entry, answers_file)
    else:
        append_answer(answer_dict, answers_file)


def main():
    args = parse_args()
    endpoint = args.otlp_endpoint or os.getenv("OTLP_ENDPOINT")
    setup_smolagents_tracing(
        endpoint=endpoint,
        enable_tracing=bool(endpoint),
        force_reinit=True
    )
    gateway = (args.llm_gateway or os.getenv("LLM_GATEWAY") or "").strip()

    if gateway and args.api_key:
        provider_specific_env = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "fireworks": "FIREWORKS_API_KEY",
            "huggingface": "HUGGINGFACEHUB_API_TOKEN",
        }
        env_key = provider_specific_env.get(gateway.lower())
        if env_key:
            os.environ[env_key] = args.api_key

    if gateway and args.api_base:
        provider_base_env = {
            "openai": "OPENAI_API_BASE",
            "anthropic": "ANTHROPIC_BASE_URL",
        }
        base_key = provider_base_env.get(gateway.lower())
        if base_key:
            os.environ[base_key] = args.api_base

    if gateway:
        os.environ["LLM_GATEWAY"] = gateway

    normalized_model_id = args.model_id
    if gateway:
        prefix = f"{gateway}/"
        if not normalized_model_id.startswith(prefix):
            normalized_model_id = f"{prefix}{normalized_model_id}"

    logger.warning(f"Starting run with arguments: {args}")

    ctx_path = download_context(str(Path().resolve()), args.hf_token)

    runs_dir = Path().resolve() / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if not args.timestamp else args.timestamp
    base_filename = runs_dir / f"{normalized_model_id.replace('/', '_').replace('.', '_')}/{args.split}/{int(timestamp)}"

    # save config
    os.makedirs(base_filename, exist_ok=True)
    with open(base_filename / "config.yaml", "w", encoding="utf-8") as f:
        if args.use_reasoning:
            args.system_prompt = reasoning_llm_system_prompt
        else:
            args.system_prompt = chat_llm_system_prompt
        args.timestamp = timestamp
        args_dict = vars(args)
        yaml.dump(args_dict, f, default_flow_style=False)

    # Load dataset with user-chosen split
    data = datasets.load_dataset(REPO_ID, name="tasks", split=args.split, download_mode='reuse_dataset_if_exists', token=args.hf_token)

    if args.max_tasks >= 0 and args.tasks_ids is not None:
        logger.error(f"Can not provide {args.max_tasks=} and {args.tasks_ids=} at the same time")
    total = len(data) if args.max_tasks < 0 else min(len(data), args.max_tasks)

    tasks_to_run = get_tasks_to_run(data, total, base_filename, args.tasks_ids)

    for task in tasks_to_run:
        run_single_task(
            task=task, 
            model_id=normalized_model_id,
            api_base=args.api_base, 
            api_key=args.api_key, 
            ctx_path=ctx_path,
            base_filename=base_filename,
            is_dev_data=True, 
            max_steps=args.max_steps,
            use_reasoning=args.use_reasoning)
    

    # with ThreadPoolExecutor(max_workers=args.concurrency) as exe:
    #     futures = [
    #         exe.submit(
    #             run_single_task,
    #            task,
    #            args.model_id,
    #            args.api_base,
    #            args.api_key,
    #            ctx_path,
    #            base_filename,
    #            (args.split == "dev"),
    #            args.max_steps
    #         )
    #         for task in tasks_to_run
    #     ]
    #     for f in tqdm(as_completed(futures), total=len(tasks_to_run), desc="Processing tasks"):
    #         f.result()

    logger.warning("All tasks processed.")


if __name__ == "__main__":
    main()
