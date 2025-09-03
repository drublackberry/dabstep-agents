import json
from tqdm import tqdm
import logging
import threading
from smolagents import CodeAgent, OpenAIServerModel
from custom_agent import CustomCodeAgent
from custom_litellm import LiteLLMModelWithBackOff
from huggingface_hub import hf_hub_download
from constants import REPO_ID, ADDITIONAL_AUTHORIZED_IMPORTS
from pathlib import Path
from prompts import reasoning_llm_system_prompt, chat_llm_system_prompt
import pandas as pd
from scorer import question_scorer

append_answer_lock = threading.Lock()
append_console_output_lock = threading.Lock()

class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        tqdm.write(self.format(record))

def read_only_open(*a, **kw):
    if (len(a) > 1 and isinstance(a[1], str) and a[1] != 'r') or kw.get('mode', 'r') != 'r':
        raise Exception("Only mode='r' allowed for the function open")
    return open(*a, **kw)

def download_context(base_dir: str, hf_token: str = None) -> str:
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

    root_dir = Path(__file__).resolve().parent.parent
    full_path = Path(base_dir) / Path(ctx_files[0]).parent
    relative_path = full_path.relative_to(root_dir)
    return str(relative_path)

def is_reasoning_llm(model_id: str) -> bool:
    # TODO: expose a list in a YAML file
    reasoning_llm_list = [
        "openai/o1",
        "openai/o3",
        "openai/o3-mini",
        "deepseek/deepseek-reasoner"
    ]
    return model_id in reasoning_llm_list

def get_tasks_to_run(data, total: int, base_filename: Path, tasks_ids: list[int]):
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
    jsonl_file.parent.mkdir(parents=True, exist_ok=True)
    with append_answer_lock, open(jsonl_file, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(entry) + "\n")


def append_console_output(captured_text: str, txt_file: Path) -> None:
    txt_file.parent.mkdir(parents=True, exist_ok=True)
    with append_console_output_lock, open(txt_file, "a", encoding="utf-8") as fp:
        fp.write(captured_text + "\n")

def create_code_agent_with_reasoning_llm(model_id: str, api_base=None, api_key=None, max_steps=10, ctx_path=None):
    agent = CustomCodeAgent(
        system_prompt=reasoning_llm_system_prompt,
        tools=[],
        model=LiteLLMModelWithBackOff(
            model_id=model_id, api_base=api_base, api_key=api_key, max_tokens=None, max_completion_tokens=3000),
        additional_authorized_imports=ADDITIONAL_AUTHORIZED_IMPORTS,
        max_steps=max_steps,
        verbosity_level=3,
    )
    agent.python_executor.static_tools.update({"open": read_only_open})

    agent.system_prompt = agent.system_prompt.format(ctx_path=ctx_path)
    return agent


def create_code_agent_with_chat_llm(model_id: str, api_base=None, api_key=None, max_steps=10):


    # use the default system prompt
    agent = CodeAgent(
        tools=[],
        model=OpenAIServerModel(model_id=model_id, api_base=api_base, api_key=api_key, max_tokens=3000),
        additional_authorized_imports=ADDITIONAL_AUTHORIZED_IMPORTS,
        max_steps=max_steps,
        verbosity_level=3,
        executor_type="local",
    )

    return agent

# ported from leaderboard
def evaluate(agent_answers: pd.DataFrame, tasks_with_gt: pd.DataFrame, submission_id: str = ""):
    task_scores = []
    for index, row in tasks_with_gt.iterrows():
          correct_answer = row["answer"]
          level = str(row["level"])
          task_id = str(row["task_id"])

          if task_id not in agent_answers["task_id"].values:
              raise KeyError(f"Task ID: {task_id} not found. Are you sure you submitted the correct file?")

          agent_answer = agent_answers.loc[agent_answers.task_id == task_id, "agent_answer"].values[0]
          # num_steps = agent_answers.loc[agent_answers.task_id == task_id, "num_steps"].values[0]
          score = question_scorer(agent_answer, correct_answer)

          task_scores.append(
              {
                  "submission_id": submission_id,
                  "task_id": task_id,
                  "score": score,
                  "level": level,
                  "agent_answer": agent_answer,
                  # "num_steps": num_steps,
              }
          )

    return task_scores
