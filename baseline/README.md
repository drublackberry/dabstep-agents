1. Setup environment
```
export MODEL_ID=openai/o3-mini
export API_KEY=<your_api_key>
pip install -r baseline/requirements.txt
```

2. Launch trace collector
```
python -m phoenix.server.main serve &
```

3. Launch baseline
```
# Runs 10 tasks from dev split
python baseline/run.py --model-id $MODEL_ID --api-key $API_KEY --max-tasks 10 --split dev --concurrency 1

# Run all tasks from the dev split
python baseline/run.py --model-id $MODEL_ID --api-key $API_KEY --max-tasks -1 --split dev --concurrency 1

# Run against 10 tasks from the full benchmark (default) split
python baseline/run.py --model-id $MODEL_ID --api-key $API_KEY --max-tasks 10 --split default --concurrency 1

# Run 10 tasks in parallel
python baseline/run.py --model-id $MODEL_ID --api-key $API_KEY --max-tasks 10 --split default --concurrency 10

# Run specific task ids
python baseline/run.py --model-id $MODEL_ID --api-key $API_KEY --tasks-ids 49 5 1273 --split default --concurrency 3
```