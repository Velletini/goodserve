import json
import asyncio
import aiohttp
import time
import os
import random
import uvicorn 

TARGET_URL = "http://localhost:9000/v1/completions" 
TRACE_FILE = "/path/to/test_data/trace.jsonl"


GLOBAL_BASE_DEADLINE_MS = 1000  

SCALE_OUT = [1]

TIME_SCALE = 0.001  # 1ms 

CONTENT_FILES = [
    "/path/to/dataset/test1.jsonl",
    "/path/to/dataset/test2.jsonl",
    "/path/to/dataset/test3.jsonl",
]
# ===========================================

def get_dataset_name(file_path):
    basename = os.path.basename(file_path)  
    name = os.path.splitext(basename)[0]    
    return name

def load_prompts(file_path):
    dataset_items = []
    if not os.path.exists(file_path):
        return []
    
    dataset_name = get_dataset_name(file_path)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        index = 0 
        for line in f:
            if not line.strip(): continue
            try:
                data = json.loads(line)
                if "prompt" in data:
                    
                    raw_problem = data["prompt"]
                    prompt = raw_problem
                       
                    proxy_request_id = f"{dataset_name}-{index}"
                    
                    dataset_items.append({
                        "formatted_prompt": prompt,
                        "raw_problem": raw_problem,
                        "proxy_request_id": proxy_request_id,
                        "dataset_name": dataset_name,
                        "index": index
                    })
                    
                    index += 1  # 递增索引
            except: continue
    return dataset_items

async def send_request(session, req_id, prompt, problem, proxy_request_id, deadline_ms, scale):
    payload = {
        "prompt": prompt,
        "problem": problem,
        "proxy_request_id": proxy_request_id,  
        "deadline": deadline_ms, 
        "stream": False,
        "scale": scale,
        "temperature": 0,
        "top_p": 0.95,
        "max_tokens": 4096,
        "model": "/path/to/models/llama-model" 
    }
    try:
        async with session.post(TARGET_URL, json=payload, timeout=None) as response:
            if response.status != 200:
                print(f"[Req {req_id}]  Failed: {response.status}")
    except Exception as e:
        print(f"[Req {req_id}]  Error: {e}")

async def run_experiment(scale, trace_timestamps, raw_data_list):

    current_deadline = int(GLOBAL_BASE_DEADLINE_MS * scale)
    print(f"\n>>> 新一轮实验 | Scale: {scale}x | Deadline: {current_deadline}ms <<<")

    queues = [list(d_list) for d_list in raw_data_list if d_list]
    total_items = sum(len(q) for q in queues)
    
    if total_items == 0:
        return

    # Trace 参数
    trace_len = len(trace_timestamps)
    max_trace_time = trace_timestamps[-1]

    async with aiohttp.ClientSession() as session:
        start_wall_time = time.time()
        tasks = []
        sent_count = 0
        
        while sent_count < total_items:

            loop_index = sent_count // trace_len
            trace_index = sent_count % trace_len

            target_ts_ms = trace_timestamps[trace_index] + (loop_index * (max_trace_time + 1))
            

            available_queues = [q for q in queues if len(q) > 0]
            if not available_queues: break
            
            current_queue = random.choice(available_queues)
            item = current_queue.pop(0)
            
            prompt_str = item["formatted_prompt"]
            problem_str = item["raw_problem"]
            proxy_request_id = item["proxy_request_id"]
            
            target_time_sec = target_ts_ms * TIME_SCALE
            current_elapsed = time.time() - start_wall_time
            wait_time = target_time_sec - current_elapsed
            
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            
            task = asyncio.create_task(
                send_request(session, sent_count, prompt_str, problem_str, proxy_request_id, current_deadline, scale)
            )
            tasks.append(task)
            sent_count += 1
            
            if sent_count % 200 == 0:
                print(f"  [Scale {scale}] 进度: {sent_count}/{total_items}")

        await asyncio.gather(*tasks)

        await asyncio.sleep(3)

async def main():
    trace_timestamps = []
    if os.path.exists(TRACE_FILE):
        with open(TRACE_FILE, 'r') as f:
            for line in f:
                try: trace_timestamps.append(json.loads(line)["timestamp"])
                except: pass
    else:
        trace_timestamps = [i * 10 for i in range(1000)]
    trace_timestamps.sort()

    raw_data_list = []
    for path in CONTENT_FILES:
        d_list = load_prompts(path)
        if d_list:
            raw_data_list.append(d_list)
            print(f"  - {os.path.basename(path)}: {len(d_list)} 条")

    if not raw_data_list:
        return

    for scale in SCALE_OUT:
        await run_experiment(scale, trace_timestamps, raw_data_list)

if __name__ == "__main__":
    asyncio.run(main())