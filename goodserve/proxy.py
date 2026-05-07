import asyncio
import logging
import httpx
import json
import time
import random
import os
from pathlib import Path
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List, Union
from fastapi import FastAPI, Request
from starlette.responses import JSONResponse
import uvicorn
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import joblib
from dataclasses import dataclass
from bert import DistilBertLengthPredictor  
from mlp import MLPPredictor  

os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

RESULTS_FILE = "/path/to/results/experiment_results.txt"

class LazyLengthPredictor:
    _distribution_cache: Dict[str, dict] = {}
    
    def __init__(self, 
                 data_path: str = "/path/to/dataset/all_data.jsonl",
                 bin_size: int = 10):
        self.data_path = Path(data_path)
        self.bin_size = bin_size
        self.bins_list: List[int] = []
        self.probs_list: List[float] = []
        
        cache_key = f"{data_path}_{bin_size}"
        if cache_key in self._distribution_cache:
            cached_data = self._distribution_cache[cache_key]
            self.bins_list = cached_data['bins']
            self.probs_list = cached_data['probs']
            
        else:
            self._load_distribution()
            self._distribution_cache[cache_key] = {
                'bins': self.bins_list,
                'probs': self.probs_list
            }
    
    def _load_distribution(self) -> None:
        if not self.data_path.exists():
            raise FileNotFoundError(f"not found: {self.data_path}")
        
        length_bins = defaultdict(int)
        total_samples = 0
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    token_num = data.get('token_number')
                    if token_num is None:
                        continue
                    bin_idx = token_num // self.bin_size
                    length_bins[bin_idx] += 1
                    total_samples += 1
                except (json.JSONDecodeError, KeyError):
                    continue
        
        if total_samples == 0:
            raise ValueError("no data")
        
        sorted_bins = sorted(length_bins.items())
        self.bins_list = [bin_idx for bin_idx, _ in sorted_bins]
        self.probs_list = [count / total_samples for _, count in sorted_bins]
        
    def predict_single(self, seed: Optional[int] = None) -> int:
        if not self.bins_list:
            raise RuntimeError("error")
        
        local_random = random.Random(seed)
        chosen_bin = local_random.choices(
            population=self.bins_list,
            weights=self.probs_list,
            k=1
        )[0]
        
        min_length = chosen_bin * self.bin_size
        max_length = min_length + self.bin_size - 1
        predicted_length = local_random.randint(min_length, max_length)
        return predicted_length

logging.basicConfig(level=logging.INFO, format='%(asctime)s - Proxy - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Expert(nn.Module):
    def __init__(self, input_dim):
        dim = 512 if input_dim > 5000 else 1024
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_dim, dim)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(dim, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class InputAwareRouter(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(InputAwareRouter, self).__init__()
        self.fc1 = nn.Linear(input_dim + 1, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, num_experts)

    def forward(self, x, input_texts):
        input_lengths = torch.tensor([[len(text)] for text in input_texts], 
                                     dtype=torch.float32).to(x.device)
        normalized_lengths = input_lengths / 1000.0
        x_with_length = torch.cat([x, normalized_lengths], dim=1)
        
        x = F.relu(self.bn1(self.fc1(x_with_length)))
        x = F.relu(self.bn2(self.fc2(x)))
        logits = self.fc3(x)

        return F.softmax(logits, dim=1)

class DynamicMoE(nn.Module):
    def __init__(self, experts, router):
        super(DynamicMoE, self).__init__()
        self.experts = nn.ModuleList(experts)
        self.router = router

    def forward(self, x, input_texts):
        routing_weights = self.router(x, input_texts)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        final_output = torch.sum(routing_weights.unsqueeze(-1) * expert_outputs, dim=1)
        return final_output

class LengthPredictor:
    def __init__(self, model_dir, num_experts=9):
        self.device = torch.device("cuda") 
        print(f"[LengthPredictor] Initializing from {model_dir} on GPU...")
        
        try:
            self.vectorizer = joblib.load(os.path.join(model_dir, "dynamic_moe_vectorizer.pkl"))
            expert0_state = torch.load(os.path.join(model_dir, "dynamic_moe_expert_0.pth"), map_location=self.device)
            input_dim = expert0_state['fc1.weight'].shape[1]
            
            experts = []
            for i in range(num_experts):
                expert = Expert(input_dim)
                expert.load_state_dict(torch.load(os.path.join(model_dir, f"dynamic_moe_expert_{i}.pth"), map_location=self.device))
                expert.eval()
                experts.append(expert)
            
            router = InputAwareRouter(input_dim, num_experts)
            router.load_state_dict(torch.load(os.path.join(model_dir, "dynamic_moe_router.pth"), map_location=self.device))
            router.eval()
            
            self.model = DynamicMoE(experts, router)
            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            print("[LengthPredictor] Loaded successfully.")
            
        except Exception as e:
            print(f"[LengthPredictor] FAILED to load: {e}")
            self.is_loaded = False

    def predict_batch(self, prompts: List[str], tokens_generated_list: List[int] = None) -> List[int]:
        """批量预测剩余长度"""
        if not self.is_loaded or not prompts:
            return [0] * len(prompts)
            
        if tokens_generated_list is None:
            tokens_generated_list = [0] * len(prompts)
            
        try:
            prompt_vecs = self.vectorizer.transform(prompts).toarray()
            
            tokens_gen_arr = np.array(tokens_generated_list, dtype=np.float32).reshape(-1, 1)
            feature_vecs = np.concatenate([prompt_vecs, tokens_gen_arr], axis=1)
            
            feature_tensor = torch.tensor(feature_vecs, dtype=torch.float32).to(self.device)

            with torch.no_grad():
                predicted_log = self.model(feature_tensor, prompts) 
                predicted_remaining = torch.expm1(predicted_log).squeeze(-1).cpu().tolist()
                
            if not isinstance(predicted_remaining, list):
                predicted_remaining = [predicted_remaining]
                
            return [int(max(0, val)) for val in predicted_remaining]
            
        except Exception as e:
            logger.error(f"Batch predict error: {e}")
            return [0] * len(prompts)

    def predict(self, prompt: str, tokens_generated: int = 0) -> int:
        return self.predict_batch([prompt], [tokens_generated])[0]

class StatsState:
    def __init__(self):
        self.stats = defaultdict(lambda: {
            "total": 0, 
            "violations": 0,
            "start_time": None,
            "end_time": None
        })
        self.file_lock = asyncio.Lock()

stats_state = StatsState()

async def flush_stats_to_file():
    async with stats_state.file_lock:
        try:
            sorted_scales = sorted(stats_state.stats.keys(), key=lambda x: float(x) if str(x).replace('.', '', 1).isdigit() else 999)
            with open(RESULTS_FILE, "w", encoding="utf-8") as f:
                f.write("Scale,Total_Requests,Violations,Success_Count,Duration_Sec,Goodput_RPS\n")
                for scale in sorted_scales:
                    data = stats_state.stats[scale]
                    total = data['total']
                    violations = data['violations']
                    success_count = total - violations
                    
                    duration = 0
                    if data['start_time'] is not None and data['end_time'] is not None:
                        duration = data['end_time'] - data['start_time']
                    
                    goodput = 0
                    if duration > 0:
                        goodput = success_count / duration
                    
                    f.write(f"{scale},{total},{violations},{success_count},{duration:.4f},{goodput:.4f}\n")
        except Exception as e:
            logger.error(e)


class TrieNode:
    def __init__(self):
        self.children = {}

class PrefixCacheTracker:

    def __init__(self, max_tokens_per_server=100000):
        # server_name -> Trie Root
        self.roots = defaultdict(TrieNode)
        self.max_tokens = max_tokens_per_server
        
    def record_request(self, server_name: str, token_ids: List[int]):
       
        if not token_ids:
            return
        
        node = self.roots[server_name]
        for token in token_ids:
            if token not in node.children:
                if len(node.children) > 1000: 
                    break 
                node.children[token] = TrieNode()
            node = node.children[token]
            
    def get_max_prefix_match(self, server_name: str, token_ids: List[int]) -> int:
        if server_name not in self.roots or not token_ids:
            return 0
        
        node = self.roots[server_name]
        match_len = 0
        for token in token_ids:
            if token in node.children:
                match_len += 1
                node = node.children[token]
            else:
                break
        return match_len

prefix_tracker = PrefixCacheTracker()

@dataclass
class ServerConfig:
    base_url: str
    price_per_hour: float
    tpot_ms: float         
    name: str
    tier: int                
    queueing_time: float = 0.0
    waiting_request_num: int = 0
    gpu_memory_usage: float = 0.0
    gamma_g: float = 2.0     
    current_w_g: float = 0.0 
    q_hat_g: float = 0.0     

SERVER_POOLS = [
    ServerConfig(base_url="http://backend_host_1:8070", price_per_hour=1.0, tpot_ms=10.0, name="GPU_Type_A", tier=0, gamma_g=2.5),
    ServerConfig(base_url="http://backend_host_2:8071", price_per_hour=3.0, tpot_ms=10.0, name="GPU_Type_B", tier=1, gamma_g=0.8),
    ServerConfig(base_url="http://backend_host_3:8072", price_per_hour=5.0, tpot_ms=10.0, name="GPU_Type_C", tier=2, gamma_g=1.5),
    ServerConfig(base_url="http://backend_host_4:8073", price_per_hour=8.0, tpot_ms=19.0, name="GPU_Type_D", tier=3, gamma_g=0.5),
]

MAX_MIGRATIONS = 10
REQUEST_STATES: Dict[str, Any] = {}

async def update_single_server_stats(client: httpx.AsyncClient, server: ServerConfig):
    try:
        url = f"{server.base_url}/custom/stats"
        resp = await client.get(url, timeout=0.4)
        
        if resp.status_code == 200:
            data = resp.json()
            new_tpot = data.get("avg_token_latency_ms")
            current_load = data.get("current_load", 0)
            queueing_time = data.get("ttft_prediction_ms", 50)
            waiting_request_num = data.get("current_waiting_requests", 0)
            
            server.gamma_g = float(data.get("avg_prefill_time_per_token_ms", server.gamma_g))
            server.current_w_g = float(data.get("avg_queue_wait_time_ms", queueing_time))
            
            if new_tpot is not None:
                if current_load < 2:
                    default_tpot = {
                        "GPU_Type_A": 30.0, "GPU_Type_B": 11.9, 
                        "GPU_Type_C": 20.8, "GPU_Type_D": 10.1
                    }
                    server.tpot_ms = default_tpot.get(server.name, server.tpot_ms)
                    server.queueing_time = 50
                    server.waiting_request_num = 0
                else:   
                    server.tpot_ms = float(new_tpot)
                    server.queueing_time = queueing_time
                    server.waiting_request_num = waiting_request_num
                    
            server.gpu_memory_usage = data.get("gpu_cache_usage", 0.0)
            
    except Exception as e:
        pass

async def monitor_servers_loop():
    logger.info("Starting background server stats monitor...")
    async with httpx.AsyncClient() as client:
        while True:
            start_ts = time.time()
            tasks = [update_single_server_stats(client, server) for server in SERVER_POOLS]
            await asyncio.gather(*tasks)
            elapsed = time.time() - start_ts
            sleep_time = max(0.0, 0.5 - elapsed)
            await asyncio.sleep(sleep_time)

length_predictor = None 
MLP_MODEL_DIR = "/path/to/models/mlp_baseline"

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
        with open(RESULTS_FILE, "w") as f:
            f.write("Scale,Total_Requests,Violations,Success_Count,Duration_Sec,Goodput_RPS\n")
    except Exception as e:
        logger.error(e)

    monitor_task = asyncio.create_task(monitor_servers_loop())
        
    try:

        global length_predictor
        length_predictor = MLPPredictor(MLP_MODEL_DIR)
        
        if length_predictor.is_loaded:
            logger.info("successfully loaded length predictor model")
        else:
            logger.error("(is_loaded=False)")
    except Exception as e:
        logger.error(e)
        
    global length_predictor_lazy
    length_predictor_lazy = LazyLengthPredictor()
    
    yield
    logger.info("close")

app = FastAPI(lifespan=lifespan)

def select_best_server(
    token_ids: List[int],   
    prompt_len: int,        
    remaining_tokens: int,  
    deadline_ms: float,     
    exclude_server: str = None 
) -> Optional[ServerConfig]:
    
    alpha = 0.8 
    
    # 2: C \leftarrow \emptyset
    C: List[ServerConfig] = []
    
    available_gpus = [s for s in SERVER_POOLS if s.name != exclude_server] if exclude_server else SERVER_POOLS
    if not available_gpus:
        available_gpus = SERVER_POOLS 
        
    T_hat_dict = {} 

    # 3: for all g \in \mathcal{G} do
    for g in available_gpus:
        # 4: w_g \leftarrow AVGWAITTIME(g)
        w_g = g.current_w_g
        
        # 5: \hat{q}_g \leftarrow \alpha w_g + (1 - \alpha)\hat{q}_g
        g.q_hat_g = alpha * w_g + (1 - alpha) * g.q_hat_g
        q_hat_g = g.q_hat_g
        
        # 6: H_{r,g} \leftarrow REUSEPREFIX(r, g) (Trie 精确匹配)
        H_rg = prefix_tracker.get_max_prefix_match(g.name, token_ids)
        
        # 7: T^{prefill}(r, g) \leftarrow \gamma_g \cdot (L^{in}_r - H_{r,g})
        T_prefill = g.gamma_g * max(0, prompt_len - H_rg)
        
        # 8: \hat{T}(r, g) \leftarrow \hat{q}_g + T^{prefill}(r, g) + \tau_g \cdot \hat{L}_r
        T_hat_rg = q_hat_g + T_prefill + (g.tpot_ms * remaining_tokens)
        T_hat_dict[g] = T_hat_rg
        
        # 9: if \hat{T}(r, g) \le D_r then
        if T_hat_rg <= deadline_ms:
            if g.gpu_memory_usage < 1.0: 
                # 10: C \leftarrow C \cup \{g\}
                C.append(g)
        # 11: end if
    # 12: end for

    # 13: if C \neq \emptyset then
    if C:
        # 14: g^* \leftarrow \arg\max_{g \in C} \tau_g
        g_star = max(C, key=lambda g: g.tpot_ms)
    # 15: else
    else:
        # 16: g^* \leftarrow \arg\min_{g \in \mathcal{G}} (\hat{T}(r, g) - D_r)
        g_star = min(available_gpus, key=lambda g: T_hat_dict.get(g, float('inf')) - deadline_ms)
    # 17: end if
    
    # 18: return g^*
    return g_star


@app.post("/check_migration")
async def check_migration(request: Request):
    try:
        data = await request.json()
    except Exception:
        return {"action": "CONTINUE"}
    
    proxy_request_id = data.get("proxy_request_id") or data.get("request_id")
    
    if not proxy_request_id or proxy_request_id not in REQUEST_STATES:
        logger.warning(f"Unknown ID {proxy_request_id} in check_migration")
        return {"action": "CONTINUE"}

    state = REQUEST_STATES[proxy_request_id]
    
    current_migration_count = state.get('migration_count', 0)
    if current_migration_count >= MAX_MIGRATIONS:
        return {"action": "CONTINUE"}
    
    
    problem = state.get("problem", "")
    
    token_ids = data.get("full_token_ids", [])
    if not token_ids:
        token_ids = [hash(problem[i:i+4]) for i in range(0, len(problem), 4)]
        
    prompt_len = data.get("original_prompt_len", len(token_ids))
    if prompt_len <= 0: prompt_len = len(token_ids)
    remaining_tokens = data.get("remaining_tokens", 1)
    
    now = time.time() * 1000
    remaining_deadline = state['deadline_ms'] - (now - state['start_time_ms'])

    best_server = select_best_server(
        token_ids=token_ids,
        prompt_len=prompt_len,
        remaining_tokens=remaining_tokens,
        deadline_ms=remaining_deadline,
        exclude_server=state['current_server_name']
    )

    if best_server is not None:
        migration_count = current_migration_count + 1
        logger.info(f"[{proxy_request_id}] Decision: MIGRATE #{migration_count} ({state['current_server_name']} -> {best_server.name})")
        
        prefix_tracker.record_request(best_server.name, token_ids)
        
        state['target_base_url'] = best_server.base_url
        state['target_server_name'] = best_server.name
        state['target_tier'] = best_server.tier
        state['migration_count'] = migration_count
        
        new_payload = state['original_payload'].copy()
        new_payload.pop("scale", None) 
        new_payload["prompt"] = data["full_token_ids"] 
        new_payload["proxy_request_id"] = proxy_request_id
        new_payload["request_id"] = proxy_request_id
        new_payload["deadline"] = remaining_deadline
        new_payload["stream"] = False
        new_payload['total_generated_len'] = data['total_generated_len']
        new_payload['max_tokens'] = state['initial_max_tokens'] - data['total_generated_len']
        new_payload["original_prompt_len"] = prompt_len
        
        state['target_payload'] = new_payload
        state['status'] = 'SWITCHING'
        
        if 'migration_event' in state:
            state['migration_event'].set()
        
        return {"action": "MIGRATE"}
    else:
        return {"action": "CONTINUE"}


async def post_with_migration_check(
    client: httpx.AsyncClient,
    url: str,
    payload: dict,
    migration_event: asyncio.Event
) -> tuple[httpx.Response | None, bool]:
    request_task = asyncio.create_task(client.post(url, json=payload))
    
    async def wait_for_migration():
        await migration_event.wait()
        return True
    
    migration_task = asyncio.create_task(wait_for_migration())
    
    try:
        done, pending = await asyncio.wait(
            [request_task, migration_task],
            return_when=asyncio.FIRST_COMPLETED
        )
        
        if migration_task in done:
            request_task.cancel()
            try:
                await request_task
            except asyncio.CancelledError:
                pass
            return None, True
        
        if request_task in done:
            migration_task.cancel()
            try:
                await migration_task
            except asyncio.CancelledError:
                pass
            return request_task.result(), False
            
    except Exception as e:
        for task in [request_task, migration_task]:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        raise e
    
    return None, False



@app.post("/v1/completions")
async def proxy_completions(request: Request):
    req_start_time = time.time()
    
    try:
        req_data = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    proxy_request_id = req_data.get("proxy_request_id") or req_data.get("request_id")
    deadline_ms = req_data.get("deadline", 10000)
    current_scale = req_data.get("scale", 1) 
    problem = req_data.get("problem", "default")
    max_tokens = req_data.get("max_tokens", 512)
    

    token_ids = req_data.get("full_token_ids")
    if not token_ids:
        token_ids = [hash(problem[i:i+4]) for i in range(0, len(problem), 4)]
        
    prompt_len = req_data.get("prompt_len")
    if prompt_len is None:
        prompt_len = len(token_ids)
    if prompt_len <= 0:
        prompt_len = 10
    
    if stats_state.stats[current_scale]["start_time"] is None:
        stats_state.stats[current_scale]["start_time"] = req_start_time
    
    if not proxy_request_id:
        return JSONResponse({"error": "request_id/proxy_request_id required"}, status_code=400)

    start_time = time.time()
    start_time_ms = start_time * 1000
    
    a_constant = 1.001 
    noise_multiplier = random.uniform(1.0 / a_constant, a_constant) if a_constant > 1.0 else 1.0
        
    remaining_tokens = length_predictor.predict(problem) * noise_multiplier
    

    initial_server = select_best_server(
        token_ids=token_ids,
        prompt_len=prompt_len,
        remaining_tokens=remaining_tokens,
        deadline_ms=deadline_ms
    )
    
    if initial_server is None:
        initial_server = SERVER_POOLS[3]
        
    
    prefix_tracker.record_request(initial_server.name, token_ids)
    
    logger.info(f"[{proxy_request_id}] Initial routing selected: {initial_server.name} (tier {initial_server.tier}) | Scale: {current_scale}")
    
    REQUEST_STATES[proxy_request_id] = {
        "problem": problem,
        "status": "RUNNING",
        "start_time_ms": start_time_ms,
        "deadline_ms": deadline_ms,
        "current_base_url": initial_server.base_url,
        "current_server_name": initial_server.name,
        "current_tier": initial_server.tier,
        "original_payload": req_data,
        "target_base_url": None,
        "target_payload": None,
        "target_tier": None,
        "migration_count": 0,
        "migration_history": [initial_server.name],
        "migration_event": asyncio.Event(),
        "initial_max_tokens": max_tokens
    }

    backend_payload = req_data.copy()
    backend_payload["stream"] = False
    backend_payload["proxy_request_id"] = proxy_request_id
    backend_payload["request_id"] = proxy_request_id
    backend_payload["deadline"] = deadline_ms
    backend_payload.pop("scale", None)

    final_response = None
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        current_server = initial_server
        current_payload = backend_payload
        
        while True:
            state = REQUEST_STATES.get(proxy_request_id)
            if not state:
                break
            
            state['status'] = 'RUNNING'
            state['current_base_url'] = current_server.base_url
            state['current_server_name'] = current_server.name
            state['current_tier'] = current_server.tier
            state['migration_event'].clear()
            
            current_url = f"{current_server.base_url}/v1/completions"
            
            try:
                resp, was_migrated = await post_with_migration_check(
                    client,
                    current_url,
                    current_payload,
                    state['migration_event']
                )
                
                if was_migrated:
                    state = REQUEST_STATES.get(proxy_request_id)
                    if state and state['target_base_url']:
                        target_server = next((s for s in SERVER_POOLS if s.base_url == state['target_base_url']), None)
                        if target_server:
                            state['migration_history'].append(target_server.name)
                            current_server = target_server
                            current_payload = state['target_payload']
                            logger.info(f"[{proxy_request_id}] Migrating to {target_server.name} (Migration #{state['migration_count']})")
                            continue
                    
                    logger.error(f"[{proxy_request_id}] Migration triggered but no target info")
                    final_response = {"error": "Migration failed: no target info"}
                    break
                
                if resp is None:
                    final_response = {"error": "No response received"}
                    break
                    
                if resp.status_code != 200:
                    logger.error(f"[{proxy_request_id}] Backend {current_server.name} Error: {resp.status_code}")
                    final_response = {"error": f"Backend Error: {resp.text}"}
                    break
                
                final_response = resp.json()
                break
                
            except asyncio.CancelledError:
                continue
            except httpx.TimeoutException as e:
                final_response = {"error": f"Timeout: {str(e)}"}
                break
            except Exception as e:
                final_response = {"error": str(e)}
                break

    end_time = time.time()
    total_duration_ms = (end_time - start_time) * 1000
    
    state = REQUEST_STATES.get(proxy_request_id, {})
    migration_count = state.get('migration_count', 0)
    migration_history = state.get('migration_history', [])
    
    is_success = total_duration_ms <= deadline_ms
    is_violation = not is_success
    status_msg = "SUCCESS" if is_success else "VIOLATED"
    
    logger.info(
        f"[{proxy_request_id}] Completed (Scale {current_scale}). "
        f"Duration: {total_duration_ms:.2f}ms / Deadline: {deadline_ms}ms. "
        f"Migrations: {migration_count}. Path: {' -> '.join(migration_history)}. "
        f"Result: {status_msg}"
    )
    
    stats_state.stats[current_scale]["total"] += 1
    if is_violation:
        stats_state.stats[current_scale]["violations"] += 1
        
    stats_state.stats[current_scale]["end_time"] = end_time
    await flush_stats_to_file()

    if proxy_request_id in REQUEST_STATES:
        del REQUEST_STATES[proxy_request_id]
    
    if final_response is None:
        final_response = {"error": "No response received"}
    
    if isinstance(final_response, dict):
        final_response["_proxy_stats"] = {
            "total_duration_ms": round(total_duration_ms, 2),
            "deadline_ms": deadline_ms,
            "migration_count": migration_count,
            "migration_history": migration_history,
            "slo_met": is_success,
        }
    
    return JSONResponse(final_response)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)