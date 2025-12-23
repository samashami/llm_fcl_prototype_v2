# src/policy/lmss_api.py
import os, json
from typing import Dict, Any
from openai import OpenAI

STRATEGY_PALETTE = {
    0: {"name": "Conservative", "k": 2, "lr_scale": 0.8, "replay_ratio": 0.3, "desc": "Low-impact learning to maintain stability."},
    1: {"name": "Standard",     "k": 2, "lr_scale": 1.0, "replay_ratio": 0.5, "desc": "Balanced progression."},
    2: {"name": "Consolidate",  "k": 3, "lr_scale": 0.9, "replay_ratio": 0.7, "desc": "High replay to combat forgetting."},
    3: {"name": "Aggressive",   "k": 4, "lr_scale": 1.2, "replay_ratio": 0.4, "desc": "Push accuracy if stalled."},
    4: {"name": "Recover",      "k": 2, "lr_scale": 0.5, "replay_ratio": 0.6, "desc": "Stabilize if divergence/loss bad."},
}

def get_fallback_action(n_clients: int, policy_source: str = "LMSS_FALLBACK") -> Dict[str, Any]:
    strat = STRATEGY_PALETTE[1]
    return {
        "client_selection_k": int(strat["k"]),
        "aggregation": {"method": "FedAvg"},
        "client_params": [
            {"id": i, "replay_ratio": float(strat["replay_ratio"]), "lr_scale": float(strat["lr_scale"]), "ewc_lambda": 0.0}
            for i in range(n_clients)
        ],
        "policy_source": policy_source,
    }

def lmss_decide_action_api(state: Dict[str, Any], compact_state_fn, model: str = "gpt-4o-mini") -> Dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY")
    n_clients = len(state["clients"])
    if not api_key:
        return get_fallback_action(n_clients, policy_source="LMSS_NO_API_KEY_FALLBACK")

    client = OpenAI(api_key=api_key)
    s_small = compact_state_fn(state)

    palette_text = "\n".join([f"{k}: {v['name']} - {v['desc']}" for k, v in STRATEGY_PALETTE.items()])

    prompt = f"""You are the Meta-Optimizer for a Federated Continual Learning system.
Analyze the current training state and select the best Strategy ID from the palette.

[STATE]:
{json.dumps(s_small)}

[STRATEGY PALETTE]:
{palette_text}

Respond in JSON format:
{{
  "reasoning": "one sentence",
  "strategy_id": int
}}"""

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        res = json.loads(resp.choices[0].message.content)
        strat_id = int(res.get("strategy_id", 1))
        reasoning = res.get("reasoning", "")

        strat = STRATEGY_PALETTE.get(strat_id, STRATEGY_PALETTE[1])

        print(f"\n[LMSS] strategy_id={strat_id} ({strat['name']}) | {reasoning}\n", flush=True)

        return {
            "client_selection_k": int(strat["k"]),
            "aggregation": {"method": "FedAvg"},
            "client_params": [
                {"id": i, "replay_ratio": float(strat["replay_ratio"]), "lr_scale": float(strat["lr_scale"]), "ewc_lambda": 0.0}
                for i in range(n_clients)
            ],
            "policy_source": f"LMSS_{model}_STRAT_{strat_id}",
        }
    except Exception as e:
        print(f"⚠️ LMSS API error: {e}", flush=True)
        return get_fallback_action(n_clients, policy_source="LMSS_API_ERROR_FALLBACK")