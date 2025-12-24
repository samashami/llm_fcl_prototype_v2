# src/policy/lmss_local.py
import os
import json
import re
from typing import Dict, Any, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# Discrete, safe action space (the “palette”)
STRATEGY_PALETTE = {
    0: {"name": "Conservative", "k": 2, "lr_scale": 0.8, "replay_ratio": 0.3, "desc": "Low-impact learning to maintain stability."},
    1: {"name": "Standard",     "k": 2, "lr_scale": 1.0, "replay_ratio": 0.5, "desc": "Balanced progression."},
    2: {"name": "Consolidate",  "k": 3, "lr_scale": 0.9, "replay_ratio": 0.7, "desc": "High replay to combat forgetting."},
    3: {"name": "Aggressive",   "k": 4, "lr_scale": 1.2, "replay_ratio": 0.4, "desc": "Push accuracy with higher LR/K."},
    4: {"name": "Recover",      "k": 2, "lr_scale": 0.5, "replay_ratio": 0.6, "desc": "Stabilize divergence with lower LR."},
}

_DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


_cache = {"tok": None, "mdl": None, "model_name": None}


def _extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Robust-ish extraction: find first {...} and try json.loads.
    We only need {"strategy_id": int} (and optional "reasoning").
    """
    m = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if not m:
        return None
    blob = m.group(0)
    # common tiny fixes
    blob = blob.replace("\n", " ").strip()
    try:
        return json.loads(blob)
    except Exception:
        return None


def _build_action_from_strategy(strategy_id: int, n_clients: int) -> Dict[str, Any]:
    strat = STRATEGY_PALETTE.get(int(strategy_id), STRATEGY_PALETTE[1])
    return {
        "client_selection_k": int(strat["k"]),
        "aggregation": {"method": "FedAvg"},
        "client_params": [
            {
                "id": int(i),
                "replay_ratio": float(strat["replay_ratio"]),
                "lr_scale": float(strat["lr_scale"]),
                "ewc_lambda": 0.0,
            }
            for i in range(n_clients)
        ],
        "policy_source": f"LMSS_LOCAL_{int(strategy_id)}",
    }


def lmss_decide_action_local(
    state: Dict[str, Any],
    compact_state_fn,
    model_name: str = _DEFAULT_MODEL,
    max_new_tokens: int = 96,
) -> Dict[str, Any]:
    """
    Local LMSS: LLM outputs ONLY {"strategy_id": int, "reasoning": "..."}.
    We deterministically expand to full action schema.
    """
    n_clients = len(state.get("clients", []))
    if n_clients <= 0:
        # should never happen, but don't crash
        return _build_action_from_strategy(1, 0)

    # lazy load / cache
    if _cache["tok"] is None or _cache["mdl"] is None or _cache["model_name"] != model_name:
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else None,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mdl = mdl.to(device)
        _cache["tok"], _cache["mdl"], _cache["model_name"] = tok, mdl, model_name

    tok, mdl = _cache["tok"], _cache["mdl"]

    s_small = compact_state_fn(state)

    palette_text = "\n".join(
        [f"{k}: {v['name']} — {v['desc']}" for k, v in STRATEGY_PALETTE.items()]
    )

    # Force *tiny* output space: only one integer id + optional short reasoning.
    prompt = f"""You are a Strategy Selector for Federated Continual Learning.

STATE:
{json.dumps(s_small)}

STRATEGY PALETTE:
{palette_text}

Return ONLY one JSON object in this exact format:
{{"strategy_id": <int>, "reasoning": "<one short sentence>"}}

No extra text.
"""

    inputs = tok(prompt, return_tensors="pt")
    inputs = {k: v.to(mdl.device) for k, v in inputs.items()}

    with torch.no_grad():
        out = mdl.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy => more stable
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )

    text = tok.decode(out[0], skip_special_tokens=True)

    parsed = _extract_first_json_object(text)
    if not parsed or "strategy_id" not in parsed:
        # hard fallback: Standard
        return _build_action_from_strategy(1, n_clients)

    sid = int(parsed.get("strategy_id", 1))
    action = _build_action_from_strategy(sid, n_clients)

    # Optional: print reasoning (safe)
    reasoning = parsed.get("reasoning", "")
    if reasoning:
        print(f"[LMSS_LOCAL] chose strategy={sid} | {reasoning}", flush=True)
    else:
        print(f"[LMSS_LOCAL] chose strategy={sid}", flush=True)

    return action