import json
import hashlib
import os
import time
from typing import Any, Dict

from openai import OpenAI


STRATEGY_PALETTE = {
    0: {"name": "Conservative", "k": 2, "lr": 1e-4, "lr_scale": 0.8, "replay_ratio": 0.6},
    1: {"name": "Standard", "k": 2, "lr": 1e-4, "lr_scale": 1.0, "replay_ratio": 0.5},
    2: {"name": "Consolidate", "k": 3, "lr": 1.5e-4, "lr_scale": 1.0, "replay_ratio": 0.7},
    3: {"name": "Aggressive", "k": 4, "lr": 3e-4, "lr_scale": 1.0, "replay_ratio": 0.4},
    4: {"name": "HyperDrive", "k": 4, "lr": 4.5e-4, "lr_scale": 1.0, "replay_ratio": 0.3},
}

_DEFAULT_MODEL = "openai/gpt-4o-mini"
_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


def _build_action_from_strategy(strategy_id: int, n_clients: int, policy_source: str) -> Dict[str, Any]:
    strat = STRATEGY_PALETTE.get(int(strategy_id), STRATEGY_PALETTE[1])
    return {
        "lr": float(strat["lr"]),
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
        "policy_source": policy_source,
    }


def lmss_decide_action_openrouter(
    state: Dict[str, Any],
    compact_state_fn,
    model: str = _DEFAULT_MODEL,
    deterministic: bool = False,
) -> Dict[str, Any]:
    n_clients = len(state.get("clients", []))
    if n_clients <= 0:
        return _build_action_from_strategy(1, 0, "LMSS_OPENROUTER_EMPTY_CLIENTS_FALLBACK")

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        action = _build_action_from_strategy(1, n_clients, "LMSS_OPENROUTER_NO_KEY_FALLBACK")
        action["controller_metadata"] = {
            "call_count": 0,
            "fallback": True,
            "latency_seconds": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "billed_cost": None,
            "requested_model": model,
            "response_model": None,
            "temperature": 0.0 if deterministic else None,
            "top_p": 1.0 if deterministic else None,
        }
        print(
            "[LMSS_OPENROUTER] policy_source=LMSS_OPENROUTER_NO_KEY_FALLBACK "
            "raw_response=<missing OPENROUTER_API_KEY> strategy_id=1 "
            f"applied_lr={action['lr']:.6f} applied_replay_ratio={action['client_params'][0]['replay_ratio']:.2f}",
            flush=True,
        )
        return action

    client_kwargs = dict(
        api_key=api_key,
        base_url=os.environ.get("OPENROUTER_BASE_URL", _DEFAULT_BASE_URL),
    )
    if deterministic:
        client_kwargs.update(max_retries=0, timeout=60.0)
    client = OpenAI(**client_kwargs)

    s_small = compact_state_fn(state)
    palette_text = "\n".join(
        [
            f"{k}: {v['name']} (k={v['k']}, lr={v['lr']}, replay={v['replay_ratio']})"
            for k, v in STRATEGY_PALETTE.items()
        ]
    )

    prompt = f"""You are a Strategy Selector for Federated Continual Learning.

STATE:
{json.dumps(s_small)}

STRATEGY PALETTE:
{palette_text}

Return ONLY one JSON object in this exact format:
{{"strategy_id": <int>, "reasoning": "<one short sentence>"}}

No extra text.
"""
    prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    extra_headers = {}
    referer = os.environ.get("OPENROUTER_HTTP_REFERER")
    title = os.environ.get("OPENROUTER_APP_TITLE")
    if referer:
        extra_headers["HTTP-Referer"] = referer
    if title:
        extra_headers["X-Title"] = title

    raw_content = ""
    started = time.perf_counter()
    try:
        request_kwargs = dict(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            extra_headers=extra_headers or None,
        )
        if deterministic:
            request_kwargs.update(temperature=0.0, top_p=1.0)
        resp = client.chat.completions.create(**request_kwargs)
        latency_seconds = time.perf_counter() - started
        raw_content = resp.choices[0].message.content or ""
        parsed = json.loads(raw_content)
        if "strategy_id" not in parsed:
            raise ValueError("LMSS response is missing strategy_id")
        strategy_id = int(parsed["strategy_id"])
        if strategy_id not in STRATEGY_PALETTE:
            raise ValueError(f"LMSS returned unsupported strategy_id={strategy_id}")
        action = _build_action_from_strategy(
            strategy_id,
            n_clients,
            f"LMSS_OPENROUTER_{model}_STRAT_{strategy_id}",
        )
        usage = getattr(resp, "usage", None)
        action["strategy_id"] = strategy_id
        action["reasoning"] = str(parsed.get("reasoning", ""))
        action["controller_metadata"] = {
            "call_count": 1,
            "fallback": False,
            "latency_seconds": float(latency_seconds),
            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
            "billed_cost": getattr(usage, "cost", None),
            "requested_model": model,
            "response_model": getattr(resp, "model", None),
            "temperature": 0.0 if deterministic else None,
            "top_p": 1.0 if deterministic else None,
            "prompt_sha256": prompt_sha256,
        }
        print(
            f"[LMSS_OPENROUTER] policy_source={action['policy_source']} "
            f"raw_response={raw_content} strategy_id={strategy_id} "
            f"applied_lr={action['lr']:.6f} applied_replay_ratio={action['client_params'][0]['replay_ratio']:.2f}",
            flush=True,
        )
        return action
    except Exception as e:
        latency_seconds = time.perf_counter() - started
        action = _build_action_from_strategy(1, n_clients, "LMSS_OPENROUTER_ERROR_FALLBACK")
        action["controller_metadata"] = {
            "call_count": 1,
            "fallback": True,
            "latency_seconds": float(latency_seconds),
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "billed_cost": None,
            "requested_model": model,
            "response_model": None,
            "temperature": 0.0 if deterministic else None,
            "top_p": 1.0 if deterministic else None,
            "prompt_sha256": prompt_sha256,
            "error": repr(e),
        }
        print(
            f"[LMSS_OPENROUTER] policy_source=LMSS_OPENROUTER_ERROR_FALLBACK "
            f"raw_response={raw_content or repr(e)} strategy_id=1 "
            f"applied_lr={action['lr']:.6f} applied_replay_ratio={action['client_params'][0]['replay_ratio']:.2f}",
            flush=True,
        )
        return action
