import json
import os
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
) -> Dict[str, Any]:
    n_clients = len(state.get("clients", []))
    if n_clients <= 0:
        return _build_action_from_strategy(1, 0, "LMSS_OPENROUTER_EMPTY_CLIENTS_FALLBACK")

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        action = _build_action_from_strategy(1, n_clients, "LMSS_OPENROUTER_NO_KEY_FALLBACK")
        print(
            "[LMSS_OPENROUTER] policy_source=LMSS_OPENROUTER_NO_KEY_FALLBACK "
            "raw_response=<missing OPENROUTER_API_KEY> strategy_id=1 "
            f"applied_lr={action['lr']:.6f} applied_replay_ratio={action['client_params'][0]['replay_ratio']:.2f}",
            flush=True,
        )
        return action

    client = OpenAI(
        api_key=api_key,
        base_url=os.environ.get("OPENROUTER_BASE_URL", _DEFAULT_BASE_URL),
    )

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

    extra_headers = {}
    referer = os.environ.get("OPENROUTER_HTTP_REFERER")
    title = os.environ.get("OPENROUTER_APP_TITLE")
    if referer:
        extra_headers["HTTP-Referer"] = referer
    if title:
        extra_headers["X-Title"] = title

    raw_content = ""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            extra_headers=extra_headers or None,
        )
        raw_content = resp.choices[0].message.content or ""
        parsed = json.loads(raw_content)
        strategy_id = int(parsed.get("strategy_id", 1))
        action = _build_action_from_strategy(
            strategy_id,
            n_clients,
            f"LMSS_OPENROUTER_{model}_STRAT_{strategy_id}",
        )
        print(
            f"[LMSS_OPENROUTER] policy_source={action['policy_source']} "
            f"raw_response={raw_content} strategy_id={strategy_id} "
            f"applied_lr={action['lr']:.6f} applied_replay_ratio={action['client_params'][0]['replay_ratio']:.2f}",
            flush=True,
        )
        return action
    except Exception as e:
        action = _build_action_from_strategy(1, n_clients, "LMSS_OPENROUTER_ERROR_FALLBACK")
        print(
            f"[LMSS_OPENROUTER] policy_source=LMSS_OPENROUTER_ERROR_FALLBACK "
            f"raw_response={raw_content or repr(e)} strategy_id=1 "
            f"applied_lr={action['lr']:.6f} applied_replay_ratio={action['client_params'][0]['replay_ratio']:.2f}",
            flush=True,
        )
        return action
