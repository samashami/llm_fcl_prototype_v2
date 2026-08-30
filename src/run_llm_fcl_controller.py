# src/run_llm_fcl_controller.py

import argparse, time, copy, random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch import optim
from torchvision import datasets, transforms
import pandas as pd
import math
from pathlib import Path

from src.model import build_resnet18
from src.fl import Client, Server
from src.strategies.replay import ReplayBuffer
from src.policy import Policy
from src.policy.lmss_api import lmss_decide_action_api
from src.policy.lmss_openrouter import lmss_decide_action_openrouter
from src.agent_io import save_json
from src.agent_io import write_state_json, write_action_json, validate_action
from src.mock_agent import decide_action as mock_decide_action
from src.instrumentation.subspace import (
    DEFAULT_RESNET18_TARGETS,
    SubspaceInstrumentation,
    aggregate_update_energy_records,
)
from src.instrumentation.shrinkage import load_shrinkage_schedule
from src.checkpointing import (
    CHECKPOINT_FORMAT_VERSION,
    branch_control_metadata,
    capture_rng_state,
    endpoint_state,
    fingerprint_sha256,
    load_checkpoint,
    model_state_sha256,
    protocol_manifest,
    restore_rng_state,
    requires_endpoint_equality,
    save_checkpoint,
    validate_determinism_gate,
    validate_resume_protocol,
    verify_endpoint_reference,
    write_endpoint_reference,
)
import os, json
# run_llm_fcl_controller.py
from src._bootstrap_env import *  # sets TOKENIZERS_PARALLELISM=false early

# ---------------------------
# Controller v4 hyperparams
# ---------------------------
V4_LR_MIN, V4_LR_MAX = 1e-4, 2e-3
V4_REP_MIN, V4_REP_MAX = 0.20, 0.70
V4_DEADBAND = 0.003
V4_REP_STEP_HIGH = 0.10
V4_REP_STEP_LOW  = 0.05
V4_FORGET_THR    = 0.05
V4_DIV_THR       = 0.10
V4_EMA_ALPHA     = 0.30
V4_LR_BOOST      = 1.35
V4_LR_COOLDOWN   = 1.50
V4_CLIENT_LR_MIN, V4_CLIENT_LR_MAX = 0.8, 1.2
V4_ROLLBACK_THR  = 0.015         # allow 1.5% drop before rollback
V4_WARMUP_ROUNDS = 2


def local_epoch_budget(configured_epochs, branch_local_epochs):
    """Return the production epoch budget without changing the ordinary path."""
    return configured_epochs if branch_local_epochs is None else branch_local_epochs


def should_stop_local_training(early_stop, branch_local_epochs):
    """Honor early stopping unless a resumed branch has a fixed compute budget."""
    return bool(early_stop) and branch_local_epochs is None


def make_dirichlet_client_splits(train_indices, targets, n_clients, alpha, seed):
    """Partition the supplied training indices class-wise across clients."""
    train_indices = np.asarray(train_indices, dtype=np.int64)
    targets = np.asarray(targets)
    if n_clients <= 0:
        raise ValueError("n_clients must be positive")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if len(np.unique(train_indices)) != len(train_indices):
        raise ValueError("train_indices must not contain duplicates")
    if len(train_indices) < n_clients:
        raise ValueError("Dirichlet split requires at least one sample per client")

    rng = np.random.RandomState(seed)
    train_targets = targets[train_indices]

    # A shuffled DataLoader requires non-empty clients. Redraw only in the rare
    # event that a client receives no samples; do not otherwise alter the draw.
    for _ in range(100):
        splits = [[] for _ in range(n_clients)]
        for class_id in np.unique(train_targets):
            class_indices = train_indices[train_targets == class_id].copy()
            rng.shuffle(class_indices)
            proportions = rng.dirichlet(np.full(n_clients, alpha))
            counts = rng.multinomial(len(class_indices), proportions)

            start = 0
            for client_id, count in enumerate(counts):
                end = start + count
                splits[client_id].extend(class_indices[start:end].tolist())
                start = end

        if all(splits):
            assigned = np.concatenate(
                [np.asarray(split, dtype=np.int64) for split in splits]
            )
            if len(assigned) != len(train_indices) or not np.array_equal(
                np.sort(assigned), np.sort(train_indices)
            ):
                raise RuntimeError("Dirichlet split did not assign training samples exactly once")
            return splits

    raise RuntimeError("Could not produce a non-empty Dirichlet split after 100 draws")


# --- SFT controller helper (local tiny model) ---
def _compact_state_for_sft(state):
    g = state["global"]
    keep = {
        "acc": float(round(g["acc"], 4)),
        "ema_loss": float(round(g["ema_loss"], 4)),
        "forget_mean": float(round(g["forget_mean"], 4)),
        "divergence": float(round(g["divergence"], 4)),
    }
    clients = []
    for c in state["clients"]:
        vloss = c["vloss"]
        if isinstance(vloss, float) and vloss != vloss:  # NaN
            vloss = None
        def _safe_float(x):
            try:
                x = float(x)
                # NaN check
                return None if (x != x) else x
            except Exception:
                return None

        clients.append({
            "id": int(c["id"]),
            "vloss": _safe_float(vloss),
            "vacc": _safe_float(c.get("vacc")),
            "new_batch_size": int(c["new_batch_size"]),
            "last_lr": _safe_float(c.get("last_lr")),
        })
    return {"global": keep, "clients": clients}

import re

def _balanced_json_from_text(s, anchor="ACTION:"):
    """
    Find JSON after anchor and repair common small-model errors:
    - missing quotes on keys: {aggregation: -> {"aggregation":
    - truncated JSON: add missing ] and } when possible
    """
    if anchor in s:
        s = s.split(anchor)[-1]

    start_idx = s.find("{")
    if start_idx == -1:
        return "{}"

    s = s[start_idx:].strip()

    # Repair 1: add quotes to unquoted keys
    s = re.sub(r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', s)

    # Repair 2: find balanced end
    depth = 0
    final_idx = len(s)
    for i, ch in enumerate(s):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                final_idx = i + 1
                break

    candidate = s[:final_idx]

    # Repair 3: if truncated, best-effort close
    if depth > 0:
        candidate = candidate.rstrip().rstrip(",")
        if "[" in candidate and "]" not in candidate:
            candidate += "]"
        candidate += ("}" * depth)

    return candidate

_sft_cache = {"tok": None, "mdl": None}

def sft_decide_action(state, model_dir="sft_model_distilgpt2", fewshot=True):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # lazy load (keeps it fast across rounds)
    if _sft_cache["tok"] is None:
        tok = AutoTokenizer.from_pretrained(model_dir)
        mdl = AutoModelForCausalLM.from_pretrained(model_dir)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mdl = mdl.to(device)
        print("✅ SFT model device:", next(mdl.parameters()).device, flush=True)

        tok.pad_token = tok.eos_token
        _sft_cache["tok"], _sft_cache["mdl"] = tok, mdl
    else:
        tok, mdl = _sft_cache["tok"], _sft_cache["mdl"]

    s_small = _compact_state_for_sft(state)
    num_clients = len(state["clients"])

    instruction = (
        f"TASK: Output ONE valid JSON object for an ACTION for {num_clients} clients.\n"
        f"CLIENT IDS: You MUST include exactly one entry for EACH client id 0..{num_clients-1}.\n"
        "OUTPUT RULES:\n"
        "- Output ONLY JSON. No text before or after.\n"
        "- Use ONLY the ACTION schema below.\n"
        "ACTION SCHEMA:\n"
        '{ "client_selection_k": <int>, "aggregation": {"method":"FedAvg"}, '
        '"client_params": ['
        '{"id":<int>,"replay_ratio":<float>,"lr_scale":<float>,"ewc_lambda":<float>},'
        ' ... ] }\n'
        "BOUNDS:\n"
        "- replay_ratio in [0.0, 0.7]\n"
        "- lr_scale in [0.5, 1.5]\n"
        "- ewc_lambda in [0.0, 10.0]\n"
        "IMPORTANT:\n"
        "- Do NOT output STATE.\n"
        "- Do NOT invent new client ids.\n"
    )

    demo = ""
    if fewshot:
        demo = (
            "EXAMPLE:\n"
            'STATE: {"global":{"acc":0.011,"ema_loss":4.73,"forget_mean":0.001,"divergence":0.001},'
            '"clients":[{"id":0,"vloss":4.68,"vacc":1.36,"new_batch_size":45,"last_lr":0.00012},'
            '{"id":1,"vloss":4.67,"vacc":1.34,"new_batch_size":45,"last_lr":0.00012}]}\n'
            "ACTION:\n"
            '{"client_selection_k":2,"aggregation":{"method":"FedAvg"},"client_params":['
            '{"id":0,"replay_ratio":0.5,"lr_scale":0.8,"ewc_lambda":0.0},'
            '{"id":1,"replay_ratio":0.5,"lr_scale":1.2,"ewc_lambda":0.0}]}\n\n'
        )

    # Force the model to start JSON immediately:
    prompt = (
        instruction + demo +
        f"STATE: {json.dumps(s_small, allow_nan=False)}\n\n"
        f"ACTION:\n{{\"client_selection_k\": {num_clients}, "
        f"\"aggregation\": {{\"method\": \"FedAvg\"}}, "
        f"\"client_params\": ["
    )

    inputs = tok(prompt, return_tensors="pt")
    inputs = {k: v.to(mdl.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        gen = mdl.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )

    out_text = tok.decode(gen[0], skip_special_tokens=True)

    # Your prompt contains "ACTION:\n{", so extract from there
    raw_json = _balanced_json_from_text(out_text, anchor="ACTION:")

    # DEBUG (temporary): write what we saw if parse fails
    if raw_json == "{}":
        print(f"\n==== SFT PRODUCED NO JSON ====\nOUT_TEXT_TAIL:\n{out_text[-800:]}\n", flush=True)
        n_clients = len(state["clients"])
        return {
            "client_selection_k": n_clients,
            "aggregation": {"method": "FedAvg"},
            "client_params": [
                {"id": i, "replay_ratio": 0.5, "lr_scale": 1.0, "ewc_lambda": 0.0}
                for i in range(n_clients)
            ],
        }
    try:
        act = json.loads(raw_json)
    except json.JSONDecodeError as e:
        dump = (
            "\n==== SFT JSON DECODE FAILED ====\n"
            f"Error: {e}\n"
            f"RAW_JSON_HEAD: {raw_json[:400]}\n"
            f"RAW_JSON_TAIL: {raw_json[-400:]}\n"
            "\n---- OUT_TEXT_TAIL (last 800 chars) ----\n"
            f"{out_text[-800:]}\n"
        )
        print(dump, flush=True)

        n_clients = len(state["clients"])
        return {
            "client_selection_k": n_clients,
            "aggregation": {"method": "FedAvg"},
            "client_params": [
                {"id": i, "replay_ratio": 0.5, "lr_scale": 1.0, "ewc_lambda": 0.0}
                for i in range(n_clients)
            ],
        }

    # =========================
    # STEP 2.2: sanitize client ids + enforce exact 0..n-1
    # =========================
    n_clients = len(state["clients"])
    params = act.get("client_params", [])
    by_id = {}
    for p in params:
        try:
            cid = int(p.get("id"))
        except Exception:
            continue
        if 0 <= cid < n_clients and cid not in by_id:
            by_id[cid] = p

    fixed = []
    for cid in range(n_clients):
        p = by_id.get(cid, {"id": cid})
        fixed.append({
            "id": cid,
            "replay_ratio": float(p.get("replay_ratio", 0.5)),
            "lr_scale": float(p.get("lr_scale", 1.0)),
            "ewc_lambda": float(p.get("ewc_lambda", 0.0)),
        })

    act["client_selection_k"] = n_clients
    act["aggregation"] = {"method": "FedAvg"}
    act["client_params"] = fixed

    # Your original checks (keep them)
    if "client_selection_k" not in act:
        raise RuntimeError(f"SFT action missing client_selection_k: {act}")
    if "client_params" not in act or not isinstance(act["client_params"], list) or len(act["client_params"]) == 0:
        raise RuntimeError(f"SFT action has empty/missing client_params: {act}")

    return act

# ---------------------------
# Seeding helpers
# ---------------------------
GLOBAL_SEED = 42
def seed_worker(worker_id: int):
    import numpy as _np, random as _random
    _np.random.seed(GLOBAL_SEED + worker_id)
    _random.seed(GLOBAL_SEED + worker_id)

def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------------------------
# Eval helpers
# ---------------------------
def evaluate(model, device, test_loader):
    model.eval()
    correct, total = 0, 0
    n_classes = 100
    hits = np.zeros(n_classes, dtype=np.int64)
    counts = np.zeros(n_classes, dtype=np.int64)
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
            for c in range(n_classes):
                mask = (y == c)
                if mask.any():
                    counts[c] += mask.sum().item()
                    hits[c] += (pred[mask] == c).sum().item()
    acc = correct / max(1, total)
    per_class_recall = np.array(
        [(hits[c] / counts[c]) if counts[c] > 0 else 0.0 for c in range(n_classes)],
        dtype=np.float32,
    )
    return acc, per_class_recall

def evaluate_loss(model, device, loader):
    model.eval()
    crit = torch.nn.CrossEntropyLoss()
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += float(crit(logits, y).item()) * y.size(0)
            n += y.size(0)
    return total_loss / max(1, n)

# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--subset_per_client", type=int, default=-1, help="use -1 for all data")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_interval", type=int, default=50)
    ap.add_argument("--split_mode", choices=["equal", "dirichlet"], default="equal")
    ap.add_argument("--val_size", type=int, default=5000)
    ap.add_argument("--cl_batches", type=int, default=7)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--optimizer", choices=["adam","sgd"], default="adam")
    ap.add_argument("--early_patience", type=int, default=5)
    ap.add_argument("--tag", type=str, default="controller_v4")
    ap.add_argument("--controller", choices=["v4", "mock", "fixed", "sft", "lmss_api", "lmss_local", "lmss_openrouter"], default="v4")
    ap.add_argument("--lmss_model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--measure_subspaces", action="store_true",
                    help="enable read-only protected-subspace measurements")
    ap.add_argument("--subspace_energy", type=float, default=0.95)
    ap.add_argument("--subspace_max_rank", type=int, default=32)
    ap.add_argument("--subspace_samples_per_batch", type=int, default=8)
    ap.add_argument("--subspace_samples_per_phase", type=int, default=64)
    ap.add_argument(
        "--projection_lambda",
        type=float,
        choices=[0.0, 0.25, 0.5, 0.75, 1.0],
        default=0.0,
    )
    ap.add_argument(
        "--update_control",
        choices=["projection", "shrinkage", "shrinkage_online"],
        default="projection",
    )
    ap.add_argument("--shrinkage_schedule", type=str, default=None)
    ap.add_argument(
        "--capture_common_checkpoints",
        action="store_true",
        help="save immutable pre-round-1/pre-round-5 checkpoints and parent references",
    )
    ap.add_argument("--common_checkpoint_dir", type=str, default=None)
    ap.add_argument("--resume_checkpoint", type=str, default=None)
    ap.add_argument(
        "--branch_local_epochs",
        type=int,
        default=None,
        help="fixed local epoch budget for every client in a resumed one-round branch",
    )
    ap.add_argument(
        "--one_round",
        action="store_true",
        help="execute only the checkpoint's next round, then stop",
    )
    ap.add_argument(
        "--determinism_reference",
        type=str,
        default=None,
        help="parent endpoint JSON required for a resumed lambda=0 branch",
    )
    ap.add_argument(
        "--determinism_gate_dir",
        type=str,
        default=None,
        help="directory containing the lambda=0 PASS report for this checkpoint",
    )
    ap.add_argument("--output_dir", type=str, default=".")
    args = ap.parse_args()

    if args.capture_common_checkpoints:
        if args.controller != "fixed" or args.optimizer != "adam":
            ap.error("common-checkpoint parent requires fixed controller and Adam")
        if args.update_control != "projection" or args.projection_lambda != 0.0:
            ap.error("common-checkpoint parent must use projection mode with lambda=0")
        if not args.measure_subspaces:
            ap.error("common-checkpoint parent requires --measure_subspaces")
        if args.rounds < 6:
            ap.error("common-checkpoint parent must run through round 5 (--rounds >= 6)")
        if not args.common_checkpoint_dir:
            ap.error("--capture_common_checkpoints requires --common_checkpoint_dir")
    if args.resume_checkpoint:
        if not args.one_round:
            ap.error("--resume_checkpoint requires --one_round")
        if args.controller != "fixed" or args.optimizer != "adam":
            ap.error("common-checkpoint branches require fixed controller and Adam")
        if not args.measure_subspaces:
            ap.error("common-checkpoint branches require --measure_subspaces")
        if args.branch_local_epochs is not None and not (
            1 <= args.branch_local_epochs <= args.epochs
        ):
            ap.error("--branch_local_epochs must satisfy 1 <= value <= --epochs")
        if (
            args.update_control == "projection"
            and args.projection_lambda == 0.0
            and not args.determinism_reference
        ):
            ap.error("a resumed lambda=0 branch requires --determinism_reference")
        if (
            not requires_endpoint_equality(
                args.update_control, args.projection_lambda
            )
            and not args.determinism_gate_dir
        ):
            ap.error("treatment branches require --determinism_gate_dir")
    elif (
        args.one_round
        or args.determinism_reference
        or args.determinism_gate_dir
        or args.branch_local_epochs is not None
    ):
        ap.error(
            "branch resume options "
            "require --resume_checkpoint"
        )

    branching_mode = bool(args.capture_common_checkpoints or args.resume_checkpoint)
    if branching_mode:
        torch.use_deterministic_algorithms(True)
        if args.device in {"cuda", "auto"} and torch.cuda.is_available():
            if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
                ap.error(
                    "deterministic CUDA branching requires "
                    "CUBLAS_WORKSPACE_CONFIG=:4096:8 (set it before Python starts)"
                )

    if args.update_control == "projection" and args.projection_lambda != 0.0:
        if args.controller != "fixed":
            ap.error("--projection_lambda > 0 requires --controller fixed")
        if args.optimizer != "adam":
            ap.error("--projection_lambda > 0 requires --optimizer adam")
        if not args.measure_subspaces:
            ap.error("--projection_lambda > 0 requires --measure_subspaces")
    if args.update_control in {"shrinkage", "shrinkage_online"}:
        if args.controller != "fixed":
            ap.error("shrinkage modes require --controller fixed")
        if args.optimizer != "adam":
            ap.error("shrinkage modes require --optimizer adam")
        if not args.measure_subspaces:
            ap.error("shrinkage modes require --measure_subspaces")
        if args.update_control == "shrinkage":
            if args.projection_lambda != 0.0:
                ap.error("schedule-based shrinkage mode requires --projection_lambda 0")
            if not args.shrinkage_schedule:
                ap.error("shrinkage mode requires --shrinkage_schedule")
        else:
            if args.projection_lambda not in {0.0, 0.25, 0.5, 0.75, 1.0}:
                ap.error("online shrinkage mode requires a supported --projection_lambda")
    elif args.shrinkage_schedule:
        ap.error("--shrinkage_schedule is only valid in shrinkage mode")

    controller_name_map = {
        "v4": "ControllerV4",
        "mock": "Mock",
        "fixed": "Fixed",
        "sft": "SFT_v0",
        "lmss_api": "LMSS_API",
        "lmss_local": "LMSS_LOCAL",
        "lmss_openrouter": "LMSS_OPENROUTER",
    }

    controller_name = controller_name_map.get(args.controller, args.controller)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = (
        Path(args.common_checkpoint_dir)
        if args.common_checkpoint_dir
        else output_dir / "checkpoints"
    )
    resume_payload = None
    resume_metadata = None
    if args.resume_checkpoint:
        resume_payload, resume_metadata = load_checkpoint(args.resume_checkpoint)

    protocol_fields = (
        "clients", "alpha", "epochs", "rounds", "batch_size", "lr",
        "subset_per_client", "seed", "split_mode", "val_size", "cl_batches",
        "num_workers", "optimizer", "early_patience", "controller",
        "measure_subspaces", "subspace_energy", "subspace_max_rank",
        "subspace_samples_per_batch", "subspace_samples_per_phase",
        "update_control",
    )
    protocol = {name: getattr(args, name) for name in protocol_fields}
    code_paths = (
        "src/run_llm_fcl_controller.py",
        "src/checkpointing.py",
        "src/fl.py",
        "src/strategies/replay.py",
        "src/instrumentation/subspace.py",
    )
    
    set_seeds(args.seed)
    # safe device selection with fallback for mac (no CUDA)
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        try:
            if args.device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA not available")
            if args.device == "mps" and not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
                raise RuntimeError("MPS not available")
            device = torch.device(args.device)
        except Exception as e:
            print(f"Warning: requested device '{args.device}' not available ({e}); falling back to cpu", flush=True)
            device = torch.device("cpu")
    print(f"Using device: {device}", flush=True)

    
    global GLOBAL_SEED
    GLOBAL_SEED = args.seed

    g = torch.Generator()
    g.manual_seed(args.seed)

    # Transforms
    tf_train = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
    ])
    tf_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
    ])

    # Data
    trainset_full = datasets.CIFAR100(root="./data", train=True,  download=True, transform=tf_train)
    testset       = datasets.CIFAR100(root="./data", train=False, download=True, transform=tf_test)

    total_train = len(trainset_full)  # 50_000
    val_size = args.val_size          # 5_000
    train_size = total_train - val_size

    train_subset, val_subset = torch.utils.data.random_split(
        trainset_full, [train_size, val_size], generator=g
    )
    train_indices = np.array(train_subset.indices, dtype=np.int64)
    val_indices   = np.array(val_subset.indices, dtype=np.int64)

    valset = Subset(trainset_full, val_indices)
    val_loader = DataLoader(valset, batch_size=256, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=256, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    print(f"[Split] train={len(train_indices)} val={len(val_indices)} test={len(testset)}", flush=True)

    # Split among clients
    if args.split_mode == "equal":
        rng = np.random.RandomState(args.seed)
        perm = rng.permutation(train_indices)
        sizes = [len(perm) // args.clients] * args.clients
        for i in range(len(perm) % args.clients):
            sizes[i] += 1
        splits, start = [], 0
        for s in sizes:
            splits.append(perm[start:start+s].tolist())
            start += s
    else:
        splits = make_dirichlet_client_splits(
            train_indices,
            trainset_full.targets,
            n_clients=args.clients,
            alpha=args.alpha,
            seed=args.seed,
        )
        client_sizes = [len(idxs) for idxs in splits]
        targets = np.asarray(trainset_full.targets)
        for i, idxs in enumerate(splits):
            class_ids, counts = np.unique(targets[idxs], return_counts=True)
            distribution = dict(zip(class_ids.tolist(), counts.tolist()))
            print(
                f"[Split] client {i}: total={len(idxs)} "
                f"class_distribution={distribution}",
                flush=True,
            )
        print(
            f"[Split] client sizes: min={min(client_sizes)} max={max(client_sizes)}",
            flush=True,
        )

    # Optional subsample per client
    if args.subset_per_client and args.subset_per_client > 0:
        splits = [idxs[:args.subset_per_client] for idxs in splits]

    for i, idxs in enumerate(splits):
        print(f"[Split] client {i}: {len(idxs)} images", flush=True)

    # Build CL schedule: initial ~0.466 + even splits
    def make_cl_batches(indices, num_batches=7, seed=42):
        rng_local = np.random.RandomState(seed)
        idx = np.array(indices, dtype=np.int64)
        rng_local.shuffle(idx)
        init = int(round(0.466 * len(idx)))
        init = max(1, min(len(idx) - (num_batches - 1), init))
        first = idx[:init]
        rem = idx[init:]
        if num_batches <= 1:
            return [idx.tolist()]
        per = len(rem) // (num_batches - 1)
        chunks = [rem[i*per:(i+1)*per] for i in range(num_batches - 2)]
        chunks.append(rem[(num_batches - 2)*per:])
        return [first.tolist()] + [c.tolist() for c in chunks]

    cl_schedule, cl_rows = [], []
    for cid, idxs in enumerate(splits):
        batches = make_cl_batches(idxs, num_batches=args.cl_batches, seed=args.seed + cid)
        cl_schedule.append(batches)
        sizes = [len(b) for b in batches]
        print(f"[CL] client {cid}: {sizes} (sum={sum(sizes)})", flush=True)
        for i, b in enumerate(batches, start=1):
            cl_rows.append({"run_id": "", "client": cid, "cl_batch": i, "size": len(b)})

    data_state = {
        "train_indices": train_indices.tolist(),
        "val_indices": val_indices.tolist(),
        "client_splits": [[int(index) for index in split] for split in splits],
        "cl_schedule": [
            [[int(index) for index in batch] for batch in client_batches]
            for client_batches in cl_schedule
        ],
    }
    if resume_payload is not None:
        checkpoint_manifest = resume_payload.get("manifest", {})
        validate_resume_protocol(checkpoint_manifest.get("protocol", {}), protocol)
        if resume_payload.get("data_state") != data_state:
            raise RuntimeError("train/validation split, client split, or CL schedule differs")

    # Init clients
    clients = []
    for cid, idx in enumerate(splits):
        subset = Subset(trainset_full, idx)
        loader = DataLoader(
            subset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g,
        )
        model = build_resnet18(100).to(device)
        if args.optimizer == "adam":
            opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)
        else:
            opt = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        replay = ReplayBuffer(capacity=2000)
        clients.append(Client(cid, model, opt, loader, device=device, replay=replay,
                              val_loader=val_loader, early_patience=args.early_patience))

        if cid == 0:
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            print(f"Trainable params: {trainable:,}/{total}", flush=True)

    print("✅ Client 0 model device:", next(clients[0].model.parameters()).device, flush=True)

    subspace_instrumentation = None
    if args.measure_subspaces:
        subspace_instrumentation = SubspaceInstrumentation(
            [c.model for c in clients],
            explained_energy=args.subspace_energy,
            max_rank=args.subspace_max_rank,
            samples_per_batch=args.subspace_samples_per_batch,
            samples_per_phase=args.subspace_samples_per_phase,
        )
        for client, monitor in zip(clients, subspace_instrumentation.monitors):
            client.gradient_monitor = monitor
        if args.update_control == "shrinkage":
            print("[Subspace] frozen norm-matched scalar shrinkage enabled", flush=True)
        elif args.update_control == "shrinkage_online":
            print(
                f"[Subspace] online energy-matching scalar shrinkage enabled "
                f"(lambda={args.projection_lambda:g})",
                flush=True,
            )
        elif args.projection_lambda == 0.0:
            print("[Subspace] measurement-only instrumentation enabled (lambda=0)", flush=True)
        else:
            print(
                f"[Subspace] fixed soft projection enabled "
                f"(lambda={args.projection_lambda:g})",
                flush=True,
            )

    shrinkage_schedule = None
    if args.update_control == "shrinkage":
        layer_names = [target.name for target in DEFAULT_RESNET18_TARGETS]
        expected_keys = (
            (round_id, client_id, layer_name)
            for round_id in range(args.rounds)
            for client_id in range(args.clients)
            for layer_name in layer_names
        )
        shrinkage_schedule = load_shrinkage_schedule(
            args.shrinkage_schedule, expected_keys
        )
        print(
            f"[Shrinkage] loaded {len(shrinkage_schedule)} frozen schedule entries "
            f"from {args.shrinkage_schedule}",
            flush=True,
        )
    elif args.update_control == "shrinkage_online":
        print(
            "[Shrinkage] online energy-matching mode enabled; no frozen CSV schedule required",
            flush=True,
        )

    current_manifest = protocol_manifest(protocol, code_paths) if branching_mode else None
    if resume_payload is not None:
        checkpoint_manifest = resume_payload["manifest"]
        if checkpoint_manifest["code"]["files_sha256"] != current_manifest["code"]["files_sha256"]:
            raise RuntimeError("checkpoint code-file checksums differ from current code")
        environment_keys = (
            "python", "platform", "torch", "numpy", "cuda_runtime", "cudnn",
            "cuda_device_count", "cuda_devices", "cudnn_deterministic",
            "cudnn_benchmark", "deterministic_algorithms", "cublas_workspace_config",
        )
        environment_differences = {
            key: {
                "checkpoint": checkpoint_manifest["environment"].get(key),
                "current": current_manifest["environment"].get(key),
            }
            for key in environment_keys
            if checkpoint_manifest["environment"].get(key)
            != current_manifest["environment"].get(key)
        }
        if environment_differences:
            raise RuntimeError(
                f"checkpoint environment differs from current environment: {environment_differences}"
            )

    # Server / Policy
    server = Server(device=device)
    policy = Policy()

    print("[DEBUG] Before initial FedAvg", flush=True)

    # Initial global model + metrics
    print("[DEBUG] Before initial FedAvg", flush=True)
    global_model = server.average([c.model for c in clients])
    print("[DEBUG] After initial FedAvg", flush=True)

    acc, per_class = evaluate(global_model, device, test_loader)
    print("[DEBUG] After initial evaluate()", flush=True)

    best_recall = per_class.copy()
    forgetting = np.zeros_like(per_class)
    global_loss = evaluate_loss(global_model, device, test_loader)
    ema_loss = global_loss
    div_norm = 0.0

    # --- comm accounting: approximate model size in bytes (FP32 unless changed) ---
    def _model_num_params_bytes(model) -> int:
        total = 0
        for p in model.parameters():
            total += p.numel() * (4 if p.dtype in (torch.float32, torch.int32) else 2 if p.dtype == torch.float16 else 4)
        return total

    MODEL_BYTES = _model_num_params_bytes(global_model)
    print(f"[comm] MODEL_BYTES ≈ {MODEL_BYTES:,}")

    print(f"[Round -1] acc={acc:.3f}", flush=True)

    # Local best/rollback tracking (no Server.save_state)
    best_state = copy.deepcopy(global_model.state_dict())
    best_global_acc = float(acc)
    best_hp = {"lr": args.lr, "replay_ratio": 0.50, "notes": "init (paper defaults)"}
    best_round = -1
    rollback_flag = False
    rollback_round = -1
    last_acc = acc
    last_hp = copy.deepcopy(best_hp)

    run_id = time.strftime("%Y%m%d-%H%M%S")
    # Fill run_id into CL rows for traceability
    for row in cl_rows:
        row["run_id"] = run_id

    run_logs, round_logs = [], []

    io_root = str(output_dir / "runs" / run_id)
    os.makedirs(io_root, exist_ok=True)

    def _build_state(round_id, acc_global, loss_global, ema_loss, forget_mean, forget_max, divergence, bytes_last_round, client_snapshots):
        return {
            "round_id": int(round_id),
            "global": {
                "acc": float(acc_global),
                "loss": float(loss_global),
                "ema_loss": float(ema_loss),
                "forget_mean": float(forget_mean),
                "forget_max": float(forget_max),
                "divergence": float(divergence),
                "bytes_last_round": int(bytes_last_round),
            },
            "clients": client_snapshots,  # list of dicts with vloss, vacc, last_lr, last_replay_ratio, last_ewc_lambda, etc.
        }

    def _safe_last_lr(c, fallback_lr):
        try:
            return float(c.optimizer.param_groups[0]["lr"])
        except Exception:
            return float(fallback_lr)
        
    # ---------------------------
    # Training rounds
    # ---------------------------
    bytes_last_round = 0  # carried into the next round's state
    bytes_cum = 0
    aulc_running = 0.0
    # --- metrics accumulators ---
    acc_hist = []            # for AULC
    comm_bytes_cum = 0       # cumulative comm

    parent_run_id = run_id
    start_round = 0
    source_checkpoint_sha256 = None
    starting_state_hash = None
    if resume_payload is not None:
        global_model.load_state_dict(resume_payload["global_model"])
        if len(resume_payload["clients"]) != len(clients):
            raise RuntimeError("checkpoint client count differs")
        for client, saved in zip(clients, resume_payload["clients"]):
            if int(saved["cid"]) != int(client.cid):
                raise RuntimeError("checkpoint client ordering differs")
            client.optimizer.load_state_dict(saved["optimizer"])
            client.replay.load_state_dict(saved["replay"])
            client.load_persistent_state_dict(saved["persistent"])
            client.update_energy_rows = []
        subspace_instrumentation.load_state_dict(resume_payload["subspace"])

        metrics = resume_payload["metrics"]
        acc = metrics["acc"]
        per_class = metrics["per_class"].copy()
        best_recall = metrics["best_recall"].copy()
        forgetting = metrics["forgetting"].copy()
        global_loss = metrics["global_loss"]
        ema_loss = metrics["ema_loss"]
        div_norm = metrics["div_norm"]
        last_acc = metrics["last_acc"]
        last_hp = copy.deepcopy(metrics["last_hp"])
        best_global_acc = metrics["best_global_acc"]
        best_state = copy.deepcopy(metrics["best_state"])
        best_hp = copy.deepcopy(metrics["best_hp"])
        best_round = metrics["best_round"]
        rollback_flag = metrics["rollback_flag"]
        rollback_round = metrics["rollback_round"]
        aulc_running = metrics["aulc_running"]
        bytes_last_round = metrics["bytes_last_round"]
        bytes_cum = metrics["bytes_cum"]
        acc_hist = copy.deepcopy(metrics["acc_hist"])
        comm_bytes_cum = metrics["comm_bytes_cum"]

        start_round = int(resume_payload["next_round"])
        if start_round not in {1, 5}:
            raise RuntimeError(
                f"common-state branch checkpoint must be pre-round 1 or 5, got {start_round}"
            )
        parent_run_id = str(resume_payload["parent_run_id"])
        source_checkpoint_sha256 = resume_metadata["sha256"]
        starting_state_hash = resume_metadata["starting_state_hash"]
        # This must be the final restoration action before the round starts.
        restore_rng_state(resume_payload["rng"], g)
        del resume_payload
        print(
            f"[Checkpoint] restored parent={parent_run_id} pre-round={start_round} "
            f"sha256={source_checkpoint_sha256}",
            flush=True,
        )
        if requires_endpoint_equality(args.update_control, args.projection_lambda):
            reference_document = json.loads(
                Path(args.determinism_reference).read_text(encoding="utf-8")
            )
            reference_identity = {
                "parent_run_id": reference_document.get("parent_run_id"),
                "executed_round": reference_document.get("executed_round"),
                "source_checkpoint_sha256": reference_document.get(
                    "source_checkpoint_sha256"
                ),
            }
            expected_identity = {
                "parent_run_id": parent_run_id,
                "executed_round": start_round,
                "source_checkpoint_sha256": source_checkpoint_sha256,
            }
            if reference_identity != expected_identity:
                raise RuntimeError(
                    "determinism reference does not belong to this checkpoint: "
                    f"expected={expected_identity}, found={reference_identity}"
                )
        else:
            gate_dir = Path(args.determinism_gate_dir)
            gate_path = (
                gate_dir
                / f"determinism_round_{start_round:02d}_lambda000_PASS.json"
            )
            validate_determinism_gate(
                gate_path,
                parent_run_id=parent_run_id,
                executed_round=start_round,
                source_checkpoint_sha256=source_checkpoint_sha256,
            )

    stop_round = start_round + 1 if args.one_round else args.rounds
    captured_checkpoint_metadata = {}
    branch_client_epoch_counts = {int(client.cid): 0 for client in clients}
    branch_client_optimizer_step_counts = {int(client.cid): 0 for client in clients}

    def _metric_state():
        return {
            "acc": float(acc),
            "per_class": per_class.copy(),
            "best_recall": best_recall.copy(),
            "forgetting": forgetting.copy(),
            "global_loss": float(global_loss),
            "ema_loss": float(ema_loss),
            "div_norm": float(div_norm),
            "last_acc": float(last_acc),
            "last_hp": copy.deepcopy(last_hp),
            "best_global_acc": float(best_global_acc),
            "best_state": copy.deepcopy(best_state),
            "best_hp": copy.deepcopy(best_hp),
            "best_round": int(best_round),
            "rollback_flag": bool(rollback_flag),
            "rollback_round": int(rollback_round),
            "aulc_running": float(aulc_running),
            "bytes_last_round": int(bytes_last_round),
            "bytes_cum": int(bytes_cum),
            "acc_hist": copy.deepcopy(acc_hist),
            "comm_bytes_cum": int(comm_bytes_cum),
        }

    def _checkpoint_payload(next_round):
        return {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "next_round": int(next_round),
            "parent_run_id": parent_run_id,
            "global_model": copy.deepcopy(global_model.state_dict()),
            "clients": [
                {
                    "cid": int(client.cid),
                    "optimizer": copy.deepcopy(client.optimizer.state_dict()),
                    "replay": client.replay.state_dict(),
                    "persistent": client.persistent_state_dict(),
                    # Local weights are discarded by the next-round broadcast.
                    "local_model_hash": model_state_sha256(client.model.state_dict()),
                }
                for client in clients
            ],
            "subspace": subspace_instrumentation.state_dict(),
            "metrics": _metric_state(),
            "rng": capture_rng_state(g),
            "data_state": data_state,
            "manifest": current_manifest,
        }

    for r in range(start_round, stop_round):

        # ---- Broadcast global model to all clients (FedAvg step 1) ----
        for c in clients:
            c.model.load_state_dict(global_model.state_dict())
        
        acc_delta = float(acc - last_acc)

        # --- Build and write STATE JSON (once, at round start) ---
        client_snaps = []
        for c in clients:
            # robust last_lr: if optimizer exists use it, else fallback to last chosen HP or CLI LR
            if hasattr(c, "optimizer") and getattr(c.optimizer, "param_groups", None):
                _lr_snapshot = float(c.optimizer.param_groups[0]["lr"])
            else:
                _lr_snapshot = float(last_hp.get("lr", args.lr))

            # new batch size for THIS round for this client (size of incoming CL chunk)
            batches = cl_schedule[c.cid]
            nb = len(batches[r]) if r < len(batches) else len(batches[-1])

            client_snaps.append({
                "id": int(c.cid),
                "vloss": float(getattr(c, "_last_vloss", float("nan"))),
                "vacc": float(getattr(c, "_last_vacc", float("nan"))),
                "new_batch_size": int(nb),
                "replay_capacity": int(getattr(getattr(c, "replay", None), "capacity", 2000)),
                "last_lr": _lr_snapshot,
                "last_replay_ratio": float(last_hp.get("replay_ratio", 0.50)),
                "last_ewc_lambda": float(getattr(c, "_last_ewc_lambda", 0.0)),
            })

        state = {
            "round_id": int(r),
            "global": {
                "acc": float(acc),
                "last_acc": float(last_acc),
                "loss": float(global_loss),
                "ema_loss": float(ema_loss),
                "forget_mean": float(np.mean(forgetting)) if forgetting is not None else 0.0,
                "forget_max": float(np.max(forgetting)) if forgetting is not None else 0.0,
                "divergence": float(div_norm),
                "bytes_last_round": int(bytes_last_round),
                "bytes_cum": int(bytes_cum),
            },
            "clients": client_snaps,
        }
        write_state_json(io_root, r, state)

        # =========================================================
        # Decide action ONCE (by controller) -> validate ONCE
        # =========================================================
        if args.controller == "sft":
            # SFT: the tiny local LM returns JSON; we validate & clamp it.
            print(f"[DEBUG] Calling SFT controller at round {r}", flush=True)
            raw = sft_decide_action(state, model_dir="models/sft_distilgpt2_v2")
            action = validate_action(raw, n_clients=len(clients), policy_source="SFT_v2")
            hp_lr = float(args.lr)
            rep = float(action["client_params"][0]["replay_ratio"]) if action["client_params"] else 0.50
            hp_notes = "SFT_v2"

        elif args.controller == "mock":
            # Mock: synthetic policy for plumbing / dataset creation
            raw = mock_decide_action(state, n_clients=len(clients))
            action = validate_action(raw, n_clients=len(clients), policy_source="Mock")
            hp_lr = float(args.lr)
            rep = float(action["client_params"][0]["replay_ratio"]) if action["client_params"] else 0.50
            hp_notes = "Mock"

        elif args.controller == "lmss_api":
            # LMSS via API: LLM selects strategy_id, we expand deterministically
            raw = lmss_decide_action_api(state, compact_state_fn=_compact_state_for_sft, model="gpt-4o-mini")
            action = validate_action(raw, n_clients=len(clients), policy_source=raw.get("policy_source", "LMSS_API"))
            hp_lr = float(raw.get("lr", args.lr))
            rep = float(action["client_params"][0]["replay_ratio"]) if action["client_params"] else 0.50
            hp_notes = raw.get("policy_source", "LMSS_API")

        elif args.controller == "lmss_local":
            from src.policy.lmss_local import lmss_decide_action_local

            raw = lmss_decide_action_local(
                state,
                compact_state_fn=_compact_state_for_sft,
                model_name=getattr(args, "lmss_model", "Qwen/Qwen2.5-0.5B-Instruct"),
            )
            action = validate_action(raw, n_clients=len(clients), policy_source=raw.get("policy_source", "LMSS_LOCAL"))
            hp_lr = float(raw.get("lr", args.lr))
            rep = float(action["client_params"][0]["replay_ratio"]) if action["client_params"] else 0.50
            hp_notes = raw.get("policy_source", "LMSS_LOCAL")

        elif args.controller == "lmss_openrouter":
            raw = lmss_decide_action_openrouter(
                state,
                compact_state_fn=_compact_state_for_sft,
                model=getattr(args, "lmss_model", "openai/gpt-4o-mini"),
            )
            action = validate_action(raw, n_clients=len(clients), policy_source=raw.get("policy_source", "LMSS_OPENROUTER"))
            hp_lr = float(raw.get("lr", args.lr))
            rep = float(action["client_params"][0]["replay_ratio"]) if action["client_params"] else 0.50
            hp_notes = raw.get("policy_source", "LMSS_OPENROUTER")

        elif args.controller == "v4":
            # Controller V4: compute hp (lr/rep) from simple signals
            dacc = float(acc - last_acc)
            F_t  = float(np.mean(forgetting)) if forgetting is not None else 0.0
            L_ema = float(ema_loss)
            div   = float(div_norm)

            if rollback_flag:
                lr = float(best_hp["lr"])
                rep = float(best_hp["replay_ratio"])
                notes = [f"ROLLBACK(r{rollback_round}→best r{best_round})"]
                rollback_flag = False

            elif r < V4_WARMUP_ROUNDS:
                lr, rep = float(args.lr), 0.50
                notes = ["warmup (fixed defaults)"]

            else:
                lr, rep = float(last_hp["lr"]), float(last_hp["replay_ratio"])
                notes = ["policy_v4"]

                if abs(dacc) < V4_DEADBAND:
                    notes.append(f"deadband(|dacc|<{V4_DEADBAND})")
                else:
                    if F_t > V4_FORGET_THR or div > V4_DIV_THR:
                        rep += V4_REP_STEP_HIGH
                        notes.append("replay↑ (forget/div high)")
                    else:
                        rep -= V4_REP_STEP_LOW
                        notes.append("replay↓ (forget low)")

                    if dacc < -V4_DEADBAND:
                        lr /= V4_LR_COOLDOWN
                        notes.append("lr↓ (dacc<0)")
                    elif dacc > V4_DEADBAND and L_ema > 1.5:
                        lr *= V4_LR_BOOST
                        notes.append("lr↑ (loss high & improving)")

            # clamp hp
            lr  = max(V4_LR_MIN,  min(V4_LR_MAX,  lr))
            rep = max(V4_REP_MIN, min(V4_REP_MAX, rep))
            notes.append(f"clamped(lr∈[{V4_LR_MIN},{V4_LR_MAX}], rep∈[{V4_REP_MIN:.2f},{V4_REP_MAX:.2f}])")

            # build per-client scales by vloss rank (higher loss → lower scale)
            vlosses = []
            for c in clients:
                v = getattr(c, "_last_vloss", None)
                vlosses.append(float(v) if v is not None and not np.isnan(v) else float(global_loss))
            vl_min, vl_max = float(np.min(vlosses)), float(np.max(vlosses))
            rng_v = max(1e-8, vl_max - vl_min)

            # --- ✅ WARMUP FIX: force lr_scale=1.0 during warmup rounds ---
            if r < V4_WARMUP_ROUNDS:
                lr_scales = [1.0 for _ in clients]
                notes.append("warmup: lr_scale forced to 1.0")
            else:
                lr_scales = [
                    float(
                        max(
                            V4_CLIENT_LR_MIN,
                            min(
                                V4_CLIENT_LR_MAX,
                                V4_CLIENT_LR_MIN
                                + (1.0 - ((vlosses[i] - vl_min) / rng_v))
                                * (V4_CLIENT_LR_MAX - V4_CLIENT_LR_MIN),
                            ),
                        )
                    )
                    for i in range(len(clients))
                ]

            candidate = {
                "client_selection_k": len(clients),
                "aggregation": {"method": "FedAvg"},
                "client_params": [
                    {
                        "id": int(c.cid),
                        "replay_ratio": float(rep),
                        "lr_scale": float(lr_scales[i]),
                        "ewc_lambda": float(getattr(c, "_last_ewc_lambda", 0.0)),
                    }
                    for i, c in enumerate(clients)
                ],
            }

            action = validate_action(candidate, n_clients=len(clients), policy_source="ControllerV4")
            hp_lr = float(lr)
            hp_notes = " | ".join(notes)
            
        elif args.controller == "fixed":
            # fixed (paper CL defaults)
            candidate = {
                "client_selection_k": len(clients),
                "aggregation": {"method": "FedAvg"},
                "client_params": [
                    {"id": int(c.cid), "replay_ratio": 0.50, "lr_scale": 1.0, "ewc_lambda": 0.0}
                    for c in clients
                ],
            }
            action = validate_action(candidate, n_clients=len(clients), policy_source="Fixed")
            hp_lr = float(args.lr)
            rep = 0.50
            hp_notes = "fixed (paper CL default)"
        else:
            raise ValueError(f"Unknown controller: {args.controller}")

        # =========================================================
        # Apply the validated ACTION uniformly (HP + per-client LR)
        # =========================================================
        # replay ratio comes from first client entry
        rep_from_action = (
            float(action["client_params"][0]["replay_ratio"])
            if action.get("client_params") else 0.50
        )
        hp = {"lr": hp_lr, "replay_ratio": rep_from_action, "notes": hp_notes}

        cid2scale = {int(p["id"]): float(p.get("lr_scale", 1.0)) for p in action.get("client_params", [])}
        for c in clients:
            scale = cid2scale.get(int(c.cid), 1.0)
            for pg in c.optimizer.param_groups:
                pg["lr"] = hp["lr"] * scale
            c._last_lr_scale = float(scale)

        # policy line for logs
        F_t_print = float(np.mean(forgetting)) if forgetting is not None else 0.0
        print(
            f"[Policy r={r}] acc={acc:.3f} dacc={acc_delta:+.3f} F_t={F_t_print:.3f} Div={div_norm:.3f} "
            f"-> lr={hp['lr']:.5f}, replay={hp['replay_ratio']:.2f} ({hp['notes']})",
            flush=True,
        )

        # persist chosen hp for next round snapshots
        last_hp = {"lr": hp["lr"], "replay_ratio": hp["replay_ratio"], "notes": hp["notes"]}

        # ---- Write ACTION JSON exactly once per round ----
        write_action_json(io_root, r, action, policy_source=action.get("policy_source", controller_name))

        # =========================================================
        # Local training per client
        # =========================================================
        phase_id = min(r, args.cl_batches - 1)
        if subspace_instrumentation is not None:
            subspace_instrumentation.begin_round(phase_id)

        for c in clients:
            batches = cl_schedule[c.cid]
            if r < len(batches):
                batch_indices = batches[r]
                batch_id = r
            else:
                batch_indices = batches[-1]
                batch_id = len(batches) - 1

            c.loader = DataLoader(
                Subset(trainset_full, batch_indices),
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True,
                worker_init_fn=seed_worker,
                generator=g,
            )

            print(f"[Round {r}] client {c.cid}: CL batch {batch_id+1}/{len(batches)} "
                  f"(new={len(batch_indices)}; replay≈{hp['replay_ratio']:.2f}, LR_scale={c._last_lr_scale:.2f})",
                  flush=True)

            epoch_budget = local_epoch_budget(args.epochs, args.branch_local_epochs)
            for e in range(epoch_budget):
                avg_loss, epoch_acc, stop = c.train_one_epoch(
                    replay_ratio=hp["replay_ratio"],
                    epoch=e,
                    total_epochs=args.epochs,
                    log_interval=args.log_interval,
                    projection_lambda=args.projection_lambda,
                    update_control=args.update_control,
                    shrinkage_factors=(
                        {
                            target.name: shrinkage_schedule[(r, c.cid, target.name)]
                            for target in DEFAULT_RESNET18_TARGETS
                        }
                        if shrinkage_schedule is not None
                        else None
                    ),
                    round_id=r,
                )
                run_logs.append({
                    "run_id": run_id, "tag": args.tag, "round": r, "client": c.cid,
                    "epoch": e + 1,
                    "lr": float(c.optimizer.param_groups[0]["lr"]),
                    "replay_ratio": float(hp["replay_ratio"]),
                    "cl_batch": batch_id + 1,
                    "cl_batch_size": len(batch_indices),
                    "train_loss": float(avg_loss),
                    "train_acc": float(epoch_acc),
                    "val_loss": float(getattr(c, "_last_vloss", float("nan"))),
                    "val_acc": float(getattr(c, "_last_vacc", float("nan"))),
                })
                if args.resume_checkpoint:
                    branch_client_epoch_counts[int(c.cid)] += 1
                    branch_client_optimizer_step_counts[int(c.cid)] += len(c.loader)
                if should_stop_local_training(stop, args.branch_local_epochs):
                    print(f"[Client {c.cid}] Early stopping (patience {c.early_patience})", flush=True)
                    break

        subspace_metrics = {}
        if subspace_instrumentation is not None:
            subspace_metrics = subspace_instrumentation.end_round()
            print(
                f"[Subspace r={r}] rank={subspace_metrics['protected_basis_rank']} "
                f"Ein={subspace_metrics['gradient_energy_inside']:.6g} "
                f"Eout={subspace_metrics['gradient_energy_outside']:.6g} "
                f"beta_hat={subspace_metrics['beta_hat']:.6g} "
                f"rho_hat={subspace_metrics['rho_hat']:.6g} "
                f"overhead={subspace_metrics['measurement_overhead_seconds']:.4f}s",
                flush=True,
            )

        # ---- Divergence (before FedAvg) ----
        with torch.no_grad():
            def flat_params(m: torch.nn.Module):
                return torch.cat([p.detach().float().view(-1).to(device) for p in m.parameters()])
            g_flat = flat_params(global_model)
            dists = []
            for c in clients:
                c_flat = flat_params(c.model)
                dists.append(torch.norm(c_flat - g_flat, p=2).item())
            if len(dists) > 1:
                div_norm = float(np.std(dists) / (np.median(dists) + 1e-8))
            else:
                div_norm = 0.0

        # ---- Aggregate & evaluate ----
        global_model = server.average([c.model for c in clients])
        last_acc = float(acc)
        acc, per_class = evaluate(global_model, device, test_loader)

        # running mean AULC up to round r
        aulc_running = ((aulc_running * r) + float(acc)) / max(1, (r + 1))

        # ---- Rollback check ----
        do_rollback = args.controller in ["v4", "lmss_local", "lmss_api", "lmss_openrouter", "sft"]

        if do_rollback and (acc < best_global_acc - V4_ROLLBACK_THR):
            global_model.load_state_dict(best_state)
            acc, per_class = evaluate(global_model, device, test_loader)
            forgetting = np.maximum(0.0, best_recall - per_class)

            print(
                f"[ROLLBACK r{r}] drop detected. "
                f"acc={acc:.3f} < best={best_global_acc:.3f} - {V4_ROLLBACK_THR}",
                flush=True,
            )

            rollback_flag = True
            rollback_round = r
        else:
            rollback_flag = False

        # ---- Update best state ----
        if acc > best_global_acc:
            best_global_acc = float(acc)
            best_state = copy.deepcopy(global_model.state_dict())
            best_hp = copy.deepcopy(hp)
            best_round = r

        # ---- Update loss/EMA/forgetting ----
        global_loss = evaluate_loss(global_model, device, test_loader)
        ema_loss = V4_EMA_ALPHA * global_loss + (1.0 - V4_EMA_ALPHA) * ema_loss
        forgetting = np.maximum(0.0, best_recall - per_class)
        best_recall = np.maximum(best_recall, per_class)

        # scalar forgetting metrics for logging/reward
        forget_mean_val = float(np.mean(forgetting)) if forgetting is not None else 0.0
        forget_max_val  = float(np.max(forgetting))  if forgetting is not None else 0.0

        # ---- Comm bytes for this round (used next round) ----
        model_size_bytes = sum(p.numel() for p in global_model.parameters()) * 4  # float32
        bytes_last_round = model_size_bytes * 2 * len(clients)  # up + down
        bytes_cum += int(bytes_last_round)
        print(f"[round {r}] AULC={aulc_running:.4f} | ACC={acc:.4f} | COMM_round={bytes_last_round:,} | COMM_cum={bytes_cum:,}")

        # ---- Round summary log ----
        round_logs.append({
            "run_id": run_id, "tag": args.tag, "round": r,
            "global_acc": float(acc),

            "lr": float(hp["lr"]),
            "replay_ratio": float(hp["replay_ratio"]),
            "notes": hp.get("notes", ""),

            "global_loss": float(global_loss),
            "ema_loss": float(ema_loss),

            # forgetting metrics
            "forget_mean": float(forget_mean_val),
            "forget_max": float(forget_max_val),

            # stability / divergence
            "divergence": float(div_norm),

            # best seen and rollback flag
            "best_acc_so_far": float(best_global_acc),
            "was_rollback": bool(rollback_flag),

            # communication + AULC
            "comm_bytes_round": int(bytes_last_round),
            "comm_bytes_cum": int(bytes_cum),
            "aulc_running": float(aulc_running),

            # measurement-only protected feature subspaces
            "protected_basis_exists": subspace_metrics.get("protected_basis_exists", False),
            "protected_basis_rank": subspace_metrics.get("protected_basis_rank", 0),
            "protected_basis_rank_by_layer": json.dumps(
                subspace_metrics.get("protected_basis_rank_by_layer", {}), sort_keys=True
            ),
            "protected_basis_rank_after_update": subspace_metrics.get(
                "protected_basis_rank_after_update", 0
            ),
            "protected_orthonormality_error": subspace_metrics.get(
                "protected_orthonormality_error", float("nan")
            ),
            "protected_orthonormality_error_by_layer": json.dumps(
                subspace_metrics.get("protected_orthonormality_error_by_layer", {}), sort_keys=True
            ),
            "gradient_energy_inside": subspace_metrics.get("gradient_energy_inside", float("nan")),
            "gradient_energy_outside": subspace_metrics.get("gradient_energy_outside", float("nan")),
            "beta_hat": subspace_metrics.get("beta_hat", float("nan")),
            "rho_hat": subspace_metrics.get("rho_hat", float("nan")),
            "basis_construction_seconds": subspace_metrics.get(
                "basis_construction_seconds", 0.0
            ),
            "measurement_overhead_seconds": subspace_metrics.get(
                "measurement_overhead_seconds", 0.0
            ),
            "gradient_measurement_count": subspace_metrics.get("gradient_measurement_count", 0),
        })
        
        print(f"[Round {r}] acc={acc:.3f} (best={best_global_acc:.3f})", flush=True)

        deterministic_summary = {
            key: value
            for key, value in round_logs[-1].items()
            if key not in {
                "run_id",
                "tag",
                # Wall-clock observations cannot be bitwise reproducible and
                # do not affect continuation semantics.
                "basis_construction_seconds",
                "measurement_overhead_seconds",
            }
        }
        endpoint = (
            endpoint_state(
                global_model,
                clients,
                subspace_instrumentation,
                deterministic_summary,
                history_state=_metric_state(),
                rng_state=capture_rng_state(g),
            )
            if branching_mode
            else None
        )

        if args.capture_common_checkpoints and r in {1, 5}:
            reference_path = checkpoint_dir / f"parent_endpoint_round_{r:02d}.json"
            source = captured_checkpoint_metadata[r]
            reference = write_endpoint_reference(
                reference_path,
                endpoint,
                {
                    "parent_run_id": parent_run_id,
                    "executed_round": int(r),
                    "projection_lambda": 0.0,
                    "source_checkpoint": source["checkpoint"],
                    "source_checkpoint_sha256": source["sha256"],
                    "excluded_nondeterministic_summary_fields": [
                        "basis_construction_seconds",
                        "measurement_overhead_seconds",
                    ],
                },
            )
            print(
                f"[Checkpoint] wrote parent endpoint reference {reference_path} "
                f"hash={reference['endpoint_state_hash']}",
                flush=True,
            )

        if args.resume_checkpoint:
            endpoint_hash = fingerprint_sha256(endpoint)
            branch_metadata = {
                "source_checkpoint": str(Path(args.resume_checkpoint)),
                "source_checkpoint_sha256": source_checkpoint_sha256,
                "starting_state_hash": starting_state_hash,
                "endpoint_global_model_hash": model_state_sha256(
                    global_model.state_dict()
                ),
                "endpoint_state_hash": endpoint_hash,
                "parent_run_id": parent_run_id,
                "branch_run_id": run_id,
                "executed_round": int(r),
                "projection_lambda": float(args.projection_lambda),
                **branch_control_metadata(
                    update_control=args.update_control,
                    branch_local_epochs=args.branch_local_epochs,
                    shrinkage_schedule=args.shrinkage_schedule,
                    client_epoch_counts=branch_client_epoch_counts,
                    client_optimizer_step_counts=branch_client_optimizer_step_counts,
                ),
                "determinism_gate": None,
            }
            if requires_endpoint_equality(
                args.update_control, args.projection_lambda
            ):
                gate = verify_endpoint_reference(args.determinism_reference, endpoint)
                branch_metadata["determinism_gate"] = gate
                report_name = (
                    f"determinism_round_{r:02d}_lambda000_"
                    f"{'PASS' if gate['passed'] else 'FAIL'}.json"
                )
                report_path = output_dir / report_name
                report_path.write_text(
                    json.dumps(branch_metadata, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                if not gate["passed"]:
                    raise RuntimeError(
                        "DETERMINISM GATE FAILED; no branch CSVs were written. "
                        f"Exact differences: {report_path}"
                    )
                print(f"[Determinism] exact continuation PASS: {report_path}", flush=True)
            metadata_path = output_dir / "branch_metadata.json"
            metadata_path.write_text(
                json.dumps(branch_metadata, indent=2, sort_keys=True),
                encoding="utf-8",
            )

        if args.capture_common_checkpoints and (r + 1) in {1, 5}:
            next_round = r + 1
            checkpoint_path = checkpoint_dir / f"pre_round_{next_round:02d}.pt"
            metadata = save_checkpoint(
                checkpoint_path, _checkpoint_payload(next_round)
            )
            captured_checkpoint_metadata[next_round] = metadata
            gib = metadata["size_bytes"] / (1024 ** 3)
            print(
                f"[Checkpoint] wrote {checkpoint_path} size={metadata['size_bytes']} "
                f"bytes ({gib:.3f} GiB) sha256={metadata['sha256']}",
                flush=True,
            )

    # ---------------------------
    # Write CSVs
    # ---------------------------
    results_path = output_dir / f"fcl_run_results_{run_id}_{args.tag}.csv"
    summary_path = output_dir / f"fcl_run_summary_{run_id}_{args.tag}.csv"
    cl_path = output_dir / f"fcl_run_cl_batches_{run_id}_{args.tag}.csv"
    pd.DataFrame(run_logs).to_csv(results_path, index=False)
    pd.DataFrame(round_logs).to_csv(summary_path, index=False)
    pd.DataFrame(cl_rows).to_csv(cl_path, index=False)
    update_energy_rows = [
        {"run_id": run_id, "tag": args.tag, **row}
        for client in clients
        for row in client.update_energy_rows
    ]
    step_energy_path = output_dir / f"fcl_run_update_energy_steps_{run_id}_{args.tag}.csv"
    round_energy_path = output_dir / f"fcl_run_update_energy_rounds_{run_id}_{args.tag}.csv"
    if update_energy_rows:
        pd.DataFrame(update_energy_rows).to_csv(step_energy_path, index=False)
        round_energy_rows = aggregate_update_energy_records(
            update_energy_rows, rounds=range(start_round, stop_round)
        )
        for row in round_energy_rows:
            row.update(
                {
                    "run_id": run_id,
                    "tag": args.tag,
                    "update_control": args.update_control,
                    "projection_lambda": args.projection_lambda,
                }
            )
        pd.DataFrame(round_energy_rows).to_csv(round_energy_path, index=False)
    print("✓ Wrote CSVs:",
          results_path, summary_path, cl_path, flush=True)
    if update_energy_rows:
        print("✓ Wrote update-energy CSVs:", step_energy_path, round_energy_path, flush=True)
    if subspace_instrumentation is not None:
        subspace_instrumentation.close()


if __name__ == "__main__":
    main()
