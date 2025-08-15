#!/usr/bin/env python3
# prod_grade_olmo2_fixedsplit.py
#
# Example:
#   python prod_grade_olmo2_fixedsplit.py \
#     --localization_method random_greedy \
#     --model_name allenai/OLMo-2-0425-1B \
#     --revision stage1-step140000-tokens294B \
#     --memorized_jsonl /data/mem_fixed_64_48.jsonl \
#     --clean_mode text_windows \
#     --clean_text /data/pile10k.txt \
#     --prefix_len 64 --suffix_len 48 --clean_block_size 112 \
#     --batch_size 16 --ratio 0.01 --epochs 1 --lr 0.1 --momentum 0.9 --weight_decay 5e-4 \
#     --loss_weighting 0.05 --include_gate false
#
# Alternatives for clean set:
#   --clean_mode pt_cache --clean_pt_path ../data/pythia_mem_data/pile_random_batch.pt
#
# Behavior:
# - Fixed-length memorization check: generate exactly suffix_len tokens from prefix_len prompt.
# - Only MLP up_proj & down_proj are trainable (default). Add --include_gate true to also include gate_proj.
# - Saves pre- and post-edit mem_seq and a results CSV.

import os
import math
import time
import argparse
import json
import copy
from typing import List, Dict, Any

import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

# Extend sys.path to prefer local copies under src/localize/, then fall back to project root
import sys
LOCAL_DIR = os.path.dirname(__file__)
if LOCAL_DIR not in sys.path:
    sys.path.insert(0, LOCAL_DIR)
REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

# Levenshtein and nDCG helpers (optional)
try:
    from test_memorization_levenshtein import (
        compute_memorization_metrics_levenshtein as compute_lev_metrics,
        compute_memorization_metrics_fixed_ids as compute_lev_fixed,
    )
    from data.baseline_generator import (
        get_baseline_predictions,
        TextChunkDataset as NDCGTextChunkDataset,
    )
    from evaluators import NDCGEvaluator, generate_beam_sequences
    HAVE_AUX_EVAL = True
except Exception:
    HAVE_AUX_EVAL = False

# --- existing project imports (unchanged) ---
from src.localize.neuron.neuron_utils import (
    apply_ablation_mask_to_base_model,
    set_model_attributes,
)
from neuron.neuron_utils import perplexity
from neuron.zero_out import fast_zero_out_vector  # noqa: F401
from neuron.activations import largest_act        # noqa: F401
from neuron.slimming import patch_slim, reinit_slim, slim  # noqa: F401
from neuron.hard_concrete import (               # noqa: F401
    patch_hardconcrete,
    reinit_hardconcrete,
    transpose_conv1d,
    hard_concrete,
)
from neuron.integrated_gradients import ig_full_data  # noqa: F401

from weight.greedy import do_greedy                  # noqa: F401
from weight.durable import do_durable                # noqa: F401
from weight.obs import do_obs                        # noqa: F401
from weight.random_subnet import do_random
from weight.random_subnet_greedy import do_random_greedy

from localizing_memorization import check_existance, check_basic_stats_existance
# ----------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------
# Data + utils
# --------------------------
def sort_metrics(args, perc_mem, perp, total_time):
    data = vars(args).copy()
    stat_dict = {"perc": [perc_mem], "perp": [perp], "total_time": total_time}
    data.update(stat_dict)
    return data


class FixedSplitMemDataset(Dataset):
    """
    JSONL with each line containing either:
      { "prefix_ids": [...], "suffix_ids": [...] }
    or
      { "prefix_text": "...", "suffix_text": "..." }  (tokenized here)
    or
      { "prefix": "...", "target_suffix": "..." }     (tokenized here)
    This dataset ENFORCES exact (prefix_len, suffix_len) via cropping/padding rules below.
    """
    def __init__(self, path: str, tokenizer, prefix_len: int, suffix_len: int):
        self.prefix_len = prefix_len
        self.suffix_len = suffix_len
        self.PAD = tokenizer.pad_token_id
        self.tok = tokenizer
        if self.PAD is None:
            raise ValueError("Tokenizer must define pad_token_id")

        self.items: List[Dict] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line)
                pre_ids, suf_ids = self._normalize(ex)
                if pre_ids is None:
                    continue
                self.items.append({"prefix": pre_ids, "suffix": suf_ids})

    def _normalize(self, ex):
        # get or compute ids
        if "prefix_ids" in ex and "suffix_ids" in ex:
            pre_ids, suf_ids = ex["prefix_ids"], ex["suffix_ids"]
        else:
            pre_text = ex.get("prefix_text") or ex.get("prefix") or ""
            suf_text = ex.get("suffix_text") or ex.get("target_suffix") or ""
            if not pre_text or not suf_text:
                return None, None
            pre_ids = self.tok(pre_text, add_special_tokens=False).input_ids
            suf_ids = self.tok(suf_text, add_special_tokens=False).input_ids

        # enforce suffix length: must be >= suffix_len; keep first suffix_len
        if len(suf_ids) < self.suffix_len:
            return None, None
        suf_ids = suf_ids[:self.suffix_len]

        # enforce prefix length: keep last prefix_len; left-pad if needed
        if len(pre_ids) >= self.prefix_len:
            pre_ids = pre_ids[-self.prefix_len:]
        else:
            pre_ids = [self.PAD] * (self.prefix_len - len(pre_ids)) + pre_ids

        return pre_ids, suf_ids

    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]


def make_fixed_collate(pad_id: int, prefix_len: int, suffix_len: int):
    def _collate(batch):
        B = len(batch)
        prompt = torch.full((B, prefix_len), pad_id, dtype=torch.long)
        prompt_attn = torch.zeros((B, prefix_len), dtype=torch.long)
        gold = torch.full((B, suffix_len), pad_id, dtype=torch.long)
        full = torch.full((B, prefix_len + suffix_len), pad_id, dtype=torch.long)

        for i, ex in enumerate(batch):
            pre = torch.tensor(ex["prefix"], dtype=torch.long)
            suf = torch.tensor(ex["suffix"], dtype=torch.long)
            # prompt
            prompt[i] = pre
            prompt_attn[i] = (pre != pad_id).long()
            # gold
            gold[i] = suf
            # full for mem_seq caching
            full[i] = torch.cat([pre, suf], dim=0)

        return {"prompt": prompt, "prompt_attn": prompt_attn, "gold": gold, "full": full}
    return _collate


class CleanTextDataset(Dataset):
    """Reads a single text file and yields fixed windows (block_size) of token ids."""
    def __init__(self, path: str, tokenizer, block_size: int):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        ids = tokenizer(text, add_special_tokens=False).input_ids
        self.blocks = []
        for s in range(0, len(ids) - block_size + 1, block_size):
            self.blocks.append(torch.tensor(ids[s:s + block_size], dtype=torch.long))

    def __len__(self): return len(self.blocks)
    def __getitem__(self, i): return self.blocks[i]


def load_clean_perplexity_and_extra(args, tokenizer, block_size):
    """
    Returns:
      perp_loader: DataLoader for perplexity
      extra_data:  Tensor [M, block_size] for weight methods
    """
    if args.clean_mode == "pt_cache":
        # Reuse their PT cache if given; try to conform shape to [*, block_size]
        raw = torch.load(args.clean_pt_path, map_location="cpu")
        # Attempt best-effort reshape if it's 1D/unknown
        if raw.dim() == 1:
            n = (raw.numel() // block_size) * block_size
            raw = raw[:n].view(-1, block_size)
        elif raw.dim() == 2 and raw.size(1) != block_size:
            # center crop or pad columns to block_size
            L = raw.size(1)
            if L > block_size:
                raw = raw[:, :block_size]
            else:
                pad = torch.full((raw.size(0), block_size - L), tokenizer.pad_token_id, dtype=raw.dtype)
                raw = torch.cat([raw, pad], dim=1)
        # split half for perp, half for extra
        mid = max(raw.size(0) // 2, 1)
        perp_tensor = raw[:mid]
        extra_tensor = raw[mid:] if raw.size(0) > mid else raw[:mid]
        perp_loader = DataLoader(perp_tensor, batch_size=32, shuffle=False)
        return perp_loader, extra_tensor

    # Default: process pile10k (or any text) into windows
    clean_ds = CleanTextDataset(args.clean_text, tokenizer, block_size=block_size)
    mid = max(len(clean_ds) // 2, 1)
    perp_loader = DataLoader(torch.utils.data.Subset(clean_ds, range(0, mid)), batch_size=32, shuffle=False)
    extra_loader = DataLoader(torch.utils.data.Subset(clean_ds, range(mid, len(clean_ds))), batch_size=32, shuffle=False)
    extra_batches = [b for b in extra_loader]
    extra_tensor = torch.vstack(extra_batches) if extra_batches else torch.empty(0, block_size, dtype=torch.long)
    return perp_loader, extra_tensor


def build_sequences_for_lev(jsonl_path: str, tokenizer, limit: int | None = None) -> List[Dict]:
    """
    Build a sequences list expected by compute_lev_metrics from a JSONL where
    each line may contain either ids or text for prefix/suffix.
    Output dict has keys: 'prompt', 'suffix', 'source'.
    """
    seqs: List[Dict] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            # Accept multiple field names
            p_txt = ex.get("prefix_text") or ex.get("prefix")
            s_txt = ex.get("suffix_text") or ex.get("target_suffix") or ex.get("suffix")
            # If only ids provided, decode to text
            if (p_txt is None) and ("prefix_ids" in ex):
                try:
                    p_txt = tokenizer.decode(ex["prefix_ids"], skip_special_tokens=True)
                except Exception:
                    p_txt = None
            if (s_txt is None) and ("suffix_ids" in ex):
                try:
                    s_txt = tokenizer.decode(ex["suffix_ids"], skip_special_tokens=True)
                except Exception:
                    s_txt = None
            if not p_txt or not s_txt:
                continue
            seqs.append({"prompt": p_txt, "suffix": s_txt, "source": ex.get("source", "mem_jsonl")})
            if limit is not None and len(seqs) >= limit:
                break
    return seqs


def _topk_over_loader(model, loader, k: int) -> Any:
    import numpy as _np
    device = next(model.parameters()).device
    out_chunks = []
    for batch in loader:
        if isinstance(batch, dict):
            ids = batch.get("prompt")  # not our case; perp_loader yields tensors
            if ids is None:
                ids = batch
        else:
            ids = batch
        ids = ids.to(device)
        with torch.no_grad():
            logits = model(ids).logits[:, :-1, :]
        topk_idx = torch.topk(logits, k, dim=-1).indices  # [B, L-1, k]
        flat = topk_idx.reshape(-1, k).cpu().numpy()
        out_chunks.append(flat)
    return _np.vstack(out_chunks) if out_chunks else _np.zeros((0, k), dtype="int32")


def _ndcg_from_topk(baseline_topk: Any, cand_topk: Any, k: int) -> float:
    import numpy as _np
    assert baseline_topk.shape == cand_topk.shape and baseline_topk.shape[1] == k
    # baseline positions 0..k-1 → relevance k..1
    rank_rel = _np.arange(k, 0, -1, dtype=_np.float32)
    discount = 1.0 / _np.log2(_np.arange(2, k + 2, dtype=_np.float32))
    idcg = (rank_rel * discount).sum()
    # Membership: for each token, candidate ids shape (k,), baseline shape (k,)
    # Expand to (k, k) compare, then take max relevance along baseline axis
    # Vectorised over rows using broadcasting
    # cand_topk: [N,k] -> [N,k,1]; baseline_topk: [N,k] -> [N,1,k]
    same = (cand_topk[:, :, None] == baseline_topk[:, None, :])
    # relevance per candidate position is the baseline rank value where equal
    # Build relevance map per row by multiplying with rank_rel (broadcast to [1,1,k]) and taking max over baseline axis
    rel = (same * rank_rel[None, None, :]).max(axis=2)  # [N, k]
    gains = rel * discount[None, :]
    dcg = gains.sum(axis=1)  # [N]
    return float(dcg.mean() / idcg) if dcg.size else 0.0


def _round_lev_metrics(m: Dict) -> Dict:
    if not m:
        return {}
    out: Dict = {}
    if "strict_acc" in m:
        out["strict_acc"] = round(float(m["strict_acc"]), 2)
    if "loose_acc" in m:
        out["loose_acc"] = round(float(m["loose_acc"]), 2)
    if "avg_levenshtein_norm" in m:
        out["avg_levenshtein_norm"] = round(float(m["avg_levenshtein_norm"]), 2)
    if "total" in m:
        out["total"] = int(m["total"])  # count stays integer
    return out


@torch.inference_mode()
def check_percent_memorized_fixed(mem_loader, perp_loader, suffix_len, model, pad_token_id):
    """Generate exactly suffix_len tokens; compare to gold next suffix_len tokens."""
    print(f"checking perc mem (fixed prompts, suffix_len={suffix_len})")
    memorized = 0
    total = 0
    kept_full = []

    # No progress bar
    for batch in mem_loader:
        prompt = batch["prompt"].to(model.device)
        prompt_attn = batch["prompt_attn"].to(model.device)
        gold = batch["gold"].to(model.device)     # [B, suffix_len]
        full = batch["full"]                      # keep on CPU

        outputs = model.generate(
            input_ids=prompt,
            attention_mask=prompt_attn,
            max_new_tokens=suffix_len,
            pad_token_id=pad_token_id,
            do_sample=False,
        )
        pred_next = outputs[:, -suffix_len:]      # [B, suffix_len]
        match_rows = (pred_next == gold).all(dim=1).cpu()
        if match_rows.any():
            kept_full.append(full[match_rows])

        total += prompt.size(0)
        memorized += int(match_rows.sum().item())

    perc_mem = memorized / max(total, 1)
    perp = perplexity(perp_loader, model)
    print(f"perc mem: {perc_mem:.2f}   perplexity(clean): {perp:.4f}")

    mem_seq = torch.cat(kept_full, dim=0) if kept_full else torch.empty(0, dtype=torch.long)
    return perc_mem, mem_seq, perp


def build_paths(model_name: str, revision: str | None):
    safe_model = model_name.replace("/", "__")
    safe_rev = (revision or "main").replace("/", "_")
    base_dir = os.path.join("..", "..", "model_ckpts", safe_model, safe_rev)
    edit_dir = base_dir + "_edit" + os.sep
    os.makedirs(edit_dir, exist_ok=True)
    return safe_model, safe_rev, base_dir, edit_dir


def set_trainable_mlp_projections(model, include_gate: bool):
    """
    Freeze everything except MLP {up_proj, down_proj} (and optionally gate_proj).
    Returns (num_trainable_params, total_params, list_of_kept_names).
    """
    keep_substrings = ["mlp.up_proj", "mlp.down_proj"]
    if include_gate:
        keep_substrings.append("mlp.gate_proj")

    kept = []
    total = 0
    trainable = 0

    for n, p in model.named_parameters():
        total += p.numel()
        keep = any(sub in n for sub in keep_substrings)
        p.requires_grad = bool(keep)
        if keep:
            kept.append(n)
            trainable += p.numel()

    print(f"[param-filter] trainable={trainable:,}  total={total:,}  keep={keep_substrings}")
    return trainable, total, kept


# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # model
    parser.add_argument("--model_name", type=str, default="allenai/OLMo-2-0425-1B")
    parser.add_argument("--revision", type=str, default=None)

    # method
    parser.add_argument("--localization_method", type=str, default="random_greedy",
                        choices=["greedy", "durable", "durable_agg", "random", "random_greedy", "act", "slim", "hc", "ig", "zero"])
    parser.add_argument("--ratio", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lambda_l1", type=float, default=1000.0)
    parser.add_argument("--stop_loss", type=float, default=1e-1)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--loss_weighting", type=float, default=0.05)

    # fixed split
    parser.add_argument("--prefix_len", type=int, default=64)
    parser.add_argument("--suffix_len", type=int, default=48)

    # data
    parser.add_argument("--memorized_jsonl", type=str, required=True, help="JSONL with prefix_ids/suffix_ids or text fields.")
    # Default to pt_cache with a prebuilt OLMo2-tokenized cache
    parser.add_argument("--clean_mode", type=str, default="pt_cache", choices=["text_windows", "pt_cache"])
    parser.add_argument("--clean_text", type=str, default=None, help="Only used if clean_mode=text_windows")
    parser.add_argument(
        "--clean_pt_path",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "data", "olmo2_clean_pt_cache_112.pt"),
        help="Default OLMo2 pt_cache (112-token windows)"
    )
    parser.add_argument("--clean_block_size", type=int, default=112, help="Defaults to 112 for pt_cache")

    # param subset
    parser.add_argument("--include_gate", type=lambda x: str(x).lower() in ["1", "true", "yes"], default=False)

    # misc/compat
    parser.add_argument("--seed", type=int, default=0)
    # keep these for CSV compatibility with older code (not used here)
    parser.add_argument("--prompt_len", type=int, default=0)
    parser.add_argument("--ig_steps", type=int, default=1)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--assess_mem", type=int, default=0)

    # aux metrics
    parser.add_argument("--with_lev", type=lambda x: str(x).lower() in ["1","true","yes"], default=True)
    parser.add_argument("--with_ndcg", type=lambda x: str(x).lower() in ["1","true","yes"], default=True)
    parser.add_argument("--ndcg_seq_len", type=int, default=1024)
    parser.add_argument("--ndcg_k", type=int, default=10)
    parser.add_argument("--ndcg_max_tokens", type=int, default=200000)
    parser.add_argument(
        "--ndcg_text",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "data", "pile10k_None.txt"),
        help="Path to pile10k text for nDCG (independent of clean_mode)"
    )
    parser.add_argument("--lev_limit", type=int, default=512, help="max mem examples for Levenshtein eval (None for all)")
    # Optional ordered split of mem set; 0 means no split
    parser.add_argument("--mem_train_count", type=int, default=0)
    parser.add_argument("--mem_val_count", type=int, default=0)
    # (legacy) mem split sizes were here; we now evaluate on full mem set again

    args = parser.parse_args()
    # Global seeds for reproducibility
    import numpy as _np, random as _random
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    _np.random.seed(args.seed)
    _random.seed(args.seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

    # paths
    safe_model, safe_rev, base_dir, edit_dir = build_paths(args.model_name, args.revision)
    args.model_path = os.path.join(base_dir, safe_model)  # informational
    args.results_path = os.path.join(edit_dir, "localization_results.csv")
    print("Model base dir:", base_dir)
    print("Results path:", args.results_path)
    # Beam output file lives alongside edited artifacts so both baseline and masked
    # continuations end up in the same file for a given model/revision.
    beam_file_path = os.path.join(edit_dir, f"beam_prefix_suffix_{safe_model}_{safe_rev}.txt")
    beam_prompts = [
        "For someone transitioning from a sedentary job to a more active routine, the safest way to ramp up cardio exercise is to",
        "During a summer thunderstorm, lightning forms when",
        "To keep houseplants healthy through the winter months, a gardener should first",
        "The invention of the printing press in the fifteenth century reshaped European society because",
        "When training a large language model on a multilingual corpus, one of the primary challenges is",
    ]

    # model/tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name) #, revision=args.revision)
    PAD = tokenizer.pad_token_id
    EOS = tokenizer.eos_token_id
    assert PAD is not None and EOS is not None, "pad/eos token ids must be defined"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        #revision=args.revision,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.generation_config.pad_token_id = PAD
    model = model.to(device)

    # param gating (default: up/down only; add gate via flag)
    _ = set_trainable_mlp_projections(model, include_gate=args.include_gate)

    # data loaders (full mem set or optional ordered split)
    mem_ds = FixedSplitMemDataset(args.memorized_jsonl, tokenizer, args.prefix_len, args.suffix_len)
    collate_fn = make_fixed_collate(PAD, args.prefix_len, args.suffix_len)

    from torch.utils.data import Subset
    mem_loader = None
    mem_loader_val = None
    n_train_used = 0
    n_val_used = 0
    if args.mem_train_count and args.mem_train_count > 0:
        n_total = len(mem_ds)
        n_train = min(args.mem_train_count, n_total)
        n_val = min(args.mem_val_count, max(0, n_total - n_train)) if args.mem_val_count else 0
        idx_train = list(range(0, n_train))
        idx_val = list(range(n_train, n_train + n_val)) if n_val > 0 else []
        mem_loader = DataLoader(Subset(mem_ds, idx_train), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        if idx_val:
            mem_loader_val = DataLoader(Subset(mem_ds, idx_val), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        n_train_used = len(idx_train)
        n_val_used = len(idx_val)
    else:
        mem_loader = DataLoader(mem_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        n_train_used = len(mem_ds)
        n_val_used = 0

    print(f"mem split: train={n_train_used} val={n_val_used}")

    block_size = args.clean_block_size or (args.prefix_len + args.suffix_len)
    if args.clean_mode == "text_windows" and not args.clean_text:
        raise ValueError("clean_mode=text_windows requires --clean_text")
    if args.clean_mode == "pt_cache" and not args.clean_pt_path:
        raise ValueError("clean_mode=pt_cache requires --clean_pt_path")

    perp_loader, extra_data = load_clean_perplexity_and_extra(args, tokenizer, block_size)

    # results existence check (safe)
    exists = 0
    if os.path.exists(args.results_path):
        try:
            existing_results = pd.read_csv(args.results_path)
            data = vars(args)
            ckpt_check_df = existing_results[data.keys()]
            exists = check_basic_stats_existance(data, ckpt_check_df)
        except Exception:
            exists = 0
    print("The basic stats exists:", exists)

    mem_seq_pre_path = os.path.join(edit_dir, f"mem_seq_{safe_model}_{safe_rev}_pre.pt")

    # base stats (pre-edit)
    print("PRE TRAIN")
    percent_mem, mem_seq, perp = check_percent_memorized_fixed(mem_loader, perp_loader, args.suffix_len, model, PAD)

    # Write baseline (pre-edit) beam-1 prefix/suffix pairs to file
    try:
        # Some tokenizers expose an absurdly large model_max_length (e.g., 1e30),
        # which breaks tokenization with "int too big to convert". Temporarily cap it.
        old_max_len = getattr(tokenizer, "model_max_length", None)
        need_cap = isinstance(old_max_len, int) and old_max_len > 100000
        if need_cap:
            tokenizer.model_max_length = 1024
        try:
            baseline_beams = generate_beam_sequences(
                model,
                tokenizer,
                prompts=beam_prompts,
                beam_width=5,
                max_new_tokens=50,
                early_stopping=True,
            )
        finally:
            if need_cap:
                tokenizer.model_max_length = old_max_len

        with open(beam_file_path, "w", encoding="utf-8") as f:
            f.write("BASELINE (beam_width=5, max_new_tokens=50)\n")
            for i, (prompt, beams) in enumerate(zip(beam_prompts, baseline_beams), start=1):
                full = beams[0] if beams else ""
                suffix = full[len(prompt):]
                f.write(f"Sequence {i}:\n")
                f.write(prompt + "\n")
                f.write(suffix + "\n\n")
        print(f"Wrote baseline beam prefix/suffix to: {beam_file_path}")
    except Exception as e:
        print(f"[warn] baseline beam generation/write failed: {e}")

    # Optional: Levenshtein and nDCG pre-edit
    lev_metrics = {}
    ndcg_score = None
    if HAVE_AUX_EVAL and args.with_lev:
        try:
            def _stack_pg(loader):
                ap, ag = [], []
                for b in loader:
                    ap.append(b["prompt"]) ; ag.append(b["gold"])
                return torch.vstack(ap), torch.vstack(ag)

            # Train or full set
            p_train, g_train = _stack_pg(mem_loader)
            lev_train = compute_lev_fixed(model, p_train, g_train, PAD, batch_size=args.batch_size)
            lev_train = _round_lev_metrics(lev_train)
            if lev_train:
                tag = "train" if args.mem_train_count else "all"
                print(
                    f"Levenshtein pre ({tag}): strict={lev_train.get('strict_acc')} "
                    f"loose={lev_train.get('loose_acc')} avg_norm={lev_train.get('avg_levenshtein_norm')} "
                    f"(N={lev_train.get('total')})"
                )
            # Optional val
            if mem_loader_val is not None:
                p_val, g_val = _stack_pg(mem_loader_val)
                lev_val = compute_lev_fixed(model, p_val, g_val, PAD, batch_size=args.batch_size)
                lev_val = _round_lev_metrics(lev_val)
                print(
                    f"Levenshtein pre (val): strict={lev_val.get('strict_acc')} "
                    f"loose={lev_val.get('loose_acc')} avg_norm={lev_val.get('avg_levenshtein_norm')} "
                    f"(N={lev_val.get('total')})"
                )
            elif args.mem_val_count and args.mem_val_count > 0:
                print("Levenshtein pre (val): N=0 (no val items in file/split)")
            lev_metrics = lev_train
            # (external n-8/8 validation removed)
        except Exception as e:
            print(f"[warn] Levenshtein metrics failed: {e}")
    # Compute nDCG on the same clean windows as perplexity (pt_cache), avoid baseline prints
    base_topk_for_ndcg = None
    if HAVE_AUX_EVAL and args.with_ndcg:
        try:
            base_topk_for_ndcg = _topk_over_loader(model, perp_loader, k=args.ndcg_k)
            ndcg_score = _ndcg_from_topk(base_topk_for_ndcg, base_topk_for_ndcg, args.ndcg_k)
            print(f"nDCG@{args.ndcg_k} pre (pt_cache): {ndcg_score:.4f}")
        except Exception as e:
            print(f"[warn] nDCG (pt_cache) failed: {e}")
    torch.save(mem_seq, mem_seq_pre_path)
    base_row = sort_metrics(args, percent_mem, perp, math.nan)
    # Attach aux metrics if any
    if lev_metrics:
        base_row.update({
            "lev_strict_acc": [lev_metrics.get("strict_acc")],
            "lev_loose_acc": [lev_metrics.get("loose_acc")],
            "lev_avg_norm": [lev_metrics.get("avg_levenshtein_norm")],
            "lev_total": [lev_metrics.get("total")],
        })
    if ndcg_score is not None:
        base_row.update({"ndcg": [ndcg_score]})
    base_df = pd.DataFrame.from_dict(base_row)
    base = 1

    args.unlearn_set_name = "mem"
    total_time = math.nan

    # localization
    if len(mem_seq) != 0:
        if args.localization_method in ["zero", "act", "ig", "slim", "hc"]:
            original_model = copy.deepcopy(model)
            try:
                set_model_attributes(model, args.model_name)
                set_model_attributes(original_model, args.model_name)
            except Exception as e:
                print(f"[warn] set_model_attributes failed for {args.model_name}: {e}")

        if args.localization_method in ["ig", "slim", "hc", "zero", "act"]:
            # neuron-level paths (not typical for random_greedy)
            if args.localization_method == "act":
                start = time.time()
                attributions = largest_act(
                    inner_dim=getattr(model, "inner_dim", None),
                    model=model,
                    inputs=mem_seq.to(model.device),
                    gold_set=None,
                    model_name=args.model_name,
                    prompt_len=args.prefix_len,
                )
                total_time = time.time() - start

            if args.localization_method == "slim":
                patch_slim(model); model.to(device)
                start = time.time()
                attributions = slim(
                    lr=args.lr, epoch=args.epochs, lambda_l1=args.lambda_l1,
                    stop_loss=args.stop_loss, threshold=1e-1,
                    model=model, inputs=mem_seq.to(model.device),
                    gold_set=None, batch_size=args.batch_size,
                )
                total_time = time.time() - start

            if args.localization_method == "hc":
                patch_hardconcrete(model, args.model_name, mask_p=0.5, beta=2/3); model.to(device)
                start = time.time()
                attributions = hard_concrete(
                    lr=args.lr, epoch=args.epochs, lambda_l1=args.lambda_l1,
                    stop_loss=args.stop_loss, threshold=1e-1,
                    model=model, inputs=mem_seq.to(model.device),
                    gold_set=None, batch_size=args.batch_size,
                )
                total_time = time.time() - start

            print("Applying ablation mask to model")
            model = apply_ablation_mask_to_base_model(
                attributions, model=original_model, ratio=args.ratio, model_name=args.model_name
            )

        else:
            # weight-level methods
            if args.localization_method == "greedy":
                start = time.time()
                model = do_greedy(extra_data, mem_seq, model, args.batch_size, args.ratio)
                total_time = time.time() - start

            if args.localization_method == "durable":
                start = time.time()
                model = do_durable(model, mem_seq, args.ratio, False)
                total_time = time.time() - start

            if args.localization_method == "durable_agg":
                start = time.time()
                model = do_durable(model, mem_seq, args.ratio, True)
                total_time = time.time() - start

            if args.localization_method == "random":
                start = time.time()
                model = do_random(
                    model, mem_seq, model.config.num_hidden_layers, args.ratio,
                    args.epochs, args.lr, args.momentum, args.weight_decay,
                    args.model_name, args.batch_size,
                )
                total_time = time.time() - start

            if args.localization_method == "random_greedy":
                start = time.time()
                model = do_random_greedy(
                    model,
                    mem_seq,                     # [N, prefix_len + suffix_len]
                    extra_data,                  # [M, block_size]
                    model.config.num_hidden_layers,
                    args.ratio,
                    args.epochs,
                    args.lr,
                    args.momentum,
                    args.weight_decay,
                    args.batch_size,
                    args.loss_weighting,
                    args.model_name,
                    seed=args.seed,
                    include_gate=args.include_gate,
                )
                total_time = time.time() - start

        # save edited model + post-edit eval
        method_dir = os.path.join(
            edit_dir, args.localization_method, args.unlearn_set_name, str(args.ratio)
        )
        if args.localization_method in ["hc", "slim"]:
            method_dir = os.path.join(method_dir, f"{args.epochs}/{args.lambda_l1}/{args.stop_loss}/{args.lr}")
        if args.localization_method in ["ig"]:
            method_dir = os.path.join(method_dir, "1")  # ig_steps placeholder
        if args.localization_method in ["random"]:
            method_dir = os.path.join(method_dir, f"{args.epochs}/{args.lr}/{args.momentum}/{args.weight_decay}")
        if args.localization_method in ["random_greedy"]:
            method_dir = os.path.join(method_dir, f"{args.epochs}/{args.lr}/{args.momentum}/{args.weight_decay}/{args.loss_weighting}")

        os.makedirs(method_dir, exist_ok=True)

        model_file_name = f"{safe_model}_{safe_rev}.pt"
        MODEL_PATH = os.path.join(method_dir, model_file_name)
        # Hide noisy model path prints
        torch.save({"model_state_dict": model.state_dict()}, MODEL_PATH)

        print("POST TRAIN")
        percent_mem, mem_seq_after, perp = check_percent_memorized_fixed(mem_loader, perp_loader, args.suffix_len, model, PAD)

        # Optional: post-edit Levenshtein and nDCG
        lev_metrics_post = {}
        ndcg_score_post = None
        if HAVE_AUX_EVAL and args.with_lev:
            try:
                def _stack_pg(loader):
                    ap, ag = [], []
                    for b in loader:
                        ap.append(b["prompt"]) ; ag.append(b["gold"])
                    return torch.vstack(ap), torch.vstack(ag)
                # Post on train or full
                p_train, g_train = _stack_pg(mem_loader)
                lev_post_train = compute_lev_fixed(model, p_train, g_train, PAD, batch_size=args.batch_size)
                lev_post_train = _round_lev_metrics(lev_post_train)
                tag = "train" if args.mem_train_count else "all"
                print(
                    f"Levenshtein post ({tag}): strict={lev_post_train.get('strict_acc')} "
                    f"loose={lev_post_train.get('loose_acc')} avg_norm={lev_post_train.get('avg_levenshtein_norm')} "
                    f"(N={lev_post_train.get('total')})"
                )
                if mem_loader_val is not None:
                    p_val, g_val = _stack_pg(mem_loader_val)
                    lev_post_val = compute_lev_fixed(model, p_val, g_val, PAD, batch_size=args.batch_size)
                    lev_post_val = _round_lev_metrics(lev_post_val)
                    print(
                        f"Levenshtein post (val): strict={lev_post_val.get('strict_acc')} "
                        f"loose={lev_post_val.get('loose_acc')} avg_norm={lev_post_val.get('avg_levenshtein_norm')} "
                        f"(N={lev_post_val.get('total')})"
                    )
                elif args.mem_val_count and args.mem_val_count > 0:
                    print("Levenshtein post (val): N=0 (no val items in file/split)")
                lev_metrics_post = lev_post_train
                # (external n-8/8 validation removed)
            except Exception as e:
                print(f"[warn] Levenshtein post-edit metrics failed: {e}")
        if HAVE_AUX_EVAL and args.with_ndcg and base_topk_for_ndcg is not None:
            try:
                cand_topk = _topk_over_loader(model, perp_loader, k=args.ndcg_k)
                ndcg_score_post = _ndcg_from_topk(base_topk_for_ndcg, cand_topk, args.ndcg_k)
                print(f"nDCG@{args.ndcg_k} post (pt_cache): {ndcg_score_post:.4f}")
            except Exception as e:
                print(f"[warn] nDCG post-edit (pt_cache) failed: {e}")
        # Append masked model (post-edit) beam-1 prefix/suffix pairs to same file
        try:
            old_max_len = getattr(tokenizer, "model_max_length", None)
            need_cap = isinstance(old_max_len, int) and old_max_len > 100000
            if need_cap:
                tokenizer.model_max_length = 1024
            try:
                masked_beams = generate_beam_sequences(
                    model,
                    tokenizer,
                    prompts=beam_prompts,
                    beam_width=5,
                    max_new_tokens=50,
                    early_stopping=True,
                )
            finally:
                if need_cap:
                    tokenizer.model_max_length = old_max_len

            with open(beam_file_path, "a", encoding="utf-8") as f:
                f.write("MASKED (beam_width=5, max_new_tokens=50)\n")
                for i, (prompt, beams) in enumerate(zip(beam_prompts, masked_beams), start=1):
                    full = beams[0] if beams else ""
                    suffix = full[len(prompt):]
                    f.write(f"Sequence {i}:\n")
                    f.write(prompt + "\n")
                    f.write(suffix + "\n\n")
            print(f"Appended masked beam prefix/suffix to: {beam_file_path}")
        except Exception as e:
            print(f"[warn] masked beam generation/write failed: {e}")
        mem_seq_post_path = os.path.join(method_dir, f"mem_seq_{safe_model}_{safe_rev}_post.pt")
        torch.save(mem_seq_after, mem_seq_post_path)
        print("path for the post edit mem_seq set:", mem_seq_post_path)

        ablate_row = sort_metrics(args, percent_mem, perp, total_time)
        if lev_metrics_post:
            ablate_row.update({
                "lev_strict_acc": [lev_metrics_post.get("strict_acc")],
                "lev_loose_acc": [lev_metrics_post.get("loose_acc")],
                "lev_avg_norm": [lev_metrics_post.get("avg_levenshtein_norm")],
                "lev_total": [lev_metrics_post.get("total")],
            })
        if ndcg_score_post is not None:
            ablate_row.update({"ndcg": [ndcg_score_post]})
        ablate_df = pd.DataFrame.from_dict(ablate_row)
        result = pd.concat([pd.DataFrame([]), base_df, ablate_df], axis=0, ignore_index=True)

        if os.path.exists(args.results_path):
            existing_results = pd.read_csv(args.results_path)
            existing_results = pd.concat([existing_results, result], axis=0, ignore_index=True)
            existing_results.to_csv(args.results_path, index=False)
        else:
            result.to_csv(args.results_path, index=False)

    else:
        # no matching mem examples; still write base row
        result = base_df
        if os.path.exists(args.results_path):
            existing_results = pd.read_csv(args.results_path)
            existing_results = pd.concat([existing_results, result], axis=0, ignore_index=True)
            existing_results.to_csv(args.results_path, index=False)
        else:
            result.to_csv(args.results_path, index=False)
