#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
svd_eval_multi_simple_suffix.py

Apply SVD (rank-ratio) compression sequentially across specified transformer MLP
projections (gate/up/down) and evaluate:
  - Levenshtein memorization metrics on an n-8,8 text dataset (prefix+suffix text)
  - nDCG@k on a cached pile10k stream
  - Optional clean perplexity
  - **Clean non-memorized windows (pile10k half, 64+48):**
      * avg normalized Levenshtein vs gold for BASELINE completions
      * avg normalized Levenshtein vs gold for EDITED completions
    (computed only on windows where the baseline completion != gold)

This mirrors the K-FAC simple_suffix script’s dataset handling, but uses SVD for
the projection compression and adds the two “clean non-mem” stats.
"""

import os
import sys
import io
import json
import time
import argparse
import contextlib
import hashlib
import pathlib
from datetime import datetime
from typing import Tuple, List, Dict

import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import logging as hf_logging

# ────────────────────────────────────────────────────────────────────────────────
# Ensure project root (update if needed) is on sys.path for local repo imports
# ────────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = "/mnt/polished-lake/home/siri/research/research/experiments/llm_memorization_toolkit"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Local repo imports
from test_memorization_levenshtein import (
    compute_memorization_metrics_levenshtein,
)
from evaluators import NDCGEvaluator, TextChunkDataset
from data.baseline_generator import get_baseline_predictions
from neuron.neuron_utils import perplexity

# Import the SVD treatment utilities provided
from svd_treatment import SVDTreatment

# ────────────────────────────────────────────────────────────────────────────────
# Globals and hard-coded paths (mirroring K-FAC simple_suffix script)
# ────────────────────────────────────────────────────────────────────────────────

# Central layer-of-interest (updated dynamically when applying per-layer)
LAYER_IDX = 15

# Dataset & nDCG sources
# Expects JSONL rows with text fields: 'prefix' (n-8) and 'target_suffix' (8)
DATASET_JSONL = "/mnt/polished-lake/home/siri/research/research/experiments/llm_memorization_toolkit/replicate_bsn/memorization/src/localize/olmo2_1b_large8_bfloat16.jsonl"
NDCG_FILE = "/mnt/polished-lake/home/siri/research/research/experiments/llm_memorization_toolkit/data/pile10k_None.txt"
CLEAN_BLOCK_SIZE = 112

# Cache directory for storing per-(layer, projection, rho) SVD-compressed weights
CACHE_DIR = "/mnt/polished-lake/home/siri/research/research/experiments/llm_memorization_toolkit/data/cache/svd_weights"

# Clean non-mem window spec (pile10k half windows)
NONMEM_PREFIX_LEN = 64
NONMEM_SUFFIX_LEN = 48


# ────────────────────────────────────────────────────────────────────────────────
# Utilities
# ────────────────────────────────────────────────────────────────────────────────

def _sanitize_filename_component(text: str) -> str:
    return ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in text)


def _rho_to_str(rho: float) -> str:
    return f"{rho:.3f}".replace('.', 'p')


def read_sequences_large8(jsonl_path: str) -> List[Dict[str, str]]:
    """
    Read JSONL rows with text fields and build sequences for Levenshtein eval.
    Expects keys 'prefix' and 'target_suffix' (n-8 prefix, 8-token suffix).
    Fallbacks: 'prefix_text' for prefix and 'suffix'/'suffix_text' for suffix.
    """
    sequences: List[Dict[str, str]] = []
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            ex = json.loads(line)
            p_txt = ex.get("prefix") or ex.get("prefix_text")
            s_txt = ex.get("target_suffix") or ex.get("suffix") or ex.get("suffix_text")
            if not p_txt or not s_txt:
                continue
            sequences.append({"prompt": p_txt, "suffix": s_txt, "source": ex.get("topic", "large8")})
    if not sequences:
        raise ValueError(f"No usable rows in {jsonl_path}; expected 'prefix' and 'target_suffix' text fields")
    return sequences


def load_model_and_tokenizer(model_name: str,
                             dtype: str = "float16",
                             quiet: bool = True) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    torch_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype]

    if quiet:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map="auto",
                trust_remote_code=True,
            )
    else:
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
    model.eval()
    return model, tok


class CachedTextChunkDataset(torch.utils.data.Dataset):
    """
    Tokenizes a text file once and caches token ids to disk. Subsequent runs
    reuse the cache to avoid repeated tokenization. Matches TextChunkDataset API.
    """
    def __init__(self,
                 filepath: str,
                 tokenizer,
                 seq_len: int = 1024,
                 max_tokens: int | None = None,
                 cache_dir: str | None = None):
        self.seq_len = seq_len
        self.tokens: List[int] = []

        path = pathlib.Path(filepath)
        cache_root = pathlib.Path(cache_dir) if cache_dir else pathlib.Path(os.path.dirname(__file__)) / "data"
        os.makedirs(cache_root, exist_ok=True)
        tok_name = getattr(tokenizer, "name_or_path", tokenizer.__class__.__name__)
        key = f"{path.name}::{tok_name}::{seq_len}::{max_tokens}"
        cache_hash = hashlib.md5(key.encode("utf-8")).hexdigest()[:8]
        cache_file = cache_root / f"tok_cache_{path.stem}_{cache_hash}.pt"

        if cache_file.exists():
            tensor = torch.load(cache_file, map_location="cpu")
            if not isinstance(tensor, torch.Tensor):
                raise RuntimeError(f"Corrupt token cache: {cache_file}")
            self.tokens = tensor.tolist()
            print(f"Loaded token cache: {cache_file}")
        else:
            tokens: List[int] = []
            with path.open("r", encoding="utf-8", errors="ignore") as fp:
                for line in fp:
                    ids = tokenizer.encode(line, add_special_tokens=False)
                    tokens.extend(ids)
                    if max_tokens and len(tokens) >= max_tokens:
                        tokens = tokens[:max_tokens]
                        break
            torch.save(torch.tensor(tokens, dtype=torch.long), cache_file)
            print(f"Saved token cache: {cache_file}")
            self.tokens = tokens

        n_full = len(self.tokens) // seq_len
        self.tokens = self.tokens[: n_full * seq_len]
        print(f"Total tokens: {len(self.tokens):,}")
        print(f"Sequences: {n_full:,}")

    def __len__(self):
        return len(self.tokens) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len
        return torch.tensor(self.tokens[start:end], dtype=torch.long)


# ────────────────────────────────────────────────────────────────────────────────
# SVD Application (per layer, gate/up/down projections)
# ────────────────────────────────────────────────────────────────────────────────

def _proj_name(layer_idx: int, which: str) -> str:
    assert which in ("up", "down", "gate")
    return f"model.layers.{layer_idx}.mlp.{which}_proj"


def _fro_energy_preserved(orig: torch.Tensor, approx: torch.Tensor) -> float:
    # Fraction of Frobenius norm energy preserved by approximation
    num = (orig - approx).float().pow(2).sum().item()
    den = orig.float().pow(2).sum().item() + 1e-12
    return max(0.0, 1.0 - (num / den))


def apply_pairwise_svd(model: AutoModelForCausalLM,
                       variance_up: float = 0.8,
                       variance_down: float = 0.9,
                       variance_gate: float = 1.0,
                       quiet: bool = True,
                       *,
                       model_name: str,
                       use_cache: bool = True,
                       refresh_cache: bool = False,
                       cache_dir: str = CACHE_DIR) -> None:
    """
    Apply SVD (rank-ratio) to {up, down, gate} projections of the current LAYER_IDX.
    Mirrors the K-FAC driver: supports caching and emits concise stats.
    """
    up_layer = model.model.layers[LAYER_IDX].mlp.up_proj
    down_layer = model.model.layers[LAYER_IDX].mlp.down_proj
    gate_layer = model.model.layers[LAYER_IDX].mlp.gate_proj

    # Treat 100% as "reuse original weights" (skip SVD entirely)
    reuse_up = variance_up >= 0.9999
    reuse_down = variance_down >= 0.9999
    reuse_gate = variance_gate >= 0.9999

    # Prepare cache paths
    os.makedirs(cache_dir, exist_ok=True)
    model_tag = _sanitize_filename_component(model_name)
    up_cache_path = os.path.join(cache_dir, f"{model_tag}__L{LAYER_IDX}__up__rho{_rho_to_str(variance_up)}__{up_layer.weight.dtype.__str__()}.pt")
    down_cache_path = os.path.join(cache_dir, f"{model_tag}__L{LAYER_IDX}__down__rho{_rho_to_str(variance_down)}__{down_layer.weight.dtype.__str__()}.pt")
    gate_cache_path = os.path.join(cache_dir, f"{model_tag}__L{LAYER_IDX}__gate__rho{_rho_to_str(variance_gate)}__{gate_layer.weight.dtype.__str__()}.pt")

    stats = {}

    def _apply_one(which: str, layer, rho: float, cache_path: str, reuse: bool):
        loaded_from_cache = False
        k = None
        energy = None
        orig_rank = min(layer.weight.shape)
        kept_ratio = None

        if reuse:
            msg = f"SVD {which}_proj (ρ={rho:.3f}): reused baseline weights (no SVD)"
            stats[which] = {"reuse": True, "rho": rho, "rank_kept": orig_rank, "rank_total": orig_rank, "energy_preserved": 1.0, "msg": msg}
            if quiet:
                print(msg)
            else:
                print(msg)
            return

        # Clone original BEFORE modifying for energy stats
        origW = layer.weight.detach().clone()

        if use_cache and (not refresh_cache) and os.path.exists(cache_path):
            with torch.no_grad():
                cached = torch.load(cache_path, map_location=layer.weight.device)
                layer.weight.copy_(cached.to(dtype=layer.weight.dtype, device=layer.weight.device))
            loaded_from_cache = True
            # Derive k from rho and compute energy against origW
            k = max(1, int(rho * orig_rank))
            energy = _fro_energy_preserved(origW, layer.weight.data)
            kept_ratio = k / max(1, orig_rank)
        else:
            # Apply SVD via helper on just this projection
            lname = _proj_name(LAYER_IDX, which)
            svd = SVDTreatment(model, [lname])
            # Keep specified fraction of singular values
            svd.apply_svd({lname: rho})
            k = svd.compression_stats[lname]['rank']
            kept_ratio = k / max(1, orig_rank)
            # Energy preserved (Frobenius) vs original
            energy = _fro_energy_preserved(svd.original_weights[lname], layer.weight.data)
            if use_cache:
                torch.save(layer.weight.detach().cpu(), cache_path)

        msg = (f"SVD {which}_proj (ρ={rho:.3f}): rank {k}/{orig_rank} "
               f"({kept_ratio:.1%}), energy preserved={energy:.3f}"
               + (" [cache]" if loaded_from_cache else ""))
        stats[which] = {"reuse": False, "rho": rho, "rank_kept": int(k), "rank_total": int(orig_rank),
                        "kept_ratio": float(kept_ratio), "energy_preserved": float(energy),
                        "cache": bool(loaded_from_cache), "msg": msg}
        if quiet:
            print(msg)
        else:
            print(msg)

    _apply_one("up", up_layer, variance_up, up_cache_path, reuse_up)
    _apply_one("down", down_layer, variance_down, down_cache_path, reuse_down)
    _apply_one("gate", gate_layer, variance_gate, gate_cache_path, reuse_gate)


# Default multi-layer configuration (edit or pass via --layers-json/--layers-file)
LAYER_TO_VARIANCES_DEFAULT: Dict[int, Dict[str, float]] = {
    # Example:
    # 0: {"gate": 0.6, "up": 0.6, "down": 1.0},
    # 15: {"gate": 0.6, "up": 0.9, "down": 1.0},
}


# ────────────────────────────────────────────────────────────────────────────────
# Clean non-memorized windows (pile10k half, 64+48) — helper utilities
# ────────────────────────────────────────────────────────────────────────────────

def _make_clean_half_windows(tokenizer,
                             text_path: str,
                             prefix_len: int = NONMEM_PREFIX_LEN,
                             suffix_len: int = NONMEM_SUFFIX_LEN,
                             max_tokens: int = 200_000) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Tokenize the text and take only the FIRST HALF of contiguous windows of size prefix_len+suffix_len.
    Returns:
      prompts [N, prefix_len], gold [N, suffix_len]
    """
    with open(text_path, "r", encoding="utf-8") as fh:
        text = fh.read()
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if max_tokens is not None:
        ids = ids[:max_tokens]
    block = prefix_len + suffix_len
    total_blocks = len(ids) // block
    if total_blocks == 0:
        return torch.empty(0, prefix_len, dtype=torch.long), torch.empty(0, suffix_len, dtype=torch.long)
    half_blocks = max(total_blocks // 2, 1)
    usable = ids[: half_blocks * block]
    prompts, suffixes = [], []
    for s in range(0, len(usable), block):
        seg = usable[s: s + block]
        prompts.append(seg[:prefix_len])
        suffixes.append(seg[prefix_len: prefix_len + suffix_len])
    return torch.tensor(prompts, dtype=torch.long), torch.tensor(suffixes, dtype=torch.long)


def _extract_new_tokens(gen: torch.Tensor, input_len: int, target_len: int, pad_id: int) -> torch.Tensor:
    new = gen[:, input_len:]
    if new.size(1) >= target_len:
        return new[:, :target_len]
    pad = torch.full((new.size(0), target_len - new.size(1)), pad_id, dtype=gen.dtype, device=gen.device)
    return torch.cat([new, pad], dim=1)


@torch.inference_mode()
def _batch_greedy_generate_exact(model,
                                 prompt_ids: torch.Tensor,
                                 suffix_len: int,
                                 pad_id: int,
                                 eos_id: int,
                                 batch_size: int = 32) -> torch.Tensor:
    """
    Greedy-generate exactly suffix_len tokens for each prompt row.
    Returns [N, suffix_len] on CPU.
    """
    device = next(model.parameters()).device
    outs = []
    for i in range(0, prompt_ids.size(0), batch_size):
        batch = prompt_ids[i: i + batch_size].to(device)
        gen = model.generate(
            input_ids=batch,
            attention_mask=torch.ones_like(batch),
            max_new_tokens=suffix_len,
            do_sample=False,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
            use_cache=True,
        )
        outs.append(_extract_new_tokens(gen, input_len=batch.size(1), target_len=suffix_len, pad_id=pad_id).cpu())
    return torch.cat(outs, dim=0) if outs else torch.empty(0, suffix_len, dtype=torch.long)


def _levenshtein_ints(a: List[int], b: List[int]) -> int:
    m, n = len(a), len(b)
    if m == 0: return n
    if n == 0: return m
    prev = list(range(n + 1))
    cur = [0] * (n + 1)
    for i in range(1, m + 1):
        cur[0] = i
        ai = a[i - 1]
        for j in range(1, n + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev, cur = cur, prev
    return prev[n]


def _avg_norm_lev(a: torch.Tensor, b: torch.Tensor) -> float:
    assert a.shape == b.shape
    N, L = a.size(0), a.size(1)
    if N == 0:
        return float("nan")
    tot = 0.0
    for i in range(N):
        tot += _levenshtein_ints(a[i].tolist(), b[i].tolist()) / max(1, L)
    return tot / N


def eval_lev_on_clean_pile_half_nonmem_only(
    *,
    model,
    tokenizer,
    text_path: str,
    prefix_len: int = NONMEM_PREFIX_LEN,
    suffix_len: int = NONMEM_SUFFIX_LEN,
    max_tokens: int = 200_000,
    batch_size: int = 32,
    baseline_cache_path: str,
    baseline_model=None,
) -> Dict[str, float | int | None]:
    """
    Clean half windows (64+48), drop rows where baseline==gold, then compute:
      - avg_lev_vs_gold_baseline
      - avg_lev_vs_gold_edited
    """
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id or pad_id

    prompts, gold = _make_clean_half_windows(
        tokenizer, text_path, prefix_len=prefix_len, suffix_len=suffix_len, max_tokens=max_tokens
    )
    N = prompts.size(0)
    if N == 0:
        return {
            'avg_lev_vs_gold_baseline': None,
            'avg_lev_vs_gold_edited': None,
            'num_windows_total': 0,
            'num_windows_kept': 0,
        }

    # Baseline completions cache
    if os.path.exists(baseline_cache_path):
        baseline_comps = torch.load(baseline_cache_path, map_location='cpu')
        if not (isinstance(baseline_comps, torch.Tensor) and list(baseline_comps.shape) == [N, suffix_len]):
            raise ValueError(f"Baseline cache shape mismatch: expected [{N}, {suffix_len}], got {list(baseline_comps.shape)}")
    else:
        if baseline_model is None:
            raise ValueError("Baseline cache not found. Provide `baseline_model` to build it.")
        baseline_comps = _batch_greedy_generate_exact(
            baseline_model, prompts, suffix_len=suffix_len, pad_id=pad_id, eos_id=eos_id, batch_size=batch_size
        )
        os.makedirs(os.path.dirname(baseline_cache_path), exist_ok=True)
        torch.save(baseline_comps, baseline_cache_path)

    keep_mask = (baseline_comps != gold).any(dim=1)
    if keep_mask.sum().item() == 0:
        return {
            'avg_lev_vs_gold_baseline': None,
            'avg_lev_vs_gold_edited': None,
            'num_windows_total': int(N),
            'num_windows_kept': 0,
        }

    prompts_kept = prompts[keep_mask]
    gold_kept = gold[keep_mask]
    base_kept = baseline_comps[keep_mask]

    edited_kept = _batch_greedy_generate_exact(
        model, prompts_kept, suffix_len=suffix_len, pad_id=pad_id, eos_id=eos_id, batch_size=batch_size
    )

    return {
        'avg_lev_vs_gold_baseline': float(_avg_norm_lev(base_kept, gold_kept)),
        'avg_lev_vs_gold_edited':   float(_avg_norm_lev(edited_kept, gold_kept)),
        'num_windows_total': int(N),
        'num_windows_kept':  int(keep_mask.sum().item()),
    }


# ────────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Apply SVD sequentially across multiple layers, then evaluate Levenshtein + nDCG metrics on an n-8,8 text memorization set.")
    parser.add_argument("--model-name", type=str, default="allenai/OLMo-2-0425-1B", help="HF model name")
    parser.add_argument("--dtype", type=str, choices=["float16", "bfloat16", "float32"], default="bfloat16", help="Compute dtype for model weights")
    parser.add_argument("--bs", type=int, default=32, help="Batch size for Levenshtein eval")
    parser.add_argument("--loose", type=float, default=0.75, help="Loose threshold for Levenshtein (0..1)")
    parser.add_argument("--prefix", type=int, default=64, help="Prefix token length")
    parser.add_argument("--suffix", type=int, default=48, help="Suffix token length")
    parser.add_argument("--skip-baseline", action="store_true", help="Skip baseline Levenshtein and baseline nDCG")
    parser.add_argument("--layers-json", type=str, default="", help="JSON string mapping layer -> {gate, up, down}")
    parser.add_argument("--layers-file", type=str, default="", help="Path to a JSON file with the same mapping as --layers-json")
    parser.add_argument("--order", type=str, default="", help="Comma-separated layer indices to apply in sequence; defaults to the dict insertion order")
    parser.add_argument("--use-cache", action="store_true", help="Load/save cached SVD weights per (layer, proj, rho)")
    parser.add_argument("--refresh-cache", action="store_true", help="Recompute and overwrite cache entries if present")
    parser.add_argument("--perplexity", action="store_true", help="Compute clean perplexity over pile10k windows")

    args = parser.parse_args()

    # Suppress HF progress bars / logs
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BAR", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()

    t0 = time.time()

    # Load dataset (text prefix/suffix pairs)
    prefix_len = args.prefix
    suffix_len = args.suffix
    batch_size = args.bs
    loose_threshold = args.loose
    dtype = args.dtype

    ndcg_data = NDCG_FILE
    ndcg_k = 10
    ndcg_batch = 8
    ndcg_seq_len = 1024
    ndcg_max_tokens = 200000
    print(f"ALERT: ndcg_max_tokens set to {ndcg_max_tokens} for faster grid runs")

    sequences = read_sequences_large8(DATASET_JSONL)
    _N = len(sequences)

    # Load baseline model
    model, tok = load_model_and_tokenizer(args.model_name, dtype=dtype)

    # Baseline (optional)
    if not args.skip_baseline:
        print("============================================================")
        print("BASELINE (no SVD) — Levenshtein metrics")
        print("============================================================")
        base_metrics = compute_memorization_metrics_levenshtein(
            model=model,
            sequences=sequences,
            tokenizer=tok,
            batch_size=batch_size,
            loose_threshold=loose_threshold,
        )
        print({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in base_metrics.items()})

    # nDCG setup and baseline cache (unchanged vs K-FAC script)
    print("\nSetting up nDCG evaluation and baseline cache...")
    baseline_file = get_baseline_predictions(
        model_name=args.model_name,
        data_path=ndcg_data,
        k=ndcg_k,
        seq_len=ndcg_seq_len,
        batch_size=ndcg_batch,
        max_tokens=ndcg_max_tokens,
        dtype=torch.bfloat16,
    )
    ndcg_dataset = CachedTextChunkDataset(
        filepath=ndcg_data,
        tokenizer=tok,
        seq_len=ndcg_seq_len,
        max_tokens=ndcg_max_tokens,
        cache_dir=os.path.join(os.path.dirname(__file__), "data"),
    )
    ndcg_eval = NDCGEvaluator(
        baseline_file=baseline_file,
        dataset=ndcg_dataset,
        k=ndcg_k,
        batch_size=ndcg_batch,
        dtype=torch.bfloat16,
    )
    if not args.skip_baseline:
        baseline_ndcg = ndcg_eval(model, max_tokens=ndcg_max_tokens, show_progress=False)
        print(f"Baseline NDCG@{ndcg_k}: {baseline_ndcg:.4f}")

    # Clean perplexity loader from same nDCG text, but via cached token ids and max_tokens cap
    def _build_clean_perp_loader(
        text_path: str,
        tokenizer,
        block_size: int = CLEAN_BLOCK_SIZE,
        batch_size: int = 32,
        max_tokens: int | None = None,
    ) -> DataLoader:
        ds = CachedTextChunkDataset(
            filepath=text_path,
            tokenizer=tokenizer,
            seq_len=block_size,
            max_tokens=max_tokens,
            cache_dir=os.path.join(os.path.dirname(__file__), "data"),
        )
        half = max(len(ds) // 2, 1)
        from torch.utils.data import Subset
        return DataLoader(Subset(ds, range(0, half)), batch_size=batch_size, shuffle=False)

    perp_loader = None
    if args.perplexity:
        perp_loader = _build_clean_perp_loader(
            NDCG_FILE,
            tok,
            block_size=CLEAN_BLOCK_SIZE,
            batch_size=32,
            max_tokens=ndcg_max_tokens,
        )
        if not args.skip_baseline:
            base_perp = perplexity(perp_loader, model)
            print(f"Baseline perplexity (clean windows): {base_perp:.4f}")

    # ===== CLEAN non-mem PRE: clean-half 64+48 windows, drop baseline==gold, avg L_norm =====
    clean_nonmem_cache_dir = os.path.join(os.path.dirname(NDCG_FILE), "clean_nonmem_cache")
    os.makedirs(clean_nonmem_cache_dir, exist_ok=True)
    clean_nonmem_cache_path = os.path.join(
        clean_nonmem_cache_dir,
        f"{_sanitize_filename_component(args.model_name)}__baseline_cleanhalf_p{NONMEM_PREFIX_LEN}_s{NONMEM_SUFFIX_LEN}_max{ndcg_max_tokens}.pt"
    )

    nonmem_pre = None
    try:
        nonmem_pre = eval_lev_on_clean_pile_half_nonmem_only(
            model=model,                    # baseline (pre-edit)
            tokenizer=tok,
            text_path=NDCG_FILE,
            prefix_len=NONMEM_PREFIX_LEN,
            suffix_len=NONMEM_SUFFIX_LEN,
            max_tokens=ndcg_max_tokens,
            batch_size=batch_size,
            baseline_cache_path=clean_nonmem_cache_path,  # builds if missing
            baseline_model=model,
        )
        if nonmem_pre["num_windows_kept"] > 0:
            print(f"[CLEAN non-mem PRE] avg_lev_gold baseline={nonmem_pre['avg_lev_vs_gold_baseline']:.4f} "
                  f"(kept={nonmem_pre['num_windows_kept']}/{nonmem_pre['num_windows_total']})")
        else:
            print("[CLEAN non-mem PRE] No rows after dropping baseline==gold; skipping metric.")
    except Exception as e:
        print(f"[warn] CLEAN non-mem PRE failed: {e}")

    # Resolve layer→variance mapping
    if args.layers_file:
        with open(args.layers_file, "r", encoding="utf-8") as fh:
            layer_map_raw = json.load(fh)
    elif args.layers_json:
        layer_map_raw = json.loads(args.layers_json)
    else:
        layer_map_raw = {str(k): v for k, v in LAYER_TO_VARIANCES_DEFAULT.items()}

    # Normalize keys to int, values to floats
    layer_to_variances: Dict[int, Dict[str, float]] = {}
    for k_str, ratios in layer_map_raw.items():
        li = int(k_str)
        gate = float(ratios.get("gate", 1.0))
        up = float(ratios.get("up", 1.0))
        down = float(ratios.get("down", 1.0))
        layer_to_variances[li] = {"gate": gate, "up": up, "down": down}

    if not layer_to_variances:
        raise ValueError("No layers specified. Provide --layers-json/--layers-file or edit LAYER_TO_VARIANCES_DEFAULT.")

    # Determine application order
    if args.order.strip():
        layer_order = [int(x) for x in args.order.split(',') if x.strip()]
    else:
        # Preserve insertion order from JSON parsing (Python 3.7+)
        layer_order = [int(k) for k in layer_map_raw.keys()]

    # Apply SVD sequentially across layers (in place)
    print("\nApplying SVD sequentially across layers:", layer_order)
    for li in layer_order:
        if li not in layer_to_variances:
            raise ValueError(f"Layer {li} not present in the mapping; check --order vs mapping")
        ratios = layer_to_variances[li]

        global LAYER_IDX
        LAYER_IDX = li
        print(f"\n→ Layer {li}: gate ρ={ratios['gate']:.3f}, up ρ={ratios['up']:.3f}, down ρ={ratios['down']:.3f}")
        apply_pairwise_svd(
            model,
            variance_up=ratios["up"],
            variance_down=ratios["down"],
            variance_gate=ratios["gate"],
            quiet=True,
            model_name=args.model_name,
            use_cache=args.use_cache,
            refresh_cache=args.refresh_cache,
            cache_dir=CACHE_DIR,
        )

    # Final evaluation after all layers have been transformed
    print("\n============================================================")
    print("SVD (multi-layer) — Levenshtein metrics")
    print("============================================================")
    svd_metrics = compute_memorization_metrics_levenshtein(
        model=model,
        sequences=sequences,
        tokenizer=tok,
        batch_size=batch_size,
        loose_threshold=loose_threshold,
    )
    print({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in svd_metrics.items()})

    svd_ndcg = ndcg_eval(model, max_tokens=ndcg_max_tokens, show_progress=False)
    print(f"SVD (multi-layer) NDCG@{ndcg_k}: {svd_ndcg:.4f}")
    svd_perp = None
    if args.perplexity and perp_loader is not None:
        svd_perp = perplexity(perp_loader, model)
        print(f"SVD (multi-layer) perplexity (clean windows): {svd_perp:.4f}")

    # ===== CLEAN non-mem POST (reuse baseline cache) =====
    nonmem_post = None
    try:
        nonmem_post = eval_lev_on_clean_pile_half_nonmem_only(
            model=model,                    # edited model
            tokenizer=tok,
            text_path=NDCG_FILE,
            prefix_len=NONMEM_PREFIX_LEN,
            suffix_len=NONMEM_SUFFIX_LEN,
            max_tokens=ndcg_max_tokens,
            batch_size=batch_size,
            baseline_cache_path=clean_nonmem_cache_path,  # MUST reuse same cache
            baseline_model=None,
        )
        if nonmem_post["num_windows_kept"] > 0:
            print(f"[CLEAN non-mem POST] avg_lev_gold baseline={nonmem_post['avg_lev_vs_gold_baseline']:.4f} "
                  f"edited={nonmem_post['avg_lev_vs_gold_edited']:.4f} "
                  f"(kept={nonmem_post['num_windows_kept']}/{nonmem_post['num_windows_total']})")
        else:
            print("[CLEAN non-mem POST] No rows after dropping baseline==gold; skipping metric.")
    except Exception as e:
        print(f"[warn] CLEAN non-mem POST failed: {e}")

    # Persist a compact record of the multi-layer run next to the dataset
    out_dir = os.path.dirname(os.path.abspath(DATASET_JSONL))
    os.makedirs(out_dir, exist_ok=True)
    jsonl_path = os.path.join(out_dir, "svd_eval_log_multi.jsonl")
    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": DATASET_JSONL,
        "model": args.model_name,
        "layers": layer_to_variances,
        "dtype": dtype,
        "prefix_len": int(prefix_len),
        "suffix_len": int(suffix_len),
        "batch_size": int(batch_size),
        "loose_threshold": float(loose_threshold),
        "svd_strict_acc": float(svd_metrics.get("strict_acc", 0.0)),
        "svd_loose_acc": float(svd_metrics.get("loose_acc", 0.0)),
        "svd_avg_levenshtein_norm": float(svd_metrics.get("avg_levenshtein_norm", 0.0)),
        "svd_total": int(svd_metrics.get("total", 0)),
        f"svd_ndcg@{ndcg_k}": float(svd_ndcg),
        "elapsed_sec": round(time.time() - t0, 2),
    }
    if svd_perp is not None:
        record["svd_perplexity_clean"] = float(svd_perp)

    # Add clean non-mem stats
    if nonmem_pre is not None:
        record.update({
            "clean_nonmem_pre_avg_lev_base": nonmem_pre.get("avg_lev_vs_gold_baseline"),
            "clean_nonmem_n_total": nonmem_pre.get("num_windows_total"),
            "clean_nonmem_n_kept":  nonmem_pre.get("num_windows_kept"),
        })
    if nonmem_post is not None:
        record.update({
            "clean_nonmem_post_avg_lev_base": nonmem_post.get("avg_lev_vs_gold_baseline"),
            "clean_nonmem_post_avg_lev_edit": nonmem_post.get("avg_lev_vs_gold_edited"),
            # keep counts in record too (redundant but handy for filtering)
            "clean_nonmem_n_total": nonmem_post.get("num_windows_total"),
            "clean_nonmem_n_kept":  nonmem_post.get("num_windows_kept"),
        })

    with open(jsonl_path, "a", encoding="utf-8") as jh:
        jh.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"\n✓ Appended multi-layer SVD results to: {jsonl_path}")


if __name__ == "__main__":
    main()
