"""
Offline check of the DFLASH mixed-verify target forward (no server, no draft model).

Prefills R running requests, then compares one mixed forward over
[extend rows of a new request | verify rows of the running requests] against
the separate extend forward plus the separate target-verify forward:
outputs (logits, captured hidden states, a follow-up extend chunk) and latency.

It accepts the server arguments of launch_server.py (use the DFLASH target
configuration) plus the --mv-* arguments below.

# Usage
python -m sglang.benchmark.mixed_verify <server args> --mv-running 16 --mv-ctx 140000 --mv-extend 4096 8192 16384
"""

import argparse
import dataclasses
import json
import logging
import multiprocessing
import os
import time
from array import array
from types import SimpleNamespace
from typing import List, Tuple

import numpy as np
import torch

from sglang.benchmark.one_batch import (
    TreeCacheNamespace,
    _init_process_global_configs,
    _maybe_prepare_mlp_sync_batch,
    load_model,
)
from sglang.kernels.ops.speculative.cache_locs import assign_extend_cache_locs_func
from sglang.srt.arg_groups.model_override_base import resolving_view
from sglang.srt.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
)
from sglang.srt.entrypoints.engine import _set_envs_and_config
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_mode import CaptureHiddenMode
from sglang.srt.model_executor.runner.mixed_verify_runner import MixedVerifyRunner
from sglang.srt.runtime_context import (
    SpawnRanks,
    get_schedule,
    get_spec,
    publish,
    spawn_world_rank,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.draft_worker_common import make_draft_input_v2
from sglang.srt.utils import (
    configure_logger,
    kill_process_tree,
    maybe_reindex_device_id,
)

# Token ids below this are ordinary text tokens in every supported vocabulary.
_MAX_RANDOM_TOKEN_ID = 10000


@dataclasses.dataclass
class MixedVerifyBenchArgs:
    mv_running: int = 16
    mv_ctx: int = 140000
    mv_tail: int = 512
    mv_extend: Tuple[int] = (4096, 8192, 16384)
    mv_prefix: int = 0
    mv_followup: int = 512
    mv_reps: int = 5
    mv_corpus: str = ""
    mv_exact: bool = False
    mv_noise: bool = False
    mv_accept: int = 3
    mv_result_filename: str = ""

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        d = MixedVerifyBenchArgs
        parser.add_argument("--mv-running", type=int, default=d.mv_running)
        parser.add_argument("--mv-ctx", type=int, default=d.mv_ctx)
        parser.add_argument(
            "--mv-tail",
            type=int,
            default=d.mv_tail,
            help="Tokens of each running request's last prefill chunk (one batch).",
        )
        parser.add_argument("--mv-extend", type=int, nargs="+", default=d.mv_extend)
        parser.add_argument(
            "--mv-prefix",
            type=int,
            default=d.mv_prefix,
            help="Tokens the new request already holds before the measured chunk.",
        )
        parser.add_argument("--mv-followup", type=int, default=d.mv_followup)
        parser.add_argument("--mv-reps", type=int, default=d.mv_reps)
        parser.add_argument(
            "--mv-corpus",
            type=str,
            default=d.mv_corpus,
            help="Text file the prompts are cut from; random token ids when empty.",
        )
        parser.add_argument(
            "--mv-result-filename", type=str, default=d.mv_result_filename
        )
        parser.add_argument(
            "--mv-exact",
            action="store_true",
            help="Exact-equality run instead of the latency run. Keep every "
            "request under the sparse-attention top-k so the separate forwards "
            "are reproducible.",
        )
        parser.add_argument(
            "--mv-noise",
            action="store_true",
            help="Reproducibility of the separate extend forward: two fresh "
            "requests with the same tokens, per --mv-extend length.",
        )
        parser.add_argument(
            "--mv-accept",
            type=int,
            default=d.mv_accept,
            help="Tokens of the first verify block committed before the second.",
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        return cls(
            **{
                f.name: (
                    tuple(getattr(args, f.name))
                    if f.name == "mv_extend"
                    else getattr(args, f.name)
                )
                for f in dataclasses.fields(cls)
            }
        )


class _Harness:
    """Builds the forward batches a scheduler and DFLASH worker would."""

    def __init__(self, model_runner, chunk_size: int):
        self.mr = model_runner
        self.device = model_runner.device
        self.chunk_size = chunk_size
        self.block_size = int(get_spec().speculative_num_draft_tokens)
        self.tree_cache = TreeCacheNamespace(
            page_size=get_schedule().page_size,
            device=model_runner.device,
            req_to_token_pool=model_runner.req_to_token_pool,
            token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
        )
        self._num_reqs = 0

    def new_req(self, token_ids: np.ndarray) -> Req:
        req = Req(
            rid=self._num_reqs,
            origin_input_text="",
            origin_input_ids=array("q", token_ids.tolist()),
            sampling_params=SamplingParams(temperature=0, max_new_tokens=16),
        )
        self._num_reqs += 1
        req.full_untruncated_fill_ids = req.origin_input_ids
        req.logprob_start_len = -1
        req.extend_end = 0
        return req

    def extend_batch(
        self, reqs: List[Req], ends: List[int]
    ) -> Tuple[ScheduleBatch, ForwardBatch]:
        """The next prefill chunk of each request, up to ``ends``."""
        for req, end in zip(reqs, ends):
            req.prefix_len = req.extend_end
            req.extend_end = end
        batch = ScheduleBatch.init_new(
            reqs=reqs,
            req_to_token_pool=self.mr.req_to_token_pool,
            token_to_kv_pool_allocator=self.mr.token_to_kv_pool_allocator,
            tree_cache=self.tree_cache,
            model_config=self.mr.model_config,
            enable_overlap=False,
            spec_algorithm=self.mr.spec_algorithm,
        )
        batch.prepare_for_extend()
        _maybe_prepare_mlp_sync_batch(batch, self.mr)
        if batch.input_ids is None and batch.prefill_input_ids_cpu is not None:
            batch.input_ids = batch.prefill_input_ids_cpu.to(self.device)
            batch.prefill_input_ids_cpu = None
        forward_batch = ForwardBatch.init_new(
            batch,
            self.mr,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            return_hidden_states_before_norm=False,
        )
        return batch, forward_batch

    def prefill(self, req: Req, end: int) -> None:
        """Chunked prefill of one request up to ``end``."""
        while req.extend_end < end:
            _, forward_batch = self.extend_batch(
                [req], [min(end, req.extend_end + self.chunk_size)]
            )
            self.mr.forward(forward_batch)

    def start_decode(self, batch: ScheduleBatch, bonus_tokens: torch.Tensor) -> None:
        """Move a finished prefill batch to speculative decode: reserve the
        verify window of every request."""
        batch.spec_info = make_draft_input_v2(
            bonus_tokens=bonus_tokens, new_seq_lens=batch.seq_lens
        )
        batch.prepare_for_decode()

    def verify_batch(
        self, batch: ScheduleBatch, draft_tokens: torch.Tensor, *, load_graph: bool
    ) -> ForwardBatch:
        """The target-verify forward of the running batch over ``draft_tokens``
        ([bs, block]), as DFlashWorkerV2 prepares it."""
        bs, block = draft_tokens.shape
        prefix_lens = batch.seq_lens
        positions = (
            prefix_lens.unsqueeze(1) + torch.arange(block, device=self.device)
        ).reshape(-1)
        out_cache_loc = assign_extend_cache_locs_func(
            req_pool_indices=batch.req_pool_indices,
            req_to_token=self.mr.req_to_token_pool.req_to_token,
            start_offset=prefix_lens,
            end_offset=prefix_lens + block,
            batch_size=bs,
            draft_token_num=block,
            device=self.device,
        )
        committed_lens_cpu = torch.tensor(
            [req.kv.kv_committed_len for req in batch.reqs], dtype=torch.int64
        )
        iteration_ids = self.mr.runner_ids.plan(
            req_pool_indices=batch.req_pool_indices,
            seq_lens=prefix_lens,
            seq_lens_cpu=committed_lens_cpu,
            write_virtual=out_cache_loc,
            read_extent=block,
        )
        verify_input = DFlashVerifyInput(
            draft_token=draft_tokens.reshape(-1),
            positions=positions,
            draft_token_num=block,
            custom_mask=None,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            iteration_ids=iteration_ids,
        )
        batch.out_cache_loc = out_cache_loc
        # Verify host bound = committed prefix + one verify block.
        batch.seq_lens_cpu = committed_lens_cpu + block
        batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())
        forward_batch, _ = verify_input.prepare_for_verify(
            batch,
            SimpleNamespace(model_runner=self.mr),
            defer_metadata_init=not load_graph,
        )
        batch.seq_lens_cpu = committed_lens_cpu
        batch.seq_lens_sum = int(committed_lens_cpu.sum())
        return forward_batch


    def commit(self, batch: ScheduleBatch, num_accept_tokens: int, attn_backend):
        """Commit the first tokens of the verify block ``attn_backend`` just
        ran, as DFlashWorkerV2 and the result processor do."""
        bs = len(batch.reqs)
        accept_lens = torch.full(
            (bs,), num_accept_tokens, dtype=torch.int64, device=self.device
        )
        tracked = batch.mamba_track_indices is not None
        attn_backend.update_mamba_state_after_mtp_verify(
            last_correct_step_indices=accept_lens - 1,
            mamba_track_indices=batch.mamba_track_indices,
            mamba_steps_to_track=torch.full_like(accept_lens, -1) if tracked else None,
            model=self.mr.model,
            req_pool_indices=batch.req_pool_indices,
        )
        for req in batch.reqs:
            req.kv.commit_decode(num_accept_tokens)
        batch.seq_lens = batch.seq_lens + num_accept_tokens
        batch.seq_lens_cpu = batch.seq_lens_cpu + num_accept_tokens
        batch.orig_seq_lens = batch.orig_seq_lens + num_accept_tokens
        batch.seq_lens_sum = None


class _SplitFfn:
    """Test only: inside ``active``, every weight read a mixed forward shares
    between its segments (FFN, o_proj) runs once per segment instead, so each
    row goes through exactly the kernels of its separate forward."""

    def __init__(self, model):
        from sglang.srt.models.kimi_k3 import KimiK3DecoderLayer

        self._num_extend_rows = None
        for module in model.modules():
            if isinstance(module, KimiK3DecoderLayer):
                module.mlp.forward = self._per_segment(module.mlp.forward)
                module.self_attn.project_output = self._per_segment(
                    module.self_attn.project_output
                )

    def _per_segment(self, fn):
        def wrapped(hidden_states, **kwargs):
            split = self._num_extend_rows
            if split is None:
                return fn(hidden_states, **kwargs)
            outputs = []
            for rows in (slice(0, split), slice(split, None)):
                num_rows = hidden_states[rows].shape[0]
                segment_kwargs = dict(kwargs)
                if kwargs.get("prefix_sum") is not None:
                    segment_kwargs["prefix_sum"] = kwargs["prefix_sum"][rows]
                count = kwargs.get("valid_token_count")
                if isinstance(count, torch.Tensor):
                    segment_kwargs["valid_token_count"] = torch.full_like(
                        count, num_rows
                    )
                elif count is not None:
                    segment_kwargs["valid_token_count"] = num_rows
                outputs.append(fn(hidden_states[rows], **segment_kwargs))
            return torch.cat(outputs, dim=0)

        return wrapped

    def run(self, mixed_runner, *, extend: ForwardBatch, verify: ForwardBatch):
        parent = mixed_runner.plan(extend=extend, verify=verify)
        self._num_extend_rows = parent.mixed_verify.num_extend_tokens
        try:
            return mixed_runner.forward(parent)
        finally:
            self._num_extend_rows = None


class _TokenSource:
    """Disjoint token spans: consecutive text of a corpus, or random ids."""

    # Characters tokenized per call; a span boundary may split a word.
    _PIECE_CHARS = 1 << 20

    def __init__(self, corpus: str, tokenizer):
        self._rng = np.random.RandomState(0)
        self._file = open(corpus, errors="ignore") if corpus else None
        self._tokenizer = tokenizer
        self._pending = np.empty((0,), dtype=np.int64)

    def take(self, n: int) -> np.ndarray:
        if self._file is None:
            return self._rng.randint(0, _MAX_RANDOM_TOKEN_ID, (n,), dtype=np.int64)
        while len(self._pending) < n:
            text = self._file.read(self._PIECE_CHARS)
            assert text, "corpus exhausted"
            ids = self._tokenizer.encode(text, add_special_tokens=False)
            self._pending = np.concatenate(
                [self._pending, np.asarray(ids, dtype=np.int64)]
            )
        out, self._pending = self._pending[:n], self._pending[n:]
        return out


def _snapshot(logits_output) -> dict:
    """Owned copies: a graph replay or a later forward may reuse the buffers."""
    return {
        name: tensor.float().clone()
        for name, tensor in (
            ("logits", logits_output.next_token_logits),
            ("hidden", logits_output.hidden_states),
        )
        if isinstance(tensor, torch.Tensor)
    }


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / a.norm().clamp_min(1e-9))


def _compare(ref: dict, out: dict, hidden_size: int) -> dict:
    result = {}
    for name in ref:
        a, b = ref[name], out[name]
        if a.shape != b.shape:
            result[name] = f"shape {tuple(a.shape)} vs {tuple(b.shape)}"
            continue
        result[f"{name}_rel_l2"] = _rel_l2(a, b)
        row_rel = (a - b).norm(dim=-1) / a.norm(dim=-1).clamp_min(1e-9)
        result[f"{name}_row_rel_l2_median"] = float(row_rel.median())
        result[f"{name}_row_rel_l2_max"] = float(row_rel.max())
        identical = int(((a - b).abs().amax(dim=-1) == 0).sum())
        result[f"{name}_identical_rows"] = f"{identical}/{a.shape[0]}"
    if "hidden" in ref and ref["hidden"].shape == out["hidden"].shape:
        # Captured layers in depth order: run-to-run noise grows with depth.
        a = ref["hidden"].unflatten(-1, (-1, hidden_size))
        b = out["hidden"].unflatten(-1, (-1, hidden_size))
        result["hidden_rel_l2_per_capture"] = [
            round(_rel_l2(a[:, i], b[:, i]), 5) for i in range(a.shape[1])
        ]
    if "logits" in ref and ref["logits"].shape == out["logits"].shape:
        same = ref["logits"].argmax(-1) == out["logits"].argmax(-1)
        result["argmax_match"] = f"{int(same.sum())}/{same.numel()}"
    return result


@torch.no_grad()
def _run(
    model_runner, tokenizer, bench_args: MixedVerifyBenchArgs, rank_print
) -> List[dict]:
    mr = model_runner
    device_module = torch.get_device_module(mr.device)
    harness = _Harness(mr, chunk_size=get_schedule().chunked_prefill_size)
    hidden_size = mr.model_config.hf_text_config.hidden_size

    def compare(ref: dict, out: dict) -> dict:
        return _compare(ref, out, hidden_size)

    mixed_runner = MixedVerifyRunner(mr)
    source = _TokenSource(bench_args.mv_corpus, tokenizer)
    tokens = source.take
    block = harness.block_size

    def timed(fn):
        device_module.synchronize()
        tic = time.perf_counter()
        out = fn()
        device_module.synchronize()
        return out, (time.perf_counter() - tic) * 1e3

    rank_print(
        f"logical capacity={mr.logical_max_total_num_tokens} tokens, "
        f"per-rank rows={mr.max_total_num_tokens}, block={block}"
    )

    # --- Running requests: chunked prefill, then the first verify window.
    num_running, ctx = bench_args.mv_running, bench_args.mv_ctx
    # Each request's prompt, then the text that follows it as the draft block.
    running_ids = [tokens(ctx + block) for _ in range(num_running)]
    running = [harness.new_req(ids[:ctx]) for ids in running_ids]
    tic = time.perf_counter()
    for req in running:
        harness.prefill(req, ctx - bench_args.mv_tail)
    running_batch, forward_batch = harness.extend_batch(running, [ctx] * num_running)
    mr.forward(forward_batch)
    device_module.synchronize()
    rank_print(
        f"prefilled {num_running} x {ctx} tokens in {time.perf_counter() - tic:.1f} s"
    )
    draft_tokens = torch.from_numpy(
        np.stack([ids[ctx:] for ids in running_ids])
    ).to(mr.device)
    harness.start_decode(running_batch, bonus_tokens=draft_tokens[:, 0].clone())

    def verify_graph():
        fb = harness.verify_batch(running_batch, draft_tokens, load_graph=True)
        return mr.forward(fb, skip_attn_backend_init=True)

    def verify_eager():
        fb = harness.verify_batch(running_batch, draft_tokens, load_graph=False)
        graph_runner, mr.decode_cuda_graph_runner = mr.decode_cuda_graph_runner, None
        try:
            return mr.forward(fb)
        finally:
            mr.decode_cuda_graph_runner = graph_runner

    out, _ = timed(verify_graph)
    rank_print(f"verify alone: cuda graph={out.can_run_graph}")
    verify_ref = _snapshot(out.logits_output)
    out, _ = timed(verify_graph)
    rank_print(
        "noise floor, verify graph vs graph:",
        json.dumps(compare(verify_ref, _snapshot(out.logits_output))),
    )
    out, _ = timed(verify_eager)
    rank_print(
        "verify eager vs graph:",
        json.dumps(compare(verify_ref, _snapshot(out.logits_output))),
    )

    results = []
    for extend_len in bench_args.mv_extend:
        prefix, followup = bench_args.mv_prefix, bench_args.mv_followup
        ids = tokens(prefix + extend_len + followup)
        req_alone, req_mixed = harness.new_req(ids), harness.new_req(ids)
        req_noise = harness.new_req(ids)
        for req in (req_alone, req_mixed, req_noise):
            harness.prefill(req, prefix)

        # Separate forwards, then the mixed one, on requests in the same state.
        _, fb_alone = harness.extend_batch([req_alone], [prefix + extend_len])
        out, _ = timed(lambda: mr.forward(fb_alone))
        extend_ref = _snapshot(out.logits_output)

        _, fb_mixed = harness.extend_batch([req_mixed], [prefix + extend_len])
        fb_verify = harness.verify_batch(running_batch, draft_tokens, load_graph=False)
        out, _ = timed(lambda: mixed_runner.execute(extend=fb_mixed, verify=fb_verify))
        equality = {
            "extend": compare(extend_ref, _snapshot(out.extend)),
            "verify": compare(verify_ref, _snapshot(out.verify)),
        }
        # Noise floor: the separate extend forward again, on a third request.
        _, fb_noise = harness.extend_batch([req_noise], [prefix + extend_len])
        equality["noise_floor_extend"] = compare(
            extend_ref, _snapshot(mr.forward(fb_noise).logits_output)
        )

        # The next chunk reads the KV and KDA state each path wrote.
        if followup > 0:
            end = prefix + extend_len + followup
            _, fb = harness.extend_batch([req_alone], [end])
            followup_ref = _snapshot(mr.forward(fb).logits_output)
            _, fb = harness.extend_batch([req_mixed], [end])
            equality["followup_extend"] = compare(
                followup_ref, _snapshot(mr.forward(fb).logits_output)
            )
            _, fb = harness.extend_batch([req_noise], [end])
            equality["noise_floor_followup_extend"] = compare(
                followup_ref, _snapshot(mr.forward(fb).logits_output)
            )
        # The verify window is unchanged by the mixed forward.
        out, _ = timed(verify_graph)
        equality["verify_after_mixed"] = compare(
            verify_ref, _snapshot(out.logits_output)
        )

        # Latency: the same batches again (state values do not change the cost).
        lat = {
            name: []
            for name in (
                "extend",
                "verify_graph",
                "verify_eager",
                "mixed",
                "mixed_plan",
                "mixed_forward",
            )
        }
        for _ in range(bench_args.mv_reps):
            lat["extend"].append(timed(lambda: mr.forward(fb_alone))[1])
            lat["verify_graph"].append(timed(verify_graph)[1])
            lat["verify_eager"].append(timed(verify_eager)[1])
            fb_verify = harness.verify_batch(
                running_batch, draft_tokens, load_graph=False
            )
            parent, plan_ms = timed(
                lambda: mixed_runner.plan(extend=fb_mixed, verify=fb_verify)
            )
            forward_ms = timed(lambda: mixed_runner.forward(parent))[1]
            lat["mixed_plan"].append(plan_ms)
            lat["mixed_forward"].append(forward_ms)
            lat["mixed"].append(plan_ms + forward_ms)
        med = {name: float(np.median(values)) for name, values in lat.items()}
        marginal = med["mixed"] - med["extend"]
        result = {
            "running": num_running,
            "ctx": ctx,
            "prefix": prefix,
            "extend": extend_len,
            "median_ms": med,
            "all_ms": lat,
            "marginal_ms": marginal,
            "marginal_over_verify_graph": marginal / med["verify_graph"],
            "equality": equality,
        }
        rank_print(json.dumps(result))
        results.append(result)
    return results


@torch.no_grad()
def _run_exact(
    model_runner, tokenizer, bench_args: MixedVerifyBenchArgs, rank_print
) -> List[dict]:
    """Mixed vs separate on requests whose separate forwards are reproducible:
    outputs of the step itself, then of the step after a commit, which reads
    the KV rows and KDA state the first one wrote."""
    mr = model_runner
    harness = _Harness(mr, chunk_size=get_schedule().chunked_prefill_size)
    mixed_runner = MixedVerifyRunner(mr)
    split_ffn = _SplitFfn(mr.model)
    hidden_size = mr.model_config.hf_text_config.hidden_size
    tokens = _TokenSource(bench_args.mv_corpus, tokenizer).take
    block, accept = harness.block_size, bench_args.mv_accept
    num_running, ctx = bench_args.mv_running, bench_args.mv_ctx
    extend_len, followup = bench_args.mv_extend[0], bench_args.mv_followup

    running_ids = [tokens(ctx + accept + block) for _ in range(num_running)]
    extend_ids = tokens(extend_len + followup)

    def drafts(start: int) -> torch.Tensor:
        return torch.from_numpy(
            np.stack([ids[start : start + block] for ids in running_ids])
        ).to(mr.device)

    def running_set() -> ScheduleBatch:
        reqs = [harness.new_req(ids[:ctx]) for ids in running_ids]
        batch, forward_batch = harness.extend_batch(reqs, [ctx] * num_running)
        mr.forward(forward_batch)
        harness.start_decode(batch, bonus_tokens=drafts(ctx)[:, 0].clone())
        return batch

    def verify_eager(batch: ScheduleBatch, draft_tokens: torch.Tensor) -> dict:
        fb = harness.verify_batch(batch, draft_tokens, load_graph=False)
        graph_runner, mr.decode_cuda_graph_runner = mr.decode_cuda_graph_runner, None
        try:
            return _snapshot(mr.forward(fb).logits_output)
        finally:
            mr.decode_cuda_graph_runner = graph_runner

    def extend(req: Req, end: int, eager: bool = True) -> dict:
        _, fb = harness.extend_batch([req], [end])
        graph_runner = mr.prefill_cuda_graph_runner
        if eager:
            # A mixed forward runs eager; compare against the eager extend.
            mr.prefill_cuda_graph_runner = None
        try:
            return _snapshot(mr.forward(fb).logits_output)
        finally:
            mr.prefill_cuda_graph_runner = graph_runner

    def separate(eager: bool = True) -> Tuple[ScheduleBatch, Req, dict]:
        """Verify, commit, extend: three separate steps."""
        batch, req = running_set(), harness.new_req(extend_ids)
        out = {"verify": verify_eager(batch, drafts(ctx))}
        harness.commit(batch, accept, mr.attn_backend)
        out["extend"] = extend(req, extend_len, eager=eager)
        return batch, req, out

    def mixed(run) -> Tuple[ScheduleBatch, Req, dict]:
        batch, req = running_set(), harness.new_req(extend_ids)
        _, fb_extend = harness.extend_batch([req], [extend_len])
        fb_verify = harness.verify_batch(batch, drafts(ctx), load_graph=False)
        ret = run(extend=fb_extend, verify=fb_verify)
        harness.commit(batch, accept, mixed_runner.verify_attn_backend)
        return (
            batch,
            req,
            {"verify": _snapshot(ret.verify), "extend": _snapshot(ret.extend)},
        )

    def second_step(batch: ScheduleBatch, req: Req, out: dict) -> dict:
        second = drafts(ctx + accept)
        harness.start_decode(batch, bonus_tokens=second[:, 0].clone())
        out["verify_after_commit"] = verify_eager(batch, second)
        if followup > 0:
            out["followup_extend"] = extend(req, extend_len + followup)
        return out

    universes = {
        "separate": separate(),
        "separate_again": separate(),
        "separate_graph_extend": separate(eager=False),
        "mixed": mixed(mixed_runner.execute),
        "mixed_ffn_per_segment": mixed(
            lambda **kw: split_ffn.run(mixed_runner, **kw)
        ),
    }
    outputs = {name: second_step(*state) for name, state in universes.items()}
    ref = outputs.pop("separate")
    result = {
        "running": num_running,
        "ctx": ctx,
        "extend": extend_len,
        "accept": accept,
        "vs_separate": {
            name: {
                step: _compare(ref[step], out[step], hidden_size)
                | {
                    "max_abs_diff": max(
                        float((ref[step][k] - out[step][k]).abs().max())
                        for k in ref[step]
                    )
                }
                for step in ref
            }
            for name, out in outputs.items()
        },
    }
    rank_print(json.dumps(result))
    return [result]


@torch.no_grad()
def _run_noise(
    model_runner, tokenizer, bench_args: MixedVerifyBenchArgs, rank_print
) -> List[dict]:
    """Where two separate extend forwards of the same tokens stop agreeing."""
    mr = model_runner
    harness = _Harness(mr, chunk_size=get_schedule().chunked_prefill_size)
    hidden_size = mr.model_config.hf_text_config.hidden_size
    tokens = _TokenSource(bench_args.mv_corpus, tokenizer).take
    results = []
    for extend_len in bench_args.mv_extend:
        ids = tokens(extend_len)
        outs = []
        for _ in range(2):
            _, fb = harness.extend_batch([harness.new_req(ids)], [extend_len])
            out = mr.forward(fb)
            outs.append(
                (out.can_run_graph, out.logits_output.hidden_states.float().clone())
            )
        (graph, a), (_, b) = outs
        a = a.unflatten(-1, (-1, hidden_size))
        b = b.unflatten(-1, (-1, hidden_size))
        # [rows, captures]: relative difference of each row at each capture.
        row_rel = (a - b).norm(dim=-1) / a.norm(dim=-1).clamp_min(1e-9)
        differs = row_rel > 0
        result = {
            "extend": extend_len,
            "cuda_graph": bool(graph),
            "identical_rows_per_capture": (~differs).sum(0).tolist(),
            "first_differing_row_per_capture": [
                int(col.nonzero()[0]) if bool(col.any()) else -1 for col in differs.T
            ],
            "row_rel_l2_median_per_capture": [
                round(float(x), 6) for x in row_rel.median(0).values
            ],
            "row_rel_l2_first_rows_capture0": [
                round(float(x), 6) for x in row_rel[:8, 0]
            ],
        }
        rank_print(json.dumps(result))
        results.append(result)
    return results


def _work(server_args, port_args, bench_args, gpu_id, tp_rank):
    cfg = resolving_view(server_args)
    publish(
        server_args,
        role="scheduler",
        ranks=SpawnRanks(
            world_rank=spawn_world_rank(server_args, tp_rank=tp_rank, pp_rank=0),
            gpu_id=gpu_id,
        ),
    )
    _init_process_global_configs()
    configure_logger(server_args, prefix=f" TP{tp_rank}")
    rank_print = (
        (lambda *args: print(*args, flush=True))
        if tp_rank == 0
        else (lambda *args: None)
    )

    bench_runner, tokenizer = load_model(server_args, port_args, gpu_id, tp_rank)
    run = (
        _run_exact
        if bench_args.mv_exact
        else _run_noise
        if bench_args.mv_noise
        else _run
    )
    results = run(bench_runner.torch_runner, tokenizer, bench_args, rank_print)
    if tp_rank == 0 and bench_args.mv_result_filename:
        with open(bench_args.mv_result_filename, "a") as fout:
            for result in results:
                fout.write(json.dumps(result) + "\n")

    if cfg.tp_size > 1:
        destroy_model_parallel()
        destroy_distributed_environment()


def main(server_args, bench_args):
    server_args.resolve_once()
    _set_envs_and_config(server_args)
    port_args = PortArgs.init_new(server_args)
    if server_args.tp_size == 1:
        _work(server_args, port_args, bench_args, 0, 0)
        return
    workers = []
    for tp_rank in range(server_args.tp_size):
        with maybe_reindex_device_id(tp_rank) as gpu_id:
            proc = multiprocessing.Process(
                target=_work,
                args=(server_args, port_args, bench_args, gpu_id, tp_rank),
            )
            proc.start()
            workers.append(proc)
    for proc in workers:
        proc.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    MixedVerifyBenchArgs.add_cli_args(parser)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    bench_args = MixedVerifyBenchArgs.from_cli_args(args)
    logging.basicConfig(
        level=getattr(logging, server_args.log_level.upper()),
        format="%(message)s",
    )
    try:
        main(server_args, bench_args)
    finally:
        if server_args.tp_size != 1:
            kill_process_tree(os.getpid(), include_parent=False)
