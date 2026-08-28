# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Benchmarks the cost of attention dropout in the splash attention kernel.

A `dropout_rate` of 0 removes the dropout path at trace time, so timing the same
shape and blocks at 0 and at the rate of interest isolates the dropout exactly.

    # every row of a built-in shape table
    python -m ...splash_attention_dropout_bench --preset=glm_fwd
    python -m ...splash_attention_dropout_bench --preset=glm_bwd

    # one configuration, given explicitly
    python -m ...splash_attention_dropout_bench --seq_len=8192 --mode=bwd \
        --block_q=1024 --block_kv=1024 --block_kv_compute=512 \
        --block_q_dkv=1024 --block_kv_dkv=2048 --block_kv_dkv_compute=512

    # split the cost into generating the mask bits and applying them
    python -m ...splash_attention_dropout_bench --preset=glm_bwd --marginal

Blocks tuned at `dropout_rate=0` are usually a poor choice once dropout is on,
so the interesting comparison is not one row against its own `ms@0` but the best
row of a search against the best row of a search at rate 0.

`--marginal` reports the second of two calls made in one `jit` on the same key.
For the dense scheme that lets XLA fold the two `bernoulli` draws into one, so
the difference between the one-call and two-call times is a kernel invocation
with the mask already resident, i.e. the cost of *applying* a mask with
generation excluded. The in-kernel scheme cannot benefit -- its bits are made
inside the kernel and there is nothing to hoist -- so the gap between the two
marginal numbers is what mask generation is costing. Note this is a diagnostic:
a real step draws a fresh key, so the marginal dense figure is not reachable.
"""

import dataclasses
import functools
import statistics
import time
from typing import Sequence

from absl import app
from absl import flags
import jax
from jax import random
import jax.numpy as jnp
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_kernel as splash
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_mask as mask_lib


# The dense-mask scheme is only present on branches that carry it; the flag is
# accepted either way so this file can be cherry-picked without edits.
_SUPPORTS_DENSE = any(
    f.name == "dropout_dense_mask" for f in dataclasses.fields(splash.SplashConfig)
)

_HEAD_DIM_MINOR = splash.QKVLayout.HEAD_DIM_MINOR
_SEQ_MINOR = splash.QKVLayout.SEQ_MINOR
_LAYOUTS = {"head_dim_minor": _HEAD_DIM_MINOR, "seq_minor": _SEQ_MINOR}


@dataclasses.dataclass(frozen=True)
class Row:
  """One benchmark configuration: a shape plus the blocks it was tuned to."""

  seq_len: int
  qk_head_dim: int
  v_head_dim: int
  layouts: tuple[str, str, str]
  fwd_blocks: tuple[int, int, int]
  bwd_blocks: tuple[int, int, int] | None
  use_experimental_scheduler: bool


def _rows(spec, bwd):
  hh, hs = "head_dim_minor", "seq_minor"
  lay = {1: hh, 2: hs}
  out = []
  for seq, dqk, dv, l, fwd, bwd_blocks, sched in spec:
    out.append(
        Row(seq, dqk, dv, tuple(lay[x] for x in l), fwd,
            bwd_blocks if bwd else None, sched)
    )
  return tuple(out)


# Shapes and blocks from a GLM tuning sweep on TPU7x, kept verbatim so the
# dropout-off column can be checked against the numbers that sweep reported.
_GLM_FWD = _rows(
    (
        (4096, 128, 128, (1, 2, 1), (1024, 1024, 512), None, True),
        (4096, 192, 128, (1, 1, 1), (1024, 1024, 512), None, True),
        (4096, 256, 256, (1, 2, 1), (1024, 1024, 256), None, True),
        (8192, 128, 128, (1, 1, 1), (2048, 2048, 256), None, True),
        (8192, 192, 128, (1, 2, 1), (2048, 2048, 512), None, True),
        (8192, 256, 256, (1, 2, 1), (2048, 2048, 256), None, True),
        (32768, 128, 128, (1, 1, 2), (2048, 2048, 256), None, True),
        (32768, 192, 128, (1, 2, 1), (2048, 2048, 512), None, True),
        (32768, 256, 256, (1, 1, 2), (2048, 2048, 256), None, True),
        (131072, 128, 128, (1, 1, 2), (4096, 2048, 256), None, True),
        (131072, 192, 128, (1, 1, 2), (4096, 2048, 256), None, True),
        (131072, 256, 256, (1, 2, 1), (4096, 2048, 256), None, True),
    ),
    bwd=False,
)
_GLM_BWD = _rows(
    (
        (4096, 128, 128, (1, 1, 1), (512, 1024, 512), (2048, 2048, 512), False),
        (4096, 192, 128, (2, 1, 1), (512, 1024, 512), (2048, 2048, 512), True),
        (4096, 256, 256, (1, 2, 1), (512, 1024, 512), (2048, 2048, 512), False),
        (8192, 128, 128, (2, 1, 1), (512, 1024, 512), (2048, 2048, 512), True),
        (8192, 192, 128, (2, 1, 1), (512, 1024, 512), (2048, 2048, 512), False),
        (8192, 256, 256, (1, 2, 1), (512, 1024, 512), (2048, 2048, 512), False),
        (32768, 128, 128, (2, 1, 2), (512, 1024, 512), (2048, 2048, 512), False),
        (32768, 192, 128, (2, 1, 1), (512, 1024, 512), (2048, 2048, 512), False),
        (32768, 256, 256, (2, 1, 2), (512, 1024, 512), (2048, 2048, 512), True),
        (131072, 128, 128, (2, 1, 1), (512, 1024, 512), (1024, 4096, 512), True),
    ),
    bwd=True,
)
_PRESETS = {"glm_fwd": _GLM_FWD, "glm_bwd": _GLM_BWD}

_PRESET = flags.DEFINE_enum(
    "preset", None, sorted(_PRESETS),
    "Built-in shape table to sweep. Omit to benchmark the flags below as a "
    "single configuration.")
_MODE = flags.DEFINE_enum(
    "mode", None, ["fwd", "bwd"],
    "Time the forward, or the forward and backward together (`jax.grad`). "
    "Defaults to whatever the preset implies, else `fwd`.")
_RATE = flags.DEFINE_float("rate", 0.1, "Dropout rate to compare against 0.")
_NUM_HEADS = flags.DEFINE_integer("num_heads", 32, "Number of query heads.")
_DTYPE = flags.DEFINE_string("dtype", "bfloat16", "Input dtype.")
_ITERS = flags.DEFINE_integer(
    "iters", 0, "Timed iterations. 0 picks a count from the sequence length.")
_SCHEMES = flags.DEFINE_list(
    "schemes", ["in_kernel"],
    "Mask sources to time: `in_kernel` regenerates the bits per tile, `dense` "
    "materializes the whole [head, q, kv] mask in HBM first.")
_MARGINAL = flags.DEFINE_bool(
    "marginal", False,
    "Also report the marginal cost of a second call sharing one key; see the "
    "module docstring.")
_PEAK_FLOPS = flags.DEFINE_float(
    "peak_flops", 1.15e15, "Denominator for the MFU column (TPU7x bf16).")

_SEQ_LEN = flags.DEFINE_integer("seq_len", 4096, "Query and key sequence length.")
_QK_HEAD_DIM = flags.DEFINE_integer("qk_head_dim", 128, "Head dim of q and k.")
_V_HEAD_DIM = flags.DEFINE_integer("v_head_dim", 128, "Head dim of v.")
_BLOCK_Q = flags.DEFINE_integer("block_q", 1024, "Forward query block.")
_BLOCK_KV = flags.DEFINE_integer("block_kv", 1024, "Forward key block (fetch).")
_BLOCK_KV_COMPUTE = flags.DEFINE_integer(
    "block_kv_compute", 512, "Forward key block (arithmetic).")
_BLOCK_Q_DKV = flags.DEFINE_integer("block_q_dkv", 1024, "Backward query block.")
_BLOCK_KV_DKV = flags.DEFINE_integer(
    "block_kv_dkv", 2048, "Backward key block (fetch).")
_BLOCK_KV_DKV_COMPUTE = flags.DEFINE_integer(
    "block_kv_dkv_compute", 512, "Backward key block (arithmetic).")
_DROPOUT_BLOCK_Q = flags.DEFINE_integer(
    "dropout_block_q", None,
    "Canonical dropout grid, query axis. Defaults to min(block_q, "
    "block_q_dkv).")
_DROPOUT_BLOCK_KV = flags.DEFINE_integer(
    "dropout_block_kv", None,
    "Canonical dropout grid, key axis. Defaults to min(block_kv_compute, "
    "block_kv_dkv_compute).")
_Q_LAYOUT = flags.DEFINE_enum("q_layout", "head_dim_minor", sorted(_LAYOUTS), "")
_K_LAYOUT = flags.DEFINE_enum("k_layout", "head_dim_minor", sorted(_LAYOUTS), "")
_V_LAYOUT = flags.DEFINE_enum("v_layout", "head_dim_minor", sorted(_LAYOUTS), "")
_SCHEDULER = flags.DEFINE_bool(
    "use_experimental_scheduler", True, "Passed through to SplashConfig.")


def _config(row: Row, rate: float, dense: bool) -> splash.SplashConfig:
  """Builds a SplashConfig for one row, at one rate, with one mask source."""
  kwargs = dict(
      block_q=row.fwd_blocks[0],
      block_kv=row.fwd_blocks[1],
      block_kv_compute=row.fwd_blocks[2],
      q_layout=_LAYOUTS[row.layouts[0]],
      k_layout=_LAYOUTS[row.layouts[1]],
      v_layout=_LAYOUTS[row.layouts[2]],
      use_experimental_scheduler=row.use_experimental_scheduler,
      dropout_rate=rate,
  )
  if row.bwd_blocks is not None:
    kwargs.update(
        block_q_dkv=row.bwd_blocks[0],
        block_kv_dkv=row.bwd_blocks[1],
        block_kv_dkv_compute=row.bwd_blocks[2],
    )
  if rate:
    if _DROPOUT_BLOCK_Q.value is not None:
      kwargs["dropout_block_q"] = _DROPOUT_BLOCK_Q.value
    if _DROPOUT_BLOCK_KV.value is not None:
      kwargs["dropout_block_kv"] = _DROPOUT_BLOCK_KV.value
  if dense:
    kwargs["dropout_dense_mask"] = True
  return splash.SplashConfig(**kwargs)


def _inputs(row: Row, dtype):
  k1, k2, k3, k4 = random.split(random.key(0), 4)
  n = _NUM_HEADS.value
  shape_qk = (n, row.seq_len, row.qk_head_dim)
  shape_v = (n, row.seq_len, row.v_head_dim)
  return (
      random.normal(k1, shape_qk, dtype=dtype),
      random.normal(k2, shape_qk, dtype=dtype),
      random.normal(k3, shape_v, dtype=dtype),
      random.normal(k4, shape_v, dtype=dtype),
  )


def _time_ms(fn, *args, iters: int) -> float:
  for _ in range(min(3, iters)):
    jax.block_until_ready(fn(*args))
  times = []
  for _ in range(iters):
    start = time.perf_counter()
    jax.block_until_ready(fn(*args))
    times.append((time.perf_counter() - start) * 1e3)
  return statistics.median(times)


def _measure(row: Row, mode: str, rate: float, dense: bool):
  """Returns (single call ms, marginal second call ms or None)."""
  dtype = jnp.dtype(_DTYPE.value)
  q, k, v, do = _inputs(row, dtype)
  config = _config(row, rate, dense)
  kernel = splash.make_splash_mha_single_device(
      mask_lib.CausalMask((row.seq_len, row.seq_len)), config=config
  )
  if rate:
    kernel = functools.partial(kernel, prng_key=random.key(1234))

  def wrap(f):
    if mode == "fwd":
      return jax.jit(f)
    return jax.jit(jax.grad(lambda *a: jnp.sum(f(*a) * do), argnums=(0, 1, 2)))

  iters = _ITERS.value or (20 if row.seq_len <= 8192 else 8)
  single = _time_ms(wrap(lambda a, b, c: kernel(a, b, c)), q, k, v, iters=iters)
  marginal = None
  if _MARGINAL.value:
    # Same key both times, so a mask drawn outside the kernel is drawn once.
    double = _time_ms(
        wrap(lambda a, b, c: kernel(a, b, c) + kernel(a, b, c * 2)),
        q, k, v, iters=iters,
    )
    marginal = double - single
  return single, marginal, config


def _flops(row: Row, mode: str) -> float:
  # Causal halves the attention matrix; the backward's five matmuls against the
  # forward's two is the usual 2.5x.
  fwd = (
      2 * _NUM_HEADS.value * row.seq_len * row.seq_len
      * (row.qk_head_dim + row.v_head_dim) * 0.5
  )
  return fwd if mode == "fwd" else 2.5 * fwd


def _draws(row: Row, mode: str, config: splash.SplashConfig) -> int:
  """RNG draws per tile: how many canonical blocks a kernel tile covers."""
  blocks = row.fwd_blocks if mode == "fwd" or row.bwd_blocks is None else row.bwd_blocks
  return ((blocks[0] // config.dropout_block_q)
          * (blocks[2] // config.dropout_block_kv))


def _run(rows: Sequence[Row], mode: str) -> None:
  """Times every row at rate 0 and at `--rate`, for each requested scheme."""
  rate = _RATE.value
  schemes = [s.strip() for s in _SCHEMES.value]
  for scheme in schemes:
    if scheme not in ("in_kernel", "dense"):
      raise ValueError(f"Unknown scheme {scheme!r}.")
    if scheme == "dense" and not _SUPPORTS_DENSE:
      raise ValueError(
          "This build of SplashConfig has no `dropout_dense_mask` field."
      )

  header = (
      f"{'seq':>7s} {'dqk':>4s} {'dv':>4s} {'blocks':>38s} {'grid':>12s}"
      f" {'draws':>5s} {'ms@0':>9s}"
  )
  if _MARGINAL.value:
    header += f" {'marg@0':>9s}"
  header += f" {'MFU@0':>6s}"
  for scheme in schemes:
    header += f" | {scheme + ' ms':>13s} {'over':>8s}"
    if _MARGINAL.value:
      header += f" {'marginal':>9s}"
  print(f"{mode.upper()}  {_NUM_HEADS.value} heads, {_DTYPE.value}, causal, "
        f"rate={rate}")
  print(header)
  print("-" * len(header))

  for row in rows:
    blocks = str(row.fwd_blocks)
    if row.bwd_blocks is not None:
      blocks = f"{row.fwd_blocks}->{row.bwd_blocks}"
    try:
      base, base_marginal, _ = _measure(row, mode, 0.0, False)
    except Exception as e:  # pylint: disable=broad-except
      print(f"{row.seq_len:7d} {row.qk_head_dim:4d} {row.v_head_dim:4d}"
            f" {blocks:>38s}  rate=0 failed: {type(e).__name__}")
      continue
    mfu = _flops(row, mode) / (base * 1e-3) / _PEAK_FLOPS.value

    # The dense scheme has no canonical grid -- nothing is assembled from
    # canonical blocks -- so those two columns come from the in-kernel arm only.
    cells, grid, draws = [], "-", "-"
    for scheme in schemes:
      try:
        ms, marginal, config = _measure(row, mode, rate, scheme == "dense")
        if config.dropout_block_q is not None:
          grid = f"({config.dropout_block_q},{config.dropout_block_kv})"
          draws = str(_draws(row, mode, config))
        cell = f" | {ms:12.3f}m {ms / base - 1:+7.1%}"
        if _MARGINAL.value:
          cell += f" {marginal:8.3f}m"
        cells.append(cell)
      except Exception as e:  # pylint: disable=broad-except
        cells.append(f" | {type(e).__name__[:12]:>13s} {'':>8s}"
                     + (f" {'':>9s}" if _MARGINAL.value else ""))
    line = (f"{row.seq_len:7d} {row.qk_head_dim:4d} {row.v_head_dim:4d}"
            f" {blocks:>38s} {grid:>12s} {draws:>5s} {base:8.3f}m")
    if _MARGINAL.value:
      line += f" {base_marginal:8.3f}m"
    line += f" {mfu:5.1%}"
    print(line + "".join(cells))


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError(f"Unexpected arguments: {argv[1:]}")

  if _PRESET.value is not None:
    rows = _PRESETS[_PRESET.value]
    mode = _MODE.value or ("bwd" if rows[0].bwd_blocks is not None else "fwd")
  else:
    mode = _MODE.value or "fwd"
    rows = (
        Row(
            seq_len=_SEQ_LEN.value,
            qk_head_dim=_QK_HEAD_DIM.value,
            v_head_dim=_V_HEAD_DIM.value,
            layouts=(_Q_LAYOUT.value, _K_LAYOUT.value, _V_LAYOUT.value),
            fwd_blocks=(_BLOCK_Q.value, _BLOCK_KV.value,
                        _BLOCK_KV_COMPUTE.value),
            bwd_blocks=(
                (_BLOCK_Q_DKV.value, _BLOCK_KV_DKV.value,
                 _BLOCK_KV_DKV_COMPUTE.value)
                if mode == "bwd" else None
            ),
            use_experimental_scheduler=_SCHEDULER.value,
        ),
    )
  _run(rows, mode)


if __name__ == "__main__":
  app.run(main)
