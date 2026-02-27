# random_midi_input.py
# Generate a randomized single-voice "MIDI-like" grid input for inference.
#
# Output matches your inference expectations:
#   hit: (T,) float32 in {0,1}
#   vel: (T,) float32 in [0,1] with 0 where hit==0
#
# Params:
#   n_hits: 1..128 (clamped to T)
#   vel_lo, vel_hi: 0..127 (clamped; swapped if out of order)

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import random
import torch


@dataclass(frozen=True)
class RandomMidiInputConfig:
    bars: int = 8
    steps_per_bar: int = 16  # 16ths => 8 bars * 16 = 128 steps


def _clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(x)))


def generate_random_single_voice_sample(
    n_hits: int,
    vel_lo: int,
    vel_hi: int,
    cfg: RandomMidiInputConfig = RandomMidiInputConfig(),
    rng: random.Random | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (hit, vel) where:
      hit: (T,) float32 0/1
      vel: (T,) float32 0..1, with vel=0 where hit=0

    n_hits is the exact number of 1s in hit (unless clamped to T).
    Velocities are uniform random integers in [vel_lo, vel_hi] per hit.
    """
    rng = rng or random

    T = cfg.bars * cfg.steps_per_bar
    if T <= 0:
        raise ValueError(f"Invalid cfg: bars={cfg.bars} steps_per_bar={cfg.steps_per_bar}")

    n_hits = _clamp_int(n_hits, 1, T)

    vel_lo = _clamp_int(vel_lo, 0, 127)
    vel_hi = _clamp_int(vel_hi, 0, 127)
    if vel_lo > vel_hi:
        vel_lo, vel_hi = vel_hi, vel_lo

    hit = torch.zeros(T, dtype=torch.float32)
    vel = torch.zeros(T, dtype=torch.float32)

    # Choose unique hit locations
    steps = rng.sample(range(T), k=n_hits)
    for s in steps:
        hit[s] = 1.0
        v_int = rng.randint(vel_lo, vel_hi)
        vel[s] = float(v_int) / 127.0

    # Ensure non-hit steps are 0 velocity
    vel = vel * hit
    return hit, vel