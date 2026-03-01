from __future__ import annotations
import os
import glob
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import torch
from torch.utils.data import Dataset
import mido


@dataclass
class MidiGridConfig:
    bars: int = 8
    steps_per_bar: int = 16          # 16th-notes
    drum_channel: int = 9            # MIDI channel 10 is index 9
    voice_pitch: Optional[int] = None  # e.g. 42 closed hat, 36 kick, etc.
    # If multiple notes land in same step, how to combine?
    step_merge: str = "max"          # "max" (accent) or "last"
    # Quantization:
    quantize: str = "nearest"        # "nearest" or "floor"
    # If your clips have leading silence, you can optionally shift so first hit starts at step 0:
    align_first_hit_to_zero: bool = False


def _choose_voice_pitch_from_file(mid: mido.MidiFile, drum_channel: Optional[int] = None) -> Optional[int]:
    """
    If drum_channel is None, accept notes from any channel.
    Returns the most common pitch among note_on events.
    """
    counts: Dict[int, int] = {}
    for track in mid.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                chan = getattr(msg, "channel", None)
                if drum_channel is not None and chan != drum_channel:
                    continue
                counts[msg.note] = counts.get(msg.note, 0) + 1

    if not counts:
        return None
    return max(counts.items(), key=lambda kv: kv[1])[0]


def midi_file_to_single_voice_grid(
    path: str,
    cfg: MidiGridConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      hit: (T,) float32 in {0,1}
      vel: (T,) float32 in [0,1], 0 when hit=0
    """
    mid = mido.MidiFile(path)
    ticks_per_beat = mid.ticks_per_beat

    # 16th note = 1/4 beat
    ticks_per_step = ticks_per_beat // 4
    if ticks_per_step <= 0:
        raise ValueError(f"Bad ticks_per_step computed from ticks_per_beat={ticks_per_beat}")

    T = cfg.bars * cfg.steps_per_bar
    total_ticks = T * ticks_per_step

    voice_pitch = cfg.voice_pitch
    if voice_pitch is None:
        voice_pitch = _choose_voice_pitch_from_file(mid, cfg.drum_channel)
        if voice_pitch is None:
            raise ValueError(f"No drum notes found in {path}")

    # Collect note_on events (absolute tick -> velocity) for that pitch+channel
    events: List[Tuple[int, int]] = []
    for track in mid.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if abs_tick < 0:
                continue
            if abs_tick >= total_ticks:
                # ignore anything beyond our fixed 8 bars window
                continue

            if msg.type == "note_on" and msg.velocity > 0:
                chan = getattr(msg, "channel", None)
                if (cfg.drum_channel is None or chan == cfg.drum_channel) and msg.note == voice_pitch:
                    events.append((abs_tick, int(msg.velocity)))

    hit = torch.zeros(T, dtype=torch.float32)
    vel = torch.zeros(T, dtype=torch.float32)

    if not events:
        # valid: totally empty clip
        return hit, vel

    # Optional alignment so the first hit starts at step 0
    tick_shift = 0
    if cfg.align_first_hit_to_zero:
        first_tick = min(t for t, _ in events)
        tick_shift = first_tick

    def tick_to_step(tick: int) -> int:
        tick = max(0, tick - tick_shift)
        if cfg.quantize == "floor":
            step = tick // ticks_per_step
        else:  # "nearest"
            step = int(round(tick / ticks_per_step))
        return step

    for abs_tick, v in events:
        step = tick_to_step(abs_tick)
        if 0 <= step < T:
            v01 = max(0.0, min(1.0, v / 127.0))

            if cfg.step_merge == "last":
                hit[step] = 1.0
                vel[step] = v01
            else:
                # "max" accent merge
                hit[step] = 1.0
                vel[step] = max(float(vel[step].item()), v01)

    # Safety: vel must be 0 where no hit
    vel = vel * hit
    return hit, vel


class SingleVoiceMidiDataset(Dataset):
    def __init__(self, midi_dir: str, cfg: MidiGridConfig):
        self.cfg = cfg
        self.paths = sorted(glob.glob(os.path.join(midi_dir, "*.mid")) + glob.glob(os.path.join(midi_dir, "*.midi")))
        if not self.paths:
            raise FileNotFoundError(f"No MIDI files found in {midi_dir}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path = self.paths[idx]
        hit, vel = midi_file_to_single_voice_grid(path, self.cfg)  # (T,), (T,)

        # Model expects (B,T) for hit/vel in training code
        return {
            "hit": hit,                 # (T,)
            "vel": vel,                 # (T,)
        }