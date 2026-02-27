from __future__ import annotations
import os
import glob
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Iterable

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


def _iter_note_on_events(
    mid: mido.MidiFile,
    *,
    total_ticks: int,
    drum_channel: Optional[int],
) -> Iterable[Tuple[int, int, int]]:
    """
    Yields (abs_tick, pitch, velocity) for note_on velocity>0 within [0, total_ticks).
    If drum_channel is None, accepts all channels.
    """
    for track in mid.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if abs_tick < 0:
                continue
            if abs_tick >= total_ticks:
                continue
            if msg.type == "note_on" and msg.velocity > 0:
                chan = getattr(msg, "channel", None)
                if drum_channel is not None and chan != drum_channel:
                    continue
                yield abs_tick, int(msg.note), int(msg.velocity)


def midi_file_to_single_voice_grid(
    path: str,
    cfg: MidiGridConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      hit: (T,) float32 in {0,1}
      vel: (T,) float32 in [0,1], 0 when hit=0

    One voice only: uses cfg.voice_pitch, or auto-selects most common pitch if None.
    """
    mid = mido.MidiFile(path)
    ticks_per_beat = mid.ticks_per_beat

    ticks_per_step = ticks_per_beat // 4
    if ticks_per_step <= 0:
        raise ValueError(f"Bad ticks_per_step computed from ticks_per_beat={ticks_per_beat}")

    T = cfg.bars * cfg.steps_per_bar
    total_ticks = T * ticks_per_step

    voice_pitch = cfg.voice_pitch
    if voice_pitch is None:
        voice_pitch = _choose_voice_pitch_from_file(mid, drum_channel=cfg.drum_channel)
        if voice_pitch is None:
            # No notes at all
            hit = torch.zeros(T, dtype=torch.float32)
            vel = torch.zeros(T, dtype=torch.float32)
            return hit, vel

    # Collect note_on events for this pitch
    events: List[Tuple[int, int]] = []
    for abs_tick, pitch, v in _iter_note_on_events(mid, total_ticks=total_ticks, drum_channel=cfg.drum_channel):
        if pitch == voice_pitch:
            events.append((abs_tick, v))

    hit = torch.zeros(T, dtype=torch.float32)
    vel = torch.zeros(T, dtype=torch.float32)

    if not events:
        return hit, vel

    tick_shift = 0
    if cfg.align_first_hit_to_zero:
        tick_shift = min(t for t, _ in events)

    def tick_to_step(tick: int) -> int:
        tick = max(0, tick - tick_shift)
        if cfg.quantize == "floor":
            step = tick // ticks_per_step
        else:
            step = int(round(tick / ticks_per_step))
        return int(step)

    for abs_tick, v in events:
        step = tick_to_step(abs_tick)
        if 0 <= step < T:
            v01 = max(0.0, min(1.0, v / 127.0))
            if cfg.step_merge == "last":
                hit[step] = 1.0
                vel[step] = v01
            else:
                hit[step] = 1.0
                vel[step] = max(float(vel[step].item()), v01)

    vel = vel * hit
    return hit, vel


def midi_file_to_multi_voice_grids(
    path: str,
    cfg: MidiGridConfig,
    *,
    min_hits_per_voice: int = 1,
    max_hits_per_voice: Optional[int] = None,
    include_pitch: bool = True,
) -> List[Dict[str, torch.Tensor]]:
    """
    Treats EACH pitch in the MIDI as a separate training record.

    Returns a list of dicts:
      {"hit": (T,), "vel": (T,), "pitch": scalar-int tensor}  (pitch optional)

    Notes:
    - Respects cfg.drum_channel (or all channels if cfg.drum_channel is None).
    - Uses the same quantize/merge logic as the single-voice extractor.
    - Alignment (align_first_hit_to_zero) is applied PER VOICE (per pitch).
    - Filters voices by min_hits_per_voice and (optionally) max_hits_per_voice.
    """
    mid = mido.MidiFile(path)
    ticks_per_beat = mid.ticks_per_beat

    ticks_per_step = ticks_per_beat // 4
    if ticks_per_step <= 0:
        raise ValueError(f"Bad ticks_per_step computed from ticks_per_beat={ticks_per_beat}")

    T = cfg.bars * cfg.steps_per_bar
    total_ticks = T * ticks_per_step

    # Gather events grouped by pitch
    events_by_pitch: Dict[int, List[Tuple[int, int]]] = {}
    for abs_tick, pitch, v in _iter_note_on_events(mid, total_ticks=total_ticks, drum_channel=cfg.drum_channel):
        events_by_pitch.setdefault(pitch, []).append((abs_tick, v))

    records: List[Dict[str, torch.Tensor]] = []

    for pitch, events in sorted(events_by_pitch.items(), key=lambda kv: kv[0]):
        # Build grid for this pitch
        hit = torch.zeros(T, dtype=torch.float32)
        vel = torch.zeros(T, dtype=torch.float32)

        if not events:
            continue

        tick_shift = 0
        if cfg.align_first_hit_to_zero:
            tick_shift = min(t for t, _ in events)

        def tick_to_step(tick: int) -> int:
            tick = max(0, tick - tick_shift)
            if cfg.quantize == "floor":
                step = tick // ticks_per_step
            else:
                step = int(round(tick / ticks_per_step))
            return int(step)

        hits_count = 0
        for abs_tick, v in events:
            step = tick_to_step(abs_tick)
            if 0 <= step < T:
                v01 = max(0.0, min(1.0, v / 127.0))
                if cfg.step_merge == "last":
                    if hit[step].item() == 0.0:
                        hits_count += 1
                    hit[step] = 1.0
                    vel[step] = v01
                else:
                    if hit[step].item() == 0.0:
                        hits_count += 1
                    hit[step] = 1.0
                    vel[step] = max(float(vel[step].item()), v01)

        vel = vel * hit

        if hits_count < min_hits_per_voice:
            continue
        if max_hits_per_voice is not None and hits_count > int(max_hits_per_voice):
            continue

        rec: Dict[str, torch.Tensor] = {"hit": hit, "vel": vel}
        if include_pitch:
            rec["pitch"] = torch.tensor(int(pitch), dtype=torch.int64)
        records.append(rec)

    return records


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
        hit, vel = midi_file_to_single_voice_grid(path, self.cfg)
        return {"hit": hit, "vel": vel}


class MultiVoiceMidiDataset(Dataset):
    """
    Dataset that treats each distinct pitch in each MIDI file as one record.
    This allows multiple training samples from a single MIDI file.

    It builds an index at init-time using a fast (unique-step) hit-count approximation,
    filtering by min_hits_per_voice and (optionally) max_hits_per_voice.
    """
    def __init__(
        self,
        midi_dir: str,
        cfg: MidiGridConfig,
        *,
        min_hits_per_voice: int = 1,
        max_hits_per_voice: Optional[int] = None, # Default None,
        include_pitch: bool = True,
    ):
        self.cfg = cfg
        self.include_pitch = include_pitch
        self.min_hits_per_voice = int(min_hits_per_voice)
        self.max_hits_per_voice = None if max_hits_per_voice is None else int(max_hits_per_voice)

        paths = sorted(glob.glob(os.path.join(midi_dir, "*.mid")) + glob.glob(os.path.join(midi_dir, "*.midi")))
        if not paths:
            raise FileNotFoundError(f"No MIDI files found in {midi_dir}")

        # Build an index: list of (path, pitch)
        index: List[Tuple[str, int]] = []
        for path in paths:
            try:
                mid = mido.MidiFile(path)
            except Exception:
                continue

            ticks_per_beat = mid.ticks_per_beat
            ticks_per_step = ticks_per_beat // 4
            if ticks_per_step <= 0:
                continue

            T = cfg.bars * cfg.steps_per_bar
            total_ticks = T * ticks_per_step

            pitch_to_steps: Dict[int, set] = {}
            for abs_tick, pitch, _v in _iter_note_on_events(mid, total_ticks=total_ticks, drum_channel=cfg.drum_channel):
                # approximate unique hit count by unique step index (fast prefilter)
                if cfg.quantize == "floor":
                    step = abs_tick // ticks_per_step
                else:
                    step = int(round(abs_tick / ticks_per_step))
                if 0 <= step < T:
                    pitch_to_steps.setdefault(int(pitch), set()).add(int(step))

            for pitch, steps in pitch_to_steps.items():
                n = len(steps)
                if n < self.min_hits_per_voice:
                    continue
                if self.max_hits_per_voice is not None and n > self.max_hits_per_voice:
                    continue
                index.append((path, int(pitch)))

        if not index:
            raise FileNotFoundError(
                f"No usable (file,pitch) records found in {midi_dir}. "
                f"Try lowering min_hits_per_voice / raising max_hits_per_voice, or adjusting drum_channel."
            )

        self.index = index

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path, pitch = self.index[idx]

        # Reuse your single-voice path by temporarily setting cfg.voice_pitch
        # (avoid mutating shared cfg by constructing a new config)
        cfg2 = MidiGridConfig(**{**self.cfg.__dict__, "voice_pitch": pitch})
        hit, vel = midi_file_to_single_voice_grid(path, cfg2)

        rec: Dict[str, torch.Tensor] = {"hit": hit, "vel": vel}
        if self.include_pitch:
            rec["pitch"] = torch.tensor(int(pitch), dtype=torch.int64)
        return rec