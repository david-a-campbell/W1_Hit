# midi_export.py
# Export a single-voice 16th-note step pattern (hit + continuous velocity) to a MIDI file.
# The filename is a random UUID (e.g., "3f2c9c3b3c9b4a0b8f7c9b5b4a2d1c0e.mid")

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import Optional, Tuple

import mido


@dataclass
class MidiExportConfig:
    tempo_bpm: float = 120.0
    bars: int = 8
    steps_per_bar: int = 16          # 16th notes
    drum_channel: int = 9            # MIDI channel 10 is index 9
    voice_pitch: int = 42            # e.g. 42 closed hat
    note_length_steps: float = 0.25  # 0.25 = 1/64 note, 1.0 = 1/16 note
    # If your DAW doesn't care about note length for drums, keep it short.
    min_velocity: int = 1            # MIDI velocity must be 1..127 for a note_on
    max_velocity: int = 127
    clamp_velocities: bool = True


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def export_single_voice_steps_to_midi(
    hit,
    vel,
    out_dir: str,
    cfg: MidiExportConfig,
    filename_uuid: Optional[str] = None,
) -> str:
    """
    hit: 1D array-like length T with 0/1 (or bool)
    vel: 1D array-like length T with 0..1 (continuous). Should be 0 when hit=0.

    Returns full output path to the written .mid file.
    """
    # Convert to plain Python lists
    hit_list = [int(1 if float(h) >= 0.5 else 0) for h in hit]
    vel_list = [float(v) for v in vel]

    T_expected = cfg.bars * cfg.steps_per_bar
    if len(hit_list) != T_expected or len(vel_list) != T_expected:
        raise ValueError(
            f"Expected length T={T_expected} (bars={cfg.bars}, steps_per_bar={cfg.steps_per_bar}), "
            f"got hit={len(hit_list)} vel={len(vel_list)}"
        )

    os.makedirs(out_dir, exist_ok=True)

    file_id = filename_uuid or str(uuid.uuid4())
    out_path = os.path.join(out_dir, f"{file_id}.mid")

    mid = mido.MidiFile(ticks_per_beat=480)  # common PPQ; Ableton reads fine
    track = mido.MidiTrack()
    mid.tracks.append(track)

    # Tempo meta
    tempo = mido.bpm2tempo(cfg.tempo_bpm)  # microseconds per beat
    track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))

    # 4/4 time signature meta (optional but nice)
    track.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0))

    # Timing
    ticks_per_beat = mid.ticks_per_beat
    ticks_per_step = ticks_per_beat // 4  # 16th note
    note_len_ticks = max(1, int(round(ticks_per_step * cfg.note_length_steps)))

    # We accumulate absolute tick times then convert to delta times for MIDI messages
    events: list[Tuple[int, mido.Message]] = []

    for step in range(T_expected):
        if hit_list[step] == 0:
            continue

        v01 = vel_list[step]
        if cfg.clamp_velocities:
            v01 = _clamp01(v01)

        velocity = int(round(v01 * cfg.max_velocity))
        velocity = max(cfg.min_velocity, min(cfg.max_velocity, velocity))

        start_tick = step * ticks_per_step
        end_tick = start_tick + note_len_ticks

        events.append(
            (start_tick, mido.Message("note_on", channel=cfg.drum_channel, note=cfg.voice_pitch, velocity=velocity, time=0))
        )
        events.append(
            (end_tick, mido.Message("note_off", channel=cfg.drum_channel, note=cfg.voice_pitch, velocity=0, time=0))
        )

    # Sort by time; note_off after note_on when same tick
    events.sort(key=lambda x: (x[0], 0 if x[1].type == "note_on" else 1))

    # Write with delta times
    last_tick = 0
    for abs_tick, msg in events:
        delta = abs_tick - last_tick
        msg.time = max(0, int(delta))
        track.append(msg)
        last_tick = abs_tick

    # End of track
    track.append(mido.MetaMessage("end_of_track", time=0))

    mid.save(out_path)
    return out_path


# Example usage:
#   h, v = sample_pattern(model, T=128, ...)
#   path = export_single_voice_steps_to_midi(h.numpy(), v.numpy(), out_dir="./out",
#                                           cfg=MidiExportConfig(tempo_bpm=124, voice_pitch=42))
#   print("Wrote:", path)