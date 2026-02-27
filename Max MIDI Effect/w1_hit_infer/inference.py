import os
import sys
import json
import random
import argparse
import traceback
from typing import Dict, List, Tuple
from random_midi_input import generate_random_single_voice_sample, RandomMidiInputConfig

import torch

from hit_generator import SingleVoiceTCN

# New API (added in your updated model code)
try:
    from hit_generator import sample_variation
except Exception as e:
    sample_variation = None
    _import_err = e

# MIDI parsing (same quantization idea as training)
try:
    import mido
except Exception as e:
    mido = None
    _mido_err = e


def log(debug: bool, msg: str) -> None:
    if debug:
        print(f"[inference.py] {msg}", file=sys.stderr, flush=True)


# ---------------------------
# Model loading
# ---------------------------

def find_first_pt_file(directory: str) -> str | None:
    if not os.path.isdir(directory):
        return None
    for filename in os.listdir(directory):
        if filename.endswith(".pt"):
            return os.path.join(directory, filename)
    return None


def resolve_first_existing_dir(candidates: List[str]) -> str | None:
    for d in candidates:
        if os.path.isdir(d):
            return d
    return None


def load_first_model(device: str = "cpu", debug: bool = False) -> SingleVoiceTCN:
    # Try both layouts:
    # - cwd = patch dir: w1_hit_infer/Loader exists
    # - cwd = w1_hit_infer: Loader exists
    loader_dir = resolve_first_existing_dir(["w1_hit_infer/Loader", "Loader"])
    if loader_dir is None:
        raise FileNotFoundError(
            "Could not find Loader directory. Tried: w1_hit_infer/Loader and Loader"
        )

    path = find_first_pt_file(loader_dir)
    if path is None:
        raise FileNotFoundError(f"No .pt file found in {loader_dir}")

    log(debug, f"Loading model from: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    config = checkpoint["model_config"]

    model = SingleVoiceTCN(**config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


# ---------------------------
# MIDI -> multi-sample grids
# (each pitch = one sample)
# ---------------------------

def _collect_note_ons(mid: "mido.MidiFile") -> List[Tuple[int, int, int]]:
    """
    Returns list of (abs_ticks, pitch, velocity) for NOTE_ON velocity>0 across all tracks.
    """
    events = []
    abs_ticks = 0

    for msg in mido.merge_tracks(mid.tracks):
        abs_ticks += msg.time
        if msg.type == "note_on" and msg.velocity and msg.velocity > 0:
            events.append((abs_ticks, msg.note, msg.velocity))
    return events


def midi_file_to_multi_pitch_grids(
    path: str,
    bars: int = 8,
    steps_per_bar: int = 16,
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Convert a MIDI file into multiple samples, one per pitch.
    Each sample:
      hit: (T,) float32 in {0,1}
      vel: (T,) float32 in [0,1] with 0 where hit==0
    """
    if mido is None:
        raise ImportError(
            "mido is required to parse MIDI files in inference.py. "
            "Install it (pip install mido) or run inference in the same env as training."
        ) from _mido_err

    mid = mido.MidiFile(path)
    ticks_per_beat = mid.ticks_per_beat
    ticks_per_step = ticks_per_beat // 4  # 16th notes

    if ticks_per_step <= 0:
        raise ValueError(f"Bad ticks_per_step computed from ticks_per_beat={ticks_per_beat}")

    T = bars * steps_per_bar
    total_ticks = T * ticks_per_step

    grids: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    for abs_ticks, pitch, velocity in _collect_note_ons(mid):
        if abs_ticks < 0 or abs_ticks >= total_ticks:
            continue

        step = int(abs_ticks // ticks_per_step)
        if step < 0 or step >= T:
            continue

        if pitch not in grids:
            hit = torch.zeros(T, dtype=torch.float32)
            vel = torch.zeros(T, dtype=torch.float32)
            grids[pitch] = (hit, vel)

        hit, vel = grids[pitch]
        hit[step] = 1.0
        vel_val = float(velocity) / 127.0
        if vel_val > float(vel[step]):
            vel[step] = vel_val

    for pitch, (hit, vel) in list(grids.items()):
        vel = vel * hit
        grids[pitch] = (hit, vel)

    return grids


def pick_random_nonempty_sample_from_midi_folder(
    midi_dir: str,
    bars: int = 8,
    steps_per_bar: int = 16,
    debug: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pick a random .mid/.midi file from midi_dir, parse it into pitch-samples,
    pick a random non-empty pitch sample (>=1 hit), and return (hit, vel).
    """
    # Try both layouts:
    # - midi_dir passed as "MIDI" when cwd is patch dir
    # - midi_dir passed as "MIDI" when cwd is w1_hit_infer (still ok)
    # - OR user passes w1_hit_infer/MIDI
    candidates = [midi_dir]
    if midi_dir == "MIDI":
        candidates.append("w1_hit_infer/MIDI")
    resolved = next((d for d in candidates if os.path.isdir(d)), None)

    log(debug, f"midi_dir arg={midi_dir} candidates={candidates} resolved={resolved}")

    if resolved is None:
        raise FileNotFoundError(f"MIDI folder not found. Tried: {candidates}")

    files = [
        os.path.join(resolved, f)
        for f in os.listdir(resolved)
        if f.lower().endswith((".mid", ".midi"))
    ]
    log(debug, f"Found {len(files)} midi files in {resolved}")

    if not files:
        raise FileNotFoundError(f"No .mid/.midi files found in {resolved}")

    random.shuffle(files)

    for path_ in files:
        grids = midi_file_to_multi_pitch_grids(path_, bars=bars, steps_per_bar=steps_per_bar)
        nonempty = [(p, hv) for p, hv in grids.items() if int(hv[0].sum().item()) > 0]
        log(debug, f"{os.path.basename(path_)} pitches={len(grids)} nonempty={len(nonempty)}")

        if not nonempty:
            continue
        pitch, (hit, vel) = random.choice(nonempty)
        log(debug, f"Selected pitch={pitch} hits={int(hit.sum().item())}")
        return hit, vel

    raise RuntimeError("All MIDI files parsed to empty grids within the chosen bars/length.")


# ---------------------------
# Main
# ---------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--midi_dir", type=str, default="MIDI")
    parser.add_argument("--bars", type=int, default=8)
    parser.add_argument("--steps_per_bar", type=int, default=16)
    parser.add_argument("--edit_fraction", type=float, default=0.25)
    parser.add_argument("--n_iters", type=int, default=8)
    parser.add_argument("--temperature_hit", type=float, default=0.85)
    parser.add_argument("--sigma_floor", type=float, default=0.08)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument("--n_hits", type=int, default=24)
    parser.add_argument("--vel_lo", type=int, default=40)
    parser.add_argument("--vel_hi", type=int, default=110)
    parser.add_argument("--gen_from_midi", action="store_true")
    args = parser.parse_args()

    debug = bool(args.debug)

    log(debug, f"Python={sys.version.split()[0]} torch={torch.__version__}")
    log(debug, f"cwd={os.getcwd()}")
    log(debug, f"argv={sys.argv}")

    if sample_variation is None:
        raise ImportError(
            "Could not import sample_variation from hit_generator. "
            "Make sure your updated hit_generator.py exports sample_variation."
        ) from _import_err

    model = load_first_model(device=args.device, debug=debug)

    if args.gen_from_midi:
        ref_hit, ref_vel = pick_random_nonempty_sample_from_midi_folder(
            midi_dir=args.midi_dir,
            bars=args.bars,
            steps_per_bar=args.steps_per_bar,
            debug=debug,
        )
    else:
        ref_hit, ref_vel = generate_random_single_voice_sample(
            n_hits=args.n_hits,
            vel_lo=args.vel_lo,
            vel_hi=args.vel_hi,
            cfg=RandomMidiInputConfig(bars=args.bars, steps_per_bar=args.steps_per_bar),
        )

    log(debug, f"Running sample_variation: edit_fraction={args.edit_fraction} n_iters={args.n_iters} temp={args.temperature_hit}")

    h, v = sample_variation(
        model,
        ref_hit=ref_hit,
        ref_vel=ref_vel,
        n_iters=args.n_iters,
        device=args.device,
        edit_fraction=args.edit_fraction,
        temperature_hit=args.temperature_hit,
        sigma_floor=args.sigma_floor,
    )

    hit = h.detach().cpu().numpy().astype(int).tolist()
    vel = (v.detach().cpu().numpy() * 127.0).round().astype(int).tolist()

    # Keep original output format (stdout JSON)
    print(json.dumps({"hit": hit, "vel": vel}))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Always print traceback to stderr so Node can show it
        traceback.print_exc(file=sys.stderr)
        raise