// w1_hit.js  — Variation model version (Node for Max)
// Use with: [node.script w1_hit.js]

const maxApi = require("max-api");
const path = require("path");
const fs = require("fs");
const { spawn } = require("child_process");

// ---------------- CONFIG ----------------
const CFG = {
  pythonBin: "/opt/anaconda3/envs/music-ai/bin/python",
  scriptName: "w1_hit_infer/inference.py",

  // inference defaults
  midiDir: "MIDI",
  bars: 8,
  stepsPerBar: 16,

  edit_fraction: 0.30,   // VARIATION default
  n_iters: 8,            // FILL DEPTH default
  temperature_hit: 0.85,
  sigma_floor: 0.08,
  device: "cpu",

  // note conversion
  stepBeats: 0.25,
  durBeats: 0.25,
  pitch: 36,
  velDefault: 100,

  applyToClip: 1,
  quantizeAnchor: 1,
  debug: 1,

  // log controls
  logStdout: 1,
  logStderr: 1,
  maxTailChars: 4000,
};

const SCRIPT_PATH = path.join(__dirname, CFG.scriptName);

// ---------------- STATE ----------------
let variation = CFG.edit_fraction;
let fillDepth = CFG.n_iters;
let temperature = CFG.temperature_hit;
let genFromMidi = true;
let nHits = 24;              // 1..128
let velLo = 40;              // 0..127
let velHi = 110;             // 0..127
let keepSeedInput = false; 

let lastAnchor = {
  have: false,
  pitch: CFG.pitch,
  vel: CFG.velDefault,
  beat: 0.0,
};

// ---------------- UTILS ----------------
function dpost(msg) {
  if (CFG.debug) maxApi.post("[w1_hit] " + msg);
}

function exists(p) {
  try { return fs.existsSync(p); } catch { return false; }
}

function clamp(x, lo, hi, fallback) {
  const n = Number(x);
  if (!Number.isFinite(n)) return fallback;
  return Math.max(lo, Math.min(hi, n));
}

function buildPythonArgs() {
  const args = [
    SCRIPT_PATH,
    "--midi_dir", CFG.midiDir,
    "--bars", String(CFG.bars),
    "--steps_per_bar", String(CFG.stepsPerBar),
    "--edit_fraction", String(variation),
    "--n_iters", String(fillDepth),
    "--temperature_hit", String(temperature),
    "--sigma_floor", String(CFG.sigma_floor),
    "--device", CFG.device,
    "--debug", "1",
  ];

  if (keepSeedInput) {
    args.push("--keep_seed_input");
  }

  if (genFromMidi) {
    args.push("--gen_from_midi");
  } else {
    args.push(
      "--n_hits", String(nHits),
      "--vel_lo", String(velLo),
      "--vel_hi", String(velHi),
    );
  }

  return args;
}

function tailStr(s, n) {
  if (!s) return "";
  if (s.length <= n) return s;
  return s.slice(s.length - n);
}

function parseLastJson(stdout) {
  const trimmed = (stdout || "").trim();
  const lines = trimmed.split(/\r?\n/).reverse();
  for (const s of lines) {
    if (s && s[0] === "{") {
      try { return JSON.parse(s); } catch {}
    }
  }
  throw new Error("No JSON object found in stdout (maybe Python crashed before printing JSON).");
}

function wireLineLogger(stream, label) {
  let buf = "";
  stream.on("data", (chunk) => {
    buf += chunk.toString("utf8");
    let idx;
    while ((idx = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, idx).replace(/\r$/, "");
      buf = buf.slice(idx + 1);
      if (line.length) maxApi.post(`[w1_hit][py:${label}] ${line}`);
    }
  });
}

function runPython() {
  return new Promise((resolve, reject) => {
    const args = buildPythonArgs();

    // Big debugging header
    dpost(`CWD(node)=${process.cwd()}`);
    dpost(`__dirname=${__dirname}`);
    dpost(`pythonBin=${CFG.pythonBin} exists=${exists(CFG.pythonBin)}`);
    dpost(`SCRIPT_PATH=${SCRIPT_PATH} exists=${exists(SCRIPT_PATH)}`);
    dpost(`args=${JSON.stringify(args)}`);

    // IMPORTANT: set cwd so relative MIDI/Loader paths resolve to your patch folder
    const p = spawn(CFG.pythonBin, args, {
      cwd: __dirname,
      env: process.env,
    });

    let out = "";
    let err = "";

    if (CFG.logStdout) wireLineLogger(p.stdout, "stdout");
    if (CFG.logStderr) wireLineLogger(p.stderr, "stderr");

    p.stdout.on("data", (d) => { out += d.toString("utf8"); });
    p.stderr.on("data", (d) => { err += d.toString("utf8"); });

    p.on("error", (e) => {
      // spawn failed (python path wrong, permissions, etc.)
      const msg = `spawn error: ${e && e.message ? e.message : String(e)}`;
      dpost(msg);
      reject(new Error(msg));
    });

    p.on("close", (code, signal) => {
      dpost(`python exit code=${code} signal=${signal || "none"}`);

      if (code !== 0) {
        const tail = tailStr(err || out, CFG.maxTailChars);
        reject(new Error(`Python failed (code=${code}). Tail:\n${tail}`));
        return;
      }

      try {
        const json = parseLastJson(out);
        resolve(json);
      } catch (e) {
        const tail = tailStr(out || err, CFG.maxTailChars);
        reject(new Error(`JSON parse failed: ${e.message}\nTail:\n${tail}`));
      }
    });
  });
}

// ---------------- NOTE CONVERSION ----------------
function toNotes(hit, vel) {
  const notes = [];

  const pitch = lastAnchor.have ? lastAnchor.pitch : CFG.pitch;
  const velDefault = lastAnchor.have ? lastAnchor.vel : CFG.velDefault;
  const anchorBeat = lastAnchor.have ? lastAnchor.beat : 0;

  for (let i = 0; i < hit.length; i++) {
    if (!hit[i]) continue;

    notes.push({
      pitch,
      start_time: anchorBeat + i * CFG.stepBeats,
      duration: CFG.durBeats,
      velocity: vel[i] || velDefault,
      mute: 0,
    });
  }
  return notes;
}

// ---------------- GENERATE ----------------
async function generate() {
  dpost(`generate() variation=${variation} fillDepth=${fillDepth} temperature=${temperature}`);

  try {
    const json = await runPython();

    // Validate expected output shape early
    if (!json || !Array.isArray(json.hit) || !Array.isArray(json.vel)) {
      throw new Error(`Bad JSON shape: expected {hit:[...], vel:[...]}, got: ${JSON.stringify(json).slice(0, 400)}`);
    }

    const notes = toNotes(json.hit, json.vel);

    maxApi.outlet(["hitvel", JSON.stringify(json)]);
    maxApi.outlet(["notes", JSON.stringify(notes)]);

    if (CFG.applyToClip) {
      // NOTE: delete/add are done via clipcmd downstream
      maxApi.outlet(["clipcmd", "remove_notes_extended", CFG.pitch, CFG.pitch, 0, 9999]);
      maxApi.outlet(["clipcmd", "add_new_notes", JSON.stringify({ notes })]);
    }
  } catch (e) {
    // DO NOT crash node.script — report error instead
    const msg = (e && e.message) ? e.message : String(e);
    dpost("ERROR: " + msg);
    maxApi.outlet(["error", msg]);
  }
}

// ---------------- HANDLERS ----------------
maxApi.post("[w1_hit] variation model ready");
maxApi.outlet("ready");

maxApi.addHandler("variation", (v) => {
  variation = clamp(v, 0, 1, variation);
  dpost("variation → " + variation);
});

maxApi.addHandler("fill_depth", (v) => {
  fillDepth = Math.round(clamp(v, 1, 32, fillDepth));
  dpost("fill_depth → " + fillDepth);
});

maxApi.addHandler("temperature", (v) => {
  temperature = clamp(v, 0.1, 2.0, temperature);
  dpost("temperature → " + temperature);
});

maxApi.addHandler("keep_seed_input", (v) => {
  keepSeedInput = !!Number(v);
  dpost("keep_seed_input → " + keepSeedInput);
});

maxApi.addHandler("anchor_pitch", (pitch, vel) => {
  lastAnchor = {
    have: true,
    pitch: clamp(pitch, 0, 127, CFG.pitch),
    vel: clamp(vel, 1, 127, CFG.velDefault),
    beat: 0,
  };
  dpost(`anchor_pitch → pitch=${lastAnchor.pitch} vel=${lastAnchor.vel}`);
});

maxApi.addHandler("gen_from_midi", (v) => {
  // accept 0/1, false/true, etc.
  genFromMidi = !!Number(v);
  dpost("gen_from_midi → " + genFromMidi);
});

maxApi.addHandler("n_hits", (v) => {
  nHits = Math.round(clamp(v, 1, 128, nHits));
  dpost("n_hits → " + nHits);
});

maxApi.addHandler("vel_lo", (v) => {
  velLo = Math.round(clamp(v, 0, 127, velLo));
  // keep lo <= hi
  if (velLo > velHi) velLo = velHi;
  dpost("vel_lo → " + velLo);
});

maxApi.addHandler("vel_hi", (v) => {
  velHi = Math.round(clamp(v, 0, 127, velHi));
  // keep lo <= hi
  if (velHi < velLo) velHi = velLo;
  dpost("vel_hi → " + velHi);
});

maxApi.addHandler("bang", generate);
maxApi.addHandler("int", (n) => { if (n) generate(); });