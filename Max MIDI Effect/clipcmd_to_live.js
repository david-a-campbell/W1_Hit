// clipcmd_to_live.js  (Max JS, NOT node.script)
//
// Use with wiring:
//   [node.script w1_hit.js] (LEFT outlet)
//        |
//     [route clipcmd]          <-- strips "clipcmd"
//        |
//   [js clipcmd_to_live.js]
//
// Expects messages AFTER route:
//   remove_notes_extended 0 127 0 9999
//   add_new_notes {"notes":[...]}   (JSON string)
//   (optional) reset

autowatch = 1;
outlets = 1;

var clip = null;

function ensure_clip() {
  if (!clip) {
    clip = new LiveAPI("live_set view detail_clip");
  }
}

function reset() {
  clip = null;
  post("[clipcmd_to_live] reset\n");
}

// Add anywhere in clipcmd_to_live.js

function get_playing_position() {
  ensure_clip();

  var live_api = new LiveAPI("live_set view detail_clip");
  try {
    var v = live_api.get("playing_position"); // returns [number]
    var beat = (v && v.length) ? v[0] : 0.0;
    outlet(0, "playing_position", beat);
  } catch (e) {
    post("[clipcmd_to_live] get_playing_position error: " + e + "\n");
    outlet(0, "playing_position", 0.0);
  }
}

// Called as: remove_notes_extended 0 127 0 9999
function remove_notes_extended(pitch_lo, pitch_hi, time_lo, time_hi) {
  ensure_clip();

  // Coerce to numbers
  var plo = Number(pitch_lo);
  var phi = Number(pitch_hi);
  var tlo = Number(time_lo);
  var thi = Number(time_hi);

  // Guard: never delete if any arg is missing/NaN
  if (!isFinite(plo) || !isFinite(phi) || !isFinite(tlo) || !isFinite(thi)) {
    post("[clipcmd_to_live] remove_notes_extended: BAD ARGS -> " +
      pitch_lo + " " + pitch_hi + " " + time_lo + " " + time_hi + "\n");
    return;
  }

  // Clamp and integerize pitch (Live expects MIDI note numbers)
  plo = Math.max(0, Math.min(127, Math.floor(plo)));
  phi = Math.max(0, Math.min(127, Math.floor(phi)));

  // Optional: clamp times too
  if (tlo < 0) tlo = 0;
  if (thi < tlo) thi = tlo;

  var live_api = new LiveAPI("live_set view detail_clip");
  try {
	// TODO: Fix remove call	
    //live_api.call("remove_notes_extended", plo, phi, tlo, thi);
  } catch (e) {
    post("[clipcmd_to_live] remove_notes_extended error: " + e + "\n");
  }
}

// Called as: add_new_notes "<json string>"
// where json is either {"notes":[...]} or [...]
function add_new_notes(payloadStr) {
  ensure_clip();

  if (payloadStr === undefined) {
    post("[clipcmd_to_live] add_new_notes: missing payload\n");
    return;
  }

	var live_api = new LiveAPI("live_set view detail_clip");
  try {
    var payload = JSON.parse(payloadStr);

    // Support both {"notes":[...]} and [...]
    if (payload && payload.notes) {
		live_api.call("add_new_notes", payload);
    } else if (Array.isArray(payload)) {
		live_api.call("add_new_notes", {"notes" : payload});
    } else {
      post("[clipcmd_to_live] add_new_notes: payload must be {notes:[...]} or [...]\n");
    }
  } catch (e2) {
    post("[clipcmd_to_live] add_new_notes JSON parse error: " + e2 + "\n");
    post("[clipcmd_to_live] payloadStr: " + payloadStr + "\n");
  }
}