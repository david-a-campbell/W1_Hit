# 🎛 W1 Hit
### An AI-powered MIDI rhythm generator with Ableton Live integration

**Music is defined as a sound that repeats.**  
**Rhythm shares the same definition and thus is the basis of all music.**

W1 Hit is an AI system for generating expressive single-voice drum patterns. It trains on MIDI rhythms and produces musically coherent variations, fills, and new sequences that can be inserted directly into Ableton Live using a Max for Live device.

Designed for producers, sound designers, and researchers, W1 Hit focuses on controllable generation rather than randomness — allowing users to shape output using parameters such as variation and temperature.

✨ Features

🧠 Train on single-voice MIDI drum patterns  
🎚 Generate new rhythms and variations  
🥁 Velocity-aware output (not just note placement)  
🎛 Parameter-controlled generation  
⚡ Real-time Ableton Live integration  
🎹 MIDI-native workflow (no audio preprocessing)  

🧠 How It Works

W1 Hit trains a Temporal Convolutional Network (TCN) on quantized MIDI grids representing a single drum voice.
During inference, the model generates new patterns conditioned on:
* training data distribution
* input samples
* user-controlled parameters
  
This enables musically coherent results suitable for production workflows.

🎛 Use Cases

* Drum pattern generation for producers  
* Creating variations of existing grooves  
* AI-assisted composition tools  
* Symbolic music generation research  
* Max for Live instrument development  

📂 Project Structure

```
W1_Hit/
│
├── HitGenerator/                # Training pipeline
│   ├── hit_generator.py
│   ├── mid_to_velocity.py
│   ├── midi_export.py
│   ├── Models/
│   └── notebooks
│
├── Max MIDI Effect/             # Ableton integration
│   ├── w1_hit.amxd              # Max for Live device
│   ├── w1_hit.js                # Node for Max interface
│   ├── clipcmd_to_live.js       # MIDI insertion logic
│   └── w1_hit_infer/            # Inference engine
│       ├── inference.py
│       ├── hit_generator.py
│       ├── mid_to_velocity.py
│       ├── Loader/
│       ├── MIDI/
│       └── random_midi_input.py
│
├── requirements.txt
└── README.md
```
🛠 Installation
1) Clone the repository
```
git clone https://github.com/david-a-campbell/W1_Hit.git
cd W1_Hit
```
2) Install Python dependencies
```
pip install -r requirements.txt
```
(Recommended: use a virtual environment)

### 3) Ableton Live Setup

1. Open Ableton Live  
2. Go to **User Library → Presets → MIDI Effects → Max MIDI Effect**  
3. Drag the all the files from the `Max MIDI Effect` folder into this folder  

🚀 Quick Start
Train a model

Place MIDI files into the training dataset folder (called MIDI) and run:
```
python HitGenerator/hit_generator.py
```
Trained models will be saved in:
```
HitGenerator/Models/
```

Copy your trained model to:
```
Max MIDI Effect/w1_hit_infer/Loader
```

Generate patterns:
Inference is handled by:
```
Max MIDI Effect/w1_hit_infer/inference.py
```
This script loads a trained model and generates MIDI patterns.

Use inside Ableton Live:
1. Insert the Max for Live device on a MIDI track
2. Trigger generation
3. Generated patterns are inserted into the MIDI clip

🎚 Generation Parameters

W1 Hit exposes controllable parameters to shape output:
  * Variation — degree of deviation from input pattern
  * Temperature — randomness vs determinism
  * Fill Depth — density of added notes  
These allow producers to dial in results that fit their groove.

🤝 Contributing
Contributions, issues, and suggestions are welcome.

🎧 About  
Created by W1NGY — a project exploring AI-assisted music production tools.

🇯🇲 Here are some example songs I created using W1_Hit. Samples contain vocals from Splice:  
[Riddim 1](https://drive.google.com/file/d/1_0SbdAahy8-wQNbpLn0izOkheZ4mTRIL/view?usp=sharing)  
[Riddim 2](https://drive.google.com/file/d/1A7YmxghzXO1f03oh6YUJPuZ4DZBoisaK/view?usp=sharing)  
[Riddim 3](https://drive.google.com/file/d/1oqbPUerFjXpN8j4oxPI2FK8zR71hyeVt/view?usp=sharing)  
[Riddim 4](https://drive.google.com/file/d/13NxNwOmxnKZjr7FRSggYpGjnPOHLx6pq/view?usp=sharing)  

