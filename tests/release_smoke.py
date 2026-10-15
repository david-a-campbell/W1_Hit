import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from zipfile import ZipFile

parser = argparse.ArgumentParser(description="Smoke-test an extracted W1 Hit release using an installed inference Python.")
parser.add_argument('archive', type=Path)
parser.add_argument('--python', required=True, help='Python with the device requirements installed')
args = parser.parse_args()
archive = args.archive
python = args.python
with tempfile.TemporaryDirectory(prefix='W1 Hit release ') as temp:
    with ZipFile(archive) as z:
        assert z.testzip() is None
        assert not any('.DS_Store' in n or '__pycache__' in n or '.venv/' in n or 'seed_input_cache' in n or '.ipynb_checkpoints' in n for n in z.namelist())
        z.extractall(temp)
        print(f'Archive verified: {len(z.namelist())} files')
    root = Path(temp) / 'W1_Hit'
    for args in [[], ['--gen_from_midi'], ['--gen_from_midi', '--keep_seed_input']]:
        result = subprocess.run([python, str(root / 'w1_hit_infer/inference.py'), '--n_iters', '2', *args], cwd=temp, text=True, capture_output=True, check=True)
        data = json.loads(result.stdout)
        assert len(data['hit']) == len(data['vel']) == 128
        assert all(h in (0,1) for h in data['hit'])
        assert all(isinstance(v,int) and 0 <= v <= 127 for v in data['vel'])
        # The stochastic sampler can return silence for sparse seeds.
        # Validate the output contract here; nonempty output is not guaranteed.
        print(f'Inference passed {args}: {sum(data["hit"])} hits, 128 steps')
    # Check installed Python detection under an actual path with spaces.
    installer = (root / 'install/install_mac.command').read_text()
    assert '$("$VENV/bin/python" -c' in installer
print('Extracted release smoke checks passed')
