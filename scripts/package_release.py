"""Build the complete external-download package for maxforlive.com."""
import argparse
import hashlib
import subprocess
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
DEVICE = Path('Max MIDI Effect/W1_Hit')


def build(destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    files = subprocess.check_output(['git', 'ls-files', '-z', str(DEVICE)], cwd=ROOT).decode().split('\0')
    with ZipFile(destination, 'w', ZIP_DEFLATED) as archive:
        for filename in sorted(filter(None, files)):
            path = Path(filename)
            if any(part.startswith('.') or part == '__pycache__' for part in path.parts):
                continue
            if path.name == 'seed_input_cache.pt' or path.suffix == '.pyc':
                continue
            archive.write(ROOT / path, Path('W1_Hit') / path.relative_to(DEVICE))
        archive.write(ROOT / 'README.md', 'W1_Hit/README.md')
        archive.write(ROOT / 'LICENSE', 'W1_Hit/LICENSE.txt')
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    destination.with_suffix(destination.suffix + '.sha256').write_text(f'{digest}  {destination.name}\n')
    print(f'{destination} ({destination.stat().st_size:,} bytes)\nSHA256 {digest}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', nargs='?', default='dist/W1_Hit-1.0.0.zip')
    build(parser.parse_args().destination)
