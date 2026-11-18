"""Verify and unpack the pinned source/evidence bundle; standard library only."""
from pathlib import Path
import hashlib,zipfile
ROOT=Path(__file__).resolve().parent
archive=ROOT/'freecp-y2-source.zip'
EXPECTED='f67e995907a0aac79988ab0f368c2bedd134e98108b20f92d553207365ac5fb2'
if hashlib.sha256(archive.read_bytes()).hexdigest()!=EXPECTED:
    raise SystemExit('Source archive SHA256 mismatch; obtain the tagged original file.')
target=ROOT/'source_bundle'
with zipfile.ZipFile(archive) as z:
    for n in z.namelist():
        p=Path(n)
        if p.is_absolute() or '..' in p.parts or (z.getinfo(n).external_attr >> 16) & 0o170000 == 0o120000:
            raise SystemExit('Unsafe archive member.')
    target.mkdir(exist_ok=True)
    z.extractall(target)
print('Verified source SHA256:',EXPECTED)
print('Source and evidence:',target)
print('Next: cd source_bundle; follow README.md for CPU analysis or Colab inference.')
