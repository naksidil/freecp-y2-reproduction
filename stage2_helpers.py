"""Small standard-library helpers for the isolated Colab experiment."""
from __future__ import annotations
import hashlib
import json
import shutil
import subprocess
import tarfile
import time
import urllib.request
from pathlib import Path, PurePosixPath

UPSTREAM_SHA = '6946aa7999ec5d57d1fe91968ba1ae35a1e53295'
CLIP_SHA = 'd05afc436d78f1c48dc0dbf8e5980a9d471f35f6'
BPE_GIT_BLOB_SHA = '7b5088a527f720063f044eb928eee315f63b2fc0'
VOC_MD5 = '6cd6e144f989b92b3379bac3b3de84fd'
VOC_URL = 'https://thor.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.writing')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def git_blob_sha(path):
    path = Path(path)
    h = hashlib.sha1(f'blob {path.stat().st_size}\0'.encode())
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def run_logged(command, label, artifacts, cwd=None, env=None):
    """Stream output and preserve a stage result, including on process errors."""
    artifacts = Path(artifacts)
    artifacts.mkdir(parents=True, exist_ok=True)
    status_path = artifacts / 'stages.json'
    states = json.loads(status_path.read_text()) if status_path.exists() else {}
    record = {'command': [str(x) for x in command], 'cwd': str(cwd) if cwd else None,
              'ok': False, 'started_unix': time.time()}
    states[label] = record
    save_json(status_path, states)
    try:
        with (artifacts / f'{label}.log').open('w', encoding='utf-8') as log:
            with subprocess.Popen([str(x) for x in command], cwd=cwd, env=env,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, bufsize=1) as process:
                try:
                    for line in process.stdout:
                        print(line, end='', flush=True)
                        log.write(line)
                        log.flush()
                    record['returncode'] = process.wait()
                    record['ok'] = record['returncode'] == 0
                except KeyboardInterrupt:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    record['error'] = 'Interrupted by user'
                    raise
        if not record['ok']:
            raise RuntimeError(f'{label} failed. Preserve this output and run the final ZIP cell.')
    except Exception as exc:
        record['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        record['wall_seconds'] = time.time() - record['started_unix']
        save_json(status_path, states)


def download(url, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + '.partial')
    request = urllib.request.Request(url, headers={'User-Agent': 'FreeCP-course-reproduction/0.2'})
    with urllib.request.urlopen(request, timeout=90) as src, partial.open('wb') as dst:
        count, last = 0, 0
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            dst.write(chunk)
            count += len(chunk)
            if count - last >= 128 * 1024**2:
                print(f'Downloaded {count / 1024**2:.0f} MiB', flush=True)
                last = count
    partial.replace(destination)
    return destination


def ensure_tokenizer(repo, artifacts):
    repo, artifacts = Path(repo), Path(artifacts)
    bpe = repo / 'open_clip/bpe_simple_vocab_16e6.txt.gz'
    url = f'https://raw.githubusercontent.com/openai/CLIP/{CLIP_SHA}/clip/bpe_simple_vocab_16e6.txt.gz'
    if not bpe.exists():
        download(url, bpe)
    if git_blob_sha(bpe) != BPE_GIT_BLOB_SHA:
        raise ValueError('Tokenizer file does not match the pinned OpenAI CLIP Git blob.')
    license_path = artifacts / 'licenses/OpenAI_CLIP_LICENSE.txt'
    download(f'https://raw.githubusercontent.com/openai/CLIP/{CLIP_SHA}/LICENSE', license_path)
    shutil.copy2(repo / 'LICENSE', artifacts / 'licenses/FreeCP_LICENSE.txt')
    save_json(artifacts / 'tokenizer_asset.json', {
        'url': url, 'clip_commit': CLIP_SHA, 'git_blob_sha1': BPE_GIT_BLOB_SHA,
        'sha256': digest(bpe), 'bytes': bpe.stat().st_size,
        'note': 'Missing upstream tokenizer asset supplied; no algorithm change.'})


def extract_voc_files(archive, destination):
    """Extract only regular files needed for segmentation; never extract links."""
    destination = Path(destination).resolve()
    allowed = ('VOCdevkit/VOC2012/JPEGImages/',
               'VOCdevkit/VOC2012/SegmentationClass/',
               'VOCdevkit/VOC2012/ImageSets/Segmentation/')
    with tarfile.open(archive, 'r:*') as tar:
        for member in tar:
            name = member.name.removeprefix('./')
            if not name.startswith(allowed):
                continue
            path = PurePosixPath(name)
            if '..' in path.parts or path.is_absolute() or member.issym() or member.islnk():
                raise ValueError(f'Unsafe archive entry: {name}')
            if not member.isfile():
                continue
            target = destination.joinpath(*path.parts).resolve()
            if not target.is_relative_to(destination):
                raise ValueError('Archive entry escapes destination.')
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as src, target.open('wb') as dst:
                shutil.copyfileobj(src, dst)


def prepare_voc(data_dir, artifacts):
    data_dir, artifacts = Path(data_dir), Path(artifacts)
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / 'VOCtrainval_11-May-2012.tar'
    marker = data_dir / 'voc_verified.json'
    voc = data_dir / 'VOCdevkit/VOC2012'
    if not marker.exists():
        if not archive.exists():
            if shutil.disk_usage(data_dir).free < 5 * 1024**3:
                raise RuntimeError('At least 5 GiB free space is needed before downloading VOC.')
            download(VOC_URL, archive)
        if digest(archive, 'md5') != VOC_MD5:
            raise ValueError('VOC archive checksum mismatch; do not use this archive.')
        extract_voc_files(archive, data_dir)
        save_json(marker, {'url': VOC_URL, 'archive_md5': VOC_MD5,
                          'archive_bytes': archive.stat().st_size})
    counts = {}
    for split, expected in [('train', 1464), ('val', 1449)]:
        ids = (voc / f'ImageSets/Segmentation/{split}.txt').read_text().split()
        if len(ids) != expected or len(ids) != len(set(ids)):
            raise ValueError(f'Unexpected {split} split size or duplicate image IDs.')
        for item in ids:
            if Path(item).name != item:
                raise ValueError('Invalid VOC image ID.')
            for directory, suffix in [('JPEGImages', '.jpg'), ('SegmentationClass', '.png')]:
                if not (voc / directory / (item + suffix)).is_file():
                    raise FileNotFoundError(item + suffix)
        counts[split] = len(ids)
    manifest = json.loads(marker.read_text())
    manifest.update({'root': str(voc.resolve()), 'split_counts': counts,
                     'source_metadata': 'torchvision.datasets.voc.DATASET_YEAR_DICT[2012]',
                     'scope': 'Original VOC2012 segmentation splits; no augmented labels.'})
    save_json(artifacts / 'dataset_manifest.json', manifest)
    print(json.dumps(manifest, indent=2), flush=True)
    return voc


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('operation', choices=['tokenizer', 'voc'])
    p.add_argument('path')
    p.add_argument('artifacts')
    args = p.parse_args()
    if args.operation == 'tokenizer':
        ensure_tokenizer(args.path, args.artifacts)
    else:
        prepare_voc(args.path, args.artifacts)
