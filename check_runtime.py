"""Check actual old-runtime CUDA kernels and imports; no model weights needed."""
import importlib.metadata
import json
import platform
import subprocess
import sys
import traceback
from pathlib import Path
from stage2_helpers import UPSTREAM_SHA, save_json


def check(repo, output):
    report = {'ok': False, 'python': platform.python_version(), 'checks': {}, 'packages': {}}
    try:
        if sys.version_info[:2] != (3, 10):
            raise RuntimeError('This isolated recipe requires Python 3.10.')
        report['commit'] = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
        if report['commit'] != UPSTREAM_SHA:
            raise RuntimeError('Unexpected FreeCP source revision.')
        for name in ['torch', 'torchvision', 'numpy', 'mmcv', 'mmengine', 'mmsegmentation']:
            report['packages'][name] = importlib.metadata.version(name)
        import torch
        report['cuda_available'] = torch.cuda.is_available()
        report['torch_cuda_build'] = torch.version.cuda
        if not report['cuda_available']:
            raise RuntimeError('CUDA is unavailable in the isolated environment.')
        report['gpu'] = torch.cuda.get_device_name(0)
        report['gpu_memory_gib'] = torch.cuda.get_device_properties(0).total_memory / 1024**3
        x = torch.ones((32, 32), device='cuda', dtype=torch.float16)
        if (x @ x).mean().item() != 32.0:
            raise RuntimeError('CUDA FP16 matrix check failed.')
        report['checks']['cuda_fp16_matmul'] = True
        import torchvision
        boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 9., 9.]], device='cuda')
        scores = torch.tensor([.9, .8], device='cuda')
        if torchvision.ops.nms(boxes, scores, .5).tolist() != [0]:
            raise RuntimeError('torchvision CUDA extension check failed.')
        report['checks']['torchvision_cuda_nms'] = True
        from mmcv.ops import nms
        _, indices = nms(boxes, scores, .5)
        if indices.tolist() != [0]:
            raise RuntimeError('MMCV CUDA extension check failed.')
        torch.cuda.synchronize()
        report['checks']['mmcv_cuda_nms'] = True
        sys.path.insert(0, str(Path(repo).resolve()))
        import freecp_segmentor
        import custom_datasets
        from open_clip import tokenizer
        tokens = tokenizer.tokenize(['a photo of a dog'])
        report['checks']['freecp_import'] = True
        report['checks']['tokenizer_shape'] = list(tokens.shape)
        if list(tokens.shape) != [1, 77]:
            raise RuntimeError('Unexpected tokenizer output.')
        report['ok'] = True
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        report['traceback'] = traceback.format_exc()
        raise
    finally:
        save_json(output, report)
        print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    check(Path(sys.argv[1]), Path(sys.argv[2]))
