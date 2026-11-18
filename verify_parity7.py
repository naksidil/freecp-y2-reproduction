import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image
from stage2_helpers import save_json

def parity(a,b,out):
    a,b=Path(a),Path(b)
    aa=np.load(a/'per_image_confusions.npz');bb=np.load(b/'per_image_confusions.npz')
    if not np.array_equal(aa['image_ids'],bb['image_ids']) or not np.array_equal(aa['confusion_matrices'],bb['confusion_matrices']):raise RuntimeError('Confusion parity failed.')
    for i in aa['image_ids']:
        name=str(i)+'_prediction.png'
        if not np.array_equal(np.array(Image.open(a/'figures'/name)),np.array(Image.open(b/'figures'/name))):raise RuntimeError('Pixel parity failed.')
    save_json(out,dict(ok=True,images=len(aa['image_ids']),all_pixels_identical=True))

if __name__=='__main__':parity(*sys.argv[1:])
