"""Final K=21 benchmark with fixed Y2; VOC val is previously seen, not a new holdout."""
import hashlib
import json
from pathlib import Path
from protocol7 import DEV_SHA,CONF_SHA

TAU=.5
VAL_SHA='cdc1326d12f69ce5153aa5da04a4d8783e146868d42d19f82f44e74b97ac907d'
REF_SHA='9a7122c3f68eba5cb2e2287b0872cb9c022973271af38883dedfb56cd885acca'
CONDITIONS=[dict(id='original_baseline',expanded=False,method='baseline'),dict(id='original_y2',expanded=False,method='y2')]

def scientific_protocol():
    return dict(version='09-y2-final-benchmark-v1',conditions=CONDITIONS,tau=TAU,
        development_sha256=DEV_SHA,confirmation_sha256=CONF_SHA,full_val_sha256=VAL_SHA,
        source='Official VOC2012 val1449, already inspected in earlier RP/background experiments.',
        role='Supplementary benchmark under a fixed method, NOT a new untouched confirmatory test.',
        preceding_confirmation='Stage08 run 20261112T231415_254311Z, all 6 conditions completed. Original +0.1577 and expanded +0.2894 mIoU points; both CIs cross zero; expanded gain is driven by one image.',
        selection='Keep the SAME tau=0.5 chosen from stage07 development presence; no change based on stage08.',
        contrast='Y2 minus baseline, K=21/Q=46 only. No random/expanded sweeps or threshold search.',
        rule='Same-window original tensor RP decision OR CLS float32 softmax p>=0.5, BEFORE zeroing; unchanged AP follows.',
        model='OpenAI CLIP ViT-B/16, SCLIP, RP=0.15, AP=0.8, crop384/stride112; same as stage08.',
        probability='All original 26 background and 20 foreground queries in the denominator.',
        GT='Never used to decide predictions. Joined only after inference.',
        metric='Dataset mIoU from summed per-class intersections/unions; void255 ignored.',
        uncertainty='Paired image bootstrap, 5000 draws, seed51100813; descriptive interval on a previously seen benchmark.',
        sensitivity='Leave-one-image-out, affected images/pixels, per-class effects, rescue/AP counts.',
        timing='One warm-up excluded; pipeline time includes audit transfers, not GPU-only time. Single sequential runs do not establish a speedup.',
        reference='Replay original baseline and compare its per-image confusion matrices to the archived 65.76325321069912 result.',
        reference_sha256=REF_SHA,
        no_retuning=True,confirmation300_rerun=False,no_other_experiments=True,
        result_use='Report alongside the existing 300-image confirmation; do not replace or hide its uncertainty.')

def protocol_hash():
    return hashlib.sha256(json.dumps(scientific_protocol(),sort_keys=True).encode()).hexdigest()

def verify_reviewed_evidence(folder):
    folder=Path(folder);r=json.loads((folder/'reviewed_stage8_run.json').read_text());s=json.loads((folder/'reviewed_stage8_summary.json').read_text())
    if (r['run_id']!='20261112T231415_254311Z' or not r['ok'] or len(r['completed_conditions'])!=6
        or not r['confirmation_evaluated'] or not s['confirmation_evaluated'] or s['tau']!=TAU or not s['random_budget_verified']):
        raise ValueError('Reviewed stage08 result differs.')
    if hashlib.sha256((folder/'reference_val1449_confusions.npz').read_bytes()).hexdigest()!=REF_SHA:
        raise ValueError('Archived baseline reference differs.')
    return r,s

def validate_manifest(m,cfg):
    if m['protocol_hash']!=protocol_hash() or m['condition'] not in CONDITIONS:raise ValueError('Unrecognized benchmark protocol.')
    official=m['model_kind']=='official';method=m['condition']['method']
    if m['split_role']=='full_val1449':
        if m['split_sha256']!=VAL_SHA or m['n_images']!=1449 or m['source_split']!='val' or official:raise ValueError('Invalid val split.')
    elif m['split_role']=='pilot3':
        if m['n_images']!=3 or method!='baseline' or m['source_split']!='train':raise ValueError('Invalid pilot.')
    else:raise ValueError('Only pilot3 and full_val1449 are allowed.')
    if official:
        if cfg.model.type!='FreeCPSegmentation':raise ValueError('Wrong official model.')
    elif (cfg.model.type!='Y2FreeCPSegmentation' or cfg.model.y2_tau!=(None if method=='baseline' else TAU)
          or cfg.model.rescue_mode!='y2' or list(cfg.model.extra_queries)!=[]):raise ValueError('Fixed Y2 settings changed.')
