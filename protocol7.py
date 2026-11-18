"""Predeclared Y2 diagnostics. Only the existing development200 may be opened."""
import hashlib
import json
from pathlib import Path
import numpy as np

VERSION = '07-y2-diagnostic-v1'
SEED = 'YZM5110-round2-20261111-v1'
OLD50_SHA = '4f991c55bd461523a92888da6d3c871ebe9cd0d5a93de3522115ec90f055bad8'
DEV_SHA = 'ffaf04a0466a50db5c0ba7139c4b0bd25b4e590bc70d0da3ecd0c4dd11c2c6bf'
CONF_SHA = 'bb3615e99c763b2d6c233df0358318f09b2466f8f7c3e1804e14739a86837741'
VOC_GIT_SHA = '700e173e6754e4ef0a72a46f9e79145d0f6bdf1e'
ADE_GIT_SHA = 'dba5aa2ce0c703b9dbf07dea5ee5e5230c8b51d3'
TAUS = [0.001,0.002,0.005,0.01,0.02,0.05,0.10,0.20,0.50,1.0]
CONDITIONS = [dict(id='original_vocab', expanded=False),dict(id='expanded_vocab',expanded=True)]
# Frozen before viewing diagnostics. No image/GT is consulted to select vocabulary.
# Exact duplicates of existing 46 queries are excluded separately.
# These aliases/overlapping concepts would otherwise compete with a known VOC label
# while being incorrectly mapped to background. Broad ambiguous concepts are omitted.
SEMANTIC_EXCLUSIONS = {
 'person':'VOC person', 'airplane':'VOC aeroplane', 'boat':'VOC boat/ship',
 'plant':'VOC pottedplant overlap', 'armchair':'VOC chair overlap',
 'seat':'VOC chair/sofa overlap', 'swivelchair':'VOC chair overlap',
 'stool':'VOC chair overlap', 'bench':'VOC chair/sofa overlap',
 'ottoman':'VOC chair/sofa overlap', 'coffeetable':'VOC table overlap',
 'pooltable':'VOC table overlap', 'televisionreceiver':'VOC tvmonitor',
 'monitor':'VOC tvmonitor overlap', 'screen':'VOC tvmonitor overlap',
 'crtscreen':'VOC tvmonitor overlap', 'minibike':'VOC motorbike overlap',
 'animal':'overlaps multiple VOC animals', 'van':'VOC car/bus overlap',
}

def split_digest(ids):
    return hashlib.sha256(('\n'.join(ids)+'\n').encode()).hexdigest()

def make_splits(train,val,old):
    if len(train)!=1464 or len(set(train))!=1464 or len(val)!=1449 or len(set(val))!=1449 or set(train)&set(val):
        raise ValueError('Invalid official train/val splits.')
    if len(old)!=50 or split_digest(old)!=OLD50_SHA or not set(old)<=set(train):
        raise ValueError('Previous train50 differs.')
    if any(Path(i).name!=i or not i.replace('_','').isdigit() for i in train+val):
        raise ValueError('Malformed image ID.')
    ranked=sorted(set(train)-set(old),key=lambda i:(hashlib.sha256((SEED+':'+i).encode()).hexdigest(),i))
    result=dict(development200=ranked[:200],confirmation300=ranked[200:500],pilot3=old[:3])
    if split_digest(result['development200'])!=DEV_SHA or split_digest(result['confirmation300'])!=CONF_SHA:
        raise ValueError('The reserved or development split differs from stage 06.')
    return result

def vocabulary_plan(voc_text,ade_text):
    lines=voc_text.strip().splitlines()
    bg=lines[0].split(', ');fg=lines[1:];ade=ade_text.strip().splitlines()
    if len(bg)!=26 or len(fg)!=20 or len(ade)!=150 or len(set(ade))!=150:
        raise ValueError('Unexpected source vocabulary.')
    existing={n.strip().casefold() for n in bg+fg}
    extras=[];excluded=[]
    for name in ade:
        reason='exact duplicate of existing query' if name.casefold() in existing else SEMANTIC_EXCLUSIONS.get(name)
        if reason:excluded.append(dict(name=name,reason=reason))
        else:extras.append(name)
    return dict(background_queries=bg,voc_queries=fg,extra_queries=extras,excluded=excluded,
        original_output_classes=21,original_text_queries=46,
        expanded_output_classes=21+len(extras),expanded_text_queries=46+len(extras),
        mapping='argmax over all output classes first; indices >20 map to VOC background (0).',
        annotation_limit='Additional concepts may be present in VOC images. Their presence is unknown; do not label them absent.',
        source_ade_git_blob=ADE_GIT_SHA,source_voc_git_blob=VOC_GIT_SHA)

def scientific_protocol():
    return dict(version=VERSION,conditions=CONDITIONS,development_sha256=DEV_SHA,
        confirmation_sha256=CONF_SHA,confirmation_evaluated=False,
        source='VOC2012 train; same 200 development IDs as stage 06, excluding old train50.',
        hypothesis='Existing vanilla-path window CLS scores can complement RP for rescuing locally rejected classes.',
        diagnostic_only=True,y2_segmentation_evaluated=False,
        y2_rule='Within the SAME window, retain class if S>0.15 OR softmax(cls_logits.float())[class]>=tau; pass the union to unchanged AP.',
        model='OpenAI CLIP ViT-B/16; SCLIP; original RP=0.15, AP=0.8, 384 crop/112 stride, resize short side 448.',
        cls_denominator='All 26 original background queries plus all foreground/additional queries; float32 softmax. Never foreground-only softmax.',
        vocabulary='Append unique ADE names after exact-query and declared semantic exclusions. Original 46 embeddings and prompts are preserved.',
        additional_ap_prompts='Use the pinned repository ADE20K Vicuna descriptions; no API, generated prompts or fallback.',
        mapping='Keep extended predictions for audit; map indices>20 to 0 ONLY after full argmax for VOC21 evaluation.',
        presence_labels='Only original 20 VOC classes; window labels from nearest-resized GT after model prediction. Void=255 excluded. Overlapping-window pixel counts are not unique image area.',
        diagnostic_scores=['S','CLS_softmax_p','CLS_cosine','max(S/0.15,p/0.05)'],
        combined_score='Fixed exploratory ranking score, not selected or validated; threshold 1 corresponds to OR with tau=0.05 except strict RP boundary.',
        threshold_grid=TAUS,threshold_table='Offline window presence only, NOT segmentation performance or AP output.',
        proposed_common_tau_rule='Among this fixed grid, require rescue precision>=0.80 separately at BOTH vocab sizes and >=5 true rescued window-class pairs at EACH size. Maximize pooled true rescues; tie choose higher tau. If none, return no proposal. No mIoU used.',
        uncertainty='Paired bootstrap by IMAGE, retaining all its windows; 1000 draws. Descriptive, unadjusted development intervals.',
        no_automatic_go_stop='AUROC +0.02 and area 1-2% are not established decision thresholds. No automatic validation launch.',
        exclusions=SEMANTIC_EXCLUSIONS,
        future='Y1 group-level purification is future work, NOT implemented/tested. Y3, corruption and further second-look tuning are out of scope.')

def protocol_hash():
    return hashlib.sha256(json.dumps(scientific_protocol(),sort_keys=True).encode()).hexdigest()

def slide_boxes(height,width,crop,stride):
    hg=max(height-crop+stride-1,0)//stride+1;wg=max(width-crop+stride-1,0)//stride+1
    return [(max(min(w*stride+crop,width)-crop,0),max(min(h*stride+crop,height)-crop,0),
             min(w*stride+crop,width),min(h*stride+crop,height)) for h in range(hg) for w in range(wg)]

def map_extended_prediction(pred):
    p=np.asarray(pred)
    if not np.issubdtype(p.dtype,np.integer) or np.any(p<0):raise ValueError('Invalid label map.')
    return np.where(p<=20,p,0).astype(np.uint8)

def rescue_indices(scores,probabilities,tau,mode='y2',seed=0,base_mask=None):
    s=np.asarray(scores);p=np.asarray(probabilities)
    if s.shape!=p.shape or s.ndim!=1 or not np.isfinite(s).all() or not np.isfinite(p).all():raise ValueError('Invalid scores.')
    if not 0<=tau<=1 or np.any((p<0)|(p>1)):raise ValueError('Invalid probability/threshold.')
    base=s>0.15 if base_mask is None else np.asarray(base_mask,dtype=bool)
    if base.shape!=s.shape:raise ValueError('Invalid RP decision mask.')
    rescue=(~base)&(p>=tau)
    if mode=='random':
        pool=np.flatnonzero(~base);chosen=np.random.RandomState(seed).choice(pool,size=int(rescue.sum()),replace=False)
        rescue[:]=False;rescue[chosen]=True
    elif mode!='y2':raise ValueError('Unknown rescue mode.')
    return np.flatnonzero(base|rescue),np.flatnonzero(rescue)
