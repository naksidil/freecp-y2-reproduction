"""Descriptive benchmark analysis; preserves the stage08 confirmation conclusion."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from benchmark_metrics import miou,paired_comparison,rescue_summary
from metrics import summarize_confusion
from stage2_helpers import save_json,digest
from protocol9 import CONDITIONS,protocol_hash,TAU,VAL_SHA,REF_SHA

def compare_reference(cms,ids,reference):
    with np.load(reference) as a:
        if not np.array_equal(ids,a['image_ids']) or not np.array_equal(cms.sum(2),a['confusion_matrices'].sum(2)):
            raise ValueError('Val images or GT differ from archived baseline.')
        ref=a['confusion_matrices'];changed=np.any(cms!=ref,axis=(1,2))
        return dict(archived_mIoU=float(miou(ref.sum(0))),replayed_mIoU=float(miou(cms.sum(0))),
            delta_mIoU=float(miou(cms.sum(0))-miou(ref.sum(0))),
            identical_per_image_confusions=bool(not changed.any()),different_confusion_matrices=int(changed.sum()),
            note='Confusion-matrix agreement, not a byte or pixel comparison of archived masks.')

def analyze(folder,draws=5000):
    folder=Path(folder);arrays={};stats=[];ids=None;gt=None;identity=None
    for c in CONDITIONS:
        name=c['id'];p=folder/'conditions'/name;r=json.loads((p/'evaluation_report.json').read_text())
        m=r['manifest']
        if (not r['ok'] or m['protocol_hash']!=protocol_hash() or m['condition']!=c or m['split_role']!='full_val1449'
            or m['split_sha256']!=VAL_SHA or r['real_images_evaluated']!=1449):raise ValueError('Incomplete or invalid benchmark condition.')
        hashes=json.loads((p/'result_files_sha256.json').read_text())
        if any(digest(p/n)!=h for n,h in hashes.items()):raise ValueError('Result integrity mismatch.')
        with np.load(p/'per_image_confusions.npz') as a:current_ids=a['image_ids'];cms=a['confusion_matrices']
        rows=[json.loads(line) for line in (p/'per_image.jsonl').read_text().splitlines()]
        current=[(x['image_id'],x['image_sha256'],x['gt_sha256']) for x in rows]
        if len(cms)!=1449 or [x[0] for x in current]!=current_ids.tolist():raise ValueError('Incorrect image order/count.')
        if ids is None:ids=current_ids;gt=cms.sum(2);identity=current
        elif not np.array_equal(ids,current_ids) or not np.array_equal(gt,cms.sum(2)) or identity!=current:
            raise ValueError('Unpaired images/GT.')
        arrays[name]=cms;metric=summarize_confusion(cms.sum(0))
        if abs(metric['mIoU']-r['independent_metrics_percent']['mIoU'])>1e-8:raise ValueError('Metric mismatch.')
        times=np.array([x['pipeline_test_step_seconds'] for x in rows])
        stats.append(dict(condition=c,n_images=1449,mIoU=metric['mIoU'],metrics=metric,
            mean_pipeline_seconds=float(times.mean()),median_pipeline_seconds=float(np.median(times)),
            p95_pipeline_seconds=float(np.percentile(times,95)),peak_allocated_gpu_gib=max(x['peak_allocated_gpu_gib'] for x in rows),
            wall_seconds_including_load=r['wall_seconds_including_load'],rescues=rescue_summary(rows)))
    baseline=arrays['original_baseline'];y2=arrays['original_y2']
    reference=folder/'scripts/reference_val1449_confusions.npz'
    if digest(reference)!=REF_SHA:raise ValueError('Baseline reference changed.')
    replay=compare_reference(baseline,ids,reference)
    comparison=paired_comparison(y2,baseline,draws)
    comparison['most_influential_image']=str(ids[comparison.pop('most_influential_image_index')])
    changed=[]
    for name in ids:
        a=np.array(Image.open(folder/'conditions/original_baseline/figures'/(str(name)+'_prediction.png')))
        b=np.array(Image.open(folder/'conditions/original_y2/figures'/(str(name)+'_prediction.png')))
        if a.shape!=b.shape:raise ValueError('Prediction shape mismatch.')
        changed.append(int((a!=b).sum()))
    comparison.update(method='original_y2',control='original_baseline',changed_images=int(np.count_nonzero(changed)),changed_pixels=int(sum(changed)))
    result=dict(ok=True,protocol_hash=protocol_hash(),tau=TAU,split='full_val1449',
        benchmark_evaluated=True,fresh_holdout=False,confirmation300_rerun=False,
        results=stats,comparison=comparison,baseline_replay=replay,
        interpretation='Previously inspected val benchmark, not independent confirmation. Preserve stage08 300-image results and their uncertainty. No retuning.')
    save_json(folder/'y2_benchmark_summary.json',result)
    lo,hi=comparison['paired_bootstrap_95ci']
    lines=['ASAMA 09 — SABIT Y2, VOC VAL1449 BENCHMARK',
        'Bu val kumesi daha once goruldu; yeni/temiz dogrulama olarak sunulamaz.',
        f"Orijinal FreeCP: {stats[0]['mIoU']:.5f} mIoU",f"Y2 (tau=0.5): {stats[1]['mIoU']:.5f} mIoU",
        f"Fark: {comparison['delta_mIoU']:+.5f} puan; %95 esli bootstrap [{lo:+.5f}, {hi:+.5f}]",
        f"Degisen goruntu: {comparison['changed_images']}/1449",
        f"En etkili goruntu cikarilinca fark: {comparison['delta_without_most_influential_image']:+.5f} ({comparison['most_influential_image']})",
        f"Eski baseline ile ayni confusion matrisleri: {replay['identical_per_image_confusions']}",
        '300 dogrulama sonucu raporda ayrica korunacak. Yeni esik aramasi yapilmadi.',
        'Sonuc ZIP dosyasini paylas.']
    (folder/'SONUC_OZETI.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8');print('\n'.join(lines),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args();analyze(a.run_dir)
