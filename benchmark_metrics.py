"""Unchanged verified aggregate and paired-image metrics from stage08."""
import numpy as np

def miou(cm):
    """mIoU from total confusion, in percent, with absent classes excluded."""
    cm=np.asarray(cm,dtype=np.float64)
    inter=np.diagonal(cm,axis1=-2,axis2=-1)
    union=cm.sum(-2)+cm.sum(-1)-inter
    iou=np.divide(inter,union,out=np.full_like(inter,np.nan),where=union>0)
    return np.nanmean(iou,axis=-1)*100

def paired_comparison(a,b,draws=5000):
    a=np.asarray(a);b=np.asarray(b)
    if a.shape!=b.shape or a.ndim!=3 or len(a)<2 or not np.array_equal(a.sum(2),b.sum(2)):
        raise ValueError('Unpaired predictions/ground truth.')
    rng=np.random.RandomState(51100813);ds=[];n=len(a);total_a=a.sum(0);total_b=b.sum(0)
    for _ in range(draws):
        ix=rng.randint(n,size=n);ds.append(float(miou(a[ix].sum(0))-miou(b[ix].sum(0))))
    loo=miou(total_a[None]-a)-miou(total_b[None]-b)
    delta=float(miou(total_a)-miou(total_b));j=int(np.argmax(np.abs(loo-delta)))
    return dict(delta_mIoU=delta,paired_bootstrap_95ci=np.percentile(ds,[2.5,97.5]).tolist(),
        bootstrap_draws=draws,leave_one_out_min=float(loo.min()),leave_one_out_max=float(loo.max()),
        most_influential_image_index=j,delta_without_most_influential_image=float(loo[j]))

def rescue_summary(rows):
    out=dict(rescued_window_class_pairs=0,voc_true_pairs=0,voc_false_pairs=0,
             extra_pairs_presence_unknown=0,images_with_rescue=0,rescued_pairs_surviving_ap=0,
             voc_true_pairs_surviving_ap=0,voc_false_pairs_surviving_ap=0,
             extra_pairs_surviving_ap_presence_unknown=0)
    for r in rows:
        touched=False
        for w in r['rp_audit']:
            ap=set(w['ap_kept_class_ids'])
            for c in w['rescued_class_ids']:
                out['rescued_window_class_pairs']+=1;touched=True
                survival=c in ap
                out['rescued_pairs_surviving_ap']+=int(survival)
                if c>20:
                    out['extra_pairs_presence_unknown']+=1
                    out['extra_pairs_surviving_ap_presence_unknown']+=int(survival)
                elif w['gt_valid_pixels']>0:
                    key='voc_true_pairs' if w['gt_voc_counts'][c-1]>0 else 'voc_false_pairs'
                    out[key]+=1;out[key+'_surviving_ap']+=int(survival)
        out['images_with_rescue']+=int(touched)
    denom=out['voc_true_pairs']+out['voc_false_pairs']
    out['voc_rescue_precision']=out['voc_true_pairs']/denom if denom else None
    out['note']='Overlapping windows are not independent; presence is unknown for added labels and fully void crops.'
    return out
