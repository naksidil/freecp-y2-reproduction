"""Fixed two-condition supplementary benchmark; ground truth joined only after inference."""
import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from stage2_helpers import digest,save_json,UPSTREAM_SHA
from protocol9 import protocol_hash,CONDITIONS,validate_manifest
from protocol7 import rescue_indices

def main(repo,config,output):
    repo,config,output=map(lambda p:Path(p).resolve(),[repo,config,output]);output.mkdir(parents=True,exist_ok=True)
    report=dict(ok=False,per_image=[],fresh_holdout=False,confirmation300_rerun=False,benchmark_evaluated=False,y2_segmentation_evaluated=False,
        ground_truth_used_for_decisions=False,timing_note='One warm-up excluded. test_step includes preprocessing and audit transfers; not GPU-only time.')
    matrices=[];rp=[];prob=[];cos=[];counts=[];window_images=[];image_counts=[];image_keep=[];np=None
    started=time.perf_counter()
    try:
        if subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()!=UPSTREAM_SHA:raise RuntimeError('Wrong upstream commit.')
        if subprocess.call(['git','-C',str(repo),'diff','--quiet','HEAD','--']):raise RuntimeError('Tracked upstream source modified.')
        sys.path.insert(0,str(repo));os.chdir(repo)
        import numpy as np
        import torch
        import torch.nn.functional as F
        from PIL import Image
        from mmengine.config import Config
        from mmengine.hooks import Hook
        from mmengine.runner import Runner
        import freecp_segmentor,custom_datasets,y2_segmentor
        from metrics import confusion_matrix,summarize_confusion
        cfg=Config.fromfile(str(config));m=json.loads(config.with_suffix('.manifest.json').read_text());report['manifest']=m
        validate_manifest(m,cfg)
        if digest(cfg.test_dataloader.dataset.ann_file)!=m['split_sha256']:raise RuntimeError('Split hash mismatch.')
        ids=Path(cfg.test_dataloader.dataset.ann_file).read_text().split()
        if len(ids)!=m['n_images'] or len(ids)!=len(set(ids)):raise RuntimeError('Invalid image IDs.')
        official=m['model_kind']=='official'
        if cfg.model.model_type!='SCLIP' or cfg.model.rp_thres!=.15 or cfg.model.ap_thres!=.8 or digest(cfg.model.name_path)!=m['vocabulary_sha256']:raise RuntimeError('Model differs.')
        if not official and list(cfg.model.extra_queries)!=m['extra_queries']:raise RuntimeError('Extra vocabulary differs.')
        cfg.work_dir=str(output/'mmengine');cfg.launcher='none';cfg.randomness=dict(seed=511007,deterministic=False)
        cfg.dump(str(output/'resolved_config.py'));runner=Runner.from_cfg(cfg);model=runner.model;model.eval()
        report.update(gpu=torch.cuda.get_device_name(0),n_output_classes=model.num_classes,n_text_queries=model.num_queries)
        if (model.num_classes!=m['n_output_classes'] or model.num_queries!=m['n_text_queries']):raise RuntimeError('Vocabulary dimension differs.')
        figures=output/'figures';figures.mkdir(exist_ok=True)
        report['benchmark_evaluated']=m['split_role']=='full_val1449'
        report['y2_segmentation_evaluated']=m['condition']['method']=='y2'
        report['random_rescue_evaluated']=m['condition']['method']=='random'
        warm=next(iter(runner.test_dataloader))
        with torch.no_grad():model.test_step(warm)
        torch.cuda.synchronize();del warm
        (output/'per_image.jsonl').write_text('')
        class Recorder(Hook):
            def before_test_iter(self,runner,batch_idx,data_batch=None):
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();self.t=time.perf_counter()
            def after_test_iter(self,runner,batch_idx,data_batch=None,outputs=None):
                torch.cuda.synchronize();elapsed=time.perf_counter()-self.t;peak=torch.cuda.max_memory_allocated()/1024**3
                for sample in outputs:
                    image_path=Path(sample.metainfo['img_path']);image_id=image_path.stem
                    pred=sample.pred_sem_seg.data.squeeze(0).detach().cpu().numpy()
                    raw=(sample.pred_sem_seg if official else sample.raw_pred_sem_seg).data.squeeze(0).detach().cpu().numpy()
                    if pred.min()<0 or pred.max()>20 or not torch.isfinite(sample.seg_logits.data).all():raise RuntimeError('Invalid prediction.')
                    if not np.array_equal(pred,np.where(raw<=20,raw,0)):raise RuntimeError('Incorrect extended class mapping.')
                    gtpath=Path(m['data_root'])/'SegmentationClass'/(image_id+'.png');gt=np.array(Image.open(gtpath))
                    cm=confusion_matrix(gt,pred.astype(np.int64));matrices.append(cm)
                    Image.fromarray(pred.astype('uint8')).save(figures/(image_id+'_prediction.png'))
                    Image.fromarray(raw.astype('uint8')).save(figures/(image_id+'_raw.png'))
                    audit=sample.metainfo.get('rp_audit',[]);imgidx=len(report['per_image'])
                    if not official:
                        if not audit:raise RuntimeError('Missing crop audit.')
                        hw=sample.metainfo['input_hw']
                        # EXACT nearest-neighbor geometry used by torch on the model input grid.
                        resized=F.interpolate(torch.from_numpy(gt.astype('float32'))[None,None],size=hw,mode='nearest')[0,0].numpy().astype('int64')
                        union=set()
                        for crop_index,row in enumerate(audit,1):
                            if len(row['scores'])!=model.num_classes-1:raise RuntimeError('Missing query scores.')
                            import hashlib
                            seed=int(hashlib.sha256((image_id+':'+str(crop_index)).encode()).hexdigest()[:8],16)
                            scores=np.asarray(row['scores'],dtype={'torch.float16':np.float16,'torch.float32':np.float32,'torch.float64':np.float64}[row['score_dtype']]); ps=np.asarray(row['cls_probabilities'],dtype=np.float32)
                            base_mask=np.zeros(len(scores),dtype=bool)
                            base_mask[np.asarray(row['original_rp_kept_class_ids'],dtype=int)-1]=True
                            method=m['condition']['method']
                            if method=='baseline':
                                expected_keep=np.flatnonzero(base_mask); expected_rescue=np.array([],dtype=int)
                            else:
                                expected_keep,expected_rescue=rescue_indices(scores,ps,.5,method,seed,base_mask=base_mask)
                            if (row['kept_class_ids']!=(expected_keep+1).tolist()
                                or row['rescued_class_ids']!=(expected_rescue+1).tolist()):
                                raise RuntimeError('Recorded rescue decisions disagree with fixed rule.')
                            if not set(row['ap_kept_class_ids'])<=set(row['kept_class_ids']):
                                raise RuntimeError('AP introduced an unselected class.')
                            x0,y0,x1,y1=row['box'];pixels=resized[y0:y1,x0:x1]
                            if pixels.size==0:raise RuntimeError('Empty GT crop.')
                            hist=np.bincount(pixels.reshape(-1),minlength=256)
                            row['gt_voc_counts']=hist[1:21].tolist();row['gt_valid_pixels']=int(pixels.size-hist[255])
                            union.update(row['original_rp_kept_class_ids'])
                            # Fully void crops have unknown presence, not 20 negative labels.
                            if row['gt_valid_pixels']>0:
                                rp.append(row['scores'][:20]);prob.append(row['cls_probabilities'][:20]);cos.append(row['cls_cosines'][:20])
                                counts.append(hist[1:21]);window_images.append(imgidx)
                        image_counts.append(cm.sum(1)[1:21]);image_keep.append([c in union for c in range(1,21)])
                    row=dict(image_id=image_id,image_sha256=digest(image_path),gt_sha256=digest(gtpath),
                        pipeline_test_step_seconds=elapsed,peak_allocated_gpu_gib=peak,rp_audit=audit,
                        input_hw=sample.metainfo.get('input_hw'),extra_encoder_forwards_for_cls=0,
                        extra_label_pixels=int((raw>20).sum()),extra_label_pixels_on_voc_foreground=int(((raw>20)&(gt>0)&(gt<=20)).sum()))
                    report['per_image'].append(row)
                    with (output/'per_image.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                    if len(matrices)%10==0:print(f"{m['condition']['id']}: {len(matrices)}/{len(ids)}",flush=True)
        runner.register_hook(Recorder());metrics=runner.test()
        if [x['image_id'] for x in report['per_image']]!=ids:raise RuntimeError('Image order differs.')
        independent=summarize_confusion(np.sum(matrices,axis=0))
        if any(abs(independent[k]-float(metrics[k]))>.011 for k in ['mIoU','mAcc','aAcc']):raise RuntimeError('MMSeg metric mismatch.')
        ckpt=Path.home()/'.cache/clip/ViT-B-16.pt'
        if not ckpt.is_file() or digest(ckpt)!='5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f':raise RuntimeError('Unexpected checkpoint.')
        report.update(ok=True,real_images_evaluated=len(ids),independent_metrics_percent=independent,checkpoint_sha256=digest(ckpt))
    except Exception as exc:
        report['error']=f'{type(exc).__name__}: {exc}';report['traceback']=traceback.format_exc();raise
    finally:
        report['wall_seconds_including_load']=time.perf_counter()-started
        if matrices and np is not None:
            np.savez_compressed(output/'per_image_confusions.npz',image_ids=np.array([r['image_id'] for r in report['per_image']]),confusion_matrices=np.stack(matrices))
        if rp and np is not None:
            np.savez_compressed(output/'presence_arrays.npz',rp_scores=np.asarray(rp,dtype='float32'),
                cls_probabilities=np.asarray(prob,dtype='float32'),cls_cosines=np.asarray(cos,dtype='float32'),
                gt_counts=np.asarray(counts,dtype='int64'),window_image_index=np.asarray(window_images,dtype='int64'),
                image_gt_counts=np.asarray(image_counts,dtype='int64'),image_rp_kept=np.asarray(image_keep,dtype=bool))
        save_json(output/'evaluation_report.json',report)
        if report['ok']:
            members=[output/'evaluation_report.json',output/'per_image.jsonl',output/'per_image_confusions.npz']
            if rp:members.append(output/'presence_arrays.npz')
            members+=sorted((output/'figures').glob('*.png'))
            save_json(output/'result_files_sha256.json',{str(p.relative_to(output)):digest(p) for p in members})
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();main(a.repo,a.config,a.output)
