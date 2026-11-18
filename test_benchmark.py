import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from PIL import Image
from benchmark_metrics import miou,paired_comparison
from protocol7 import rescue_indices,split_digest
from protocol9 import CONDITIONS,TAU,VAL_SHA,protocol_hash,validate_manifest,verify_reviewed_evidence
from analyze9 import analyze,compare_reference
from metrics import confusion_matrix,summarize_confusion
from run_y2_benchmark import select_run,write_config,complete
from stage2_helpers import save_json,digest

class BenchmarkTests(unittest.TestCase):
    def test_reviewed_confirmation_and_reference_are_valid(self):
        r,s=verify_reviewed_evidence(Path(__file__).parent)
        self.assertEqual(s['tau'],.5);self.assertEqual(len(s['results']),6)
        a=np.load(Path(__file__).parent/'reference_val1449_confusions.npz')
        self.assertEqual(split_digest(a['image_ids'].tolist()),VAL_SHA)
        self.assertAlmostEqual(float(miou(a['confusion_matrices'].sum(0))),65.76325321069912)
    def test_preserves_actual_rp_mask_and_inclusive_cls_gate(self):
        keep,rescue=rescue_indices([float(np.float16(.15)),.3,.1],[0,0,.5],.5,base_mask=[False,True,False])
        self.assertEqual(keep.tolist(),[1,2]);self.assertEqual(rescue.tolist(),[2])
    def test_global_metric_not_mean_image_iou(self):
        a=np.array([[[1000,0],[0,0]],[[0,0],[90,10]]])
        self.assertAlmostEqual(float(miou(a.sum(0))),summarize_confusion(a.sum(0))['mIoU'])
        self.assertNotAlmostEqual(float(miou(a.sum(0))),float(miou(a).mean()))
    def test_identical_paired_interval_is_zero(self):
        a=np.array([[[4,1],[2,3]],[[3,0],[1,6]]])
        self.assertEqual(paired_comparison(a,a,20)['paired_bootstrap_95ci'],[0,0])
    def test_rejects_gt_mismatch(self):
        a=np.array([np.eye(2,dtype=int)]*3);b=a.copy();b[0,0,0]+=1
        with self.assertRaises(ValueError):paired_comparison(a,b,10)
    def test_fixed_manifest_rejects_other_tau_and_split(self):
        m=dict(protocol_hash=protocol_hash(),condition=CONDITIONS[1],model_kind='y2',split_role='full_val1449',source_split='val',split_sha256=VAL_SHA,n_images=1449)
        cfg=SimpleNamespace(model=SimpleNamespace(type='Y2FreeCPSegmentation',y2_tau=.5,rescue_mode='y2',extra_queries=[]))
        validate_manifest(m,cfg);cfg.model.y2_tau=.2
        with self.assertRaises(ValueError):validate_manifest(m,cfg)
        cfg.model.y2_tau=.5;m['split_role']='confirmation300'
        with self.assertRaises(ValueError):validate_manifest(m,cfg)
    def test_resume_new_code_gets_new_run(self):
        with tempfile.TemporaryDirectory() as d:
            a,_,_=select_run(Path(d),'a');b,_,resumed=select_run(Path(d),'a');c,_,_=select_run(Path(d),'b')
            self.assertEqual(a,b);self.assertTrue(resumed);self.assertNotEqual(a,c)
    def test_archived_reference_enforces_gt_identity(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'ref.npz';a=np.array([[[4,1],[2,3]],[[3,0],[1,6]]]);ids=np.array(['a','b'])
            np.savez_compressed(p,image_ids=ids,confusion_matrices=a)
            self.assertTrue(compare_reference(a,ids,p)['identical_per_image_confusions'])
            b=a.copy();b[0,0,0]+=1
            with self.assertRaises(ValueError):compare_reference(b,ids,p)
    def test_two_configs_and_missing_result_guard(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'configs').mkdir();(p/'configs/cls_voc21.txt').write_text('vocab')
            split=p/'split.txt';split.write_text('a\nb\nc\n')
            for c in CONDITIONS:
                out=p/(c['id']+'.py');m=write_config(p,p,split,c,out,'full_val1449',dict(extra_queries=[]))
                ns={};exec(out.read_text(),ns)
                self.assertEqual(ns['model']['y2_tau'],None if c['method']=='baseline' else .5)
                self.assertEqual(ns['model']['extra_queries'],[]);self.assertEqual(m['source_split'],'val')
                self.assertFalse(complete(p,m))

class IntegrationTests(unittest.TestCase):
    def test_full_count_two_condition_analysis_and_reference(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'scripts').mkdir();ids=np.array([f'fake_{i:04d}' for i in range(1449)])
            gt=np.array([[0,0],[1,1]]);baseline_cms=None
            for c in CONDITIONS:
                p=root/'conditions'/c['id'];fig=p/'figures';fig.mkdir(parents=True)
                pred=gt.copy()
                if c['method']=='baseline':pred[1,1]=0
                cm=confusion_matrix(gt,pred);cms=np.repeat(cm[None],len(ids),axis=0)
                if c['method']=='baseline':baseline_cms=cms
                rows=[]
                buf=io.BytesIO();Image.fromarray(pred.astype('uint8')).save(buf,format='PNG');png=buf.getvalue()
                for name in ids:
                    for suffix in ['prediction','raw']:(fig/(name+'_'+suffix+'.png')).write_bytes(png)
                    resc=[] if c['method']=='baseline' else [1]
                    w=dict(rescued_class_ids=resc,ap_kept_class_ids=resc,gt_valid_pixels=4,gt_voc_counts=[2]+[0]*19)
                    rows.append(dict(image_id=str(name),image_sha256='same-image',gt_sha256='same-gt',rp_audit=[w],pipeline_test_step_seconds=.1,peak_allocated_gpu_gib=.5))
                np.savez_compressed(p/'per_image_confusions.npz',image_ids=ids,confusion_matrices=cms)
                (p/'per_image.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
                m=dict(protocol_hash=protocol_hash(),condition=c,split_role='full_val1449',split_sha256=VAL_SHA)
                save_json(p/'evaluation_report.json',dict(ok=True,manifest=m,real_images_evaluated=1449,
                    independent_metrics_percent=summarize_confusion(cms.sum(0)),wall_seconds_including_load=10))
                save_json(p/'result_files_sha256.json',{str(q.relative_to(p)):digest(q) for q in p.rglob('*') if q.is_file()})
            ref=root/'scripts/reference_val1449_confusions.npz';np.savez_compressed(ref,image_ids=ids,confusion_matrices=baseline_cms)
            with patch('analyze9.REF_SHA',digest(ref)),redirect_stdout(io.StringIO()):r=analyze(root,draws=10)
            self.assertTrue(r['ok']);self.assertFalse(r['fresh_holdout']);self.assertFalse(r['confirmation300_rerun'])
            self.assertEqual(r['comparison']['changed_images'],1449);self.assertGreater(r['comparison']['delta_mIoU'],0)
            self.assertTrue(r['baseline_replay']['identical_per_image_confusions'])

if __name__=='__main__':unittest.main()
