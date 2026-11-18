"""Mandatory tests inside the existing Colab environment; no model download."""
import argparse
import json
import sys
import unittest
from pathlib import Path

def main(repo,manifest):
    sys.path.insert(0,str(Path(repo).resolve()))
    import numpy as np
    import torch
    from freecp_segmentor import FreeCPSegmentation,ADE20K
    from y2_segmentor import Y2FreeCPSegmentation
    from protocol7 import vocabulary_plan
    plan=json.loads(Path(manifest).read_text())
    missing=[s for s in plan['extra_queries'] if s not in ADE20K or not ADE20K[s]]
    if missing:raise RuntimeError('Missing ADE prompts: '+repr(missing))

    class FakeNet(torch.nn.Module):
        def __init__(self):super().__init__();self.logit_scale=torch.tensor(100.).log();self.calls=0
        def cam(self,img,features,kind):
            self.calls+=1
            x=torch.zeros(1,4,46);x[:,:,26]=3.;x[:,:,27]=1.
            cls=torch.zeros(1,1,46);cls[:,:,27]=8.
            return x,cls,None,None

    def make(kind=Y2FreeCPSegmentation,tau=None):
        m=kind.__new__(kind);torch.nn.Module.__init__(m)
        m.net=FakeNet();m.model_type='SCLIP';m.num_queries=46;m.num_classes=21;m.num_bkg_prompts=26;m.rp_thres=.15
        m.query_features=torch.zeros(46,2);m.patch_size=(1,1);m._boxes=[(0,0,2,2)];m._crop_counter=0
        m.y2_tau=tau;m.rescue_mode='y2';m.image_id='synthetic';m.rp_audit=[];m.ap_seen=None
        m.refinement=lambda *args:torch.ones(1,46,2,2)
        def ap(x,y,keep,img):m.ap_seen=keep.tolist();return y,keep
        m.ambiguity_purification=ap
        return m

    class ModelTests(unittest.TestCase):
        def test_instrumentation_preserves_pinned_forward_exactly(self):
            a=make(FreeCPSegmentation);b=make();image=torch.ones(1,3,2,2);labels=torch.arange(20)
            x=a.forward_feature(image,labels);y=b.forward_feature(image,labels)
            torch.testing.assert_close(x[0],y[0],rtol=0,atol=0);torch.testing.assert_close(x[3],y[3],rtol=0,atol=0)
            self.assertEqual(b.net.calls,1);self.assertEqual(len(b.rp_audit),1)
        def test_same_window_cls_rescues_before_ap(self):
            m=make(tau=.5);out=m.forward_feature(torch.ones(1,3,2,2),torch.arange(20))
            self.assertEqual(m.rp_audit[0]['rescued_class_ids'],[2]);self.assertIn(1,m.ap_seen)
            self.assertGreater(float(out[0][0,27].sum()),0);self.assertEqual(m.net.calls,1)
        def test_softmax_includes_background_queries(self):
            m=make();m.forward_feature(torch.ones(1,3,2,2),torch.arange(20))
            vals=m.rp_audit[0]['cls_probabilities'];self.assertLess(sum(vals),1)
            expected=float(torch.softmax(torch.tensor([0.]*27+[8.]+[0.]*18),dim=0)[27])
            self.assertAlmostEqual(vals[1],expected,places=6)
        def test_rejected_maps_are_zero(self):
            m=make(tau=1.);out=m.forward_feature(torch.ones(1,3,2,2),torch.arange(20))
            self.assertEqual(m.rp_audit[0]['rescued_class_ids'],[]);self.assertEqual(float(out[0][0,27].sum()),0)
        def test_random_has_same_pre_ap_budget_as_y2(self):
            a=make(tau=.5);b=make(tau=.5);b.rescue_mode='random'
            a.forward_feature(torch.ones(1,3,2,2),torch.arange(20))
            b.forward_feature(torch.ones(1,3,2,2),torch.arange(20))
            self.assertEqual(len(a.rp_audit[0]['rescued_class_ids']),len(b.rp_audit[0]['rescued_class_ids']))
            self.assertEqual(a.rp_audit[0]['original_rp_kept_class_ids'],b.rp_audit[0]['original_rp_kept_class_ids'])
            self.assertEqual(len(a.ap_seen),len(b.ap_seen));self.assertEqual(b.net.calls,1)
        def test_no_rescue_preserves_half_precision_rp_maps(self):
            a=make(FreeCPSegmentation);b=make(tau=1.)
            logits=torch.full((1,46,2,2),.1,dtype=torch.float16)
            logits[:,26]=.1765;logits[:,27]=.3
            ref=torch.ones_like(logits);b._cls_p=torch.zeros(20);b._cls_cos=torch.zeros(20)
            b._active_box=(0,0,2,2)
            x,k=a.redundancy_purification(logits,ref.clone())
            y,l=b.redundancy_purification(logits,ref.clone())
            torch.testing.assert_close(x,y,rtol=0,atol=0);torch.testing.assert_close(k,l,rtol=0,atol=0)
        def test_gt_void_and_window_presence_geometry(self):
            import torch.nn.functional as F
            gt=torch.tensor([[1.,255.],[0.,2.]])[None,None]
            resized=F.interpolate(gt,size=(4,4),mode='nearest')[0,0].numpy().astype('int64')
            left=np.bincount(resized[:,:2].reshape(-1),minlength=256)
            self.assertEqual(int(left[1]),4);self.assertEqual(int(left[2]),0);self.assertEqual(int((resized==255).sum()),4)

    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ModelTests))
    if not result.wasSuccessful():raise SystemExit(1)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--vocabulary-manifest',required=True)
    a=p.parse_args();main(a.repo,a.vocabulary_manifest)
