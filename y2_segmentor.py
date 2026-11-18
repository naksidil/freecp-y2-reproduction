"""External instrumentation and optional Y2 gate; pinned FreeCP remains unedited."""
import hashlib
import numpy as np
import torch
import torch.nn.functional as F
from mmseg.registry import MODELS
from mmengine.structures import PixelData
from freecp_segmentor import FreeCPSegmentation, ADE20K, openai_imagenet_template
from protocol7 import slide_boxes,rescue_indices

@MODELS.register_module()
class Y2FreeCPSegmentation(FreeCPSegmentation):
    def __init__(self,extra_queries=None,y2_tau=None,rescue_mode='y2',**kwargs):
        self.extra_queries=list(extra_queries or [])
        self.y2_tau=y2_tau;self.rescue_mode=rescue_mode
        super().__init__(**kwargs)
        if (self.model_type!='SCLIP' or self.clip_type!='CLIP' or self.vit_type!='ViT-B/16' or self.dataset!='voc'
            or self.num_bkg_prompts!=26 or not self.with_bkg or self.ignore_bkg or self.slide_crop!=384
            or self.slide_stride!=112 or self.rp_thres!=.15 or self.ap_thres!=.8):
            raise ValueError('Y2 requires the fixed SCLIP/VOC configuration.')

    def create_vocabulary(self,name_path,dataset,device):
        # Original embeddings are computed exactly by the pinned implementation.
        super().create_vocabulary(name_path,dataset,device)
        if self.num_queries!=46 or self.num_classes!=21:raise ValueError('Expected original VOC vocabulary.')
        if not self.extra_queries:return
        initial=[];descriptive=[]
        with torch.no_grad():
            for name in self.extra_queries:
                if name not in ADE20K or not ADE20K[name]:raise ValueError('Missing pinned ADE descriptions: '+name)
                for output,prompts in [(initial,[t(name) for t in openai_imagenet_template]),(descriptive,ADE20K[name])]:
                    feature=self.net.encode_text(self.tokenizer(prompts).to(device))
                    feature/=feature.norm(dim=-1,keepdim=True)
                    feature=feature.mean(dim=0);feature/=feature.norm()
                    output.append(feature.unsqueeze(0))
        n=len(initial)
        self.query_features=torch.cat([self.query_features]+initial,dim=0)
        self.query_featuresLLM=torch.cat([self.query_featuresLLM]+descriptive,dim=0)
        self.query_idx=torch.cat([self.query_idx,torch.arange(21,21+n,device=device,dtype=torch.int64)])
        self.num_queries+=n;self.num_classes+=n

    @torch.no_grad()
    def predict(self,inputs,data_samples):
        if len(data_samples)!=1 or inputs.shape[0]!=1:raise ValueError('Batch size must be one.')
        self.rp_audit=[];self.image_id=data_samples[0].metainfo['img_path'].split('/')[-1].split('.')[0]
        self.input_hw=list(inputs.shape[-2:])
        labels=torch.arange(self.num_classes-1,device=inputs.device)
        logits,_=self.forward_slide(inputs,labels,[s.metainfo for s in data_samples],self.slide_stride,self.slide_crop)
        result=super().postprocess_result(logits,None,data_samples,labels)
        raw=result[0].pred_sem_seg.data.clone()
        mapped=torch.where(raw<=20,raw,torch.zeros_like(raw))
        result[0].set_data({'raw_pred_sem_seg':PixelData(data=raw),'pred_sem_seg':PixelData(data=mapped)})
        result[0].set_metainfo(dict(rp_audit=self.rp_audit,input_hw=self.input_hw,
            n_output_classes=self.num_classes,n_text_queries=self.num_queries,
            y2_tau=self.y2_tau,extra_encoder_forwards_for_cls=0))
        return result

    def forward_slide(self,img,seg_label,img_metas,stride=112,crop_size=224):
        self._boxes=slide_boxes(img.shape[-2],img.shape[-1],crop_size,stride);self._crop_counter=0
        result=super().forward_slide(img,seg_label,img_metas,stride,crop_size)
        if self._crop_counter!=len(self._boxes):raise RuntimeError('Crop geometry differs.')
        return result

    def forward_feature(self,img,cls_label=None,logit_size=None):
        # Same operation order as pinned FreeCP forward_feature for SCLIP/nonempty labels.
        self._active_box=self._boxes[self._crop_counter];self._crop_counter+=1
        if isinstance(img,list):img=img[0]
        if cls_label is None or len(cls_label)!=self.num_classes-1:raise ValueError('Full vocabulary required.')
        w,h=img[0].shape[-2]//self.patch_size[0],img[0].shape[-1]//self.patch_size[1]
        logits,cls_logits,attn_weight_list,_=self.net.cam(img,self.query_features,self.model_type)
        selected=torch.cat([cls_logits[:,:,:self.num_bkg_prompts],cls_logits[:,:,self.num_bkg_prompts+cls_label]],dim=-1)
        self._cls_p=selected.float().softmax(dim=-1).reshape(-1)[self.num_bkg_prompts:]
        self._cls_cos=(selected.float()/self.net.logit_scale.exp().float()).reshape(-1)[self.num_bkg_prompts:]
        logits=torch.cat([logits[:,:,:self.num_bkg_prompts],logits[:,:,self.num_bkg_prompts+cls_label]],dim=-1)
        logits=logits.softmax(dim=-1)
        out_dim=logits.shape[-1];logits=logits.permute(0,2,1).reshape(-1,out_dim,w,h)
        normalized_logits=(logits-logits.min())/(logits.max()-logits.min())
        logits_ref=self.refinement(img,normalized_logits,attn_weight_list,cls_label)
        logits_ref,candidate_classes=self.redundancy_purification(logits,logits_ref)
        if len(candidate_classes)>1:logits_ref,candidate_classes=self.ambiguity_purification(logits,logits_ref,candidate_classes,img)
        self.rp_audit[-1]['ap_kept_class_ids']=(candidate_classes+1).detach().cpu().tolist()
        size=img.shape[-2:] if logit_size is None else logit_size
        return F.interpolate(logits_ref,size=size,mode='bilinear'),cls_logits,None,candidate_classes

    def redundancy_purification(self,logits,logits_ref):
        x=logits[0,self.num_bkg_prompts:];y=logits_ref[0,self.num_bkg_prompts:]
        s=(x*y).sum((-1,-2))/(x.sum((-1,-2))+y.sum((-1,-2))+1e-8)
        if not torch.isfinite(s).all() or not torch.isfinite(self._cls_p).all():raise RuntimeError('Nonfinite scores.')
        base=torch.where(s>self.rp_thres)[0];rescued=[]
        if self.y2_tau is None:
            purified,kept=super().redundancy_purification(logits,logits_ref)
        else:
            seed=int(hashlib.sha256((self.image_id+':'+str(self._crop_counter)).encode()).hexdigest()[:8],16)
            ids,rescue=rescue_indices(s.detach().cpu().numpy(),self._cls_p.detach().cpu().numpy(),self.y2_tau,self.rescue_mode,seed,base_mask=(s>self.rp_thres).detach().cpu().numpy())
            kept=torch.as_tensor(ids,device=s.device,dtype=torch.long);rescued=(rescue+1).tolist()
            # Union is decided BEFORE zeroing and is passed to the unchanged AP.
            purified=logits_ref
            selected=set(ids.tolist())
            for c in range(len(s)):
                if c not in selected:purified[:,self.num_bkg_prompts+c]*=0.0
        self.rp_audit.append(dict(box=list(self._active_box),score_dtype=str(s.dtype),scores=s.detach().float().cpu().tolist(),
            cls_probabilities=self._cls_p.detach().cpu().tolist(),cls_cosines=self._cls_cos.detach().cpu().tolist(),
            original_rp_kept_class_ids=(base+1).detach().cpu().tolist(),
            kept_class_ids=(kept+1).detach().cpu().tolist(),rescued_class_ids=rescued))
        return purified,kept
