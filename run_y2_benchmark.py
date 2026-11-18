"""Fixed two-condition full-val supplementary benchmark. No installation or threshold selection."""
import hashlib
import json
import os
import shutil
import subprocess
import traceback
from datetime import datetime,timezone
from pathlib import Path
from stage2_helpers import save_json,digest,git_blob_sha,UPSTREAM_SHA,BPE_GIT_BLOB_SHA,run_logged
from protocol7 import make_splits,vocabulary_plan,split_digest,VOC_GIT_SHA,ADE_GIT_SHA
from protocol9 import CONDITIONS,scientific_protocol,protocol_hash,verify_reviewed_evidence,TAU,VAL_SHA

def fingerprint(folder):
    h=hashlib.sha256()
    for p in sorted(Path(folder).iterdir()):
        if p.is_file() and p.suffix in {'.py','.json','.txt','.md','.npz'}:h.update(p.name.encode()+b'\0'+p.read_bytes())
    return h.hexdigest()

def select_run(root,code_hash):
    pointer=root/'latest_y2_benchmark_work.json'
    if pointer.is_file():
        old=json.loads(pointer.read_text());folder=Path(old['artifact']).resolve()
        if folder.parent!=(root/'runs').resolve() or not folder.name.startswith('y2benchmark_'):raise ValueError('Invalid resume path.')
        if old.get('package_fingerprint')==code_hash and (folder/'y2_benchmark_run.json').is_file():return folder,json.loads((folder/'y2_benchmark_run.json').read_text()),True
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');folder=root/'runs'/('y2benchmark_'+run_id);folder.mkdir(parents=True)
    state=dict(ok=False,run_id=run_id,package_fingerprint=code_hash,protocol_hash=protocol_hash(),completed_conditions=[],attempts=[],benchmark_evaluated=False)
    save_json(pointer,dict(artifact=str(folder),package_fingerprint=code_hash));save_json(folder/'y2_benchmark_run.json',state)
    return folder,state,False

def write_config(repo,voc,split,condition,dest,role,plan,official=False):
    extras=plan['extra_queries'] if condition['expanded'] else []
    kind='FreeCPSegmentation' if official else 'Y2FreeCPSegmentation'
    tau=None if condition['method']=='baseline' else TAU
    mode='random' if condition['method']=='random' else 'y2'
    options='' if official else f', extra_queries={extras!r}, y2_tau={tau!r}, rescue_mode={mode!r}'
    dest.write_text(f'_base_ = {str(repo/"configs/cfg_voc21.py")!r}\n'
        f'model = dict(type={kind!r}, model_type="SCLIP",clip_type="CLIP",vit_type="ViT-B/16",rp_thres=.15,ap_thres=.8,'
        f'name_path={str(repo/"configs/cls_voc21.txt")!r}{options})\n'
        f'test_dataloader = dict(num_workers=0,persistent_workers=False,dataset=dict(data_root={str(voc)!r},ann_file={str(split)!r}))\n')
    manifest=dict(protocol_hash=protocol_hash(),upstream_commit=UPSTREAM_SHA,condition=condition,source_split='val' if role=='full_val1449' else 'train',split_role=role,
        split_sha256=digest(split),n_images=len(split.read_text().split()),data_root=str(voc),model_kind='official' if official else condition['method'],
        vocabulary_sha256=digest(repo/'configs/cls_voc21.txt'),extra_queries=extras,n_output_classes=21+len(extras),n_text_queries=46+len(extras))
    save_json(dest.with_suffix('.manifest.json'),manifest);return manifest

def complete(out,manifest):
    try:
        r=json.loads((out/'evaluation_report.json').read_text());hashes=json.loads((out/'result_files_sha256.json').read_text())
        n=manifest['n_images'];expected=2*n+(3 if manifest['model_kind']=='official' else 4)
        if len(hashes)!=expected or any(not (out/p).is_file() or digest(out/p)!=h for p,h in hashes.items()):return False
        return r['ok'] is True and r['manifest']==manifest and r['real_images_evaluated']==n
    except (OSError,KeyError,ValueError):return False

def main(root='/content/freecp_t4'):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True);source=Path(__file__).resolve().parent
    code_hash=fingerprint(source);folder,state,resumed=select_run(root,code_hash);state['ok']=False
    state.pop('error',None);state.pop('traceback',None);state['attempts'].append(dict(started_utc=datetime.now(timezone.utc).isoformat(),resumed=resumed));code=1
    try:
        scripts=folder/'scripts'
        if not scripts.exists():
            scripts.mkdir()
            for p in source.iterdir():
                if p.is_file() and p.suffix in {'.py','.json','.txt','.md','.npz'}:shutil.copy2(p,scripts/p.name)
        if fingerprint(scripts)!=code_hash:raise RuntimeError('Resume source differs.')
        verify_reviewed_evidence(scripts)
        python=root/'env_py310/bin/python';repo=root/'FreeCP';voc=root/'data/VOCdevkit/VOC2012'
        for p in [python,repo/'configs/cfg_voc21.py',voc/'ImageSets/Segmentation/train.txt',voc/'ImageSets/Segmentation/val.txt']:
            if not p.is_file():raise FileNotFoundError(f'{p} missing. Run stage 02 setup in this VM first. Stages 03-08 need not be rerun. Reviewed evidence is included in this package.')
        for p,sha in [(repo/'configs/cls_voc21.txt',VOC_GIT_SHA),(repo/'configs/cls_ade20k.txt',ADE_GIT_SHA),(repo/'open_clip/bpe_simple_vocab_16e6.txt.gz',BPE_GIT_BLOB_SHA)]:
            if git_blob_sha(p)!=sha:raise RuntimeError('Pinned asset changed: '+str(p))
        # Downloaded ADE dataset is NOT required; only the repository's text list/prompts.
        plan=vocabulary_plan((repo/'configs/cls_voc21.txt').read_text(),(repo/'configs/cls_ade20k.txt').read_text())
        save_json(folder/'vocabulary_manifest.json',plan);save_json(folder/'protocol.json',scientific_protocol())
        train=(voc/'ImageSets/Segmentation/train.txt').read_text().split();val=(voc/'ImageSets/Segmentation/val.txt').read_text().split()
        splits=make_splits(train,val,(scripts/'previous_train50.txt').read_text().split())
        if split_digest(val)!=VAL_SHA:raise RuntimeError('Official val order/hash differs.')
        splits['full_val1449']=val
        sd=folder/'splits';sd.mkdir(exist_ok=True)
        for name,ids in splits.items():
            p=sd/(name+'.txt');raw='\n'.join(ids)+'\n'
            if p.exists() and p.read_text()!=raw:raise RuntimeError('Saved split changed.')
            p.write_text(raw)
        save_json(folder/'split_manifest.json',{k:dict(n=len(v),sha256=split_digest(v)) for k,v in splits.items()})
        for r in json.loads((scripts/'reference_pilot.json').read_text()):
            for field,sub,suffix in [('image_sha256','JPEGImages','.jpg'),('gt_sha256','SegmentationClass','.png')]:
                if digest(voc/sub/(r['image_id']+suffix))!=r[field]:raise RuntimeError('Pilot data changed.')
        for i in splits['full_val1449']:
            for sub,suffix in [('JPEGImages','.jpg'),('SegmentationClass','.png')]:
                if not (voc/sub/(i+suffix)).is_file():raise FileNotFoundError('Missing val file: '+i)
        env=os.environ.copy();env.update(MPLBACKEND='Agg',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1')
        def run(cmd,label,cwd=None):run_logged(cmd,label,folder,cwd=cwd,env=env)
        run([python,scripts/'check_runtime.py',repo,folder/'runtime_report.json'],'runtime_check',repo)
        freeze=subprocess.check_output([str(python),'-m','pip','freeze'],text=True,env=env)
        if (folder/'pip_freeze.txt').exists() and (folder/'pip_freeze.txt').read_text()!=freeze:raise RuntimeError('Environment changed since resume.')
        (folder/'pip_freeze.txt').write_text(freeze)
        run([python,'-m','unittest','-v','test_metrics','test_benchmark'],'unit_tests',scripts)
        run([python,scripts/'test_model7.py','--repo',repo,'--vocabulary-manifest',folder/'vocabulary_manifest.json'],'model_tests',repo)
        cd=folder/'configs';cd.mkdir(exist_ok=True)
        def evaluate(name,condition,role,official=False):
            cfg=cd/(name+'.py');out=folder/('pilot' if role=='pilot3' else 'conditions')/name
            manifest=write_config(repo,voc,sd/(role+'.txt'),condition,cfg,role,plan,official)
            if complete(out,manifest):print('RESUME: '+name,flush=True)
            else:
                if out.exists():
                    partial=folder/'partial_runs';partial.mkdir(exist_ok=True);out.rename(partial/(name+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')))
                run([python,scripts/'evaluate9.py','--repo',repo,'--config',cfg,'--output',out],name,repo)
            if not complete(out,manifest):raise RuntimeError('Incomplete condition: '+name)
            return out
        print('ON KONTROL: orijinal 3 goruntude piksel esligi.',flush=True)
        original=evaluate('official',CONDITIONS[0],'pilot3',True);instrumented=evaluate('instrumented',CONDITIONS[0],'pilot3')
        run([python,scripts/'verify_parity7.py',original,instrumented,folder/'baseline_parity.json'],'baseline_parity',repo)
        state['baseline_parity_passed']=True
        print('SABIT BENCHMARK: 2 x 1449 val goruntusu. K=21; tau=0.5. Yeni dogrulama kumesi DEGIL.',flush=True)
        state['completed_conditions']=[];state['benchmark_evaluated']=True;state['fresh_holdout']=False
        save_json(folder/'y2_benchmark_run.json',state)
        for c in CONDITIONS:
            out=evaluate(c['id'],c,'full_val1449');state['completed_conditions'].append(c['id']);save_json(folder/'y2_benchmark_run.json',state)
            print(f"TAMAMLANDI {len(state['completed_conditions'])}/2: {c['id']}",flush=True)
        run([python,scripts/'analyze9.py','--run-dir',folder],'benchmark_analysis',repo)
        state['ok']=True;code=0
    except (Exception,KeyboardInterrupt) as exc:
        state['error']=f'{type(exc).__name__}: {exc}';state['traceback']=traceback.format_exc();print(state['traceback'],flush=True)
    finally:
        state['attempts'][-1].update(finished_utc=datetime.now(timezone.utc).isoformat(),ok=state['ok']);save_json(folder/'y2_benchmark_run.json',state)
        archive=shutil.make_archive(str(root/('freecp_y2_benchmark_'+state['run_id'])),'zip',root_dir=folder)
        save_json(root/'latest_y2_benchmark_result.json',dict(ok=state['ok'],archive=archive,artifact=str(folder),completed_conditions=len(state['completed_conditions']),error=state.get('error')))
        print('RESULT_ZIP='+archive,flush=True);print('2/2 sabit benchmark tamamlandi.' if state['ok'] else 'Islem durdu; hata ZIP dosyasinda.',flush=True)
    return code

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',default='/content/freecp_t4');a=p.parse_args();raise SystemExit(main(a.root))
