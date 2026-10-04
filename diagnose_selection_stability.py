"""Separate track randomness from VAE randomness, using the canonical evaluator unchanged."""
import argparse,concurrent.futures,hashlib,json,subprocess,sys,time
from pathlib import Path
from datetime import datetime
ROOT=Path(__file__).resolve().parent
CANDIDATE=ROOT/'outputs/controller_warmstart_20260930_171351/checkpoints'

def worker(jobfile):
    job=json.loads(Path(jobfile).read_text());source=(ROOT/'evaluate_baseline.py').read_text(encoding='utf-8')
    if job['offset'] is not None:
        needle='                torch.manual_seed(seed)'
        assert source.count(needle)==1
        source=source.replace(needle,f"                torch.manual_seed(seed * 1009 + {job['offset']})")
    ns={'__name__':'stability_evaluator','__file__':str(ROOT/'evaluate_baseline.py')}
    exec(compile(source,str(ROOT/'evaluate_baseline.py'),'exec'),ns)
    sys.argv=['evaluate_baseline.py','--seeds',*[str(s) for s in job['seeds']], '--no-video',
              '--checkpoint-dir',job['checkpoints'],'--output-dir',job['output']]
    ns['main']()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--worker',type=Path);args=ap.parse_args()
    if args.worker:return worker(args.worker)
    out=ROOT/'outputs'/('selection_stability_'+datetime.now().strftime('%Y%m%d_%H%M%S'));out.mkdir()
    validation=list(range(20000,20008));fresh=list(range(40000,40016));offsets=[1000001,2000003,3000017]
    # Confirm the new tracks were not used in the preceding warm-start experiment.
    oldconfig=json.loads((CANDIDATE.parent/'config.json').read_text())
    assert not set(fresh)&set(oldconfig['training_seeds']+oldconfig['selection_seeds']+oldconfig['final_test_seeds'])
    hashes={}
    for label,cp in [('original',ROOT/'World-Model-2018-main/checkpoints'),('candidate',CANDIDATE)]:
        hashes[label]={n:hashlib.sha256((cp/f'{n}.pth').read_bytes()).hexdigest() for n in ['vae','memory','controller']}
    for n in ['vae','memory']:assert hashes['original'][n]==hashes['candidate'][n]
    config=dict(validation_seeds=validation,fresh_seeds=fresh,noise_offsets=offsets,
       latent_seed_formula='track_seed * 1009 + offset; original_noise uses track_seed',
       max_steps=1000,steering_delta=1.0,latent_mode='sample',models=hashes,training=False,
       evaluator_sha256=hashlib.sha256((ROOT/'evaluate_baseline.py').read_bytes()).hexdigest(),
       scope='8 previous selection tracks + 16 fresh tracks; 3 new noise streams each model. Additionally original noise on 8 selection tracks. 160 episodes.',
       caution='Candidate fixed before this experiment. Fresh scores not used to select a checkpoint. Only three noise replicates per track.')
    (out/'config.json').write_text(json.dumps(config,indent=2));jobs=[]
    for offset in [None]+offsets:
        for label,cp in [('original',ROOT/'World-Model-2018-main/checkpoints'),('candidate',CANDIDATE)]:
            name=label+('_original_noise' if offset is None else f'_noise_{offset}')
            job=dict(name=name,model=label,offset=offset,seeds=validation if offset is None else validation+fresh,
                     checkpoints=str(cp),output=str(out/name))
            path=out/(name+'.job.json');path.write_text(json.dumps(job,indent=2));jobs.append((job,path))
    state=dict(status='running',completed=[],failed=[]);start=time.perf_counter()
    def save():
        state['elapsed_seconds']=time.perf_counter()-start
        t=out/'status.tmp';t.write_text(json.dumps(state,indent=2));t.replace(out/'status.json')
    def run(item):
        job,path=item
        with (out/(job['name']+'.log')).open('w',encoding='utf-8') as log:
            code=subprocess.call([sys.executable,'-B',str(Path(__file__).resolve()),'--worker',str(path)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        return job['name'],code
    print('STABILITY_OUTPUT',out,flush=True);save()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        pending={pool.submit(run,j):j for j in jobs}
        while pending:
            done,_=concurrent.futures.wait(pending,timeout=30,return_when=concurrent.futures.FIRST_COMPLETED)
            for f in done:
                name,code=f.result();del pending[f]
                state['completed' if code==0 else 'failed'].append(name);save();print('JOB_FINISHED',name,code,flush=True)
            counts={j['name']:len(json.loads((Path(j['output'])/'episodes.json').read_text())) for j,_ in jobs if (Path(j['output'])/'episodes.json').exists()}
            print('PROGRESS',sum(counts.values()),'/160',flush=True)
    state['status']='completed' if not state['failed'] else 'failed';save()
    assert hashlib.sha256((ROOT/'evaluate_baseline.py').read_bytes()).hexdigest()==config['evaluator_sha256']
    print('FINAL_STATUS',json.dumps(state),flush=True)
    if state['failed']:raise RuntimeError('Some evaluation jobs failed; see logs')
if __name__=='__main__':main()
