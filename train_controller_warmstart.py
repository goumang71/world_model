"""Warm-start CMA-ES diagnostic; frozen V/M and matched steering mapping.
Run from project root with .venv/Scripts/python.exe -B train_controller_warmstart.py.
"""
import argparse, hashlib, json, shutil, sys, time
from pathlib import Path
from datetime import datetime
from train_controller import Evaluator
ROOT=Path(__file__).resolve().parent
REPO=ROOT/'World-Model-2018-main'

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--generations',type=int,default=20)
    ap.add_argument('--sigma',type=float,default=.05)
    ap.add_argument('--output-dir',type=Path)
    args=ap.parse_args()
    if not 1<=args.generations<=100 or args.sigma<=0:ap.error('generations 1..100 and positive sigma required')
    sys.path.insert(0,str(REPO))
    import numpy as np
    import torch
    import cma
    from models.vae import VAE
    from models.mdnrnn import MDNRNN
    from models.controller import Controller
    torch.set_num_threads(1);torch.manual_seed(20260930);np.random.seed(20260930)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out=(args.output_dir or ROOT/'outputs'/('controller_warmstart_'+datetime.now().strftime('%Y%m%d_%H%M%S'))).resolve()
    out.mkdir(parents=True,exist_ok=False)
    cp=out/'checkpoints';cp.mkdir();gd=out/'generations';gd.mkdir()
    hashes={}
    for name in ['vae','memory','controller']:
        src=REPO/'checkpoints'/f'{name}.pth';hashes[name]=hashlib.sha256(src.read_bytes()).hexdigest();shutil.copy2(src,cp/src.name)
    for src in [ROOT/'train_controller.py',Path(__file__).resolve(),ROOT/'evaluate_baseline.py']:
        shutil.copy2(src,out/('source_'+src.name))
    v=VAE().to(device);m=MDNRNN(32,3,256,5).to(device);c=Controller(288,3).to(device)
    for model,name in [(v,'vae'),(m,'memory'),(c,'controller')]:
        model.load_state_dict(torch.load(cp/f'{name}.pth',map_location=device,weights_only=True));model.eval().requires_grad_(False)
    x0=torch.nn.utils.parameters_to_vector(c.parameters()).detach().cpu().numpy().astype(np.float64).copy()
    incumbent=x0.copy()
    config=dict(initialization='downloaded pretrained controller; fresh CMA optimizer (not optimizer resume)',
        generations=args.generations,population=16,rollouts=3,sigma=args.sigma,optimizer_seed=20260930,
        steering_delta=1.0,steps=1000,latent='sample',preprocessing='crop bottom 12; torch bicubic 96x96',
        brake='0.8*sigmoid(raw_brake)*(1-sigmoid(raw_gas))',frozen=['vae','memory'],
        training_seeds=list(range(10000,10000+3*args.generations)),selection_seeds=list(range(20000,20008)),
        final_test_seeds=list(range(30000,30024)),selection_every=5,source_sha256=hashes,
        torch=torch.__version__,cma=cma.__version__,device=str(device),
        note='Small-budget diagnostic, not a convergence claim. Paired batched evaluations for candidate and incumbent.')
    (out/'config.json').write_text(json.dumps(config,indent=2))
    state=dict(status='verifying',generation=0,selected_generation=0,history=[],config=config)
    begin=time.perf_counter();e=None;one=None
    def save_status():
        state['elapsed_seconds']=time.perf_counter()-begin
        t=out/'status.tmp';t.write_text(json.dumps(state,indent=2));t.replace(out/'status.json')
    def save_params(params,path):
        model=Controller(288,3)
        torch.nn.utils.vector_to_parameters(torch.tensor(params,dtype=torch.float32),model.parameters())
        torch.save(model.state_dict(),path)
    print('OUTPUT',out,flush=True);save_status()
    try:
        # Singleton evaluation must reproduce the current canonical evaluator.
        one=Evaluator(1,v,m,device,steering_delta=1.0)
        score,steps=one.evaluate([x0],[42])
        one.close();one=None
        state['singleton_parity']=dict(seed=42,score=float(score[0]),steps=int(steps[0]),reference=832.862190812706)
        if abs(score[0]-832.862190812706)>1e-5:
            raise RuntimeError('Singleton training/evaluation parity failed; inspect before optimizing')
        print('PARITY',state['singleton_parity'],flush=True)
        e=Evaluator(16,v,m,device,steering_delta=1.0)
        seeds=config['selection_seeds']
        base,_=e.evaluate([x0]*16,seeds+seeds)
        if not np.allclose(base[:8],base[8:],rtol=0,atol=1e-6):raise RuntimeError('Paired duplicate baseline mismatch')
        best=float(base[:8].mean());state['baseline_selection_returns']=base[:8].tolist();state['selection_mean']=best
        print('BASELINE_SELECTION',best,base[:8].tolist(),flush=True)
        state['status']='training';save_status()
        es=cma.CMAEvolutionStrategy(x0,args.sigma,{'popsize':16,'seed':20260930,'verbose':-9})
        for generation in range(1,args.generations+1):
            tic=time.perf_counter();solutions=es.ask();scores=[]
            trainseeds=list(range(10000+(generation-1)*3,10000+generation*3))
            for seed in trainseeds:
                values,_=e.evaluate(solutions,[seed]*16);scores.append(values)
            rewards=np.mean(scores,axis=0);es.tell(solutions,(-rewards).tolist())
            winner=np.asarray(solutions[int(np.argmax(rewards))]).copy()
            save_params(winner,gd/f'controller_gen{generation:03d}.pth')
            row=dict(generation=generation,training_seeds=trainseeds,mean_return=float(rewards.mean()),best_return=float(rewards.max()),
              worst_return=float(rewards.min()),sigma=float(es.sigma))
            if generation%5==0 or generation==args.generations:
                val,_=e.evaluate([winner]*8+[incumbent]*8,seeds+seeds)
                row.update(selection_returns=val[:8].tolist(),selection_mean=float(val[:8].mean()),incumbent_recheck=float(val[8:].mean()))
                if abs(float(val[8:].mean())-best)>1e-5:raise RuntimeError('Incumbent re-evaluation changed')
                if row['selection_mean']>best:
                    incumbent=winner.copy();best=row['selection_mean'];save_params(incumbent,cp/'controller.pth')
                    state['selected_generation']=generation;state['selection_mean']=best
                row['selected_generation']=state['selected_generation']
            row['seconds']=time.perf_counter()-tic
            state['history'].append(row);state['generation']=generation
            temp=out/'cma_state.tmp';temp.write_bytes(es.pickle_dumps());temp.replace(out/'cma_state.pkl')
            save_status();print(json.dumps(row),flush=True)
        state['status']='final_evaluation';save_status()
        final=[]
        for start in range(0,24,8):
            testseeds=config['final_test_seeds'][start:start+8]
            vals,steps=e.evaluate([x0]*8+[incumbent]*8,testseeds+testseeds)
            for j,seed in enumerate(testseeds):final.append(dict(seed=seed,baseline=float(vals[j]),selected=float(vals[j+8]),
                difference=float(vals[j+8]-vals[j]),baseline_steps=int(steps[j]),selected_steps=int(steps[j+8])))
            (out/'final_episodes.json').write_text(json.dumps(final,indent=2))
            print('FINAL_BATCH',start, 'old',float(vals[:8].mean()),'selected',float(vals[8:].mean()),flush=True)
        diffs=np.array([r['difference'] for r in final]);old=np.array([r['baseline'] for r in final]);new=np.array([r['selected'] for r in final])
        result=dict(selected_generation=state['selected_generation'],selection_baseline=float(base[:8].mean()),selection_selected=best,
          n=24,baseline_mean=float(old.mean()),selected_mean=float(new.mean()),mean_difference=float(diffs.mean()),
          baseline_std=float(old.std(ddof=1)),selected_std=float(new.std(ddof=1)),
          paired_difference_standard_error=float(diffs.std(ddof=1)/np.sqrt(24)),
          improved=int((diffs>1e-6).sum()),worse=int((diffs<-1e-6).sum()),unchanged=int((np.abs(diffs)<=1e-6).sum()),
          test_used_for_selection=False,episodes=final)
        (out/'comparison.json').write_text(json.dumps(result,indent=2));state['comparison']=result
        state['status']='completed';print('COMPARISON',json.dumps(result),flush=True)
    except BaseException as exc:
        state['status']='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed';state['error']=repr(exc);raise
    finally:
        if one is not None:one.close()
        if e is not None:e.close()
        state['source_weights_unchanged']=all(hashlib.sha256((REPO/'checkpoints'/f'{n}.pth').read_bytes()).hexdigest()==h for n,h in hashes.items())
        save_status();print('STATUS',state['status'],'seconds',state['elapsed_seconds'],flush=True)

if __name__=='__main__':main()
