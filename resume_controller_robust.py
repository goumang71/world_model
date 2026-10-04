"""Resume a locally produced robust-training run by verified CMA replay.
No environment training is repeated for completed generations.
"""
import argparse,copy,hashlib,json,pickle,shutil,sys
from pathlib import Path
from datetime import datetime
ROOT=Path(__file__).resolve().parent

def vector(path):
    import numpy as np
    import torch
    weights=torch.load(path,map_location='cpu',weights_only=True)
    return np.concatenate([v.numpy().ravel() for v in weights.values()]).astype(np.float64)

def restore_optimizer(x0,config,state,out):
    import cma
    import numpy as np
    es=cma.CMAEvolutionStrategy(x0,config['sigma'],{'popsize':config['population'],'seed':config['optimizer_seed'],'verbose':-9})
    verified=[]
    for row in state['history']:
        solutions=es.ask();fitness=np.asarray(row['per_case_returns']).mean(0)
        es.tell(solutions,(-fitness).tolist())
        winner=np.asarray(solutions[int(np.argmax(fitness))],dtype=np.float32)
        expected=vector(out/'generations'/f"controller_gen{row['generation']:03d}.pth").astype(np.float32)
        if not np.array_equal(winner,expected):raise RuntimeError(f"CMA replay differs at generation {row['generation']}")
        verified.append(row['generation'])
    rng=np.random.get_state()
    # This pickle was generated locally by the preceding authorized run.
    with (out/'cma_state.pkl').open('rb') as f: checkpoint=pickle.load(f)
    checks=dict(iteration=es.countiter==checkpoint.countiter==state['generation'],
                mean=np.array_equal(es.mean,checkpoint.mean),sigma=es.sigma==checkpoint.sigma,
                covariance=np.array_equal(es.sm.C,checkpoint.sm.C))
    checks={k:bool(v) for k,v in checks.items()}
    np.random.set_state(rng)
    if not all(checks.values()):raise RuntimeError(f'CMA replay verification failed: {checks}')
    (out/'resume_verification.json').write_text(json.dumps(dict(verified_generations=verified,checks=checks,
        method='Replay saved ask/tell scores to recover optimizer and NumPy RNG; no rollout replay'),indent=2))
    print('RESUME_VERIFIED',verified,checks,flush=True)
    return es

def restore_highest(state,out):
    rows=[r for r in state['history'] if 'validation' in r]
    if not rows:return -float('inf'),None,None
    best=max(rows,key=lambda r:r['validation']['mean_return'])
    return best['validation']['mean_return'],vector(out/'generations'/f"controller_gen{best['generation']:03d}.pth"),best['generation']

def main():
    ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path);args=ap.parse_args();out=args.directory.resolve()
    state=json.loads((out/'status.json').read_text());cfg=json.loads((out/'config.json').read_text())
    if state['status']=='completed':raise RuntimeError('Training is already complete; run analyze_robust_training.py instead')
    for name in ['train_controller_robust.py','controller_evaluation.py','train_controller.py','evaluate_baseline.py']:
        if (ROOT/name).read_bytes()!=(out/('source_'+name)).read_bytes():raise RuntimeError(f'Source changed: {name}')
    for name,expected in cfg['source_sha256'].items():
        if hashlib.sha256((ROOT/'World-Model-2018-main/checkpoints'/f'{name}.pth').read_bytes()).hexdigest()!=expected:raise RuntimeError(f'Original weights changed: {name}')
    shutil.copy2(out/'status.json',out/('status_before_resume_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.json'))
    shutil.copy2(Path(__file__).resolve(),out/'source_resume_controller_robust.py')
    source=(out/'source_train_controller_robust.py').read_text(encoding='utf-8-sig')
    def change(old,new):
        nonlocal source
        if source.count(old)!=1:raise RuntimeError(f'Unexpected source for resume patch: {old[:90]}')
        source=source.replace(old,new)
    change("out=(args.output_dir or ROOT/'outputs'/('controller_robust_'+datetime.now().strftime('%Y%m%d_%H%M%S'))).resolve();out.mkdir(parents=True,exist_ok=False)","out=RESUME_DIR")
    change("cp=out/'checkpoints';cp.mkdir();gd=out/'generations';gd.mkdir()","cp=out/'checkpoints';gd=out/'generations'")
    change(";shutil.copy2(src,cp/src.name)","")
    change("  shutil.copy2(ROOT/name,out/('source_'+name))","  pass  # Preserve original source snapshots")
    change("torch.load(cp/f'{name}.pth',map_location=device,weights_only=True)","torch.load((REPO/'checkpoints' if name=='controller' else cp)/f'{name}.pth',map_location=device,weights_only=True)")
    change("(out/'config.json').write_text(json.dumps(config,indent=2));start=time.perf_counter()","assert config==RESUME_CONFIG;start=time.perf_counter()-RESUME_STATE['elapsed_seconds']")
    change("state=dict(status='initial_validation',generation=0,selected_generation=0,history=[]);train=None;canonical=None","state=copy.deepcopy(RESUME_STATE);state.pop('error',None);state['status']='resuming';train=None;canonical=None")
    change("baseline_rows=evaluate(x0,valcases,'validation_baseline');incumbent_stats=summarize(baseline_rows);incumbent=x0.copy()","baseline_rows=json.loads((out/'validation_baseline.json').read_text());assert summarize(baseline_rows)==state['baseline_validation'];incumbent_stats=state['selected_validation'];incumbent=vector(cp/'controller.pth')")
    change("state['baseline_validation']=incumbent_stats;state['selected_validation']=incumbent_stats","state['selected_validation']=incumbent_stats")
    change("es=cma.CMAEvolutionStrategy(x0,.05,{'popsize':16,'seed':20260930,'verbose':-9})","es=restore_optimizer(x0,config,state,out)")
    change("highest_mean=-float('inf');highest_params=None;highest_gen=None","highest_mean,highest_params,highest_gen=restore_highest(state,out)")
    change("for gen in range(1,args.generations+1):","for gen in range(RESUME_STATE['generation']+1,args.generations+1):")
    change("tested_cp=out/'tested_checkpoints';tested_cp.mkdir()","tested_cp=out/'tested_checkpoints';tested_cp.mkdir(exist_ok=True)")
    (out/'source_resume_effective.py').write_text(source,encoding='utf-8')
    sys.argv=['train_controller_robust.py','--generations',str(cfg['generations']),
        '--tracks-per-generation',str(cfg['tracks_per_generation']),'--noise-repeats',str(cfg['noise_repeats']),
        '--validation-every',str(cfg['validation_every'])]
    ns=dict(__name__='resumed_robust_training',__file__=str(ROOT/'train_controller_robust.py'),
        RESUME_DIR=out,RESUME_CONFIG=cfg,RESUME_STATE=state,copy=copy,vector=vector,
        restore_optimizer=restore_optimizer,restore_highest=restore_highest)
    exec(compile(source,str(out/'source_resume_effective.py'),'exec'),ns)
    ns['main']()
if __name__=='__main__':main()
