"""CMA-ES retraining of the author's CarRacing controller action mapping."""
import argparse,hashlib,json,shutil,sys,time
from pathlib import Path
from datetime import datetime
from paper_train_evaluator import Evaluator
from paper_controller_evaluation import PaperCanonicalEvaluator,summarize,should_replace
from controller_evaluation import CanonicalEvaluator
from paper_controller import PaperController,warm_start_from_local
ROOT=Path(__file__).resolve().parent;REPO=ROOT/'World-Model-2018-main'

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--generations',type=int,default=8)
 ap.add_argument('--tracks-per-generation',type=int,default=4)
 ap.add_argument('--noise-repeats',type=int,default=2)
 ap.add_argument('--validation-every',type=int,default=4)
 ap.add_argument('--training-seed-start',type=int,required=True)
 ap.add_argument('--validation-seed-start',type=int,required=True)
 ap.add_argument('--test-seed-start',type=int,required=True)
 ap.add_argument('--output-dir',type=Path)
 args=ap.parse_args()
 if min(args.generations,args.tracks_per_generation,args.noise_repeats,args.validation_every)<1:ap.error('Budgets must be positive')
 train_range=set(range(args.training_seed_start,args.training_seed_start+args.generations*args.tracks_per_generation))
 val_range=set(range(args.validation_seed_start,args.validation_seed_start+32))
 test_range=set(range(args.test_seed_start,args.test_seed_start+32))
 if any(s<0 for s in (args.training_seed_start,args.validation_seed_start,args.test_seed_start)):ap.error('Track seeds must be nonnegative')
 if train_range&val_range or train_range&test_range or val_range&test_range:ap.error('Training, validation and test track seeds must be disjoint')
 if args.noise_repeats<2:ap.error('At least two latent-noise replicates required')
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
 out=(args.output_dir or ROOT/'outputs'/('paper_controller_train_'+datetime.now().strftime('%Y%m%d_%H%M%S'))).resolve();out.mkdir(parents=True,exist_ok=False)
 cp=out/'checkpoints';cp.mkdir();gd=out/'generations';gd.mkdir()
 hashes={}
 for name in ['vae','memory','controller']:
  src=REPO/'checkpoints'/f'{name}.pth';hashes[name]=hashlib.sha256(src.read_bytes()).hexdigest();shutil.copy2(src,cp/src.name)
 for name in ['train_paper_controller.py','paper_controller.py','paper_controller_evaluation.py','paper_train_evaluator.py','controller_evaluation.py','evaluate_baseline.py']:
  shutil.copy2(ROOT/name,out/('source_'+name))
 v=VAE().to(device);m=MDNRNN(32,3,256,5).to(device);original_c=Controller(288,3).to(device)
 for model,name in [(v,'vae'),(m,'memory'),(original_c,'controller')]:
  model.load_state_dict(torch.load(cp/f'{name}.pth',map_location=device,weights_only=True));model.eval().requires_grad_(False)
 c=warm_start_from_local(PaperController(288,3).to(device),original_c.state_dict()).eval()
 x_original=torch.nn.utils.parameters_to_vector(original_c.parameters()).detach().cpu().numpy().astype(np.float64).copy()
 x0=torch.nn.utils.parameters_to_vector(c.parameters()).detach().cpu().numpy().astype(np.float64).copy()
 # Preserve separate, explicit copies of both initial policies in this run.
 shutil.copy2(cp/'controller.pth',out/'controller_original.pth')
 torch.save(c.state_dict(),out/'paper_initial.pth')
 def cases(seeds,repeats,offset):return [dict(seed=s,latent_seed=offset+s*1009+r*1000003) for s in seeds for r in range(repeats)]
 valcases=cases(range(args.validation_seed_start,args.validation_seed_start+32),2,110000001)
 testcases=cases(range(args.test_seed_start,args.test_seed_start+32),3,120000001)
 config=dict(generations=args.generations,population=16,tracks_per_generation=args.tracks_per_generation,
   noise_repeats=args.noise_repeats,training_episodes=args.generations*16*args.tracks_per_generation*args.noise_repeats,
   sigma=.05,optimizer_seed=20261003,initialization='downloaded C with gas logits halved to preserve gas, fresh CMA optimizer',
   training_seeds=list(range(args.training_seed_start,args.training_seed_start+args.generations*args.tracks_per_generation)),
   latent_seed_formula='split_offset + track_seed*1009 + replicate*1000003',training_offset=100000001,
   validation_cases=valcases,test_cases=testcases,validation_every=args.validation_every,
   selection_gate='higher mean AND no more severe failures AND no higher fully-off fraction than incumbent',
   severe_failure='return<500 OR fully-off fraction>0.2 OR boundary exit',
   canonical_validation=True,training_inference='canonical_per_car',steering_delta=None,steps=1000,latent='sample',preprocessing='crop12 bicubic96',
   controller_action='steer=tanh(raw0), gas=(tanh(raw1)+1)/2, brake=clip(tanh(raw2),0,1)',
   comparator='original pretrained C and original action mapping',
   frozen=['vae','memory'],environment_reward_unchanged=True,source_sha256=hashes,
   final_test_policy='selected controller; if none accepted, best validation-mean trained candidate is tested for diagnosis only',
   torch=torch.__version__,cma=cma.__version__)
 (out/'config.json').write_text(json.dumps(config,indent=2));start=time.perf_counter()
 state=dict(status='initial_validation',generation=0,selected_generation=0,history=[]);train=None;canonical=None;old_eval=None
 def status():
  state['elapsed_seconds']=time.perf_counter()-start;t=out/'status.tmp';t.write_text(json.dumps(state,indent=2));t.replace(out/'status.json')
 def save_params(params,path):
  model=PaperController(288,3);torch.nn.utils.vector_to_parameters(torch.tensor(params,dtype=torch.float32),model.parameters());torch.save(model.state_dict(),path)
 def evaluate(params,items,label,evaluator=None):
  rows=[]
  evaluator=canonical if evaluator is None else evaluator
  for i in range(0,len(items),8):
   chunk=items[i:i+8];assert len(chunk)==8
   rows+=evaluator.evaluate([params]*8,[x['seed'] for x in chunk],[x['latent_seed'] for x in chunk])
   (out/(label+'.json')).write_text(json.dumps(rows,indent=2))
   print('EVALUATION',label,len(rows),'/',len(items),flush=True)
  return rows
 print('ROBUST_OUTPUT',out,flush=True);status()
 try:
  canonical=PaperCanonicalEvaluator(8,v,m,device)
  # Verify the actual eight-lane paper evaluator against saved direct-map runs.
  parity=canonical.evaluate([x0]*8,[100000]*8,[160900001,161900004]*4)
  for i,r in enumerate(parity):
   saved=ROOT/'outputs'/'paper_mapping_development_candidate_20261003'/'cases'/f"track_100000_noise_{r['latent_seed']}.json"
   ref=json.loads(saved.read_text(encoding='utf-8'))
   if abs(r['episode_return']-ref['episode_return'])>1e-5 or r['fully_off_track_steps']!=ref['fully_off_steps']:raise RuntimeError('Eight-lane paper parity failed')
  state['canonical_parity_passed']=True;print('CANONICAL_PARITY passed',flush=True)
  old_eval=CanonicalEvaluator(8,v,m,device)
  baseline_rows=evaluate(x_original,valcases,'validation_baseline',old_eval);old_eval.close();old_eval=None
  incumbent_stats=summarize(baseline_rows);incumbent=x0.copy()
  initial_paper_rows=evaluate(x0,valcases,'validation_paper_initial');state['paper_initial_validation']=summarize(initial_paper_rows)
  state['baseline_validation']=incumbent_stats;state['selected_validation']=incumbent_stats
  print('BASELINE_VALIDATION',json.dumps(incumbent_stats),flush=True)
  train=Evaluator(16,v,m,device)
  parity_cases=cases(range(140000,140008),1,130000001)
  parity_seeds=[x['seed'] for x in parity_cases]
  parity_latents=[x['latent_seed'] for x in parity_cases]
  train_scores,train_steps=train.evaluate([x0]*16,parity_seeds*2,latent_seeds=parity_latents*2)
  canonical_rows=canonical.evaluate([x0]*8,parity_seeds,parity_latents)
  if not np.array_equal(train_scores[:8],np.array([x['episode_return'] for x in canonical_rows])):
   raise RuntimeError('Training and canonical returns differ')
  if not np.array_equal(train_steps[:8],np.array([x['steps'] for x in canonical_rows])):
   raise RuntimeError('Training and canonical episode lengths differ')
  if not np.array_equal(train_scores[:8],train_scores[8:]) or not np.array_equal(train_steps[:8],train_steps[8:]):
   raise RuntimeError('Training lanes with duplicate cases differ')
  state['training_canonical_parity_passed']=True;status();print('TRAINING_CANONICAL_PARITY passed',flush=True)
  es=cma.CMAEvolutionStrategy(x0,.05,{'popsize':16,'seed':20261003,'verbose':-9})
  state['status']='training';status();highest_mean=-float('inf');highest_params=None;highest_gen=None
  for gen in range(1,args.generations+1):
   tic=time.perf_counter();solutions=es.ask()
   tracks=list(range(args.training_seed_start+(gen-1)*args.tracks_per_generation,args.training_seed_start+gen*args.tracks_per_generation))
   traincases=cases(tracks,args.noise_repeats,100000001);scores=[]
   for j,case in enumerate(traincases):
    vals,_=train.evaluate(solutions,[case['seed']]*16,latent_seeds=[case['latent_seed']]*16)
    scores.append(vals);print('TRAIN_CASE',gen,j+1,'/',len(traincases),flush=True)
   scores=np.stack(scores);fitness=scores.mean(0);es.tell(solutions,(-fitness).tolist())
   winner=np.array(solutions[int(np.argmax(fitness))],copy=True);save_params(winner,gd/f'controller_gen{gen:03d}.pth')
   row=dict(generation=gen,training_cases=traincases,mean_return=float(fitness.mean()),best_return=float(fitness.max()),
      per_case_returns=scores.tolist(),sigma=float(es.sigma))
   if gen%args.validation_every==0 or gen==args.generations:
    state['status']='validating';status()
    valrows=evaluate(winner,valcases,f'validation_gen{gen:03d}');stats=summarize(valrows)
    accepted=should_replace(stats,incumbent_stats)
    row.update(validation=stats,accepted=accepted)
    if stats['mean_return']>highest_mean:
     highest_mean=stats['mean_return'];highest_params=winner.copy();highest_gen=gen
     save_params(winner,out/'best_validation_mean_candidate.pth')
    if accepted:
     incumbent=winner.copy();incumbent_stats=stats;save_params(incumbent,cp/'controller.pth')
     state['selected_generation']=gen;state['selected_validation']=stats
    print('VALIDATION',gen,json.dumps(stats),'ACCEPTED',accepted,flush=True)
   row['seconds']=time.perf_counter()-tic;state['history'].append(row);state['generation']=gen
   temp=out/'cma_state.tmp';temp.write_bytes(es.pickle_dumps());temp.replace(out/'cma_state.pkl')
   state['status']='training';status();print('GENERATION',gen,'COMPLETE',round(row['seconds'],1),flush=True)
  train.close();train=None
  if state['selected_generation']:
   tested=incumbent;tested_gen=state['selected_generation'];diagnostic_only=False
  else:
   tested=highest_params;tested_gen=highest_gen;diagnostic_only=True
  state.update(status='final_evaluation',tested_generation=tested_gen,tested_candidate_diagnostic_only=diagnostic_only);status()
  tested_cp=out/'tested_checkpoints';tested_cp.mkdir()
  for name in ['vae','memory']:shutil.copy2(cp/f'{name}.pth',tested_cp/f'{name}.pth')
  save_params(tested,tested_cp/'controller.pth')
  old_eval=CanonicalEvaluator(8,v,m,device)
  original_rows=evaluate(x_original,testcases,'test_original',old_eval);old_eval.close();old_eval=None
  candidate_rows=evaluate(tested,testcases,'test_candidate')
  a=summarize(original_rows);b=summarize(candidate_rows);diff=np.array([n['episode_return']-o['episode_return'] for o,n in zip(original_rows,candidate_rows)]).reshape(32,3)
  rng=np.random.default_rng(20261003);boot=rng.choice(diff.mean(1),(20000,32),replace=True).mean(1)
  result=dict(original=a,candidate=b,mean_difference=float(diff.mean()),cluster_bootstrap_95_interval=np.percentile(boot,[2.5,97.5]).tolist(),
   improved_pairs=int((diff>1e-6).sum()),worse_pairs=int((diff<-1e-6).sum()),tied_pairs=int((np.abs(diff)<=1e-6).sum()),
   tracks=32,noise_repeats=3,selected_generation=state['selected_generation'],tested_generation=tested_gen,
   diagnostic_only=diagnostic_only,test_used_for_selection=False,original_rows=original_rows,candidate_rows=candidate_rows)
  (out/'comparison.json').write_text(json.dumps(result,indent=2));state['status']='completed';state['comparison_summary']={k:v for k,v in result.items() if not k.endswith('_rows')}
  print('RESULT',json.dumps(state['comparison_summary']),flush=True)
 except BaseException as exc:
  state['status']='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed';state['error']=repr(exc);raise
 finally:
  if train is not None:train.close()
  if canonical is not None:canonical.close()
  if old_eval is not None:old_eval.close()
  state['source_weights_unchanged']=all(hashlib.sha256((REPO/'checkpoints'/f'{n}.pth').read_bytes()).hexdigest()==h for n,h in hashes.items())
  status();print('STATUS',state['status'],state['elapsed_seconds'],flush=True)
if __name__=='__main__':main()
