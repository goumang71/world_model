import sys,json
from pathlib import Path
import numpy as np
import torch
from train_controller import Evaluator
ROOT=Path(__file__).resolve().parent

def main():
 source=(ROOT/'evaluate_baseline.py').read_text();base=[];batch=[]
 def capture(dest):
  def f(image,z,mu,hidden,actions,obs):
   if len(dest)<250:dest.append({k:v.detach().cpu().numpy().copy() for k,v in [('image',image),('z',z),('mu',mu),('h',hidden[0]),('actions',actions)]})
  return f
 old='                    previous_steering = action[0, 0].item()'
 ns={'__name__':'diagnostic','__file__':str(ROOT/'evaluate_baseline.py'),'capture':capture(base)}
 exec(compile(source.replace(old,old+'\n                    capture(image,z,mu,hidden,action,obs)'),str(ROOT/'evaluate_baseline.py'),'exec'),ns)
 sys.argv=['evaluate_baseline.py','--seed','42','--episodes','1','--no-video','--output-dir',str(ROOT/'outputs/warmstart_parity_reference')]
 ns['main']()
 dev='cuda';v=ns['VAE']().to(dev);m=ns['MDNRNN'](32,3,256,5).to(dev);c=ns['Controller'](288,3).to(dev)
 for model,name in [(v,'vae'),(m,'memory'),(c,'controller')]:
  model.load_state_dict(torch.load(ROOT/'World-Model-2018-main/checkpoints'/f'{name}.pth',map_location=dev,weights_only=True));model.eval().requires_grad_(False)
 e=Evaluator(1,v,m,dev,steering_delta=1.0);e.trace_callback=capture(batch)
 try:print('BATCH',e.evaluate([torch.nn.utils.parameters_to_vector(c.parameters()).detach().cpu().numpy().astype(np.float64)],[42]))
 finally:e.close()
 for k in base[0]:
  diffs=np.array([np.max(np.abs(a[k]-b[k])) for a,b in zip(base,batch)])
  print(k,'firstdiff',np.where(diffs>0)[0][:5].tolist(),'first10',diffs[:10].tolist(),'max',float(diffs.max()))
 np.savez_compressed(ROOT/'outputs/warmstart_parity_traces.npz',**{prefix+'_'+k:np.stack([r[k] for r in rows]) for prefix,rows in [('baseline',base),('batch',batch)] for k in rows[0]})
if __name__=='__main__':main()
