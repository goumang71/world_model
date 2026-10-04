import sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'World-Model-2018-main'))
from controller_evaluation import CanonicalEvaluator,should_replace

def main():
    import numpy as np
    import torch
    from models.vae import VAE
    from models.mdnrnn import MDNRNN
    from models.controller import Controller
    old=dict(mean_return=700.,severe_failures=2,fully_off_fraction=.1,episodes=64)
    assert should_replace(dict(old,mean_return=710.),old)
    assert not should_replace(dict(old,mean_return=710.,severe_failures=3),old)
    assert not should_replace(dict(old,mean_return=710.,fully_off_fraction=.11),old)
    assert not should_replace(dict(old,mean_return=699.,severe_failures=0),old)
    torch.set_num_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    dev='cuda' if torch.cuda.is_available() else 'cpu'
    v=VAE().to(dev);m=MDNRNN(32,3,256,5).to(dev);c=Controller(288,3).to(dev)
    for model,name in [(v,'vae'),(m,'memory'),(c,'controller')]:
        model.load_state_dict(torch.load(ROOT/'World-Model-2018-main/checkpoints'/f'{name}.pth',weights_only=True,map_location=dev));model.eval().requires_grad_(False)
    x=torch.nn.utils.parameters_to_vector(c.parameters()).detach().cpu().numpy().astype(np.float64)
    e=CanonicalEvaluator(2,v,m,dev);start=time.perf_counter()
    try:rows=e.evaluate([x,x],[42,43],[42,43])
    finally:e.close()
    references=[(832.862190812706,24),(833.9933993399243,27)]
    print(json.dumps(rows,indent=2),flush=True)
    for row,(score,off) in zip(rows,references):
        assert abs(row['episode_return']-score)<1e-5,(row,score)
        assert row['fully_off_track_steps']==off
    result=dict(passed=True,selection_gate_checks=4,canonical_parity=rows,seconds=time.perf_counter()-start)
    p=ROOT/'outputs/robust_training_verification.json';p.write_text(json.dumps(result,indent=2))
    print('VERIFIED',p,flush=True)
if __name__=='__main__':main()
