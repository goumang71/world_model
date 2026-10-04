"""Frozen-model diagnostics on actual evaluator trajectories; no policy modifications."""
import sys, json, hashlib
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from PIL import Image, ImageDraw
ROOT=Path(__file__).resolve().parent
source=(ROOT/'evaluate_baseline.py').read_text(encoding='utf-8')
out=ROOT/'outputs'/('model_diagnosis_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
out.mkdir()
records=[]
def capture(image,z,mu,logvar,hidden,action,env,step,seed):
    car=env.unwrapped.car
    track=np.asarray(env.unwrapped.track)
    pos=np.asarray(car.hull.position)
    nearest=int(np.argmin(((track[:,2:4]-pos)**2).sum(1)))
    angles=track[np.arange(nearest-3,nearest+4)%len(track),1]
    bend=float(np.abs((np.diff(angles)+np.pi)%(2*np.pi)-np.pi).sum())
    row=dict(seed=seed,step=step+1,wheels=sum(bool(w.tiles) for w in car.wheels),bend=bend,
      speed=float(np.linalg.norm(np.asarray(car.hull.linearVelocity))))
    for name,value in [('z',z),('mu',mu),('logvar',logvar),('h',hidden[0]),('c',hidden[1]),('action',action)]:
        row[name]=value.detach().cpu().numpy().reshape(-1)
    if step%5==0:
        row['image']=image[0].detach().cpu().numpy()
    records.append(row)
needle='                    previous_steering = action[0, 0].item()'
assert source.count(needle)==1
modified=source.replace(needle,needle+'\n                    capture(image,z,mu,logvar,hidden,action,env,step,seed)')
ns={'__name__':'diagnostic_evaluator','__file__':str(ROOT/'evaluate_baseline.py'),'capture':capture}
exec(compile(modified,str(ROOT/'evaluate_baseline.py'),'exec'),ns)
sys.argv=['evaluate_baseline.py','--seeds','42','52','4470','--no-video','--output-dir',str(out/'evaluation')]
ns['main']()
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
models=[]
for cls,args,name in [(ns['VAE'],(3,32),'vae'),(ns['MDNRNN'],(32,3,256,5),'memory'),(ns['Controller'],(288,3),'controller')]:
    model=cls(*args).to(device).eval()
    model.load_state_dict(torch.load(ROOT/'World-Model-2018-main/checkpoints'/f'{name}.pth',map_location=device,weights_only=True))
    models.append(model)
v,m,c=models
arrays={k:np.stack([r[k] for r in records]) for k in ['z','mu','logvar','h','c','action']}
for k in ['seed','step','wheels','bend','speed']: arrays[k]=np.array([r[k] for r in records])
np.savez_compressed(out/'trajectory_latents.npz',**arrays)
images=[i for i,r in enumerate(records) if 'image' in r]
np.savez_compressed(out/'observations.npz',indices=images,images=np.stack([records[i]['image'] for i in images]))
recons={}; vae_rows=[]
with torch.inference_mode():
    for start in range(0,len(images),64):
        ids=images[start:start+64]
        x=torch.tensor(np.stack([records[i]['image'] for i in ids]),device=device)
        mean=torch.tensor(arrays['mu'][ids],device=device); z=torch.tensor(arrays['z'][ids],device=device)
        rm=v.decode(mean); rz=v.decode(z)
        for j,i in enumerate(ids):
            r=records[i]
            category='offroad' if r['wheels']==0 else ('edge' if r['wheels']<4 else ('bend' if r['bend']>=.1 else 'straight'))
            row=dict(index=i,seed=r['seed'],step=r['step'],category=category,
                mean_mse=float(((rm[j]-x[j])**2).mean()),sample_mse=float(((rz[j]-x[j])**2).mean()))
            vae_rows.append(row); recons[i]=(rm[j].cpu().numpy(),rz[j].cpu().numpy())
    # Score only within-episode, adjacent transitions. Predict from (z_t, a_t, h_t).
    ids=np.array([i for i in range(len(records)-1) if records[i]['seed']==records[i+1]['seed']])
    mdn_rows=[]
    for start in range(0,len(ids),128):
        ix=ids[start:start+128]
        def tensor(key,which=ix): return torch.tensor(arrays[key][which],device=device,dtype=torch.float32)
        z=tensor('z').unsqueeze(1); a=tensor('action').unsqueeze(1)
        hidden=(tensor('h').unsqueeze(0),tensor('c').unsqueeze(0))
        target=tensor('z',ix+1); target_mean=tensor('mu',ix+1)
        metrics={}
        for variant in ['normal','zero_action','zero_history']:
            pi,mean,sigma,_=m(z,torch.zeros_like(a) if variant=='zero_action' else a,
              h=tuple(torch.zeros_like(t) for t in hidden) if variant=='zero_history' else hidden)
            pi=pi[:,0];mean=mean[:,0];sigma=sigma[:,0]
            lp=torch.distributions.Normal(mean,sigma).log_prob(target[:,None]).sum(-1)
            metrics[variant+'_nll_per_dim']=(-torch.logsumexp(pi.log()+lp,dim=-1)/32).cpu().numpy()
            expected=(pi[:,:,None]*mean).sum(1)
            metrics[variant+'_mean_mse']=((expected-target_mean)**2).mean(-1).cpu().numpy()
            if variant=='normal':
                metrics['mixture_entropy']=(-(pi*pi.log()).sum(-1)).cpu().numpy()
        metrics['persistence_mean_mse']=((tensor('mu')-target_mean)**2).mean(-1).cpu().numpy()
        for j,i in enumerate(ix):
            mdn_rows.append(dict(seed=int(arrays['seed'][i]),step=int(arrays['step'][i]),
                **{k:float(val[j]) for k,val in metrics.items()}))
    state=torch.tensor(np.concatenate([arrays['z'],arrays['h']],1),device=device)
    raw=c.model(state)
    unfiltered=c(state).cpu().numpy()
    raw_noise=(torch.tensor(arrays['logvar'],device=device).exp()*c.model[0].weight[0,:32].square()).sum(-1).sqrt().cpu().numpy()
summary=dict(protocol=dict(seeds=[42,52,4470],max_steps=1000,steering_delta=1.0,latent='sample',
    evaluator_sha256=hashlib.sha256(source.encode()).hexdigest(),note='Existing policy trajectories; not independent of historical test-seed selection. No retraining.'),vae={},mdn={},controller={})
for category in ['straight','bend','edge','offroad']:
    rows=[r for r in vae_rows if r['category']==category]
    if not rows: continue
    summary['vae'][category]=dict(n=len(rows),**{k:float(np.mean([r[k] for r in rows])) for k in ['mean_mse','sample_mse']})
    # Choose time-spaced examples in each category, after initial camera zoom.
    eligible=[r for r in rows if r['step']>100]; selected=[]
    for r in eligible:
        if all(r['seed']!=s['seed'] or abs(r['step']-s['step'])>=100 for s in selected): selected.append(r)
        if len(selected)==6: break
    canvas=Image.new('RGB',(720,50+len(selected)*170),'white');draw=ImageDraw.Draw(canvas)
    draw.text((10,8),category+' | input / decode(mu) / decode(actual sampled z)',fill='black')
    for j,r in enumerate(selected):
        i=r['index'];y=40+j*170
        draw.text((8,y),f"seed {r['seed']} step {r['step']} MSE {r['sample_mse']:.4f}",fill='black')
        for col,img in enumerate((records[i]['image'],*recons[i])):
            pic=Image.fromarray(np.uint8(np.clip(img.transpose(1,2,0),0,1)*255)).resize((144,144))
            canvas.paste(pic,(30+col*230,y+20))
    canvas.save(out/f'vae_{category}.png')
for seed in [42,52,4470]:
    rows=[r for r in mdn_rows if r['seed']==seed]
    summary['mdn'][str(seed)]={k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k not in ['seed','step']}
    mask=arrays['seed']==seed; actions=arrays['action'][mask]
    summary['controller'][str(seed)]=dict(raw_steering_saturation=float((np.abs(unfiltered[mask,0])>.95).mean()),
       executed_steering_saturation=float((np.abs(actions[:,0])>.95).mean()),
       raw_steering_noise_std_mean=float(raw_noise[mask].mean()),gas_mean=float(actions[:,1].mean()),brake_mean=float(actions[:,2].mean()),
       fully_off_steps=int((arrays['wheels'][mask]==0).sum()))
(out/'summary.json').write_text(json.dumps(summary,indent=2))
(out/'vae_metrics.json').write_text(json.dumps(vae_rows))
(out/'mdn_metrics.json').write_text(json.dumps(mdn_rows))
assert hashlib.sha256((ROOT/'evaluate_baseline.py').read_text(encoding='utf-8').encode()).hexdigest()==summary['protocol']['evaluator_sha256']
print(json.dumps(summary,indent=2),flush=True)
print('DIAGNOSIS_OUTPUT',out,flush=True)
