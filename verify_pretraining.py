"""Regression checks for boundaries/clipping and a representative GPU memory probe."""
import json
import sys
from datetime import datetime
from pathlib import Path

from verify_training import definitions, REPO, ROOT


def main():
    sys.path.insert(0, str(REPO))
    import h5py
    import numpy as np
    import torch
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    from tqdm import tqdm
    import os
    from models.vae import VAE
    from models.mdnrnn import MDNRNN, sequence_nll_loss, gaussian_nll_loss
    import rollout_data

    torch.set_num_threads(1)
    torch.manual_seed(321)
    out = ROOT / 'outputs' / ('pretraining_checks_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True)
    report = dict(status='running', checks=[], cuda_profile={})
    try:
        # A terminal OR truncated transition must trigger reset and be recorded.
        class FakeEnv:
            def __init__(self): self.step_id = 0; self.resets = 0
            def reset(self, seed=None):
                self.resets += 1
                return np.full((96,96,3), self.resets, dtype=np.uint8), {}
            def step(self, action):
                self.step_id += 1
                return np.zeros((96,96,3),dtype=np.uint8), -0.1, self.step_id==2, self.step_id==4, {}
            def close(self): pass
        fake = FakeEnv()
        real_make = rollout_data.gym.make
        rollout_data.gym.make = lambda *a, **kw: fake
        try:
            path = rollout_data.collect_episode('test', 6, 0, 0, out/'fake', seed=1)
        finally:
            rollout_data.gym.make = real_make
        with h5py.File(path, 'r') as f:
            assert f['dones'][:].tolist() == [False,True,False,True,False,False]
            assert f['images'][2,0,0,0] == 2 and f['images'][4,0,0,0] == 3
            assert np.allclose(f['rewards'][:], -0.1)
        assert fake.resets == 3
        report['checks'].append('collection resets on termination and truncation, preserves fractional rewards')

        model = MDNRNN(32,3,16,5)
        z = torch.randn(1,6,32,requires_grad=True)
        target = torch.randn(1,6,32,requires_grad=True)
        action = torch.randn(1,6,3)
        done = torch.tensor([[False,True,False,False,True,False]])
        loss,_ = sequence_nll_loss(model,z,target,action,done)
        expected = 0
        for lo,hi in [(0,1),(2,4),(5,6)]:
            pi,mu,sigma,_ = model(z[:,lo:hi],action[:,lo:hi])
            expected += gaussian_nll_loss(pi,mu,sigma,target[:,lo:hi])*(hi-lo)/4
        torch.testing.assert_close(loss,expected)
        loss.backward()
        assert not target.grad[:,[1,4]].any() and not z.grad[:,[1,4]].any()
        full,_ = sequence_nll_loss(model,z,target,action,torch.zeros_like(done))
        pi,mu,sigma,_ = model(z,action)
        torch.testing.assert_close(full,gaussian_nll_loss(pi,mu,sigma,target))
        try:
            sequence_nll_loss(model,z,target,action,torch.ones_like(done))
        except ValueError: pass
        else: raise AssertionError('All-boundary batch must be rejected')
        report['checks'].append('boundary links excluded, LSTM reset, no-boundary loss unchanged')

        scope = dict(torch=torch, os=os, tqdm=tqdm, h5py=h5py, Dataset=Dataset, transforms=transforms)
        # Instrument the actual trainer: gradients must exist at clipping and be clipped at optimizer.step.
        for source in ['memory.py','3-Memory (rnn-mdn).ipynb']:
            definitions(REPO/source,['train_model'],scope)
            memory = MDNRNN(32,3,16,5)
            calls = []
            original_clip = torch.nn.utils.clip_grad_norm_
            def checked_clip(parameters,*args,**kwargs):
                params=list(parameters)
                assert any(p.grad is not None for p in params)
                norm=original_clip(params,*args,**kwargs)
                calls.append(float(norm))
                return norm
            class CheckedAdam(torch.optim.Adam):
                def step(self,*args,**kwargs):
                    assert calls
                    norm=torch.stack([p.grad.norm() for p in memory.parameters() if p.grad is not None]).norm()
                    assert norm <= 1.00001, norm
                    return super().step(*args,**kwargs)
            batch=(torch.randn(1,8,32),torch.full((1,8,32),10.),torch.zeros(1,8,3),
                   torch.zeros(1,8),torch.zeros(1,8))
            torch.nn.utils.clip_grad_norm_ = checked_clip
            try:
                scope['train_model'](memory,[batch],CheckedAdam(memory.parameters(),lr=1e-4),1,'cpu',save=False,path=str(out))
            finally:
                torch.nn.utils.clip_grad_norm_ = original_clip
            assert len(calls)==1 and calls[0]>1
            report['checks'].append(f'{source}: backward -> clipping -> update verified; preclip norm={calls[0]:.3f}')

        # Single 1000-frame chunk probes realistic sequence length; random pixels are only for memory sizing.
        data=out/'profile_data.h5'
        with h5py.File(data,'w') as f:
            f.create_dataset('images',data=np.zeros((1,1000,96,96,3),dtype=np.uint8),compression='lzf')
            f.create_dataset('actions',data=np.zeros((1,1000,3),dtype=np.float32))
            f.create_dataset('rewards',data=np.zeros((1,1000),dtype=np.float32))
            f.create_dataset('dones',data=np.zeros((1,1000),dtype=bool))
        device=torch.device('cuda')
        vision=VAE().to(device)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        optim=torch.optim.AdamW(vision.parameters(),lr=1e-3)
        images=torch.zeros(64,3,96,96,device=device)
        recon,mu,lv=vision(images)
        loss,*_=vision.loss(recon,images,mu,lv)
        loss.backward();optim.step();torch.cuda.synchronize()
        report['cuda_profile']['vae_batch64_peak_allocated_mib']=torch.cuda.max_memory_allocated()/1024**2
        del images,recon,mu,lv,loss,optim
        vision.zero_grad(set_to_none=True);vision.eval()
        definitions(REPO/'memory.py',['CarRacingDataset','train_model'],scope)
        dataset=scope['CarRacingDataset'](str(data),vae=vision,device=device,encode_batch_size=64)
        batches=[]
        original_encode=vision.encode
        def checked_encode(images):
            batches.append(len(images))
            assert len(images)<=64
            return original_encode(images)
        vision.encode=checked_encode
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        memory=MDNRNN(32,3,256,5).to(device)
        scope['train_model'](memory,DataLoader(dataset,batch_size=1,num_workers=0),
                             torch.optim.Adam(memory.parameters(),lr=1e-4),1,device,save=False,path=str(out))
        torch.cuda.synchronize()
        assert sum(batches)==1000 and len(batches)==16
        report['cuda_profile']['memory_1000frames_encode_batch64_peak_allocated_mib']=torch.cuda.max_memory_allocated()/1024**2
        report['cuda_profile']['gpu']=torch.cuda.get_device_name()
        report['cuda_profile']['total_vram_mib']=torch.cuda.get_device_properties(0).total_memory/1024**2
        dataset.h5_file.close();dataset.h5_file=None
        report['checks'].append('1000-frame Memory update succeeds with 16 bounded VAE encoding batches')
        # Batch size one must retain the batch axis in the original VAE trainer.
        definitions(REPO/'2-Vision (VAE).ipynb',['train_vae'],scope)
        scope['train_vae'](vision,[(torch.zeros(1,3,96,96),None,None,None)],
                            torch.optim.AdamW(vision.parameters(),lr=1e-3),num_epochs=1,folder=str(out/'batch1'))
        report['checks'].append('VAE training supports a final batch of size one')
        report['status']='passed'
    except Exception as exc:
        report.update(status='failed',error=repr(exc))
        raise
    finally:
        (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print('REPORT',out/'report.json',flush=True)
        print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
