"""Read-only ablation of controller mapping using the current evaluator in memory."""
import sys, json, hashlib
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
ROOT = Path(__file__).resolve().parent
source = (ROOT / 'evaluate_baseline.py').read_text(encoding='utf-8')
assert source.count('max_steering_change = 1.0') == 1
out = ROOT / 'outputs' / ('controller_action_diagnosis_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
out.mkdir()
summary = []
for coupled, limit in [(True,1.0),(False,1.0),(True,2.0),(False,2.0)]:
    name = f'coupled_{int(coupled)}_steer_{limit}'
    modified = source.replace('max_steering_change = 1.0', f'max_steering_change = {limit}')
    ns = {'__name__':'diagnostic_evaluator', '__file__':str(ROOT / 'evaluate_baseline.py')}
    exec(compile(modified, str(ROOT / 'evaluate_baseline.py'), 'exec'), ns)
    Base = ns['Controller']
    samples = []
    class ProbeController(Base):
        def forward(self, x):
            raw = self.model(x)
            steer = torch.tanh(raw[:,0:1])
            gas = torch.sigmoid(raw[:,1:2])
            base_brake = torch.sigmoid(raw[:,2:3]) * 0.8
            brake = base_brake * (1-gas) if coupled else base_brake
            samples.append([float(gas.item()), float(base_brake.item()), float(brake.item())])
            return torch.cat([steer,gas,brake],dim=1)
    ns['Controller'] = ProbeController
    sys.argv = ['evaluate_baseline.py','--seeds','42','43','52','--no-video','--output-dir',str(out / name)]
    ns['main']()
    report = json.loads((out / name / 'evaluation.json').read_text())
    a = np.asarray(samples)
    row = dict(variant=name, coupled=coupled, max_steering_change=limit,
               mean_return=report['mean_return'], episodes=report['episodes'],
               gas_mean=float(a[:,0].mean()), ungated_brake_mean=float(a[:,1].mean()),
               executed_brake_mean=float(a[:,2].mean()),
               brake_above_point1_fraction=float((a[:,2]>.1).mean()),
               suppressed_brake_fraction=float(((a[:,1]>.1)&(a[:,2]<.01)).mean()))
    summary.append(row)
    (out / 'comparison.json').write_text(json.dumps(dict(seeds=[42,43,52],
        evaluator_sha256=hashlib.sha256(source.encode()).hexdigest(), results=summary),indent=2))
print('DIAGNOSIS_OUTPUT',out,flush=True)
