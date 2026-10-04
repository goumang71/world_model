"""Resumable single-factor action diagnosis; original evaluator/weights stay unchanged."""
import argparse, hashlib, json, sys, traceback
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
ROOT = Path(__file__).resolve().parent
VARIANTS = {
    'baseline': '',
    'smooth_05': 'action[:, 0] = 0.5 * action[:, 0] + 0.5 * previous_steering',
    'steer_gain_075': 'action[:, 0] *= 0.75',
    'gas_cap_075': 'action[:, 1].clamp_(max=0.75)',
    'brake_ungated': 'action[:, 2] = torch.sigmoid(controller.model(torch.cat([z, hidden[0].squeeze(0)], dim=-1))[:, 2]) * 0.8',
}

def write(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    tmp.replace(path)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output-dir', type=Path, default=ROOT/'outputs'/'action_cause_diagnosis_20261002')
    args = p.parse_args(); out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    source = (ROOT/'evaluate_baseline.py').read_text(encoding='utf-8')
    hashes = {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in
              [ROOT/'evaluate_baseline.py', Path(__file__), *sorted((ROOT/'World-Model-2018-main'/'checkpoints').glob('*.pth'))]}
    config = dict(seeds=[42,43,52,4470,60,61], variants=VARIANTS, hashes=hashes,
                  max_steps=1000, latent_seed='same as track seed', purpose='diagnosis, not random generalization test',
                  interventions='each independently before existing steering slew clamp; executed action fed into RNN')
    if (out/'config.json').exists():
        if json.loads((out/'config.json').read_text(encoding='utf-8')) != config:
            raise RuntimeError('Protocol/source changed; use a new output directory')
    else:
        write(out/'config.json',config)
        (out/'source_evaluate_baseline.py').write_text(source,encoding='utf-8')
        (out/'source_diagnose_action_causes.py').write_bytes(Path(__file__).read_bytes())
    complete=[]
    try:
        for variant, intervention in VARIANTS.items():
            for seed in config['seeds']:
                case=out/variant/f'seed_{seed}'
                if (case/'evaluation.json').exists():
                    complete.append(f'{variant}/{seed}'); continue
                if case.exists():
                    case.rename(case.with_name(case.name+'_partial_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
                write(out/'status.json',dict(status='running',current=f'{variant}/{seed}',completed=complete,total=30))
                text=source
                needle='                    action = action.clone()'
                assert text.count(needle)==1
                text=text.replace(needle, needle+'\n                    raw_action_for_trace = action[0].cpu().tolist()\n                    '+(intervention or 'pass'))
                needle='                        car = env.unwrapped.car'
                text=text.replace(needle, '                        if SAVE_FRAMES and step % 5 == 0: FRAMES.append(obs.copy())\n'+needle)
                needle='row_trace = dict(step=step+1, action=action[0].cpu().tolist(),'
                assert text.count(needle)==1
                text=text.replace(needle,needle+' raw_action=raw_action_for_trace,')
                ns={'__name__':'action_cause_evaluator','__file__':str(ROOT/'evaluate_baseline.py'), 'SAVE_FRAMES':variant=='baseline','FRAMES':[]}
                exec(compile(text,str(ROOT/'evaluate_baseline.py'),'exec'),ns)
                sys.argv=['evaluate_baseline.py','--seeds',str(seed),'--no-video','--record-trace','--output-dir',str(case)]
                ns['main']()
                if ns['FRAMES']: np.savez_compressed(case/'frames.npz',frames=np.asarray(ns['FRAMES']),steps=np.arange(len(ns['FRAMES']))*5+1)
                complete.append(f'{variant}/{seed}')
                write(out/'status.json',dict(status='running',completed=complete,total=30))
        # Confirm instrumentation has not changed canonical baseline trajectories.
        expected={42:(832.862190812706,24),43:(833.9933993399243,27),52:(726.666666666656,82)}
        for seed,(score,off) in expected.items():
            row=json.loads((out/'baseline'/f'seed_{seed}'/'evaluation.json').read_text())['episodes'][0]
            assert abs(row['episode_return']-score)<1e-8 and row['fully_off_track_steps']==off, (seed,row)
        write(out/'status.json',dict(status='completed',completed=complete,total=30,baseline_parity=True))
    except BaseException:
        write(out/'status.json',dict(status='interrupted',completed=complete,total=30,error=traceback.format_exc()))
        raise
if __name__=='__main__': main()
