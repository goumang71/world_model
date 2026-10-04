"""Paired rollout diagnostics; metrics describe policy behavior, not model accuracy."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent


def summarize(folder):
    report=json.loads((folder/'evaluation.json').read_text())
    episodes=[]
    for episode in report['episodes']:
        trace=json.loads((folder/f'trace_seed{episode["seed"]}.json').read_text())
        actions=np.array([r['action'] for r in trace])
        alternate=np.array([r['mean_z_action_same_history'] for r in trace])
        steer=actions[:,0]
        raw=np.array([r['raw_steering_z']+r['raw_steering_h']+r['raw_steering_bias'] for r in trace])
        # Separate float32 dot products differ slightly from the fused Linear.
        assert np.allclose(np.tanh(raw),steer,atol=1e-5)
        mask=np.array([r['straight_proxy'] and r['road_wheels_before']==4 and r['speed']>1 and r['step']>50 for r in trace])
        pairs=mask[1:] & mask[:-1]
        changes=np.abs(np.diff(steer))
        flips=(steer[1:]*steer[:-1]<0)&(np.abs(steer[1:])>=.05)&(np.abs(steer[:-1])>=.05)
        probes=np.array([r['fixed_state_action_std'] for r in trace if 'fixed_state_action_std' in r])
        row=dict(episode,mean_abs_steering_change=float(changes.mean()),
                 steering_saturation_fraction=float(np.mean(np.abs(steer)>.95)),
                 strong_sign_flips=int(flips.sum()),
                 straight_pair_count=int(pairs.sum()),
                 straight_abs_delta_sum=float(changes[pairs].sum()),
                 straight_strong_flips=int(flips[pairs].sum()),
                 direct_sampling_action_difference=float(np.abs(actions-alternate).mean(axis=0)[0]),
                 fixed_state_steering_std=float(probes[:,0].mean()),
                 raw_sampling_std=float(np.mean([r['raw_steering_sampling_std'] for r in trace])))
        episodes.append(row)
    count=sum(r['straight_pair_count'] for r in episodes)
    summary=dict(mean_return=report['mean_return'],lap_finished_count=report['lap_finished_count'],
                 mean_coverage=float(np.mean([r['track_coverage'] for r in episodes])),
                 off_track_fraction=float(np.mean([r['off_track_fraction'] for r in episodes])),
                 fully_off_track_fraction=float(np.mean([r['fully_off_track_steps']/r['steps'] for r in episodes])),
                 mean_abs_steering_change=float(np.mean([r['mean_abs_steering_change'] for r in episodes])),
                 steering_saturation_fraction=float(np.mean([r['steering_saturation_fraction'] for r in episodes])),
                 strong_sign_flips_per_episode=float(np.mean([r['strong_sign_flips'] for r in episodes])),
                 straight_pair_count=count,
                 straight_mean_abs_steering_change=sum(r['straight_abs_delta_sum'] for r in episodes)/count if count else None,
                 straight_strong_flips_per_1000_pairs=1000*sum(r['straight_strong_flips'] for r in episodes)/count if count else None,
                 fixed_state_steering_std=float(np.mean([r['fixed_state_steering_std'] for r in episodes])),
                 direct_sampling_action_difference=float(np.mean([r['direct_sampling_action_difference'] for r in episodes])),
                 episodes=episodes)
    return report,summary


def main():
    out=ROOT/'outputs/latent_diagnosis';out.mkdir(exist_ok=True)
    sample_report,sample=summarize(ROOT/'outputs/latent_diagnosis_sample')
    mean_report,mean=summarize(ROOT/'outputs/latent_diagnosis_mean')
    for key in ['seeds','checkpoints','environment','max_episode_steps','lap_complete_percent','torch','gymnasium']:
        assert sample_report[key]==mean_report[key],key
    original=json.loads((ROOT/'outputs/baseline_fixed_42_61/evaluation.json').read_text())
    assert all(abs(a['episode_return']-b['episode_return'])<1e-7 for a,b in zip(sample_report['episodes'],original['episodes']))
    result=dict(sample=sample,mean=mean,
                mean_improved_tracks=sum(b['episode_return']>a['episode_return'] for a,b in zip(sample['episodes'],mean['episodes'])),
                limitations=['One sampling stream per track; not a multi-noise-seed significance study',
                             'Trajectories differ after actions diverge; straight masks are geometric proxies',
                             'Fixed-state probes isolate direct z noise, not historical noise in h',
                             'No VAE reconstruction or MDN prediction accuracy benchmark is performed'])
    (out/'comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(12,10),layout='constrained')
    seeds=[e['seed'] for e in sample['episodes']]
    axes[0].plot(seeds,[e['track_coverage']*100 for e in sample['episodes']],'o-',label='Sample z',color='#dd6644')
    axes[0].plot(seeds,[e['track_coverage']*100 for e in mean['episodes']],'o-',label='Mean z',color='#2277bb')
    axes[0].set(ylabel='Track coverage (%)',xlabel='Track seed',title='Identical pretrained weights; 1000-step limit');axes[0].legend()
    for ax,seed in zip(axes[1:],[42,43]):
        for mode,color in [('sample','#dd6644'),('mean','#2277bb')]:
            trace=json.loads((ROOT/f'outputs/latent_diagnosis_{mode}/trace_seed{seed}.json').read_text())
            ax.plot([r['step'] for r in trace],[r['action'][0] for r in trace],label=mode,color=color,alpha=.75,linewidth=.7)
        ax.set(xlabel='Environment step',ylabel='Steering command',title=f'Seed {seed}: each policy follows its own trajectory',ylim=(-1.05,1.05));ax.legend()
    fig.savefig(out/'comparison.png',dpi=150);plt.close(fig)
    for mode,summary in [('sample',sample),('mean',mean)]:
        print(mode,json.dumps({k:v for k,v in summary.items() if k!='episodes'},indent=2))
    print('mean_improved_tracks',result['mean_improved_tracks'])


if __name__=='__main__':
    main()
