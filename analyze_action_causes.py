"""Summarize saved action ablations, including pre-departure windows."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs'/'action_cause_diagnosis_20261002'

def stats(rows):
    if not rows: return None
    a=np.array([r['action'] for r in rows]); raw=np.array([r['raw_action'] for r in rows])
    std=np.array([r['raw_steering_sampling_std'] for r in rows])
    # mu = realized z contribution minus sampled deviation is unavailable;
    # same-history mean-z steering directly measures the mean raw logit's sign.
    return dict(n=len(rows),speed_mean=float(np.mean([r['speed'] for r in rows])),speed_max=float(max(r['speed'] for r in rows)),
        gas_mean=float(a[:,1].mean()),brake_mean=float(a[:,2].mean()),brake_over_01_fraction=float((a[:,2]>.1).mean()),
        raw_saturated_fraction=float((abs(raw[:,0])>.95).mean()),executed_saturated_fraction=float((abs(a[:,0])>.95).mean()),
        large_opposite_commands=int(np.sum((a[1:,0]*a[:-1,0]<0)&(abs(a[1:,0]-a[:-1,0])>.8))),
        raw_large_opposite_commands=int(np.sum((raw[1:,0]*raw[:-1,0]<0)&(abs(raw[1:,0]-raw[:-1,0])>1.5))),
        steering_total_variation=float(np.abs(np.diff(a[:,0])).sum()),raw_sampling_std_mean=float(std.mean()),
        sampled_vs_mean_z_opposite_fraction=float(np.mean([r['raw_action'][0]*r['mean_z_action_same_history'][0]<0 for r in rows])),
        straight_proxy_fraction=float(np.mean([r['straight_proxy'] for r in rows])))

def main():
    config=json.loads((OUT/'config.json').read_text(encoding='utf-8'))
    results=[]; traces={}
    for variant in config['variants']:
        for seed in config['seeds']:
            folder=OUT/variant/f'seed_{seed}'
            if not (folder/'evaluation.json').exists():continue
            row=json.loads((folder/'evaluation.json').read_text())['episodes'][0]
            trace=json.loads((folder/f'trace_seed{seed}.json').read_text());traces[variant,seed]=trace
            off=np.array([r['road_wheels_after']==0 for r in trace]); events=[];i=0
            while i<len(off):
                if not off[i]:i+=1;continue
                start=i
                while i<len(off) and off[i]:i+=1
                if i-start>=10 and any(r['road_wheels_after']>0 for r in trace[:start]):
                    # End pre-window at first fully-off state; include only states still contacting road.
                    pre=trace[max(0,start-50):start]
                    events.append(dict(start_step=start+1,duration=i-start,pre50=stats(pre),
                        pre50_contact_only=stats([r for r in pre if r['road_wheels_before']>0])))
            first_full = next((j for j in range(1,len(trace)) if off[j] and any(r['road_wheels_after']>0 for r in trace[:j])), None)
            first_evidence = None if first_full is None else dict(step=first_full+1, pre50=stats(trace[max(0,first_full-50):first_full]))
            row.update(first_full_departure=first_evidence,variant=variant,fully_off_fraction=row['fully_off_track_steps']/row['steps'],events=events,
                       contact_only=stats([r for r in trace if r['road_wheels_before']>0]))
            results.append(row)
    summary=[]
    for variant in config['variants']:
        rows=[r for r in results if r['variant']==variant]
        if not rows:continue
        summary.append(dict(variant=variant,episodes=len(rows),mean_return=float(np.mean([r['episode_return'] for r in rows])),
            fully_off_fraction=sum(r['fully_off_track_steps'] for r in rows)/sum(r['steps'] for r in rows),
            sustained_departures=sum(len(r['events']) for r in rows),
            severe_failures=sum(r['episode_return']<500 or r['fully_off_fraction']>.2 or (r['terminated'] and not r['lap_finished']) for r in rows)))
    (OUT/'analysis.json').write_text(json.dumps(dict(summary=summary,episodes=results),indent=2),encoding='utf-8')
    lines=['# Action cause diagnosis','', 'Six prespecified diagnostic tracks; one latent-noise stream per track. Not a random generalization estimate.', '',
        '|Variant|Episodes|Mean return|Fully off-road steps|Severe failures|Sustained departures|', '|---|---:|---:|---:|---:|---:|']
    for r in summary:lines.append(f"|{r['variant']}|{r['episodes']}|{r['mean_return']:.2f}|{r['fully_off_fraction']:.2%}|{r['severe_failures']}|{r['sustained_departures']}|")
    lines+=['','## Paired returns','', '|Seed|'+'|'.join(config['variants'])+'|','|---|'+'---:|'*len(config['variants'])]
    for seed in config['seeds']:
        lines.append('|'+str(seed)+'|'+ '|'.join(next((f"{r['episode_return']:.2f}" for r in results if r['seed']==seed and r['variant']==v),'pending') for v in config['variants'])+'|')
    lines+=['','## Baseline pre-departure evidence','', 'Each window includes at most 50 preceding frames with wheel contact, before the first sustained departure (zero wheels for >=10 consecutive steps, excluding initial reset). Speed is in simulator units. These are descriptive associations, not proof of excess speed or noise causality.','',
        '|Seed|First sustained off step|Duration|Contact frames in pre-window|Mean speed|Mean gas|Mean brake|Raw large reversals|Executed large reversals|Sample/mean-z sign disagreement|', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in results:
        if r['variant']!='baseline' or not r['events']:continue
        e=r['events'][0];s=e['pre50_contact_only']
        if not s:continue
        lines.append(f"|{r['seed']}|{e['start_step']}|{e['duration']}|{s['n']}|{s['speed_mean']:.2f}|{s['gas_mean']:.3f}|{s['brake_mean']:.5f}|{s['raw_large_opposite_commands']}|{s['large_opposite_commands']}|{s['sampled_vs_mean_z_opposite_fraction']:.1%}|")
    lines+=['','## First full departure, including brief recoverable excursions','',
        '|Seed|Step|Pre-window mean speed|Mean gas|Mean brake|Raw reversals|Executed reversals|Sample/mean-z sign disagreement|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in results:
        if r['variant']!='baseline' or not r['first_full_departure']:continue
        e=r['first_full_departure'];s=e['pre50']
        lines.append(f"|{r['seed']}|{e['step']}|{s['speed_mean']:.2f}|{s['gas_mean']:.3f}|{s['brake_mean']:.6f}|{s['raw_large_opposite_commands']}|{s['large_opposite_commands']}|{s['sampled_vs_mean_z_opposite_fraction']:.1%}|")
    lines+=['','## Limits','', 'Interventions alter future images and hidden states. Matched RNG seeds match noise draws, not states after trajectories diverge. Gas cap may reduce distance reached within 1000 steps. Steering smoothing adds lag and can hurt tight turns. No reward, world-model weights, or default evaluator were changed. The brake-ungated probe is diagnostic, not an endorsed action mapping.','', 'analysis.json contains every sustained event and both complete/contact-only pre-windows. Raw action is captured before interventions and steering slew limiting. Large reversal thresholds: raw delta>1.5, executed delta>0.8, both require a sign change. Contact-only windows can have gaps.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    for seed in [52,4470]:
        rows=traces.get(('baseline',seed)); base=next((r for r in results if r['variant']=='baseline' and r['seed']==seed),None)
        if not rows or not base['events']:continue
        event=base['events'][0]; t=event['start_step']; segment=[r for r in rows if max(1,t-70)<=r['step']<=t+40]
        x=[r['step'] for r in segment];a=np.array([r['action'] for r in segment]);raw=np.array([r['raw_action'] for r in segment])
        fig,axs=plt.subplots(4,1,figsize=(11,9),sharex=True)
        axs[0].plot(x,raw[:,0],alpha=.55,label='raw steer');axs[0].plot(x,a[:,0],label='executed steer');axs[0].legend()
        axs[1].plot(x,a[:,1],label='gas');axs[1].plot(x,a[:,2],label='brake');axs[1].legend()
        axs[2].plot(x,[r['speed'] for r in segment]);axs[2].set_ylabel('Speed (sim units)')
        axs[3].step(x,[r['road_wheels_after'] for r in segment],where='post');axs[3].set_ylabel('Road wheels');axs[3].set_xlabel('Step')
        for ax in axs:ax.axvline(t,color='red',linestyle='--');ax.grid(alpha=.2)
        fig.suptitle(f'Seed {seed}: first sustained departure at step {t}');fig.tight_layout();fig.savefig(OUT/f'event_{seed}.png');plt.close(fig)
        path=OUT/'baseline'/f'seed_{seed}'/'frames.npz'
        if path.exists():
            data=np.load(path);targets=[t-50,t-30,t-10,t,t+10,t+30];fig,axs=plt.subplots(1,6,figsize=(14,3))
            for ax,target in zip(axs,targets):
                j=int(np.argmin(abs(data['steps']-target)));ax.imshow(data['frames'][j]);ax.set_title(f"Step {data['steps'][j]}");ax.axis('off')
            fig.tight_layout();fig.savefig(OUT/f'frames_{seed}.png');plt.close(fig)
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
