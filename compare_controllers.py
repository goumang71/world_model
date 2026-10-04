"""Compare two evaluations only when seed sets and rollout settings match."""
import argparse
import json
from pathlib import Path
import statistics


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('baseline',type=Path)
    parser.add_argument('candidate',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    base=json.loads(args.baseline.read_text(encoding='utf-8'))
    candidate=json.loads(args.candidate.read_text(encoding='utf-8'))
    for key in ['environment','lap_complete_percent','domain_randomize','continuous','max_episode_steps',
                'seeds','stochastic_z','torch','gymnasium','numpy','device','cudnn_deterministic','cudnn_benchmark']:
        if base[key]!=candidate[key]: raise ValueError(f'Evaluation protocol differs: {key}')
    for name in ['vae','memory']:
        if base['checkpoints'][name]['sha256']!=candidate['checkpoints'][name]['sha256']:
            raise ValueError(f'Frozen {name} weights changed')
    old={r['seed']:r['episode_return'] for r in base['episodes']}
    new={r['seed']:r['episode_return'] for r in candidate['episodes']}
    if old.keys()!=new.keys(): raise ValueError('Episode seed sets differ')
    rows=[dict(seed=seed,baseline=old[seed],candidate=new[seed],difference=new[seed]-old[seed]) for seed in old]
    report=dict(baseline_mean=statistics.mean(old.values()),candidate_mean=statistics.mean(new.values()),
                mean_difference=statistics.mean(r['difference'] for r in rows),
                improved_tracks=sum(r['difference']>0 for r in rows),total_tracks=len(rows),
                episodes=rows,baseline_report=str(args.baseline.resolve()),candidate_report=str(args.candidate.resolve()))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    text=['# 新旧控制器固定赛道对比','',
          f"旧模型平均 {report['baseline_mean']:.2f}；新模型平均 {report['candidate_mean']:.2f}；差值 {report['mean_difference']:+.2f}。",'',
          '| 种子 | 旧控制器 | 新控制器 | 差值 |','|---|---:|---:|---:|']
    text.extend(f"| {r['seed']} | {r['baseline']:.2f} | {r['candidate']:.2f} | {r['difference']:+.2f} |" for r in rows)
    args.output.with_suffix('.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='episodes'},indent=2))


if __name__=='__main__':
    main()
