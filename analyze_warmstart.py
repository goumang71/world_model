"""Summarize canonical, paired warm-start evaluation after training finishes."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np

def main():
 ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path);args=ap.parse_args();p=args.directory.resolve()
 base=json.loads((p/'canonical_baseline/evaluation.json').read_text());new=json.loads((p/'canonical_candidate/evaluation.json').read_text())
 for k in ['environment','max_episode_steps','seeds','stochastic_z','lap_complete_percent','domain_randomize','continuous','torch','gymnasium','numpy','device','cudnn_deterministic','cudnn_benchmark']:
  assert base[k]==new[k],k
 for name in ['vae','memory']:assert base['checkpoints'][name]['sha256']==new['checkpoints'][name]['sha256']
 config=json.loads((p/'config.json').read_text());assert base['seeds']==config['final_test_seeds']
 assert not set(base['seeds']) & set(config['training_seeds']+config['selection_seeds'])
 root=Path(__file__).resolve().parent
 assert hashlib.sha256((p/'source_evaluate_baseline.py').read_bytes()).hexdigest()==hashlib.sha256((root/'evaluate_baseline.py').read_bytes()).hexdigest()
 rows=[]
 for b,n in zip(base['episodes'],new['episodes']):
  assert b['seed']==n['seed']
  rows.append(dict(seed=b['seed'],baseline=b['episode_return'],candidate=n['episode_return'],difference=n['episode_return']-b['episode_return'],
     baseline_fully_off=b['fully_off_track_steps'],candidate_fully_off=n['fully_off_track_steps']))
 diffs=np.array([r['difference'] for r in rows]);rng=np.random.default_rng(20260930)
 ci=np.percentile(rng.choice(diffs,(20000,len(diffs)),replace=True).mean(1),[2.5,97.5]).tolist()
 def metrics(report):
  r=report['episodes'];scores=np.array([x['episode_return'] for x in r]);steps=sum(x['steps'] for x in r)
  return dict(mean=float(scores.mean()),std=float(scores.std(ddof=1)),median=float(np.median(scores)),minimum=float(scores.min()),
     below_500=int((scores<500).sum()),fully_off_fraction=sum(x['fully_off_track_steps'] for x in r)/steps,
     any_wheel_off_fraction=sum(x['off_track_steps'] for x in r)/steps,mean_coverage=float(np.mean([x['track_coverage'] for x in r])),
     environment_lap_finished=sum(x['lap_finished'] for x in r))
 state=json.loads((p/'status.json').read_text())
 result=dict(n=len(rows),baseline=metrics(base),candidate=metrics(new),mean_difference=float(diffs.mean()),
   improved=int((diffs>1e-6).sum()),worse=int((diffs<-1e-6).sum()),tied=int((np.abs(diffs)<=1e-6).sum()),
   paired_bootstrap_95_interval=ci,selected_generation=state['selected_generation'],rows=rows,
   caution='24 prespecified seeds only; bootstrap interval assumes exchangeable episodes. No claim of 100-track or paper reproduction.')
 (p/'canonical_comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
 b=result['baseline'];n=result['candidate']
 lines=['# 预训练控制器附近继续优化：正式复评','',
  f"第{state['selected_generation']}代由8条验证赛道选中。正式评估使用24条事先固定、未参与训练或选择的赛道。",'',
  '|指标|旧控制器|候选控制器|','|---|---:|---:|',
  f"|平均分|{b['mean']:.2f}|{n['mean']:.2f}|",f"|样本标准差|{b['std']:.2f}|{n['std']:.2f}|",
  f"|最低分|{b['minimum']:.2f}|{n['minimum']:.2f}|",f"|低于500分局数|{b['below_500']}|{n['below_500']}|",
  f"|四轮完全离道步数占比|{b['fully_off_fraction']:.2%}|{n['fully_off_fraction']:.2%}|",
  f"|至少一轮离道步数占比|{b['any_wheel_off_fraction']:.2%}|{n['any_wheel_off_fraction']:.2%}|",
  f"|平均已访问赛道块比例|{b['mean_coverage']:.2%}|{n['mean_coverage']:.2%}|",'',
  f"平均差值{result['mean_difference']:+.2f}；改善{result['improved']}局、下降{result['worse']}局、持平{result['tied']}局。成对bootstrap 95%区间约[{ci[0]:.2f}, {ci[1]:.2f}]；仅24局，不能承诺泛化或达到论文900分。",'',
  '## 设置与限制','',
  '冻结VAE/MDN，仅从已有867参数控制器附近启动新的CMA-ES。20代、16候选、每候选3局、sigma初始0.05。原奖励、采样z、bicubic、转向变化上限1.0及原油门刹车映射保持一致；每局1000步。',
  '训练10000起、验证20000..20007、最终测试30000..30023。最终结果来自evaluate_baseline.py；批量训练器输出另见comparison.json。批量与单局浮点差异可导致不同轨迹，不能混报。',
  '未覆盖原始权重，默认run_baseline.py仍使用旧模型。coverage不是几何完圈率。',
  '单局训练评估与正式评估seed42一致性验证通过。冻结权重哈希一致，评估源文件与启动时快照一致。',
  '', '|seed|旧分数|新分数|差值|旧完全离道步数|新完全离道步数|','|---|---:|---:|---:|---:|---:|']
 lines += [f"|{r['seed']}|{r['baseline']:.2f}|{r['candidate']:.2f}|{r['difference']:+.2f}|{r['baseline_fully_off']}|{r['candidate_fully_off']}|" for r in rows]
 (p/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(1,2,figsize=(11,4))
 old=np.array([r['baseline'] for r in rows]);fresh=np.array([r['candidate'] for r in rows])
 ax[0].scatter(old,fresh);ax[0].plot([-100,1000],[-100,1000],'k--',lw=1);ax[0].set(xlabel='Original score',ylabel='Warm-start score',title='24 paired tracks; above line = improved')
 ax[1].bar(np.arange(len(rows)),diffs,color=['#2b8a6e' if d>=0 else '#c45850' for d in diffs]);ax[1].axhline(0,color='black',lw=.8)
 ax[1].set(xlabel='Track index (seed 30000 + index)',ylabel='Score difference',title=f'Mean difference {diffs.mean():+.1f}')
 fig.tight_layout();fig.savefig(p/'comparison.png',dpi=150);plt.close(fig)
 print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
