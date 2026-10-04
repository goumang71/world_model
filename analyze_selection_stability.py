"""Paired track/noise stability analysis. No checkpoint selection."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np

def main():
 ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path);args=ap.parse_args();p=args.directory.resolve()
 config=json.loads((p/'config.json').read_text());state=json.loads((p/'status.json').read_text());assert state['status']=='completed'
 offsets=config['noise_offsets'];models=['original','candidate'];reports={};rows=[]
 for model in models:
  for offset in [None]+offsets:
   name=model+('_original_noise' if offset is None else f'_noise_{offset}')
   d=json.loads((p/name/'evaluation.json').read_text());reports[model,offset]=d
   assert not d['interrupted']
   expected=config['validation_seeds'] if offset is None else config['validation_seeds']+config['fresh_seeds']
   assert d['seeds']==expected
   for n in ['vae','memory','controller']:assert d['checkpoints'][n]['sha256']==config['models'][model][n]
   for e in d['episodes']:
    rows.append(dict(model=model,noise_offset=offset,latent_seed=e['seed'] if offset is None else e['seed']*1009+offset,**e))
 for offset in [None]+offsets:
  a=reports['original',offset];b=reports['candidate',offset]
  for k in ['environment','max_episode_steps','lap_complete_percent','domain_randomize','continuous','stochastic_z','torch','gymnasium','numpy','device','cudnn_deterministic','cudnn_benchmark']:assert a[k]==b[k],k
 rng=np.random.default_rng(20260930)
 def group(seeds,noise_values):
  matrices={};metrics={}
  for model in models:
   by={(r['seed'],r['noise_offset']):r for r in rows if r['model']==model}
   mat=np.array([[by[s,n]['episode_return'] for n in noise_values] for s in seeds]);matrices[model]=mat
   episodes=[by[s,n] for s in seeds for n in noise_values];total=sum(r['steps'] for r in episodes)
   metrics[model]=dict(mean=float(mat.mean()),std_over_episodes=float(mat.std(ddof=1)),minimum=float(mat.min()),
     mean_within_track_std=float(mat.std(axis=1,ddof=1).mean()) if len(noise_values)>1 else None,
     mean_within_track_range=float(np.ptp(mat,axis=1).mean()),below_500=int((mat<500).sum()),episodes=len(episodes),
     fully_off_fraction=sum(r['fully_off_track_steps'] for r in episodes)/total,
     any_wheel_off_fraction=sum(r['off_track_steps'] for r in episodes)/total)
  delta=matrices['candidate']-matrices['original'];track_delta=delta.mean(1)
  boot=rng.choice(track_delta,(20000,len(seeds)),replace=True).mean(1)
  tracks=[]
  for j,s in enumerate(seeds):
   tracks.append(dict(seed=s,original=matrices['original'][j].tolist(),candidate=matrices['candidate'][j].tolist(),
      differences=delta[j].tolist(),mean_difference=float(track_delta[j]),winner_flips=bool((delta[j]>1e-6).any() and (delta[j]<-1e-6).any())))
  # Rechoose one of the measured noise streams per fixed validation track.
  picks=rng.integers(len(noise_values),size=(20000,len(seeds)))
  repeated=delta[np.arange(len(seeds))[None,:],picks].mean(1)
  return dict(n_tracks=len(seeds),noise_replicates=len(noise_values),metrics=metrics,mean_difference=float(delta.mean()),
    track_cluster_bootstrap_95_interval=np.percentile(boot,[2.5,97.5]).tolist(),
    improved_pairs=int((delta>1e-6).sum()),worse_pairs=int((delta<-1e-6).sum()),tied_pairs=int((np.abs(delta)<=1e-6).sum()),
    tracks_with_winner_flip=sum(t['winner_flips'] for t in tracks),
    measured_noise_resampling_candidate_win_fraction=float((repeated>0).mean()),
    measured_noise_resampling_difference_95_range=np.percentile(repeated,[2.5,97.5]).tolist(),tracks=tracks)
 result=dict(original_validation=group(config['validation_seeds'],[None]),
   repeated_validation=group(config['validation_seeds'],offsets),fresh=group(config['fresh_seeds'],offsets),
   caveats=['Only three new noise streams per track.','Within-track repeated episodes are not independent tracks; bootstrap clusters by track.',
     'Noise-resampling win fraction is conditional on observed three streams, not a universal probability.',
     'Candidate fixed beforehand; no model training or checkpoint selection in this experiment.'])
 (p/'analysis.json').write_text(json.dumps(result,indent=2));(p/'all_episodes.json').write_text(json.dumps(rows,indent=2))
 lines=['# 选模稳定性：赛道与潜变量噪声分离','',
  '固定原始控制器和此前第10代候选，不训练、不重选模型。8条原验证赛道、16条新赛道，各测3组潜变量噪声；另复查原验证赛道的原始噪声。共160局，1000步/局。',
  '所有结果使用未修改的正式评估流程；独立脚本仅在内存中替换torch随机种子，环境种子与NumPy种子不变。配置、噪声种子公式、模型与源码哈希见config.json；逐局显式潜变量种子见all_episodes.json。',
  '', '|分组|原控制器均分|候选均分|候选减原模型|','|---|---:|---:|---:|']
 labels=[('original_validation','原验证8赛道×原噪声'),('repeated_validation','原验证8赛道×3组新噪声'),('fresh','新16赛道×3组新噪声')]
 for key,label in labels:
  g=result[key];lines.append(f"|{label}|{g['metrics']['original']['mean']:.2f}|{g['metrics']['candidate']['mean']:.2f}|{g['mean_difference']:+.2f}|")
 for key,label in labels[1:]:
  g=result[key];a=g['metrics']['original'];b=g['metrics']['candidate'];ci=g['track_cluster_bootstrap_95_interval']
  lines += ['',f'## {label}','',f"平均同赛道采样标准差：原模型{a['mean_within_track_std']:.2f}，候选{b['mean_within_track_std']:.2f}。",
   f"完全离道步数占比：原模型{a['fully_off_fraction']:.2%}，候选{b['fully_off_fraction']:.2%}。低于500分局数：原模型{a['below_500']}/{a['episodes']}，候选{b['below_500']}/{b['episodes']}。",
   f"同赛道换采样后新旧胜负翻转：{g['tracks_with_winner_flip']}/{g['n_tracks']}条。平均分差按赛道聚类bootstrap 95%区间[{ci[0]:.2f}, {ci[1]:.2f}]。",
   f"仅在已测3组噪声内，每条固定赛道任选一组重新组合，候选均分胜出的比例{g['measured_noise_resampling_candidate_win_fraction']:.1%}；这不是总体真实胜率。",
   '', '|赛道|旧模型3次分数|候选3次分数|平均差值|胜负翻转|','|---|---|---|---:|---|']
  for r in g['tracks']:
   old=', '.join(f'{x:.1f}' for x in r['original']);new=', '.join(f'{x:.1f}' for x in r['candidate'])
   lines.append(f"|{r['seed']}|{old}|{new}|{r['mean_difference']:+.1f}|{'是' if r['winner_flips'] else '否'}|")
 lines += ['', '## 限制','', '只有3次采样/赛道，不能精确估计尾部失败概率；新赛道16条，不能代替论文100条随机赛道复现。原验证赛道已参与候选选择，不能当成独立测试。检查可识别随机性与场景泛化不稳，但不能单独证明VAE、MDN或控制器结构哪个是唯一根因。']
 (p/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 tracks=result['fresh']['tracks'];fig,axes=plt.subplots(1,2,figsize=(9,9))
 for ax,model,title in zip(axes,models,['Original controller','Warm-start candidate']):
  values=np.array([r[model] for r in tracks]);im=ax.imshow(values,vmin=-100,vmax=1000,cmap='RdYlGn',aspect='auto')
  ax.set_xticks([0,1,2],['Noise A','Noise B','Noise C']);ax.set_yticks(np.arange(len(tracks)),[str(r['seed']) for r in tracks]);ax.set_title(title)
  for (i,j),value in np.ndenumerate(values):ax.text(j,i,f'{value:.0f}',ha='center',va='center',fontsize=9,color='black')
 fig.suptitle('Same track, different latent sampling (1000-step limit)');fig.tight_layout(rect=(0,0,1,.97));fig.savefig(p/'fresh_noise_scores.png',dpi=140);plt.close(fig)
 print(json.dumps({k:{kk:vv for kk,vv in v.items() if kk!='tracks'} for k,v in result.items() if k!='caveats'},indent=2))
if __name__=='__main__':main()
