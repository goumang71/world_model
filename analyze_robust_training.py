"""Report actual results of the multi-noise training run."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np

def main():
 ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path);args=ap.parse_args();p=args.directory.resolve()
 state=json.loads((p/'status.json').read_text());cfg=json.loads((p/'config.json').read_text());d=json.loads((p/'comparison.json').read_text())
 assert state['status']=='completed' and state['source_weights_unchanged']
 old=d['original_rows'];new=d['candidate_rows'];expected=cfg['test_cases']
 assert len(old)==len(new)==len(expected)==48
 for a,b,c in zip(old,new,expected):assert (a['seed'],a['latent_seed'])==(b['seed'],b['latent_seed'])==(c['seed'],c['latent_seed'])
 assert not {r['seed'] for r in old}&set(cfg['training_seeds'])
 assert not {r['seed'] for r in old}&{r['seed'] for r in cfg['validation_cases']}
 root=Path(__file__).resolve().parent
 for n in ['vae','memory','controller']:assert hashlib.sha256((root/'World-Model-2018-main/checkpoints'/f'{n}.pth').read_bytes()).hexdigest()==cfg['source_sha256'][n]
 assert (p/'source_evaluate_baseline.py').read_bytes()==(root/'evaluate_baseline.py').read_bytes()
 def stats(rows):
  scores=np.array([r['episode_return'] for r in rows]);matrix=scores.reshape(16,3)
  return dict(mean=float(scores.mean()),std=float(scores.std(ddof=1)),minimum=float(scores.min()),
    mean_within_track_std=float(matrix.std(1,ddof=1).mean()),severe_failures=sum(r['severe_failure'] for r in rows),
    fully_off_fraction=sum(r['fully_off_track_steps'] for r in rows)/sum(r['steps'] for r in rows),
    boundary_exits=sum(r['boundary_exit'] for r in rows))
 a=stats(old);b=stats(new);out=dict(original=a,candidate=b,mean_difference=d['mean_difference'],
   cluster_bootstrap_95_interval=d['cluster_bootstrap_95_interval'],improved_pairs=d['improved_pairs'],
   worse_pairs=d['worse_pairs'],tied_pairs=d['tied_pairs'],selected_generation=d['selected_generation'],
   tested_generation=d['tested_generation'],diagnostic_only=d['diagnostic_only'])
 (p/'report_summary.json').write_text(json.dumps(out,indent=2))
 lines=['# 多赛道、多采样控制器训练：实际结果','',
  f"从下载的原控制器附近训练{cfg['generations']}代，每代16候选×{cfg['tracks_per_generation']}条赛道×{cfg['noise_repeats']}组采样，共{cfg['training_episodes']}局。冻结VAE/MDN，保留环境奖励和原动作映射。",'',
  '验证为32赛道×2组噪声；最终测试16条独立赛道×3组噪声。候选必须均分提高且严重失败次数、完全离道比例均不增加，才能替换验证基准。验证和最终测试采用已通过正式评估一致性检查的单环境神经网络推理。', '',
  '|验证检查点|均分|严重失败/64|完全离道占比|通过门槛|','|---|---:|---:|---:|---|']
 base=state['baseline_validation'];lines.append(f"|原模型|{base['mean_return']:.2f}|{base['severe_failures']}|{base['fully_off_fraction']:.2%}|基准|")
 for row in state['history']:
  if 'validation' in row:
   v=row['validation'];lines.append(f"|第{row['generation']}代|{v['mean_return']:.2f}|{v['severe_failures']}|{v['fully_off_fraction']:.2%}|{'是' if row['accepted'] else '否'}|")
 lines += ['',f"最终通过验证的代数：{d['selected_generation']}（0表示保留原模型）。独立测试的训练候选：第{d['tested_generation']}代。",'']
 if d['diagnostic_only']:lines.append('没有训练候选通过验证门槛；下表测试的是验证均分最高的候选，仅作诊断，checkpoints/controller.pth仍是原权重。')
 lines += ['', '|独立测试指标|原控制器|训练候选|','|---|---:|---:|',
  f"|均分|{a['mean']:.2f}|{b['mean']:.2f}|",f"|平均同赛道采样标准差|{a['mean_within_track_std']:.2f}|{b['mean_within_track_std']:.2f}|",
  f"|最低分|{a['minimum']:.2f}|{b['minimum']:.2f}|",f"|严重失败/48|{a['severe_failures']}|{b['severe_failures']}|",
  f"|完全离道步数占比|{a['fully_off_fraction']:.2%}|{b['fully_off_fraction']:.2%}|",'',
  f"成对改善{d['improved_pairs']}局、下降{d['worse_pairs']}局、持平{d['tied_pairs']}局；均分差{d['mean_difference']:+.2f}。按赛道聚类bootstrap 95%区间{d['cluster_bootstrap_95_interval']}。",'',
  '严重失败预定义为得分<500、四轮完全离道>20%步数或跑出场地边界。只测了16条独立赛道、每条3次，不能精确估计尾部失败率或声称论文900分已复现。训练代数少且只检查定期训练优胜者，不能认为已找到最优控制器。',
  '这轮与旧训练方法没有在相同优化预算和同一测试组上同时重新训练，因此不能由一次新旧权重对照证明方法普遍优越。源权重与正式评估脚本未被覆盖。',
  '', '|赛道|采样种子|原分数|候选分数|原完全离道步数|候选完全离道步数|','|---|---|---:|---:|---:|---:|']
 for o,n in zip(old,new):lines.append(f"|{o['seed']}|{o['latent_seed']}|{o['episode_return']:.2f}|{n['episode_return']:.2f}|{o['fully_off_track_steps']}|{n['fully_off_track_steps']}|")
 (p/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 x=np.arange(16);va=np.array([r['episode_return'] for r in old]).reshape(16,3);vb=np.array([r['episode_return'] for r in new]).reshape(16,3)
 fig,ax=plt.subplots(figsize=(10,4))
 ax.errorbar(x-.1,va.mean(1),yerr=va.std(1,ddof=1),fmt='o',capsize=3,label='Original')
 ax.errorbar(x+.1,vb.mean(1),yerr=vb.std(1,ddof=1),fmt='o',capsize=3,label='Trained candidate')
 ax.set(xlabel='Track index (seed 70000 + index)',ylabel='Score',title='Three latent-noise runs per track; bars = sample standard deviation');ax.legend();fig.tight_layout();fig.savefig(p/'comparison.png',dpi=150);plt.close(fig)
 print(json.dumps(out,indent=2))
if __name__=='__main__':main()
