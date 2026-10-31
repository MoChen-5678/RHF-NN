#!/usr/bin/env python3
"""Read-only continuous comparison of two independently monitored searches."""
import argparse, datetime as dt, hashlib, json, os, subprocess, time, traceback
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path('/home/ubuntu/rhf/DFT-RHF-NoRefresh')
LOCAL=Path('/home/ubuntu/rhf/demo-meta-pcn/runtime/monitor-local-20261006')
REMOTE=ROOT/'runtime/monitor-remote-20261006'
OUT=ROOT/'runtime/monitor-combined-20261006'
NAMES=['m_sigma','g_sigma','g_omega','g_rho','kappa_rho','f_pi','a_rho','a_rho_tensor','a_pi']
REMOTE_ROOT='/home/mayx/demo-meta-pcn/runtime/monitor-remote-20261006'
def read(p): return json.loads(p.read_text())
def write(p,d):
 tmp=p.with_suffix(p.suffix+'.tmp'); tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)); tmp.replace(p)
def sync():
 REMOTE.mkdir(parents=True,exist_ok=True)
 args=['scp','-q','-o','ControlPath=/tmp/rhf-ca.sock','-o','ConnectTimeout=10','-P','7789']
 for name in ['snapshot.json','curve_data.json']:
  temp=REMOTE/(name+'.incoming')
  subprocess.run(args+[f'mayx@127.0.0.1:{REMOTE_ROOT}/{name}',str(temp)],check=True,timeout=25)
  read(temp); temp.replace(REMOTE/name)
def save(fig,name):
 fig.tight_layout(); fig.savefig(OUT/(name+'.png'),dpi=180); fig.savefig(OUT/(name+'.pdf')); plt.close(fig)
def best_records(records):
 best=float('inf'); selected=[]; current=None
 for r in records:
  if r['x2']<best: best=r['x2']; current=r
  selected.append(current)
 return selected

def render(curves, snapshots):
 fig,axes=plt.subplots(2,2,figsize=(13,9))
 for host, color in [('local','tab:blue'),('7789','tab:orange')]:
  task=next(t for t in curves[host]['tasks'] if t['mode']=='single'); rr=task['evaluations']
  if not rr: continue
  x=np.arange(1,len(rr)+1); best=best_records(rr)
  axes[0,0].plot(x,[r['x2'] for r in best],label=host,color=color)
  axes[0,0].scatter(x,[r['x2'] for r in rr],s=8,color=color,alpha=.3)
  stamps=[dt.datetime.fromisoformat(r['utc']) for r in rr]
  start=stamps[0]
  snap=next(t for t in snapshots[host]['tasks'] if t['mode']=='single')
  if snap.get('run_created_utc'): start=dt.datetime.fromisoformat(snap['run_created_utc'])
  hours=[(t-start).total_seconds()/3600 for t in stamps]
  axes[0,1].plot(hours,[r['x2'] for r in best],label=host,color=color)
  for ax, key in [(axes[1,0],'target'),(axes[1,1],'raw_target')]:
   vals=[]
   for r in best:
    obs=next((o for o in r.get('selected_observations',[]) if o.get('observable')=='binding_energy_per_a'),None)
    vals.append((obs['prediction']-obs.get(key,obs['target']))*obs['A']/1000 if obs else np.nan)
   ax.plot(x,vals,label=host,color=color)
 for ax in axes.flat: ax.grid(alpha=.25); ax.legend(); ax.set_xlabel('Completed physical parameter evaluations')
 for ax in axes[0]: ax.set_yscale('log'); ax.axhline(1,color='gray',ls=':'); ax.set_ylabel('Best weighted X² (candidates: dots)')
 axes[0,1].set_xlabel('Hours from run start (first evaluation if unavailable)')
 axes[1,0].set_ylabel('Total binding error vs adjusted target (MeV)')
 axes[1,1].set_ylabel('Total binding error vs raw experiment (MeV)')
 fig.suptitle('Pb208: local vs 7789 | target shift = -2 MeV / nucleus')
 save(fig,'pb208-comparison')
 for relative in [False,True]:
  fig,axes=plt.subplots(3,3,figsize=(14,10))
  for host,color in [('local','tab:blue'),('7789','tab:orange')]:
   rr=next(t for t in curves[host]['tasks'] if t['mode']=='single')['evaluations']
   if not rr: continue
   arr=np.array([r['parameters'] for r in best_records(rr)]); base=np.array(rr[0]['parameters'])
   if relative: arr=(arr-base)/np.maximum(abs(base),1e-30)*100
   for j,ax in enumerate(axes.flat): ax.plot(np.arange(1,len(rr)+1),arr[:,j],label=host,color=color); ax.set_title(NAMES[j]); ax.grid(alpha=.25); ax.ticklabel_format(useOffset=False,axis='y'); ax.set_xlabel('Completed evaluations'); ax.set_ylabel('Change from initial (%)' if relative else 'Best parameter value'); ax.legend()
  fig.suptitle('Pb208 best physical parameter history'+(' (relative)' if relative else ' (raw)')); save(fig,'parameters-relative' if relative else 'parameters-raw')
 # Partial observables remain distinct from a complete joint objective.
 fig,axes=plt.subplots(1,2,figsize=(15,6))
 for host,color in [('local','tab:blue'),('7789','tab:orange')]:
  t=next(t for t in snapshots[host]['tasks'] if t['mode']=='cross')
  rows=t.get('per_nucleus_x2',[])
  for observable,ax in [('binding_energy_per_a',axes[0]),('beta_decay_energy',axes[1])]:
   vals=[]
   for row in rows:
    for c in row.get('components',[]):
     if c.get('observable')==observable: vals.append((row['Z'],row['N'],c.get('adjusted_residual')))
   vals=sorted(v for v in vals if v[2] is not None)
   labels=[f'Z{z}N{n}' for z,n,v in vals]
   ax.plot(labels,[v for z,n,v in vals],'.-',label=f'{host}: {t["completed_nuclei"]}/{t["required_nuclei"]}',color=color,alpha=.8)
 for ax,title in zip(axes,['B/A residual (keV)','Beta-Q residual (keV)']): ax.set_title(title); ax.set_ylabel('Prediction minus adjusted target'); ax.tick_params(axis='x',rotation=90,labelsize=7); ax.grid(alpha=.2); ax.legend()
 fig.suptitle('Ca40/Ca48 crosses: current partial vector, not a full fitted objective'); save(fig,'ca-partial-observables')


 history=[]
 for line in (OUT/'history.jsonl').read_text().splitlines():
     try: history.append(json.loads(line))
     except ValueError: pass
 fig,axes=plt.subplots(1,2,figsize=(13,5))
 for host,color in [('local','tab:blue'),('7789','tab:orange')]:
     for mode,ax in [('cross',axes[0]),('single',axes[1])]:
         points=[(dt.datetime.fromisoformat(r['utc']),t) for r in history for t in r['tasks'] if t['host']==host and t['mode']==mode]
         if not points: continue
         x=[(t-points[0][0]).total_seconds()/3600 for t,v in points]
         field='completed_nuclei' if mode=='cross' else 'completed_evaluations'
         ax.plot(x,[v[field] for t,v in points],'.-',label=host,color=color)
 axes[0].axhline(86,color='gray',ls=':'); axes[0].set_ylabel('Current Ca vector: completed nuclei / 86')
 axes[1].set_ylabel('Pb208 completed parameter evaluations')
 for ax in axes: ax.set_xlabel('Hours since comparison monitor started'); ax.grid(alpha=.25); ax.legend()
 fig.suptitle('Training progress sampled every 60 seconds'); save(fig,'training-progress')

def sample():
 sync()
 snaps={'local':read(LOCAL/'snapshot.json'),'7789':read(REMOTE/'snapshot.json')}
 curves={'local':read(LOCAL/'curve_data.json'),'7789':read(REMOTE/'curve_data.json')}
 now=dt.datetime.now(dt.timezone.utc).isoformat(); data={'updated_utc':now,'snapshots':snaps}
 comparison={}
 for mode in ['single','cross']:
  rr={h:next(t for t in curves[h]['tasks'] if t['mode']==mode)['evaluations'] for h in curves}
  n=min(map(len,rr.values()))
  comparison[mode]={'equal_completed_count':n,'best_x2_at_equal_count':{h:min((r['x2'] for r in rs[:n]),default=None) for h,rs in rr.items()}}
 data['comparison']=comparison; write(OUT/'latest.json',data)
 history={'utc':now,'tasks':[{**{k:t.get(k) for k in ['mode','name','status','alive','completed_evaluations','best_x2','completed_nuclei','required_nuclei','accepted_steps','rss_mb']},'host':h} for h,s in snaps.items() for t in s['tasks']]}
 with (OUT/'history.jsonl').open('a') as f: f.write(json.dumps(history)+'\n')
 lines=['# 两端实时监控汇总','',f'更新时间（UTC）：{now}','','|位置|任务|运行|完整参数评价|最佳 X²|逐核进度|','|---|---|---|---:|---:|---:|']
 for h,s in snaps.items():
  for t in s['tasks']:
   score='尚无完整结果' if t['best_x2'] is None else f'{t["best_x2"]:.8g}'
   lines.append(f'|{h}|{t["mode"]}|{t["status"]}, alive={t["alive"]}|{t["completed_evaluations"]}|{score}|{t["completed_nuclei"]}/{t["required_nuclei"]}|')
 c=comparison['single']; lines+=['',f'Pb208 按相同的 {c["equal_completed_count"]} 次完整物理评价比较：{c["best_x2_at_equal_count"]}。','','图表：[Pb208 分数与误差](pb208-comparison.png)，[九参数原值](parameters-raw.png)，[九参数相对变化](parameters-relative.png)，[Ca 当前部分观测量](ca-partial-observables.png)，[训练进度](training-progress.png)。','','Ca 未完成整组核的计算前，不报告完整 X²。结合能同时保留原始实验值与每核下调 2 MeV 后的目标。这里只记录现有训练，不修改物理程序、输入参数或运行进程。']
 (OUT/'README.md').write_text('\n'.join(lines)+'\n')
 # A semantic digest prevents needless figure rendering while still tracking each sample.
 semantic=json.dumps({'curves':{h:v['tasks'] for h,v in curves.items()},'tasks':{h:v['tasks'] for h,v in snaps.items()}},sort_keys=True)
 digest=hashlib.sha256(semantic.encode()).hexdigest()
 old=(OUT/'digest.txt').read_text() if (OUT/'digest.txt').exists() else None
 if digest!=old: render(curves,snaps); (OUT/'digest.txt').write_text(digest)
 print(now,'sampled',flush=True)

def main():
 parser=argparse.ArgumentParser(); parser.add_argument('--once',action='store_true'); parser.add_argument('--interval',type=int,default=60); args=parser.parse_args(); OUT.mkdir(parents=True,exist_ok=True); (OUT/'monitor.pid').write_text(str(os.getpid()))
 while True:
  started=time.monotonic()
  try: sample()
  except Exception: traceback.print_exc()
  if args.once: break
  time.sleep(max(1,args.interval-(time.monotonic()-started)))
if __name__=='__main__': main()
