"""Independent offline numerical audit; requires numpy and pandas, not MuJoCo."""
from pathlib import Path
import hashlib, json, sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent.parent
def csv(name): return pd.read_csv(ROOT/name)
def js(name): return json.loads((ROOT/name).read_text(encoding='utf8'))
def eq(a,b): np.testing.assert_allclose(a,b,rtol=1e-9,atol=1e-12)
def ok(t): print('PASS | '+t)

def hashes():
    manifest=js('12_附件目录与复现说明/来源与校验清单.json')
    assert manifest['schema_version']==2
    listed={r['file'] for r in manifest['files']}
    actual={p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file() and p.name!='来源与校验清单.json' and '__pycache__' not in p.parts}
    assert listed==actual,('清单覆盖差异',sorted(listed^actual))
    for r in manifest['files']:
        p=ROOT/r['file'];assert p.is_file() and p.stat().st_size==r['bytes'],r['file']
        assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'],r['file']
    evidence=manifest['original_evidence_144'];assert len(evidence)==144
    mapped={r['source']:r['sha256'] for r in evidence}
    text_ext={'.csv','.json','.md','.txt','.log','.py','.ps1','.yaml','.toml','.xml','.html'}
    for r in evidence:
        p=ROOT/r['current_file'];raw=p.read_bytes()
        if p.suffix.lower() in text_ext:
            text=p.read_text(encoding='utf-8-sig').replace('\r\n','\n')
            for e in reversed(manifest['reversible_text_edits'].get(r['current_file'],[])):
                start=e['start'];after=e['after'];assert text[start:start+len(after)]==after,r['current_file']
                text=text[:start]+e['before']+text[start+len(after):]
            raw=text.encode('utf8')
        assert hashlib.sha256(raw).hexdigest()==r['normalized_sha256'],r['file']
    old=csv('01_综合测试与环境记录/补充运行_20261115/测试与校验/V2_原始来源清单.csv');assert len(old)==45
    for r in old.itertuples(index=False):assert mapped[r.path]==r.sha256,r.path
    ok(f"{len(manifest['files'])}份现存文件哈希与覆盖一致；144项原证据、45项历史V2来源完整，原内容或可逆编辑前内容一致")

def physics():
    for task in ('PINCH','WRAP'):
        pre='02_MuJoCo三任务批量实验/补充运行_20261115/'+task+'/'
        d=csv(pre+'当前物理运行.csv'); s=js(pre+'当前物理运行_summary.json'); idx=csv('05_MuJoCo任务截图组/'+task+'/截图索引.csv'); frame=csv(pre+'实时帧记录.csv'); a=d[d.task_active]
        assert len(d)==len(frame)==800 and len(idx)==5 and d.frame.is_unique
        assert a.frame.min()==(171 if task=='PINCH' else 489) and a.frame.max()==(407 if task=='PINCH' else 727)
        eq(d.simulation_time_s,frame.simulation_time_s) if 'simulation_time_s' in d and 'simulation_time_s' in frame else None
        eq(a.task_contact_proxy.mean(),s['contact_frame_ratio']); eq(a.hold_stable_proxy.mean(),s['stable_frame_ratio'])
        assert bool(a.contact_success_proxy.any())==s['contact_success_proxy'] and bool(a.iloc[-1].stable_success_proxy)==s['stable_success_proxy']
        for r in idx.itertuples(index=False):
            f=frame.loc[frame.frame==r.frame].iloc[0];assert f.task_contact_proxy==r.task_contact_proxy and f.fixture_held==r.fixture_held and (ROOT/'05_MuJoCo任务截图组'/task/r.filename).is_file()
        ok(f'{task} 实测800帧：接触{a.task_contact_proxy.mean():.2%}，新版稳定{a.hold_stable_proxy.mean():.2%}，稳定判定={s["stable_success_proxy"]}')

def benchmark_and_pose():
    pre='02_MuJoCo三任务批量实验/'
    run=csv(pre+'benchmark_runs.csv'); assert len(run)==9
    for r in run.itertuples(index=False):
        d=csv(pre+f'bench_{r.task}_trial{r.trial:02d}.csv'); a=d[d.task_active]
        assert len(d)==800 and len(a)==r.active_frames
        eq(a.task_contact_proxy.mean(),r.contact_frame_ratio);eq(a.stable_contact_proxy.mean(),r.stable_frame_ratio);eq(d.pipeline_latency_ms.quantile(.95),r.p95_pipeline_latency_ms)
        if r.task in ('pinch','wrap') and r.trial==0:
            assert len([c for c in d if c.startswith('q_')])==11
            key=[0,int(a.frame.iloc[0]),int(a.frame.iloc[len(a)//2]),int(a.frame.iloc[-1])]
            print(f'  {r.task} 重建关键帧：{key}')
    stats=csv('09_Benchmark汇总展示页/09_三任务统计.csv').set_index('task')
    for t,g in run.groupby('task'):
        s=stats.loc[t];eq([g.contact_success_proxy.sum(),g.stable_success_proxy.sum(),g.task_metric_pass.sum(),g.contact_frame_ratio.mean(),g.stable_frame_ratio.mean(),g.p95_pipeline_latency_ms.mean()],s[['contact_pass','stable_pass','required_metric_pass','mean_contact_ratio','mean_stable_ratio','mean_p95_ms']].to_numpy(float))
    ok('9次历史任务逐帧复算与图册汇总相同；三维姿态关键帧及11关节字段存在')
    pre='03_StageB七方法消融/'
    detail=csv(pre+'paired_cases.csv'); stats=csv(pre+'summary_by_condition.csv').set_index('condition')
    cols=['ridge6_original','pca4_original','ridge6_robust','pca4_robust','fixed_50_50','calibrated_scalar_blend','tqhg_v2_original']
    assert len(detail)==1464
    eq(detail.groupby('condition')[cols].mean().loc[stats.index].to_numpy(),stats[cols].to_numpy())
    ok(f'Stage B 1464条配对明细复算7方法均值一致，五列屏蔽 robust={stats.loc["mask_five","ridge6_robust"]:.5f}')

def recognition():
    base=ROOT/'图册配套依据/07-08_识别与鲁棒性'; poses=['张开','对食指','对中指','三指捏','半握','握拳']; sensors=[f'sensor_{i}' for i in range(1,16)]
    frames=[]
    for f in sorted((base/'原始采集数据').glob('*/captured_samples.csv')):
        d=pd.read_csv(f);d=d[d.pose.isin(poses)&d.sample_source.eq('captured')].dropna(subset=sensors).copy();d['session']=f.parent.name;d['device']=f.parent.name.split('_')[2];frames.append(d)
    d=pd.concat(frames,ignore_index=True);assert len(d)==5100 and d.device.nunique()==12 and d.session.nunique()==27
    result=[]; predicted=[]
    for fold,device in enumerate(sorted(d.device.unique())):
        tr=d[d.device!=device]; te=d[d.device==device]; x=tr[sensors].to_numpy(); z=te[sensors].to_numpy(); y=tr.pose.to_numpy();truth=te.pose.to_numpy()
        mu=x.mean(0);scale=np.maximum(x.std(0),1e-8);x=(x-mu)/scale;z=(z-mu)/scale
        tdev=tr.device.to_numpy();cent=np.stack([np.mean([x[(y==p)&(tdev==dv)].mean(0) for dv in tr.device.unique() if np.any((y==p)&(tdev==dv))],0) for p in poses])
        for n in (0,1,3,5):
            for rep in range(1 if n==0 else 10):
                missing=np.random.default_rng(20261114+fold*1000+n*100+rep).choice(15,n,replace=False);observed=np.ones_like(z,dtype=bool);observed[:,missing]=False
                a=((np.where(observed,z,0)[:,None,:]-cent[None,:,:])**2).sum(2);b=(((z[:,None,:]-cent[None,:,:])**2)*observed[:,None,:]).sum(2)/(15-n)
                for method,dist in [('均值填补',a),('可用通道距离',b)]:
                    pred=np.array(poses)[dist.argmin(1)];result.append((device,n,rep,method,float(np.mean(pred==truth))))
                    if n==0 and method=='可用通道距离':predicted.extend(zip(te.session,te.sample_index,truth,pred))
    est=pd.DataFrame(result,columns=['device','missing','repeat','method','accuracy']).set_index(['device','missing','repeat','method']).sort_index()
    old=pd.read_csv(base/'计算结果/13_设备留出与缺失实验明细.csv').set_index(est.index.names).sort_index();assert est.index.equals(old.index);eq(est.accuracy,old.accuracy)
    pred=pd.DataFrame(predicted,columns=['session','sample_index','truth','prediction']).sort_values(['session','sample_index']).reset_index(drop=True)
    old=pd.read_csv(base/'计算结果/13_无缺失逐帧预测.csv')[pred.columns].sort_values(['session','sample_index']).reset_index(drop=True);pd.testing.assert_frame_equal(pred,old)
    group=est.reset_index().groupby(['missing','method','device']).accuracy.mean().groupby(['missing','method']).mean()
    old=pd.read_csv(base/'计算结果/13_设备等权指标汇总.csv').set_index(['missing','method']).sort_index();eq(group,old.accuracy)
    g12=d[d.device.str.startswith('G12')]; assert g12.device.nunique()==8 and g12[['sensor_7','sensor_10','sensor_13']].eq(0).all().all()
    ok(f'27组采集记录重新训练/预测，12设备留出、5100帧、744次方法/掩码明细均吻合；逐帧准确率{(pred.truth==pred.prediction).mean():.2%}，五列屏蔽{group.loc[(5,"可用通道距离")]:.2%}')

def filter_test():
    pre='图册配套依据/09_滤波抗噪/';d=csv(pre+'原始输入/synthetic_glove_v1.csv');channels=[c for c in d if c.endswith(('_mcp','_pip','_dip'))];metrics=[]
    for tid,g in d.groupby('trial_id'):
        ref=g.sort_values('frame')[channels].to_numpy();assert len(ref)==800
        scale=np.maximum(np.std(ref,axis=0),1e-8);noisy=ref+np.random.default_rng(700+int(tid)).normal(size=ref.shape)*scale*.12
        for alpha in [1.,.7,.45,.25]:
            out=noisy.copy()
            for i in range(1,len(out)):out[i]=alpha*noisy[i]+(1-alpha)*out[i-1]
            metrics.append((tid,alpha,np.sqrt(np.mean(((out-ref)/scale)**2)),np.sqrt(np.mean((np.diff(out,axis=0)/scale)**2))))
    m=pd.DataFrame(metrics,columns=['trial','alpha','nrmse','frame_diff_rms']).set_index(['trial','alpha']).sort_index();old=csv(pre+'计算结果/14_滤波压力实验明细.csv').set_index(['trial','alpha']).sort_index();assert len(m)==32;eq(m.to_numpy(),old[m.columns].to_numpy())
    ok('8条合成序列×4种EMA参数，32项滤波指标独立复算一致')

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    for fn in (hashes,physics,benchmark_and_pose,recognition,filter_test):fn()
    print('全部数值核验完成；此脚本不重新运行MuJoCo、不验证硬件或生成新的实验结论。')
