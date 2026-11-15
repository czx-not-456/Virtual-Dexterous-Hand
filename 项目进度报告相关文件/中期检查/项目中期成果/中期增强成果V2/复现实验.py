"""Reproduce V2 supplemental results from the original project data.

Run from any working directory; output is placed under 中期增强成果V2/结果.
No hardware or MuJoCo dependency is needed for this offline analysis.
"""
from pathlib import Path
import argparse
import json
import hashlib
import struct
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.spatial.transform import Rotation
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import confusion_matrix, f1_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "结果"
OUT.mkdir(exist_ok=True)
BENCH = ROOT / "test/test_results_v100_20261008/benchmark"
STAGE = ROOT / "experiments/实验01_StageB七方法消融/results/published_example"
MODEL = ROOT / "数据手套数据/CH-M6型号_虚拟手模型"
POSES = ["张开", "对食指", "对中指", "三指捏", "半握", "握拳"]
SENSORS = [f"sensor_{i}" for i in range(1, 16)]
COLORS = ["#176F91", "#E28C36", "#4B9B80", "#566DAD", "#9E5579", "#6D7886", "#313D51"]
sources = []
for name in ["Noto Sans CJK SC", "Microsoft YaHei", "SimHei", "PingFang SC", "Noto Sans CJK JP"]:
    if any(f.name == name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name
        break
plt.rcParams.update({"axes.unicode_minus":False, "font.size":11, "axes.spines.top":False,
                     "axes.spines.right":False, "figure.facecolor":"white", "axes.grid":True,
                     "grid.alpha":.18, "axes.axisbelow":True})

def read(path):
    sources.append(path)
    return pd.read_csv(path)

def save(fig, name):
    fig.savefig(OUT / name, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)

def csv(df,name):
    df.to_csv(OUT/name,index=False,encoding="utf-8-sig")

def recognition():
    groups=[]
    for p in sorted((ROOT/"数据手套数据/records/chm6_calibrations").glob("*/captured_samples.csv")):
        d=read(p)
        d=d[d.pose.isin(POSES)&d.sample_source.eq("captured")].dropna(subset=SENSORS).copy()
        d["session"]=p.parent.name
        d["device"]=p.parent.name.split("_")[2]
        groups.append(d)
    d=pd.concat(groups,ignore_index=True)
    assert d.device.nunique()>1 and set(d.pose)==set(POSES)
    results=[]; preds=[]
    # Leave one physical device serial out; fit normalisation and prototypes on training devices only.
    for fold,device in enumerate(sorted(d.device.unique())):
        tr=d[d.device!=device]; te=d[d.device==device]
        assert not set(tr.session)&set(te.session)
        x=tr[SENSORS].to_numpy(); y=tr.pose.to_numpy()
        z=te[SENSORS].to_numpy(); truth=te.pose.to_numpy()
        mu=x.mean(0); scale=np.maximum(x.std(0),1e-8)
        x=(x-mu)/scale; z=(z-mu)/scale
        # Equal weight across training devices within each pose.
        cent=np.stack([np.mean([x[(y==p)&(tr.device.to_numpy()==dv)].mean(0)
                               for dv in tr.device.unique() if np.any((y==p)&(tr.device.to_numpy()==dv))],0)
                       for p in POSES])
        for nmask in [0,1,3,5]:
            for repeat in range(1 if nmask==0 else 10):
                rng=np.random.default_rng(20261114+fold*1000+nmask*100+repeat)
                observed=np.ones_like(z,dtype=bool)
                # A fixed failure pattern per device/repeat, retained for all held-out frames.
                missing=rng.choice(15,nmask,replace=False)
                observed[:,missing]=False
                dist=(z[:,None,:]-cent[None,:,:])**2
                baseline=((np.where(observed,z,0)[:,None,:]-cent[None,:,:])**2).sum(2)
                masked=(dist*observed[:,None,:]).sum(2)/(15-nmask)
                for method,distances in [("均值填补",baseline),("可用通道距离",masked)]:
                    pred=np.array(POSES)[distances.argmin(1)]
                    results.append(dict(device=device,missing=nmask,repeat=repeat,method=method,
                                        frames=len(te),accuracy=np.mean(pred==truth),
                                        macro_f1=f1_score(truth,pred,labels=POSES,average="macro",zero_division=0),
                                        missing_channels=" ".join(str(i+1) for i in missing)))
                    if nmask==0 and method=="可用通道距离":
                        preds.extend(dict(device=device,session=s,sample_index=i,truth=t,prediction=a)
                                     for s,i,t,a in zip(te.session,te.sample_index,truth,pred))
    r=pd.DataFrame(results); p=pd.DataFrame(preds)
    csv(r,"13_设备留出与缺失实验明细.csv");csv(p,"13_无缺失逐帧预测.csv")
    per=r.groupby(["missing","method","device"],as_index=False)[["accuracy","macro_f1"]].mean()
    summary=per.groupby(["missing","method"],as_index=False).agg(accuracy=("accuracy","mean"),
             device_sd=("accuracy","std"),macro_f1=("macro_f1","mean"),devices=("device","size"))
    csv(summary,"13_设备等权指标汇总.csv")
    cm=confusion_matrix(p.truth,p.prediction,labels=POSES)
    norm=cm/cm.sum(1,keepdims=True)
    fig,ax=plt.subplots(figsize=(7.7,5.8),layout="constrained")
    im=ax.imshow(norm,cmap="Blues",vmin=0,vmax=1); ax.grid(False)
    ax.set(xticks=range(6),yticks=range(6),xticklabels=POSES,yticklabels=POSES,
           xlabel="预测标签",ylabel="记录标签",title="六类标定动作识别  按设备留出")
    for i in range(6):
        for j in range(6):
            ax.text(j,i,f"{norm[i,j]:.0%}\n{cm[i,j]}",ha="center",va="center",
                    color="white" if norm[i,j]>.55 else "#172F48",fontsize=10)
    fig.colorbar(im,ax=ax,label="行内占比");save(fig,"13_六类手势混淆矩阵.png")
    fig,ax=plt.subplots(figsize=(8.8,4.8),layout="constrained")
    for i,(method,g) in enumerate(summary.groupby("method",sort=True)):
        ax.errorbar(g.missing,g.accuracy*100,yerr=g.device_sd*100,marker="o",capsize=4,
                    linewidth=2.1,label=method,color=COLORS[i])
    ax.set(xticks=[0,1,3,5],xlabel="缺失传感器数量（共15路）",ylabel="设备等权准确率（%）",
           title="缺失通道压力测试  同一掩码配对比较",ylim=(85,105),yticks=[85,90,95,100,105])
    ax.legend();save(fig,"13_传感器缺失鲁棒性.png")
    fig,ax=plt.subplots(figsize=(9,4.5),layout="constrained")
    g=per[(per.missing==0)&(per.method=="可用通道距离")].sort_values("accuracy")
    ax.barh(range(len(g)),g.accuracy*100,color=COLORS[0]);ax.set(yticks=range(len(g)),yticklabels=g.device,
           xlabel="留出设备准确率（%）",xlim=(0,105),title="不同设备的迁移表现")
    save(fig,"13_设备泛化差异.png")
    return {"sessions":int(d.session.nunique()),"devices":int(d.device.nunique()),"frames":len(d),
            "summary":summary.to_dict("records"),"pooled_accuracy":float(np.mean(p.truth==p.prediction))}

def benchmark():
    run=read(BENCH/"benchmark_runs.csv")
    tasks=["neutral","pinch","wrap"]
    fig,axs=plt.subplots(1,2,figsize=(10,4.6),layout="constrained")
    for i,task in enumerate(tasks):
        g=run[run.task==task]
        x=np.arange(len(g))+i*4
        axs[0].bar(x-.17,g.contact_frame_ratio*100,width=.34,color=COLORS[0],label="接触代理" if i==0 else None)
        axs[0].bar(x+.17,g.stable_frame_ratio*100,width=.34,color=COLORS[1],label="稳定代理" if i==0 else None)
        axs[1].bar(x,g.p95_pipeline_latency_ms,color=COLORS[i],width=.65)
    for a in axs:
        a.set_xticks([1,5,9],["NEUTRAL","PINCH","WRAP"])
    axs[0].set(title="各试次有效帧占比",ylabel="占比（%）",ylim=(0,105));axs[0].legend()
    axs[1].set(title="各试次处理延迟 P95",ylabel="毫秒")
    save(fig,"09_九次试验汇总.png")
    summary=run.groupby("task",as_index=False).agg(trials=("trial","size"),
           contact_pass=("contact_success_proxy","sum"),stable_pass=("stable_success_proxy","sum"),
           mean_contact_ratio=("contact_frame_ratio","mean"),mean_stable_ratio=("stable_frame_ratio","mean"),
           mean_p95_ms=("p95_pipeline_latency_ms","mean"))
    csv(summary,"09_三任务统计.csv")
    for task in ["pinch","wrap"]:
        d=read(BENCH/f"bench_{task}_trial00.csv")
        fig,axs=plt.subplots(3,1,figsize=(9,7.0),sharex=True,layout="constrained")
        x=d.frame; active=d.task_active.astype(bool)
        for a in axs:
            a.fill_between(x,0,1,where=active,transform=a.get_xaxis_transform(),color="#DEF2EC",alpha=.7,label="任务有效区间")
        axs[0].plot(x,d.pinch_distance_m*1000,color=COLORS[0],label="指尖距（mm）")
        ax2=axs[0].twinx();ax2.plot(x,d.wrap_score,color=COLORS[1],alpha=.8,label="包络评分")
        axs[0].set_ylabel("指尖距离（mm）");ax2.set_ylabel("包络评分")
        axs[1].step(x,d.task_contact_proxy.astype(int),where="post",label="接触代理",color=COLORS[0])
        axs[1].step(x,d.stable_contact_proxy.astype(int),where="post",label="稳定代理",color=COLORS[1])
        axs[1].set(yticks=[0,1],ylabel="代理指标");axs[1].legend(loc="upper right",ncol=2)
        axs[2].plot(x,d.pipeline_latency_ms,color=COLORS[2],linewidth=1);axs[2].axhline(d.pipeline_latency_ms.quantile(.95),color=COLORS[1],linestyle="--",label="P95")
        axs[2].set(xlabel="帧编号",ylabel="处理延迟（ms）");axs[2].legend()
        fig.suptitle(f"{task.upper()}  trial 00  历史 MuJoCo 日志")
        save(fig,f"08_{task}_任务时序.png")
    return summary.to_dict("records")

def stageb():
    d=read(STAGE/"summary_by_condition.csv")
    methods=["ridge6_original","pca4_original","ridge6_robust","pca4_robust","fixed_50_50","calibrated_scalar_blend","tqhg_v2_original"]
    fig,ax=plt.subplots(figsize=(10,5),layout="constrained")
    for i,m in enumerate(methods):
        ax.plot(d.condition,d[m],marker="o",color=COLORS[i],label=m,linewidth=1.7)
    ax.set(title="Stage B 七方法  历史配对消融结果",xlabel="通道缺失条件",ylabel="六维软件教师标签归一化 MAE")
    ax.legend(ncol=2,fontsize=9);save(fig,"10_StageB七方法对比.png")
    return d[["condition"]+methods].to_dict("records")

def smooth():
    d=read(ROOT/"datasets/synthetic_glove_v1.csv")
    channels=[c for c in d if c.endswith(("_mcp","_pip","_dip"))]
    metrics=[]; example=None
    for tid,g in d.groupby("trial_id"):
        g=g.sort_values("frame"); ref=g[channels].to_numpy()
        # The original synthetic trace is the reference, not physical ground truth.
        rng=np.random.default_rng(700+int(tid))
        scale=np.maximum(np.std(ref,axis=0),1e-8)
        noise=rng.normal(size=ref.shape)*scale*.12
        noisy=ref+noise
        for alpha in [1.,.7,.45,.25]:
            out=noisy.copy()
            for i in range(1,len(out)): out[i]=alpha*noisy[i]+(1-alpha)*out[i-1]
            metrics.append(dict(trial=tid,alpha=alpha,nrmse=float(np.sqrt(np.mean(((out-ref)/scale)**2))),
                                frame_diff_rms=float(np.sqrt(np.mean((np.diff(out,axis=0)/scale)**2)))))
            if tid==0 and alpha==.45: example=(g.frame.to_numpy(),ref[:,3],noisy[:,3],out[:,3])
    r=pd.DataFrame(metrics);csv(r,"14_滤波压力实验明细.csv")
    avg=r.groupby("alpha",as_index=False)[["nrmse","frame_diff_rms"]].mean();csv(avg,"14_滤波指标汇总.csv")
    fig,axs=plt.subplots(2,1,figsize=(9,6.4),layout="constrained")
    x,ref,noisy,filtered=example
    axs[0].plot(x,noisy,color="#9BA9B4",alpha=.8,label="注入噪声")
    axs[0].plot(x,ref,color=COLORS[0],linewidth=1.4,label="原始合成输入")
    axs[0].plot(x,filtered,color=COLORS[1],linewidth=1.7,label="EMA α=0.45")
    axs[0].set(xlabel="帧",ylabel="传感器数值",title="trial 0 食指 MCP 通道");axs[0].legend(ncol=3,fontsize=9)
    axs[1].plot(avg.alpha,avg.nrmse,marker="o",label="相对原始输入的归一化RMSE",color=COLORS[0])
    axs[1].plot(avg.alpha,avg.frame_diff_rms,marker="s",label="归一化帧间差分 RMS",color=COLORS[1])
    axs[1].set(xlabel="EMA 系数 α（1=不滤波）",ylabel="无量纲",title="8 条合成序列的噪声注入实验");axs[1].legend(fontsize=9)
    save(fig,"14_滤波平滑与偏差.png")
    return avg.to_dict("records")

def flow():
    fig,ax=plt.subplots(figsize=(11,4),layout="constrained");ax.axis("off")
    items=[("15路数据手套","标定记录 / 合成输入"),("预处理","归一化 / 滤波"),("意图与映射","意图识别 / 协同映射"),
           ("CH-M6灵巧手","11关节 / MuJoCo"),("量化评估","接触代理 / 延迟")]
    for i,(a,b) in enumerate(items):
        x=.1+i*.2
        ax.text(x,.62,a,ha="center",va="center",weight="bold",fontsize=12,transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=.8",fc="#EAF3F7",ec="#176F91"))
        ax.text(x,.34,b,ha="center",va="center",fontsize=10,transform=ax.transAxes)
        if i<4: ax.annotate("",xy=(x+.145,.62),xytext=(x+.07,.62),xycoords=ax.transAxes,
                             arrowprops=dict(arrowstyle="->",lw=1.8,color="#66798F"))
    ax.text(.5,.9,"虚拟灵巧手系统  数据与评估流程",ha="center",transform=ax.transAxes,fontsize=18,weight="bold")
    ax.text(.5,.12,"新增验证：设备留出识别 / 缺失通道压力测试 / 合成输入滤波实验",ha="center",transform=ax.transAxes,color="#176F91")
    save(fig,"07_系统流程.png")

def load_mesh(path):
    # Binary STL: 80 byte header, count, then triangles with vertices.
    dtype=np.dtype([("normal","<f4",3),("verts","<f4",(3,3)),("attr","<u2")])
    with path.open("rb") as f:
        f.read(80); n=struct.unpack("<I",f.read(4))[0]
        a=np.fromfile(f,dtype=dtype,count=n)
    assert len(a)==n
    sources.append(path)
    return a["verts"].astype(float)

def pose_replay(skip=False):
    if skip: return
    # Offline FK and orthographic projection directly from original MJCF + STL,
    # using recorded joint vectors. No new physics, object pose or contact claim.
    xml=MODEL/"CH-M6_L.xml"; sources.append(xml); root=ET.parse(xml).getroot()
    meshes={p.stem:load_mesh(p) for p in (MODEL/"meshes").glob("*.STL")}
    basis=np.array([.8,.3,.6]);basis/=np.linalg.norm(basis)
    right=np.cross([0,1,0],basis);right/=np.linalg.norm(right)
    up=np.cross(basis,right)
    def vec(el,k,default):return np.fromstring(el.get(k,default),sep=" ")
    def parts(body,rot,offset,row):
        pos=vec(body,"pos","0 0 0");q=vec(body,"quat","1 0 0 0")
        offset=rot@pos+offset;rot=rot@Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
        for j in body.findall("joint"):
            axis=vec(j,"axis","0 0 1");axis/=np.linalg.norm(axis)
            theta=float(row["q_"+j.get("name")])
            # All hand joints in this model have zero local joint origin.
            assert np.allclose(vec(j,"pos","0 0 0"),0)
            rot=rot@Rotation.from_rotvec(axis*theta).as_matrix()
        out=[]
        for geom in body.findall("geom"):
            mesh=geom.get("mesh")
            if mesh in meshes:
                out.append(meshes[mesh]@rot.T+offset)
        for sub in body.findall("body"):out.extend(parts(sub,rot,offset,row))
        return out
    def render(row):
        triangles=np.concatenate(parts(root.find("worldbody/body"),np.eye(3),np.zeros(3),row))
        # Fixed camera and scale across all frames.
        center=np.array([0,.075,0]); pts=triangles-center
        sx=pts@right;sy=pts@up;depth=(pts@basis).mean(1)
        normal=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
        n=np.linalg.norm(normal,axis=1);normal/=np.maximum(n,1e-12)[:,None]
        light=np.clip(normal@basis,0,1)
        col=np.clip(np.array([78,120,147])+light[:,None]*110,0,255).astype(np.uint8)
        im=Image.new("RGB",(500,560),"#F1F5F8");drawer=ImageDraw.Draw(im)
        coords=np.stack([250+sx*2450,285-sy*2450],axis=-1).astype(int)
        for k in np.argsort(depth):
            drawer.polygon([tuple(t) for t in coords[k]],fill=tuple(col[k]))
        return im
    for task in ["pinch","wrap"]:
        d=read(BENCH/f"bench_{task}_trial00.csv"); active=d[d.task_active.astype(bool)]
        idx=[0,int(active.frame.iloc[0]),int(active.frame.iloc[len(active)//2]),int(active.frame.iloc[-1])]
        frames=[render(d.loc[d.frame==i].iloc[0]) for i in idx]
        fig,axs=plt.subplots(1,4,figsize=(11,4.2),layout="constrained")
        for a,im,i in zip(axs,frames,idx):a.imshow(im);a.axis("off");a.set_title(f"帧 {i}")
        fig.suptitle(f"{task.upper()}  CH-M6 原模型网格与历史关节姿态重建（非物理仿真）")
        save(fig,f"05_{task}_三维姿态重建.png")
        seq=np.unique(np.linspace(0,len(d)-1,22).astype(int))
        ims=[render(d.iloc[i]) for i in seq]
        ims[0].save(OUT/f"06_{task}_历史姿态回放.gif",save_all=True,append_images=ims[1:],duration=260,loop=0)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--skip-3d",action="store_true");args=ap.parse_args()
    info={"recognition":recognition(),"benchmark":benchmark(),"stageb":stageb(),"smooth":smooth()}
    flow();pose_replay(args.skip_3d)
    (OUT/"summary.json").write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding="utf-8")
    csv(pd.DataFrame([{"path":str(x.relative_to(ROOT)),"sha256":hashlib.sha256(x.read_bytes()).hexdigest()}
                      for x in sorted(set(sources))]),"数据来源与校验.csv")
    print(json.dumps(info["recognition"],ensure_ascii=False,indent=2))
    print("Results:",OUT)

if __name__=="__main__":main()
