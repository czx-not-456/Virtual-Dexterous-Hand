"""Validate generated outputs and execute selected fixture-free project tests."""
from pathlib import Path
import importlib.util
import inspect
import os
import sys
import json
import traceback
from datetime import datetime, timezone
import platform
from importlib.metadata import version, PackageNotFoundError
import hashlib
import pandas as pd
import numpy as np
from PIL import Image

from project_paths import HERE, ROOT
OUT=HERE/"结果"
sys.path.insert(0,str(ROOT))
os.chdir(ROOT)
checks=[]

def check(name, fn):
    try:
        fn()
        checks.append((name,"PASS",""))
    except Exception:
        checks.append((name,"FAIL",traceback.format_exc()))

def prediction():
    p=pd.read_csv(OUT/"13_无缺失逐帧预测.csv")
    assert len(p)==5100 and p.device.nunique()==12 and p.session.nunique()==27
    assert not p.duplicated(["session","sample_index"]).any()
    report=json.loads((OUT/"summary.json").read_text(encoding="utf8"))
    assert np.isclose((p.truth==p.prediction).mean(),report["recognition"]["pooled_accuracy"])

def paired():
    r=pd.read_csv(OUT/"13_设备留出与缺失实验明细.csv")
    assert len(r)==12*(1+3*10)*2
    for _,g in r.groupby(["device","missing","repeat"]):
        assert len(g)==2 and g.missing_channels.nunique(dropna=False)==1
        assert g.frames.nunique()==1
    assert r.accuracy.between(0,1).all()
    a=r[r.missing==0].pivot(index="device",columns="method",values="accuracy")
    assert np.allclose(a.iloc[:,0],a.iloc[:,1])

def benchmarks():
    r=pd.read_csv(OUT/"09_三任务统计.csv").set_index("task")
    assert r.trials.sum()==9 and r.loc["pinch","contact_pass"]==3 and r.loc["pinch","stable_pass"]==0
    assert r.loc["wrap","stable_pass"]==3
    assert (r.required_metric_pass==3).all()

def filter_metrics():
    r=pd.read_csv(OUT/"14_滤波压力实验明细.csv")
    assert len(r)==32 and r.trial.nunique()==8
    assert np.isfinite(r[["nrmse","frame_diff_rms"]]).all().all()

def images():
    for p in OUT.glob("*.png"):
        with Image.open(p) as i:
            assert i.width>500 and i.height>300
            i.verify()
    for p in OUT.glob("*.gif"):
        with Image.open(p) as i: assert i.n_frames>1
    assert len(list(OUT.glob("*.png")))==11 and len(list(OUT.glob("*.gif")))==2

def checksums():
    s=pd.read_csv(OUT/"数据来源与校验.csv")
    old=pd.read_csv(HERE/"历史记录/V2_原始来源清单.csv").set_index("path")
    assert len(s)==45
    for r in s.itertuples(index=False):
        path=ROOT/r.path; b=path.read_bytes()
        raw=hashlib.sha256(b).hexdigest()
        lf=hashlib.sha256(b.replace(b"\r\n",b"\n")).hexdigest() if path.suffix.lower() in {".csv",".xml",".md",".yaml",".json",".txt"} else raw
        assert old.loc[r.path,"sha256"]==r.original_sha256
        assert raw==r.current_sha256 and lf==r.lf_sha256
        expected="byte_match" if raw==r.original_sha256 else ("crlf_to_lf_match" if lf==r.original_sha256 else "content_mismatch")
        assert r.comparison==expected and expected!="content_mismatch",r.path
    # Exercise the newline-only normalization branch without modifying source data.
    assert hashlib.sha256(b"row1\r\nrow2\r\n".replace(b"\r\n",b"\n")).hexdigest()==hashlib.sha256(b"row1\nrow2\n").hexdigest()

for n,fn in [("逐帧预测数量与唯一性",prediction),("缺失掩码配对与一致性",paired),
             ("九次Benchmark指标",benchmarks),("滤波实验完整性",filter_metrics),
             ("图像及动画可读性",images),("原始数据SHA256",checksums)]:
    check(n,fn)

for module in ["test_features_intent","test_calibration","test_mapping","test_robot_architecture","test_glove_adapter"]:
    p=ROOT/"test/tests_20261012"/(module+".py")
    spec=importlib.util.spec_from_file_location(module,p)
    mod=importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception:
        checks.append((module,"FAIL",traceback.format_exc()));continue
    for name,fn in inspect.getmembers(mod,inspect.isfunction):
        if name.startswith("test_") and not inspect.signature(fn).parameters:
            check(module+"."+name,fn)

result=pd.DataFrame(checks,columns=["name","status","details"])
result.to_csv(OUT/"11_测试记录.csv",index=False,encoding="utf-8-sig")
versions={}
for pkg in ["numpy","pandas","matplotlib","scipy","scikit-learn","Pillow","PyYAML","mujoco","pytest"]:
    try: versions[pkg]=version(pkg)
    except PackageNotFoundError: versions[pkg]="not installed"
env={"timestamp_utc":datetime.now(timezone.utc).isoformat(),"python":sys.version,
     "platform":platform.platform(),"dependencies":versions,
     "script_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob("*.py")},
     "integrity_checks":6,"selected_fixture_free_tests":len(result)-6,
     "passed":int(sum(result.status=="PASS")),"full_pytest_executed":False,
     "mujoco_physics_included_in_these_checks":False,
     "scope":"current execution, not the recipient's Windows checkout"}
(OUT/"11_本次执行环境.json").write_text(json.dumps(env,ensure_ascii=False,indent=2),encoding="utf8")
log=result.to_string(index=False)+"\n"+json.dumps(env,ensure_ascii=False,indent=2)
(OUT/"11_本次执行完整日志.txt").write_text(log,encoding="utf8")
print(result[["name","status"]].to_string(index=False))
print(f"PASS {sum(result.status=='PASS')}/{len(result)}")
if any(result.status=="FAIL"):
    print(result.loc[result.status=="FAIL","details"].to_string(index=False))
    raise SystemExit(1)
