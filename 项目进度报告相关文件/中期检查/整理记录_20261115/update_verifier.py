from pathlib import Path
import json
O=Path(__file__).resolve().parent;P=json.loads((O/'整理操作计划.json').read_text(encoding='utf8'));B=Path(P['base']);D=P['dirs']
src=B/'图册配套依据/核验图册数据.py';dest=B/D['12']/'核验成果.py';s=src.read_text(encoding='utf8');s=s.replace('ROOT=Path(__file__).resolve().parent','ROOT=Path(__file__).resolve().parent.parent')
a=s.index('def hashes():');z=s.index('\ndef physics():',a)
h='''def hashes():
    manifest=js('MANIFEST_PATH')
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
            text=p.read_text(encoding='utf-8-sig').replace('\\r\\n','\\n')
            for e in reversed(manifest['reversible_text_edits'].get(r['current_file'],[])):
                start=e['start'];after=e['after'];assert text[start:start+len(after)]==after,r['current_file']
                text=text[:start]+e['before']+text[start+len(after):]
            raw=text.encode('utf8')
        assert hashlib.sha256(raw).hexdigest()==r['normalized_sha256'],r['file']
    old=csv('V2_PATH');assert len(old)==45
    for r in old.itertuples(index=False):assert mapped[r.path]==r.sha256,r.path
    ok(f"{len(manifest['files'])}份现存文件哈希与覆盖一致；144项原证据、45项历史V2来源完整，原内容或可逆编辑前内容一致")
'''.replace('MANIFEST_PATH',D['12']+'/来源与校验清单.json').replace('V2_PATH',D['01']+'/补充运行_20261115/测试与校验/V2_原始来源清单.csv')
s=s[:a]+h+s[z:]
s=s.replace("pre='02_MuJoCo实测/'+task+'/'","pre='"+D['02']+"/补充运行_20261115/'+task+'/'")
s=s.replace("idx=csv(pre+'截图索引.csv')","idx=csv('"+D['05']+"/'+task+'/截图索引.csv')")
s=s.replace("(ROOT/pre/r.filename).is_file()","(ROOT/'"+D['05']+"'/task/r.filename).is_file()")
s=s.replace("pre='04-06_历史任务与消融/历史Benchmark/'","pre='"+D['02']+"/'")
s=s.replace("04-06_历史任务与消融/重分析结果/09_三任务统计.csv",D['09']+'/09_三任务统计.csv')
s=s.replace("pre='04-06_历史任务与消融/StageB消融/'","pre='"+D['03']+"/'")
s=s.replace("base=ROOT/'07-08_识别与鲁棒性'","base=ROOT/'图册配套依据/07-08_识别与鲁棒性'")
s=s.replace("pre='09_滤波抗噪/'","pre='图册配套依据/09_滤波抗噪/'")
# Add raw summary, screenshot time/count and paired mask checks; do not weaken old assertions.
s=s.replace("assert len(d)==len(frame)==800 and len(idx)==5 and d.frame.is_unique","assert len(d)==len(frame)==800 and len(idx)==5 and d.frame.is_unique\n        assert a.frame.min()==(171 if task=='PINCH' else 489) and a.frame.max()==(407 if task=='PINCH' else 727)\n        eq(d.simulation_time_s,frame.simulation_time_s) if 'simulation_time_s' in d and 'simulation_time_s' in frame else None")
# Original numeric checks remain in place.
dest.write_text(s,encoding='utf8');assert src.resolve().is_relative_to(B.resolve());src.unlink()
print('VERIFIER_RELOCATED_AND_UPDATED',dest)
