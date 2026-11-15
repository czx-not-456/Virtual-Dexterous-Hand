from pathlib import Path
import json,hashlib,os,urllib.parse,collections
O=Path(__file__).resolve().parent;P=json.loads((O/'整理操作计划.json').read_text(encoding='utf8'));B=Path(P['base']);D=P['dirs'];edits=json.loads((O/'可逆文本修订.json').read_text(encoding='utf8'))
# Rebuild final index after all entrypoints have been created.
p=B/D['12']/'附件目录.md';t=p.read_text(encoding='utf8').split('## 文件逐项索引')[0]+'## 文件逐项索引\n\n'
files=sorted(p for p in B.rglob('*') if p.is_file())
manifest=B/D['12']/'来源与校验清单.json'
for f in sorted(set(files+[manifest])):
 if f==p:continue
 rel=os.path.relpath(f,p.parent).replace('\\','/');label=f.relative_to(B).as_posix();t+='- ['+label+']('+urllib.parse.quote(rel,safe='/._-')+')\n'
p.write_text(t,encoding='utf8')
# Every one of 144 original evidence records keeps its source, raw hash and normalized-content hash.
evidence=[]
for e in P['evidence144']:
 old='图册配套依据/'+e['file'];assert old in P['final'],('未覆盖的原证据',old)
 row=dict(e);row['previous_package_file']=old;row['current_file']=P['final'][old];row['normalized_sha256']=P['metadata'][old]['normalized_sha256'];assert (B/row['current_file']).is_file();evidence.append(row)
origins=collections.defaultdict(list)
for old,final in P['final'].items():
 if not (B/final).is_file():continue
 row={'previous_package_file':old,**P['metadata'][old]};origins[final].append(row)
for r in P['top45']:
 old=r['成果相对路径'].replace('\\','/');final=P['final'].get(old)
 if final:origins[final].append({'project_source':r['项目内原始路径'],'original_sha256':r['SHA256'].lower()})
for r in P['ref26']:
 old=r['参考资料路径'].replace('\\','/');final=P['final'].get(old)
 if final and (B/final).is_file():origins[final].append({'project_source':r['项目内来源'],'original_sha256':r['SHA256'].lower()})
entries=[]
for f in sorted(p for p in B.rglob('*') if p.is_file() and p!=manifest):
 n=f.relative_to(B).as_posix();entries.append({'file':n,'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'origins':origins.get(n,[{'source':'2026-11-15依据现有成果整理或修订；未新增实验成绩'}])})
obj={'schema_version':2,'updated':'2026-11-15','scope':'现存文件的唯一有效来源/哈希清单；不含本清单自身，避免自引用哈希','normalization':'仅文本UTF-8 BOM及CRLF/LF；无数值忽略','files':entries,'original_evidence_144':evidence,'reversible_text_edits':edits,'invalid_previous_reference_paths':[r['参考资料路径'] for r in P['ref26'] if r['参考资料路径'].replace('\\','/') not in P['metadata']]}
manifest.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf8');print('MANIFEST',len(entries),'files plus manifest;',len(evidence),'original evidence records;',len(obj['invalid_previous_reference_paths']),'stale paths retired')

