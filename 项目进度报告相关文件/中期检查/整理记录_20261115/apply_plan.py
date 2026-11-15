from pathlib import Path
import json,hashlib
O=Path(__file__).resolve().parent;plan=json.loads((O/'整理操作计划.json').read_text(encoding='utf8'));B=Path(plan['base']).resolve();actions=[]
def checked(n):
 p=(B/n).resolve();assert p.is_relative_to(B) and p!=B;return p
# Check originals have not changed since the concrete plan was prepared.
for n,m in plan['metadata'].items():assert hashlib.sha256(checked(n).read_bytes()).hexdigest()==m['sha256'],n
for src,dst in plan['move'].items():
 p,q=checked(src),checked(dst);assert not q.exists();q.parent.mkdir(parents=True,exist_ok=True);p.rename(q);actions.append({'action':'move','from':src,'to':dst})
for n,reason in plan['delete'].items():
 p=checked(n);assert p.is_file();p.unlink();actions.append({'action':'delete','from':n,'reason':reason})
protected={checked(n) for n in plan['dirs'].values()}|{checked('图册配套依据')}
for p in sorted((p for p in B.rglob('*') if p.is_dir()),key=lambda p:len(p.parts),reverse=True):
 assert p.resolve().is_relative_to(B)
 if p not in protected and not any(p.iterdir()):p.rmdir()
(O/'实际操作记录.json').write_text(json.dumps(actions,ensure_ascii=False,indent=2),encoding='utf8');print('DONE',len(actions),'remaining files',sum(p.is_file() for p in B.rglob('*')))
