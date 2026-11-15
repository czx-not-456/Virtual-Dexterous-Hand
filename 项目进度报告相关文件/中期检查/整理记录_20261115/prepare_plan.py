from pathlib import Path
import json,hashlib,csv,collections
B=Path('项目进度报告相关文件/中期检查/项目中期成果').resolve(); O=B.parent/'整理记录_20261115';O.mkdir(exist_ok=True)
D={p.name[:2]:p.name for p in B.iterdir() if p.is_dir() and p.name[:2].isdigit()}
T={'.csv','.json','.md','.txt','.log','.py','.ps1','.yaml','.toml','.xml','.html'}
def norm(p):
 raw=p.read_bytes()
 return p.read_text(encoding='utf-8-sig').replace('\r\n','\n').encode() if p.suffix.lower() in T else raw
files=sorted(p for p in B.rglob('*') if p.is_file());meta={p.relative_to(B).as_posix():{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'normalized_sha256':hashlib.sha256(norm(p)).hexdigest(),'bytes':p.stat().st_size} for p in files}
old_evidence=json.loads((B/'图册配套依据/原件来源及哈希.json').read_text(encoding='utf8'));old_top=list(csv.DictReader((B/'文件来源与校验清单.csv').open(encoding='utf-8-sig')));old_ref=list(csv.DictReader((B/'待制作材料_参考资料来源清单.csv').open(encoding='utf-8-sig')))
# First collapse identical normalized files, preserving canonical historical results.
groups=collections.defaultdict(list)
for n,m in meta.items():groups[m['normalized_sha256']].append(n)
rep={n:n for n in meta};deletes={};moves={}
for v in groups.values():
 if len(v)<2:continue
 keep=next((n for n in v if n.startswith('图册配套依据/01_系统与测试/关键代码快照/README.md')),v[0])
 for n in v:
  rep[n]=keep
  if n!=keep:deletes[n]='重复，保留 '+keep
# Fully sourced replacement screenshots supersede anonymous display exports.
for n in meta:
 if n.startswith(D['05']+'/') or n.startswith(D['08']+'/'):
  deletes[n]='展示图缺少可靠运行溯源，改用已核验、带索引的现有材料'
for n in ['待制作展示材料说明.md','待制作材料_参考资料来源清单.csv','文件来源与校验清单.csv','图册配套依据/00_成果依据索引.docx','图册配套依据/原件来源及哈希.json','图册配套依据/本次独立数值核验记录.txt',D['02']+'/benchmark_dashboard.html']:
 deletes[n]='过时或重复入口，内容由新说明、统一来源清单及展示页替代'
# Assign unique evidence to numbered tasks.
for n in meta:
 if n in deletes:continue
 dest=n
 prefix='图册配套依据/'
 if n.startswith(prefix+'01_系统与测试/07_系统流程.png'):dest=D['07']+'/07_系统流程.png'
 elif n.startswith(prefix+'01_系统与测试/关键代码快照/') or n.startswith(prefix+'01_系统与测试/测试与校验/'):
  dest=D['01']+'/补充运行_20261115/'+n.split('01_系统与测试/',1)[1]
 elif n.startswith(prefix+'01_系统与测试/原复现脚本/') or n.startswith(prefix+'02_MuJoCo实测/原录制脚本/'):
  dest=D['12']+'/复现参考脚本/'+Path(n).name
 elif n.startswith(prefix+'02_MuJoCo实测/'):
  task=Path(n).parts[2];name=Path(n).name
  if n.endswith('.png') or name=='截图索引.csv':dest=D['05']+'/'+task+'/'+name
  elif name in ['任务演示_慢放.mp4','录制说明.json']:dest=D['06']+'/'+task+'/'+name
  else:dest=D['02']+'/补充运行_20261115/'+task+'/'+name
 elif n.startswith(prefix+'04-06_历史任务与消融/StageB消融/原代码/'):
  dest=D['03']+'/原代码/'+Path(n).name
 elif n.startswith(prefix+'04-06_历史任务与消融/重分析结果/'):
  name=Path(n).name
  number=name[:2];dest=D[number]+'/'+name
 elif n.startswith(D['06']+'/现成参考资料/CAMERA_CONTROLS.md'):dest=D['07']+'/CAMERA_CONTROLS.md'
 elif n.startswith(D['12']+'/现成参考资料/'):dest=D['12']+'/历史项目导航/'+Path(n).name
 if dest!=n:moves[n]=dest
# Resolve every deleted duplicate to final canonical path.
final={n:moves.get(rep[n],rep[n]) for n in meta if rep[n] not in deletes}
plan={'base':str(B),'dirs':D,'metadata':meta,'delete':deletes,'move':moves,'final':final,'evidence144':old_evidence,'top45':old_top,'ref26':old_ref}
(O/'整理操作计划.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf8')
print('BEFORE',len(files),'DELETIONS',len(deletes),'MOVES',len(moves),'DUPLICATE_REMOVALS',sum('重复，' in v for v in deletes.values()))
# Validate all mutation paths and collisions before execution.
for n in list(deletes)+list(moves)+list(moves.values()):
 p=(B/n).resolve();assert p.is_relative_to(B) and p!=B,n
assert len(set(moves.values()))==len(moves)
for src,dst in moves.items():assert not (B/dst).exists() or dst==src,(src,dst)
print('PATHS_AND_DESTINATIONS_VALID')
