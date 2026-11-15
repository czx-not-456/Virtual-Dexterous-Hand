from pathlib import Path
import json,hashlib,collections,re,urllib.parse,zipfile,xml.etree.ElementTree as E,csv,ast
from html.parser import HTMLParser
from PIL import Image
import cv2
O=Path(__file__).resolve().parent;P=json.loads((O/'整理操作计划.json').read_text(encoding='utf8'));B=Path(P['base']);D=P['dirs'];errors=[];groups=collections.defaultdict(list);links=0;imgs=0;frames=0
T={'.csv','.json','.md','.txt','.log','.py','.ps1','.yaml','.toml','.xml','.html'}
def check(target,parent,source):
 global links
 target=urllib.parse.unquote(target)
 if re.match(r'^(https?:|mailto:|#)',target):return
 if not target:return
 links+=1;p=(parent/target.split('#')[0]).resolve()
 if not p.exists():errors.append(('LINK',source,target))
class Parse(HTMLParser):
 def __init__(self,p):super().__init__();self.p=p
 def handle_starttag(self,tag,attrs):
  a=dict(attrs)
  for k in ['href','src']:
   if k in a:check(a[k],self.p.parent,self.p.relative_to(B).as_posix())
for p in B.rglob('*'):
 if not p.is_file():continue
 n=p.relative_to(B).as_posix();raw=p.read_bytes();suffix=p.suffix.lower()
 if suffix in T:
  text=p.read_text(encoding='utf-8-sig').replace('\r\n','\n');raw=text.encode()
  if suffix=='.json':json.loads(text)
  if suffix=='.py':ast.parse(text)
  if suffix=='.csv':
   rows=list(csv.reader(text.splitlines()));assert all(len(r)==len(rows[0]) for r in rows),n
  if suffix=='.md':
   for label,target in re.findall(r'\[([^\]\n]+)\]\(([^)\n]+)\)',text):check(target,p.parent,n)
  if suffix=='.html':Parse(p).feed(text)
 groups[hashlib.sha256(raw).hexdigest()].append(n)
 if suffix in ['.png','.gif']:
  im=Image.open(p)
  for i in range(getattr(im,'n_frames',1)):im.seek(i);im.load()
  imgs+=1
 if suffix=='.mp4':
  cap=cv2.VideoCapture(str(p));expected=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS);count=0
  while True:
   ok,frame=cap.read()
   if not ok:break
   assert frame is not None;count+=1
  cap.release();assert count==expected==400 and fps==10,(n,count,expected,fps);frames+=count
 if suffix=='.docx':
  with zipfile.ZipFile(p) as z:
   assert z.testzip() is None
   for name in z.namelist():
    if name.endswith('.xml'):E.fromstring(z.read(name))
   for r in E.fromstring(z.read('word/_rels/document.xml.rels')):
    if r.attrib.get('TargetMode')=='External':check(r.attrib['Target'],p.parent,n)
dup=[v for v in groups.values() if len(v)>1]
assert not dup,dup
for n in list(D.values())+['图册配套依据']:assert (B/n).is_dir()
assert (B/'项目中期成果说明.md').is_file() and (B/'中期检查_成果图册.docx').is_file()
empty=[p.relative_to(B).as_posix() for p in B.rglob('*') if p.is_dir() and not any(p.iterdir())]
assert not empty,empty
report={'files':sum(p.is_file() for p in B.rglob('*')),'bytes':sum(p.stat().st_size for p in B.rglob('*') if p.is_file()),'normalized_duplicate_groups':dup,'checked_links':links,'link_errors':errors,'decoded_images_and_gifs':imgs,'decoded_video_frames':frames,'empty_directories':empty,'required_structure':'PASS'}
(O/'结构链接及媒体检查.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(report,ensure_ascii=False));assert not errors
