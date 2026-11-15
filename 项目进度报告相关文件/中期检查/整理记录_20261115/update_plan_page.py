from pathlib import Path
import json
from PIL import Image,ImageDraw,ImageFont
O=Path(__file__).resolve().parent;P=json.loads((O/'整理操作计划.json').read_text(encoding='utf8'));B=Path(P['base']);D=P['dirs'];rows=json.loads((O/'计划页内容.json').read_text(encoding='utf8'))
rows=[[t.replace('Stage B','阶段B').replace('Stage C','阶段C') for t in row] for row in rows]
im=Image.new('RGB',(1920,1350),'white');d=ImageDraw.Draw(im)
def f(n,bold=False):return ImageFont.truetype('C:/Windows/Fonts/msyhbd.ttc' if bold else 'C:/Windows/Fonts/msyh.ttc',n)
ink='#203b47';d.text((80,55),'当前问题与下一阶段计划',font=f(60,True),fill=ink);d.text((82,150),'虚拟灵巧手中期成果  |  2026-11-15修订  |  依据已保存结果，不新增实验结论',font=f(27),fill=ink)
widths=[290,450,540,480];xs=[80]
for w in widths:xs.append(xs[-1]+w)
def wrap(t,font,width):
 lines=[];s=''
 for c in t:
  if d.textlength(s+c,font=font)>width:lines.append(s);s=c
  else:s+=c
 if s:lines.append(s)
 return lines
headers=['工作方向','当前问题','下一阶段工作','验收依据'];y=230;d.rectangle((80,y,1840,y+82),fill='#145568')
for j,t in enumerate(headers):d.text((xs[j]+18,y+20),t,font=f(32,True),fill='white')
y+=82
for i,row in enumerate(rows):
 d.rectangle((80,y,1840,y+210),fill='#eef4f6' if i%2==0 else '#f7f9fa')
 for j,t in enumerate(row):
  font=f(29,j==0);lines=wrap(t,font,widths[j]-40);assert len(lines)<=5,(i,j,lines)
  for k,line in enumerate(lines):d.text((xs[j]+18,y+25+k*35),line,font=font,fill=ink)
 y+=210
d.text((82,1190),'材料来源：02历史Benchmark及补充运行、03 Stage B、01历史系统审查；详细链接见同目录“计划说明.md”。',font=f(24),fill=ink)
d.text((82,1240),'负责人和时间节点尚未指定；真实手套通信、动态泛化与完整Stage C验证仍待开展。',font=f(24),fill=ink)
pre=B/D['11']/'11_当前问题与下一阶段计划';im.save(str(pre)+'.png');im.save(str(pre)+'.pdf','PDF',resolution=160.0)
print('PLAN_PNG_AND_PDF_UPDATED')

