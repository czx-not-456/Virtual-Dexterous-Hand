from pathlib import Path
import json,zipfile,html,urllib.parse,re
O=Path(__file__).resolve().parent;P=json.loads((O/'整理操作计划.json').read_text(encoding='utf8'));B=Path(P['base']);D=P['dirs']
p=B/D['12']/'复现说明.md';p.write_text(p.read_text(encoding='utf8').replace('--trial 0','--dataset-trial 0'),encoding='utf8')
p=B/'中期检查_成果图册.docx'
with zipfile.ZipFile(p) as z:parts={n:z.read(n) for n in z.namelist()}
x=parts['word/document.xml'].decode('utf8');r=parts['word/_rels/document.xml.rels'].decode('utf8')
x=x.replace('使用MuJoCo 3.3.7及CH-M6虚拟手','2026年11月15日补充运行，使用MuJoCo 3.3.7及CH-M6虚拟手')
ns='http://schemas.openxmlformats.org/wordprocessingml/2006/main';esc=lambda s:html.escape(s,quote=True)
def para(s,size=22,bold=False):return '<w:p><w:pPr><w:spacing w:after="120"/></w:pPr><w:r><w:rPr><w:rFonts w:ascii="Arial" w:eastAsia="Microsoft YaHei"/><w:sz w:val="'+str(size)+'"/>'+('<w:b/>' if bold else '')+'</w:rPr><w:t>'+esc(s)+'</w:t></w:r></w:p>'
append='<w:p><w:r><w:br w:type="page"/></w:r></w:p>'+para('成果附件导航',30,True)+para('资料按01至12归类，配套依据保留独有数据和模型。链接相对于本图册所在目录。')
targets=[('成果范围与指标口径','项目中期成果说明.md')]+[(name,name+'/') for _,name in sorted(D.items())]+[('图册章节与完整文件索引',D['12']+'/附件目录.md'),('可核验数据与复现说明',D['12']+'/复现说明.md')]
ids=[int(v) for v in re.findall(r'Id="rId(\d+)"',r)];start=max(ids+[0])+1
for i,(label,target) in enumerate(targets,start):
 rid='rId'+str(i);uri=urllib.parse.quote(target,safe='/._-')
 rel='<Relationship Id="'+rid+'" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink" Target="'+uri+'" TargetMode="External"/>'
 r=r.replace('</Relationships>',rel+'</Relationships>')
 append+='<w:p><w:pPr><w:spacing w:after="95"/></w:pPr><w:hyperlink r:id="'+rid+'"><w:r><w:rPr><w:rFonts w:eastAsia="Microsoft YaHei"/><w:color w:val="176F91"/><w:sz w:val="22"/><w:u w:val="single"/></w:rPr><w:t>'+esc(label)+'</w:t></w:r></w:hyperlink></w:p>'
append+=para('历史稳定指标与新版滚动窗口定义不同；接触帧占比不是独立抓取成功概率。材料齐备不等于硬件在线通信、连续动作泛化或提举任务已经完成。',20)
pos=x.rfind('<w:sectPr');assert pos>0;x=x[:pos]+append+x[pos:]
parts['word/document.xml']=x.encode();parts['word/_rels/document.xml.rels']=r.encode()
with zipfile.ZipFile(p,'w',compression=zipfile.ZIP_DEFLATED) as z:
 for n,data in parts.items():z.writestr(n,data)
print('DOCX_UPDATED',len(targets),'relative navigation links')
