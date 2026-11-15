from pathlib import Path
import json,pypdfium2 as pdfium
from PIL import Image,ImageDraw,ImageFont
O=Path(__file__).resolve().parent;P=json.loads((O/'整理操作计划.json').read_text(encoding='utf8'));B=Path(P['base']);D=P['dirs'];V=O/'版式检查';pdf=pdfium.PdfDocument(V/'成果图册_版式检查.pdf')
for i in range(len(pdf)):
 p=pdf[i];im=p.render(scale=1.4).to_pil();im.save(V/f'图册_page_{i+1:02d}.png');t=p.get_textpage().get_text_range();(V/f'图册_page_{i+1:02d}.txt').write_text(t,encoding='utf8');print('PAGE',i+1,len(t),t[:35].replace('\r',' ').replace('\n',' '))
pdf.close()
pdf=pdfium.PdfDocument(B/D['11']/'11_当前问题与下一阶段计划.pdf');assert len(pdf)==1;pdf[0].render(scale=1.8).to_pil().save(V/'计划页_PDF渲染.png');pdf.close()
print('PDF_RENDER_COMPLETE')
