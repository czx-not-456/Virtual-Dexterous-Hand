from pathlib import Path
import json,subprocess,pypdfium2 as pdfium
O=Path(__file__).resolve().parent;P=json.loads((O/'整理操作计划.json').read_text(encoding='utf8'));B=Path(P['base']);D=P['dirs'];V=O/'版式检查'
for no in ['09','10']:
 path=next((B/D[no]).glob('*.html'));out=V/(no+'_展示页.pdf')
 cmd=['C:/Program Files/Google/Chrome/Application/chrome.exe','--headless','--disable-gpu','--no-first-run','--no-default-browser-check','--allow-file-access-from-files','--no-pdf-header-footer','--user-data-dir='+str(V/('chrome_'+no)),'--print-to-pdf='+str(out),path.as_uri()]
 r=subprocess.run(cmd,capture_output=True,timeout=45,creationflags=subprocess.CREATE_NO_WINDOW);assert out.is_file(),r.stderr.decode(errors='replace')[-1000:]
 pdf=pdfium.PdfDocument(out);print(no,'PAGES',len(pdf));assert len(pdf)==1
 pdf[0].render(scale=1.25).to_pil().save(V/(no+'_展示页.png'));pdf.close()
pdf=pdfium.PdfDocument(B/D['11']/'11_当前问题与下一阶段计划.pdf');pdf[0].render(scale=1.8).to_pil().save(V/'计划页_PDF渲染.png');pdf.close()
