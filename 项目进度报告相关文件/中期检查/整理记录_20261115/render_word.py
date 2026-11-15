from pathlib import Path
import json,win32com.client
O=Path(__file__).resolve().parent;P=json.loads((O/'整理操作计划.json').read_text(encoding='utf8'));B=Path(P['base']);out=O/'版式检查';out.mkdir(exist_ok=True);app=None;doc=None
try:
 app=win32com.client.DispatchEx('Word.Application');app.Visible=False;app.DisplayAlerts=0
 doc=app.Documents.Open(str(B/'中期检查_成果图册.docx'),ReadOnly=True,AddToRecentFiles=False,Visible=False)
 doc.ExportAsFixedFormat(str(out/'成果图册_版式检查.pdf'),17)
 print('DOCX_RENDER_PAGES',doc.ComputeStatistics(2))
finally:
 if doc is not None:doc.Close(False)
 if app is not None:app.Quit()
