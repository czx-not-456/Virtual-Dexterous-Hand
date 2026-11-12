from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
import zipfile,os,shutil
root=Path(__file__).resolve().parent.parent;qa=root/'.文档修改校验'
s=(root/'revise_midterm.py').read_text(encoding='utf8')
exec(s[s.index('# Rebuild code-generated'):s.index('def put(p,s):')])
for f in [root/'中期检查/修订稿/中期检查报告.docx',root/'中期检查/中期检查报告.docx']:
    tmp=f.with_suffix('.qa.docx')
    with zipfile.ZipFile(f) as src,zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as dst:
        for item in src.infolist():
            data=src.read(item.filename)
            if item.filename=='word/media/image2.png':data=(qa/'figure2.png').read_bytes()
            dst.writestr(item,data)
    os.replace(tmp,f)
try:
    shutil.copy2(root/'中期检查/修订稿/1_已取得成果.docx',root/'中期检查/1_已取得成果.docx')
    print('第一份原文件已更新')
except PermissionError:print('第一份仍被占用；其余五份原文件已更新')
