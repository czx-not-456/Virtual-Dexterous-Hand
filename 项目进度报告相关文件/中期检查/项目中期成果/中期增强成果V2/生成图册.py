"""Build the figure atlas from results, after running 复现实验.py and 验证成果.py."""
from pathlib import Path
import json
import pandas as pd
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement

HERE=Path(__file__).resolve().parent
O=HERE/"结果"
info=json.loads((O/"summary.json").read_text(encoding="utf-8"))
rec=info["recognition"]; sr=pd.DataFrame(rec["summary"])
bench=pd.DataFrame(info["benchmark"]).set_index("task")
sm=pd.DataFrame(info["smooth"]).set_index("alpha")
tests=pd.read_csv(O/"11_测试记录.csv")
doc=Document()
sec=doc.sections[0];sec.page_width=Inches(8.5);sec.page_height=Inches(11)
sec.left_margin=sec.right_margin=Inches(.76);sec.top_margin=Inches(.66);sec.bottom_margin=Inches(.65)
for name in ["Normal","Title","Heading 1","Heading 2","Caption"]:
    st=doc.styles[name];st.font.name="Calibri";st.font.color.rgb=RGBColor(0,0,0)
    st.element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"),"Noto Sans CJK SC")
    st.font.size=Pt({"Normal":11,"Title":23,"Heading 1":17,"Heading 2":13,"Caption":9}[name])
    st.paragraph_format.space_after=Pt(5)
    st.paragraph_format.line_spacing=1.13
for name in ["Heading 1","Heading 2"]:
    doc.styles[name].paragraph_format.space_before=Pt(9)
doc.styles["Title"].font.bold=True
for border in doc.styles["Title"].element.xpath(".//w:pBdr"):
    border.getparent().remove(border)

def p(text,style=None):
    return doc.add_paragraph(text,style=style)
def head(text):doc.add_heading(text,level=1)
def fig(name,caption,width=6.85):
    para=doc.add_paragraph();para.alignment=WD_ALIGN_PARAGRAPH.CENTER
    para.paragraph_format.space_after=Pt(2)
    r=para.add_run();r.add_picture(str(O/name),width=Inches(width))
    drawing=r._r.xpath(".//wp:docPr")[0];drawing.set("descr",caption)
    cap=p(caption,"Caption");cap.alignment=WD_ALIGN_PARAGRAPH.CENTER
def table(headers,rows):
    t=doc.add_table(rows=1,cols=len(headers));t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.style="Table Grid"
    for c,v in zip(t.rows[0].cells,headers):c.text=str(v)
    for row in rows:
        for c,v in zip(t.add_row().cells,row):c.text=str(v)
    for i,row in enumerate(t.rows):
        for c in row.cells:
            c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            pr=c._tc.get_or_add_tcPr();sh=OxmlElement("w:shd");sh.set(qn("w:fill"),"DDEAF3" if i==0 else ("F4F7FA" if i%2==0 else "FFFFFF"));pr.append(sh)
            for pp in c.paragraphs:
                pp.paragraph_format.space_after=Pt(4);pp.paragraph_format.space_before=Pt(4)
                pp.alignment=WD_ALIGN_PARAGRAPH.CENTER
                for r in pp.runs:r.font.size=Pt(10);r.bold=(i==0)
    borders=OxmlElement("w:tblBorders")
    for side in ("top","bottom","left","right","insideH","insideV"):
        el=OxmlElement("w:"+side);el.set(qn("w:val"),"single");el.set(qn("w:sz"),"4");el.set(qn("w:color"),"D9D9D9");borders.append(el)
    t._tbl.tblPr.append(borders)
def page():doc.add_page_break()
def acc(k,m):return float(sr[(sr.missing==k)&(sr.method==m)].accuracy.iloc[0])

p("虚拟灵巧手项目中期增强成果图册","Title")
p("河海大学大学生创新创业训练计划项目  |  中期检查补充材料 V2.0")
p("项目：基于数据手套的手部动作采集与虚拟灵巧手交互系统设计")
head("01  总体成果与系统框架")
p("现阶段已形成数据采集、预处理、动作映射、虚拟手输出及量化评估链路。本次新增按设备留出的六类标定姿态识别与传感器缺失压力验证，并对历史仿真日志开展三维姿态重建和统计分析。")
fig("07_系统流程.png","图 1  工程模块与新增验证流程")
table(["证据对象","规模","用途"],[
    ["手套采集记录",f"{rec['sessions']} 组 / {rec['devices']} 台设备 / {rec['frames']} 帧","设备留出、鲁棒性"],
    ["历史 MuJoCo 日志","3 任务 × 3 次","接触、稳定性、延迟"],
    ["合成输入","8 条 × 800 帧","滤波压力验证"],
    ["工程与成果检查",f"{sum(tests.status=='PASS')}/{len(tests)} 项通过","功能及数据完整性"]])
p("说明：三维关键帧为历史关节向量驱动原 CH-M6 网格的运动学重建；本次未重新运行 MuJoCo 物理仿真。姿态识别是标定标签的离线识别，不等同连续动态手势识别。")

page();head("02  捏合与抓握的三维姿态展示")
p("依据 MJCF 的关节轴、层级关系和原始 STL 网格，读取历史 trial 00 的 11 关节角度进行正向运动学重建。统一相机和比例，展示初始、任务开始、中段、结束四个关键时刻。")
fig("05_pinch_三维姿态重建.png","图 2  PINCH 关键帧（关节数据驱动网格重建）")
fig("05_wrap_三维姿态重建.png","图 3  WRAP 关键帧（关节数据驱动网格重建）")
p("配套 GIF 记录两个任务的历史姿态变化，便于现场离线预览。由于日志未包含完整物体位姿，本重建不呈现物体、碰撞或接触力；现场可通过实时演示脚本运行原 MuJoCo 环境。")

page();head("03  任务时序与稳定性分析")
fig("08_pinch_任务时序.png","图 4  PINCH trial 00 历史日志，绿色背景为有效区间",6.6)
p(f"三次捏合的有效帧平均接触代理占比为 {bench.loc['pinch','mean_contact_ratio']:.2%}，而稳定代理占比仅 {bench.loc['pinch','mean_stable_ratio']:.2%}。虽能建立接触，但持续稳定性仍是主要改进点。")
p("延迟为历史日志中的算法链路处理延迟，不能等同端到端硬件响应时间。")

page();head("04  抓握时序及九次试验汇总")
fig("08_wrap_任务时序.png","图 5  WRAP trial 00 历史日志",6.3)
table(["任务","接触代理通过","稳定代理通过","平均稳定帧占比","平均试次P95"],[
    [k.upper(),f"{int(r.contact_pass)}/3",f"{int(r.stable_pass)}/3",f"{r.mean_stable_ratio:.2%}",f"{r.mean_p95_ms:.1f} ms"]
    for k,r in bench.loc[["neutral","pinch","wrap"]].iterrows()])
p("NEUTRAL 为无抓握对照，接触和稳定代理不是其目标。PINCH 与 WRAP 的接触代理均 3/3 通过，只有 WRAP 的稳定代理 3/3 通过；这并非实体提举成功率。")

page();head("05  Stage B 缺失条件消融与历史实验")
fig("09_九次试验汇总.png","图 6  三任务各三次历史试验的代理占比与延迟")
fig("10_StageB七方法对比.png","图 7  既有 Stage B 七方法消融结果重绘")
p("历史 Stage B 比较六维软件教师标签归一化 MAE。五路缺失时，ridge6_robust 为 0.12188，低于 ridge6_original 的 0.12652；校准融合与鲁棒单模型接近，未体现明显额外增益。软件教师标签并非实体机器人真值。")

page();head("06  新增六类姿态识别与缺失鲁棒性")
p(f"从 {rec['sessions']} 组 captured_samples.csv 中筛出六类共有姿态，共 {rec['frames']} 帧，按 {rec['devices']} 台设备序列号逐一留出；标准化与原型均仅由训练设备拟合，以避免同设备相邻帧跨训练与测试。")
fig("13_六类手势混淆矩阵.png","图 8  无缺失预测混淆矩阵（行内占比及帧数）",5.6)
table(["指标","无缺失","缺1路","缺3路","缺5路"],[
  [m]+[f"{acc(k,m):.2%}" for k in (0,1,3,5)] for m in ["均值填补","可用通道距离"]])
p(f"无缺失时，按设备等权准确率为 {acc(0,'可用通道距离'):.2%}，逐帧汇总为 {rec['pooled_accuracy']:.2%}。注意这些是静态标定记录标签，不是独立参与者或动态任务真值。")

page();head("07  缺失压力结果与设备差异")
fig("13_传感器缺失鲁棒性.png","图 9  设备等权准确率（误差线为设备间标准差）")
fig("13_设备泛化差异.png","图 10  留出各设备的无缺失准确率")
p(f"同一设备的两种方法使用相同掩码；各非零缺失条件重复 10 次。五路缺失时，可用通道距离法达 {acc(5,'可用通道距离'):.2%}，比均值填补的 {acc(5,'均值填补'):.2%} 高 {(acc(5,'可用通道距离')-acc(5,'均值填补'))*100:.2f} 个百分点。缺失为人为屏蔽，尚未验证真实传感器故障。")

page();head("08  滤波抗扰与下一阶段工作")
fig("14_滤波平滑与偏差.png","图 11  合成输入噪声压力测试（8 条序列）",6.6)
p(f"在每通道人为加入标准差为原序列标准差 12% 的高斯噪声后，α=0.7 的归一化 RMSE 为 {sm.loc[.7,'nrmse']:.4f}，低于不滤波的 {sm.loc[1.,'nrmse']:.4f}；α=0.25 虽最平滑，却产生更大的偏差。原序列是合成参考，不是物理真值。")
p("下一阶段：优先优化拇指与食指接触稳定性，扩大随机物体位姿与动态动作测试，验证真实传感器缺失及在线识别，再开展实体持续持物和提举验证。")
p(f"复现：复现实验.py 生成图表、CSV 与原始数据来源 SHA256；验证成果.py 执行 {len(tests)} 项检查，当前 {sum(tests.status=='PASS')} 项通过。实时演示.ps1 用于 Windows MuJoCo 现场展示。")
doc.save(HERE/"中期检查_增强成果图册V2.docx")
