from pathlib import Path
import shutil, re, json
from datetime import datetime
from docx import Document
from docx.text.paragraph import Paragraph
from docx.oxml.ns import qn
from PIL import Image, ImageDraw, ImageFont

root=Path(__file__).parent
folder=root/'中期检查'
backup=folder/('修改前备份_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
backup.mkdir()
files=[f for f in folder.glob('*.docx') if not f.name.startswith('~$')]
for f in files: shutil.copy2(f,backup/f.name)
output=folder/'修订稿';output.mkdir(exist_ok=True)
qa=root/'.文档修改校验'; qa.mkdir(exist_ok=True)
log=[]
# Rebuild code-generated diagrams, preserving the document's image dimensions.
fontpath='C:/Windows/Fonts/msyh.ttc'
def font(n):return ImageFont.truetype(fontpath,n)
def label(draw,xy,text,size=28,fill='#183954',anchor='mm'):
    draw.text(xy,text,font=font(size),fill=fill,anchor=anchor)
im=Image.new('RGB',(2000,840),'white');d=ImageDraw.Draw(im)
boxes=[(35,80,625,310),(705,80,1295,310),(1375,80,1965,310),(1375,510,1965,740),(705,510,1295,740),(35,510,625,740)]
titles=['离线数据输入','标定与滤波','姿态解算与分层特征','意图识别','意图驱动动作映射','MuJoCo 仿真与评测']
subs=['合成数据／实采标定文件回放','量程归一化／平滑／输出低通','15 维语义关节／指尖位置','中性／捏合／包络','15→11 关节目标适配','接触与稳定代理／耗时']
for b,t,s in zip(boxes,titles,subs):
    d.rounded_rectangle(b,20,fill='#eef3f8',outline='#426f9b',width=4)
    cx=(b[0]+b[2])//2;cy=(b[1]+b[3])//2
    label(d,(cx,cy-30),t,36);label(d,(cx,cy+40),s,27)
for a,b in [((625,195),(705,195)),((1295,195),(1375,195)),((1670,310),(1670,510)),((1375,625),(1295,625)),((705,625),(625,625))]:
    d.line([a,b],fill='#426f9b',width=6)
    x,y=b;dx=b[0]-a[0];dy=b[1]-a[1]
    pts=[(x,y),(x-18,y-12),(x-18,y+12)] if dx>0 else ([(x,y),(x+18,y-12),(x+18,y+12)] if dx<0 else [(x,y),(x-12,y-18),(x+12,y-18)])
    d.polygon(pts,fill='#426f9b')
label(d,(1000,405),'真实手套连续采集：待接入；60 Hz 为配置频率，实时性尚未达标',31)
im.save(qa/'figure1.png')
im=Image.new('RGB',(2000,840),'white');d=ImageDraw.Draw(im)
label(d,(510,50),'(a) 历史仿真代理成功率',37,fill='black')
label(d,(1510,50),'(b) 单帧耗时 P95 跨运行均值',37,fill='black')
for left,maxval,unit in [(120,100,'成功率（%）'),(1120,140,'耗时（ms）')]:
    bottom=655;top=190;right=left+780
    label(d,(left+70,105),unit,26,fill='black')
    d.line([(left,top),(left,bottom),(right,bottom)],fill='black',width=3)
    for v in range(0,maxval+1,20):
        y=bottom-(bottom-top)*v/maxval
        d.line([(left-8,y),(right,y)],fill='#dddddd',width=1)
        label(d,(left-15,y),str(v),25,fill='black',anchor='rm')
    for j,name in enumerate(['中性','捏合','包络']):label(d,(left+140+j*250,700),name,30,fill='black')
for j,vals in enumerate([(None,None),(100,0),(100,100)]):
    x=120+140+j*250
    if j==0:label(d,(x,590),'N/A',32,fill='#555555');continue
    for k,v in enumerate(vals):
        xx=x-85+k*85;y=655-465*v/100
        if v:d.rectangle((xx,y,xx+75,655),fill=['#a0c2da','#306a99'][k])
        label(d,(xx+37,y-23),str(v),27,fill='black')
for j,v in enumerate([123.06,116.21,119.36]):
    x=1120+140+j*250;y=655-465*v/140
    d.rectangle((x-65,y,x+65,655),fill='#80a6c3');label(d,(x,y-23),f'{v:.1f}',28,fill='black')
y=655-465*16.67/140
for x in range(1120,1900,30):d.line((x,y,x+18,y),fill='#bd4141',width=4)
label(d,(1520,y-25),'16.67 ms 帧预算',25,fill='#bd4141')
d.rectangle((200,110,230,135),fill='#a0c2da');label(d,(245,123),'接触代理',24,anchor='lm')
d.rectangle((470,110,500,135),fill='#306a99');label(d,(515,123),'历史稳定接触代理',24,anchor='lm')
label(d,(1000,770),'每任务 3 次 × 800 帧；成功率按运行次数统计；中性任务抓取指标不适用。',28,fill='black')
label(d,(1000,813),'历史结果不代表释放夹具后连续一秒稳定抓握代理已通过验证。',27,fill='#555555')
im.save(qa/'figure2.png')
def put(p,s):
    old=p.text
    if old==s:return
    if p.runs:
        p.runs[0].text=s
        for r in p.runs[1:]:r.text=''
    else:p.add_run(s)
    log.append({'file':current,'before':old,'after':s})

plans=[
'2026 年 11 月—2027 年 1 月：2026 年 12 月前完成设备接口与 CH-M6 参数资料的初步核对；至 2027 年 1 月完成手套 SDK 或通信协议接入、连续采集、时间戳同步及掉帧处理，形成自然连续动作数据。分阶段计时，先以单帧管线耗时 P95 不超过 33 ms 为目标，再向 16.67 ms 优化，并同步报告精度变化。',
'2027 年 1 月—2027 年 2 月：补齐个性化映射的前向推理实现与预处理定义，完成固定样本输出一致性验证后，在连续动作数据上开展对照；核验 CH-M6 关节运动范围、耦合关系及控制维度，形成参数对照记录。',
'2027 年 2 月—2027 年 3 月：引入 YCB 标准物体模型，扩展抬升、搬运与放置任务；完成方案 C、D、E 及双专家门控的任务级 MuJoCo 对照，区分历史稳定接触判据与释放夹具后连续一秒的稳定抓握代理；补齐 Unity 接收端与展示场景。',
'2027 年 3 月—2027 年 4 月：在统一软硬件条件下开展重复实验，预先明确数据划分、重复次数、成功判据及统计方式，补充跨被试验证，分别报告标定误差、仿真代理指标、单帧耗时和端到端延迟。',
'2027 年 4 月—2027 年 5 月：整理代码、配置与实验数据，按实际完成情况申请软件著作权、完成项目论文与总结报告，准备结题验收。']
expect='姿态参数化表达、运动学模型及动作映射算法已完成阶段性实现，并已打通离线数据驱动的仿真评测流程。真实设备连续采集与 60 Hz 实时驱动目标尚未完成，其中原计划 5 月至 8 月建立的稳定连续采集链路存在延期；当前进度应表述为部分环节滞后，后续优先补齐设备接入与实时性验证。'
common={
'已完成以下成果':'已取得以下阶段性成果',
'完整软件闭环':'离线数据处理与仿真评测流程',
'总体进度按计划进行':'软件与仿真工作已取得阶段性进展，设备接入与实时性验证存在延期',
'释放后连续一秒':'释放外部辅助夹具后连续一秒',
'真实 MuJoCo 集成测试':'MuJoCo 仿真集成测试',
'真实MuJoCo集成测试':'MuJoCo 仿真集成测试',
'五个量':'五类特征',
'方案 E采用':'方案 E 采用','原始XML':'原始 XML',
'2026 年 10 月12 日':'2026 年 10 月 12 日','2026 年11 月':'2026 年 11 月',
'尚未在本环境完成':'尚未完成',
'项目未反序列化运行，仅复算了留出姿态的预测值':'尚未加载权重执行前向推理，目前仅依据交付的预测结果复核留出姿态的误差指标',
'否则映射结果的物理意义有限':'当前结果主要适用于所用仿真模型，向实物系统迁移的有效性仍需验证',
'若与实物不符，映射结果的物理意义有限':'相关参数尚未完成实物一致性核验，当前结果向实物系统迁移的有效性仍需验证',
'投影到 CH-M6 的 11 个执行器':'映射为 CH-M6 模型的 11 维关节目标，并由位置执行器驱动',
'单帧 P95 耗时均值为 116—123 ms':'各任务的单帧耗时 P95 跨运行均值为 116.2—123.1 ms',
'单帧 P95 管线耗时均值为 116—123 ms':'各任务的单帧管线耗时 P95 跨运行均值为 116.2—123.1 ms',
'单帧 P95 耗时均值为 116—123 ms':'各任务的单帧耗时 P95 跨运行均值为 116.2—123.1 ms',
'输出护栏':'输出约束检查','模型输出双重护栏':'模型输出双重约束检查',
'PWL/LoRA 前向推理代码':'交付权重所对应的前向推理代码',
}

for f in files:
    current=f.name; doc=Document(f)
    ps=[Paragraph(el,doc._body) for el in doc.element.body.iter(qn('w:p'))]
    pi=0
    for p in ps:
        s=p.text
        if not s:continue
        if re.match(r'202[67] 年 \d+ 月—2027 年 \d+ 月：',s):
            if pi<5:put(p,plans[pi]);pi+=1
            continue
        if s.startswith('对照申报书 2026 年 9 月至 12 月'):put(p,expect);continue
        for a,b in common.items():s=s.replace(a,b)
        if s.startswith('项目自立项以来'):s=s.replace('已完成以下成果','已取得以下阶段性成果')
        if s.startswith('5. 取得阶段性量化结果'):
            s=s.replace('降低 31.08%；','降低 31.08%，但捏合相对向量误差增加 10.24%，体现多目标权衡；')
        if '190 帧' in s:
            s=s.replace('190 帧实采标定记录','190 帧实采标定记录（来自 27 组快照中的一组；24 帧六姿态样本用于另一组离线对照）')
        if '6 维目标' in s or '六维标定目标' in s:
            s += ' 六维软件命令依据既有关节映射与五项被动耦合关系展开为 11 维模型关节目标；该软件适配不代表实物控制维度已核验。'
        if s.startswith('3. 形成标准任务'):
            s=s.replace('实验数据集','仿真评测数据集').replace('可阅读报告','可读报告')
        if s.startswith('问题：单帧') or s.startswith('问题：各任务'):
            s += ' 该均值由每类任务 3 次运行各自计算 P95 后取平均获得，不是合并全部帧计算的 P95，也不等于端到端交互延迟。'
        if s.startswith('解决方案：对数据处理'):
            s=s.replace('对雅可比与候选姿态做热启动与缓存','采用上一帧解进行热启动、缓存可复用中间量，并评估雅可比更新频率对耗时和精度的影响')
        if s.startswith('1. 建成虚拟'):s=s.replace('建成','形成').replace('实现从数据输入到接触任务评测的离线数据处理与仿真评测流程','打通基于合成数据和实采标定文件回放的数据输入、动作映射与接触任务评测流程')
        if s.startswith('本项目一期获批经费'):
            s='根据现有经费记录，一期三笔金额分别为 2026 年 9 月 29 日的 1730.47 元、66.50 元和 2026 年 10 月 15 日的 551.00 元，合计 2347.97 元。截至中期检查，材料记载已使用及已报销金额均为 2347.97 元，未报销金额为 0 元。三笔记录属于经费拨款还是报销回款、项目获批总额及一期下达额度，应以立项批复和财务台账核定；不能仅据报销合计推定获批总额或账户结余。'
        if s.startswith('一期经费已全部执行完毕'):
            s='后续经费安排以学校实际下达额度和财务台账为依据。优先保障真实设备接入、系统测试、连续动作实验及标准物体等必要耗材支出，再根据成果进度与学校规定安排论文发表、软件著作权登记及必要的专利申请费用。各类成果费用分别按实际业务和财务科目列支，所有支出据实报销、专款专用，并保留票据及项目用途记录。'
        if s=='3. 其它（报销成员入住支出）：551.00 元。':s='3. 其它（原记录为成员入住相关支出，具体用途及财务科目需按票据核实）：551.00 元。'
        if s=='2. 试剂等耗材费（实验耗材采购）：1730.47 元。':s='2. 试剂等耗材费（实验耗材采购，具体物品和科目按票据核实）：1730.47 元。'
        if s.startswith('项目获批总经费：'):s='项目获批总经费：按立项批复核定（现有一期三笔记录合计 2347.97 元）'
        if s=='报销成员入住支出':s='成员入住相关支出（用途及科目按票据核实）'
        if s=='实验耗材采购':s='实验耗材采购（具体物品及科目按票据核实）'
        if s.startswith('（√）按计划进行'):s='（  ）按计划进行、（  ）进度提前、（√）进度滞后（设备接入与实时性验证环节）'
        if s=='2026.11—2026.12':s='2026.11—2027.1'
        if s.startswith('接入真实数据手套 SDK'):s=plans[0].split('：',1)[1]
        if s.startswith('2026 年 10 月 12 日基线下') or s.startswith('1. 工程与可复现性'):
            s=s.replace('54 项通过，其中包含三类任务各 800 帧的 MuJoCo 仿真集成测试','54 项通过；三类任务各 800 帧的集成测试记录见 10 月 8 日基线材料')
        if s.startswith('3. 形成标准任务'):s=s.replace('含三类任务各 800 帧的仿真集成测试','三类任务各 800 帧的集成测试记录见 10 月 8 日基线材料')
        if s.startswith('工程方面，'):s=s.replace('含三类任务各 800 帧的仿真集成测试','三类任务各 800 帧的集成测试记录见 10 月 8 日基线材料')
        if s.startswith('该项目自立项以来'):
            s='指导教师意见参考稿（须由指导教师审阅确认）：项目已完成离线数据接入、姿态解算、动作映射与 MuJoCo 仿真评测的阶段性实现，并形成实验记录。真实设备连续采集与实时性目标尚未完成，部分环节存在延期。建议优先补齐设备链路、性能优化及规范重复实验。经费执行情况须结合票据和财务台账审核；是否同意通过中期检查，由指导教师审阅后填写。'
        if s=='指导教师签字：上传系统可用电子签':s='指导教师签字：'
        if s.startswith('综上，项目已完成'):s='项目已形成离线数据处理、映射与仿真评测原型；真实设备接入、实时性优化及任务级重复实验仍需完成。下一阶段将优先补齐上述环节。'
        put(p,s)
    # Remove actual duplicate ID paragraphs, retaining the first.
    ids=[p for p in ps if p.text=='S202610294063']
    for p in ids[1:]:p._element.getparent().remove(p._element)
    if f.name=='中期检查报告.docx':
        for row in doc.tables[0].rows:
            if any('2026.5' in c.text for c in row.cells):
                cell=row.cells[-1];put(cell.paragraphs[0],'部分完成（连续采集链路延期）')
        for p in [Paragraph(el,doc._body) for el in doc.element.body.iter(qn('w:p'))]:
            if p.text.startswith('特征模块'):pass
            if p.text.startswith('映射部分把申报书'):
                put(p,'映射框架支持末端位置、相对向量、拓扑相似、时序平滑、欠驱耦合和碰撞六类代价，按意图调配权重，将 15 维人手语义表示映射为 11 维模型关节目标。本次基线协同惩罚配置为空，不支持欠驱耦合改善结论。已实现方案 B/C/D/E 与双专家门控，并归档 10 组离线对照；其中 C、D、E 的任务级 MuJoCo 效果尚未验证。新增稳定抓握代理在释放辅助夹具后检查连续一秒的接触与物体相对位姿，不应与历史稳定接触指标混用。')
            if p.text.startswith('4. 任务表现与实时性'):
                put(p,'4. 任务表现与实时性。2026 年 10 月 8 日合成数据基线中，每类任务运行 3 次、每次 800 帧。图 2 成功率为满足相应代理判据的运行次数除以 3；中性任务不涉及抓取，接触与稳定指标不适用。包络任务达到历史稳定接触代理，捏合达到双指腹接触代理但未达到自由物体稳定成功。各次运行先计算单帧管线耗时 P95，再按任务取平均，结果为 116.2—123.1 ms，单次 P95 范围为 109.9—132.8 ms，均超过 16.67 ms 帧预算；该统计不等于端到端交互延迟，也不代表新增连续一秒稳定抓握代理的验证结果。')
            if p.text.startswith('标定环节按通道以最小'):
                put(p,'标定采用通道量程归一化、5 帧滑动平均与系数为 0.45 的输出低通滤波，最大输出角速度限制为 360°/s，输入频率按 60 Hz 配置。已完成合成数据与实采标定文件回放验证；连续传感器输入下的有效性仍需复核，60 Hz 配置不代表实时性能已达标。')
    target=output/f.name
    doc.save(target)
    if f.name=='中期检查报告.docx':
        import zipfile, os
        temp=target.with_suffix('.tmp.docx')
        with zipfile.ZipFile(target) as src,zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as dst:
            for member in src.infolist():
                data=src.read(member.filename)
                if member.filename=='word/media/image1.png':data=(qa/'figure1.png').read_bytes()
                if member.filename=='word/media/image2.png':data=(qa/'figure2.png').read_bytes()
                dst.writestr(member,data)
        os.replace(temp,target)
    # Structural verification after saving.
    check=Document(target)
    assert check.element.body is not None
    print(f.name,'paragraphs',len(ps))

(qa/'修改记录.json').write_text(json.dumps(log,ensure_ascii=False,indent=2),encoding='utf8')
print('BACKUP',backup)
