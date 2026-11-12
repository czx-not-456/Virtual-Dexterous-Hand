from pathlib import Path
import re,json,shutil,zipfile
from docx import Document
from docx.text.paragraph import Paragraph
from docx.oxml.ns import qn
root=Path(__file__).resolve().parent.parent
folder=root/'中期检查'
def put(p,s):
    if not p.runs:p.add_run(s)
    else:
        p.runs[0].text=s
        for r in p.runs[1:]:r.text=''
for f in (folder/'修订稿').glob('*.docx'):
    doc=Document(f)
    for p in [Paragraph(el,doc._body) for el in doc.element.body.iter(qn('w:p'))]:
        s=p.text.replace('各任务各任务的','各任务的').replace('。 六维','。六维').replace('。 该均值','。该均值').replace('索取 交付','索取交付').replace('降低31.08%','降低 31.08%').replace('的MuJoCo','的 MuJoCo')
        s=s.replace('六项代价重定向','支持六类代价的重定向框架')
        if s.startswith('问题：各任务的单帧'):s=s.replace('各任务的单帧','2026 年 10 月 8 日合成数据基线中，各任务的单帧')
        if s.startswith('解决方案：对数据处理'):
            s='解决方案：分别记录数据处理、映射优化与仿真耗时；评估降低迭代次数、采用上一帧解热启动、缓存可复用中间量及适度降维的效果，明确精度损失；对雅可比更新频率开展耗时与精度对照。先以单帧管线耗时 P95 不超过 33 ms 为目标，再向 16.67 ms 优化，同时测量端到端延迟。'
        if s.startswith('映射部分将多目标'):
            s=s.replace('将多目标优化思路落地为','支持').replace('六项代价','六类代价')
            s+=' 当前基线协同惩罚配置为空，不能推断欠驱耦合改善；新增连续一秒评价方法的实现不代表其已通过任务级验证。'
        if s.startswith('工程方面，'):
            s=s.replace('190 帧标定记录','一组 190 帧实采标定记录（来自 27 组快照之一；与 24 帧样本的重叠关系尚未核实）')
            s+=' 上述耗时先按每次运行计算 P95，再对每任务 3 次运行取平均；任务结果采用 10 月 8 日历史稳定接触判据，不代表新增连续一秒稳定抓握代理已通过验证。'
        if s.startswith('4. 建立多方案'):
            s+=' 其中方案 C、D、E 的任务级效果仍待验证。'
        if s.startswith('5. 取得阶段性量化结果'):
            s=s.replace('意图驱动动态映射','2026 年 10 月 8 日基线中，意图驱动动态映射')
            s+=' 上述接触结果为合成数据历史代理指标，不等同于新增连续一秒稳定抓握验证或真实抓取效果。'
        if s.startswith('1. 工程与可复现性'):
            s='1. 工程与可复现性。历史记录显示，2026 年 10 月 12 日基线全量自动化测试 54 项通过；10 月 8 日基线包含三类任务各 800 帧的集成测试。该记录不代表当前 v1.3.0 全部新增模块已重新通过测试。'
        if s.startswith('3. 实采标定数据'):
            s='3. 实采标定数据。选取 27 组快照中的一组 190 帧记录，基础映射相对六维目标的角度均方根误差为 6.9585°；按既有耦合关系展开至 11 关节后，平均指尖位置差为 5.8224 mm。24 帧六姿态样本用于另一组离线对照，二者是否重叠仍需原始数据核实。结果不代表连续动作泛化性能。'
        if s.startswith('按申报书提出的三层结构') and f.name=='中期检查报告.docx':
            s='姿态模块由归一化信号与关节运动范围解算 15 维关节角；以 12 个非拇指关节归一化弯曲量均值构造包络量，以拇指—食指指尖距离表征捏合。结合前向运动学结果，每帧输出关节角、归一化关节角、五指三维指尖位置、包络量与捏合距离五类特征。'
        if s.startswith('意图识别采用带迟滞') and f.name=='中期检查报告.docx':
            s='规则识别器输出中性、捏合和包络三类意图。捏合进入／退出阈值为 0.060／0.072 m，包络为 0.68／0.58；采用 7 帧滑窗及连续 4 帧驻留判决。张开阶段归为中性。尚缺少带标注的真实连续动作数据，本阶段不报告识别准确率，仅验证链路与时序稳定性。'
        if s.startswith('评测端输出逐帧 CSV'):
            s='评测输出逐帧数据、运行汇总与可读报告，记录接触及稳定代理、连续成功帧数、首次接触延迟、单帧管线耗时 P95、关节误差、姿态漂移和物体位移。接触及稳定代理均不等同于抓起、搬运或放置成功。'
        if s.startswith('标准任务仿真评测方案与实验数据集（'):
            s='标准任务仿真评测方案与仿真数据集（历史基线 54 项测试通过；10 组离线对照脚本与结果归档）'
        if s.startswith('实现方案 B/C/D/E 多路线映射与 Stage C'):
            s=s.replace('Stage C 任务雅可比门控接口','任务雅可比门控接口')
        if s!=p.text:put(p,s)
    doc.save(f)
    paras=[Paragraph(el,doc._body).text for el in doc.element.body.iter(qn('w:p'))]
    if f.name=='中期检查报告.docx':
        a=next(i for i,s in enumerate(paras) if s.startswith('本项目严格'))
        b=next(i for i,s in enumerate(paras) if s.startswith('三、经费'))
        n=len(''.join(paras[a:b]));print('报告第二部分字符数（含空格）',n);assert n<3000
        assert sum(s=='S202610294063' for s in paras)==1
    assert zipfile.is_zipfile(f)
    assert not any('各任务各任务' in s for s in paras)
results=[]
for f in (folder/'修订稿').glob('*.docx'):
    try:
        shutil.copy2(f,folder/f.name)
        results.append({'file':f.name,'original_updated':True})
    except PermissionError:
        results.append({'file':f.name,'original_updated':False,'reason':'文件被占用'})
print(json.dumps(results,ensure_ascii=False,indent=2))
(root/'.文档修改校验'/'交付状态.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf8')
