from pathlib import Path
import json,csv,os,re,html,urllib.parse,hashlib
O=Path(__file__).resolve().parent;P=json.loads((O/'整理操作计划.json').read_text(encoding='utf8'));B=Path(P['base']);D=P['dirs'];edits={}
def write(n,t):
 p=B/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(t.strip()+'\n',encoding='utf8')
def link(label,n,ctx=''):
 return '['+label+']('+urllib.parse.quote(os.path.relpath(B/n,(B/ctx).parent).replace('\\','/'),safe='/._-')+')'
def edit(n,old,new):
 p=B/n;t=p.read_text(encoding='utf-8-sig').replace('\r\n','\n');start=t.index(old);edits.setdefault(n,[]).append({'start':start,'before':old,'after':new});p.write_text(t[:start]+new+t[start+len(old):],encoding='utf8')
# Fix repeated-event filename without removing either event from the original five-row index.
n=D['05']+'/PINCH/截图索引.csv';edit(n,'05_有效区间接触数最大帧.png','04_首次任务接触代理.png')
# Preserve the original reports' claims, updating stale link targets only.
for p in list(B.rglob('*.md')):
 n=p.relative_to(B).as_posix();t=p.read_text(encoding='utf-8-sig').replace('\r\n','\n')
 for m in reversed(list(re.finditer(r'(?<!!)\[([^\]\n]+)\]\(([^)\n]+)\)',t))):
  label,target=m.groups();decoded=urllib.parse.unquote(target)
  if re.match(r'^(https?:|mailto:|#)',decoded):continue
  dest=(p.parent/decoded.split('#')[0]).resolve();new=None
  if dest.is_relative_to(B) and dest.exists():continue
  # Resolve link using the original pre-move document's location.
  original=next((old for old,final in P['final'].items() if final==n),n)
  olddest=(B/original).parent/decoded.split('#')[0]
  if olddest.resolve().is_relative_to(B):
   oldname=olddest.resolve().relative_to(B).as_posix();final=P['final'].get(oldname)
   if final and (B/final).exists():new=link(label,final,n)
  if new is None:new=label+'（历史工程路径：`'+decoded+'`）'
  old=m.group(0);edit(n,old,new)
# Main entry and numbered folder instructions.
rows=[('01','综合测试与环境记录','历史52项pytest及2026-11-15补充环境/20项检查；当前Windows未做全量复测'),('02','MuJoCo三任务批量实验','历史9次×800帧；另存新版PINCH/WRAP各800帧日志'),('03','Stage B七方法消融','24条六姿态诊断记录，软件教师标签归一化MAE'),('04','其他方案离线结果','C/D/E离线探索JSON；不属于物理抓取成功率'),('05','MuJoCo任务截图组','带帧号和CSV索引的2026-11-15物理渲染截图'),('06','任务演示视频','PINCH/WRAP各40秒；三倍慢放，合成输入'),('07','系统流程图','系统与独立离线评估流程；另有历史相机操作参考'),('08','任务时序曲线','可追溯2026-10-08 trial00历史曲线，与新版视频分开解释'),('09','Benchmark汇总展示页','一页HTML及历史统计图、汇总CSV'),('10','Stage B消融展示页','一页HTML及七方法对照图'),('11','当前问题与下一阶段计划','修订后的PNG/PDF及可编辑计划说明'),('12','附件目录与复现说明','完整索引、核验入口、统一来源清单和复现参考脚本')]
main='''# 项目中期成果说明

初次整理：2026-11-14；成果归类与说明更新：2026-11-15。

本包展示虚拟灵巧手的输入处理、映射、MuJoCo任务和离线评估成果。01—04最初汇集历史报告与数据；随后补入2026-11-15物理运行、姿态识别及滤波分析。本次仅整理现有材料、修正说明和制作展示页，未重跑MuJoCo、采集硬件数据或产生新的实验成绩。

## 成果入口

'''
main+=link('中期检查成果图册','中期检查_成果图册.docx','项目中期成果说明.md')+'是主要展示入口；'+link('完整附件目录与阅读顺序',D['12']+'/附件目录.md','项目中期成果说明.md')+'定位全部材料。\n\n| 编号 | 任务与材料 | 成果范围 |\n|---|---|---|\n'
for no,title,desc in rows:main+=f'| {no} | {link(title,D[no]+"/","项目中期成果说明.md")} | {desc} |\n'
main+='''
## 实验口径

历史基线为2026-10-08 Windows、Python 3.11.9、MuJoCo 3.13.0记录。三任务各3次、每次800帧，共7200帧；PINCH接触3/3、历史稳定0/3；WRAP接触3/3、历史稳定3/3；NEUTRAL仅要求流程完成，其接触/稳定指标不适用。固定基座接触代理不等同于抓起、搬运或放置成功。历史P95延迟约110—133毫秒，不支持严格60Hz实时执行。

补充物理运行记录为2026-11-15 Linux、Python 3.12.14、MuJoCo 3.3.7，以合成输入trial 0执行PINCH和WRAP各800帧。PINCH接触帧占比99.16%、新版稳定帧占比58.65%、新版稳定通过；WRAP对应93.31%、0%、未通过。帧占比以任务有效帧为分母，不能理解为独立试验成功概率。新版相对手掌滚动窗口与历史稳定定义不同，不直接比较优劣。截图和录制时延含渲染开销，不能用于控制器延迟排名。

Stage B基于24条、6类姿态的诊断记录和软件教师标签；20组缺失掩码不构成480条独立动作。增强Ridge6表现较强，不宣称融合/降维方法全面领先；理想选择器不可部署。识别分析为27组、12个设备标识、5100帧静态标定数据：设备等权准确率99.58%、逐帧汇总98.53%，不代表新被试连续动态手势准确率。15个CSV列的硬件含义仍需核实。滤波实验为8条合成序列的独立离线分析，不是硬件真值测试。

## 文件与核验

资料以01—12为主；图册配套依据保留三维模型/重建、姿态识别、缺失鲁棒性与滤波等独有证据，不保存编号目录的重复副本。历史日志、JSON和代码快照中的原路径作为溯源信息保留，不是当前文件导航。来源、去重映射和内容变更记录已合并到12中的统一来源清单；历史来源哈希仍可追溯。

材料归档任务01—12现已齐备；当前Windows全量测试、真实手套在线通信、连续动态泛化、自由物体持续稳定持物及Stage C完整验证仍未完成，见11计划。材料齐备不等同于全部研究目标完成。
'''
write('项目中期成果说明.md',main)
# Folder-specific source explanations.
for no in ['01','02','03','04']:
 title=next(r[1] for r in rows if r[0]==no);desc=next(r[2] for r in rows if r[0]==no)
 write(D[no]+'/阅读说明.md',f'# {title}\n\n{desc}。\n\n本目录保存历史报告、原始记录或离线结果；日期以文件内部记录为准。历史文档中的旧命令、未随包收录附件及原工程路径仅用于追溯，不能视为当前可直接执行的复现指引。\n\n'+link('当前附件索引与复现说明',D['12']+'/复现说明.md',D[no]+'/阅读说明.md')+'。\n\n'+('补充运行_20261115保存新版物理日志；原目录中的bench_*及benchmark_*属于历史基线，两批稳定指标定义不同。' if no=='02' else '未重跑实验或改写原始日志和数值数据。'))
shot='''# 任务截图说明

本组全部采用2026-11-15物理运行的原始渲染帧，与06视频及02补充运行使用同一批800帧记录。输入为合成数据trial 0，固定基座，MuJoCo 3.3.7。原先无可靠运行索引的五张展示导出图已由本组替代。

| 任务 | 图像 | 帧号 | 含义 |
|---|---|---:|---|
| PINCH | 01_初始全景.png、02_初始近景.png | 0 | 第一次物理步进后的状态 |
| PINCH | 03_任务开始.png | 171 | 活动区间开始，尚未达到任务接触代理 |
| PINCH | 04_首次任务接触代理.png | 173 | 首次接触，仍含夹具辅助；也是接触数最大帧 |
| WRAP | 01_初始全景.png、02_初始近景.png | 0 | 第一次物理步进后的状态 |
| WRAP | 03_任务开始.png | 489 | 活动区间开始 |
| WRAP | 04_首次任务接触代理.png | 503 | 首次任务接触代理 |
| WRAP | 05_有效区间接触数最大帧.png | 523 | 有效区间接触数最大 |

PINCH两个事件发生在同一173帧，只保留一张图片；两条事件记录在截图索引中均保留并指向同一文件。接触代理不等同于提举成功，也不能由一张静态图推断持续稳定。

'''
for task in ['PINCH','WRAP']:
 ctx=D['05']+'/截图说明.md';shot+=link(task+'截图索引',D['05']+'/'+task+'/截图索引.csv',ctx)+'；'+link(task+'物理逐帧日志',D['02']+'/补充运行_20261115/'+task+'/当前物理运行.csv',ctx)+'。\n\n'
write(D['05']+'/截图说明.md',shot)
ctx=D['06']+'/视频说明.md';t='# 任务视频说明\n\n两段视频来自2026-11-15物理运行，合成输入trial 0，各800控制帧。每隔2控制帧取1帧，以10fps编码，各400视频帧、40秒，为相对名义60Hz控制的三倍慢放；离线渲染，不表示实际60Hz实时运行或硬件在线采集。\n\n'
for task in ['PINCH','WRAP']:t+=link(task+'演示',D['06']+'/'+task+'/任务演示_慢放.mp4',ctx)+'；'+link('录制环境及来源哈希',D['06']+'/'+task+'/录制说明.json',ctx)+'；'+link('对应逐帧日志',D['02']+'/补充运行_20261115/'+task+'/当前物理运行.csv',ctx)+'。\n\n'
t+='PINCH首次接触含夹具辅助，新版稳定通过；WRAP新版稳定未通过。固定基座，不证明提举或搬运成功。';write(ctx,t)
write(D['07']+'/流程图说明.md','# 系统流程图\n\n'+link('系统与数据评估流程图',D['07']+'/07_系统流程.png',D['07']+'/流程图说明.md')+'对应图册图1。上方为数据输入、预处理、意图与映射、CH-M6/MuJoCo及量化评估主链路；下方为独立离线验证。历史系统代码快照保存在01补充运行目录，不能把独立离线识别实验理解为当前主控制环的部署成绩。\n\n手套实时通信、硬件通道定义和连续动作验证仍待完成。CAMERA_CONTROLS.md为原项目相机操作参考。')
ctx=D['08']+'/曲线说明.md';t='''# 历史任务时序曲线

两张图均为2026-10-08合成输入trial 00历史记录的重分析，已在图中标注日期与口径，替代此前无法明确溯源的两张曲线。它们不是2026-11-15视频运行的曲线。

PINCH任务有效帧171—407；WRAP为489—727。横轴为控制帧号。指尖距离来自人体手模型，非机器手实测距离；接触/稳定为历史代理，绿色区域为有效区间，伺服误差仅在启用时绘制。新版稳定使用另一套滚动窗口定义，不能混用。

'''
for task in ['pinch','wrap']:t+=link(task.upper()+'曲线',D['08']+'/08_'+task+'_任务时序.png',ctx)+'；'+link('唯一原始逐帧CSV',D['02']+'/bench_'+task+'_trial00.csv',ctx)+'。\n\n'
t+=link('原重分析代码（源码佐证）',D['12']+'/复现参考脚本/复现实验.py',ctx)+'；'+link('复算入口',D['12']+'/核验成果.py',ctx)+'。';write(ctx,t)
# Standalone summary pages, built only from saved results.
style='''body{font:16px/1.55 "Microsoft YaHei",sans-serif;color:#17343f;margin:32px auto;max-width:1100px;padding:0 24px}h1{font-size:28px;margin:0 0 8px}h2{font-size:19px}p{margin:8px 0}table{border-collapse:collapse;width:100%;font-size:14px;margin:14px 0}th,td{padding:8px;border-bottom:1px solid #d9e3e6;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#e9f2f4}img{display:block;max-width:100%;max-height:400px;margin:12px auto}a{color:#176f91}.note{background:#f0f5f6;padding:12px;border-radius:5px}@page{size:A4 landscape;margin:12mm}@media print{body{margin:0;max-width:none;padding:0;font-size:12px}h1{font-size:23px}table{font-size:11px}img{max-height:260px}a{color:inherit;text-decoration:none}}'''
def href(ctx,target,label):return '<a href="'+html.escape(urllib.parse.quote(os.path.relpath(B/target,(B/ctx).parent).replace('\\','/'),safe='/._-'))+'">'+html.escape(label)+'</a>'
def page(ctx,title,body):write(ctx,'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>'+title+'</title><style>'+style+'</style><h1>'+title+'</h1>'+body+'</html>')
ctx=D['09']+'/09_Benchmark汇总展示页.html';data=list(csv.DictReader((B/D['02']/'benchmark_summary.csv').open(encoding='utf-8-sig')))
body='<p>2026-10-08历史基线｜合成输入｜固定基座｜三任务各3次、每次800帧，共7200帧</p><table><tr><th>任务</th><th>任务规定指标通过</th><th>接触通过</th><th>历史稳定通过</th><th>接触帧占比</th><th>历史稳定帧占比</th><th>平均试次P95</th></tr>'
for r in data:
 applicable=r['task']!='neutral';body+='<tr><td>'+r['task'].upper()+'</td><td>3/3</td><td>'+('3/3' if applicable else '不适用')+'</td><td>'+('0/3' if r['task']=='pinch' else '3/3' if applicable else '不适用')+'</td><td>'+(f"{float(r['mean_contact_ratio']):.2%}" if applicable else '不适用')+'</td><td>'+(f"{float(r['mean_stable_ratio']):.2%}" if applicable else '不适用')+'</td><td>'+f"{float(r['mean_p95_latency_ms']):.2f} ms"+'</td></tr>'
body+='</table><img src="09_九次试验汇总.png" alt="历史9次任务的接触稳定帧占比与试次P95"><p class="note">NEUTRAL仅要求流程完成；PINCH要求接触；WRAP要求历史稳定代理。任务规定指标100%不等于三项抓取成功率100%。接触/稳定成功按连续活动帧判定，帧占比按有效帧计算。固定手掌、夹具和代理指标不能证明提举/搬运；旧稳定定义不与新版StableGrasp-v2直接比较。P95超过60Hz单帧16.67ms预算。</p><p>来源：'+href(ctx,D['02']+'/benchmark_runs.csv','9次结果明细')+' · '+href(ctx,D['02']+'/benchmark_summary.csv','原汇总')+' · '+href(ctx,D['02']+'/BENCHMARK_REPORT.md','历史报告')+'。统计图及09_三任务统计.csv为同一历史数据的重分析。</p>'
page(ctx,'三任务历史Benchmark汇总',body)
ctx=D['10']+'/10_StageB消融展示页.html';data=list(csv.DictReader((B/D['03']/'summary_by_condition.csv').open(encoding='utf-8-sig')));cols=['ridge6_original','pca4_original','ridge6_robust','pca4_robust','fixed_50_50','calibrated_scalar_blend','tqhg_v2_original'];labels=['原Ridge6','原PCA4','增强Ridge6','增强PCA4','固定混合','校准融合','原TQHG-v2']
body='<p>软件教师六维标签归一化MAE｜24条诊断记录、6类姿态｜各故障条件20组共享缺失掩码</p><table><tr><th>条件</th>'+''.join('<th>'+c+'</th>' for c in labels)+'</tr>'
for r in data:body+='<tr><td>'+r['condition']+'</td>'+''.join(f'<td>{float(r[c]):.5f}</td>' for c in cols)+'</tr>'
body+='</table><img src="10_StageB七方法对比.png" alt="七种方法在各缺失条件下的软件教师标签误差"><p class="note">增强Ridge6在五列屏蔽下MAE为0.12188，原Ridge6为0.12652；增强与原模型训练条件不同，差异不能直接归因于维数。融合/降维方法没有全面领先。20组掩码不是480条独立动作；这不是新的盲测泛化或物理抓握验证。报告中的理想选择器不可部署。</p><p>来源：'+href(ctx,D['03']+'/REPORT.md','完整实验设置与限制')+' · '+href(ctx,D['03']+'/summary_by_condition.csv','汇总表')+' · '+href(ctx,D['03']+'/paired_cases.csv','1464条配对明细')+'。</p>'
page(ctx,'Stage B七方法离线消融',body)
# Revision basis is explicit and contains no invented deadlines/owners.
planrows=[['手套实时输入','标定样本可离线回放；实时通信仍待接入。','对接SDK或通信协议，记录连续动作、时间戳与掉帧。','连续采集日志；核对帧序、采样间隔及掉帧。'],['模型与稳定持物','关节耦合需核对；新版WRAP单次运行稳定未通过。','核对关节范围/耦合，分别复测夹具辅助及释放后保持。','参数对照；报告释放后稳定帧占比及独立试验通过数。'],['映射与处理延迟','历史合成输入P95约110—133ms，超过60Hz预算。','在相同输入与环境下分段计时，再优化求解。','对照总管线P95、误差和接触保持；60Hz预算16.67ms。'],['方法对比与泛化','Stage B为24条诊断记录；Stage C缺完整保存结果。','补独立连续轨迹与物体初态，再验证真实任务效果。','按独立试验汇总；区分离线标签误差与物理任务指标。']]
ctx=D['11']+'/计划说明.md';t='# 当前问题与下一阶段计划\n\n依据已有历史记录和2026-11-15补充运行修订；不把Stage C设计描述为已完成实测。负责人和时间节点尚未指定，不虚构安排。\n\n| 工作方向 | 当前问题 | 下一阶段工作 | 验收依据 |\n|---|---|---|---|\n'
for row in planrows:t+='| '+' | '.join(row)+' |\n'
t+='\n依据：'+link('历史延迟及任务结果',D['02']+'/benchmark_runs.csv',ctx)+'；'+link('新版WRAP运行',D['02']+'/补充运行_20261115/WRAP/当前物理运行_summary.json',ctx)+'；'+link('Stage B诊断限制',D['03']+'/REPORT.md',ctx)+'；'+link('历史系统审查',D['01']+'/REVIEW_NOTES.md',ctx)+'。\n\n完整Stage C结果未随包保存，应先补齐结果再评价优劣；本次未运行Stage C。PDF和PNG为同一计划的分享/展示格式，保留两者。'
write(ctx,t)
(O/'计划页内容.json').write_text(json.dumps(planrows,ensure_ascii=False),encoding='utf8')
# Reproduction guide: actual local entry points, no automatic experiment reruns.
ctx=D['12']+'/复现说明.md';t='''# 附件核验与复现说明

本包可独立核验已保存的数值；完整物理重跑需要原项目源码、配置、模型和环境。本次整理只做离线核验，没有重跑MuJoCo、全量pytest或硬件实验。

## 一 独立核验保存的数据

在本文件所在12目录打开PowerShell，使用有NumPy、Pandas及Pillow的Python环境：

```powershell
python .\核验成果.py
```

若依赖缺失，在所选环境安装：`python -m pip install numpy pandas Pillow`。视频解码完整性检查另需OpenCV：`python -m pip install opencv-python`。这些是依赖说明，本次整理不会自动安装。

脚本从自身位置定位成果根目录。它核验统一清单所列文件、144项原证据及45项历史V2来源，随后复算两次物理运行、9次历史Benchmark、Stage B均值、设备留出识别和滤波指标。删除重复副本并未删除任一原始证据项的校验覆盖；换行/BOM差异通过规范化比较，索引/文档修订通过可逆编辑记录核对整理前内容。清单本身不对自己计算递归哈希。

## 二 原项目中的物理重跑

先切换至包含pyproject.toml、src和datasets的完整项目根目录；确认Python环境及MuJoCo、模型与输入依赖。以下命令仅说明当前源码入口，未在本次执行：

```powershell
.\.venv\Scripts\python.exe -m src.main --help
.\.venv\Scripts\python.exe -m src.main --sim mujoco --input sim --sim-source dataset --steps 800 --task pinch --trial 0 --output-dir "$env:TEMP\dexterous_reproduce" --run-name pinch_check
```

请先以`--help`核对本地版本参数，再选择任务。新运行会写入指定临时输出目录，不覆盖历史归档。当前Windows源码/依赖与2026-10-08或2026-11-15环境不同，结果不得直接冒充历史复现一致。

复现参考脚本目录保留原离线生成、校验和录制脚本。它们是历史源码佐证，会依赖原工程路径并生成输出，不是本包独立核验入口；合并了重复的project_paths.py，保留了脚本内容。原脚本内路径、哈希及历史输出名称按原文留存。历史项目导航文档中的命令仅供追溯。

## 三 指标与缺失材料

历史稳定、StableGrasp-v2、接触成功与有效帧占比分开解释，见顶层说明。历史52项pytest与补充运行20项检查来自不同环境和范围，不合并为当前72项测试。历史报告提及但未随包收录的手套标定/任务附件不能被当作本包已验证材料。

原环境、源码快照、录制元数据和历史日志分别保存在01、02及06。完整当前环境应在另一次实际复现时记录；不能将旧environment.json当作今天的Windows环境。
'''
# Only include options verified in source later; --trial is reviewed before delivery.
write(ctx,t)
# Complete folder index, generated once rather than copied into each folder.
ctx=D['12']+'/附件目录.md';t='# 中期成果附件目录\n\n主展示文件：'+link('中期检查_成果图册','中期检查_成果图册.docx',ctx)+'；'+link('成果范围与口径','项目中期成果说明.md',ctx)+'。\n\n| 编号 | 任务 | 当前材料状态 |\n|---|---|---|\n'
for no,title,desc in rows:t+=f'| {no} | {link(title,D[no]+"/",ctx)} | 已归档；{desc} |\n'
t+='\n## 图册章节与唯一证据\n\n| 图册章节 | 主要证据 |\n|---|---|\n'
chapter=[('一 系统',D['07']+'/07_系统流程.png'),('二 物理截图',D['05']+'/截图说明.md'),('二 对应新版运行',D['02']+'/补充运行_20261115/'),('三 三维姿态','图册配套依据/03_三维姿态/'),('四、五 历史时序',D['08']+'/曲线说明.md'),('六 历史任务统计',D['09']+'/09_Benchmark汇总展示页.html'),('六 Stage B',D['10']+'/10_StageB消融展示页.html'),('七、八 识别与鲁棒性','图册配套依据/07-08_识别与鲁棒性/'),('九 滤波抗噪','图册配套依据/09_滤波抗噪/')]
for title,n in chapter:t+='| '+title+' | '+link(n,n,ctx)+' |\n'
t+='\n'+link('复现说明',D['12']+'/复现说明.md',ctx)+'；'+link('统一来源与校验清单',D['12']+'/来源与校验清单.json',ctx)+'；'+link('离线核验脚本',D['12']+'/核验成果.py',ctx)+'。\n\n历史V2来源CSV为2026-11-15运行时保存的历史审计证据，不是另一套当前文件清单。当前有效路径、大小和哈希统一以本目录JSON清单为准。\n\n## 文件逐项索引\n\n'
for p in sorted(B.rglob('*')):
 if p.is_file() and p!=B/ctx:t+='- '+link(p.relative_to(B).as_posix(),p.relative_to(B).as_posix(),ctx)+'\n'
write(ctx,t)
ctx='图册配套依据/配套依据说明.md';write(ctx,'# 图册配套依据\n\n本目录保留三维模型及运动学重建、静态标定识别/缺失实验、合成序列滤波的独有材料。它们与01—12不是相同编号任务，具体对应图册章节见'+link('统一附件目录',D['12']+'/附件目录.md',ctx)+'。\n\n系统代码和测试记录在01，物理日志在02，截图在05，视频在06，流程图在07，历史时序在08，统计与消融展示在09/10；这里通过目录引用，避免重复副本。核验入口已移至12。')
(O/'可逆文本修订.json').write_text(json.dumps(edits,ensure_ascii=False,indent=2),encoding='utf8')
print('DOCUMENTATION_AND_HTML_READY','reversible_edits',sum(map(len,edits.values())))
