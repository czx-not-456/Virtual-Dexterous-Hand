"""Capture live MuJoCo physics from the existing src.main control loop.
No replacement joint trajectory and no changes to original source files.
The capture adds overhead; its timing cannot benchmark the controller.
"""
from pathlib import Path
import argparse, csv, hashlib, json, os, shutil, subprocess, sys, time
from datetime import datetime, timezone
from project_paths import ROOT, HERE
from PIL import Image, ImageDraw

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--task",choices=["pinch","wrap"],required=True)
    p.add_argument("--trial",type=int,default=0)
    args=p.parse_args()
    os.chdir(ROOT);sys.path.insert(0,str(ROOT))
    import mujoco
    from src.simulation import mujoco_env
    from src import main as controller
    try:
        ffmpeg=shutil.which("ffmpeg")
        if not ffmpeg:
            import imageio_ffmpeg
            ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as e:
        raise RuntimeError("Install ffmpeg or pip install imageio-ffmpeg") from e
    # Avoid overwriting previous attempts, including interrupted runs.
    stamp=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    dest=HERE/"真实仿真录制"/f"{args.task}_trial{args.trial}_{stamp}"
    dest.mkdir(parents=True)
    camera="closeup" if args.task=="pinch" else "overview"
    width,height=640,480
    records=[];shots=[]
    original=mujoco_env.MujocoSimulator
    started=time.perf_counter()
    class Recorder(original):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw)
            self.model.vis.global_.offwidth=max(width,self.model.vis.global_.offwidth)
            self.model.vis.global_.offheight=max(height,self.model.vis.global_.offheight)
            self.renderer=mujoco.Renderer(self.model,height=height,width=width)
            self.frame=-1;self.best_contact=-1;self.best_record=None
            self.proc=subprocess.Popen([ffmpeg,"-hide_banner","-loglevel","error","-y","-f","rawvideo","-pix_fmt","rgb24",
                "-s",f"{width}x{height}","-r","10","-i","-","-an","-c:v","libx264","-pix_fmt","yuv420p",
                str(dest/"任务演示_慢放.mp4")],stdin=subprocess.PIPE)
            self.saved=set()
        def image(self,preset):
            cam=mujoco.MjvCamera();cam.type=mujoco.mjtCamera.mjCAMERA_FREE
            cfg=self.camera_presets[preset];cam.lookat[:]=cfg["lookat"];cam.distance=cfg["distance"]
            cam.azimuth=cfg["azimuth"];cam.elevation=cfg["elevation"]
            self.renderer.update_scene(self.data,camera=cam)
            return Image.fromarray(self.renderer.render().copy())
        def step(self,*a,**kw):
            q=super().step(*a,**kw);self.frame+=1
            return q
        def contact_metrics(self):
            metrics=super().contact_metrics()
            f=self.frame;active=bool(self._task_active)
            rec={"frame":f,"simulation_time_s":float(self.data.time),"active":active,
                 "task_contact_proxy":bool(metrics["task_contact_proxy"]),
                 "stable_contact_proxy":bool(metrics["stable_contact_proxy"]),
                 "contact_count":int(metrics["contact_count"]),
                 "fixture_held":bool(self._object_held_by_fixture)}
            records.append(rec)
            def shot(name,preset):
                path=dest/(name+".png");self.image(preset).save(path)
                shots.append(dict(rec,filename=path.name,camera=preset))
                self.saved.add(name)
            if f==0:
                shot("01_初始全景","overview");shot("02_初始近景","closeup")
            if active and "03_任务开始" not in self.saved:shot("03_任务开始",camera)
            if active and rec["task_contact_proxy"] and "04_首次任务接触代理" not in self.saved:
                shot("04_首次任务接触代理",camera)
            if active and rec["contact_count"]>self.best_contact:
                self.best_contact=rec["contact_count"]
                self.image(camera).save(dest/"05_有效区间接触数最大帧.png")
                self.best_record=dict(rec,filename="05_有效区间接触数最大帧.png",camera=camera)
            if f%2==0:
                im=self.image(camera);draw=ImageDraw.Draw(im)
                draw.rectangle((0,0,width,44),fill=(18,32,48))
                draw.text((8,5),f"{args.task.upper()}  trial={args.trial}  frame={f}  sim={self.data.time:.2f}s",fill="white")
                draw.text((8,23),f"Physical MuJoCo | 3x slow motion | contact proxy={int(rec['task_contact_proxy'])} | fixture={int(rec['fixture_held'])}",fill="white")
                self.proc.stdin.write(im.tobytes())
            return metrics
        def close(self):
            if self.best_record:shots.append(self.best_record)
            self.proc.stdin.close();rc=self.proc.wait()
            self.renderer.close();super().close()
            if rc:raise RuntimeError(f"ffmpeg failed: {rc}")
    mujoco_env.MujocoSimulator=Recorder
    sys.argv=["src.main","--sim","mujoco","--input","dataset","--dataset-trial",str(args.trial),
              "--task",args.task,"--steps","800","--camera",camera,"--output-dir",str(dest),"--run-name","当前物理运行"]
    controller.main()
    def csvfile(name,rows):
        with (dest/name).open("w",encoding="utf-8-sig",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    csvfile("实时帧记录.csv",records);csvfile("截图索引.csv",shots)
    meta={"task":args.task,"trial":args.trial,"input":"synthetic dataset","mujoco":mujoco.__version__,
          "execution_utc":datetime.now(timezone.utc).isoformat(),"recorded_control_frames":len(records),
          "video_fps":10,"sample_every_control_frames":2,"playback_duration_s":len(records[::2])/10,
          "nominal_control_hz":60,"playback":"1/3 nominal simulation speed, offline rendering; not realtime",
          "wall_elapsed_s":time.perf_counter()-started,"source_main_sha256":hashlib.sha256((ROOT/"src/main.py").read_bytes()).hexdigest(),
          "source_simulator_sha256":hashlib.sha256((ROOT/"src/simulation/mujoco_env.py").read_bytes()).hexdigest(),
          "note":"Physical runtime framebuffer output, not Viewer window chrome; capture overhead invalidates latency benchmarking. Current v2 stability differs from 2026-10-08 legacy."}
    (dest/"录制说明.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf8")
    print(dest)
    if len(records)!=800:raise RuntimeError("Incomplete simulation; see logs, do not archive as complete.")

if __name__=="__main__":main()
