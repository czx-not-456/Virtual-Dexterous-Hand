from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Protocol
import numpy as np
from src.hand_model.robot_hand import RobotHandModel

class TaskSpaceSimulator(Protocol):
    def joint_positions(self) -> np.ndarray: ...
    def task_contact_target(self, finger: str, task_name: str, clearance_m: float, vertical_margin_m: float) -> np.ndarray: ...
    def fingertip_position(self, finger: str) -> np.ndarray: ...
    def fingertip_jacobian(self, finger: str, joint_names: tuple[str, ...]) -> np.ndarray: ...

@dataclass(slots=True)
class ServoDiagnostics:
    active: bool=False; thumb_error_m: float=0.0; index_error_m: float=0.0
    mean_error_m: float=0.0; max_error_m: float=0.0; corrected_digits: int=0
    def as_dict(self):
        return {"task_servo_active":self.active,"servo_thumb_error_m":self.thumb_error_m,"servo_index_error_m":self.index_error_m,"servo_mean_error_m":self.mean_error_m,"servo_max_error_m":self.max_error_m,"servo_corrected_digits":self.corrected_digits}

class ObjectAwareTaskServo:
    """Variable-DoF damped-least-squares task servo for CH-M6 digits."""
    def __init__(self, robot: RobotHandModel, cfg: dict | None) -> None:
        self.robot, self.cfg = robot, dict(cfg or {})
        self.enabled=bool(self.cfg.get("enabled",False)); self.gain=float(self.cfg.get("gain",.7))
        self.damping=float(self.cfg.get("damping",.02)); self.blend=float(self.cfg.get("blend",.8))
        self.max_joint_step=math.radians(float(self.cfg.get("max_joint_step_deg",4.0)))
        self.stop_error_m=float(self.cfg.get("stop_error_m",.0045)); self.idx={n:i for i,n in enumerate(robot.joint_order)}
    def reset(self)->None: return None
    def _task_cfg(self, task_name, task_mode):
        cfg=self.cfg.get(task_name); cfg=cfg if isinstance(cfg,dict) else self.cfg.get(task_mode)
        return dict(cfg) if isinstance(cfg,dict) else None
    def _active_fingers(self, mode):
        return ("thumb","index") if mode=="pinch" else tuple(self.robot.finger_joints) if mode=="wrap" else ()
    def _contacted(self,finger,contact,policy):
        if policy=="never": return False
        tip=bool(contact.get(f"{finger}_tip_contact",False))
        return tip if policy=="tip" else tip or bool(contact.get(f"{finger}_contact",False))
    def _dls(self,J,error):
        J=np.asarray(J,float); A=J@J.T+(self.damping**2)*np.eye(3)
        dq=J.T@np.linalg.pinv(A)@(self.gain*np.asarray(error,float))
        return np.clip(dq,-self.max_joint_step,self.max_joint_step)
    def apply(self,q_target:np.ndarray,*,simulator:TaskSpaceSimulator,task_name:str,intent_value:str,contact:dict,task_mode:str|None=None):
        q_target=np.asarray(q_target,float); diag=ServoDiagnostics(); mode=str(task_mode or task_name); cfg=self._task_cfg(task_name,mode)
        if not self.enabled or cfg is None or str(intent_value)!=str(cfg.get("active_intent","")): return q_target.copy(),diag
        clearance=float(cfg.get("target_offset_m",cfg.get("clearance_m",.007))); margin=float(cfg.get("vertical_margin_m",.01))
        policy=str(cfg.get("stop_on_contact","any")).lower(); blend=float(np.clip(cfg.get("command_blend",self.blend),0,1))
        actual=np.asarray(simulator.joint_positions(),float); cmd=q_target.copy(); errors=[]; diag.active=True
        prior=dict(cfg.get("posture_prior_deg",{})); prior_blend=float(cfg.get("posture_prior_blend",0))
        for finger in self._active_fingers(mode):
            joints=self.robot.joints_for_finger(finger); indices=np.array([self.idx[j] for j in joints],int)
            current=np.asarray(simulator.fingertip_position(finger),float); target=np.asarray(simulator.task_contact_target(finger,mode,clearance_m=clearance,vertical_margin_m=margin),float)
            error=target-current; err=float(np.linalg.norm(error)); errors.append(err)
            if finger=="thumb": diag.thumb_error_m=err
            if finger=="index": diag.index_error_m=err
            if self._contacted(finger,contact,policy): cmd[indices]=actual[indices]; continue
            nominal=q_target[indices].copy()
            if prior_blend>0:
                p=nominal.copy(); found=False
                for k,j in enumerate(joints):
                    if j in prior: p[k]=math.radians(float(prior[j])); found=True
                if found: nominal=(1-prior_blend)*nominal+prior_blend*p
            if err<=self.stop_error_m: cmd[indices]=blend*actual[indices]+(1-blend)*nominal; continue
            desired=actual[indices]+self._dls(simulator.fingertip_jacobian(finger,joints),error)
            cmd[indices]=blend*desired+(1-blend)*nominal; diag.corrected_digits+=1
        cmd=np.clip(cmd,self.robot.ranges[:,0],self.robot.ranges[:,1])
        if errors: diag.mean_error_m=float(np.mean(errors)); diag.max_error_m=float(np.max(errors))
        return cmd,diag