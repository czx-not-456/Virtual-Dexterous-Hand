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
    def __init__(self, robot: RobotHandModel, cfg: dict | None, *,
                 coupled_ablation: str = "original") -> None:
        self.robot, self.cfg = robot, dict(cfg or {})
        if coupled_ablation not in {"original", "prior", "hold", "both"}:
            raise ValueError("coupled_ablation must be original/prior/hold/both")
        self.coupled_ablation = coupled_ablation
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

    def apply_coupled(self, q_target: np.ndarray, *, simulator: TaskSpaceSimulator,
                      task_name: str, intent_value: str, contact: dict,
                      mapper, task_mode: str | None = None):
        """Object-target servo with Jacobian in the six CONTROL coordinates.

        Unlike apply(), never requests eleven unrelated joint corrections.
        All proposed joint commands are C @ u, where C is the CH-M6 software
        coupling map. This is a local DLS step, NOT a grasp feasibility proof.
        The mapper's final stage still enforces channel limits and rate bounds.
        """
        q_target = np.asarray(q_target, float)
        diag = ServoDiagnostics()
        mode = str(task_mode or task_name)
        cfg = self._task_cfg(task_name, mode)
        if not self.enabled or cfg is None or str(intent_value) != str(cfg.get("active_intent", "")):
            return q_target.copy(), diag
        actual = np.asarray(simulator.joint_positions(), float)
        u_actual = mapper.encode_joints(actual)
        u_nominal = mapper.encode_joints(q_target)
        clearance = float(cfg.get("target_offset_m", cfg.get("clearance_m", .007)))
        margin = float(cfg.get("vertical_margin_m", .01))
        policy = str(cfg.get("stop_on_contact", "any")).lower()
        blend = float(np.clip(cfg.get("command_blend", self.blend), 0, 1))
        # Ablations are opt-in and default to the exact original controller.
        # Project posture priors to the SAME 6-D coupled manifold; do not
        # introduce independent 11-DoF corrections as an unfair advantage.
        add_prior = self.coupled_ablation in {"prior", "both"}
        add_hold = self.coupled_ablation in {"hold", "both"}
        prior_nominal = u_nominal.copy()
        if add_prior:
            prior = dict(cfg.get("posture_prior_deg", {}))
            prior_blend = float(np.clip(cfg.get("posture_prior_blend", 0.), 0., 1.))
            if prior and prior_blend > 0.:
                q_prior = q_target.copy()
                for name, angle in prior.items():
                    if name in self.idx:
                        index = self.idx[name]
                        q_prior[index] = (1. - prior_blend) * q_prior[index] + (
                            prior_blend * math.radians(float(angle)))
                prior_nominal = mapper.encode_joints(q_prior)
        active_cols = []
        hold_cols = []
        tasks_J, tasks_e = [], []
        errors = []
        diag.active = True
        for finger in self._active_fingers(mode):
            joint_names = self.robot.joints_for_finger(finger)
            indices = np.array([self.idx[name] for name in joint_names], dtype=int)
            current_tip = np.asarray(simulator.fingertip_position(finger), float)
            target_tip = np.asarray(simulator.task_contact_target(
                finger, mode, clearance_m=clearance, vertical_margin_m=margin), float)
            error = target_tip - current_tip
            err = float(np.linalg.norm(error))
            errors.append(err)
            if finger == "thumb": diag.thumb_error_m = err
            if finger == "index": diag.index_error_m = err
            if self._contacted(finger, contact, policy):
                if add_hold:
                    # Copy the currently observed control pose of this digit,
                    # just as apply() does per joint, but remain strictly 6-D.
                    hold_cols.extend(np.flatnonzero(np.any(
                        np.abs(mapper.C[indices, :]) > 1e-12, axis=0)).tolist())
                continue
            if err <= self.stop_error_m:
                continue
            J6 = simulator.fingertip_jacobian(finger, joint_names) @ mapper.C[indices, :]
            tasks_J.append(J6)
            tasks_e.append(error)
            diag.corrected_digits += 1
            active_cols.extend(np.flatnonzero(np.any(np.abs(mapper.C[indices, :]) > 1e-12, axis=0)).tolist())
        u_cmd = u_nominal.copy()
        frozen = np.unique(np.asarray(hold_cols, dtype=int))
        if tasks_J:
            J = np.vstack(tasks_J)
            e = np.concatenate(tasks_e)
            active_cols = np.unique(np.asarray(active_cols, dtype=int))
            # No correction may overwrite a control held by a contacted digit.
            # The CH-M6 coupling columns do not cross fingers; this also stays
            # defined if a later hardware coupling shares columns.
            if frozen.size:
                active_cols = np.setdiff1d(active_cols, frozen)
            if active_cols.size:
                J_active = J[:, active_cols]
                damping_sq = self.damping ** 2
                d_control = J_active.T @ np.linalg.solve(
                    J_active @ J_active.T + damping_sq * np.eye(len(e)), self.gain * e)
                max_norm_step = self.max_joint_step / np.deg2rad(mapper.limits_deg[active_cols])
                step = np.clip(d_control, -max_norm_step, max_norm_step)
                desired = u_actual.copy()
                desired[active_cols] += step
                u_cmd[active_cols] = blend * desired[active_cols] + (1 - blend) * prior_nominal[active_cols]
        if frozen.size:
            u_cmd[frozen] = u_actual[frozen]
        u_cmd = np.clip(u_cmd, mapper.lower, mapper.upper)
        if errors:
            diag.mean_error_m = float(np.mean(errors))
            diag.max_error_m = float(np.max(errors))
        return mapper.decode_controls(u_cmd), diag
