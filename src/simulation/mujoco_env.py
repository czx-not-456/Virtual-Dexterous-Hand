from __future__ import annotations

from pathlib import Path
import math
import xml.etree.ElementTree as ET
import numpy as np

from src.common import ROOT
from src.hand_model.robot_hand import RobotHandModel
from src.simulation.ch_m6_adapter import build_ch_m6_scene
from src.evaluation.relative_grasp_stability import RelativeGraspHold
from src.simulation.contact_detection import ContactDetectionConfig, ForceContactDebouncer


class MujocoSimulator:
    """CH-M6 MuJoCo driver with runtime scene/contact adaptation.

    核心变化：
    - CH-M6 原生 11 DoF / 11 position actuator 控制接口；
    - 15 维标准人手姿态由配置映射为 11 维 CH-M6 目标；
    - 支持 neutral / PINCH / WRAP 三种标准场景；
    - 记录 thumb/fingers/palm 接触区、指尖接触率、物体姿态漂移等指标。
    """

    def __init__(
        self,
        robot: RobotHandModel,
        model_path: str = "数据手套数据/CH-M6型号_虚拟手模型/CH-M6_L.xml",
        render: bool = False,
        camera: str = "overview",
        task_object_body: str = "grasp_object",
        task_name: str = "wrap",
        task_spec: dict | None = None,
        auto_reset_object: bool = True,
        object_workspace_radius_m: float = 0.28,
        object_min_z_m: float = 0.045,
        stability_orientation_deg: float = 3.0,
        hold_window_s: float = 1.0,
        hold_max_translation_m: float = 0.001,
        hold_max_rotation_deg: float = 1.0,
        camera_presets: dict | None = None,
        scene_config: dict | None = None,
        control_dt_s: float | None = None,
        contact_detection: dict | None = None,
    ) -> None:
        try:
            import mujoco
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("未安装 mujoco。请执行 pip install -r requirements.txt") from exc

        self.mujoco = mujoco
        self.robot = robot
        self.render = render
        self.viewer = None
        self.camera_name = camera
        self.task_name = str(task_name)
        self.task_spec = dict(task_spec or {})
        self.task_mode = str(self.task_spec.get("interaction", task_name))
        self.auto_reset_object = bool(auto_reset_object)
        self._task_active = False
        self.object_workspace_radius_m = float(object_workspace_radius_m)
        self.object_min_z_m = float(object_min_z_m)
        self.stability_orientation_deg = float(stability_orientation_deg)
        self.camera_presets = dict(camera_presets or {})
        self._contact_streak_frames = 0
        self._contact_filter = ForceContactDebouncer(ContactDetectionConfig(**(contact_detection or {})))
        self._object_held_by_fixture = False
        self._object_reset_this_step = False
        self._hold_tracker = RelativeGraspHold(
            sampling_hz=1.0 / float(control_dt_s) if control_dt_s is not None else 60.0,
            hold_window_s=hold_window_s,
            max_translation_m=hold_max_translation_m,
            max_rotation_deg=hold_max_rotation_deg,
        )
        # The glove/control loop runs at 60 Hz, while the MJCF physics timestep is
        # 3 ms. Older versions advanced MuJoCo only once per control frame, so
        # 16.67 ms of control time advanced only 3 ms of physics. That made the
        # index finger lag badly during the short PINCH window. We keep an exact
        # time accumulator and execute 5/6 substeps per control tick on average.
        self.control_dt_s = float(control_dt_s) if control_dt_s is not None else None
        self._physics_time_accumulator_s = 0.0
        self.last_physics_substeps = 0

        path = (ROOT / model_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"MuJoCo 模型文件不存在：{path}")

        scene_cfg = {
            "physics_timestep_s": 0.002,
            "palm_pos_m": [0.0, 0.0, 0.240],
            "palm_quat_wxyz": [0.0, 0.0, 1.0, 0.0],
            "table": {"pos": [0.0, 0.10, 0.035], "size": [0.24, 0.20, 0.035]},
            "task_object_body": task_object_body,
        }
        scene_cfg.update(dict(scene_config or {}))
        scene_cfg["task_object_body"] = task_object_body
        # Project-level adapter reads the teacher-provided XML/STL as immutable assets.
        # Scene entities and contact proxies exist only in this in-memory XML.
        xml_text, assets = build_ch_m6_scene(path, robot, scene_cfg)
        if task_spec:
            xml_text = self._configure_task_xml(xml_text, task_object_body, task_spec)
        self.model = mujoco.MjModel.from_xml_string(xml_text, assets=assets)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.physics_dt_s = float(self.model.opt.timestep)
        if self.control_dt_s is None:
            self.control_dt_s = self.physics_dt_s
        if self.control_dt_s <= 0.0:
            raise ValueError("control_dt_s must be positive")
        if self.physics_dt_s <= 0.0:
            raise ValueError("MuJoCo timestep must be positive")

        self.act_ids: list[int] = []
        for actuator in robot.actuator_order:
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator)
            if aid < 0:
                raise RuntimeError(f"Missing CH-M6 actuator {actuator} in MJCF")
            self.act_ids.append(aid)
        if len(self.act_ids) != robot.effective_actuators:
            raise RuntimeError(
                f"Actuator count mismatch: model={len(self.act_ids)}, config={robot.effective_actuators}"
            )

        self._object_body_id: int | None = None
        self._object_geom_ids: set[int] = set()
        self._object_qpos_adr: int | None = None
        self._object_dof_adr: int | None = None
        self._object_qpos0: np.ndarray | None = None
        self._init_task_object(task_object_body)
        palm_name = str(self.robot.simulation_names["palm_body"])
        self._palm_body_id = int(mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, palm_name))
        if self._palm_body_id < 0:
            raise RuntimeError(f"Missing palm body for stability metric: {palm_name}")

        if render:
            import mujoco.viewer

            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._setup_camera(camera)
            self.viewer.sync()

    @staticmethod
    def _configure_task_xml(xml_text: str, body_name: str, spec: dict) -> str:
        root = ET.fromstring(xml_text)
        body = next((b for b in root.iter("body") if b.get("name") == body_name), None)
        if body is None:
            raise ValueError(f"Task body not found in MJCF: {body_name}")
        geom = next((g for g in body.findall("geom") if g.get("name") == "task_object_geom"), None)
        if geom is None:
            raise ValueError("task_object_geom not found in task body")
        if "pos" in spec:
            body.set("pos", " ".join(str(float(x)) for x in spec["pos"]))
        if "type" in spec:
            geom.set("type", str(spec["type"]))
        if "size" in spec:
            geom.set("size", " ".join(str(float(x)) for x in spec["size"]))
        if "mass_density" in spec:
            geom.set("density", str(float(spec["mass_density"])))
        return ET.tostring(root, encoding="unicode")

    def _setup_camera(self, camera: str) -> None:
        """设置可交互的自由相机初始视角。

        v0.3 使用 mjCAMERA_FIXED，所以 overview/closeup/side 会锁死视点。
        v0.3.1 将这些名称改成 FREE camera 的“初始预设”：启动后仍可
        用 MuJoCo Viewer 的鼠标操作任意旋转、平移和缩放。
        """
        if self.viewer is None:
            return

        self.viewer.cam.type = self.mujoco.mjtCamera.mjCAMERA_FREE
        self.viewer.cam.fixedcamid = -1

        name = camera.lower()
        if name == "free":
            return

        preset = self.camera_presets.get(name)
        if preset is None:
            available = ", ".join(sorted(self.camera_presets)) or "无"
            raise ValueError(f"自由相机预设不存在：{camera}；可用预设：{available}, free")

        lookat = preset.get("lookat")
        if lookat is not None:
            if len(lookat) != 3:
                raise ValueError(f"camera preset {camera}.lookat 必须包含 3 个数")
            self.viewer.cam.lookat[:] = np.asarray(lookat, dtype=float)
        if "distance" in preset:
            self.viewer.cam.distance = float(preset["distance"])
        if "azimuth" in preset:
            self.viewer.cam.azimuth = float(preset["azimuth"])
        if "elevation" in preset:
            self.viewer.cam.elevation = float(preset["elevation"])

    def _init_task_object(self, body_name: str) -> None:
        bid = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid < 0:
            return
        self._object_body_id = int(bid)
        for gid in range(self.model.ngeom):
            if int(self.model.geom_bodyid[gid]) == bid:
                self._object_geom_ids.add(int(gid))

        jadr = int(self.model.body_jntadr[bid])
        jnum = int(self.model.body_jntnum[bid])
        if jadr < 0 or jnum < 1:
            return
        jid = jadr
        if int(self.model.jnt_type[jid]) != int(self.mujoco.mjtJoint.mjJNT_FREE):
            return
        self._object_qpos_adr = int(self.model.jnt_qposadr[jid])
        self._object_dof_adr = int(self.model.jnt_dofadr[jid])
        qadr = self._object_qpos_adr
        self._object_qpos0 = self.data.qpos[qadr : qadr + 7].copy()

    def set_camera(self, camera: str) -> None:
        if self.viewer is None:
            return
        self._setup_camera(camera)
        self.camera_name = camera

    def is_running(self) -> bool:
        if self.viewer is None:
            return True
        return bool(self.viewer.is_running())

    def reset_task_object(self) -> None:
        if self._object_qpos_adr is None or self._object_qpos0 is None:
            return
        qadr = self._object_qpos_adr
        self.data.qpos[qadr : qadr + 7] = self._object_qpos0
        if self._object_dof_adr is not None:
            dadr = self._object_dof_adr
            self.data.qvel[dadr : dadr + 6] = 0.0
        self._contact_streak_frames = 0
        self._contact_filter.reset()
        self._hold_tracker.reset()
        self._object_reset_this_step = True
        self.mujoco.mj_forward(self.model, self.data)

    def _maybe_reset_task_object(self) -> None:
        if not self.auto_reset_object or self._object_qpos_adr is None or self._object_qpos0 is None:
            return
        qadr = self._object_qpos_adr
        pos = np.asarray(self.data.qpos[qadr : qadr + 3], dtype=float)
        home = np.asarray(self._object_qpos0[:3], dtype=float)
        xy_offset = float(np.linalg.norm(pos[:2] - home[:2]))
        if pos[2] < self.object_min_z_m or xy_offset > self.object_workspace_radius_m:
            self.reset_task_object()

    def _hold_task_object_pose(self, *, forward: bool = False) -> None:
        """Keep the free task object at its reproducible initial pose.

        PINCH uses this only during the approach phase.  It prevents one digit
        from knocking the tall calibration block over before the opposite pad
        arrives.  Once both fingertip pads touch, the caller releases the object
        again, so the stability metric still has to survive free-body physics.
        """
        if self._object_qpos_adr is None or self._object_qpos0 is None:
            return
        qadr = self._object_qpos_adr
        self.data.qpos[qadr : qadr + 7] = self._object_qpos0
        if self._object_dof_adr is not None:
            dadr = self._object_dof_adr
            self.data.qvel[dadr : dadr + 6] = 0.0
        if forward:
            self.mujoco.mj_forward(self.model, self.data)

    def step(self, target_q: np.ndarray, *, hold_object: bool = False) -> np.ndarray:
        self._object_reset_this_step = False
        self._object_held_by_fixture = bool(hold_object)
        if hold_object:
            self._hold_tracker.reset()
        target_q = np.asarray(target_q, dtype=float)
        controls = self.robot.actuator_targets_from_joint_targets(target_q)
        for aid, u in zip(self.act_ids, controls):
            self.data.ctrl[aid] = float(u)

        # Advance approximately one control period of physical time. For the
        # default 60 Hz controller and 3 ms MJCF timestep this alternates between
        # 5 and 6 physics steps (average 5.56), instead of the single 3 ms step
        # used in v0.4.2. The accumulator avoids long-term clock drift.
        #
        # During PINCH approach, ``hold_object`` makes the free object behave as a
        # temporary fixture: each substep starts from the same initial object pose.
        # This removes the common failure mode where the first fingertip contact
        # topples the block before the second fingertip arrives.
        if hold_object:
            self._hold_task_object_pose()
        self._physics_time_accumulator_s += float(self.control_dt_s)
        substeps = 0
        max_substeps = 64  # defensive guard for accidental configuration errors
        while (
            self._physics_time_accumulator_s + 1e-12 >= self.physics_dt_s
            and substeps < max_substeps
        ):
            self.mujoco.mj_step(self.model, self.data)
            if hold_object:
                self._hold_task_object_pose()
            self._physics_time_accumulator_s -= self.physics_dt_s
            substeps += 1
        self.last_physics_substeps = substeps

        # Recompute derived poses/contact pairs for the held pose so the contact
        # metrics and Viewer correspond to the same object state.
        if hold_object:
            self._hold_task_object_pose(forward=True)

        if substeps >= max_substeps and self._physics_time_accumulator_s >= self.physics_dt_s:
            # Do not allow an invalid control period to create an unbounded loop.
            self._physics_time_accumulator_s = 0.0

        self._maybe_reset_task_object()
        if self.viewer is not None:
            self.viewer.sync()
        return self.joint_positions()

    def joint_positions(self) -> np.ndarray:
        q = []
        for j in self.robot.joint_order:
            jid = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, j)
            if jid < 0:
                raise RuntimeError(f"Missing joint {j} in MJCF")
            adr = int(self.model.jnt_qposadr[jid])
            q.append(float(self.data.qpos[adr]))
        return np.asarray(q)

    def _other_contact_geom(self, contact) -> int | None:
        g1, g2 = int(contact.geom1), int(contact.geom2)
        if g1 in self._object_geom_ids and g2 not in self._object_geom_ids:
            return g2
        if g2 in self._object_geom_ids and g1 not in self._object_geom_ids:
            return g1
        return None

    def contact_metrics(self) -> dict[str, float | int | bool]:
        """Force-confirmed hand/object contact vs temporally filtered control state.

        MuJoCo can expose *inactive* geometric contacts in data.contact. Only
        active constraints with positive normal force are physical evidence.
        Raw force-backed contact is used for task success and free-body hold;
        debounced flags are for contact-assist and task-space servo only.
        """
        fingers = ("thumb", "index", "middle", "ring", "little")
        tip_names = {f: self.robot.tip_geom_name(f) for f in fingers}
        forces = {f: 0.0 for f in fingers}
        forces.update({f"{f}_tip": 0.0 for f in fingers})
        forces['palm'] = 0.0
        # Sum actual world-frame contact forces *on the object* by the hand digit.
        # Contact frame rows are world-frame axes (MuJoCo convention).
        force_on_object = {f: np.zeros(3, dtype=float) for f in fingers}
        geometry_hits = {key: False for key in forces}
        hand_contact_count = 0
        environment_contact_count = 0
        candidate_count = 0
        total_hand_normal_force_n = 0.0
        max_contact_penetration_m = 0.0
        for i in range(int(self.data.ncon)):
            c = self.data.contact[i]
            other = self._other_contact_geom(c)
            if other is None:
                continue
            # Classify *geometric candidates* before force gating so the
            # ablation can reproduce the original optimistic ncon baseline.
            gname = self.mujoco.mj_id2name(
                self.model, self.mujoco.mjtObj.mjOBJ_GEOM, other) or ""
            bid = int(self.model.geom_bodyid[other])
            bname = self.mujoco.mj_id2name(
                self.model, self.mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            digit = self.robot.finger_for_body(bname)
            is_palm = bname == str(self.robot.simulation_names["palm_body"])
            if digit in fingers:
                geometry_hits[digit] = True
                if gname == tip_names[digit]:
                    geometry_hits[f'{digit}_tip'] = True
            elif is_palm:
                geometry_hits['palm'] = True
            # A gap-zone contact may appear in ncon without entering the
            # physics solver. It must not be reported as tactile contact.
            if int(c.efc_address) < 0:
                candidate_count += 1
                continue
            spatial_force = np.zeros(6, dtype=float)
            self.mujoco.mj_contactForce(self.model, self.data, i, spatial_force)
            normal_force_n = float(spatial_force[0])
            if not np.isfinite(normal_force_n) or normal_force_n <= 0.0:
                candidate_count += 1
                continue
            if digit is None and not is_palm:
                environment_contact_count += 1
                continue
            hand_contact_count += 1
            total_hand_normal_force_n += normal_force_n
            max_contact_penetration_m = max(max_contact_penetration_m, max(0.0, -float(c.dist)))
            if digit in fingers:
                forces[digit] += normal_force_n
                if hasattr(c, 'frame'):
                    # mj_contactForce is expressed in frame axes. Its world
                    # vector acts on geom2; the equal/opposite acts on geom1.
                    axes = np.asarray(c.frame, dtype=float).reshape(3, 3)
                    world_on_geom2 = axes.T @ spatial_force[:3]
                    sign = +1.0 if int(c.geom2) in self._object_geom_ids else -1.0
                    force_on_object[digit] += sign * world_on_geom2
                if gname == tip_names[digit]:
                    forces[f'{digit}_tip'] += normal_force_n
            elif is_palm:
                forces['palm'] += normal_force_n

        raw_flags = {name: value >= self._contact_filter.cfg.enter_force_n
                     for name, value in forces.items()}
        mode = self._contact_filter.cfg.mode
        if mode == 'geometry_only':
            filtered = dict(geometry_hits)
        elif mode == 'force_only':
            filtered = dict(raw_flags)
        elif mode == 'temporal_only':
            # Hysteresis in time, no contact force criterion: this includes
            # inactive gap contacts, precisely what this ablation tests.
            filtered = self._contact_filter.update({
                key: self._contact_filter.cfg.enter_force_n if hit else 0.0
                for key, hit in geometry_hits.items()})
        else:
            filtered = self._contact_filter.update(forces)
        raw_tips = {f: raw_flags[f'{f}_tip'] for f in fingers}
        tip_flags = {f: filtered[f'{f}_tip'] for f in fingers}
        raw_pinch = raw_tips['thumb'] and raw_tips['index']
        raw_wrap = raw_flags['thumb'] and any(raw_flags[f] for f in fingers[1:])
        pinch_control = tip_flags['thumb'] and tip_flags['index']
        wrap_control = filtered['thumb'] and any(filtered[f] for f in fingers[1:])
        raw_task_proxy = bool(raw_pinch if self.task_mode == 'pinch' else raw_wrap)
        filtered_task_proxy = bool(pinch_control if self.task_mode == 'pinch' else wrap_control)
        if raw_task_proxy:
            self._contact_streak_frames += 1
        else:
            self._contact_streak_frames = 0

        orient_deg = self.object_orientation_drift_deg()
        stable_orientation = orient_deg <= self.stability_orientation_deg
        # In particular, do not use the debounced estimate for 'success': it
        # may remain True for a short time *after* force goes to zero.
        stable_contact_proxy = bool(raw_task_proxy and stable_orientation and not self._object_held_by_fixture)
        free_object = bool(self._task_active and self._object_qpos_adr is not None
                           and not self._object_held_by_fixture and not self._object_reset_this_step)
        if self._object_body_id is not None:
            object_pos, object_rot = self.task_object_pose()
        else:
            object_pos = object_rot = None
        hold_info = self._hold_tracker.update(
            contact=raw_task_proxy, free_object=free_object,
            palm_pos=np.asarray(self.data.xpos[self._palm_body_id], dtype=float).copy(),
            palm_rot=np.asarray(self.data.xmat[self._palm_body_id], dtype=float).reshape(3, 3).copy(),
            object_pos=object_pos, object_rot=object_rot,
        )
        sections = (int(filtered['thumb']) + int(any(filtered[f] for f in fingers[1:]))
                    + int(filtered['palm']))
        return {
            'contact_count': int(hand_contact_count),
            'environment_contact_count': int(environment_contact_count),
            'inactive_contact_candidates': int(candidate_count),
            'contact_sections': int(sections),
            **{f'raw_{f}_contact': bool(raw_flags[f]) for f in fingers},
            **{f'{f}_normal_force_n': float(forces[f]) for f in fingers},
            **{f'{f}_force_on_object_{axis}_n': float(force_on_object[f][i])
               for f in fingers for i, axis in enumerate('xyz')},
            **{f'{f}_contact': bool(filtered[f]) for f in fingers},
            'finger_contact': bool(any(filtered[f] for f in fingers[1:])),
            'palm_contact': bool(filtered['palm']),
            **{f'{f}_tip_contact': bool(tip_flags[f]) for f in fingers},
            **{f'raw_{f}_tip_contact': bool(raw_tips[f]) for f in fingers},
            'raw_pinch_contact_proxy': bool(raw_pinch),
            'raw_wrap_contact_proxy': bool(raw_wrap),
            'filtered_task_contact_proxy': filtered_task_proxy,
            'hand_normal_force_n': float(total_hand_normal_force_n),
            'thumb_tip_normal_force_n': float(forces['thumb_tip']),
            'index_tip_normal_force_n': float(forces['index_tip']),
            'max_contact_penetration_m': float(max_contact_penetration_m),
            'tip_contact_ratio': float(sum(tip_flags.values())/len(fingers)),
            'contact_streak_frames': int(self._contact_streak_frames),
            'orientation_drift_deg': float(orient_deg),
            'stable_orientation': bool(stable_orientation),
            'pinch_contact_proxy': bool(raw_pinch),
            'wrap_contact_proxy': bool(raw_wrap),
            'task_contact_proxy': raw_task_proxy,
            'stable_contact_proxy': stable_contact_proxy,
            'object_displacement_m': float(self.object_displacement_m()),
            'object_pos_x_m': float(object_pos[0]) if object_pos is not None else float('nan'),
            'object_pos_y_m': float(object_pos[1]) if object_pos is not None else float('nan'),
            'object_pos_z_m': float(object_pos[2]) if object_pos is not None else float('nan'),
            'object_reset_this_step': bool(self._object_reset_this_step),
            **hold_info,
        }

    def object_displacement_m(self) -> float:
        if self._object_qpos_adr is None or self._object_qpos0 is None:
            return 0.0
        qadr = self._object_qpos_adr
        return float(np.linalg.norm(self.data.qpos[qadr : qadr + 3] - self._object_qpos0[:3]))

    def object_orientation_drift_deg(self) -> float:
        if self._object_qpos_adr is None or self._object_qpos0 is None:
            return 0.0
        qadr = self._object_qpos_adr
        q0 = np.asarray(self._object_qpos0[3:7], dtype=float)
        q1 = np.asarray(self.data.qpos[qadr + 3 : qadr + 7], dtype=float)
        n0 = np.linalg.norm(q0)
        n1 = np.linalg.norm(q1)
        if n0 <= 1e-12 or n1 <= 1e-12:
            return 0.0
        dot = float(np.clip(abs(np.dot(q0 / n0, q1 / n1)), 0.0, 1.0))
        return math.degrees(2.0 * math.acos(dot))

    def set_task_active(self, active: bool, *, hold_when_inactive: bool = True) -> bool:
        """设置当前任务是否进入有效交互阶段。

        返回 True 表示本次调用发生了 inactive->active 跳变，并已将任务物体
        复位到标准初始位姿。这样 WRAP 圆柱不会在前置 NEUTRAL/PINCH 动作中
        被提前碰倒，PINCH 薄块也能从可复现实验初态开始。
        """
        active = bool(active)
        rising = active and not self._task_active
        if rising:
            self.reset_task_object()
        elif not active and hold_when_inactive:
            self.reset_task_object()
        self._task_active = active
        if not active:
            self._hold_tracker.reset()
        return rising

    def task_object_pose(self) -> tuple[np.ndarray, np.ndarray]:
        if self._object_body_id is None:
            raise RuntimeError("Task object body is unavailable")
        center = np.asarray(self.data.xpos[self._object_body_id], dtype=float).copy()
        R = np.asarray(self.data.xmat[self._object_body_id], dtype=float).reshape(3, 3).copy()
        return center, R

    def fingertip_position(self, finger: str) -> np.ndarray:
        sid = self.mujoco.mj_name2id(
            self.model, self.mujoco.mjtObj.mjOBJ_SITE, self.robot.tip_site_name(finger)
        )
        if sid < 0:
            raise RuntimeError(f"Missing fingertip site: {finger}_tip")
        return np.asarray(self.data.site_xpos[sid], dtype=float).copy()

    def fingertip_jacobian(self, finger: str, joint_names: tuple[str, ...]) -> np.ndarray:
        sid = self.mujoco.mj_name2id(
            self.model, self.mujoco.mjtObj.mjOBJ_SITE, self.robot.tip_site_name(finger)
        )
        if sid < 0:
            raise RuntimeError(f"Missing fingertip site: {finger}_tip")
        jacp = np.zeros((3, self.model.nv), dtype=float)
        jacr = np.zeros((3, self.model.nv), dtype=float)
        self.mujoco.mj_jacSite(self.model, self.data, jacp, jacr, sid)
        cols: list[int] = []
        for name in joint_names:
            jid = self.mujoco.mj_name2id(
                self.model, self.mujoco.mjtObj.mjOBJ_JOINT, name
            )
            if jid < 0:
                raise RuntimeError(f"Missing joint for Jacobian: {name}")
            cols.append(int(self.model.jnt_dofadr[jid]))
        return jacp[:, cols].copy()

    def task_contact_target(
        self,
        finger: str,
        task_name: str,
        *,
        clearance_m: float,
        vertical_margin_m: float,
    ) -> np.ndarray:
        """Generate a reachable fingertip-center target on the current object surface.

        ``task_name`` is the interaction mode (``pinch``/``wrap``/``sphere``), not
        necessarily the catalog name. This lets multiple standard objects reuse the
        same control law. The target follows the current free object's pose.
        """
        center, R = self.task_object_pose()
        tip = self.fingertip_position(finger)
        local_tip = R.T @ (tip - center)
        spec = self.task_spec
        kind = str(spec.get("type", "cylinder"))
        size = [float(x) for x in spec.get("size", [])]
        c = float(clearance_m)
        margin = max(0.0, float(vertical_margin_m))
        mode = str(spec.get("interaction", task_name))

        if mode == "pinch" and kind == "box" and len(size) >= 3:
            half = np.asarray(size[:3], dtype=float)
            face = dict(spec.get("contact_faces", {}).get(finger, {}))
            axis = int(face.get("axis", 0))
            sign = float(face.get("sign", 1.0 if local_tip[axis] >= 0 else -1.0))
            local_target = np.clip(local_tip, -half + margin, half - margin)
            local_target[axis] = sign * (half[axis] + c)
            return center + R @ local_target
        if mode in {"wrap", "sphere"}:
            if kind == "sphere" and len(size) >= 1:
                radius = size[0]
                v = local_tip.copy()
                target_radius = radius + c
                n = float(np.linalg.norm(v))
                if n < 1e-9:
                    v = np.array([0.0, 1.0, 0.0])
                    n = 1.0
                local_target = v / n * target_radius
                return center + R @ local_target

            if kind == "cylinder" and len(size) >= 2:
                radius, half_h = size[:2]
                radial = local_tip[:2].copy()
                n = float(np.linalg.norm(radial))
                if n < 1e-9:
                    default_dir = {
                        "thumb": np.array([0.0, -1.0]),
                        "index": np.array([-0.35, 1.0]),
                        "middle": np.array([0.0, 1.0]),
                        "ring": np.array([0.35, 1.0]),
                        "little": np.array([0.65, 0.85]),
                    }
                    radial = default_dir.get(finger, np.array([0.0, 1.0]))
                    n = float(np.linalg.norm(radial))
                xy = radial / n * (radius + c)
                z_margin = min(margin, max(0.0, half_h * 0.35))
                z = float(np.clip(local_tip[2], -half_h + z_margin, half_h - z_margin))
                local_target = np.array([xy[0], xy[1], z], dtype=float)
                return center + R @ local_target

            if kind == "box" and len(size) >= 3:
                half = np.asarray(size[:3], dtype=float)
                # Closest point on the box; then move outward by fingertip-center
                # clearance. This supports box/cuboid envelope-grasp benchmarks.
                surface = np.clip(local_tip, -half, half)
                outside = local_tip - surface
                n = float(np.linalg.norm(outside))
                if n < 1e-9:
                    # If numerically inside, choose the nearest face.
                    face_gap = half - np.abs(local_tip)
                    axis = int(np.argmin(face_gap))
                    normal = np.zeros(3, dtype=float)
                    normal[axis] = 1.0 if local_tip[axis] >= 0.0 else -1.0
                    surface[axis] = normal[axis] * half[axis]
                else:
                    normal = outside / n
                local_target = surface + c * normal
                return center + R @ local_target

        # Unknown task/geometry: do not inject an unsafe correction.
        return tip.copy()

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
