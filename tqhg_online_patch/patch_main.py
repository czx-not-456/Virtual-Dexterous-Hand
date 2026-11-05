from pathlib import Path


def patch(source: str) -> str:
    def replace(old, new):
        nonlocal source
        n = source.count(old)
        if n != 1:
            raise ValueError(f"Expected one patch anchor; got {n}: {old[:90]!r}")
        source = source.replace(old, new)

    if 'TQHG_ONLINE_INTEGRATED_v1' in source:
        raise ValueError('Online patch has already been installed')
    if '--coupled-servo-ablation' not in source or '--servo-trace' not in source:
        raise ValueError('First install CHM6_TEQS_Coupled_Servo_Diagnostic_Ablations.zip')

    replace('from src.retargeting.task_effect_spectrum import TaskEffectSpectrumMapper\n',
            'from src.retargeting.task_effect_spectrum import TaskEffectSpectrumMapper\n'
            'from src.retargeting.tqhg_online import TQHGOnlineMapper  # TQHG_ONLINE_INTEGRATED_v1\n')
    replace('from src.retargeting.optimizer import IntentDrivenRetargeter\n',
            'from src.retargeting.optimizer import IntentDrivenRetargeter, RetargetResult\n')
    replace('EVALUATION_COLUMNS = [\n',
            'ONLINE_COLUMNS = ["mapping_w_task", "mapping_w_null", "mapping_visible_rank", "mapping_box_clipped"]\n\n'
            'EVALUATION_COLUMNS = [\n')
    replace('    return p\n\n\ndef _empty_contact_metrics()',
            '    p.add_argument("--mapping-method", choices=["legacy", "ridge6", "pca4", "fixed-blend", "single-gate", "tqhg"],\n'
            '                   default="legacy", help="对照用15->6映射。legacy维持原控制流程")\n'
            '    p.add_argument("--tqhg-calibration", type=Path, default=Path("configs/tqhg_online_v1.json"),\n'
            '                   help="独立历史训练和校准形成的6维编码器与残差数据")\n'
            '    return p\n\n\ndef _empty_contact_metrics()')
    replace('    driver = None\n    sim_adapter = None\n',
            '    online_mapper = None\n'
            '    if args.mapping_method != "legacy":\n'
            '        if input_mode != "glove" or args.task_effect_spectrum or args.adaptive_synergy or args.reachability_synergy or args.credible_contact_synergy:\n'
            '            raise SystemExit("--mapping-method 非legacy只支持 --input glove，不能同时启用方案B/C/D/E")\n'
            '        if latent_mapper is not None:\n'
            '            raise SystemExit("在线映射使用独立 --tqhg-calibration 模型；请去掉 --latent-model")\n'
            '        online_mapper = TQHGOnlineMapper(robot, _project_path(args.tqhg_calibration))\n'
            '        print(f"[INFO] 在线统一6维映射={args.mapping_method}；模型={args.tqhg_calibration}")\n'
            '    driver = None\n    sim_adapter = None\n')
    replace('            latent_mapper=latent_mapper,\n            latent_source=args.latent_source,\n',
            '            latent_mapper=latent_mapper if online_mapper is None else None,\n'
            '            latent_source=args.latent_source,\n')
    replace('    if args.coupled_servo_ablation != "original" and not args.task_effect_spectrum:\n',
            '    if args.coupled_servo_ablation != "original" and not (args.task_effect_spectrum or online_mapper is not None):\n')
    replace('    if args.task_effect_spectrum:\n        print(f"[INFO] Coupled servo ablation=',
            '    if args.task_effect_spectrum or online_mapper is not None:\n        print(f"[INFO] Coupled servo ablation=')
    replace('                       + (SERVO_TRACE_COLUMNS if args.servo_trace else [])),\n',
            '                       + (SERVO_TRACE_COLUMNS if args.servo_trace else []) + ONLINE_COLUMNS),\n')
    old = '''            result = retargeter.solve(
                q_h,
                features.fingertip_positions,
                intent,
                q_prev_opt,
                dynamic=True,
                q_base=hand_frame.robot_joint_targets_rad,
                sensors_deg=hand_frame.metadata.get('raw_sensors_deg') if adaptive_mapper else None,
                controls_deg=(np.array([
                    hand_frame.metadata['base_mapping_outputs_deg'][name]
                    for name in glove_adapter.base_mapping.output_names]) if adaptive_mapper else None),
            )
            optimizer_ms = (time.perf_counter() - opt_t0) * 1000.0
'''
    new = '''            if online_mapper is not None:
                control = online_mapper.predict(
                    hand_frame.metadata['raw_sensors_deg'], method=args.mapping_method,
                    simulator=simulator, task_mode=task_mode)
                q_mapped = online_mapper.decode_controls(control)
                # Direct online mapping comparison: SAME downstream coupled Hold
                # for Ridge6/PCA4/fixed blend/single gate/TQHG; no hidden NLP.
                result = RetargetResult(q_mapped, q_mapped.copy(), 0.0, True, 0, {})
            else:
                result = retargeter.solve(
                    q_h,
                    features.fingertip_positions,
                    intent,
                    q_prev_opt,
                    dynamic=True,
                    q_base=hand_frame.robot_joint_targets_rad,
                    sensors_deg=hand_frame.metadata.get('raw_sensors_deg') if adaptive_mapper else None,
                    controls_deg=(np.array([
                        hand_frame.metadata['base_mapping_outputs_deg'][name]
                        for name in glove_adapter.base_mapping.output_names]) if adaptive_mapper else None),
                )
            optimizer_ms = (time.perf_counter() - opt_t0) * 1000.0
'''
    replace(old, new)
    replace('                    if isinstance(adaptive_mapper, TaskEffectSpectrumMapper):\n                        q_target, servo_diag = task_servo.apply_coupled(',
            '                    if isinstance(adaptive_mapper, TaskEffectSpectrumMapper) or online_mapper is not None:\n                        q_target, servo_diag = task_servo.apply_coupled(')
    replace('                            contact=last_contact, mapper=adaptive_mapper)\n',
            '                            contact=last_contact, mapper=(online_mapper if online_mapper is not None else adaptive_mapper))\n')
    replace('            q_filtered = (q_target if isinstance(adaptive_mapper, TaskEffectSpectrumMapper)\n',
            '            q_filtered = (q_target if online_mapper is not None else q_target if isinstance(adaptive_mapper, TaskEffectSpectrumMapper)\n')
    replace('            if adaptive_mapper is not None:\n                # Preserve the complete six-command manifold, including residuals.\n',
            '            if online_mapper is not None:\n'
            '                q_out = online_mapper.project_and_rate_limit(q_filtered, q_prev_command, vmax, dt)\n'
            '                q_prev_command = q_out.copy()\n'
            '            elif adaptive_mapper is not None:\n'
            '                # Preserve the complete six-command manifold, including residuals.\n')
    replace('                    *trace_values,\n                    *q_actual.tolist(),\n',
            '                    *trace_values,\n'
            '                    *([online_mapper.last_gate.get("w_task", float("nan")),\n'
            '                        online_mapper.last_gate.get("w_null", float("nan")),\n'
            '                        online_mapper.last_gate.get("visible_rank", float("nan")),\n'
            '                        float(online_mapper.last_gate.get("box_clipped", False))]\n'
            '                      if online_mapper is not None else [float("nan")] * len(ONLINE_COLUMNS)),\n'
            '                    *q_actual.tolist(),\n')
    replace('                "coupled_servo_ablation": args.coupled_servo_ablation,\n',
            '                "coupled_servo_ablation": args.coupled_servo_ablation,\n'
            '                "mapping_method": args.mapping_method,\n'
            '                "tqhg_calibration": str(args.tqhg_calibration) if online_mapper is not None else None,\n'
            '                "online_comparison_bypasses_legacy_optimizer": bool(online_mapper is not None),\n')
    return source
