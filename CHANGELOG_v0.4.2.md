# v0.4.2 — PINCH + WRAP consolidated contact-control release

This release consolidates the validated WRAP contact tuning and the PINCH fingertip-contact fix into one package.

## WRAP retained from validated v0.4.0 tuning

- `task_space_servo.enabled: true`
- global servo gain `0.88`
- global DLS damping `0.018`
- global command blend `0.96`
- maximum per-frame joint step `6.5 deg`
- stop error `0.8 mm`
- WRAP surface clearance `5.5 mm`
- WRAP still freezes a digit on any contact (`stop_on_contact` defaults to `any`) to avoid excessive penetration.

These are the settings that produced `success_proxy=True` in the validated WRAP run.

## PINCH fix retained from v0.4.1

- PINCH stops a digit only when its fingertip pad contacts the object (`stop_on_contact: tip`).
- PINCH uses a signed virtual contact target of `-2.0 mm` (`target_offset_m`) so the controller keeps closing until physical contact occurs.
- PINCH keeps a strong posture prior and task-space command blend for thumb/index alignment.
- The thin pinch block uses a wider footprint while preserving a thin pinch direction to reduce premature tipping.

## Compatibility

The CLI and dataset format are unchanged. Existing commands for `pinch`, `wrap`, `sphere`, `card`, `bottle`, and `box` continue to work.
