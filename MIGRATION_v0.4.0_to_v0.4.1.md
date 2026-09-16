# v0.4.1 PINCH contact fix

This patch keeps the v0.4.0 synthetic dataset and the successful WRAP setup, and targets the remaining precision-pinch failure.

## What changed

1. `task_space_servo.pinch.stop_on_contact: tip`
   - non-tip phalanx contact no longer freezes thumb/index before their pads touch.
2. `target_offset_m: -0.002`
   - uses a small virtual penetration target. MuJoCo collision stops the real pad at the object surface.
3. Stronger pinch preshape (`posture_prior_blend=0.90`, `command_blend=0.90`).
4. PINCH object widened in X while staying thin in Y, reducing toppling without making the pinch gap easier in Y.
5. Terminal prints `Ttip`, `Itip`, `Terr`, and `Ierr` every 30 active frames.
6. Keeps the WRAP-success servo tuning: gain 0.88, max step 6.5 deg, stop error 0.8 mm, wrap clearance 5.5 mm.

## Run

```powershell
python -m src.main --sim mujoco --render --realtime --input dataset --dataset datasets\synthetic_glove_v1.csv --dataset-trial 0 --task pinch --steps 600 --camera closeup
```

Success target:

```text
Ttip=1 Itip=1
[RESULT] task=pinch success_proxy=True ...
```

`success_proxy` remains the fixed-base stable-contact proxy; it is not lift/transport success.
