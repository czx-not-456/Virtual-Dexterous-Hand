# v0.4.3

## PINCH physics-clock fix

- Preserve the v0.4.2 WRAP contact parameters and PINCH fingertip-only contact logic.
- Synchronize MuJoCo physical time with the 60 Hz glove/control loop.
- The MJCF timestep is 3 ms, so each 16.67 ms control tick now executes 5/6 physics substeps (5.56 average) via an accumulator.
- This specifically addresses the observed PINCH trace where index fingertip target error fell only from ~121 mm to ~86 mm before the PINCH window ended.
- Add a startup diagnostic showing control timestep, MuJoCo timestep, and average substep count.
