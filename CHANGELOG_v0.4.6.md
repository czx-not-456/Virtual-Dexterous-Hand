# v0.4.6 — PINCH reachability fix

## Why v0.4.5 stalled
The index fingertip error dropped rapidly and then plateaued near 7–8 mm while
`index_tip_contact` stayed false. This was not a timing problem: the old PINCH
box ended at z=0.202 m, while the one-tendon index trajectory approaches the
object around z≈0.209 m. With an 8 mm fingertip pad, the minimum sphere-to-box
distance was still about 10.3 mm, so true fingertip contact was geometrically
impossible even though the servo kept converging.

## Changes
- PINCH calibration block: 90 × 30 × 140 mm, bottom still on the table.
- PINCH fingertip-center offset: +7 mm (no virtual interpenetration).
- Reachability-aware index prior: MCP/PIP/DIP = 70/95/70 deg.
- Synthetic trial cycle extended to 800 frames with 240-frame PINCH and WRAP phases.
- Added regression tests for actual index-pad/box geometric intersection.

WRAP control and the v0.4.3 physics-clock synchronization are unchanged.
