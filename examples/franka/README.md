# Franka Panda visualisations

Three scripts, all run from the repo root with the project venv.

| script | shows |
|---|---|
| `run_fields.py` | the CERG navigation field around the moving arm: attraction, repulsion, their sum |
| `dsm_explainer.py` | one DSM evaluation: the predicted rollout as ghosts, the hard-obstacle margin, then the live loop |
| `emin_press.py` | E_min: a contact tip pressing on a solid sphere, with δ_s(t) and E(t) beside the render |

`run_fields.py` opens a live window by default and can write an `.mp4` instead;
the other two write an `.mp4` and then play it back in a window.
Videos go in `examples/franka/videos/`; play one with `xdg-open <file>`.

## Fields: `run_fields.py`

```
python examples/franka/run_fields.py                       # scenario around, 3D view
python examples/franka/run_fields.py --scenario soft --solid
```

One window opens, laid out like the videos: the MuJoCo render on the left,
with the field arrows drawn on the robot, and the matplotlib 3D quiver with
the Panda's own meshes on the right. Drag the right half to rotate it.
`--separate-viewer` gives the old two windows, with the interactive MuJoCo
viewer; `--no-viewer` gives the quiver alone.

### Scenarios (`--scenario`)

| name | world | what happens |
|---|---|---|
| `around` (default) | joint-1 swing, hard sphere r 0.12 on the flange path | arm lifts over it, reaches goal in ~5 s |
| `stall` | joint-1 swing, hard sphere r 0.22 on the wrist path | attraction and repulsion cancel, arm stops short |
| `soft` | joint-1 swing, soft sphere r 0.18 on the flange path | arm sinks in ~5 cm, slides through, reaches goal |
| `attraction` | forward reach, no obstacle | pure pull to the goal |
| `hard`, `total` | forward reach, hard sphere on the path | goal pose is inside the sphere, so these stall at once |

Scenario defaults that differ from `configs/panda_fields.yaml`, each
overridable by flag: `around`/`stall` use `kappa_hard = 10`; `soft` uses
`fd = 10` and `E_max = 2`. The script prints the values it runs with.

### What to draw (`--field`)

| flag | arrows |
|---|---|
| default | the scenario's own choice |
| `--field attraction` | blue pull toward the goal only |
| `--field repulsion` | red (hard) or amber (soft) push off the obstacle only |
| `--field both` | both, in their own colours |
| `--field sum` | one dark field, attraction + repulsion added; thick arrows on the links are where each link is sent |

The robot always moves under the full governor. The flag changes only what is drawn.

Hard spheres also show two rings: solid at ζ_h where the repulsion starts,
dashed at δ_h where it reaches unit strength. Soft spheres show the surface
ring and, unless `--solid`, the dark core deeper than δ_s.

### Soft contact

```
python examples/franka/run_fields.py --scenario soft                      # virtual sphere, arm passes through
python examples/franka/run_fields.py --scenario soft --solid              # real sphere r 0.12 inside the soft one
python examples/franka/run_fields.py --scenario soft --solid --E-max 0.6  # presses more gently
python examples/franka/run_fields.py --scenario soft --delta-s 0.05       # softer field, shallower sinking
```

`--solid` adds a collision sphere to the MuJoCo scene (`cerg/models/franka/scene_cerg_solid.xml`),
concentric with the soft constraint and smaller by about a link thickness.
The arm is stopped at its surface; the bottom text line shows the contact
force, and the terminal prints peak force, contact time and energy against `E_max`.

Knobs: `--delta-s`, `--fd` (soft push is `Kp·depth/(δ_s·fd)`), `--E-max`,
`--solid-radius`, `--kappa-hard`, `--radius`, `--offset`.

### Two runs back to back

```
python examples/franka/run_fields.py --scenario soft --compare-delta-s 0.10 0.05
python examples/franka/run_fields.py --scenario soft --compare-E-max 2 0.6
```

Run 1 plays, then the scene resets and run 2 plays in the same figure and the
same video. Each run has its own colour for its penetration path. The
terminal prints a comparison table at the end.

### Recording

Videos are written to the path you give `--video`; keep them in
`examples/franka/videos/`, which is where the existing ones live.

```
python examples/franka/run_fields.py --scenario around --viz-decimation 33 --video examples/franka/videos/around.mp4
python examples/franka/run_fields.py --scenario soft --solid --duration 6 --viz-decimation 33 --video examples/franka/videos/soft_solid.mp4
python examples/franka/run_fields.py --save-frame out.png     # one PNG of the final figure instead
```

The video is the MuJoCo render on the left and the matplotlib figure on the
right, 30 fps. `--viz-decimation 33` makes one frame per 33 ms of simulation,
so the video plays in real time; the default (100 in 3D) plays 3x fast.
`--video` implies `--no-viewer`. A 6 s run records in about half a minute.

Other options: `--duration` (default 12 s), `--loop PERIOD` to swap the goal
back and forth, `--view 2d` for the Cartesian slice instead of 3D,
`--attraction grid|projected|both`, `--dpi`, `--no-dashboard`, `--realtime`.

### Energy strip

A live plot of closed-loop energy against `E_max` under the 3D view is
implemented but switched off: two commented lines in `run_fields.py` where
`FieldView3D` is built (`energy_strip=...`, `t_end=...`). Uncomment to
bring it back. It only appears for soft scenarios.

## DSM: `dsm_explainer.py`

```
python examples/franka/dsm_explainer.py                       # writes videos/dsm_explainer.mp4
python examples/franka/dsm_explainer.py --horizon 0.1         # shorter prediction, writes dsm_explainer_h0.1.mp4
python examples/franka/dsm_explainer.py --scenario hard --duration 6 --out videos/dsm_hard.mp4
python examples/franka/dsm_explainer.py --scenario soft          # writes videos/dsm_explainer_soft.mp4
```

`--scenario soft` is the same world as `run_fields.py --scenario soft --solid`:
a soft constraint around a real sphere. The right panel then plots Δ_s (min
predicted clearance), Δ_e = E_max − E, their max, which is the soft DSM term,
and the DSM. E_max defaults to 1 J there so that the energy reaches the bound
while the arm presses on the sphere; `--E-max` overrides it.

Act 1 reveals the DSM's own rollout from the start pose as ghosts, with
poses that enter the obstacle in red. Act 2 runs the real loop with the
arm, the reference q_v and the goal as ghosts, plus the predicted rollout
each frame. The right panel plots predicted distance over the horizon and
the DSM history. The world is the `around` scenario of `run_fields.py`.

Flags: `--scenario around|hard|soft`, `--E-max`, `--duration`, `--frame-every`, `--fps`,
`--horizon`, `--out`, `--height`, `--no-show`.

## E_min press: `emin_press.py`

```
python examples/franka/emin_press.py                    # E_min = 0.3, 0.6, 0.75 J -> videos/emin_press.mp4
python examples/franka/emin_press.py --E-min 0 0.6      # baseline (E_min off) against 0.6 J
python examples/franka/emin_press.py --E-min 0.6 --out examples/franka/videos/emin06.mp4
```

One video, the runs back to back, then a playback window (`--no-show` skips it).
The default video is about 18 s and renders in about half a minute.

The world: a contact tip, a 1.5 cm dome on a 3 cm post
(`FrankaPandaRobot(tip_radius=0.015, tip_length=0.03)`), presses down on a
solid sphere of radius 0.35 m (`scene_cerg_solid.xml`). The soft constraint
sits on the contact surface (sphere radius + dome radius), so signed
distance 0 is tip contact. The goal q_r, the same in every run, puts the
dome 0.22 m inside the sphere. Gains and everything else come from
`configs/panda_fields.yaml`, except the soft-field damping: fd = 10 here
(the YAML has 1), as in `run_fields.py --scenario soft`. The soft push on the
dome is `Kp·depth/(δ_s·fd)` with Kp = 20, so q_v's dome settles about
δ_s·fd/20 deep: at fd = 1 that is under 2 cm while the core is visible, at
fd = 10 it is 6–10 cm. `--fd 1` gives the YAML behaviour.

While E < E_min, δ_s grows (dδ_s/dt = κ_δs (E_min − E)), the soft push
weakens, q_v sinks into the sphere and the PD spring stretches until E
reaches E_min. Then δ_s stops: it never shrinks. With E_min = 0, δ_s stays
at δ_i and the tip only rests on the sphere.

Each run stops `--hold` seconds (default 2) after E reaches E_min: the dome
has been in contact for a second, E ≥ E_min, and δ_s has not moved for 0.5 s.
`--duration` (default 20 s) caps a run that never gets there; the terminal
says which happened.

Left, the render: the arm; the reference q_v as a blue ghost, its dome a
blue ball; the goal as a green marker inside the sphere; the sphere
translucent so both show; a blue line from the dome to q_v's dome (the PD
stretch); a red arrow for the contact force; and the δ_s core, a dark ball
of radius R − δ_s as `run_fields.py` draws it. The caption gives t, E, δ_s,
the contact force, the core radius, how deep q_v's dome is, and whether E is
still rising or has reached E_min.

Right: δ_s(t) on top, with a dotted line at R where the core disappears;
E(t) below, with a dashed line at each E_min. The E axis is scaled to the
E_min values (the title says so if anything runs off the top). One colour
per run; finished runs stay, faded.

After each run the terminal prints when E reached E_min, the final E, δ_s
and contact force, and every robot geom that touched the sphere; it should
be only `contact_tip`. Final frames are saved next to the video as
`<name>_key<run>.png`.

The default E_min values show the core at three sizes:

| E_min | E reached at | E at the end | δ_s | core R − δ_s | contact force |
|---|---|---|---|---|---|
| 0.3 J | 4.1 s | 0.31 J | 0.20 m | 16 cm | 9.7 N |
| 0.6 J | 2.9 s | 0.63 J | 0.31 m | 6 cm | 13.9 N |
| 0.75 J | 2.1 s | 0.89 J | 0.37 m | gone at 1.4 s | 17.1 N |

E can end above E_min, most at 0.75 J: δ_s also grows during the approach,
while E is below E_min before contact, and it never shrinks, so at contact
the soft push is already weaker than E_min needs.

Limits. If you change `GOAL_DEPTH`, `TIP_LEN` or `SPHERE_R` at the top of the
script, check the contact list it prints. A deeper goal or a shorter post
can let q_v slide sideways along the soft surface until the wrist turns
about the tip and link6 or link7 lands on the sphere.

Flags: `--E-min` (up to 4 values), `--fd`, `--hold`, `--duration`,
`--frame-every` (0.04 plays in real time; 0.08 plays 2x), `--fps`, `--out`,
`--height`, `--no-show`.

## RRR arm

The 3-link arm has its own sphere demos under `examples/rrr/`:

```
python examples/rrr/sphere_demo.py --kind hard --scenario two      # Meshcat + plots
python examples/rrr/sphere_anim.py --kind soft --scenario link3    # 2D matplotlib animation
```

Scenarios: `link3` `tip` `wrist` `link2` `big` `two` `three` `blocked`.
Flags: `--kind hard|soft|both`, `--E-max`, `--bend`, `--offset`, `--no-constraint`.
