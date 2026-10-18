# Methods report: zero-shot MAST equilibria from a free-boundary Grad-Shafranov solve constrained by Thomson scattering

**Entrant:** Matteo Passero (Codabench, solo). **Status:** final pipeline, code frozen 2026-10-18. The Final-phase MAST
file is produced by the configuration of Development submission 992762 ("v20", S_MAST 0.7106).

## 1. Summary
- **DIII-D file (Challenge 1 / the G_ratio denominator):** PCA (20 components) of the flux map with ridge regression from
  72 engineered inputs (coil currents and their rates, solenoid change since breakdown, F-coil/Ip ratios, Thomson-derived
  plasma edges, |Ip|, |B_T|), trained on 300 DIII-D training shots. One physics correction: on reversed-current shots
  (Ip < 0, ~8% of DIII-D shots) EFIT stores the map with the opposite sign relative to the coil flux, so targets are
  sign-normalised by sign(Ip) before fitting and restored after. Trains in ~7 s on one CPU core.
- **MAST file (Challenge 2):** no learned model. Every frame is a free-boundary Grad-Shafranov equilibrium computed
  from the measured coil currents, the measured plasma current and the Thomson profiles. Three settings enter: the ratio
  of total to electron pressure (2.42, measured on DIII-D training data), the electron temperature at the last closed
  flux surface (30 eV; DIII-D gives 40 eV, MAST label-free tests and the Development board both preferred 30 eV) and the
  Spitzer exponent of the current profile (1.0, chosen on the Development board between 2.5, 1.5 and 1.0).

## 2. MAST pipeline
1. **Coil flux (exact).** Green's functions of all 812 shipped conductor elements (each subdivided 3 x 3) on the 65 x 65
   grid; P-coil columns are total kA-turn shared by their elements; the solenoid column is the circuit current, and
   each of its 656 conductor elements carries HALF of it (f_sol = 0.5): MAST's solenoid is "324 turns of 12 mm x 18 mm
   ... copper wound two-in-hand in four layers" (Development of a high field solenoid magnet for the MAST Spherical
   Tokamak, IEEE, ieeexplore.ieee.org/document/1027724), i.e. two parallel conductors per winding position. The 656
   shipped elements are exactly those 12 x 18 mm conductors (2 x 328). Independently supported by label-free tests:
   with f_sol = 0.5 the correction field needed for radial equilibrium shows no dependence on the solenoid current.
2. **Plasma edge from Thomson.** Core + edge Thomson Te profiles (despiked, 5-point running median) give the outboard
   midplane radius where Te crosses the separatrix temperature Te_sep. On DIII-D, Te at EFIT's LCFS has a median of
   40 eV (6,263 frames of 200 training shots, where the vertical core chord crosses the LCFS). On MAST we re-checked it
   label-free, because the laser crosses the plasma twice: pin the outboard edge at the Te_sep crossing, solve, and test
   (i) whether Te is equal on the inboard and outboard sides of each flux surface and (ii) where the model's inboard edge
   lands relative to the laser's inboard Te_sep crossing. On 36 shots in two independent sets, 40 eV was the worst value
   on both tests (isotherm mismatch 0.045-0.049 of peak Te; model inboard edge 5-6 cm inside the laser's) and 25-30 eV the
   flat optimum (0.038-0.040). We use 30 eV, the smallest step from the DIII-D calibration inside that optimum. A colder
   MAST separatrix is physically expected (less exhaust power and shorter connection lengths than DIII-D). The
   Development board agreed: 40 -> 30 eV raised S_MAST by 0.008 (v16 -> v18), with flux, q95/beta_N and shape all better.
3. **Free-boundary solve (Picard) with a kinetic current profile.** J = R p'(psi) + c_f (R0/R) g(psiN), where the
   pressure p = 2.42 ne Te comes from the Thomson profiles mapped onto the current iterate's flux surfaces, and
   g = (Te/Te_axis)^x shapes the remaining current; c_f normalises the total to the measured Ip. Spitzer conductivity
   suggests x = 1.5 for a fully relaxed current; the final pipeline uses x = 1.0 (a broader current). On the Development
   board x = 2.5 / 1.5 / 1.0 scored S_MAST 0.585 / 0.692 / 0.711 (v19 / v18 / v20), and every term improved from 1.5 to
   1.0 (flux R^2 0.945 -> 0.961, q95 R^2 0.505 -> 0.583, li R^2 -0.85 -> -0.07). Of our two label-free tests, the
   inboard-edge test (item 5) had pointed to 1.0 and the isotherm test to 1.5 or more; the board sided with the inboard
   edge. (Parametric fallback: f = (1 - psiN^alpha_m)^alpha_n with beta_p.) On a DIII-D bench with the true external
   field (60 held-out shots, 3,417 frames) the kinetic and parametric profiles tie (S 0.875 vs 0.878; li R^2 +0.20 vs
   -0.06).
   Plasma flux by Green's-function convolution (FFT along Z);
   magnetic axis = interior maximum; LCFS = lowest flux level whose axis-enclosing region does not reach the domain edge
   (snapped to the X-point value). A uniform vertical field B_v is adjusted each iteration so that the plasma's outboard
   midplane edge matches the Thomson edge (it stands for the unmeasured vessel currents and profile error). Frames whose
   solve fails a sanity check are replaced by interpolation in time between neighbouring good frames of the same shot.
   The Picard iteration does not settle to a unique fixed point (a small oscillation persists), so the returned map is
   the average of the last 25 of 60 iterates; on the DIII-D bench this made the score independent of the iteration count
   (0.887-0.888 for 60/100/200 iterations, vs 0.849-0.899 without averaging).
4. **Scalars.** q95 by integrating dl/(R^2 Bp) around the psiN = 0.95 contour of our own map, with F = mu0 N_TF I_TF / 2 pi
   (N_TF = 24, MAST engineering value). Code check: on the TRUE DIII-D maps it reproduces EFIT's q95 with R^2 = 0.9998.
   beta_N from the Thomson electron pressure mapped onto our flux surfaces, volume-averaged over our plasma, times 2.42
   (fitted on DIII-D: R^2 = 0.897 against EFIT beta_N on 3,810 frames).
5. **Inboard edge: a check, not a constraint.** We pin only the OUTBOARD edge, so the model's inboard midplane edge is a
   free prediction that can be compared with the laser's inboard Te_sep crossing (innermost laser channel R = 0.234 m).
   Imposing it instead, by adjusting the current peaking per frame, was tested and rejected on the Development board
   (section 2b).
6. **Vertical position: held at Z = 0** by a correction current in the P6 radial-field pair that cancels the net vertical
   force, as MAST's own vertical control does with P6 (median 0.57 kA, p99 12.6 kA, i.e. the size of the real P6
   currents). Why: the free solutions (zero net vertical force from the measured coils) drift slowly up and down within a
   shot (axis-height std 4.4 cm, top/bottom of the plasma 11-13 cm; not frame noise, the 5-frame running-median residual
   is ~0), and the boundary score is a worst-point distance. With the hold, usable frames rose from 98.6% to 99.7% overall
   and from 88.7% to 100% on the 63 shots with an up-down asymmetric divertor-coil setting. The Development board agreed
   in both pairs we scored: +0.0035 (v11 -> v16) and +0.013 (v21 -> v19).
7. **Choices informed by Development-phase feedback (disclosed per Rule 3):** we compared complete pipeline variants on
   the Development board using only our own submissions' aggregate and per-term scores and Detailed Results: the
   solenoid factor (0.5 vs 1.0), the inner wall / solenoid clearance fix (v11), the inboard-edge fit (v12) and its
   one-sided versions (v13-v15, rejected), the vertical hold (v16, kept), Te_sep 40 vs 30 eV (v18, 30 kept) and the
   Spitzer exponent 2.5 / 1.5 / 1.0 (v19 / v18 / v20, 1.0 kept), plus free height at x = 2.5 (v21). Each variant was a
   physics change motivated and checked label-free first, and from v16 on each scored submission changed one thing
   against an already scored one. Per-shot tables were not used for tuning. No probe submissions
   (constant/shifted/rescaled predictions) were made. Complete list of MAST pipelines scored on the Development board
   (S_MAST and its terms; DIII-D file identical in all, S_DIII-D 0.9023):

   | submission | pipeline | what it tested | S_MAST | R2psi | R2(q95,bN) | D_LCFS | Consistency |
   |---|---|---|---|---|---|---|---|
   | 984876 | v9 | kinetic profile + iterate averaging + edges read at Z = 0 | 0.6314 | 0.9430 | <0 | 0.2654 | 0.1963 |
   | 984987 | v10 | + vertical hold (bugged: the correction never acted) | 0.6258 | 0.932 | <0 | 0.251 | 0.193 |
   | 985136 | v11 | inner wall at 0.18 m (solenoid behind the armour, no clearance) | 0.6808 | 0.9447 | 0.3247 | 0.2547 | 0.1900 |
   | 985219 | v12 | parametric profile + inboard edge fitted to the laser | 0.6613 | 0.9536 | 0.1894 | 0.2177 | 0.1508 |
   | 987456 | v13 | kinetic + one-sided inboard edge | 0.6511 | 0.9365 | 0.2466 | 0.2404 | 0.1155 |
   | 987457 | v14 | parametric + one-sided inboard edge | 0.6800 | 0.9437 | 0.3366 | 0.2239 | 0.1641 |
   | 987458 | v15 | v13 + no-collapse retry | 0.6714 | 0.9445 | 0.2724 | 0.2350 | 0.1727 |
   | 992795 | v16 | v11 + vertical hold at Z = 0 (fixed) | 0.6843 | 0.9403 | 0.3551 | 0.2204 | 0.1799 |
   | 992764 | v18 | v16 + Te_sep 30 eV | 0.6924 | 0.9445 | 0.3729 | 0.2166 | 0.1934 |
   | 992763 | v19 | v18 + Spitzer exponent 2.5 | 0.5849 | 0.8373 | 0.1514 | 0.2591 | 0.1379 |
   | **992762** | **v20 (final)** | **v18 + Spitzer exponent 1.0** | **0.7106** | **0.9611** | **0.4172** | **0.2065** | **0.2004** |
   | 992761 | v21 | free height + Te_sep 30 eV + exponent 2.5 | 0.5715 | 0.8495 | 0.1006 | 0.3429 | 0.1173 |

## 2b. Tested and NOT used (negative results)
- **"Fixing" the 63 asymmetric-P2 shots.** On 5% of shots the upper divertor coil runs at about -100 kA-turn while the
  lower sits near 0; 18 of our 20 worst shots are among them. Every label-free test (isotherm mismatch, sane-solve
  fraction, vertical position, correction field) preferred the MEASURED currents over all "repaired" versions
  (P2 = 0, upper = lower, lower = upper, flipped sign). The data are right; MAST ran these shots asymmetric.
- **Force-balance beta gauge.** Idea: the correction field B_v measures the missing pressure through Shafranov's radial
  force balance (Lambda = beta_p + li/2), after removing its eddy-current part (linear in the coil and Ip ramp rates).
  On MAST the leftover is slow within a shot and anti-correlated with core Te across shots (-0.41), as unmeasured
  fast-ion pressure would be. But on DIII-D, where labels are allowed (83 shots, 3,482 frames, outboard edge from the
  true LCFS), the same solver and procedure gave a leftover almost uncorrelated with the model's true beta_p error
  (+0.07), and the correction LOWERED the beta_N R^2 at every strength (0.76 -> 0.50 at half strength; 0.67 with
  per-shot medians): coil and vessel modelling errors swamp the pressure signal. Not used.
- **More pressure for unmeasured fast ions.** At Spitzer exponent 1.0, raising the total/electron pressure ratio from
  2.42 to 3.0 / 3.6 made the label-free isotherm mismatch worse (0.047 -> 0.050 / 0.053) with the inner edge unchanged
  (12 MAST shots). The correction field B_v vanishes near 3.6, but B_v already failed as a pressure gauge on DIII-D.
  Not used, and never submitted (a pressure-only change is too close to a rescaled beta_N).
- **Fitting the inboard edge to the laser.** Closes the inboard-edge gap but makes early-shot plasmas taller (kappa
  1.69 -> 1.97) and q95 larger; on the Development board q95 R^2 fell 0.41 -> 0.13 and the total fell (v12-v15 vs v11).
- **Current diffusion: a time-dependent Spitzer exponent.** Early in a shot the induced current has not yet soaked into
  the core (skin effect), so the current should be broader than later. Label-free, the exponent that puts the model's
  inboard edge on the laser's rises with time (~0.5 or less before 150 ms, ~0.9 at 200-250 ms). A ramp x = 0.3 before
  125 ms rising to 1.0 at 227 ms ("v23"), checked on the 1,182 public-test shots not used to choose it, moved the
  inboard edge in the first 200 ms from 5-7 cm inside the laser's to within 2.5 cm, also on the frames where the laser
  really saw the edge. A ramp up to 1.5 late ("v22") put the late inboard edge 2.5-4 cm inside the laser's, so the late
  value 1.45 suggested by the first 24-shot scan did not hold up. We did NOT use the ramp: it makes early plasmas much
  taller (kappa 1.68 -> 2.18 in the first 125 ms) and raises early q95 (median +3), the same signature as the inboard-edge
  fits above that the board scored lower, and the isotherm test also got worse (0.049 -> 0.054). The Development phase
  closed before it could be scored, and we did not send an unscored change to the blind Final.
- **Te_sep 25 eV** (inside the label-free optimum 25-30 eV): built, not scored before the Development phase closed. Not used.

## 3. Label-free validation on MAST
Te is constant on a flux surface, and the MAST laser sees both sides of the axis. On the final pipeline's public-test
maps (201 shots, every 3rd frame), inboard and outboard Te agree on the same flux surface to a median 4.9% of the peak
Te and the laser's inboard 30 eV point lands at psiN = 0.983. Of all 69,075 frames, 99.6% pass the solver's sanity check; the axis
stays within 0.1 cm of Z = 0 (90th percentile). Used to compare settings without any MAST label (src/labelfree.py, src/scan_labelfree.py).

## 4. Data provenance (per machine)
- **DIII-D:** competition `diii_d_train` (300 shots for the model; 200 shots for the 40 eV and 150 shots for the 2.42
  calibrations; 60 held-out shots for local scoring), revision 1e280905. The official Ip time fix (data_fixes.fix_d3d_ip_times).
- **MAST:** competition `mast_public_test` / `mast_private_test` INPUTS only (coil currents, Ip, TF current, Thomson Te/ne,
  coil geometry, grid). **No MAST or MAST-U equilibrium data from any source** was used to train, calibrate or
  validate anything, and we never scored against MAST equilibria offline: not FAIR-MAST, not the three labelled demo
  shots (never opened), not `fusion_scoring/masks/mast_envelope.npz`. The one MAST-label-derived input to our choices is
  the competition's own feedback, which Rule 3 explicitly permits: the Development-phase scores and Detailed Results
  of our own candidate submissions, used to select between complete pipeline variants (section 2, item 7). No probes.
  The MAST domain boundary (R 0.18-1.56 m, |Z| <= 1.60 m, 7 cm clearance from every in-vessel P-coil conductor; the
  solenoid sits behind the centre-column armour and gets no clearance) comes from machine geometry.
  The TF turn count (24) is MAST engineering data.
- Thomson time arrays are cleaned with the organizers' supported rule (keep a sample only if its time is later than
  every earlier one; starter issue #12). No external datasets, no pre-trained models.

## 5. Compute
DIII-D model: ~7 s on one CPU core (plus ~1 h of data streaming). MAST physics: ~0.13 s per frame per core on an Apple M4
(69,075 public-test frames in ~17 min on 9 cores). No GPU. Compute-light.

## 6. AI-assistant disclosure (Rule 4)
The code, the physics derivations and this report were produced with an AI coding assistant (Anthropic Claude, agentic,
in Claude Code), directed and checked by the entrant: Claude wrote the code and ran the experiments; the entrant set the
goals, reviewed each result and its explanation, and takes responsibility for the submission. All predictions submitted
were produced by this pipeline.
