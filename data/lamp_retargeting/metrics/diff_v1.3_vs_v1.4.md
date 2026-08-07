# Metrics diff: v1.3 -> v1.4

mapping_version: ['1.3'] -> ['1.4']; constants_sha1: ['a49d0111c5bb'] -> ['1e4c4ad2e6e1']

| flag | old | new |
|---|---|---|
| DEGENERATE | 2 | 2 |
| DYN | 17 | 0 |
| LOWCORR | 193 | 194 |
| RATE | 527 | 527 |
| SAT | 23 | 23 |
| SHORT | 154 | 154 |
| STATIC | 45 | 45 |
| TINY | 50 | 50 |

| stat | old | new |
|---|---|---|
| mean sat_max | 0.003 | 0.003 |
| mean rate_cap_frac | 0.157 | 0.158 |
| mean jerk_rms | 4972.320 | 5059.445 |
| mean light_step_per_s | 0.000 | 0.000 |
| median r_head | 0.918 | 0.918 |
| median r_lift | 0.619 | 0.619 |
| median r_yaw | 0.923 | 0.923 |

## Flags cleared (fixed) (18)

- anim_bored_01: DYN
- anim_cozmosays_getout_long_01_head_angle_-20: DYN
- anim_energy_getin_01: DYN
- anim_energy_requestlvlone_01: DYN
- anim_keepaway_fakeout_06: DYN
- anim_keepaway_losehand_03: DYN
- anim_launch_search_head_angle_-20: DYN
- anim_launch_wakeup_05: LOWCORR
- anim_meetcozmo_celebration_02_head_angle_40: DYN
- anim_meetcozmo_reenrollment_sayname_03: DYN
- anim_memorymatch_pointcenter_fast_02: DYN
- anim_petdetection_cat_01_head_angle_-20: DYN
- anim_pounce_fail_03: DYN
- anim_reacttocliff_faceplantroll_armup_02: DYN
- anim_reacttocliff_turtleroll_05: DYN
- anim_repair_fix_getin_01: DYN
- anim_speedtap_loseround_intensity02_02: DYN
- anim_speedtap_wingame_intensity03_03: DYN

## Flags appeared (regressed) (2)

- anim_meetcozmo_sayname_02_head_angle_20: LOWCORR
- anim_memorymatch_failhand_01: LOWCORR

