# Metrics diff: v1.0-baseline -> v1.1

mapping_version: ['1.0'] -> ['1.1']; constants_sha1: ['678e3a5a74fe'] -> ['2fa8bc4f5012']

| flag | old | new |
|---|---|---|
| DEGENERATE | 2 | 2 |
| DYN | 31 | 31 |
| FLICKER | 242 | 0 |
| LOWCORR | 167 | 169 |
| RATE | 419 | 403 |
| SAT | 40 | 39 |
| SHORT | 154 | 154 |
| STATIC | 39 | 39 |
| TINY | 50 | 50 |

| stat | old | new |
|---|---|---|
| mean sat_max | 0.004 | 0.004 |
| mean rate_cap_frac | 0.095 | 0.080 |
| mean jerk_rms | 30357.706 | 23949.461 |
| mean light_step_per_s | 0.665 | 0.000 |
| median r_head | 0.963 | 0.962 |
| median r_lift | 0.668 | 0.675 |
| median r_yaw | 0.969 | 0.966 |

## Flags cleared (fixed) (253)

- anim_bored_getout_02: FLICKER
- anim_bouncer_timeout_01: FLICKER
- anim_codelab_duck_01: FLICKER
- anim_codelab_kitchen_yucky_01: FLICKER
- anim_codelab_rooster_01_head_angle_20: FLICKER
- anim_codelab_rooster_01_head_angle_40: FLICKER
- anim_cozmosays_badword_01_head_angle_-20: FLICKER
- anim_cozmosays_badword_01_head_angle_20: FLICKER
- anim_cozmosays_badword_01_head_angle_40: FLICKER
- anim_cozmosays_badword_02_head_angle_-20: FLICKER
- anim_cozmosays_badword_02_head_angle_20: FLICKER
- anim_cozmosays_badword_02_head_angle_40: FLICKER
- anim_cozmosays_getout_long_01_head_angle_-20: FLICKER
- anim_cozmosays_getout_long_01_head_angle_20: FLICKER
- anim_cozmosays_getout_long_01_head_angle_40: FLICKER
- anim_cozmosays_getout_short_01: FLICKER
- anim_cozmosays_getout_short_01_head_angle_-20: FLICKER
- anim_cozmosays_getout_short_01_head_angle_40: FLICKER
- anim_dizzy_pickup_02: FLICKER
- anim_driving_upset_end_01: FLICKER
- anim_driving_upset_loop_01: FLICKER
- anim_driving_upset_loop_02: FLICKER
- anim_driving_upset_start_01: FLICKER
- anim_energy_cubedown_01: FLICKER
- anim_energy_cubedownlvl2_03: FLICKER
- anim_energy_failgetout_01: FLICKER
- anim_energy_getout_01: FLICKER
- anim_energy_idlel1_01: FLICKER
- anim_energy_successgetout_01: FLICKER
- anim_explorer_driving01_turbo_01: FLICKER
- anim_explorer_driving01_turbo_01_head_angle_-10: FLICKER
- anim_explorer_driving01_turbo_01_head_angle_-20: FLICKER
- anim_explorer_driving01_turbo_01_head_angle_10: FLICKER
- anim_explorer_driving01_turbo_01_head_angle_20: FLICKER
- anim_explorer_driving01_turbo_01_head_angle_30: FLICKER
- anim_explorer_driving01_turbo_01_head_angle_40: FLICKER
- anim_explorer_idle_02: FLICKER
- anim_explorer_idle_02_head_angle_-10: FLICKER
- anim_explorer_idle_02_head_angle_-20: FLICKER
- anim_explorer_idle_02_head_angle_10: FLICKER
- ... and 213 more

## Flags appeared (regressed) (5)

- anim_codelab_cubetap: LOWCORR
- anim_keepaway_fakeout_06: LOWCORR
- anim_memorymatch_pointsmallleft_01: LOWCORR
- anim_memorymatch_pointsmallleft_02: LOWCORR
- anim_reacttocliff_stuckleftside_01: LOWCORR

