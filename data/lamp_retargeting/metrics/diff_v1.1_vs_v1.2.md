# Metrics diff: v1.1 -> v1.2

mapping_version: ['1.1'] -> ['1.2']; constants_sha1: ['2fa8bc4f5012'] -> ['b7e804532df6']

| flag | old | new |
|---|---|---|
| DEGENERATE | 2 | 2 |
| DYN | 31 | 26 |
| LOWCORR | 169 | 192 |
| RATE | 403 | 521 |
| SAT | 39 | 32 |
| SHORT | 154 | 154 |
| STATIC | 39 | 41 |
| TINY | 50 | 50 |

| stat | old | new |
|---|---|---|
| mean sat_max | 0.004 | 0.003 |
| mean rate_cap_frac | 0.080 | 0.136 |
| mean jerk_rms | 23949.461 | 11553.799 |
| mean light_step_per_s | 0.000 | 0.000 |
| median r_head | 0.962 | 0.945 |
| median r_lift | 0.675 | 0.647 |
| median r_yaw | 0.966 | 0.940 |

## Flags cleared (fixed) (16)

- anim_cozmosays_app_getout_01: DYN
- anim_cozmosays_getout_long_01_head_angle_-20: SAT
- anim_energy_eat_02: SAT
- anim_explorer_driving01_turbo_01: RATE
- anim_freeplay_reacttoface_identified_03_head_angle_20: DYN
- anim_gamesetup_getout_01: DYN
- anim_guarddog_fakeout_01: RATE
- anim_memorymatch_pointsmallleft_01: LOWCORR
- anim_memorymatch_pointsmallleft_02: LOWCORR
- anim_play_idle_03: DYN
- anim_reacttoblock_react_short_01_head_angle_20: DYN
- anim_reacttocliff_turtleroll_02: SAT
- anim_reacttocliff_turtleroll_06: SAT
- anim_reacttocliff_wheely_02: SAT
- anim_repair_severe_idle_03: SAT
- anim_vc_reaction_yesfaceheardyou_01_head_angle_-20: SAT

## Flags appeared (regressed) (144)

- anim_bored_02: RATE
- anim_bored_event_01: RATE
- anim_bored_event_02: RATE
- anim_bored_event_03: RATE
- anim_bouncer_ideatoplay_01: RATE
- anim_bouncer_intoscore_01: RATE
- anim_codelab_bored_01: RATE
- anim_codelab_magicfortuneteller_inquistive: RATE
- anim_codelab_rooster_01_head_angle_20: LOWCORR
- anim_codelab_rooster_01_head_angle_40: LOWCORR
- anim_codelab_staring_loop: STATIC
- anim_codelab_tinyorchestra_conducting: RATE
- anim_cozmosays_getin_long_01_head_angle_40: RATE
- anim_cozmosays_getin_medium_01_head_angle_20: RATE
- anim_cozmosings_80_song_01: LOWCORR
- anim_cozmosings_getin_02: RATE
- anim_cozmosings_getin_03: RATE
- anim_cozmosings_getout_01: RATE
- anim_dizzy_pickup_02: RATE
- anim_dizzy_pickup_03: RATE
- anim_dizzy_reaction_hard_01: RATE
- anim_dizzy_reaction_medium_01: RATE
- anim_dizzy_reaction_soft_03: RATE
- anim_energy_cubedownlvl1_03: RATE
- anim_energy_cubedownlvl2_02: RATE
- anim_energy_cubedownlvl2_03: LOWCORR
- anim_energy_cubeshake_02: RATE
- anim_energy_cubeshake_lv2_03: RATE
- anim_energy_eat_01: RATE
- anim_energy_eat_03: RATE
- anim_energy_eat_04: RATE
- anim_energy_eat_lvl2_05: RATE
- anim_energy_getout_01: RATE
- anim_energy_idlel2_search_01: RATE
- anim_energy_react_stop_01: RATE
- anim_energy_reacttocliff_lv2_01: RATE
- anim_energy_requestlvlone_01: RATE
- anim_energy_requestlvltwo_01: RATE
- anim_energy_shortreact_lvl2_01: RATE
- anim_explorer_driving01_turbo_01_head_angle_40: RATE
- ... and 104 more

