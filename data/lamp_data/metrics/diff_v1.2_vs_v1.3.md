# Metrics diff: v1.2 -> v1.3

mapping_version: ['1.2'] -> ['1.3']; constants_sha1: ['b7e804532df6'] -> ['a49d0111c5bb']

| flag | old | new |
|---|---|---|
| DEGENERATE | 2 | 2 |
| DYN | 26 | 17 |
| LOWCORR | 192 | 193 |
| RATE | 521 | 527 |
| SAT | 32 | 23 |
| SHORT | 154 | 154 |
| STATIC | 41 | 45 |
| TINY | 50 | 50 |

| stat | old | new |
|---|---|---|
| mean sat_max | 0.003 | 0.003 |
| mean rate_cap_frac | 0.136 | 0.157 |
| mean jerk_rms | 11553.799 | 4972.320 |
| mean light_step_per_s | 0.000 | 0.000 |
| median r_head | 0.945 | 0.918 |
| median r_lift | 0.647 | 0.619 |
| median r_yaw | 0.940 | 0.923 |

## Flags cleared (fixed) (43)

- anim_codelab_ghoulish_creeping_01: DYN
- anim_cozmosays_getin_medium_01_head_angle_20: RATE
- anim_cozmosays_getin_medium_01_head_angle_40: RATE
- anim_cozmosings_getout_01: DYN
- anim_dizzy_pickup_02: RATE
- anim_energy_cubedownlvl1_02: SAT
- anim_energy_react_stop_01: SAT
- anim_energy_requestshortlvl1_01: RATE
- anim_freeplay_reacttoface_identified_01_head_angle_40: RATE
- anim_freeplay_reacttoface_sayname_01_head_angle_20: RATE
- anim_freeplay_reacttoface_sayname_02_head_angle_40: RATE
- anim_gamesetup_reaction_01: RATE
- anim_gif_idk_01: RATE
- anim_greeting_happy_03: DYN
- anim_guarddog_getout_playersucess_01: RATE
- anim_guarddog_getout_timeout_02: RATE
- anim_hiccup_01: RATE
- anim_hiccup_02: RATE
- anim_hiking_react_03: DYN;LOWCORR
- anim_memorymatch_failhand_01: LOWCORR
- anim_memorymatch_failhand_player_02: RATE
- anim_memorymatch_failhand_player_03: DYN
- anim_memorymatch_pointsmallright_01: LOWCORR
- anim_memorymatch_pointsmallright_02: LOWCORR
- anim_memorymatch_solo_failhand_player_03: DYN
- anim_memorymatch_successhand_cozmo_01: RATE
- anim_peekaboo_requestonce_01: SAT
- anim_petdetection_cat_01_head_angle_-20: SAT
- anim_petdetection_misc_01_head_angle_20: DYN
- anim_pyramid_reacttocube_happy_mid_01: RATE
- anim_reacttoblock_react_01_head_angle_-20: SAT
- anim_reacttocliff_faceplantroll_armup_02: LOWCORR
- anim_reacttocliff_pickup_01: LOWCORR
- anim_reacttocliff_pickup_03: RATE
- anim_reacttocliff_stuckleftside_01: LOWCORR
- anim_reacttocliff_turtleroll_01: SAT
- anim_reacttocliff_wheely_01: SAT
- anim_repair_fix_wheels_back_01: DYN
- anim_repair_severe_fix_fail_01: LOWCORR
- anim_repair_severe_interruption_edge_01: SAT
- ... and 3 more

## Flags appeared (regressed) (37)

- anim_bouncer_requesttoplay_01: RATE
- anim_bouncer_rerequest_01: RATE
- anim_bouncer_wait_03: STATIC
- anim_codelab_kitchen_eating_01: RATE
- anim_cozmosays_getin_short_01_head_angle_20: RATE
- anim_cozmosays_getin_short_01_head_angle_40: RATE
- anim_dizzy_shake_loop_01: LOWCORR
- anim_dizzy_shake_stop_01: LOWCORR
- anim_explorer_huh_01_head_angle_-10: LOWCORR
- anim_explorer_huh_01_head_angle_-20: LOWCORR
- anim_guarddog_cubedisconnect_01: RATE
- anim_hiking_react_02: RATE
- anim_hiking_rtpmarker_01: RATE
- anim_keepalive_blink_01: STATIC
- anim_keepaway_getout_02: RATE
- anim_launch_wakeup_05: LOWCORR
- anim_meetcozmo_lookface_getout_head_angle_40: STATIC
- anim_meetcozmo_sayname_01_head_angle_-20: RATE
- anim_peekaboo_success_03: LOWCORR
- anim_repair_fix_idle_01: RATE
- anim_repair_fix_idle_fullyfull_03: RATE
- anim_repair_severe_fix_raiselift_01: RATE
- anim_repair_severe_reaction_01: RATE
- anim_rtpkeepaway_playeryes_01: LOWCORR
- anim_sdk_speak_01: STATIC
- anim_sparking_fail_01: RATE
- anim_sparking_idle_01: RATE
- anim_speedtap_losegame_intensity02_01: RATE
- anim_speedtap_loseround_intensity01_01: RATE
- anim_speedtap_loseround_intensity02_01: RATE
- anim_speedtap_loseround_intensity02_02: RATE
- anim_speedtap_tap_02: LOWCORR
- anim_speedtap_winhand_03: RATE
- anim_test_shiver: RATE
- anim_vc_reaction_yesfaceheardyou_01_head_angle_20: LOWCORR
- anim_workout_lowenergy_getin_01: RATE
- anim_workout_lowenergy_getready_01: RATE

