"""The catalog: every project and its videos. A video is named '<project>/<video>' everywhere (its folder is
dsa.paths.video_dir). Downloading them: dsa.ingest.download.
"""
from __future__ import annotations

import json
import re

from dsa.paths import ROOT

EXTRA = 2400                            # atp_hardcourt: seconds after the first window
PAD_S = 1800                            # atp_hardcourt: each later piece of a match (dsa.ingest.pad)

PROJECTS = {
    "federer_usopen": {
        "what": "Federer at the US Open, 2003-2019: every year he played (not 2016; none after 2019), about 40 "
                "minutes a year from two to four matches. Full matches: the US Open's YouTube channel where it "
                "has them, a few minutes of each from where play starts; thin years add highlights (all of "
                "them, nearly all play). All right-handed but Lopez (2007) and Verdasco (2012).",
        "who": "Roger Federer",          # step 4 (dsa.scene.names) finds him among Player 1 / 2
        # name: (YouTube id, match, second play starts (`start`; None = not looked yet), seconds kept; 0 = all)
        "videos": {
            "2003_blake": ("toVXmR3x9p0", "R3 vs Blake", 0, 1200),
            "2003_nalbandian": ("uGamKdkzb6c", "R4 vs Nalbandian", 0, 1200),
            "2004_agassi": ("-6hZMQ8rfEc", "QF vs Agassi", 60, 800),
            "2004_henman": ("YL2ZUzDynPE", "SF vs Henman", 0, 800),
            "2004_hewitt": ("nDusNa6zpGI", "Final vs Hewitt", 60, 800),
            "2005_nalbandian": ("99ak0zC1L-s", "QF vs Nalbandian", 30, 800),
            "2005_hewitt": ("dDwmhJKLLUw", "SF vs Hewitt", 0, 800),
            "2005_agassi": ("NZH2CvUL7_0", "Final vs Agassi", 0, 800),
            "2006_blake": ("BYMs5_O3q7w", "QF vs Blake", 0, 800),
            "2006_davydenko": ("LTqLwkpPiNI", "SF vs Davydenko", 30, 800),
            "2006_roddick": ("yhFXsQcnXsE", "Final vs Roddick", 120, 800),
            "2007_isner": ("5b2WtbcVsuU", "R3 vs Isner", 0, 600),
            "2007_lopez": ("ChwSRjsr_jk", "R4 vs Lopez", 60, 600),
            "2007_roddick": ("xUA01wmi4Sc", "QF vs Roddick", 0, 600),
            "2007_djokovic": ("9WGj0lA8XTQ", "Final vs Djokovic", 30, 600),
            "2008_andreev": ("Hq51hKuJ8LQ", "R4 vs Andreev, highlights", 0, 0),
            "2008_djokovic": ("d1jiZT7oJDg", "SF vs Djokovic", 0, 600),
            "2008_murray": ("4y-5TWeS-mM", "Final vs Murray", 30, 600),
            "2009_soderling": ("Ij4WWbKZtn8", "QF vs Soderling, highlights", 30, 0),
            "2009_djokovic": ("MhFjQ2Ydq8s", "SF vs Djokovic", 30, 600),
            "2009_delpotro": ("AlJFsGQpsas", "Final vs del Potro", 0, 600),
            "2010_dabul": ("sDYGLhHuHrA", "R1 vs Dabul, highlights", 0, 0),
            "2010_soderling": ("nCgAOMCQw5U", "QF vs Soderling", 0, 900),
            "2010_djokovic": ("55tIwiOMszY", "SF vs Djokovic", 30, 900),
            "2011_cilic": ("OiWaRzcPOWY", "R3 vs Cilic", 30, 800),
            "2011_tsonga": ("sBBYSbZYnl8", "QF vs Tsonga", 0, 800),
            "2011_djokovic": ("VbVfqI2cLWo", "SF vs Djokovic", 30, 800),
            "2012_young": ("1LVVlU6PVDg", "R1 vs Young, highlights", 0, 0),
            "2012_verdasco": ("bHxhiUW300Q", "R3 vs Verdasco, highlights", 30, 0),
            "2012_berdych": ("jTmACT3OsjA", "QF vs Berdych", 0, 600),
            "2013_mannarino": ("NYdkiKk1b9M", "R3 vs Mannarino, highlights", 0, 0),
            "2013_robredo": ("6JW-xcPvhlY", "R4 vs Robredo", 0, 1800),
            "2014_monfils": ("SLSa-62hgRA", "QF vs Monfils", 0, 1200),
            "2014_cilic": ("cPGmbC9BrzU", "SF vs Cilic", 0, 1200),
            "2015_isner": ("j2a06T_Cqso", "R4 vs Isner", 30, 800),
            "2015_wawrinka": ("Qa14hLnC-B4", "SF vs Wawrinka", 30, 800),
            "2015_djokovic": ("634UMLDrVzc", "Final vs Djokovic", 30, 800),
            "2017_tiafoe": ("60Kzsr6ZbRU", "R1 vs Tiafoe", 0, 800),
            "2017_youzhny": ("i8TAarlV4_Q", "R2 vs Youzhny", 30, 800),
            "2017_delpotro": ("Q5WjrTBMzMg", "QF vs del Potro", 30, 800),
            "2018_kyrgios": ("bvMGZX3-DzU", "R3 vs Kyrgios", 0, 1200),
            "2018_millman": ("9ktmxppoVog", "R4 vs Millman", 30, 1200),
            "2019_nagal": ("1zXOBeomGno", "R1 vs Nagal", 30, 800),
            "2019_goffin": ("YVSkXLGQwqc", "R4 vs Goffin", 30, 800),
            "2019_dimitrov": ("VKUWE65LVu0", "QF vs Dimitrov", 60, 800),
        },
    },
    "tests": {
        "what": "5-minute test clips: two TV matches and three of our own phone sessions (30 fps, at most 1080p).",
        "seconds": 300,
        "videos": {   # name: (recording on the external drive, start s)
            "broadcast_usopen2015_final": ("data/datasets/e2e_spot/videos/usopen_2015_mens_final_federer_djokovic.mp4", 4900),
            "broadcast_wimbledon2019_womens_final": ("data/datasets/e2e_spot/videos/wimbledon_2019_womens_final_halep_williams.mp4", 1400),
            "own_drew": ("data/raw/tennis_videos/peter_drew_202609.MOV", 3702),
            "own_dylan": ("data/raw/tennis_videos/peter_dylan_202609.MOV", 1580),
            "own_richard": ("data/raw/tennis_videos/peter_richard_202609.MOV", 1135),
        },
    },
    "own": {
        "what": "Our own phone sessions (30 fps, 1080p), Peter against a friend each time, for how his strokes compare "
                "with the atp_hardcourt players'. The phone sits lower and closer than a TV camera.",
        "who": "Peter",                  # named by hand (Opus can't know him): names.json
        "players": ["Peter", "Richard Pham"],
        "heights": {"Peter": 1.83, "Richard Pham": 1.79},   # m, as they told us (6'0", about 5'10.5")
        "compare_with": "atp_hardcourt",     # dsa.measure.profiles: boxes against its players; a tab each on its page
        "strokes": True,                 # F3ED on phone footage too (it is trained on TV: checked on a sample)
        "no_clips": ["2025_richard"],    # the app's graphics over the picture: measured, never shown as a clip
        # (video, frame) of balls fed to the hitter, seen by eye: never shown as a clip (technique's feed test only
        # catches a ball he drops and hits himself; one fed from the other end rises above him like a rally ball)
        "fed": [["2026_richard_a", 9475]],
        # name: (recording on the external drive or "yt:<id>", match, start s, seconds kept; 0 = all)
        "videos": {
            # one session in three pieces (players.json ids are per piece): Peter in pink, then blue; Richard in grey,
            # then shirtless
            "2026_richard_a": ("data/raw/tennis_videos/peter_richard_202609.MOV", "Practice vs Richard Pham", 0, 1135),
            "2026_richard": ("data/raw/tennis_videos/peter_richard_202609.MOV", "Practice vs Richard Pham", 1135, 1200),
            "2026_richard_c": ("data/raw/tennis_videos/peter_richard_202609.MOV", "Practice vs Richard Pham", 2335, 0),
            "2025_richard": ("data/raw/tennis_videos/peter_richard_20250614.mp4", "Practice vs Richard Pham", 0, 0),   # SwingVision export, points only; YouTube kbBfuwNfNj8, 60 fps
        },
    },
    "others": {
        "what": "Footage other players sent us (SwingVision and YouTube, phones behind the baseline), for how their "
                "strokes compare with the atp_hardcourt players'. Each match whole, in pieces of about 20 min (_x<k>): "
                "20 min gave 10-15 routine near-end forehands per player but only 2-5 backhands and serves. Who is who: names.json, by hand. "
                "Players by letter: who they are is in data/raw/other_players, not here.",
        "players": ["Player A", "Player B", "Player C", "Player D", "Player E", "Player F", "Player G"],
        # m, guessed from the footage (none told us yet): only for dsa.measure.profiles' phone foreshortening
        "heights": {"Player A": 1.63, "Player B": 1.80, "Player C": 1.80, "Player D": 1.55, "Player E": 1.80,
                    "Player F": 1.78, "Player G": 1.73},
        "compare_with": "atp_hardcourt",
        "strokes": True,
        # dsa.measure.technique's feed test off: filmed low from just behind the near player, the ball from the other
        # end never shows above his head (it kept 1 of 38 forehands in G's, 0 of 11 in C's YouTube one; elsewhere it
        # drops one or two)
        "phone": True,                         # dsa.measure.technique.PHONE_DEPTH
        "no_feed_check": ["2026_g", "2025_c_yt", "2025_c_yt_x2"],
        "blur": {"2026_g": [[0, 0, 460, 230]]},    # dsa.pages.exemplars: the app's scoreboard, with both players' names
        # name: (recording on the external drive, match, start s, seconds kept; 0 = all, (player, opponent)).
        # Left out: three 12-16 s screen recordings of the SwingVision app (portrait, its UI on top)
        "videos": {
            "2026_a": ("data/raw/other_players/player_a.mp4", "Player A vs Opponent A", 0, 1200, ('Player A', 'Opponent A')),
            "2026_a_x2": ("data/raw/other_players/player_a.mp4", "Player A vs Opponent A", 1200, 1456, ('Player A', 'Opponent A')),
            "2026_a_x3": ("data/raw/other_players/player_a.mp4", "Player A vs Opponent A", 2656, 0, ('Player A', 'Opponent A')),
            "2024_b": ("data/raw/other_players/player_b.mp4", "Player B vs Opponent B", 0, 1200, ('Player B', 'Opponent B')),
            "2024_b_x2": ("data/raw/other_players/player_b.mp4", "Player B vs Opponent B", 1200, 1328, ('Player B', 'Opponent B')),
            "2024_b_x3": ("data/raw/other_players/player_b.mp4", "Player B vs Opponent B", 2528, 1328, ('Player B', 'Opponent B')),
            "2024_b_x4": ("data/raw/other_players/player_b.mp4", "Player B vs Opponent B", 3856, 1328, ('Player B', 'Opponent B')),
            "2024_b_x5": ("data/raw/other_players/player_b.mp4", "Player B vs Opponent B", 5184, 0, ('Player B', 'Opponent B')),
            "2025_c_sv": ("data/raw/other_players/player_c_swingvision.mp4", "Player C vs Opponent C, college match", 0, 900, ('Player C', 'Opponent C')),
            "2025_c_sv_x2": ("data/raw/other_players/player_c_swingvision.mp4", "Player C vs Opponent C, college match", 900, 1190, ('Player C', 'Opponent C')),
            "2025_c_sv_x3": ("data/raw/other_players/player_c_swingvision.mp4", "Player C vs Opponent C, college match", 2090, 0, ('Player C', 'Opponent C')),
            "2025_c_yt": ("data/raw/other_players/player_c_youtube.mp4", "Player C vs Opponent C2, league playoff", 0, 900, ('Player C', 'Opponent C2')),
            "2025_c_yt_x2": ("data/raw/other_players/player_c_youtube.mp4", "Player C vs Opponent C2, league playoff", 900, 0, ('Player C', 'Opponent C2')),
            "2026_g": ("data/raw/other_players/player_g.mp4", "Player G vs Opponent G", 0, 0, ('Player G', 'Opponent G')),
            "2026_d": ("data/raw/other_players/player_d_match.mp4", "Player D vs Opponent D", 0, 1200, ('Player D', 'Opponent D')),
            "2026_d_x2": ("data/raw/other_players/player_d_match.mp4", "Player D vs Opponent D", 1200, 1396, ('Player D', 'Opponent D')),
            "2026_d_x3": ("data/raw/other_players/player_d_match.mp4", "Player D vs Opponent D", 2596, 1396, ('Player D', 'Opponent D')),
            "2026_d_x4": ("data/raw/other_players/player_d_match.mp4", "Player D vs Opponent D", 3992, 1396, ('Player D', 'Opponent D')),
            "2026_d_x5": ("data/raw/other_players/player_d_match.mp4", "Player D vs Opponent D", 5388, 0, ('Player D', 'Opponent D')),
            "2026_d_drill": ("data/raw/other_players/player_d_drill.mp4", "Player D, drill: a coach feeds from the far end", 0, 0, ("Player D (drill)", "coach")),   # fed balls from 3 m inside the baseline: kept apart from his match
            "2026_e": ("data/raw/other_players/player_e.mp4", "Player E vs Opponent E", 0, 0, ('Player E', 'Opponent E')),
            "2026_f": ("data/raw/other_players/player_f.mp4", "Player F vs Opponent F", 0, 0, ('Player F', 'Opponent F')),
        },
    },
    "atp_hardcourt": {
        "what": "Today's and recent ATP players at the Australian Open and US Open (hard courts, the same "
                "broadcasts), three to four matches each, for how their strokes and movement differ. Matches between "
                "two listed players count for both. Full matches: a few minutes of each from where play starts.",
        # Dimitrov dropped (too few routine backhands); his matches stay, his strokes among the others'
        "players": ["Jannik Sinner", "Carlos Alcaraz", "Novak Djokovic", "Daniil Medvedev", "Frances Tiafoe",
                    "Alexander Zverev", "Ben Shelton", "Nick Kyrgios", "Andrey Rublev", "Taylor Fritz", "Tommy Paul"],
        # m, listed heights (ATP): with "own"'s, how much a phone foreshortens the body against TV (dsa.measure.profiles)
        "heights": {"Jannik Sinner": 1.91, "Carlos Alcaraz": 1.83, "Novak Djokovic": 1.88, "Daniil Medvedev": 1.98,
                    "Frances Tiafoe": 1.88, "Alexander Zverev": 1.98, "Ben Shelton": 1.93, "Nick Kyrgios": 1.93,
                    "Andrey Rublev": 1.88, "Taylor Fritz": 1.96, "Tommy Paul": 1.85},
        # the GPU step (dsa.gpu.process): skeletons at the near end only (all the analysis uses; the far end is three
        # quarters of them) and a court check every 10 frames; checked on federer_usopen to change next to nothing
        "gpu": {"near_only": True, "court_every": 10},
        "fine": False,      # step 2b (dsa.gpu.fine): the near player's skeleton at the upload's 50 / 60 fps around each hit;
                            # off since fine_measures is (done for the first 17 matches' first hour)
        "hz": 60,           # dsa.measure.technique's time grid for those skeletons
        # but only 6 of the 17 uploads are 50 / 60 fps (all US Open): measuring those from fine.npz and the rest
        # from 25 / 30 fps would make frame rate look like style, so every match is measured at 25 / 30
        "fine_measures": False,
        # name: (YouTube id, match, second play starts (None = not looked yet), seconds kept; 0 = all,
        # (player 1, player 2)). Names start with the year and the event (ao, uso). The 2024 US Open quarterfinal
        # ended with Dimitrov retiring hurt.
        "videos": {
            "2024_ao_sinner_medvedev": ("qkNLSXDAZtQ", "Australian Open 2024 final, Sinner vs Medvedev", 30, 1200, ("Jannik Sinner", "Daniil Medvedev")),
            "2025_ao_sinner_zverev": ("9_3APf0X_-8", "Australian Open 2025 final, Sinner vs Zverev", 0, 1200, ("Jannik Sinner", "Alexander Zverev")),
            "2025_uso_sinner_alcaraz": ("TDdEVBgDqK4", "US Open 2025 final, Sinner vs Alcaraz", 30, 1200, ("Jannik Sinner", "Carlos Alcaraz")),
            "2025_ao_sinner_shelton": ("9TpapKO3bsg", "Australian Open 2025 semifinal, Sinner vs Shelton", 0, 1200, ("Jannik Sinner", "Ben Shelton")),
            "2022_uso_alcaraz_tiafoe": ("monj8eP5LQM", "US Open 2022 semifinal, Alcaraz vs Tiafoe", 30, 1200, ("Carlos Alcaraz", "Frances Tiafoe")),
            "2024_ao_zverev_alcaraz": ("7hxSj-2iq9M", "Australian Open 2024 quarterfinal, Zverev vs Alcaraz", 0, 1200, ("Alexander Zverev", "Carlos Alcaraz")),
            "2025_uso_djokovic_alcaraz": ("duiRquc8Jjc", "US Open 2025 semifinal, Djokovic vs Alcaraz", 0, 1200, ("Novak Djokovic", "Carlos Alcaraz")),
            "2023_uso_djokovic_medvedev": ("1R4qtc1H4wM", "US Open 2023 final, Djokovic vs Medvedev", 30, 1200, ("Novak Djokovic", "Daniil Medvedev")),
            "2023_uso_djokovic_shelton": ("47mFjMai8H0", "US Open 2023 semifinal, Djokovic vs Shelton", 30, 1200, ("Novak Djokovic", "Ben Shelton")),
            "2021_ao_djokovic_zverev": ("QVw6bS-GwgU", "Australian Open 2021 quarterfinal, Djokovic vs Zverev", 60, 1200, ("Novak Djokovic", "Alexander Zverev")),
            "2022_ao_medvedev_kyrgios": ("hg7nlXw2IHI", "Australian Open 2022 second round, Medvedev vs Kyrgios", 0, 1200, ("Daniil Medvedev", "Nick Kyrgios")),
            "2022_uso_kyrgios_medvedev": ("XdKQEuP0r5o", "US Open 2022 fourth round, Kyrgios vs Medvedev", 30, 1200, ("Nick Kyrgios", "Daniil Medvedev")),
            "2023_uso_tiafoe_shelton": ("qwg_pIJlNtI", "US Open 2023 quarterfinal, Tiafoe vs Shelton", 60, 1200, ("Frances Tiafoe", "Ben Shelton")),
            "2024_uso_tiafoe_dimitrov": ("fBBnCKcOMK4", "US Open 2024 quarterfinal, Tiafoe vs Dimitrov", 30, 1200, ("Frances Tiafoe", "Grigor Dimitrov")),
            "2019_uso_medvedev_dimitrov": ("mLGcEqse4H4", "US Open 2019 semifinal, Medvedev vs Dimitrov", 0, 1200, ("Daniil Medvedev", "Grigor Dimitrov")),
            "2022_uso_kyrgios_khachanov": ("mvbdH85TLV4", "US Open 2022 quarterfinal, Kyrgios vs Khachanov", 30, 1200, ("Nick Kyrgios", "Karen Khachanov")),
            "2021_ao_kyrgios_thiem": ("VqvBtbBf5qE", "Australian Open 2021 third round, Kyrgios vs Thiem", 60, 1200, ("Nick Kyrgios", "Dominic Thiem")),
        },
    },
    "quiz": {
        "what": "The guess-the-players game (dsa.pages.quiz): ten levels, three players each and one match per pair "
                "of them. Full matches: their first 30 minutes (uploads open with walk-ons and warm-ups, so 15 "
                "minutes held only 2-8 minutes of play); highlights and condensed matches: all of it.",
        "seconds": 1800,
        "min_depth": {"l9_draper_giron": 0.3, "l10_giron_tien": 0.3},   # outer courts: lower main camera (dsa.scene.court)
        # name: (YouTube id, match, level, (player 1, player 2), seconds kept; 0 = all). The name's l<n> is the
        # level first planned; the three 1995-2004 broadcasts are out of the game (level None): their court
        # and tracking are too poor, and they have no scoreboard serve marker to name the players by.
        "videos": {
            "l1_federer_nadal": ("KTCDxjJvs2U", "Australian Open 2017 final", 1, ("Roger Federer", "Rafael Nadal"), 1800),
            "l1_djokovic_federer": ("TUikJi0Qhhw", "Wimbledon 2019 final", 1, ("Novak Djokovic", "Roger Federer"), 1800),
            "l1_djokovic_nadal": ("qg4wqo02gfc", "US Open 2013 final", 1, ("Novak Djokovic", "Rafael Nadal"), 1800),
            "l2_alcaraz_sinner": ("ckbX699wngs", "Roland-Garros 2025 final", 2, ("Carlos Alcaraz", "Jannik Sinner"), 1800),
            "l2_alcaraz_zverev": ("gkhOwQ3eM88", "Australian Open 2026 semifinal", 2, ("Carlos Alcaraz", "Alexander Zverev"), 1800),
            "l2_sinner_zverev": ("9_3APf0X_-8", "Australian Open 2025 final", 2, ("Jannik Sinner", "Alexander Zverev"), 1800),
            "l3_agassi_sampras": ("wTNvz2F682k", "US Open 1995 final", None, ("Andre Agassi", "Pete Sampras"), 1800),
            "l3_roddick_sampras": ("nq94E9eDia0", "US Open 2002 quarterfinal", None, ("Andy Roddick", "Pete Sampras"), 1800),
            "l3_agassi_roddick": ("86Db0SUSXU0", "Cincinnati 2004 semifinal", None, ("Andre Agassi", "Andy Roddick"), 1800),
            "l4_medvedev_tsitsipas": ("pFIP6kY8yoA", "Australian Open 2022 semifinal", 2, ("Daniil Medvedev", "Stefanos Tsitsipas"), 1800),
            "l4_kyrgios_medvedev": ("XdKQEuP0r5o", "US Open 2022 round 4", 3, ("Nick Kyrgios", "Daniil Medvedev"), 1800),
            "l4_kyrgios_tsitsipas": ("6dzIBWFGYTQ", "Wimbledon 2022 round 3, highlights", 3, ("Nick Kyrgios", "Stefanos Tsitsipas"), 0),
            "l5_fritz_shelton": ("0EQ52HULPbw", "Indian Wells 2023, highlights", 4, ("Taylor Fritz", "Ben Shelton"), 0),
            "l5_fritz_tiafoe": ("4AV40TOmfe0", "US Open 2024 semifinal", 4, ("Taylor Fritz", "Frances Tiafoe"), 1800),
            "l5_shelton_tiafoe": ("8MOWcvOKwlA", "US Open 2024 round 3", 4, ("Ben Shelton", "Frances Tiafoe"), 1800),
            "l6_dimitrov_rublev": ("IDRsM8sDdTc", "US Open 2024 round 4", 5, ("Grigor Dimitrov", "Andrey Rublev"), 1800),
            "l6_deminaur_dimitrov": ("4USAx07Ypq4", "US Open 2019 round 4, highlights", 5, ("Alex de Minaur", "Grigor Dimitrov"), 0),
            "l6_deminaur_rublev": ("oWi78iQGPi8", "Australian Open 2024 round 4", 5, ("Alex de Minaur", "Andrey Rublev"), 1800),
            "l7_berrettini_rune": ("uJyijHso1_Y", "Australian Open 2025 round 2", 6, ("Matteo Berrettini", "Holger Rune"), 1800),
            "l7_berrettini_ruud": ("G7T5Z3WwE1M", "US Open 2022 quarterfinal", 6, ("Matteo Berrettini", "Casper Ruud"), 1800),
            "l7_rune_ruud": ("tJ2_uu8HkS8", "Red Bull Bassline exhibition, Australian Open 2025", 6, ("Holger Rune", "Casper Ruud"), 0),
            "l8_korda_paul": ("tQd0KIzHFPE", "US Open 2022 round 2", 7, ("Sebastian Korda", "Tommy Paul"), 1800),
            "l8_hurkacz_korda": ("zVIG57uwLEE", "Australian Open 2023 round 4", 7, ("Hubert Hurkacz", "Sebastian Korda"), 1800),
            "l8_hurkacz_mensik": ("Pcn-Su3IU5I", "Australian Open 2024 round 2, highlights", 8, ("Hubert Hurkacz", "Jakub Mensik"), 0),
            "l9_fonseca_mensik": ("beh_Kwo6UoI", "Next Gen ATP Finals 2024, highlights", 9, ("Joao Fonseca", "Jakub Mensik"), 0),
            "l9_draper_giron": ("Gb5blpgTiqA", "Australian Open 2024 round 1, condensed", 9, ("Jack Draper", "Marcos Giron"), 0),
            "l9_augeraliassime_lehecka": ("kCAJdZRzEsM", "Australian Open 2023 round 4, condensed", 8, ("Felix Auger-Aliassime", "Jiri Lehecka"), 0),
            "l10_michelsen_nakashima": ("9C0u0R3Zj2w", "US Open 2026 round 2, condensed", 10, ("Alex Michelsen", "Brandon Nakashima"), 0),
            "l10_giron_tien": ("4NrbPy_MKrE", "Australian Open 2026 round 1", 10, ("Marcos Giron", "Learner Tien"), 1800),
            "l10_brooksby_fonseca": ("V5TzhUetgm4", "Wimbledon 2025 round 1, highlights", 9, ("Jenson Brooksby", "Joao Fonseca"), 0),
        },
    },
}

# atp_hardcourt: minutes 20-60 of each match as its own video "<name>_b" (the first 20 are done; nothing is redone)
_atp = PROJECTS["atp_hardcourt"]["videos"]
for _v, (_yt, _m, _at, _s, _p) in list(_atp.items()):
    _atp[f"{_v}_b"] = (_yt, _m, None if _at is None else _at + _s, EXTRA, _p)

# atp_hardcourt, more footage, a piece at a time (dsa.ingest.pad): more matches (their first hour) and later
# pieces of any match, PAD_S each ("<match>_x<k>"). Only the pieces listed in PAD are part of the project.
MORE = {
    # in order: matches between two listed players who are short of strokes (both players' strokes count); one
    # against someone not listed (Korda, de Minaur, Dimitrov) gives half as much for the same price: spares
    "2024_uso_fritz_tiafoe": ("4AV40TOmfe0", "US Open 2024 semifinal, Fritz vs Tiafoe", ("Taylor Fritz", "Frances Tiafoe")),   # SSD (raw/quiz)
    "2022_uso_rublev_tiafoe": ("OsQzOCVrSzk", "US Open 2022 quarterfinal, Rublev vs Tiafoe", ("Andrey Rublev", "Frances Tiafoe")),
    "2023_ao_rublev_djokovic": ("GRVEHRyNNB8", "Australian Open 2023 quarterfinal, Rublev vs Djokovic", ("Andrey Rublev", "Novak Djokovic")),
    "2021_uso_tiafoe_rublev": ("ssrxfLAB_Y8", "US Open 2021 third round, Tiafoe vs Rublev", ("Frances Tiafoe", "Andrey Rublev")),
    "2024_uso_fritz_zverev": ("FgcirxnayKs", "US Open 2024 quarterfinal, Fritz vs Zverev", ("Taylor Fritz", "Alexander Zverev")),
    "2024_ao_djokovic_fritz": ("ZarYCPgfCQA", "Australian Open 2024 quarterfinal, Djokovic vs Fritz", ("Novak Djokovic", "Taylor Fritz")),
    "2023_ao_shelton_paul": ("KMflCz1WDpA", "Australian Open 2023 quarterfinal, Shelton vs Paul", ("Ben Shelton", "Tommy Paul")),
    "2023_ao_djokovic_paul": ("7dSqbzWPxek", "Australian Open 2023 semifinal, Djokovic vs Paul", ("Novak Djokovic", "Tommy Paul")),
    "2025_ao_paul_zverev": ("udi_GsVOefY", "Australian Open 2025 quarterfinal, Paul vs Zverev", ("Tommy Paul", "Alexander Zverev")),
    "2023_uso_paul_shelton": ("fy2W6WHf5eQ", "US Open 2023 fourth round, Paul vs Shelton", ("Tommy Paul", "Ben Shelton")),
    "2024_uso_shelton_tiafoe": ("8MOWcvOKwlA", "US Open 2024 third round, Shelton vs Tiafoe", ("Ben Shelton", "Frances Tiafoe")),   # SSD
    "2026_uso_zverev_shelton": ("r6iz35hizA8", "US Open 2026 final, Zverev vs Shelton", ("Alexander Zverev", "Ben Shelton")),
    # spares (dsa.ingest.pad: only once no later piece of a match in helps)
    "2023_uso_djokovic_fritz": ("B9iBZKUAxD8", "US Open 2023 quarterfinal, Djokovic vs Fritz", ("Novak Djokovic", "Taylor Fritz")),
    "2022_ao_fritz_tiafoe": ("i9onbTfuNwU", "Australian Open 2022 second round, Fritz vs Tiafoe", ("Taylor Fritz", "Frances Tiafoe")),
    "2024_uso_dimitrov_rublev": ("IDRsM8sDdTc", "US Open 2024 fourth round, Dimitrov vs Rublev", ("Grigor Dimitrov", "Andrey Rublev")),   # SSD
    "2024_ao_deminaur_rublev": ("oWi78iQGPi8", "Australian Open 2024 fourth round, de Minaur vs Rublev", ("Alex de Minaur", "Andrey Rublev")),   # SSD
    "2022_uso_korda_paul": ("tQd0KIzHFPE", "US Open 2022 second round, Korda vs Paul", ("Sebastian Korda", "Tommy Paul")),   # SSD
}
PAD = ROOT / "projects/atp_hardcourt/pad.json"           # the pieces added so far, in order


def started(yt: str) -> int | None:
    """Where play starts in an upload, once dsa.ingest.download start has looked (raw/starts/<id>/start.json)."""
    f = ROOT / "raw/starts" / yt / "start.json"
    return json.loads(f.read_text())["start"] if f.exists() else None


def piece(name: str) -> tuple | None:
    """A MORE match's first hour ("<match>") or a later piece of any match ("<match>_x<k>": PAD_S from the end of
    what came before), as a video entry; a later piece is None while its match's start is unknown."""
    base, _, k = name.partition("_x")
    if base in MORE:
        yt, m, p = MORE[base]
        at, first = started(yt), 3600
    else:
        yt, m, at, _, p = _atp[base]
        first = 3600                                           # minutes 0-20 and 20-60
    if not k:                       # a start not known yet is found by dsa.ingest.download get, once downloaded
        return (yt, m, at, first, p)
    return None if at is None else (yt, m, at + first + (int(k) - 1) * PAD_S, PAD_S, p)


for _v in (json.loads(PAD.read_text()) if PAD.exists() else []):
    if (_e := piece(_v)) is not None:
        _atp[_v] = _e


def match(video: str) -> str:
    """The match a video is part of (a "_b" video and the later pieces "_x<k>" (dsa.ingest.pad) continue it)."""
    return re.sub(r"_x\d+$", "", video).removesuffix("_b")


def pair(spec: str) -> tuple[str, str]:
    """The video's two players: first the one names.json's "who" numbers, then the other. With a project player
    ("who"): them and the opponent from the match title ("R3 vs Blake"); otherwise the two listed for the video."""
    project, video = spec.split("/")
    p, info = PROJECTS[project], PROJECTS[project]["videos"][video]
    if "who" in p:
        return p["who"], info[1].split(" vs ")[1].split(",")[0]
    return next(x for x in info if isinstance(x, tuple))
