"""Steps 3-4 on made-up data: court, tracking, players, hits, names, cleaning, points."""
import numpy as np

from dsa.scene import court as C
from dsa.scene import events, identity, tracks


def _court_image() -> np.ndarray:
    """The reference court seen by a camera behind the near baseline: 1280 x 720 image points."""
    import cv2

    src = np.float32([C.REF[0], C.REF[1], C.REF[2], C.REF[3]])
    dst = np.float32([[440, 200], [840, 200], [200, 650], [1080, 650]])
    m = cv2.getPerspectiveTransform(src, dst)
    return cv2.perspectiveTransform(np.float32(C.REF)[:, None], m).reshape(14, 2)


def test_court_roundtrip_and_area():
    import cv2

    pts = _court_image()
    assert C.court_ok(pts, (1280, 720))
    assert not C.court_ok(pts[[1, 0, 3, 2] + list(range(4, 14))], (1280, 720))       # left / right flipped
    m = C.homography(pts)
    back = cv2.perspectiveTransform(pts[:, None], m).reshape(14, 2)
    assert np.allclose(back, C.REF, atol=0.5)
    inside = np.array([[C.MID, C.NEAR + 2 * C.METRE], [C.MID, C.FAR - 7 * C.METRE]])
    outside = np.array([[C.MID, C.NEAR + 9 * C.METRE], [C.LEFT - 3 * C.METRE, C.NET]])  # far back; umpire's chair
    assert C.in_area(inside).all() and not C.in_area(outside).any()


def test_main_camera_only():
    pts = _court_image()                                         # far baseline 400 px, near 880 px wide
    assert C.main_camera(pts, 1280)
    low = pts.copy()
    low[[2, 3], 0] = [-200, 1480]                                # near baseline 4 x the far one: low camera
    assert not C.main_camera(low, 1280)
    tilted = pts.copy()
    tilted[1, 1] += 80                                           # far baseline sloping: side camera
    assert not C.main_camera(tilted, 1280)
    assert not C.main_camera(pts + [400, 0], 1280)               # court off to one side


def test_tracking_restarts_at_cuts():
    boxes = [np.array([[100, 100, 150, 250, 0.9]], np.float32) for _ in range(40)]
    ids = tracks.track(boxes, 30, breaks=[20])
    assert len({int(i[0]) for i in ids[5:20]}) == 1 and len({int(i[0]) for i in ids[25:]}) == 1
    assert int(ids[10][0]) != int(ids[30][0])


def test_tv_court_holds_then_drops():
    n = C.COURT_HOLD + 10
    checks = np.full((n, 14, 2), np.nan, np.float32)
    checks[0] = _court_image()
    court, _ = C.tv(checks, n)
    assert court[C.COURT_HOLD] is not None and court[C.COURT_HOLD + 1] is None


def test_stitch_joins_a_hidden_person():
    fps = 30
    boxes = [np.array([[100 + i, 100, 150 + i, 250, 0.9]], np.float32) if i < 30 or i >= 40
             else np.zeros((0, 5), np.float32) for i in range(70)]
    ids = [np.array([1 if i < 30 else 2]) if len(b) else np.zeros(0, int) for i, b in enumerate(boxes)]
    pp = tracks.people(boxes, ids, fps)
    assert pp["person_of"][1] == pp["person_of"][2]


def test_still_people_are_marked():
    fps = 30
    boxes = [np.array([[10, 10, 50, 110, 0.9], [300 + 5 * i, 100, 350 + 5 * i, 200, 0.9]], np.float32)
             for i in range(150)]
    ids = [np.array([1, 2]) for _ in boxes]
    ppl = tracks.people(boxes, ids, fps)["people"]
    assert ppl[1]["still"] and not ppl[2]["still"]


def test_contact_goes_to_the_nearest_wrist():
    n = 40
    ball = np.zeros((n, 4), np.float32)
    for t in range(n):                                   # in along +x, turns back at frame 20 by x = 500
        ball[t] = (1, 500 - 10 * abs(t - 20), 300, 1)
    boxes = [np.array([[450, 250, 520, 450, 0.9], [900, 250, 970, 450, 0.9]], np.float32) for _ in range(n)]
    ids = [np.array([1, 2]) for _ in range(n)]
    assert 20 in events.turns(ball)
    assert events.contacts(ball, boxes, ids, {1: 1, 2: 2}) == [(20, 1)]


def test_identity_splits_by_colour():
    import cv2

    h, w = 400, 600
    kp = np.full((65, 3), np.nan, np.float32)
    body = {5: (100, 100), 6: (200, 100), 11: (110, 250), 12: (190, 250), 13: (115, 350), 14: (185, 350)}

    def person(dx):
        k = kp.copy()
        for i, (x, y) in body.items():
            k[i] = (x + dx, y, 0.9)
        return k

    red, blue = np.zeros((h, w, 3), np.uint8), np.zeros((h, w, 3), np.uint8)
    for img, (a, b) in ((red, ((0, 0, 220), (220, 0, 0))), (blue, ((220, 0, 0), (0, 0, 220)))):
        img[:, :300], img[:, 300:] = a, b                  # left person one colour, right person the other
    # frames 0-9: pieces 1 (left, red) and 2 (right, blue) together; frames 10-19: piece 3 alone, on the right,
    # in red (piece 1's colour, after a change of end)
    frames = [red] * 10 + [cv2.flip(red, 1)] * 10
    picks = [[1, 2]] * 10 + [[None, 3]] * 10
    per_frame = [np.zeros((2, 5), np.float32)] * 10 + [np.zeros((1, 5), np.float32)] * 10
    ids = [np.array([1, 2])] * 10 + [np.array([3])] * 10
    kps = [np.stack([person(0), person(300)])] * 10 + [np.stack([person(300)])] * 10
    r = identity.split(picks, per_frame, ids, {1: 1, 2: 2, 3: 3}, kps, iter(frames))
    assert r["player"][1] != r["player"][2] and r["player"][3] == r["player"][1]


def test_identity_returns_each_players_look():
    import cv2

    h, w = 400, 600
    kp = np.full((65, 3), np.nan, np.float32)
    body = {5: (100, 100), 6: (200, 100), 11: (110, 250), 12: (190, 250), 13: (115, 350), 14: (185, 350)}
    k = [kp.copy(), kp.copy()]
    for i, (x, y) in body.items():
        k[0][i], k[1][i] = (x, y, 0.9), (x + 300, y, 0.9)
    img = np.zeros((h, w, 3), np.uint8)
    img[:, :300], img[:, 300:] = (0, 0, 220), (220, 0, 0)
    r = identity.split([[1, 2]] * 5, [np.zeros((2, 5), np.float32)] * 5, [np.array([1, 2])] * 5, {1: 1, 2: 2},
                       [np.stack(k)] * 5, iter([img] * 5))
    red = identity.describe(img, k[0])
    assert identity.chi2(r["centres"][r["player"][1]], red) < 1e-9


def test_line_up_renumbers_later_pieces_to_match_the_first(tmp_path, monkeypatch):
    import json

    red, blue = np.eye(4)[0], np.eye(4)[1]                  # two players' looks
    def piece(name, looks, p1_near):
        (tmp_path / name).mkdir()
        ends = [1, 2] if p1_near else [2, 1]
        (tmp_path / name / "players.json").write_text(json.dumps({
            "fps": 30.0, "near_far": [ends, [ends[0], None]], "boxes": [[None, None]] * 2,
            "contacts": [[0, 1], [1, None]], "looks": None if looks is None else [x.tolist() for x in looks]}))
    monkeypatch.setattr(identity, "video_dir", lambda spec: tmp_path / spec.split("/")[1])
    piece("a", [red, blue], True)
    piece("b", [red, blue], False)          # numbered as the first; ends changed
    piece("c", [blue, red], True)           # numbered the other way
    piece("d", None, True)                  # one player only: no looks
    specs = [f"p/{v}" for v in "abcd"]
    assert identity.line_up(specs) == ["p/c"]
    c = json.loads((tmp_path / "c" / "players.json").read_text())
    assert c["near_far"] == [[2, 1], [2, None]] and c["contacts"] == [[0, 2], [1, None]]
    assert c["looks"] == [red.tolist(), blue.tolist()]
    assert identity.line_up(specs) == []    # a rerun leaves it


def test_names_crops_prefer_the_near_end():
    from dsa.scene.names import CROPS, moments

    b = [0, 0, 10, 20]
    # player 1 near in frames 0-9 then far; player 2 only ever far
    players = {"near_far": [[1, 2]] * 10 + [[None, 1]] * 10,
               "boxes": [[b, b]] * 10 + [[None, b]] * 10}
    one = moments(players, 1)
    assert len(one) == CROPS and all(i < 10 for i, _ in one)
    assert len(moments(players, 2)) == CROPS
    # a moment at the near end is a mix-up, not where player 2 plays
    players["near_far"][0] = [2, 1]
    assert all(i > 0 for i, _ in moments(players, 2))


def test_far_player_beats_line_judges_behind_and_beside_the_court():
    from dsa.scene.court import FAR, METRE
    from dsa.scene.players import SINGLES, off_court

    # a player 1 m and one 5.5 m behind the far baseline; a line judge 7.8 m behind it, one sitting 3.6 m
    # beside the singles sideline; someone on the court
    xy = np.array([[832, FAR - 1 * METRE], [832, FAR - 5.5 * METRE], [500, FAR - 7.8 * METRE],
                   [SINGLES[1] + 3.6 * METRE, FAR + 3 * METRE], [832, 1000]], np.float32)
    near, deep, back, beside, on = off_court(xy)
    assert near < back and deep < beside and on == 0


def test_far_player_is_the_other_one():
    import cv2

    kp = np.full((65, 3), np.nan, np.float32)
    body = {5: (100, 100), 6: (200, 100), 11: (110, 250), 12: (190, 250), 13: (115, 350), 14: (185, 350)}

    def person(dx):
        k = kp.copy()
        for i, (x, y) in body.items():
            k[i] = (x + dx, y, 0.9)
        return k

    red = np.zeros((400, 600, 3), np.uint8)
    red[:, :300], red[:, 300:] = (0, 0, 220), (220, 0, 0)
    # near: piece 1 (left, red) then piece 2 (left, blue); far: piece 3 throughout, though its player changes
    frames = [red] * 10 + [cv2.flip(red, 1)] * 10
    picks = [[1, 3]] * 10 + [[2, 3]] * 10
    per_frame = [np.zeros((2, 5), np.float32)] * 20
    ids = [np.array([1, 3])] * 10 + [np.array([2, 3])] * 10
    kps = [np.stack([person(0), person(300)])] * 20
    r = identity.split(picks, per_frame, ids, {1: 1, 2: 2, 3: 3}, kps, iter(frames), tv=True)
    a, b = r["frames"][0], r["frames"][15]
    assert a[0] != b[0] and a[1] == 1 - a[0] and b[1] == 1 - b[0]


def test_clean_drops_a_ball_spike_and_fills_short_gaps_not_across_cuts():
    from dsa.scene import clean

    b = np.zeros((12, 4), np.float32)
    for i in (0, 1, 2, 4, 5, 6, 9, 10, 11):
        b[i] = (1, 10 * i, 100, 1)
    b[5, 1:3] = (900, 600)                          # a false detection far from both neighbours
    c = clean.ball(b, breaks=[8])
    assert c[5, 0] == 1 and abs(c[5, 1] - 50) < 1   # dropped, then filled from 4 and 6
    assert c[3, 0] == 1 and abs(c[3, 1] - 30) < 1   # one-frame gap filled
    assert c[7, 0] == 0 and c[8, 0] == 0            # not filled across the cut at 8


def test_clean_drops_a_few_frames_of_ball_elsewhere():
    from dsa.scene import clean

    b = np.zeros((12, 4), np.float32)
    for i in range(12):
        b[i] = (1, 400 + 12 * i, 300, 1)
    b[5:8, 1:3] = (670, 560)                        # three frames of a ball lying on the court
    c = clean.ball(b)
    assert all(abs(c[i, 1] - (400 + 12 * i)) < 1 for i in range(12))


def test_clean_undoes_short_left_right_flips_of_a_whole_side_only():
    from dsa.scene import clean

    n = 60
    kp = np.full((n, 17, 3), 1.0)
    for j in range(17):                             # left joints at x 100, right at x 140, walking right slowly
        kp[:, j, 0] = (100 if j % 2 else 140) + np.arange(n)
        kp[:, j, 1] = 10 * j
    flip = kp.copy()
    legs = [11, 13, 15]
    flip[10:14, legs + [12, 14, 16]] = kp[10:14, [12, 14, 16] + legs]    # the model flips the legs for 4 frames
    flip[30:, legs + [12, 14, 16]] = kp[30:, [12, 14, 16] + legs]        # and for good from frame 30
    c = clean.joints(flip, np.full(n, 300.0))
    assert (c[10:14, 11, 0] < c[10:14, 12, 0]).all()                     # short flip undone, legs together
    assert (c[10:14, 13, 0] < c[10:14, 14, 0]).all()
    assert (c[35:, 11, 0] > c[35:, 12, 0]).all()                         # a long flip is the model's call
    assert (c[:, 5, 0] < c[:, 6, 0]).all()                               # arms untouched


def test_points_end_at_serves_cuts_and_pauses():
    from dsa.scene.results import points

    h = lambda f, shot="stroke": {"frame": f, "shot": shot}                                   # noqa: E731
    hits = ([h(100, "serve")] + [h(100 + 30 * k) for k in range(1, 7)]          # 7 hits
            + [h(400, "serve"), h(430)]                                          # next serve: 2 hits
            + [h(1000, "serve"), h(1030), h(1300)])                              # a 4 s pause after 2
    got = points(hits, [(0, 1100), (1100, 2000)], fps=50)
    assert [p[0] for p in got] == [7, 2, 2]
    assert got[0][1:] == (25, 380) and got[2][2] == 1100                          # 1.5 s before, 2 s after, in its piece


def _look(shirt_l, shorts_l, light=0.0):
    """A look: grey head and shins, the shirt and shorts at the given lightness, all `light` lighter."""
    from dsa.scene.who import PARTS, STATS, VARIANTS

    look = np.full((len(PARTS), STATS), 128.0)
    for p, l in (("torso", shirt_l), ("shorts", shorts_l)):
        look[PARTS.index(p)] = [l, 128, 128, l - 10, l + 10]
    look[:, [0, 3, 4]] += light
    return np.tile(look, (len(VARIANTS), 1))


def test_who_features_hold_in_sun_and_shade():
    from dsa.scene.who import FEATURES, features

    sun, shade = features(_look(200, 40, 0)), features(_look(200, 40, -60))
    i = FEATURES.index("shorts L")
    assert sun[i] == shade[i] == -160                    # black shorts under a white shirt, whatever the light


def test_who_fit_follows_the_scoreboard_not_the_light():
    from dsa.scene.who import fit

    rng = np.random.default_rng(0)
    units = []
    for j in range(40):                          # black shorts (player 0) or white (player 1), in sun or shade
        k, light = j % 2, (0 if j < 20 else -70)
        look = _look(190 + rng.normal(0, 3), (40 if k == 0 else 190) + rng.normal(0, 3), light)
        units.append((("v", j, 0, 89), 90, look, ("v", 0)))
    anchors = {0: 0, 1: 1, 22: 0, 23: 1}                 # four serves read off the scoreboard
    lab, w, d = fit(units, anchors)
    assert all(k in (None, j % 2) for j, k in enumerate(lab)) and lab.count(None) <= 2
