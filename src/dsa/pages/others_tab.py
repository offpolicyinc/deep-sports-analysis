"""Claude as a Tennis Coach, its Other players tab: Figure 4 (the heatmap and its clips) for the players who sent us
their footage (the others project) against the pros, and what Claude, given only those measurements, told each of
them to fix.

    python -m dsa.pages.others_tab

Writes data/projects/others/review/tab/: tab.html (the tab's figure, its data and Claude's box, ids ending "-o" so
it sits on the page beside Figure 4), coach.json (Claude's answers) and ours' clips, rendered (when missing) beside the
pros' in atp_hardcourt/review/profiles/ (the page's clips) and copied here.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import shutil
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from dsa.ingest.projects import PROJECTS
from dsa.pages import explainer
from dsa.pages.profiles_report import (KINDS, SENS, SHORT, guests, heat_z, heatmap, label_of, load, surname,
                                       tests, value)
from dsa.paths import PROJECTS as ROOT_PROJECTS

PROJECT, GUESTS = "atp_hardcourt", "others"
MEDIA = "{{MEDIA}}/claude-tennis-coach/"          # where the page finds its clips

PROMPT = """You are a tennis coach talking to one amateur player after watching their match. Below are \
measurements of their strokes (filmed on a phone from behind the baseline) next to 11 ATP pros (filmed on TV from \
behind the baseline). For each measurement: the player's typical value, the pros' typical value and range, and z, how \
far the player is from the pros (|z| above 1 is a clear difference). Lengths are in torso lengths. Rows marked \
"camera: medium" depend on where the camera stood; lean on them less. With fewer than 5 of a stroke, say the read is \
a first look.

Coach, don't diagnose. Use the numbers to decide what matters, then say it the way you would on court: plain words, \
"you", short sentences, no jargon (no z scores, no torso lengths, no degrees unless they help). Pick the one thing \
per stroke that would help most; if a stroke is already close to the pros, say so in a line. Don't guess at what the \
numbers can't show (grip, footwork between shots, tactics).

For each stroke: a headline under eight words, one or two sentences on what you see and why it matters, at most two \
short supporting points (each in plain words, one number at most), and one fix the player can feel or practise. \
Then one sentence on what to work on first.

{name}: {strokes}.

{rows}"""

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["strokes", "order"], "properties": {
    "strokes": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                                           "required": ["kind", "headline", "lead", "points", "fix"], "properties": {
                                               "kind": {"type": "string", "enum": KINDS},
                                               "headline": {"type": "string"}, "lead": {"type": "string"},
                                               "points": {"type": "array", "items": {"type": "string"}, "maxItems": 2},
                                               "fix": {"type": "string"}}}},
    "order": {"type": "string"}}}


def rows(res: dict, extra: dict, q: str, order: list[tuple[str, str]], zs: dict, pros: list[str]) -> tuple[str, str]:
    """What Claude is given about player q: his stroke counts, and one line per trait in the figure he is measured
    on: his median and middle half, the pros' median and range, z."""
    g, gres = extra[q]
    units = {(r["kind"], r["name"]): r["unit"] for r in tests(PROJECT)}
    counts = ", ".join(f"{gres[k]['strokes'][q]} {k}s" for k in KINDS if q in gres[k]["box"])
    out = []
    for k in KINDS:
        if q not in gres[k]["box"]:
            continue
        out.append(f"\n{k.upper()}")
        for kk, n in order:
            if kk != k or q not in zs[k, n]:
                continue
            u, j, gj = units.get((k, n), ""), res[k]["features"].index(n), gres[k]["features"].index(n)
            meds = [res[k]["box"][p][j][2] for p in pros if p in res[k]["box"]]
            b = gres[k]["box"][q][gj]
            out.append(f"- {SHORT.get(n, n)} ({explainer.SAY.get(n, '').strip()}): {value(b[2], u)} "
                       f"(middle half {value(b[1], u)} to {value(b[3], u)}); pros {value(float(np.median(meds)), u)} "
                       f"(lowest {value(min(meds), u)}, highest {value(max(meds), u)}); z {zs[k, n][q]:+.1f}"
                       + ("; camera: medium" if SENS.get(n) == "medium" else ""))
    return counts, "\n".join(out)


def coach(extra: dict, order: list[tuple[str, str]], cache) -> dict[str, dict]:
    """Each of ours: Claude's answer, from a blank context, to the measurements alone (dsa.opus, cached)."""
    from dsa import opus

    res, _ = load(PROJECT)
    pros = list(PROJECTS[PROJECT]["players"])
    zs = heat_z(PROJECT, res, pros, extra)

    def one(q):
        counts, rs = rows(res, extra, q, order, zs, pros)
        a, sec = opus.ask(PROMPT.format(name=label_of(q, extra), strokes=counts, rows=rs), [], SCHEMA, cache,
                          effort="high")
        print(q, f"{sec:.0f} s", flush=True)
        return q, a
    with ThreadPoolExecutor(len(extra)) as ex:
        return dict(ex.map(one, extra))


def box(extra: dict, answers: dict) -> str:
    """Claude's answers, one player at a time, laid out as the page's Figure 7 (figure.claudefig: its .seg buttons,
    one .adv row per stroke and its fix, then what to work on first)."""
    who, e = list(extra), html.escape
    key = lambda q: label_of(q, extra).lower().replace(" ", "-")            # noqa: E731
    buttons = "".join(f'<button data-who="{key(q)}" aria-pressed="{str(i == 0).lower()}">{label_of(q, extra)}</button>'
                      for i, q in enumerate(who))
    parts = []
    for i, q in enumerate(who):
        a = answers[q]
        rows = "".join(f'<div class="adv"><h4>{s["kind"].capitalize()}</h4><div><p class="adv-h">{e(s["headline"])}</p>'
                       f'<p>{e(s["lead"])}</p><ul>{"".join(f"<li>{e(p)}</li>" for p in s["points"])}</ul>'
                       f'<p class="adv-fix"><span>Fix</span>{e(s["fix"])}</p></div></div>\n' for s in a["strokes"])
        parts.append(f'<div data-for="{key(q)}"{"" if i == 0 else " hidden"}>\n{rows}'
                     f'<div class="adv"><h4>In order</h4><div><p>{e(a["order"])}</p></div></div>\n</div>\n')
    return (f'<section class="wrap"><figure class="claudefig" id="claude-o"><div class="claude"><div class="claude-head">'
            f'<span class="claude-tag">Claude\'s advice for</span><span class="seg">{buttons}</span></div>\n'
            f'{"".join(parts)}</div></figure></section>')                   # .wrap: the text's width, as Figure 7


def figure(svg: str, data: dict, names: list[str], n_ours: int) -> str:
    """Figure 4's markup again (as the page lays it out), every id ending "-o"."""
    svg = re.sub(r'<text x="0" y="[\d.]+" class="hrow" id="zdef">.*?</text><text[^>]*>[^<]*</text>', "", svg)
    pick = "".join(f'<option value="{q}">{q}</option>' for q in names)
    js = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    return (f'<figure class="heatfig" id="figure-o">\n<div class="heatwrap">\n<div class="scroll">\n{svg}</div>\n'
            f'<div class="explain">\n<div class="ex-cap"><b id="ex-title-o"></b>\n<p id="ex-say-o"></p></div>\n'
            f'<div class="ex-grid">\n<figure><video id="ex-me-o" muted playsinline preload="auto"></video>'
            f'<figcaption id="ex-mcap-o"></figcaption>\n<div class="ex-pick" id="ex-pick-o"></div></figure>\n'
            f'<figure><video id="ex-avg-o" muted playsinline preload="auto"></video><figcaption id="ex-acap-o"></figcaption>\n'
            f'<label class="ex-ref">Compare with <select id="ex-ref-o"><option value="">Average</option>{pick}</select>'
            f'</label></figure>\n<figure>\n<svg id="ex-fig-o" viewBox="0 0 320 320" role="img"></svg>'
            f'<figcaption id="ex-fcap-o"></figcaption></figure>\n<figure>\n<svg id="ex-box-o" viewBox="0 0 320 320" '
            f'role="img"></svg><figcaption id="ex-bcap-o"></figcaption></figure></div></div>\n</div>\n'
            f'<script type="application/json" id="ex-data-o">{js}</script>\n'
            f'<figcaption class="heatcap">Figure 8. Stroke fingerprints of the 11 ATP pros and {n_ours} players who '
            f'sent us their matches. Each column is a player; each row is a stroke trait. Color is a z score of the '
            f'player\'s median minus the pros\' median. Point at or click a square to see, on the left, that player\'s '
            f'typical stroke. Clicking "Compare with" swaps the average for one player.</figcaption></figure>')


def build() -> None:
    out = ROOT_PROJECTS / GUESTS / "review" / "tab"
    out.mkdir(parents=True, exist_ok=True)
    clips_dir = ROOT_PROJECTS / PROJECT / "review" / "profiles"
    pros = sorted(PROJECTS[PROJECT]["players"], key=surname)
    extra = guests(PROJECT, [GUESTS], least=1)       # everyone who sent us footage, however few strokes
    clips = explainer.videos(PROJECT, pros, clips_dir, False, extra)           # only the clips not made yet
    ours = {label_of(q, extra) for q in extra}
    for c in clips.values():                         # ours' clips beside the tab; every clip named as the page has it
        for q, ix in c["players"].items():
            for f, _ in ix if q in ours else []:
                shutil.copy2(clips_dir / f, out / f)
    page = {k: {"avg": [MEDIA + c["avg"][0], c["avg"][1]],
                "players": {q: [[MEDIA + f, lab] for f, lab in ix] for q, ix in c["players"].items()}}
            for k, c in clips.items()}
    svg, cols, order = heatmap(PROJECT, pros, extra)
    data, names = explainer.panel_data(PROJECT, cols, order, page, extra)
    answers = coach(extra, order, out / "coach")
    (out / "coach.json").write_text(json.dumps({q: answers[q] for q in extra}, indent=1))
    (out / "tab.html").write_text(figure(svg, data, names, len(extra)) + "\n" + box(extra, answers) + "\n")
    print(out / "tab.html", ", ".join(label_of(q, extra) for q in extra))


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    build()
