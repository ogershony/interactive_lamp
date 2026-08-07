#!/usr/bin/env python3
"""
Optional human-curation tooling for retarget runs: GIF rendering, a local
web review app, a mapping A/B panel, batch verdicts, and contact sheets.

    uv run data/scripts/curate.py render  [--run R] [--clips A B] [--force]
    uv run data/scripts/curate.py serve   [--run R] [--port 7788]
    uv run data/scripts/curate.py panel   [--run R] [--tag NAME]
    uv run data/scripts/curate.py verdict --flag STATIC --verdict drop \
        --tags source-static [--run R] [--note "..."] [--overwrite]
    uv run data/scripts/curate.py sheet --clips-file list.txt --out DIR \
        [--run R] [--per-sheet 4]

render  writes side-by-side Cozmo|lamp GIFs to data/lamp_data/review_gifs/,
        cached by npz sha1 (only clips whose qpos actually changed between
        runs re-render); resumable.
serve   browser UI at http://localhost:7788 -- keyboard: k keep, d drop,
        f fix-mapping, 1-9/0/- toggle reason tags, n note, arrows navigate,
        u back. Verdicts persist to data/lamp_data/curation.csv (atomic
        rewrite, git-tracked).
panel   renders a fixed ~10-clip panel into review_gifs/panel/<tag>/ for
        side-by-side comparison of mapping variants at /panel.
verdict batch-applies a curation verdict to all clips carrying a metric
        flag; existing verdicts are never touched unless --overwrite is
        given (human review beats batch rules).
sheet   contact-sheet PNGs for visual clip audits: 4 clips per sheet,
        each clip as two rows of 8 keyframes (Cozmo source over lamp
        retarget) with a metrics caption. clips-file: one clip name per
        line (extra tab-separated columns ignored).
"""

import argparse
import csv
import datetime
import hashlib
import json
import os
import pathlib
import sys
import tempfile
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

DATA = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DATA))
from pipeline import (  # noqa: E402
    METRICS_DIR, NPZ_IN, Lamp, LampRenderer, _font, latest_run,
    load_labels, pick_gif_samples, read_run_json, run_dir, top_emotions)

LAMP_DATA = DATA / "lamp_data"
GIF_DIR = LAMP_DATA / "review_gifs"
PANEL_DIR = GIF_DIR / "panel"
MANIFEST_PATH = GIF_DIR / "render_manifest.json"
CURATION_CSV = LAMP_DATA / "curation.csv"

CURATION_FIELDS = ["clip_name", "verdict", "reason_tags", "note",
                   "reviewed_at", "run_reviewed", "mapping_version_reviewed"]
VERDICTS = ("keep", "drop", "fix_mapping")
# sticky tags mark source-intrinsic problems: a drop with one of these
# stays dropped across mapping iterations
STICKY_TAGS = ["source-static", "too-short", "source-degenerate",
               "untranslatable"]
TAGS = STICKY_TAGS + ["jarring", "saturated", "wrong-read", "off-character",
                      "not-cute", "timing", "other"]
TAG_KEYS = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "-"]

# fixed A/B panel: 4 known problem clips (worst saturation / rate-cap in
# v1.0) + the 6 per-emotion exemplars appended at render time
PANEL_PROBLEM_CLIPS = ["anim_speedtap_playerno_01", "anim_rtpmemorymatch_no_01",
                       "anim_explorer_huh_01", "anim_bored_event_02"]

GIF_W, GIF_H = 600, 243   # composed 840x340 canvas downscaled


# ---------------------------------------------------------------------------
# curation.csv
# ---------------------------------------------------------------------------

def load_curation():
    if not CURATION_CSV.exists():
        return {}
    with open(CURATION_CSV, newline="") as fh:
        return {r["clip_name"]: r for r in csv.DictReader(fh)}


def save_curation(cur):
    """Atomic rewrite, sorted by clip name."""
    CURATION_CSV.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=CURATION_CSV.parent, suffix=".tmp")
    with os.fdopen(fd, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CURATION_FIELDS)
        w.writeheader()
        for name in sorted(cur):
            w.writerow({k: cur[name].get(k, "") for k in CURATION_FIELDS})
    os.replace(tmp, CURATION_CSV)


# ---------------------------------------------------------------------------
# GIF rendering
# ---------------------------------------------------------------------------

def npz_sha1(path):
    return hashlib.sha1(path.read_bytes()).hexdigest()[:12]


def load_manifest():
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {}


def save_manifest(man):
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(man, indent=0, sort_keys=True))


class GifFactory:
    """Reusable renderers; composes the side-by-side frame like
    pipeline.render_gif but downscaled and optimized for a ~900-GIF corpus."""

    def __init__(self, lamp, labels):
        from cozmo_model import ClipRenderer
        from PIL import Image  # noqa: F401  (import check up front)
        self.cozmo = ClipRenderer(420, 300)
        self.lampr = LampRenderer(lamp)
        self.labels = labels
        self.font = _font(15)
        self.small = _font(12)

    def render(self, stem, z_src, z_out, out_path, stride=1):
        from PIL import Image, ImageDraw
        q = z_out["qpos"]
        light = z_out["light01"]
        row = self.labels.get(stem, {})
        emo = top_emotions(row) if row else ""
        frames = []
        T = len(q)
        for i in range(0, T, stride):
            canvas = Image.new("RGB", (840, 340), (16, 16, 20))
            canvas.paste(Image.fromarray(self.cozmo.frame(z_src, i)),
                         (0, 40))
            canvas.paste(
                Image.fromarray(self.lampr.frame(q[i], float(light[i]))),
                (420, 40))
            dr = ImageDraw.Draw(canvas)
            dr.text((10, 4), stem, font=self.font, fill=(240, 240, 240))
            dr.text((10, 22), emo, font=self.small, fill=(170, 170, 180))
            dr.text((770, 4), f"t={z_src['t'][i]:5.2f}s", font=self.small,
                    fill=(170, 170, 180))
            dr.text((430, 22), "cozmo (source)", font=self.small,
                    fill=(120, 200, 220))
            dr.text((826, 22), "lamp", font=self.small,
                    fill=(250, 210, 120), anchor="ra")
            frames.append(canvas.resize((GIF_W, GIF_H)))
        if not frames:
            return False
        out_path.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(out_path, save_all=True, append_images=frames[1:],
                       duration=33 * stride, loop=0, optimize=True)
        return True

    def close(self):
        self.cozmo.close()
        self.lampr.close()


def cmd_render(args):
    run = args.run or latest_run()
    out_dir = run_dir(run)
    stems = args.clips or sorted(p.stem for p in out_dir.glob("*.npz"))
    lamp = Lamp()
    labels = load_labels()
    factory = GifFactory(lamp, labels)
    man = load_manifest()
    done = skipped = 0
    t0 = time.time()
    try:
        for i, stem in enumerate(stems):
            src = out_dir / f"{stem}.npz"
            sha = npz_sha1(src)
            gif = GIF_DIR / f"{stem}.gif"
            if (not args.force and gif.exists()
                    and man.get(stem, {}).get("npz_sha1") == sha):
                skipped += 1
                continue
            ok = factory.render(stem, np.load(NPZ_IN / f"{stem}.npz"),
                                np.load(src), gif, stride=args.stride)
            if ok:
                man[stem] = dict(npz_sha1=sha, run=run)
                save_manifest(man)   # incremental: resumable
                done += 1
            if done and done % 10 == 0:
                rate = (time.time() - t0) / done
                left = len(stems) - i - 1
                print(f"  {done} rendered, {skipped} cached, "
                      f"~{rate * left / 60:.0f} min left")
    finally:
        factory.close()
    print(f"render done: {done} rendered, {skipped} cache hits "
          f"({(time.time() - t0) / 60:.1f} min)")


def cmd_panel(args):
    run = args.run or latest_run()
    out_dir = run_dir(run)
    info = read_run_json(run)
    tag = args.tag or f"{info['mapping_version']}-{info['constants_sha1'][:6]}"
    labels = load_labels()
    stems = [s for s in PANEL_PROBLEM_CLIPS
             if (out_dir / f"{s}.npz").exists()]
    stems += [s for s in pick_gif_samples(labels) if s not in stems]
    lamp = Lamp()
    factory = GifFactory(lamp, labels)
    try:
        for stem in stems:
            gif = PANEL_DIR / tag / f"{stem}.gif"
            factory.render(stem, np.load(NPZ_IN / f"{stem}.npz"),
                           np.load(out_dir / f"{stem}.npz"), gif)
            print(f"  panel {tag}/{stem}.gif")
    finally:
        factory.close()
    print(f"panel '{tag}' done ({len(stems)} clips) -- view at /panel")


# ---------------------------------------------------------------------------
# review server
# ---------------------------------------------------------------------------

def read_metrics(run):
    path = METRICS_DIR / f"{run}.csv"
    if not path.exists():
        sys.exit(f"no metrics for run '{run}' -- run "
                 f"`uv run data/pipeline.py metrics --run {run}` first")
    with open(path, newline="") as fh:
        return {r["clip_name"]: r for r in csv.DictReader(fh)}


def build_state(run, run_info, metrics, labels, curation, rendered):
    """Queue + per-clip payload for the UI."""
    clips = []
    for stem in sorted(metrics):
        m = metrics[stem]
        cur = curation.get(stem)
        stale = bool(cur) and cur.get("run_reviewed") != run
        sticky = bool(cur) and any(t in STICKY_TAGS for t in
                                   (cur.get("reason_tags") or "").split(";"))
        verdict = cur["verdict"] if cur else ""
        sev = float(m["severity"] or 0)
        flags = m["flags"]
        if verdict == "fix_mapping" and stale:
            group = 0          # mapping changed since -- re-check first
        elif not verdict and flags:
            group = 1          # flagged, never reviewed: worst first
        elif not verdict:
            group = 2          # unreviewed, unflagged: unbiased hash order
        elif verdict == "drop" and stale and not sticky:
            group = 3          # mapping-relative drop: one-key reconfirm
        elif verdict == "keep" and stale:
            group = 4          # spot-check for regressions
        else:
            group = 5          # done (current verdicts + sticky drops)
        h = int(hashlib.sha1(stem.encode()).hexdigest()[:8], 16)
        clips.append(dict(
            clip=stem, group=group, severity=sev, hash=h,
            T=int(m["T"]), dur=float(m["dur_s"] or 0), flags=flags,
            sat_max=float(m["sat_max"] or 0),
            rate_cap_frac=float(m["rate_cap_frac"] or 0),
            emotions=top_emotions(labels[stem]) if stem in labels else "",
            desc=labels.get(stem, {}).get("descriptions", ""),
            verdict=verdict,
            tags=(cur.get("reason_tags") or "") if cur else "",
            note=(cur.get("note") or "") if cur else "",
            stale=stale, sticky=sticky,
            reviewed_at=(cur.get("reviewed_at") or "") if cur else "",
            gif_v=rendered.get(stem, {}).get("npz_sha1", ""),
        ))
    clips.sort(key=lambda c: (c["group"], -c["severity"], c["hash"],
                              c["clip"]))
    return dict(run=run, mapping_version=run_info["mapping_version"],
                tags=TAGS, tag_keys=TAG_KEYS, sticky_tags=STICKY_TAGS,
                clips=clips)


def list_panel_tags():
    if not PANEL_DIR.is_dir():
        return []
    tags = [d for d in PANEL_DIR.iterdir() if d.is_dir()]
    tags.sort(key=lambda d: d.stat().st_mtime)
    return [dict(tag=d.name,
                 clips=sorted(p.stem for p in d.glob("*.gif")))
            for d in tags]


def make_handler(ctx):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json",
                  cache=False):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            if cache:
                self.send_header("Cache-Control",
                                 "public, max-age=31536000, immutable")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/":
                self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            elif path == "/panel":
                self._send(200, PANEL_PAGE.encode(),
                           "text/html; charset=utf-8")
            elif path == "/api/state":
                state = build_state(ctx["run"], ctx["run_info"],
                                    ctx["metrics"], ctx["labels"],
                                    load_curation(), load_manifest())
                self._send(200, json.dumps(state).encode())
            elif path == "/api/panel":
                self._send(200, json.dumps(list_panel_tags()).encode())
            elif path.startswith("/gifs/") or path.startswith("/panel/"):
                rel = pathlib.Path(path).relative_to("/")
                f = (GIF_DIR / rel.relative_to("gifs")
                     if path.startswith("/gifs/") else GIF_DIR / rel)
                f = f.resolve()
                if GIF_DIR.resolve() not in f.parents or not f.exists():
                    self._send(404, b"not found", "text/plain")
                    return
                self._send(200, f.read_bytes(), "image/gif", cache=True)
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path != "/api/verdict":
                self._send(404, b"not found", "text/plain")
                return
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n))
            stem = req["clip_name"]
            if stem not in ctx["metrics"] or \
                    req["verdict"] not in VERDICTS:
                self._send(400, b'{"error":"bad request"}')
                return
            tags = [t for t in req.get("reason_tags", []) if t in TAGS]
            cur = load_curation()
            cur[stem] = dict(
                clip_name=stem,
                verdict=req["verdict"],
                reason_tags=";".join(tags),
                note=req.get("note", "").replace("\n", " ").strip(),
                reviewed_at=datetime.datetime.now()
                .isoformat(timespec="seconds"),
                run_reviewed=ctx["run"],
                mapping_version_reviewed=ctx["run_info"]["mapping_version"],
            )
            save_curation(cur)
            self._send(200, b'{"ok":true}')

    return Handler


def cmd_serve(args):
    run = args.run or latest_run()
    ctx = dict(run=run, run_info=read_run_json(run),
               metrics=read_metrics(run), labels=load_labels())
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(ctx))
    n_gifs = len(list(GIF_DIR.glob("*.gif"))) if GIF_DIR.is_dir() else 0
    print(f"reviewing run '{run}' ({len(ctx['metrics'])} clips, "
          f"{n_gifs} GIFs rendered)")
    print(f"  http://localhost:{args.port}/        review queue")
    print(f"  http://localhost:{args.port}/panel   mapping A/B panel")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


# ---------------------------------------------------------------------------
# UI (single-file HTML/JS, no external assets)
# ---------------------------------------------------------------------------

PAGE = r"""<!doctype html>
<meta charset="utf-8"><title>lamp retarget review</title>
<style>
 body{background:#101014;color:#ddd;font:14px system-ui;margin:0}
 #top{display:flex;gap:16px;align-items:center;padding:8px 14px;
      background:#17171d;position:sticky;top:0}
 #gifbox{text-align:center;padding:10px}
 #gifbox img{max-width:96vw;image-rendering:auto;background:#000}
 #meta{max-width:820px;margin:0 auto;padding:0 14px}
 .badge{display:inline-block;padding:1px 8px;border-radius:9px;
        margin-right:6px;font-size:12px;background:#2a2a33}
 .badge.warn{background:#5a3a13}.badge.bad{background:#6b2020}
 .chip{display:inline-block;padding:1px 8px;border-radius:9px;margin:2px;
       background:#1e3a4a;font-size:12px}
 #desc{color:#9a9aa5;font-style:italic;margin:6px 0}
 .vbtn{padding:6px 14px;margin-right:8px;border:0;border-radius:6px;
       font-size:14px;cursor:pointer;color:#fff;background:#333}
 .vbtn.sel-keep{background:#1d6b2f}.vbtn.sel-drop{background:#8a2828}
 .vbtn.sel-fix{background:#8a6a1d}
 .tag{padding:3px 9px;margin:2px;border:1px solid #444;border-radius:6px;
      display:inline-block;cursor:pointer;font-size:12px;color:#bbb}
 .tag.on{background:#3a3a6b;color:#fff;border-color:#7a7ad0}
 .tag.sticky{border-style:dashed}
 #note{width:100%;background:#1b1b22;color:#ddd;border:1px solid #333;
       padding:6px;border-radius:6px;margin-top:6px}
 #help{color:#666;font-size:12px;margin:10px 0 30px}
 select{background:#1b1b22;color:#ddd;border:1px solid #333;padding:3px}
 .stale{color:#d0a040}
 kbd{background:#26262e;padding:0 5px;border-radius:4px}
</style>
<div id=top>
 <b id=run></b>
 <span id=progress></span>
 <select id=filter>
  <option value=queue>queue</option><option value=all>all</option>
  <option value=unreviewed>unreviewed</option>
  <option value=flagged>flagged</option><option value=stale>stale</option>
  <option value=kept>kept</option><option value=dropped>dropped</option>
  <option value=done>done</option>
 </select>
 <span id=pos></span>
 <a href=/panel style="color:#7aa">panel</a>
</div>
<div id=gifbox><img id=gif alt="run render first"></div>
<div id=meta>
 <div id=badges></div>
 <div id=chips></div>
 <div id=desc></div>
 <div id=verdict-row style="margin:8px 0">
  <button class=vbtn id=b-keep>Keep (k)</button>
  <button class=vbtn id=b-drop>Drop (d)</button>
  <button class=vbtn id=b-fix>Fix mapping (f)</button>
  <span id=vstate></span>
 </div>
 <div id=tags></div>
 <input id=note placeholder="note (n to focus, Esc to leave; saved with next verdict)">
 <div id=help>
  <kbd>k</kbd> keep &nbsp;<kbd>d</kbd> drop &nbsp;<kbd>f</kbd> fix-mapping
  &nbsp;<kbd>1</kbd>-<kbd>-</kbd> tags &nbsp;<kbd>n</kbd> note
  &nbsp;<kbd>&larr;</kbd><kbd>&rarr;</kbd> navigate &nbsp;<kbd>u</kbd> back.
  Verdict saves + advances. Dashed tags are sticky (source-intrinsic:
  the drop survives mapping changes).
 </div>
</div>
<script>
let S=null, order=[], idx=0, hist=[], tagsOn=new Set();
const $=id=>document.getElementById(id);
function cur(){return S.clips[order[idx]]}
function applyFilter(){
  const f=$('filter').value;
  order=S.clips.map((c,i)=>i).filter(i=>{
    const c=S.clips[i];
    if(f==='all')return true;
    if(f==='queue')return c.group<5;
    if(f==='unreviewed')return !c.verdict;
    if(f==='flagged')return !!c.flags;
    if(f==='stale')return c.stale;
    if(f==='kept')return c.verdict==='keep';
    if(f==='dropped')return c.verdict==='drop';
    if(f==='done')return c.group===5;
    return true});
  idx=Math.min(idx,Math.max(0,order.length-1)); hist=[]; show();
}
function badge(txt,cls){return `<span class="badge ${cls||''}">${txt}</span>`}
function show(){
  if(!order.length){$('gif').src='';$('badges').innerHTML='(empty filter)';return}
  const c=cur();
  $('gif').src=`/gifs/${c.clip}.gif?v=${c.gif_v}`;
  let b=badge(`${c.T} f / ${c.dur.toFixed(2)} s`);
  if(c.flags)for(const f of c.flags.split(';'))b+=badge(f,'bad');
  if(c.sat_max>0.01)b+=badge(`sat ${(c.sat_max*100).toFixed(0)}%`,'warn');
  if(c.rate_cap_frac>0.05)b+=badge(`rate ${(c.rate_cap_frac*100).toFixed(0)}%`,'warn');
  if(c.stale)b+=badge('stale review','warn');
  if(c.sticky)b+=badge('sticky drop');
  $('badges').innerHTML=`<b>${c.clip}</b><br>`+b;
  $('chips').innerHTML=c.emotions.split(', ').filter(x=>x)
    .map(e=>`<span class=chip>${e}</span>`).join('');
  $('desc').textContent=c.desc;
  tagsOn=new Set(c.tags.split(';').filter(x=>x));
  $('note').value=c.note||'';
  renderTags(); renderVerdict();
  $('pos').textContent=`${idx+1}/${order.length}`;
  const done=S.clips.filter(x=>x.group===5||(x.verdict&&!x.stale)).length;
  $('progress').textContent=`${done}/${S.clips.length} reviewed`;
}
function renderVerdict(){
  const c=cur();
  for(const[b,v,cls]of[['b-keep','keep','sel-keep'],['b-drop','drop','sel-drop'],
      ['b-fix','fix_mapping','sel-fix']])
    $(b).className='vbtn '+(c.verdict===v?cls:'');
  $('vstate').innerHTML=c.verdict?
    `<span class=${c.stale?'stale':''}>${c.verdict}${c.stale?' (stale: '+c.reviewed_at+')':''}</span>`:'';
}
function renderTags(){
  $('tags').innerHTML=S.tags.map((t,i)=>
   `<span class="tag ${tagsOn.has(t)?'on':''} ${S.sticky_tags.includes(t)?'sticky':''}"
     data-t="${t}">${S.tag_keys[i]} ${t}</span>`).join('');
  for(const el of document.querySelectorAll('.tag'))
    el.onclick=()=>{toggleTag(el.dataset.t)};
}
function toggleTag(t){tagsOn.has(t)?tagsOn.delete(t):tagsOn.add(t);renderTags()}
async function verdict(v){
  const c=cur();
  await fetch('/api/verdict',{method:'POST',
    body:JSON.stringify({clip_name:c.clip,verdict:v,
      reason_tags:[...tagsOn],note:$('note').value})});
  c.verdict=v;c.tags=[...tagsOn].join(';');c.note=$('note').value;
  c.stale=false;c.group=5;
  c.sticky=[...tagsOn].some(t=>S.sticky_tags.includes(t));
  hist.push(idx);
  if(idx<order.length-1){idx++} show();
}
function nav(d){if(d<0&&hist.length){idx=hist.pop()}
  else{idx=Math.max(0,Math.min(order.length-1,idx+d))} show()}
document.addEventListener('keydown',e=>{
  if(document.activeElement===$('note')){
    if(e.key==='Escape')$('note').blur(); return}
  if(e.key==='k')verdict('keep');
  else if(e.key==='d')verdict('drop');
  else if(e.key==='f')verdict('fix_mapping');
  else if(e.key==='n'){e.preventDefault();$('note').focus()}
  else if(e.key==='ArrowRight'||e.key==='l')nav(1);
  else if(e.key==='ArrowLeft'||e.key==='j')nav(-1);
  else if(e.key==='u')nav(-1);
  else{const i=S.tag_keys.indexOf(e.key);
       if(i>=0&&i<S.tags.length)toggleTag(S.tags[i])}
});
$('b-keep').onclick=()=>verdict('keep');
$('b-drop').onclick=()=>verdict('drop');
$('b-fix').onclick=()=>verdict('fix_mapping');
$('filter').onchange=applyFilter;
fetch('/api/state').then(r=>r.json()).then(s=>{S=s;
  $('run').textContent=`run ${s.run} (v${s.mapping_version})`;applyFilter()});
</script>
"""

PANEL_PAGE = r"""<!doctype html>
<meta charset="utf-8"><title>mapping A/B panel</title>
<style>
 body{background:#101014;color:#ddd;font:14px system-ui;margin:0;padding:12px}
 table{border-collapse:collapse}
 td,th{padding:4px;border:1px solid #26262e;vertical-align:top}
 img{width:420px;display:block}
 th{position:sticky;top:0;background:#17171d}
 a{color:#7aa}
</style>
<a href=/>&larr; review</a>
<h3>mapping variants, side by side (columns = panel tags, oldest first)</h3>
<div id=grid>loading...</div>
<script>
fetch('/api/panel').then(r=>r.json()).then(tags=>{
  if(!tags.length){document.getElementById('grid').textContent=
    'no panels yet: uv run data/scripts/curate.py panel --tag <name>';return}
  const clips=[...new Set(tags.flatMap(t=>t.clips))];
  let h='<table><tr><th>clip</th>'+
    tags.map(t=>`<th>${t.tag}</th>`).join('')+'</tr>';
  for(const c of clips){
    h+=`<tr><td>${c}</td>`+tags.map(t=>t.clips.includes(c)
      ?`<td><img loading=lazy src="/panel/${t.tag}/${c}.gif"></td>`
      :'<td>-</td>').join('')+'</tr>'}
  document.getElementById('grid').innerHTML=h+'</table>';
});
</script>
"""


# ---------------------------------------------------------------------------
# Batch verdicts by metric flag
# ---------------------------------------------------------------------------

def cmd_verdict(args):
    tags = [t for t in args.tags.split(";") if t]
    bad = [t for t in tags if t not in TAGS]
    if bad:
        sys.exit(f"unknown tags {bad}; valid: {TAGS}")

    run = args.run or latest_run()
    info = read_run_json(run)
    path = METRICS_DIR / f"{run}.csv"
    if not path.exists():
        sys.exit(f"no metrics for run '{run}' -- run pipeline.py metrics first")
    with open(path, newline="") as fh:
        matched = [r["clip_name"] for r in csv.DictReader(fh)
                   if args.flag in (r["flags"] or "").split(";")]
    if not matched:
        sys.exit(f"no clips carry flag {args.flag} in run {run}")

    cur = load_curation()
    applied, skipped = [], []
    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    for stem in matched:
        if stem in cur and cur[stem].get("verdict") and not args.overwrite:
            skipped.append((stem, cur[stem]["verdict"]))
            continue
        cur[stem] = dict(
            clip_name=stem, verdict=args.verdict,
            reason_tags=";".join(tags),
            note=args.note if args.note is not None
            else f"auto: {args.flag} flag",
            reviewed_at=stamp, run_reviewed=run,
            mapping_version_reviewed=info["mapping_version"])
        applied.append(stem)
    save_curation(cur)
    print(f"{args.verdict} applied to {len(applied)} clips "
          f"(flag {args.flag}, run {run})")
    if skipped:
        print(f"kept {len(skipped)} existing verdicts: "
              + ", ".join(f"{s}({v})" for s, v in skipped[:8])
              + (" ..." if len(skipped) > 8 else ""))


# ---------------------------------------------------------------------------
# Contact sheets for visual audits
# ---------------------------------------------------------------------------

TILE_W, TILE_H = 190, 136
N_FRAMES = 8
CAP_H = 34


def cmd_sheet(args):
    from PIL import Image, ImageDraw
    from cozmo_model import ClipRenderer

    run = args.run or latest_run()
    out_dir = run_dir(run)
    labels = load_labels()
    stems = [ln.split("\t")[0].strip()
             for ln in open(args.clips_file) if ln.strip()]
    dest = pathlib.Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)

    lamp = Lamp()
    cozmo = ClipRenderer(420, 300)
    lampr = LampRenderer(lamp)
    font = _font(13)

    clip_h = 2 * TILE_H + CAP_H
    sheet_w = N_FRAMES * TILE_W

    def strip(stem):
        z_src = np.load(NPZ_IN / f"{stem}.npz")
        z = np.load(out_dir / f"{stem}.npz")
        q, light = z["qpos"], z["light01"]
        T = len(q)
        idx = np.linspace(0, T - 1, min(N_FRAMES, T)).astype(int)
        im = Image.new("RGB", (sheet_w, clip_h), (12, 12, 16))
        for k, i in enumerate(idx):
            c = Image.fromarray(cozmo.frame(z_src, int(i))).resize(
                (TILE_W, TILE_H))
            l = Image.fromarray(
                lampr.frame(q[i], float(light[i]))).resize((TILE_W, TILE_H))
            im.paste(c, (k * TILE_W, 0))
            im.paste(l, (k * TILE_W, TILE_H))
        row = labels.get(stem, {})
        rng = float(np.ptp(q, axis=0).max()) if T else 0.0
        desc = (row.get("descriptions", "") or "")[:110]
        cap = (f"{stem}  T={T} ({T * 0.033:.1f}s)  range={rng:.2f}rad  "
               f"light[{light.min():.2f}-{light.max():.2f}]  "
               f"{top_emotions(row) if row else ''}")
        dr = ImageDraw.Draw(im)
        dr.text((6, 2 * TILE_H + 2), cap, font=font, fill=(240, 240, 160))
        dr.text((6, 2 * TILE_H + 18), desc, font=font, fill=(160, 160, 170))
        return im

    sheets = []
    for s0 in range(0, len(stems), args.per_sheet):
        group = stems[s0:s0 + args.per_sheet]
        sheet = Image.new("RGB", (sheet_w, clip_h * len(group)), (0, 0, 0))
        for gi, stem in enumerate(group):
            sheet.paste(strip(stem), (0, gi * clip_h))
        path = dest / f"sheet_{s0 // args.per_sheet:03d}.png"
        sheet.save(path)
        sheets.append(path)
        print(f"wrote {path} ({', '.join(group)})")
    cozmo.close()
    lampr.close()
    print(f"{len(sheets)} sheets, {len(stems)} clips")


def main():
    p = argparse.ArgumentParser(
        description="curation tooling: render | serve | panel | verdict "
                    "| sheet")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("render", help="render review GIFs (sha1-cached)")
    pr.add_argument("--run", default=None)
    pr.add_argument("--clips", nargs="+", default=None)
    pr.add_argument("--force", action="store_true")
    pr.add_argument("--stride", type=int, default=1,
                    help="render every Nth frame (default 1)")

    ps = sub.add_parser("serve", help="run the review web app")
    ps.add_argument("--run", default=None)
    ps.add_argument("--port", type=int, default=7788)

    pp = sub.add_parser("panel", help="render the A/B panel for this run")
    pp.add_argument("--run", default=None)
    pp.add_argument("--tag", default=None,
                    help="column label (default: version-constsha)")

    pv = sub.add_parser("verdict",
                        help="batch verdict for all clips with a flag")
    pv.add_argument("--flag", required=True,
                    help="metric flag to match, e.g. STATIC, DEGENERATE")
    pv.add_argument("--verdict", required=True, choices=VERDICTS)
    pv.add_argument("--tags", default="",
                    help="semicolon-joined reason tags")
    pv.add_argument("--note", default=None,
                    help="note (default: 'auto: <FLAG> flag')")
    pv.add_argument("--run", default=None)
    pv.add_argument("--overwrite", action="store_true",
                    help="replace existing verdicts too")

    pc = sub.add_parser("sheet", help="contact-sheet PNGs for clip audits")
    pc.add_argument("--clips-file", required=True)
    pc.add_argument("--out", required=True)
    pc.add_argument("--run", default=None)
    pc.add_argument("--per-sheet", type=int, default=4)

    args = p.parse_args()
    {"render": cmd_render, "serve": cmd_serve, "panel": cmd_panel,
     "verdict": cmd_verdict, "sheet": cmd_sheet}[args.cmd](args)


if __name__ == "__main__":
    main()
