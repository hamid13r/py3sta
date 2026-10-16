"""``py3sta explore``: a self-contained HTML page to look at the particles in 2D (the 06_embed map, two MSA
factors, or a classify run's UMAP), coloured by any classification / tomogram / score, and to draw particle
groups with the lasso. Exported groups (JSON) become a classification through a classify run
``{method: groups, file: groups/NAME.json}``.

The page embeds the data and plotly.js (from the installed ``plotly`` package, or ``plotly_js``), so it
works offline and can be copied anywhere.

Groups file (format py3sta-groups, version 1)::

    {"format": "py3sta-groups", "version": 1, "project": ..., "cycle": 0, "created": ISO time, "axes": "...",
     "groups": [{"name": "good", "particles": ["TS_01/12", ...]}, ...]}

Group k (1-based, file order) becomes class k; particles in no group are class 0.
"""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import ConfigError

FORMAT = "py3sta-groups"
GROUPS_DIR = "groups"
HTML_FILE = "explore.html"


# ------------------------------------------------------------- groups file
def write_groups(path, groups: dict, project: str = "", cycle: int = 0, axes: str = "") -> Path:
    """Write {name: [particle names]} as a groups file (what the page exports)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"format": FORMAT, "version": 1, "project": project, "cycle": cycle,
           "created": dt.datetime.now().isoformat(timespec="seconds"), "axes": axes,
           "groups": [{"name": str(k), "particles": [str(x) for x in v]} for k, v in groups.items()]}
    path.write_text(json.dumps(doc, indent=1))
    return path


def load_groups(path, names) -> tuple[np.ndarray, list]:
    """(class per particle of ``names``: 1..G in file order, 0 = in no group; group names)."""
    path = Path(path)
    try:
        doc = json.loads(path.read_text())
    except ValueError as e:
        raise ConfigError(f"{path}: not a JSON file ({e})")
    if not isinstance(doc, dict) or doc.get("format") != FORMAT:
        raise ConfigError(f"{path}: not a py3sta groups file (format {FORMAT!r}; export it from 'py3sta explore')")
    groups = doc.get("groups") or []
    names = [str(x) for x in names]
    index = {nm: i for i, nm in enumerate(names)}
    labels = np.zeros(len(names), int)
    seen, gnames, unknown, twice = {}, [], [], []
    for g, grp in enumerate(groups, 1):
        gname = str(grp.get("name") or f"group {g}")
        gnames.append(gname)
        for nm in grp.get("particles", []):
            nm = str(nm)
            if nm not in index:
                unknown.append(nm)
                continue
            if nm in seen:
                twice.append(f"{nm} ({seen[nm]}, {gname})")
                continue
            seen[nm] = gname
            labels[index[nm]] = g
    if unknown:
        raise ConfigError(f"{path.name}: {len(unknown)} particle(s) not among the active particles of this cycle: "
                          + ", ".join(unknown[:5]) + (" ..." if len(unknown) > 5 else ""))
    if twice:
        raise ConfigError(f"{path.name}: {len(twice)} particle(s) in two groups: " + ", ".join(twice[:5]))
    if not gnames:
        raise ConfigError(f"{path.name}: no groups")
    return labels, gnames


# -------------------------------------------------------------------- data
def _r(v, nd=4):
    return [None if not np.isfinite(x) else x for x in np.round(np.asarray(v, float), nd).tolist()]


def explore_data(project, n: int, pcs: int = 8) -> dict:
    """Everything the page shows, as JSON-ready lists aligned with the particle names of msa.star."""
    from ..io.particles import ParticleSet
    from ..io.star import read_star
    from .classify import CLASS_COLUMN
    from .embed import EMBED_FILE
    from .project import PARTICLE_NAME, load_cycle_input
    from .steps import _particle_block, get_step, pose_star

    cdir = project.cycle_dir(n)
    msa_star = cdir / get_step("msa").folder / "msa.star"
    if not msa_star.is_file():
        raise FileNotFoundError(f"{msa_star}: run the msa step first (explore shows the MSA factors)")
    sc = _particle_block(read_star(msa_star))
    names = sc[PARTICLE_NAME].astype(str).to_numpy()
    path, _ = pose_star(project, n, "embed")
    path = path if path.is_file() else project.cycle_input(n)
    cps = load_cycle_input(project, n, path).particles          # active particles, rlnTomoName added
    df = cps.df.astype({PARTICLE_NAME: str}).set_index(PARTICLE_NAME)
    missing = set(names) - set(df.index)
    if missing:
        raise ValueError(f"{len(missing)} particle(s) of msa.star are not active in {path.name} (re-run msa)")
    df = df.loc[names].reset_index()
    ps = ParticleSet(df, cps.ts_apix)

    axes, num, cat = {}, {}, {}
    emb = cdir / get_step("embed").folder / EMBED_FILE
    if emb.is_file():
        e = _particle_block(read_star(emb)).astype({PARTICLE_NAME: str}).set_index(PARTICLE_NAME).loc[names]
        method = project.cycle_config(n).embed.method
        axes[f"06_embed ({method})"] = [_r(e["py3staEmbed1"]), _r(e["py3staEmbed2"])]
    pc_cols = [c for c in sc.columns if c.startswith("py3staPC")][:pcs]
    for c in pc_cols:
        num[c.replace("py3sta", "")] = _r(sc[c])
    classifications = []
    comp = cdir / get_step("classify").folder / "compare.star"
    if comp.is_file():
        for row in _particle_block(read_star(comp)).itertuples():
            run, tag = str(row.py3staRun), str(row.py3staTag)
            f = comp.parent / run / f"class_{tag}.star"
            if not f.is_file():
                continue
            cs = _particle_block(read_star(f)).astype({PARTICLE_NAME: str}).set_index(PARTICLE_NAME)
            if not set(names) <= set(cs.index):
                continue
            label = f"{run}/{tag}"
            cat[label] = cs.loc[names, CLASS_COLUMN].astype(int).astype(str).tolist()
            classifications.append(label)
            embf = comp.parent / run / "embedding.npy"
            if embf.is_file() and f"{run} (UMAP)" not in axes:
                E = np.load(embf)
                if E.ndim == 2 and E.shape[1] >= 2 and len(E) == len(names):
                    axes[f"{run} (UMAP)"] = [_r(E[:, 0]), _r(E[:, 1])]
    cat["tomogram"] = df["rlnTomoName"].astype(str).tolist()
    if "rlnRandomSubset" in df:
        cat["half set"] = df["rlnRandomSubset"].astype(int).astype(str).tolist()
    if "py3staRefCC" in df:
        num["alignment CC"] = _r(df["py3staRefCC"])
    beam = np.abs(ps.matrices[:, :, 2])                     # beam direction in the reference frame (sign-free)
    for i, ax in enumerate("xyz"):
        num[f"|beam {ax}| (wedge)"] = _r(beam[:, i], 3)
    num["height z (px)"] = _r(ps.centers_px[:, 2], 1)
    if len(pc_cols) >= 2:
        axes.setdefault("PC1 / PC2", None)                  # the page builds it from the PC scores
    return {"project": project.root.name, "cycle": n, "names": names.tolist(), "axes": axes,
            "pcs": [c.replace("py3sta", "") for c in pc_cols], "numeric": num, "categorical": cat,
            "classifications": classifications, "groups_dir": str(cdir / GROUPS_DIR),
            "created": dt.datetime.now().isoformat(timespec="seconds")}


# -------------------------------------------------------------------- page
def plotly_source(plotly_js=None) -> str:
    if plotly_js:
        return Path(plotly_js).read_text()
    try:
        import plotly
    except ImportError:
        raise ConfigError("explore embeds plotly.js: pip install plotly (or give --plotly-js plotly.min.js)")
    f = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
    if not f.is_file():
        raise ConfigError(f"{f} not found: give --plotly-js plotly.min.js")
    return f.read_text()


@dataclass
class ExploreReport:
    path: Path
    n_particles: int
    size_mb: float
    groups_dir: Path
    axes: list
    colourings: int


def build_explore(project, n: int, out=None, plotly_js=None, pcs: int = 8) -> ExploreReport:
    from .steps import get_step

    data = explore_data(project, n, pcs)
    out = Path(out) if out else project.cycle_dir(n) / get_step("embed").folder / HTML_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    html = (PAGE.replace("@@TITLE@@", f"{data['project']} cycle {n:03d}")
                .replace("@@PLOTLY@@", plotly_source(plotly_js).replace("</script", "<\\/script"))
                .replace("@@DATA@@", payload)
                .replace("@@APP@@", APP))
    out.write_text(html)
    return ExploreReport(out, len(data["names"]), out.stat().st_size / 1e6, Path(data["groups_dir"]),
                         list(data["axes"]), len(data["numeric"]) + len(data["categorical"]))


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>py3sta explore: @@TITLE@@</title>
<style>
:root { --bg:#fafafa; --fg:#222; --muted:#666; --line:#ddd; --panel:#fff; --accent:#2b6cb0; }
@media (prefers-color-scheme: dark) { :root { --bg:#181a1b; --fg:#e6e6e6; --muted:#9a9a9a; --line:#333; --panel:#222; --accent:#63a4ff; } }
body { margin:0; font:14px/1.4 system-ui, sans-serif; background:var(--bg); color:var(--fg); }
header { padding:10px 16px; border-bottom:1px solid var(--line); display:flex; gap:16px; align-items:baseline; flex-wrap:wrap; }
header h1 { font-size:16px; margin:0; } header span { color:var(--muted); }
main { display:flex; gap:12px; padding:12px 16px; flex-wrap:wrap; }
#plot { flex:1 1 640px; min-height:640px; background:var(--panel); border:1px solid var(--line); border-radius:6px; }
aside { flex:0 1 360px; display:flex; flex-direction:column; gap:12px; }
section { background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:10px; }
section h2 { font-size:13px; margin:0 0 8px; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); }
label { display:block; margin:4px 0 2px; color:var(--muted); font-size:12px; }
select, input, button { font:inherit; color:inherit; background:var(--bg); border:1px solid var(--line); border-radius:4px; padding:4px 6px; }
select, input[type=text] { width:100%; box-sizing:border-box; }
button { cursor:pointer; } button.primary { background:var(--accent); color:#fff; border-color:var(--accent); }
.row { display:flex; gap:6px; align-items:center; margin-top:6px; flex-wrap:wrap; }
table { width:100%; border-collapse:collapse; font-size:13px; } td, th { padding:3px 4px; border-bottom:1px solid var(--line); text-align:left; }
td.num { text-align:right; } .muted { color:var(--muted); font-size:12px; } .note { color:#c05621; font-size:12px; min-height:1em; }
.swatch { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:4px; vertical-align:middle; }
code { font-size:12px; word-break:break-all; }
</style></head><body>
<header><h1>py3sta explore</h1><span>@@TITLE@@</span><span id="count"></span></header>
<main><div id="plot"></div>
<aside>
<section><h2>View</h2>
 <label for="axes">Axes</label><select id="axes"></select>
 <div id="pcpick" class="row" style="display:none"><select id="pcx"></select><select id="pcy"></select></div>
 <label for="colour">Colour by</label><select id="colour"></select>
 <div class="muted" id="hint">Lasso or box-select points (toolbar), then add them as a group.</div>
</section>
<section><h2>Groups</h2>
 <div class="row"><span id="selinfo" class="muted">no selection</span></div>
 <div class="row"><input type="text" id="gname" placeholder="group name (optional)"></div>
 <div class="row"><button class="primary" id="add">Add selection as group</button><button id="undo">Undo</button><button id="clearsel">Clear selection</button></div>
 <div class="note" id="note"></div>
 <table id="gtable"><thead><tr><th>group</th><th>particles</th><th>main tomograms</th><th></th></tr></thead><tbody></tbody></table>
 <div class="muted" id="rest"></div>
</section>
<section><h2>Export</h2>
 <label for="setname">File name</label><input type="text" id="setname" value="groups">
 <div class="row"><button class="primary" id="export">Export groups (JSON)</button><label class="muted" style="margin:0"><input type="file" id="load" accept=".json" style="display:none"><span style="cursor:pointer;text-decoration:underline" onclick="document.getElementById('load').click()">load a groups file</span></label></div>
 <div class="muted" style="margin-top:6px">Save it to <code id="gdir"></code> and add a classify run:<br><code id="snippet"></code></div>
</section>
</aside></main>
<script>@@PLOTLY@@</script>
<script type="application/json" id="data">@@DATA@@</script>
<script>@@APP@@</script>
</body></html>
"""

APP = r"""
(function () {
  'use strict';
  var D = JSON.parse(document.getElementById('data').textContent);
  var N = D.names.length;
  var PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
    '#17becf', '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#c49c94', '#f7b6d2', '#dbdb8d', '#9edae5', '#393b79'];
  var groups = [], history = [], selected = [], listening = false;
  var el = function (id) { return document.getElementById(id); };

  function option(sel, value, text) { var o = document.createElement('option'); o.value = value; o.textContent = text || value; sel.appendChild(o); }
  Object.keys(D.axes).forEach(function (k) { option(el('axes'), k); });
  D.pcs.forEach(function (p, i) { option(el('pcx'), p); option(el('pcy'), p); });
  if (D.pcs.length > 1) { el('pcy').value = D.pcs[1]; }
  option(el('colour'), 'cat:groups (current)', 'groups (current)');
  D.classifications.forEach(function (k) { option(el('colour'), 'cat:' + k, 'class ' + k); });
  Object.keys(D.categorical).forEach(function (k) { if (D.classifications.indexOf(k) < 0) { option(el('colour'), 'cat:' + k, k); } });
  Object.keys(D.numeric).forEach(function (k) { option(el('colour'), 'num:' + k, k); });
  el('colour').value = D.classifications.length ? 'cat:' + D.classifications[0] : 'cat:tomogram';
  el('gdir').textContent = D.groups_dir + '/';
  el('count').textContent = N + ' particles';

  function xy() {
    var a = el('axes').value;
    el('pcpick').style.display = D.axes[a] ? 'none' : 'flex';
    if (D.axes[a]) { return { x: D.axes[a][0], y: D.axes[a][1], lx: a + ' 1', ly: a + ' 2' }; }
    return { x: D.numeric[el('pcx').value], y: D.numeric[el('pcy').value], lx: el('pcx').value, ly: el('pcy').value };
  }
  function groupLabels() {
    var lab = new Array(N).fill('(none)');
    groups.forEach(function (g) { g.idx.forEach(function (i) { lab[i] = g.name; }); });
    return lab;
  }
  function colouring() {
    var v = el('colour').value, kind = v.slice(0, 3), key = v.slice(4);
    if (kind === 'num') { return { kind: 'num', key: key, values: D.numeric[key] }; }
    return { kind: 'cat', key: key, values: key === 'groups (current)' ? groupLabels() : D.categorical[key] };
  }
  function sortCats(cats) {
    return cats.sort(function (a, b) { var x = Number(a), y = Number(b);
      if (!isNaN(x) && !isNaN(y)) { return x - y; } return a < b ? -1 : a > b ? 1 : 0; });
  }
  function draw() {
    var p = xy(), c = colouring(), traces = [];
    if (c.kind === 'num') {
      var idx = []; for (var i = 0; i < N; i++) { idx.push(i); }
      traces.push({ type: 'scattergl', mode: 'markers', x: p.x, y: p.y, customdata: idx,
        text: D.names, hovertemplate: '%{text}<br>' + c.key + ' %{marker.color}<extra></extra>',
        marker: { size: 4, color: c.values, colorscale: 'Viridis', showscale: true, colorbar: { title: { text: c.key }, thickness: 12 } } });
    } else {
      var by = {};
      for (var j = 0; j < N; j++) { var k = String(c.values[j]); (by[k] = by[k] || []).push(j); }
      sortCats(Object.keys(by)).forEach(function (k, n) {
        var ids = by[k];
        traces.push({ type: 'scattergl', mode: 'markers', name: k + ' (' + ids.length + ')', customdata: ids,
          x: ids.map(function (i) { return p.x[i]; }), y: ids.map(function (i) { return p.y[i]; }),
          text: ids.map(function (i) { return D.names[i]; }), hovertemplate: '%{text}<br>' + c.key + ' ' + k + '<extra></extra>',
          marker: { size: 4, color: (k === '(none)' || k === '0') && c.key === 'groups (current)' ? '#bbbbbb' : PALETTE[n % PALETTE.length] } });
      });
    }
    var dark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    var layout = { dragmode: 'lasso', uirevision: el('axes').value + el('pcx').value + el('pcy').value,
      margin: { l: 50, r: 10, t: 10, b: 45 }, xaxis: { title: { text: p.lx }, zeroline: false }, yaxis: { title: { text: p.ly }, zeroline: false },
      legend: { itemsizing: 'constant' }, paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
      font: { color: dark ? '#e6e6e6' : '#222' } };
    Plotly.react('plot', traces, layout, { displaylogo: false, responsive: true });
    if (!listening) {
      el('plot').on('plotly_selected', function (ev) {
        selected = ev && ev.points ? ev.points.map(function (q) { return q.customdata; }) : [];
        el('selinfo').textContent = selected.length ? selected.length + ' particle(s) selected' : 'no selection';
      });
      el('plot').on('plotly_deselect', function () { selected = []; el('selinfo').textContent = 'no selection'; });
      listening = true;
    }
  }
  function snapshot() { history.push(groups.map(function (g) { return { name: g.name, idx: new Set(g.idx) }; })); }
  function tomoSummary(g) {
    var t = {}, tomos = D.categorical.tomogram; g.idx.forEach(function (i) { t[tomos[i]] = (t[tomos[i]] || 0) + 1; });
    return Object.keys(t).sort(function (a, b) { return t[b] - t[a]; }).slice(0, 3)
      .map(function (k) { return k + ' ' + Math.round(100 * t[k] / g.idx.size) + '%'; }).join(', ');
  }
  function render() {
    var body = el('gtable').querySelector('tbody'); body.innerHTML = ''; var used = 0;
    groups.forEach(function (g, i) {
      used += g.idx.size;
      var tr = document.createElement('tr');
      var name = document.createElement('td'); name.innerHTML = '<span class="swatch" style="background:' + PALETTE[i % PALETTE.length] + '"></span>';
      var a = document.createElement('a'); a.href = '#'; a.textContent = g.name; a.title = 'rename';
      a.onclick = function (e) { e.preventDefault(); var v = prompt('Group name', g.name); if (v) { snapshot(); g.name = v.trim(); render(); refresh(); } };
      name.appendChild(a);
      var cnt = document.createElement('td'); cnt.className = 'num'; cnt.textContent = g.idx.size;
      var tm = document.createElement('td'); tm.className = 'muted'; tm.textContent = tomoSummary(g);
      var del = document.createElement('td'); var b = document.createElement('button'); b.textContent = 'delete';
      b.onclick = function () { snapshot(); groups.splice(i, 1); render(); refresh(); }; del.appendChild(b);
      tr.appendChild(name); tr.appendChild(cnt); tr.appendChild(tm); tr.appendChild(del); body.appendChild(tr);
    });
    el('rest').textContent = groups.length ? (N - used) + ' particle(s) in no group (class 0)' : '';
    var file = (el('setname').value || 'groups').replace(/[^A-Za-z0-9_.-]/g, '_');
    el('snippet').textContent = 'lasso: {method: groups, file: groups/' + file + '.json}';
  }
  function refresh() { if (el('colour').value === 'cat:groups (current)') { draw(); } }
  el('add').onclick = function () {
    if (!selected.length) { el('note').textContent = 'select particles first (lasso or box tool)'; return; }
    snapshot();
    var moved = 0, sel = new Set(selected);
    groups.forEach(function (g) { sel.forEach(function (i) { if (g.idx.delete(i)) { moved++; } }); });
    groups = groups.filter(function (g) { return g.idx.size > 0; });
    var name = el('gname').value.trim() || 'group ' + (groups.length + 1);
    groups.push({ name: name, idx: sel });
    el('note').textContent = moved ? moved + ' particle(s) moved from other groups to "' + name + '"' : '';
    el('gname').value = ''; render();
    if (el('colour').value !== 'cat:groups (current)') { el('colour').value = 'cat:groups (current)'; }
    draw();
  };
  el('undo').onclick = function () { if (history.length) { groups = history.pop(); render(); refresh(); el('note').textContent = ''; } };
  el('clearsel').onclick = function () { selected = []; el('selinfo').textContent = 'no selection'; draw(); };
  el('setname').oninput = render;
  el('export').onclick = function () {
    if (!groups.length) { el('note').textContent = 'no groups to export'; return; }
    var doc = { format: 'py3sta-groups', version: 1, project: D.project, cycle: D.cycle, created: new Date().toISOString(),
      axes: el('axes').value, groups: groups.map(function (g) {
        return { name: g.name, particles: Array.from(g.idx).sort(function (a, b) { return a - b; }).map(function (i) { return D.names[i]; }) }; }) };
    var file = (el('setname').value || 'groups').replace(/[^A-Za-z0-9_.-]/g, '_') + '.json';
    var blob = new Blob([JSON.stringify(doc, null, 1)], { type: 'application/json' });
    var a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = file; document.body.appendChild(a); a.click();
    setTimeout(function () { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
    el('note').textContent = 'downloaded ' + file + ': move it to ' + D.groups_dir + '/';
  };
  el('load').onchange = function (ev) {
    var f = ev.target.files[0]; if (!f) { return; }
    var rd = new FileReader();
    rd.onload = function () {
      try {
        var doc = JSON.parse(rd.result), index = {}, missing = 0;
        if (doc.format !== 'py3sta-groups') { throw new Error('not a py3sta groups file'); }
        D.names.forEach(function (nm, i) { index[nm] = i; });
        snapshot();
        groups = doc.groups.map(function (g) { var s = new Set();
          g.particles.forEach(function (nm) { if (nm in index) { s.add(index[nm]); } else { missing++; } });
          return { name: g.name, idx: s }; });
        el('setname').value = f.name.replace(/\.json$/, '');
        el('note').textContent = 'loaded ' + groups.length + ' group(s)' + (missing ? '; ' + missing + ' unknown particle(s) skipped' : '');
        render(); el('colour').value = 'cat:groups (current)'; draw();
      } catch (e) { el('note').textContent = 'could not load: ' + e.message; }
    };
    rd.readAsText(f); ev.target.value = '';
  };
  ['axes', 'pcx', 'pcy', 'colour'].forEach(function (id) { el(id).onchange = draw; });
  render(); draw();
})();
"""
