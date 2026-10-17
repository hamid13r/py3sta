"""py3sta dashboard (Dash): pages render what dashboard.data reads from the project folder.

Shared workstations: the server listens on 127.0.0.1 by default (reach it with an SSH tunnel) and every request
needs the instance token, given once as ``?token=...`` in the URL (then kept in a cookie).
"""
from __future__ import annotations

import hmac
import secrets
from pathlib import Path

import numpy as np

from . import data

REFRESH_MS = 10_000
COOKIE = "py3sta_token"
STATE_COLOUR = {"done": "#2e7d32", "stale": "#e09400", "missing": "#9e9e9e", "running": "#1565c0",
                "failed": "#c62828", "interrupted": "#ef6c00"}
CSS = {"font-family": "system-ui, sans-serif", "margin": "0 16px", "color": "#1b1b1b"}


def new_token() -> str:
    return secrets.token_urlsafe(16)


def _load(root):
    from ..pipeline import load_project

    return load_project(root)


# ------------------------------------------------------------- renderers
def state_badge(cell: data.StepCell):
    from dash import html

    tip = cell.state + (": " + "; ".join(cell.reasons) if cell.reasons else "")
    if cell.edited:
        tip += f" (edited: {', '.join(cell.edited)})"
    if cell.seconds is not None and cell.state == "done":
        tip += f"\n{cell.seconds:.0f} s, {cell.finished}" + (f", {cell.device}" if cell.device else "")
    if cell.n_warnings:
        tip += f"\n{cell.n_warnings} warning(s)"
    label = cell.state + (" ⚠" if cell.n_warnings else "") + (" ✎" if cell.edited else "")
    return html.Span(label, title=tip, className="badge",
                     style={"background": STATE_COLOUR.get(cell.state, "#616161"), "color": "white",
                            "padding": "2px 6px", "border-radius": "4px", "font-size": "12px",
                            "white-space": "nowrap"})


def overview_page(root):
    """Cycles table (state of every step, counts, resolution, average thumbnail) + resolution history."""
    from dash import dcc, html

    project = _load(root)
    ov = data.overview(project)
    folders = [c.folder for c in ov.cycles[0].steps] if ov.cycles else []
    head = html.Tr([html.Th(h) for h in ["cycle", "particles", "excluded", "ctf", "FSC 0.143", "average"] + folders])
    body = []
    for r in ov.cycles:
        avg = data.cycle_average(project, r.n)
        thumb = html.Img(src=data.thumbnail(avg), style={"height": "56px"}) if avg else ""
        res = f"{r.fsc143:g} Å" if r.fsc143 is not None else "—"
        if r.fsc143_last is not None and r.fsc143_last != r.fsc143:
            res += f" ({r.fsc143_last:g} Å last)"
        cells = [html.Td(f"{r.n:03d}"), html.Td(r.n_particles if r.n_particles is not None else "—"),
                 html.Td(r.n_excluded if r.n_excluded is not None else "—"), html.Td(r.ctf_mode or "—"),
                 html.Td(res), html.Td(thumb)]
        cells += [html.Td(state_badge(c)) for c in r.steps]
        body.append(html.Tr(cells, title=r.error or ""))
    table = html.Table([html.Thead(head), html.Tbody(body)], className="cycles",
                       style={"border-collapse": "collapse", "width": "100%"})
    hist = data.resolution_history(ov)
    children = [html.P(ov.stack, style={"color": "#555"}), html.Div(table, style={"overflow-x": "auto"})]
    if hist:
        import plotly.graph_objects as go

        fig = go.Figure(go.Scatter(x=[h[0] for h in hist], y=[h[1] for h in hist], mode="lines+markers",
                                   text=[f"{h[3]} particles" for h in hist], name="FSC 0.143"))
        fig.update_layout(height=260, margin=dict(l=50, r=20, t=30, b=40), title="resolution per cycle",
                          xaxis_title="cycle", yaxis_title="Å (FSC 0.143)", xaxis_dtick=1)
        children.append(dcc.Graph(figure=fig, config={"displaylogo": False}))
    return html.Div(children)


def log_view(root, n, step):
    """Log text, warnings and error of one step."""
    from dash import html

    if n is None or step is None:
        return html.P("choose a cycle and a step")
    lg = data.step_log(_load(root), int(n), step)
    if not lg["manifest"]:
        return html.P(f"{step}: not run yet")
    out = [html.P(f"status: {lg['status']}")]
    if lg["error"]:
        out.append(html.Pre(lg["error"] + "\n" + (lg["traceback"] or ""), style={"color": STATE_COLOUR["failed"]}))
    if lg["warnings"]:
        out.append(html.Ul([html.Li(w) for w in lg["warnings"]], style={"color": STATE_COLOUR["stale"]}))
    out.append(html.Pre(lg["log"] or "(no log.txt: the step ran before logs were saved)",
                        style={"background": "#f5f5f5", "padding": "8px", "max-height": "60vh", "overflow": "auto",
                               "font-size": "12px"}))
    return html.Div(out)


def _heat(fig, img, row, col, mask=None):
    import plotly.graph_objects as go

    fig.add_trace(go.Heatmap(z=img, colorscale="gray", showscale=False, hoverinfo="skip"), row=row, col=col)
    if mask is not None:
        fig.add_trace(go.Contour(z=mask, contours=dict(start=0.5, end=0.5, coloring="lines"), showscale=False,
                                 line=dict(color="#e53935", width=1), hoverinfo="skip"), row=row, col=col)


def _square_axes(fig, n):
    for i in range(1, n + 1):
        sfx = "" if i == 1 else str(i)
        fig.layout[f"xaxis{sfx}"].update(visible=False, constrain="domain")
        fig.layout[f"yaxis{sfx}"].update(visible=False, scaleanchor=f"x{sfx}", constrain="domain")


def map_figure(root, n, rel, k=0, project_views=False, lowpass=None, show_mask=False, positions=(50, 50, 50)):
    """XY / XZ / YZ slices (or projections) of map ``rel`` (relative to the cycle folder), volume ``k``;
    ``positions``: slice positions along z, y, x in % of the box."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    project = _load(root)
    if n is None or not rel:
        return go.Figure()
    vols = data.load_volumes(project.cycle_dir(int(n)) / rel)
    k = int(np.clip(k or 0, 0, len(vols) - 1))
    vol = data.filtered(vols[k], project.config.apix, lowpass)
    indices = [int(round((p if p is not None else 50) / 100 * (sz - 1))) for p, sz in zip(positions, vol.shape)]
    mask = None
    mpath = data.step_folder(project, int(n), "reference") / "mask.mrc"
    if show_mask and mpath.is_file():
        m = data.load_volumes(mpath)[0]
        mask = data.ortho_views(m, indices, project_views) if m.shape == vol.shape else None
        if project_views and mask is not None:
            mask = [v / max(float(v.max()), 1e-6) for v in mask]          # projected mask: contour at half max
    views = data.ortho_views(vol, indices, project_views)
    names = ["XY", "XZ", "YZ"]
    fig = make_subplots(rows=1, cols=3, subplot_titles=[f"{t} {'projection' if project_views else 'slice'}" for t in names],
                        horizontal_spacing=0.03)
    for i, img in enumerate(views):
        _heat(fig, img, 1, i + 1, None if mask is None else mask[i])
    _square_axes(fig, 3)
    title = rel + (f" [{k + 1}/{len(vols)}]" if len(vols) > 1 else "") + (f", low-pass {lowpass:g} Å" if lowpass else "")
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10), title=title, showlegend=False)
    return fig


def gallery_figure(root, n, rel, project_views=True, lowpass=None, axis=0, cols=6):
    """Every volume of a stack (class averages, eigenvolumes, references) as one view each."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    project = _load(root)
    if n is None or not rel:
        return go.Figure()
    vols = data.load_volumes(project.cycle_dir(int(n)) / rel)
    if len(vols) < 2:
        return go.Figure()
    cols = min(cols, len(vols))
    rows = -(-len(vols) // cols)
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[str(i + 1) for i in range(len(vols))],
                        horizontal_spacing=0.01, vertical_spacing=0.06)
    for i, v in enumerate(vols):
        _heat(fig, data.view(data.filtered(v, project.config.apix, lowpass), axis, project=project_views),
              i // cols + 1, i % cols + 1)
    _square_axes(fig, len(vols))
    fig.update_layout(height=170 * rows + 40, margin=dict(l=10, r=10, t=40, b=10), showlegend=False,
                      title=f"{rel}: all {len(vols)} volumes ({['XY', 'XZ', 'YZ'][axis]} "
                            f"{'projections' if project_views else 'central slices'})")
    return fig


def plots_page(root, n):
    """FSC of every cycle, scree, classification comparison, and the step PNGs of cycle n."""
    import plotly.graph_objects as go
    from dash import dash_table, dcc, html

    project = _load(root)
    out = []
    curves = data.all_fsc(project)
    if curves:
        fig = go.Figure()
        for c, (r, f) in curves.items():
            fig.add_trace(go.Scatter(x=1 / r, y=f, mode="lines", name=f"cycle {c:03d}",
                                     hovertemplate="%{customdata:.1f} Å: %{y:.3f}", customdata=r))
        fig.add_hline(y=0.143, line_dash="dot", line_color="#888")
        fig.update_layout(height=320, margin=dict(l=50, r=20, t=40, b=40), title="half-map FSC (01_average)",
                          xaxis_title="1/Å", yaxis_title="FSC", yaxis_range=[-0.1, 1.02])
        out.append(dcc.Graph(figure=fig, config={"displaylogo": False}))
    if n is None:
        return html.Div(out)
    n = int(n)
    sc = data.scree(project, n)
    if sc is not None:
        fig = go.Figure(go.Bar(x=sc["py3staFactor"], y=sc["py3staVarFraction"], name="variance fraction"))
        fig.add_trace(go.Scatter(x=sc["py3staFactor"], y=sc["py3staVarCumulative"], yaxis="y2", name="cumulative"))
        fig.update_layout(height=280, margin=dict(l=50, r=50, t=40, b=40), title=f"cycle {n:03d}: MSA scree",
                          xaxis_title="factor", yaxis2=dict(overlaying="y", side="right", range=[0, 1]))
        out.append(dcc.Graph(figure=fig, config={"displaylogo": False}))
    cmp_ = data.classify_compare(project, n)
    if cmp_ is not None:
        df = cmp_.rename(columns=lambda c: c.replace("py3sta", ""))
        out += [html.H4(f"cycle {n:03d}: classifications (05_classify/compare.star)"),
                dash_table.DataTable(df.round(3).to_dict("records"), [{"name": c, "id": c} for c in df.columns],
                                     sort_action="native", style_table={"overflow-x": "auto"},
                                     style_cell={"font-size": "12px", "padding": "2px 6px"})]
    pngs = data.cycle_pngs(project, n)
    if pngs:
        rel_cycle = project.cycle_dir(n).relative_to(project.root)
        out.append(html.H4(f"cycle {n:03d}: step diagnostics"))
        out.append(html.Div([html.Figure([html.Img(src=f"/files/{rel_cycle}/{p}", style={"max-width": "100%"}),
                                          html.Figcaption(p, style={"font-size": "12px", "color": "#555"})],
                                         style={"margin": "0 0 16px 0"}) for p in pngs]))
    return html.Div(out)


# ------------------------------------------------------------------- app
def create_app(root, token: str | None = None):
    """Dash app for the project at ``root``; ``token`` None = no token check (tests, single-user laptops)."""
    import flask
    from dash import Dash, Input, Output, State, dcc, html

    root = Path(root).resolve()
    project = _load(root)
    server = flask.Flask(__name__)

    if token:
        @server.before_request
        def _check_token():
            given = flask.request.args.get("token") or flask.request.cookies.get(COOKIE) or ""
            if not hmac.compare_digest(given, token):
                return flask.Response("py3sta dashboard: open the URL with ?token=... printed at start-up\n", 403)

        @server.after_request
        def _keep_token(resp):
            if flask.request.args.get("token") == token:
                resp.set_cookie(COOKIE, token, httponly=True, samesite="Strict")
            return resp

    app = Dash(__name__, server=server, title=f"py3sta · {root.name}", update_title=None,
               suppress_callback_exceptions=True)
    cycles = [{"label": f"cycle-{n:03d}", "value": n} for n in project.cycles]
    last = project.cycles[-1] if project.cycles else None
    steps = [{"label": s.folder, "value": s.name} for s in data.STEPS if s.implemented]
    app.layout = html.Div(style=CSS, children=[
        html.H2(f"py3sta · {root.name}", style={"margin-bottom": "0"}),
        html.Div(str(root), style={"color": "#777", "font-size": "13px"}),
        dcc.Interval(id="tick", interval=REFRESH_MS),
        dcc.Tabs(id="tabs", value="cycles", children=[
            dcc.Tab(label="Cycles", value="cycles", children=[html.Div(id="overview")]),
            dcc.Tab(label="Maps", value="maps", children=[
                html.Div([dcc.Dropdown(id="map-cycle", options=cycles, value=last, clearable=False,
                                       style={"width": "160px"}),
                          dcc.Dropdown(id="map-file", placeholder="map", style={"min-width": "320px", "flex": "1"}),
                          dcc.RadioItems(id="map-mode", options=["slices", "projections"], value="slices", inline=True),
                          dcc.Input(id="map-lowpass", type="number", placeholder="low-pass Å", debounce=True,
                                    min=0, style={"width": "110px"}),
                          dcc.Checklist(id="map-mask", options=[{"label": "mask", "value": "mask"}], value=[],
                                        inline=True)],
                         style={"display": "flex", "gap": "12px", "margin": "12px 0", "flex-wrap": "wrap",
                                "align-items": "center"}),
                html.Div([html.Span("volume"), dcc.Slider(id="map-k", min=1, max=1, step=1, value=1)],
                         id="map-k-box", style={"display": "none"}),
                dcc.Graph(id="map-ortho", config={"displaylogo": False}),
                html.Div([html.Div([html.Span(lab), dcc.Slider(id=f"map-{ax}", min=0, max=100, step=1, value=50,
                                                               marks=None, tooltip={"placement": "bottom"})],
                                   style={"flex": "1", "min-width": "160px"})
                          for lab, ax in (("z (XY)", "z"), ("y (XZ)", "y"), ("x (YZ)", "x"))],
                         style={"display": "flex", "gap": "16px", "flex-wrap": "wrap"}),
                dcc.Graph(id="map-gallery", config={"displaylogo": False})]),
            dcc.Tab(label="Plots", value="plots", children=[
                dcc.Dropdown(id="plots-cycle", options=cycles, value=last, clearable=False,
                             style={"width": "160px", "margin": "12px 0"}),
                html.Div(id="plots")]),
            dcc.Tab(label="Logs", value="logs", children=[
                html.Div([dcc.Dropdown(id="log-cycle", options=cycles, value=project.cycles[-1] if cycles else None,
                                       clearable=False, style={"width": "160px"}),
                          dcc.Dropdown(id="log-step", options=steps, value="average", clearable=False,
                                       style={"width": "180px"})],
                         style={"display": "flex", "gap": "8px", "margin": "12px 0", "flex-wrap": "wrap"}),
                html.Div(id="log")]),
        ]),
    ])

    @app.callback(Output("overview", "children"), Input("tick", "n_intervals"), Input("tabs", "value"))
    def _overview(_n, tab):
        return overview_page(root) if tab == "cycles" else []

    @app.callback(Output("log", "children"), Input("log-cycle", "value"), Input("log-step", "value"),
                  Input("tick", "n_intervals"))
    def _log(n, step, _t):
        return log_view(root, n, step)

    @app.callback(Output("log-cycle", "options"), Output("map-cycle", "options"), Output("plots-cycle", "options"),
                  Input("tick", "n_intervals"))
    def _cycles(_t):
        opts = [{"label": f"cycle-{n:03d}", "value": n} for n in _load(root).cycles]
        return opts, opts, opts

    @app.callback(Output("map-file", "options"), Output("map-file", "value"), Input("map-cycle", "value"),
                  State("map-file", "value"))
    def _map_files(n, cur):
        if n is None:
            return [], None
        files = data.cycle_maps(_load(root), int(n))
        default = "01_average/average.mrc" if "01_average/average.mrc" in files else (files[0] if files else None)
        return files, cur if cur in files else default

    @app.callback(Output("map-k", "max"), Output("map-k", "value"), Output("map-k-box", "style"),
                  Input("map-cycle", "value"), Input("map-file", "value"))
    def _map_k(n, rel):
        k = len(data.load_volumes(_load(root).cycle_dir(int(n)) / rel)) if n is not None and rel else 1
        return k, 1, {"display": "block" if k > 1 else "none"}

    @app.callback(Output("map-ortho", "figure"), Output("map-gallery", "figure"), Output("map-gallery", "style"),
                  Input("map-cycle", "value"), Input("map-file", "value"), Input("map-k", "value"),
                  Input("map-mode", "value"), Input("map-lowpass", "value"), Input("map-mask", "value"),
                  Input("map-z", "value"), Input("map-y", "value"), Input("map-x", "value"))
    def _map(n, rel, k, mode, lowpass, mask, z, y, x):
        proj = mode == "projections"
        ortho = map_figure(root, n, rel, (k or 1) - 1, proj, lowpass, "mask" in (mask or []), (z, y, x))
        gal = gallery_figure(root, n, rel, True, lowpass)
        return ortho, gal, {"display": "block" if gal.data else "none"}

    @app.callback(Output("plots", "children"), Input("plots-cycle", "value"), Input("tabs", "value"))
    def _plots(n, tab):
        return plots_page(root, n) if tab == "plots" else []

    @server.route("/files/<path:rel>")
    def _file(rel):
        if not rel.endswith(".png"):
            return flask.Response("only PNG files are served\n", 404)
        try:
            p = data.project_file(_load(root), rel)
        except ValueError:
            return flask.Response("outside the project\n", 404)
        if not p.is_file():
            return flask.Response("not found\n", 404)
        return flask.send_file(p, mimetype="image/png", max_age=0)

    return app


def serve(root, host: str = "127.0.0.1", port: int = 8050, token: str | None = None, debug: bool = False):
    create_app(root, token).run(host=host, port=port, debug=debug)
