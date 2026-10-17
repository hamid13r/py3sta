"""py3sta dashboard (Dash): pages render what dashboard.data reads from the project folder.

Shared workstations: the server listens on 127.0.0.1 by default (reach it with an SSH tunnel) and every request
needs the instance token, given once as ``?token=...`` in the URL (then kept in a cookie).
"""
from __future__ import annotations

import hmac
import secrets
from pathlib import Path

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


# ------------------------------------------------------------------- app
def create_app(root, token: str | None = None):
    """Dash app for the project at ``root``; ``token`` None = no token check (tests, single-user laptops)."""
    import flask
    from dash import Dash, Input, Output, dcc, html

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
    steps = [{"label": s.folder, "value": s.name} for s in data.STEPS if s.implemented]
    app.layout = html.Div(style=CSS, children=[
        html.H2(f"py3sta · {root.name}", style={"margin-bottom": "0"}),
        html.Div(str(root), style={"color": "#777", "font-size": "13px"}),
        dcc.Interval(id="tick", interval=REFRESH_MS),
        dcc.Tabs(id="tabs", value="cycles", children=[
            dcc.Tab(label="Cycles", value="cycles", children=[html.Div(id="overview")]),
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

    @app.callback(Output("log-cycle", "options"), Input("tick", "n_intervals"))
    def _cycles(_t):
        return [{"label": f"cycle-{n:03d}", "value": n} for n in _load(root).cycles]

    return app


def serve(root, host: str = "127.0.0.1", port: int = 8050, token: str | None = None, debug: bool = False):
    create_app(root, token).run(host=host, port=port, debug=debug)
