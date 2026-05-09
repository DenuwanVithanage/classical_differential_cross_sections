"""
3D Scatter Plot with true log-scale Z axis — Plotly (interactive) version
Usage: python plot3d_plotly.py yourfile.dat
       python plot3d_plotly.py yourfile.dat --cols 0 1 2
       python plot3d_plotly.py yourfile.dat --save plot.html
"""

import sys
import argparse
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Interactive 3D scatter with true log-Z")
    p.add_argument("datafile")
    p.add_argument("--cols", nargs=3, type=int, default=[0, 1, 2],
                   metavar=("X", "Y", "Z"))
    p.add_argument("--xlabel", default="v<sub>f</sub>")
    p.add_argument("--ylabel", default="j<sub>f</sub>")
    p.add_argument("--zlabel", default="d<sup>2</sup>σ/dvdj (Å<sup>2</sup>)")
    p.add_argument("--cmap", default="Viridis",
                   help="Plotly colorscale: Viridis, Plasma, Inferno, Magma, Hot, etc.")
    p.add_argument("--ptsize", type=float, default=2)
    p.add_argument("--title", default="Classical differential cross sections")
    p.add_argument("--save", default="")
    p.add_argument("--no-logz", action="store_true")
    p.add_argument("--max-pts", type=int, default=500_000)
    return p.parse_args()


def load_data(path, cols):
    data = np.loadtxt(path, comments=["#", "%"])
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data[:, cols[0]], data[:, cols[1]], data[:, cols[2]]


def main():
    args = parse_args()

    try:
        import plotly.graph_objects as go
    except ImportError:
        sys.exit("Plotly not installed.  Run:  pip install plotly")

    try:
        x, y, z = load_data(args.datafile, args.cols)
    except Exception as e:
        sys.exit(f"Error loading '{args.datafile}': {e}")

    print(f"Loaded {len(x):,} points")

    # ── Drop non-positive Z ──────────────────────────────────────────────────
    if not args.no_logz:
        mask = z > 0
        n_bad = (~mask).sum()
        if n_bad:
            print(f"  Dropping {n_bad:,} points with Z <= 0")
        x, y, z = x[mask], y[mask], z[mask]

    # ── Downsample ───────────────────────────────────────────────────────────
    if args.max_pts and len(x) > args.max_pts:
        idx = np.random.choice(len(x), args.max_pts, replace=False)
        x, y, z = x[idx], y[idx], z[idx]
        print(f"  Downsampled to {args.max_pts:,} points")

    # ── TRUE log-Z: pass log10(z) as spatial Z, then relabel the axis ticks ──
    if args.no_logz:
        z_plot  = z
        z_color = z
        z_axis_title = args.zlabel
        tickvals = None
        ticktext = None
    else:
        z_plot  = np.log10(z)         # spatial position on Z axis
        z_color = z_plot              # colour matches spatial position

        zmin, zmax = z_plot.min(), z_plot.max()
        tick_powers = np.arange(np.floor(zmin), np.ceil(zmax) + 1)
        tickvals = tick_powers.tolist()
        ticktext = [f"10<sup>{int(p)}</sup>" for p in tick_powers]
        z_axis_title = f"{args.zlabel} (log scale)"

    # ── Hover shows original Z values, not log10 ─────────────────────────────
    hover = (
        f"<b>{args.xlabel}</b>: %{{x:.4g}}<br>"
        f"<b>{args.ylabel}</b>: %{{y:.4g}}<br>"
        f"<b>{args.zlabel}</b>: %{{customdata:.4g}}<extra></extra>"
    )

    # ── Colorbar ticks also show real values ──────────────────────────────────
    if args.no_logz:
        colorbar_cfg = dict(title=dict(text=args.zlabel, side="right"), thickness=16,len=0.5,x=0.9)
    else:
        colorbar_cfg = dict(
            title=dict(text=z_axis_title, side="right"),
            thickness=16,
            tickvals=tickvals,
            ticktext=ticktext,
            len=0.5,x=0.9
        )

    trace = go.Scatter3d(
        x=x, y=y, z=z_plot,
        mode="markers",
        customdata=z,           # original Z for hover
        hovertemplate=hover,
        marker=dict(
            size=args.ptsize,
            color=z_color,
            colorscale=args.cmap,
            showscale=True,
            colorbar=colorbar_cfg,
            opacity=0.85,
            line=dict(width=0),
        ),
    )

    layout = go.Layout(
        width=1000,
        height=1000,
        scene=dict(
            xaxis=dict(title=args.xlabel,    showbackground=True,
                       backgroundcolor="rgb(245,245,250)"),
            yaxis=dict(title=args.ylabel,    showbackground=True,
                       backgroundcolor="rgb(245,245,250)"),
            zaxis=dict(
                title=z_axis_title,
                showbackground=True,
                backgroundcolor="rgb(245,245,250)",
                tickvals=tickvals,          # relabel spatial ticks
                ticktext=ticktext,
            ),
            aspectmode="auto",
        ),
        margin=dict(l=0, r=50, b=0, t=0),
        template="plotly_white",
    )

    fig = go.Figure(data=[trace], layout=layout)

    if args.save:
        fig.write_html(args.save, include_plotlyjs="cdn")
        print(f"Saved -> {args.save}  (open in any browser)")
    else:
        fig.show()


if __name__ == "__main__":
    main()
