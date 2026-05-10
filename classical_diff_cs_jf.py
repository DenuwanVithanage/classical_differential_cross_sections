#!/usr/bin/env python3

import struct
import numpy as np
import sys
import os

HEADER_FMT = "<46s 2x i 20f"
HEADER_SIZE = struct.calcsize(HEADER_FMT)

TRAJ_FMT = "<5d 4f"
TRAJ_SIZE = struct.calcsize(TRAJ_FMT)


def read_big(path):
    with open(path, "rb") as f:
        header = f.read(HEADER_SIZE)
        if len(header) != HEADER_SIZE:
            raise ValueError("Failed to read full header.")

        pot_raw, ver, *floats = struct.unpack(HEADER_FMT, header)

        bmax = floats[0]
        vi = floats[1]
        ji = floats[2]

        data = f.read()

    n = len(data) // TRAJ_SIZE
    rem = len(data) % TRAJ_SIZE
    if rem != 0:
        print(f"Warning: data size {len(data)} is not a multiple of record size {TRAJ_SIZE}. Ignoring last {rem} bytes.")

    b = np.empty(n, dtype=np.float64)
    vf = np.empty(n, dtype=np.float64)
    jf = np.empty(n, dtype=np.float64)

    off = 0
    for i in range(n):
        rec = data[off:off + TRAJ_SIZE]
        off += TRAJ_SIZE
        b_i, cth, phii, psiji, psivi, v_i, j_i, csif, scang = struct.unpack(TRAJ_FMT, rec)
        b[i] = b_i
        vf[i] = v_i
        jf[i] = j_i

    return {
        "bmax": bmax,
        "vi": vi,
        "ji": ji,
        "vf": vf,
        "jf": jf,
        "b": b,
        "version": ver,
    }


def make_weights(b, bmax, mode="uniform_b"):
    """
    mode:
      uniform_b  -> trajectories sampled uniformly in b
                    weight = (2*pi*bmax/N) * b
      uniform_b2 -> trajectories sampled uniformly in b^2
                    weight = pi*bmax^2/N
    """
    N = len(b)
    if N == 0:
        raise ValueError("No trajectories found.")

    if mode == "uniform_b":
        return (4.0 * np.pi * bmax / N) * b
    elif mode == "uniform_b2":
        return np.full(N, np.pi * bmax * bmax / N, dtype=np.float64)
    else:
        raise ValueError("weight mode must be 'uniform_b' or 'uniform_b2'")


def filter_data(vf, jf, b, weights,
                vmin=None, vmax=None,
                jmin=None, jmax=None,
                require_finite_b=True):
    mask = np.isfinite(vf) & np.isfinite(jf) & np.isfinite(weights)
    if require_finite_b:
        mask &= np.isfinite(b)

    if vmin is not None:
        mask &= (vf >= vmin)
    if vmax is not None:
        mask &= (vf <= vmax)
    if jmin is not None:
        mask &= (jf >= jmin)
    if jmax is not None:
        mask &= (jf <= jmax)

    return vf[mask], jf[mask], b[mask], weights[mask], mask


def build_edges(values, d, user_min=None, user_max=None, center_on_multiples=True):
    """
    If center_on_multiples=True and d=0.1, then bin centers land on:
      ..., -0.1, 0.0, 0.1, 0.2, ...
    by shifting edges by half a bin:
      ..., -0.05, 0.05, 0.15, 0.25, ...

    This fixes the old behavior where bins of width 0.1 were centered at
    0.15, 0.25, 0.35, ...
    """
    if len(values) == 0:
        raise ValueError("No values left after filtering.")

    lo = np.min(values) if user_min is None else user_min
    hi = np.max(values) if user_max is None else user_max

    if center_on_multiples:
        lo_edge = np.floor((lo - 0.5 * d) / d) * d + 0.5 * d
        hi_edge = np.ceil((hi - 0.5 * d) / d) * d + 0.5 * d
    else:
        lo_edge = np.floor(lo / d) * d
        hi_edge = np.ceil(hi / d) * d

    edges = np.arange(lo_edge, hi_edge + d, d)

    if len(edges) < 2:
        edges = np.array([lo_edge, lo_edge + d], dtype=float)

    return edges


def compute_2d_differential(vf, jf, weights, dv, dj,
                            vmin=None, vmax=None,
                            jmin=None, jmax=None):
    v_edges = build_edges(vf, dv, vmin, vmax, center_on_multiples=True)
    j_edges = build_edges(jf, dj, jmin, jmax, center_on_multiples=True)

    sigma2d, v_edges, j_edges = np.histogram2d(
        vf, jf,
        bins=[v_edges, j_edges],
        weights=weights
    )

    dsigma_dvdj = sigma2d / (dv * dj)

    v_cent = 0.5 * (v_edges[:-1] + v_edges[1:])
    j_cent = 0.5 * (j_edges[:-1] + j_edges[1:])

    return v_cent, j_cent, dsigma_dvdj, v_edges, j_edges


def compute_1d_differentials(dsigma_dvdj, dv, dj):
    dsigma_dv = np.sum(dsigma_dvdj, axis=1) * dj
    dsigma_dj = np.sum(dsigma_dvdj, axis=0) * dv
    return dsigma_dv, dsigma_dj


def total_cross_section_from_2d(dsigma_dvdj, dv, dj):
    return np.sum(dsigma_dvdj) * dv * dj


def jf_profile_at_target_v(vf, jf, weights, V_target, dV_window, j_edges, dj):
    """
    Build dσ/dj_f for a selected target v_f = V_target by summing over
    vf in [V_target - dV_window/2, V_target + dV_window/2).
    """
    lo = V_target - 0.5 * dV_window
    hi = V_target + 0.5 * dV_window

    mask = (vf >= lo) & (vf < hi)

    jf_sel = jf[mask]
    w_sel = weights[mask]

    hist, _ = np.histogram(jf_sel, bins=j_edges, weights=w_sel)
    dsigma_dj = hist / dj

    return dsigma_dj, mask.sum(), lo, hi


def write_2d_surface(outfile, v_cent, j_cent, dsigma_dvdj, skip_nonpositive=True):
    with open(outfile, "w") as f:
        f.write("# vf jf dsigma_dvdj\n")
        for i, v in enumerate(v_cent):
            for j, J in enumerate(j_cent):
                val = dsigma_dvdj[i, j]
                if skip_nonpositive and val <= 0.0:
                    continue
                f.write(f"{v:12.6f} {J:12.6f} {val:16.8e}\n")
            f.write("\n")


def write_2d_lines_fixed_j(outfile, v_cent, j_cent, dsigma_dvdj, skip_nonpositive=False):
    with open(outfile, "w") as f:
        f.write("# blocks of fixed jf; columns: vf jf dsigma_dvdj\n")
        for j, J in enumerate(j_cent):
            f.write(f"# jf = {J:.6f}\n")
            for i, v in enumerate(v_cent):
                val = dsigma_dvdj[i, j]
                if skip_nonpositive and val <= 0.0:
                    continue
                f.write(f"{v:12.6f} {J:12.6f} {val:16.8e}\n")
            f.write("\n")


def write_1d_curve(outfile, x, y, xlabel, ylabel):
    with open(outfile, "w") as f:
        f.write(f"# {xlabel} {ylabel}\n")
        for xi, yi in zip(x, y):
            if yi <= 0.0:
                continue
            f.write(f"{xi:12.6f} {yi:16.8e}\n")


def write_jf_profiles_for_selected_v(root, vf, jf, weights, j_edges, j_cent, dj,
                                     V_targets, dV_window):
    manifest = f"{root}.jf_profiles_manifest.txt"
    with open(manifest, "w") as mf:
        mf.write("# target_vf  lo  hi  ntraj  filename\n")

        for V in V_targets:
            prof, ntraj, lo, hi = jf_profile_at_target_v(
                vf, jf, weights,
                V_target=V,
                dV_window=dV_window,
                j_edges=j_edges,
                dj=dj
            )

            tag = str(V).replace("-", "m").replace(".", "p")
            win_tag = str(dV_window).replace("-", "m").replace(".", "p")
            outfile = f"{root}.jf_profile_v{tag}_win{win_tag}.dat"

            with open(outfile, "w") as f:
                f.write(f"# target_vf = {V:.6f}\n")
                f.write(f"# window = [{lo:.6f}, {hi:.6f})\n")
                f.write(f"# ntraj = {ntraj}\n")
                f.write("# jf dsigma_dj\n")
                for J, y in zip(j_cent, prof):
                    if y <= 0.0:
                        continue
                    f.write(f"{J:12.6f} {y:16.8e}\n")

            mf.write(f"{V:12.6f} {lo:12.6f} {hi:12.6f} {ntraj:10d} {outfile}\n")

    return manifest


def print_stats(name, arr):
    finite = np.isfinite(arr)
    bad = np.count_nonzero(~finite)
    good = arr[finite]
    if len(good) == 0:
        print(f"{name}: all bad")
        return
    print(f"{name}: total={len(arr)}, finite={len(good)}, bad={bad}")
    print(f"{name}: min={good.min():.8g}, max={good.max():.8g}, mean={good.mean():.8g}")


def parse_targets(arg):
    """
    Examples:
      "0,1,2,3"
      "-1,0,1,2"
      "0:5:1" -> 0,1,2,3,4,5
    """
    arg = arg.strip()
    if ":" in arg:
        parts = arg.split(":")
        if len(parts) != 3:
            raise ValueError("Range form must be start:stop:step")
        start = float(parts[0])
        stop = float(parts[1])
        step = float(parts[2])
        vals = np.arange(start, stop + 0.5 * step, step)
        return [float(x) for x in vals]
    else:
        return [float(x) for x in arg.split(",") if x.strip()]


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python classical_diff_cs.py file.big [dv] [dj] [weight_mode] [dV_window] [V_targets]")
        print("")
        print("Examples:")
        print("  python classical_diff_cs.py e2627_v0j18.big 0.1 0.1 uniform_b 1 0,1,2,3")
        print("  python classical_diff_cs.py e2627_v0j18.big 0.1 0.1 uniform_b2 1 0:5:1")
        return

    path = sys.argv[1]
    dv = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
    dj = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
    weight_mode = sys.argv[4] if len(sys.argv) > 4 else "uniform_b"
    dV_window = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
    V_targets = parse_targets(sys.argv[6]) if len(sys.argv) > 6 else [0, 1, 2, 3]

    print("Reading trajectories...")
    data = read_big(path)

    bmax = data["bmax"]
    vi = data["vi"]
    ji = data["ji"]
    vf = data["vf"]
    jf = data["jf"]
    b = data["b"]

    print("Trajectories:", len(vf))
    print("bmax:", bmax)
    print("vi:", vi)
    print("ji:", ji)
    print("weight_mode:", weight_mode)
    print("dV_window:", dV_window)
    print("V_targets:", V_targets)

    print_stats("vf", vf)
    print_stats("jf", jf)
    print_stats("b", b)

    weights = make_weights(b, bmax, mode=weight_mode)

    vf2, jf2, b2, w2, mask = filter_data(
        vf, jf, b, weights,
        vmin=None, vmax=None,
        jmin=None, jmax=None
    )

    print("Used trajectories after filtering:", len(vf2))

    v_cent, j_cent, dsigma_dvdj, v_edges, j_edges = compute_2d_differential(
        vf2, jf2, w2,
        dv=dv, dj=dj,
        vmin=None, vmax=None,
        jmin=None, jmax=None
    )

    dsigma_dv, dsigma_dj = compute_1d_differentials(dsigma_dvdj, dv, dj)
    sigma_total = total_cross_section_from_2d(dsigma_dvdj, dv, dj)

    root = os.path.splitext(os.path.basename(path))[0]

    out2d = f"{root}.d2cs_vf_jf.dat"
    out2d_lines = f"{root}.d2cs_vf_jf_lines.dat"
    out1dv = f"{root}.dcs_vf_total.dat"
    out1dj = f"{root}.dcs_jf_total.dat"

    write_2d_surface(out2d, v_cent, j_cent, dsigma_dvdj, skip_nonpositive=True)
    write_2d_lines_fixed_j(out2d_lines, v_cent, j_cent, dsigma_dvdj, skip_nonpositive=False)
    write_1d_curve(out1dv, v_cent, dsigma_dv, "vf", "dsigma_dv_total")
    write_1d_curve(out1dj, j_cent, dsigma_dj, "jf", "dsigma_dj_total")

    manifest = write_jf_profiles_for_selected_v(
        root=root,
        vf=vf2,
        jf=jf2,
        weights=w2,
        j_edges=j_edges,
        j_cent=j_cent,
        dj=dj,
        V_targets=V_targets,
        dV_window=dV_window
    )

    print("")
    print("Saved:")
    print("  2D surface             :", out2d)
    print("  2D line blocks fixed j :", out2d_lines)
    print("  total 1D dσ/dv         :", out1dv)
    print("  total 1D dσ/dj         :", out1dj)
    print("  selected-v manifest    :", manifest)
    print("")
    print(f"Integrated total cross section from 2D grid = {sigma_total:.8e}")


if __name__ == "__main__":
    main()
