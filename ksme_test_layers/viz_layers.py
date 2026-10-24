"""
가시화 유틸 — 1/4 단위셀 해를 전체 기둥(액적 6개)으로 펼쳐 저장/렌더링
"""
import numpy as np
from dolfinx import plot as dplot, fem
import basix.ufl
import params_layers as P
import mesh_column as MC


def to_pyvista(msh, w, mirror_y=True, mirror_z=True):
    """DOLFINx 해 -> pyvista UnstructuredGrid (P1 보간, y<0 / z<0 로 미러)."""
    import pyvista as pv
    V1 = fem.functionspace(msh, basix.ufl.element(
        "Lagrange", msh.basix_cell(), 1, shape=(3,)))
    u1 = fem.Function(V1); u1.interpolate(w.sub(0))
    Q1 = fem.functionspace(msh, basix.ufl.element("Lagrange", msh.basix_cell(), 1))
    p1 = fem.Function(Q1); p1.interpolate(w.sub(1))
    cells, types, xx = dplot.vtk_mesh(V1)
    g = pv.UnstructuredGrid(cells, types, xx)
    g.point_data["u"] = u1.x.array.reshape(-1, 3)
    g.point_data["p"] = p1.x.array
    for ax, do in ((1, mirror_y), (2, mirror_z)):
        if not do:
            continue
        m = g.copy()
        m.points[:, ax] *= -1.0
        m.point_data["u"][:, ax] *= -1.0
        g = g.merge(m, merge_points=False)
    g.point_data["speed"] = np.linalg.norm(g.point_data["u"], axis=1)
    return g


def save_solution(msh, w, path):
    try:
        to_pyvista(msh, w).save(path)
    except Exception as e:            # pyvista 없거나 렌더 불가 시 건너뜀
        print(f"    (vtu 저장 생략: {e})")


def sample_plane(msh, w, plane="y", value=0.0, nx=420, nz=340):
    """
    미러링을 고려해 단면 격자에서 속도를 샘플링한다.
    plane='y' : y=value 평면 (x-z 단면),  plane='z' : z=value 평면 (x-y 단면)
    반환: (A, B, U_mag, ux)  격자 좌표와 값
    """
    from dolfinx import geometry as dgeom
    V, dofs = w.function_space.sub(0).collapse()
    uu = fem.Function(V); uu.x.array[:] = w.x.array[dofs]; uu.x.scatter_forward()
    tree = dgeom.bb_tree(msh, msh.topology.dim)
    xs = np.linspace(P.CELL_X0, P.CELL_X1, nx)
    if plane == "y":
        bs = np.linspace(-P.CELL_ZMAX, P.CELL_ZMAX, nz)
        XX, BB = np.meshgrid(xs, bs, indexing='ij')
        pts = np.stack([XX.ravel(), np.full(XX.size, value), BB.ravel()], 1)
    else:
        bs = np.linspace(-P.CELL_YMAX, P.CELL_YMAX, nz)
        XX, BB = np.meshgrid(xs, bs, indexing='ij')
        pts = np.stack([XX.ravel(), BB.ravel(), np.full(XX.size, value)], 1)
    sy = np.where(pts[:, 1] < 0, -1.0, 1.0)
    sz = np.where(pts[:, 2] < 0, -1.0, 1.0)
    q = np.ascontiguousarray(np.stack(
        [pts[:, 0], np.abs(pts[:, 1]), np.abs(pts[:, 2])], 1))
    out = np.full((len(q), 3), np.nan)
    cand = dgeom.compute_collisions_points(tree, q)
    adj = dgeom.compute_colliding_cells(msh, cand, q)
    off, arr = adj.offsets, adj.array
    has = np.diff(off) > 0
    idx = np.flatnonzero(has)
    if len(idx):
        out[idx] = np.atleast_2d(uu.eval(q[idx], arr[off[:-1][has]].astype(np.int32)))
    out[:, 1] *= sy; out[:, 2] *= sz
    mag = np.linalg.norm(out, axis=1).reshape(XX.shape)
    return XX, BB, mag, out[:, 0].reshape(XX.shape)


def droplet_circles(ax, plane="y", **kw):
    """단면도에 액적 단면을 그린다."""
    import matplotlib.patches as mp
    style = dict(fc="0.85", ec="0.25", lw=0.8, zorder=5)
    style.update(kw)
    if plane == "y":
        for zk in MC.full_column_z():
            ax.add_patch(mp.Circle((0.0, zk*1e3), P.A_DROP*1e3, **style))
    else:
        ax.add_patch(mp.Circle((0.0, 0.0), P.A_DROP*1e3, **style))
