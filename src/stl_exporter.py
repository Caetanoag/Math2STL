from typing import cast

import numpy as np
import pyvista as pv
import sympy as sp


def export_stl_pyvista(
    expr: sp.Expr,
    vars_list: list[str] | list[sp.Symbol],
    output_path: str = "modelo.stl",
    x_range: tuple[float, float] = (-10, 10),
    y_range: tuple[float, float] = (-10, 10),
    num_points: int = 200,
    base_thickness: float = 2.0,
) -> pv.PolyData:
    if len(vars_list) != 2:
        raise ValueError(f"Esperadas 2 variáveis, recebeu {len(vars_list)}.")

    x_sym, y_sym = [sp.Symbol(str(v)) for v in vars_list]

    x_vals = np.linspace(x_range[0], x_range[1], num_points, dtype=np.float32)
    y_vals = np.linspace(y_range[0], y_range[1], num_points, dtype=np.float32)
    x, y = np.meshgrid(x_vals, y_vals)

    f = sp.lambdify((x_sym, y_sym), expr, modules=["numpy"])
    z = f(x, y)

    if np.isscalar(z) or z.ndim == 0:
        z = np.full_like(x, float(z), dtype=np.float32)
    else:
        z = np.asarray(z, dtype=np.float32)

    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    z_min = float(np.min(z)) - base_thickness

    top_grid = pv.StructuredGrid(x, y, z)
    top = top_grid.extract_surface().triangulate()

    bot_grid = pv.StructuredGrid(x, y, np.full_like(z, z_min))
    bot = bot_grid.extract_surface().triangulate()

    boundary = top.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=False,
    )
    walls = boundary.extrude((0.0, 0.0, -1.0), capping=False)
    walls.points[boundary.n_points :, 2] = z_min
    walls = walls.triangulate()

    solid = pv.merge([top, bot, walls]).clean()
    solid = cast(
        pv.PolyData,
        solid.compute_normals(consistent_normals=True, auto_orient_normals=True),
    )

    solid.save(output_path)
    return solid
