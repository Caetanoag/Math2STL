import numpy as np
import pyvista as pv
import sympy as sp
from matplotlib import pyplot as plt


def plot_expression_2d(
    expr: sp.Expr,
    vars_list: list[str] | list[sp.Symbol],
    x_range: tuple[float, float] = (-10, 10),
    y_range: tuple[float, float] = (-10, 10),
    num_points: int = 100,
) -> None:
    """
    Plota uma superfície 3D z = f(x, y) a partir de uma
    expressão SymPy com duas variáveis.
    """
    if len(vars_list) != 2:
        raise ValueError(
            f"Esperadas exatamente 2 variáveis, mas recebeu {len(vars_list)}."
        )

    x_sym, y_sym = [sp.Symbol(str(v)) for v in vars_list]

    x_vals = np.linspace(x_range[0], x_range[1], num_points)
    y_vals = np.linspace(y_range[0], y_range[1], num_points)
    x, y = np.meshgrid(x_vals, y_vals)

    # Compila e avalia na malha
    f = sp.lambdify((x_sym, y_sym), expr, modules=["numpy"])
    z = f(x, y)

    if np.isscalar(z) or z.ndim == 0:
        z = np.full_like(x, float(z))

    # Configuração da figura 3D
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    surface = ax.plot_surface(
        x,
        y,
        z,
        cmap="viridis",
        edgecolor="none",
        alpha=0.9,
    )

    fig.colorbar(surface, ax=ax, shrink=0.5, aspect=12, label=f"f({x_sym}, {y_sym})")
    ax.set_title(f"Superfície: {expr}")
    ax.set_xlabel(str(x_sym))
    ax.set_ylabel(str(y_sym))
    ax.set_zlabel(f"f({x_sym}, {y_sym})")

    plt.tight_layout()
    plt.show()


def plot_expression_gpu(
    expr: sp.Expr,
    vars_list: list[str] | list[sp.Symbol],
    x_range: tuple[float, float] = (-10, 10),
    y_range: tuple[float, float] = (-10, 10),
    num_points: int = 1000,  # Suporta resoluções muito maiores sem travar
) -> pv.StructuredGrid:
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

    grid = pv.StructuredGrid(x, y, z)
    grid["Altura (Z)"] = z.ravel(order="F")

    plotter = pv.Plotter()
    plotter.add_mesh(grid, cmap="viridis", show_edges=False)
    plotter.add_axes()
    plotter.show_grid()
    plotter.show()

    return grid
