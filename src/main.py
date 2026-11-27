import re
from pathlib import Path

import pyvista as pv

from src.dxf_exporter import export_mdf_base_dxf, load_dxf_for_preview
from src.parser import parse_equation
from src.stl_exporter import export_stl_pyvista


def preview_assembly(
    solid_mesh: pv.PolyData,
    dxf_path: str,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    margin_mm: float = 12.0,
    footer_height_mm: float = 34.0,
    mdf_thickness: float = 3.0,
) -> None:
    board_mesh, engravings = load_dxf_for_preview(dxf_path, mdf_thickness=mdf_thickness)

    aligned_stl = solid_mesh.copy()
    z_bottom = float(aligned_stl.bounds[4])

    dx = margin_mm - x_range[0]
    dy = (margin_mm + footer_height_mm) - y_range[0]
    dz = -z_bottom

    aligned_stl.translate((dx, dy, dz), inplace=True)

    plotter = pv.Plotter(title="math2stl - Pré-visualização da Montagem")

    if board_mesh is not None:
        plotter.add_mesh(board_mesh, color="#c19a6b", label="Base MDF")

    if engravings.n_points > 0:
        plotter.add_mesh(
            engravings, color="#3b2615", line_width=2, label="Gravação Laser"
        )

    plotter.add_mesh(
        aligned_stl, color="#4a90e2", smooth_shading=True, label="Modelo 3D (STL)"
    )

    plotter.add_legend()
    plotter.add_axes()
    plotter.show_grid()
    plotter.show()


def sanitize_filename(text: str) -> str:
    rhs = text.split("=", 1)[-1].strip()
    rhs = rhs.replace("**", "pow").replace("^", "pow")
    rhs = rhs.replace("*", "_").replace("/", "_div_")
    rhs = rhs.replace("+", "_plus_").replace("-", "_minus_")
    clean = re.sub(r"[^a-zA-Z0-9_.]+", "_", rhs)
    clean = re.sub(r"_+", "_", clean).strip("_")
    return clean or "custom"


def _prompt_range(label: str, default: tuple[float, float]) -> tuple[float, float]:
    """Lê um intervalo 'min, max' no terminal ou retorna o padrão."""
    raw = input(f"{label} [padrão {default[0]}, {default[1]}]: ").strip()
    if not raw:
        return default

    parts = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError(
            f"Formato inválido para {label}. Use 'min, max' (ex: -30, 30)."
        )

    low, high = float(parts[0]), float(parts[1])
    if low >= high:
        raise ValueError(
            f"O valor mínimo ({low}) deve ser menor que o máximo ({high})."
        )
    return (low, high)


def _prompt_float(label: str, default: float) -> float:
    raw = input(f"{label} [padrão {default}]: ").strip()
    return float(raw) if raw else default


def _prompt_int(label: str, default: int) -> int:
    raw = input(f"{label} [padrão {default}]: ").strip()
    return int(raw) if raw else default


def generate_project_files(
    equation_str: str,
    x_range: tuple[float, float] = (-25.0, 25.0),
    y_range: tuple[float, float] = (-25.0, 25.0),
    num_points: int = 200,
    base_thickness: float = 2.0,
    logo_dxf_path: str | None = "assets/logo.dxf",
    preview: bool = True,
) -> Path:
    func_name, var_names, expr, raw_rhs, _ = parse_equation(equation_str)

    if len(var_names) != 2:
        raise ValueError(
            f"O fluxo STL + Base MDF requer 2 variáveis (ex: f(x, y)). Recebido: {var_names}"
        )

    safe_eq_name = sanitize_filename(equation_str)
    output_dir = Path(f"equations/equation-{safe_eq_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    stl_path = output_dir / f"{func_name}_{safe_eq_name}.stl"
    dxf_path = output_dir / f"base_mdf_{safe_eq_name}.dxf"

    print(f"\n[1/3] Gerando malha 3D em: {stl_path}")
    solid_mesh = export_stl_pyvista(
        expr=expr,
        vars_list=var_names,
        output_path=str(stl_path),
        x_range=x_range,
        y_range=y_range,
        num_points=num_points,
        base_thickness=base_thickness,
    )

    width_mm = float(x_range[1] - x_range[0])
    depth_mm = float(y_range[1] - y_range[0])
    margin_mm = 12.0
    footer_height_mm = 34.0

    var_x, var_y = str(var_names[0]), str(var_names[1])
    ranges_label = (
        f"{var_x}: [{x_range[0]:g}, {x_range[1]:g}]   "
        f"{var_y}: [{y_range[0]:g}, {y_range[1]:g}]"
    )

    valid_logo = (
        logo_dxf_path if (logo_dxf_path and Path(logo_dxf_path).exists()) else None
    )

    print(f"[2/3] Gerando base de corte laser (MDF) em: {dxf_path}")
    export_mdf_base_dxf(
        equation_text=equation_str,
        output_path=str(dxf_path),
        model_size_mm=(width_mm, depth_mm),
        margin_mm=margin_mm,
        footer_height_mm=footer_height_mm,
        corner_radius_mm=4.0,
        ranges_text=ranges_label,
        logo_dxf_path=valid_logo,
        logo_scale=None,
    )

    print(f"[3/3] Concluído! Arquivos salvos em '{output_dir.resolve()}'")
    print(
        f"      Dimensões do modelo: {width_mm:.1f} x {depth_mm:.1f} mm | "
        f"Manifold: {solid_mesh.is_manifold} | Arestas abertas: {solid_mesh.n_open_edges}"
    )

    if preview:
        preview_assembly(
            solid_mesh=solid_mesh,
            dxf_path=str(dxf_path),
            x_range=x_range,
            y_range=y_range,
            margin_mm=margin_mm,
            footer_height_mm=footer_height_mm,
            mdf_thickness=3.0,
        )

    return output_dir


if __name__ == "__main__":
    print("=== math2stl ===")
    equacao = (
        input("Equação [padrão f(x, y) = 4 * sin(x/4) * cos(y/4)]: ").strip()
        or "f(x, y) = 4 * sin(x/4) * cos(y/4)"
    )

    x_lim = _prompt_range("Intervalo X em mm (min, max)", (-20.0, 20.0))
    y_lim = _prompt_range("Intervalo Y em mm (min, max)", (-20.0, 20.0))
    espessura_base = _prompt_float("Espessura mínima da base STL (mm)", 2.0)
    resolucao = _prompt_int("Resolução da grade (pontos por eixo)", 250)

    generate_project_files(
        equation_str=equacao,
        x_range=x_lim,
        y_range=y_lim,
        num_points=resolucao,
        base_thickness=espessura_base,
        logo_dxf_path="assets/logo.dxf",
        preview=True,
    )
