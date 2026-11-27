from pathlib import Path

import ezdxf
import numpy as np
import pyvista as pv
from ezdxf import path as ezdxf_path
from ezdxf import units
from ezdxf.addons import Importer, text2path
from ezdxf.enums import TextEntityAlignment


def export_mdf_base_dxf(
    equation_text: str,
    output_path: str = "base_mdf.dxf",
    model_size_mm: tuple[float, float] = (100.0, 100.0),
    margin_mm: float = 15.0,
    footer_height_mm: float = 25.0,
    corner_radius_mm: float = 4.0,
    logo_dxf_path: str | None = None,
    logo_scale: float = 1.0,
    draw_model_guide: bool = True,
) -> None:
    """
    Gera um arquivo DXF para corte e gravação a laser de uma base em MDF.

    Camadas (Layers) padrão laser:
    - CORTE_EXTERNO (Vermelho - ACI 1): Contorno da placa de MDF.
    - GRAVACAO_TEXTO (Azul - ACI 5): Equação matemática gravada.
    - GRAVACAO_LOGO (Verde - ACI 3): Logotipo vetorial.
    - GUIA_POSICAO (Cinza - ACI 8): Marcação leve onde o modelo 3D será colado.
    """
    model_w, model_h = model_size_mm
    total_w = model_w + 2 * margin_mm
    total_h = model_h + 2 * margin_mm + footer_height_mm

    doc = ezdxf.new(dxfversion="R2010", setup=True)
    doc.units = units.MM  # Define escala explícita em milímetros
    msp = doc.modelspace()

    # Configuração das camadas (cores padrão reconhecidas por LightBurn / RDWorks / LaserCAD)
    doc.layers.add("CORTE_EXTERNO", color=1)  # Vermelho = Corte
    doc.layers.add("GRAVACAO_TEXTO", color=5)  # Azul = Gravação (Fill/Scan)
    doc.layers.add("GRAVACAO_LOGO", color=3)  # Verde = Gravação Logo
    doc.layers.add("GUIA_POSICAO", color=8)  # Cinza = Marcação fraca (opcional)

    # 1. Contorno externo da base (com ou sem cantos arredondados)
    _add_rounded_rect(
        msp=msp,
        x=0.0,
        y=0.0,
        width=total_w,
        height=total_h,
        radius=corner_radius_mm,
        layer="CORTE_EXTERNO",
    )

    # 2. Guia opcional na área superior onde o modelo 3D (STL) será colado
    model_x0 = margin_mm
    model_y0 = margin_mm + footer_height_mm
    if draw_model_guide:
        _add_rounded_rect(
            msp=msp,
            x=model_x0,
            y=model_y0,
            width=model_w,
            height=model_h,
            radius=0.0,
            layer="GUIA_POSICAO",
        )

    # 3. Texto da equação na área inferior (rodapé)
    footer_center_y = margin_mm + (footer_height_mm / 2.0)
    text_x = margin_mm if logo_dxf_path else (total_w / 2.0)
    alignment = (
        TextEntityAlignment.MIDDLE_LEFT
        if logo_dxf_path
        else TextEntityAlignment.MIDDLE_CENTER
    )

    # Ajusta altura da fonte para caber confortavelmente na largura disponível
    max_text_width = total_w * 0.65 if logo_dxf_path else total_w * 0.85
    font_height = min(5.0, max_text_width / max(len(equation_text) * 0.65, 1.0))

    msp.add_text(
        equation_text,
        height=font_height,
        dxfattribs={"layer": "GRAVACAO_TEXTO"},
    ).set_placement((text_x, footer_center_y), align=alignment)

    # 4. Inserção do logotipo vetorial (.dxf) no canto inferior direito
    if logo_dxf_path and Path(logo_dxf_path).exists():
        _insert_dxf_logo(
            target_doc=doc,
            logo_path=logo_dxf_path,
            insert_xy=(total_w - margin_mm, footer_center_y),
            scale=logo_scale,
            layer="GRAVACAO_LOGO",
        )

    doc.saveas(output_path)


def _add_rounded_rect(
    msp: ezdxf.layouts.Modelspace,
    x: float,
    y: float,
    width: float,
    height: float,
    radius: float,
    layer: str,
) -> None:
    """Desenha um retângulo fechado com cantos arredondados via bulge no LWPOLYLINE."""
    r = min(radius, width / 2.0, height / 2.0)
    if r <= 0:
        pts = [(x, y), (x + width, y), (x + width, y + height), (x, y + height)]
        msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": layer})
        return

    # Bulge = tan(45° / 2) = 0.41421356 para arcos de 90 graus
    b = 0.41421356237309503
    points = [
        (x + r, y, 0, 0, 0),
        (x + width - r, y, 0, 0, b),
        (x + width, y + r, 0, 0, 0),
        (x + width, y + height - r, 0, 0, b),
        (x + width - r, y + height, 0, 0, 0),
        (x + r, y + height, 0, 0, b),
        (x, y + height - r, 0, 0, 0),
        (x, y + r, 0, 0, b),
    ]
    msp.add_lwpolyline(points, format="xyseb", close=True, dxfattribs={"layer": layer})


def _insert_dxf_logo(
    target_doc: ezdxf.document.Drawing,
    logo_path: str,
    insert_xy: tuple[float, float],
    scale: float,
    layer: str,
) -> None:
    """Importa entidades de um arquivo DXF de logotipo como um bloco posicionado."""
    logo_doc = ezdxf.readfile(logo_path)
    block_name = "LOGO_PROJETO"
    if block_name not in target_doc.blocks:
        block = target_doc.blocks.new(name=block_name)
        importer = Importer(logo_doc, target_doc)
        importer.import_entities(logo_doc.modelspace(), target_layout=block)
        importer.finalize()

    target_doc.modelspace().add_blockref(
        name=block_name,
        insert=insert_xy,
        dxfattribs={
            "layer": layer,
            "xscale": scale,
            "yscale": scale,
        },
    )


def load_dxf_for_preview(
    dxf_path: str,
    mdf_thickness: float = 3.0,
) -> tuple[pv.PolyData | None, pv.PolyData]:
    """
    Lê um arquivo DXF gerado pelo projeto e retorna:
    1. A placa de MDF extrudada (sólido 3D com topo em Z = 0).
    2. As linhas de gravação (texto, logo e guias) posicionadas em Z = 0.05.
    """
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    board_mesh: pv.PolyData | None = None
    engraving_lines: list[pv.PolyData] = []

    for entity in msp:
        layer = entity.dxf.layer
        dxftype = entity.dxftype()
        paths: list[ezdxf_path.Path] = []

        # 1. Converte a entidade DXF conforme o tipo
        if dxftype == "TEXT":
            paths = text2path.make_paths_from_entity(entity)
        elif dxftype == "INSERT":
            # Explode temporariamente os itens do bloco da logo
            for virtual_ent in entity.virtual_entities():
                p = ezdxf_path.make_path(virtual_ent)
                paths.extend(p.sub_paths() if p.has_sub_paths else [p])
        else:
            p = ezdxf_path.make_path(entity)
            paths = list(p.sub_paths()) if p.has_sub_paths else [p]

        # 2. Converte cada Path em geometria 3D do PyVista
        for p in paths:
            vertices = [(v.x, v.y, 0.0) for v in p.flattening(distance=0.1)]
            if len(vertices) < 2:
                continue

            pts = np.array(vertices, dtype=np.float32)

            if layer == "CORTE_EXTERNO" and len(pts) >= 3:
                # Remove último ponto se for idêntico ao primeiro para evitar face degenerada
                if np.allclose(pts[0], pts[-1]):
                    pts = pts[:-1]
                poly = pv.PolyData(pts)
                poly.faces = np.hstack([[len(pts)], np.arange(len(pts))])
                board_mesh = poly.triangulate().extrude(
                    (0.0, 0.0, -mdf_thickness), capping=True
                )
            else:
                pts[:, 2] = 0.05
                line = pv.lines_from_points(pts)
                engraving_lines.append(line)

    engravings = pv.merge(engraving_lines) if engraving_lines else pv.PolyData()
    return board_mesh, engravings
