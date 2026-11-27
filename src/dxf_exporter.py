from pathlib import Path
from typing import cast

import numpy as np
import pyvista as pv
from ezdxf import bbox, units
from ezdxf import path as ezdxf_path
from ezdxf.addons import text2path
from ezdxf.addons.importer import Importer
from ezdxf.document import Drawing
from ezdxf.entities import Hatch
from ezdxf.entities.dxfgfx import DXFGraphic
from ezdxf.entities.insert import Insert
from ezdxf.entities.text import Text
from ezdxf.enums import TextEntityAlignment
from ezdxf.filemanagement import new as ezdxf_new
from ezdxf.filemanagement import readfile as ezdxf_readfile
from ezdxf.layouts.layout import Modelspace


def export_mdf_base_dxf(
    equation_text: str,
    output_path: str = "base_mdf.dxf",
    model_size_mm: tuple[float, float] = (100.0, 100.0),
    margin_mm: float = 12.0,
    footer_height_mm: float = 34.0,
    corner_radius_mm: float = 4.0,
    ranges_text: str | None = None,
    logo_dxf_path: str | None = None,
    logo_scale: float | None = None,
    draw_model_guide: bool = True,
) -> None:
    model_w, model_h = model_size_mm
    has_logo = bool(logo_dxf_path and Path(logo_dxf_path).exists())
    has_ranges = bool(ranges_text)

    total_w = model_w + 2 * margin_mm
    total_h = model_h + 2 * margin_mm + footer_height_mm

    doc = ezdxf_new(dxfversion="R2010", setup=True)
    doc.units = units.MM
    msp = doc.modelspace()

    doc.layers.add("CORTE_EXTERNO", color=1)
    doc.layers.add("GRAVACAO_TEXTO", color=5)
    doc.layers.add("GRAVACAO_LOGO", color=3)
    doc.layers.add("GUIA_POSICAO", color=8)

    if "TEXTO_LASER" not in doc.styles:
        doc.styles.add("TEXTO_LASER", font="arial.ttf")

    _add_rounded_rect(
        msp=msp,
        x=0.0,
        y=0.0,
        width=total_w,
        height=total_h,
        radius=corner_radius_mm,
        layer="CORTE_EXTERNO",
    )

    if draw_model_guide:
        _add_rounded_rect(
            msp=msp,
            x=margin_mm,
            y=margin_mm + footer_height_mm,
            width=model_w,
            height=model_h,
            radius=0.0,
            layer="GUIA_POSICAO",
        )

    eq_y, ranges_y, logo_y, max_logo_h = _compute_footer_layout(
        margin_mm=margin_mm,
        footer_height_mm=footer_height_mm,
        has_logo=has_logo,
        has_ranges=has_ranges,
    )

    center_x = total_w / 2.0
    max_text_width = total_w * 0.85

    eq_font_height = min(4.8, max_text_width / max(len(equation_text) * 0.65, 1.0))
    _add_vector_text(
        msp=msp,
        text=equation_text,
        center_xy=(center_x, eq_y),
        height=eq_font_height,
        layer="GRAVACAO_TEXTO",
    )

    if has_ranges and ranges_text is not None:
        range_font_height = min(3.2, max_text_width / max(len(ranges_text) * 0.65, 1.0))
        _add_vector_text(
            msp=msp,
            text=ranges_text,
            center_xy=(center_x, ranges_y),
            height=range_font_height,
            layer="GRAVACAO_TEXTO",
        )

    if has_logo and logo_dxf_path is not None:
        _insert_dxf_logo(
            target_doc=doc,
            logo_path=logo_dxf_path,
            center_xy=(center_x, logo_y),
            max_size_mm=(model_w * 0.80, max_logo_h),
            manual_scale=logo_scale,
            layer="GRAVACAO_LOGO",
        )

    doc.saveas(output_path)


def _compute_footer_layout(
    margin_mm: float,
    footer_height_mm: float,
    has_logo: bool,
    has_ranges: bool,
) -> tuple[float, float, float, float]:
    """Calcula coordenadas Y de equação, intervalos e altura máxima da logo."""
    if has_logo and has_ranges:
        return (
            margin_mm + (footer_height_mm * 0.80),
            margin_mm + (footer_height_mm * 0.56),
            margin_mm + (footer_height_mm * 0.22),
            footer_height_mm * 0.32,
        )
    if has_logo:
        return (
            margin_mm + (footer_height_mm * 0.72),
            0.0,
            margin_mm + (footer_height_mm * 0.26),
            footer_height_mm * 0.38,
        )
    if has_ranges:
        return (
            margin_mm + (footer_height_mm * 0.65),
            margin_mm + (footer_height_mm * 0.32),
            0.0,
            0.0,
        )
    return (margin_mm + (footer_height_mm * 0.50), 0.0, 0.0, 0.0)


def _add_vector_text(
    msp: Modelspace,
    text: str,
    center_xy: tuple[float, float],
    height: float,
    layer: str,
) -> None:
    """Gera contornos fechados (LWPOLYLINE) e preenchimento (HATCH) com ilhas preservadas."""
    temp_text = msp.add_text(
        text,
        height=height,
        dxfattribs={"layer": layer, "style": "TEXTO_LASER"},
    )
    temp_text.set_placement(center_xy, align=TextEntityAlignment.MIDDLE_CENTER)

    created_entities = text2path.explode(
        temp_text,
        kind=text2path.Kind.HATCHES | text2path.Kind.LWPOLYLINES,
        target=msp,
    )
    for ent in created_entities:
        ent.dxf.layer = layer


def _add_rounded_rect(
    msp: Modelspace,
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
    target_doc: Drawing,
    logo_path: str,
    center_xy: tuple[float, float],
    max_size_mm: tuple[float, float],
    manual_scale: float | None,
    layer: str,
) -> None:
    """Importa o DXF da logo, redimensiona para max_size_mm e centraliza."""
    logo_doc = ezdxf_readfile(logo_path)
    block_name = "LOGO_PROJETO"

    if block_name not in target_doc.blocks:
        block = target_doc.blocks.new(name=block_name)
        importer = Importer(logo_doc, target_doc)
        importer.import_entities(logo_doc.modelspace(), target_layout=block)
        importer.finalize()

    ext = bbox.extents(target_doc.blocks[block_name])
    if ext.has_data:
        orig_w = float(ext.size.x)
        orig_h = float(ext.size.y)
        local_center_x = (ext.extmin.x + ext.extmax.x) / 2.0
        local_center_y = (ext.extmin.y + ext.extmax.y) / 2.0

        if manual_scale is not None:
            scale = manual_scale
        else:
            scale_x = max_size_mm[0] / orig_w if orig_w > 0 else 1.0
            scale_y = max_size_mm[1] / orig_h if orig_h > 0 else 1.0
            scale = min(scale_x, scale_y)
    else:
        local_center_x, local_center_y = 0.0, 0.0
        scale = manual_scale if manual_scale is not None else 1.0

    insert_x = center_xy[0] - (local_center_x * scale)
    insert_y = center_xy[1] - (local_center_y * scale)

    target_doc.modelspace().add_blockref(
        name=block_name,
        insert=(insert_x, insert_y),
        dxfattribs={
            "layer": layer,
            "xscale": scale,
            "yscale": scale,
        },
    )


def _extract_entity_paths(entity: DXFGraphic) -> list[ezdxf_path.Path]:
    """Converte uma entidade DXF em uma lista de objetos Path do ezdxf."""
    dxftype = entity.dxftype()
    if dxftype == "TEXT":
        return text2path.make_paths_from_entity(cast(Text, entity))
    if dxftype == "HATCH":
        return list(ezdxf_path.from_hatch(cast(Hatch, entity)))
    if dxftype == "INSERT":
        paths: list[ezdxf_path.Path] = []
        for virtual_ent in cast(Insert, entity).virtual_entities():
            p = ezdxf_path.make_path(virtual_ent)
            paths.extend(p.sub_paths() if p.has_sub_paths else [p])
        return paths

    p = ezdxf_path.make_path(entity)
    return list(p.sub_paths()) if p.has_sub_paths else [p]


def load_dxf_for_preview(
    dxf_path: str,
    mdf_thickness: float = 3.0,
) -> tuple[pv.PolyData | None, pv.PolyData]:
    """Lê o DXF e retorna a placa MDF extrudada e as linhas de gravação."""
    doc = ezdxf_readfile(dxf_path)
    board_mesh: pv.PolyData | None = None
    engraving_lines: list[pv.PolyData] = []

    for entity in doc.modelspace():
        layer = entity.dxf.layer
        for p in _extract_entity_paths(entity):
            vertices = [(v.x, v.y, 0.0) for v in p.flattening(distance=0.05)]
            if len(vertices) < 2:
                continue

            pts = np.array(vertices, dtype=np.float64)
            if layer == "CORTE_EXTERNO" and len(pts) >= 3:
                if np.allclose(pts[0], pts[-1]):
                    pts = pts[:-1]
                poly = pv.PolyData(pts)
                poly.faces = np.hstack([[len(pts)], np.arange(len(pts))])
                board_mesh = poly.triangulate().extrude(
                    (0.0, 0.0, -mdf_thickness), capping=True
                )
            else:
                pts[:, 2] = 0.05
                engraving_lines.append(pv.lines_from_points(pts))

    engravings = pv.merge(engraving_lines) if engraving_lines else pv.PolyData()
    return board_mesh, engravings
