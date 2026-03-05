#!/usr/bin/env python3
"""
build_arm_xml.py — extract a kinematic chain from a MuJoCo XML.

Extracts a named body and all its descendants from a source MuJoCo XML,
producing a self-contained standalone model with:
  - Only the assets (meshes, materials, textures) referenced by the chain
  - meshdir rewritten to an absolute path so meshes resolve correctly
  - Default classes preserved
  - A fixed world mount body at a configurable position / orientation

Robot-agnostic: works with any MuJoCo XML and any body name.

CLI usage
---------
    python examples/build_arm_xml.py \\
        --source /path/to/robot.xml \\
        --body   shoulder_link \\
        --output /path/to/output.xml \\
        [--mount-pos "0 0.2 0.85"] \\
        [--mount-quat "1 0 0 0"]   \\
        [--model-name my_arm]

API usage
---------
    from examples.build_arm_xml import extract_arm
    extract_arm(
        source="/path/to/robot.xml",
        body_name="shoulder_link",
        output="/path/to/output.xml",
        mount_pos="0 0.2 0.85",
    )
"""

from __future__ import annotations

import argparse
import copy
import xml.etree.ElementTree as ET
from pathlib import Path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _mesh_name(elem: ET.Element) -> str:
    """Return the effective name of a <mesh> element.

    MuJoCo defaults the mesh name to the stem of its file attribute when no
    explicit name is provided.
    """
    explicit = elem.get("name")
    if explicit:
        return explicit
    file_attr = elem.get("file", "")
    return Path(file_attr).stem


def _find_body(root: ET.Element, name: str) -> ET.Element | None:
    """Depth-first search for a <body name=…> element anywhere under root."""
    if root.tag == "body" and root.get("name") == name:
        return root
    for child in root:
        result = _find_body(child, name)
        if result is not None:
            return result
    return None


def _collect_used_assets(body: ET.Element) -> tuple[set[str], set[str], set[str]]:
    """Walk a body subtree and collect names of referenced assets.

    Returns (mesh_names, material_names, texture_names).
    """
    meshes: set[str] = set()
    materials: set[str] = set()
    textures: set[str] = set()

    def walk(elem: ET.Element) -> None:
        if elem.tag == "geom":
            m = elem.get("mesh")
            if m:
                meshes.add(m)
            mat = elem.get("material")
            if mat:
                materials.add(mat)
        for child in elem:
            walk(child)

    walk(body)
    return meshes, materials, textures


def _indent(elem: ET.Element, level: int = 0, indent: str = "  ") -> None:
    """In-place pretty-print indentation (works on Python 3.8 and 3.9+)."""
    pad = "\n" + indent * level
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = pad + indent
        if not elem.tail or not elem.tail.strip():
            elem.tail = pad
        for child in elem:
            _indent(child, level + 1, indent)
        # Last child's tail should close the parent
        if not child.tail or not child.tail.strip():
            child.tail = pad
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = pad


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _prune_bodies(elem: ET.Element, exclude: set[str]) -> None:
    """Remove any child <body> whose name is in *exclude* (in-place)."""
    to_remove = [
        child for child in elem
        if child.tag == "body" and child.get("name") in exclude
    ]
    for child in to_remove:
        elem.remove(child)
    for child in elem:
        _prune_bodies(child, exclude)


def extract_arm(
    source: str | Path,
    body_name: str,
    output: str | Path,
    mount_pos: str = "0 0 0",
    mount_quat: str = "1 0 0 0",
    model_name: str | None = None,
    exclude_bodies: list[str] | None = None,
) -> None:
    """Extract *body_name* and its descendants into a standalone MuJoCo XML.

    Parameters
    ----------
    source:
        Path to the source MuJoCo XML (may reference meshes via a relative
        ``meshdir``; this will be rewritten to an absolute path).
    body_name:
        Name of the root body of the kinematic chain to extract (e.g.
        ``"left_ur_arm_shoulder_link"``).
    output:
        Destination path for the generated XML.
    mount_pos:
        World-frame position of the mount body as a space-separated string
        (default ``"0 0 0"``).
    mount_quat:
        World-frame orientation of the mount body as ``"w x y z"``
        (default ``"1 0 0 0"``).
    model_name:
        Name for the ``<mujoco model=…>`` attribute.  Defaults to body_name.
    exclude_bodies:
        Optional list of body names to prune from the extracted subtree
        (e.g. finger links, sensor mounts).  All descendants are also removed.
    """
    exclude_bodies = set(exclude_bodies or [])
    source = Path(source).resolve()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model_name = model_name or body_name

    # ── Parse source ────────────────────────────────────────────────────────
    tree = ET.parse(source)
    src_root = tree.getroot()  # <mujoco>

    # ── Resolve meshdir to absolute path ────────────────────────────────────
    compiler_src = src_root.find("compiler")
    meshdir_abs = ""
    if compiler_src is not None:
        meshdir_rel = compiler_src.get("meshdir", "")
        if meshdir_rel:
            candidate = source.parent / meshdir_rel
            meshdir_abs = str(candidate.resolve())
        # Preserve other compiler attrs (angle, etc.)
        compiler_attrs = dict(compiler_src.attrib)
    else:
        compiler_attrs = {}

    compiler_attrs["angle"] = compiler_attrs.get("angle", "radian")
    compiler_attrs["fusestatic"] = "false"
    if meshdir_abs:
        compiler_attrs["meshdir"] = meshdir_abs
    elif "meshdir" in compiler_attrs:
        # make whatever was there absolute
        compiler_attrs["meshdir"] = str((source.parent / compiler_attrs["meshdir"]).resolve())

    # ── Find the target body ─────────────────────────────────────────────────
    worldbody_src = src_root.find("worldbody")
    if worldbody_src is None:
        raise ValueError(f"No <worldbody> found in {source}")

    arm_body = _find_body(worldbody_src, body_name)
    if arm_body is None:
        raise ValueError(
            f"Body '{body_name}' not found in {source}.\n"
            f"Available top-level bodies: "
            + ", ".join(b.get("name", "?") for b in worldbody_src if b.tag == "body")
        )

    # Deep-copy so we don't mutate the original tree
    arm_body = copy.deepcopy(arm_body)

    # Prune unwanted sub-chains (fingers, sensors, etc.)
    if exclude_bodies:
        _prune_bodies(arm_body, exclude_bodies)
        print(f"  pruned bodies: {sorted(exclude_bodies)}")

    # ── Collect referenced assets ────────────────────────────────────────────
    used_meshes, used_materials, used_textures = _collect_used_assets(arm_body)

    # Build name-keyed lookup tables from source <asset>
    src_asset = src_root.find("asset")
    mesh_map: dict[str, ET.Element] = {}
    material_map: dict[str, ET.Element] = {}
    texture_map: dict[str, ET.Element] = {}

    if src_asset is not None:
        for elem in src_asset:
            if elem.tag == "mesh":
                mesh_map[_mesh_name(elem)] = elem
            elif elem.tag == "material":
                name = elem.get("name", "")
                if name:
                    material_map[name] = elem
            elif elem.tag == "texture":
                name = elem.get("name", "")
                if name:
                    texture_map[name] = elem

    # ── Also collect materials referenced by materials (via texture attr) ───
    for mat_name in list(used_materials):
        mat_elem = material_map.get(mat_name)
        if mat_elem is not None:
            tex = mat_elem.get("texture")
            if tex:
                used_textures.add(tex)

    # ── Build output XML ─────────────────────────────────────────────────────
    out_root = ET.Element("mujoco", attrib={"model": model_name})

    # compiler
    ET.SubElement(out_root, "compiler", attrib=compiler_attrs)

    # option
    option_src = src_root.find("option")
    if option_src is not None:
        out_root.append(copy.deepcopy(option_src))
    else:
        ET.SubElement(out_root, "option", attrib={"gravity": "0 0 -9.81", "timestep": "0.001"})

    # default (preserve all classes)
    default_src = src_root.find("default")
    if default_src is not None:
        out_root.append(copy.deepcopy(default_src))

    # asset — include only what the arm actually references
    if used_meshes or used_materials or used_textures:
        asset_out = ET.SubElement(out_root, "asset")
        for tex_name in sorted(used_textures):
            if tex_name in texture_map:
                asset_out.append(copy.deepcopy(texture_map[tex_name]))
        for mat_name in sorted(used_materials):
            if mat_name in material_map:
                asset_out.append(copy.deepcopy(material_map[mat_name]))
        for mesh_name in sorted(used_meshes):
            if mesh_name in mesh_map:
                asset_out.append(copy.deepcopy(mesh_map[mesh_name]))
            else:
                print(f"  [warn] mesh '{mesh_name}' referenced but not found in <asset>")

    # worldbody → mount → arm chain
    worldbody_out = ET.SubElement(out_root, "worldbody")
    mount = ET.SubElement(
        worldbody_out,
        "body",
        attrib={"name": f"{body_name}_mount", "pos": mount_pos, "quat": mount_quat},
    )
    mount.append(arm_body)

    # ── Write ────────────────────────────────────────────────────────────────
    _indent(out_root)
    ET.ElementTree(out_root).write(
        str(output), xml_declaration=True, encoding="unicode"
    )
    print(f"Extracted '{body_name}' → {output}")
    print(f"  meshes   : {len(used_meshes)}")
    print(f"  materials: {len(used_materials)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract a kinematic chain from a MuJoCo XML into a standalone file."
    )
    p.add_argument("--source",     required=True, help="Source MuJoCo XML path")
    p.add_argument("--body",       required=True, help="Root body name to extract")
    p.add_argument("--output",     required=True, help="Output XML path")
    p.add_argument("--mount-pos",  default="0 0 0",   help='Mount position "x y z"')
    p.add_argument("--mount-quat", default="1 0 0 0", help='Mount orientation "w x y z"')
    p.add_argument("--model-name", default=None,      help="Model name (defaults to body name)")
    p.add_argument(
        "--exclude-bodies", nargs="*", default=[],
        metavar="BODY",
        help="Body names to prune from the extracted chain (e.g. finger links)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    extract_arm(
        source=args.source,
        body_name=args.body,
        output=args.output,
        mount_pos=args.mount_pos,
        mount_quat=args.mount_quat,
        model_name=args.model_name,
        exclude_bodies=args.exclude_bodies,
    )
