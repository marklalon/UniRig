import os
import tempfile
import numpy as np

from ..bpy_env_wrapper import run_module_in_bpy_env


def _run_worker(op: str, payload: dict):
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        payload_path = tmp.name
    try:
        np.savez(payload_path, **payload)
        code = run_module_in_bpy_env(
            module="src.data.bpy_worker",
            module_args=["--op", op, "--payload", payload_path],
            extra_env={"UNIRIG_BPY_WORKER": "1"},
        )
        if code != 0:
            raise RuntimeError(f"bpy worker failed with exit code {code} for op={op}")
    finally:
        if os.path.exists(payload_path):
            os.remove(payload_path)


def export_fbx_via_bpy_env(
    path,
    vertices,
    joints,
    skin,
    parents,
    names,
    faces,
    extrude_size,
    group_per_vertex,
    add_root,
    do_not_normalize,
    use_extrude_bone,
    use_connect_unique_child,
    extrude_from_parent,
    tails,
):
    payload = {
        "path": path,
        "joints": joints,
        "parents": np.array(parents, dtype=object),
        "names": np.array(names, dtype=object),
        "extrude_size": extrude_size,
        "group_per_vertex": group_per_vertex,
        "add_root": add_root,
        "do_not_normalize": do_not_normalize,
        "use_extrude_bone": use_extrude_bone,
        "use_connect_unique_child": use_connect_unique_child,
        "extrude_from_parent": extrude_from_parent,
    }
    if vertices is not None:
        payload["vertices"] = vertices
    if skin is not None:
        payload["skin"] = skin
    if faces is not None:
        payload["faces"] = faces
    if tails is not None:
        payload["tails"] = tails

    _run_worker("export_fbx", payload)


def export_render_via_bpy_env(path, vertices, faces, bones, resolution):
    payload = {
        "path": path,
        "resolution": np.array(resolution),
    }
    if vertices is not None:
        payload["vertices"] = vertices
    if faces is not None:
        payload["faces"] = faces
    if bones is not None:
        payload["bones"] = bones
    _run_worker("export_render", payload)
