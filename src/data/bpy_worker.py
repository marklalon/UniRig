import argparse
import os
import numpy as np

from .exporter import Exporter


def _to_python(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def run_export_fbx(payload_path: str):
    data = np.load(payload_path, allow_pickle=True)
    exporter = Exporter()

    parents = _to_python(data["parents"])
    names = _to_python(data["names"])

    exporter._export_fbx(
        path=str(data["path"].item() if hasattr(data["path"], "item") else data["path"]),
        vertices=data["vertices"] if "vertices" in data else None,
        joints=data["joints"],
        skin=data["skin"] if "skin" in data else None,
        parents=parents,
        names=names,
        faces=data["faces"] if "faces" in data else None,
        extrude_size=float(data["extrude_size"]),
        group_per_vertex=int(data["group_per_vertex"]),
        add_root=bool(data["add_root"]),
        do_not_normalize=bool(data["do_not_normalize"]),
        use_extrude_bone=bool(data["use_extrude_bone"]),
        use_connect_unique_child=bool(data["use_connect_unique_child"]),
        extrude_from_parent=bool(data["extrude_from_parent"]),
        tails=data["tails"] if "tails" in data else None,
    )


def run_export_render(payload_path: str):
    data = np.load(payload_path, allow_pickle=True)
    exporter = Exporter()
    resolution = tuple(_to_python(data["resolution"]))

    exporter._export_render(
        path=str(data["path"].item() if hasattr(data["path"], "item") else data["path"]),
        vertices=data["vertices"] if "vertices" in data else None,
        faces=data["faces"] if "faces" in data else None,
        bones=data["bones"] if "bones" in data else None,
        resolution=resolution,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", required=True, choices=["export_fbx", "export_render"])
    parser.add_argument("--payload", required=True)
    args = parser.parse_args()

    os.environ["UNIRIG_BPY_WORKER"] = "1"

    if args.op == "export_fbx":
        run_export_fbx(args.payload)
    elif args.op == "export_render":
        run_export_render(args.payload)


if __name__ == "__main__":
    main()
