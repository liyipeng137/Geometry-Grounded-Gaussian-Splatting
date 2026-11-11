"""Initialization selection and frozen-input reuse; independent of Gaussian training."""

import hashlib
from pathlib import Path


def validate_initialization_options(args):
    """A mesh path enables sampling; an external PLY already contains XYZ/RGB."""
    if getattr(args, "init_mesh", "") and getattr(args, "init_ply", ""):
        raise ValueError("--init_mesh and --init_ply are mutually exclusive")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_checkpoint_initialization(args, checkpoint_path, split):
    """Reuse the checkpoint's frozen colored PLY without reading its source mesh.

    input.ply is the run-owned snapshot. The original initialization path is a
    fallback for older runs. An explicit replacement must have identical bytes;
    silently generating a new initial cloud would make the saved proof false.
    """
    validate_initialization_options(args)
    initial = split["initialization"]
    mesh = getattr(args, "init_mesh", "")
    if mesh:
        if (initial.get("requested_source") != "mesh"
                or Path(mesh).expanduser().resolve() != Path(initial["mesh_path"]).resolve()):
            raise ValueError("Resume cannot change initialization mesh; omit --init_mesh to reuse the checkpoint input")
    explicit = getattr(args, "init_ply", "")
    candidates = ([Path(explicit).expanduser().resolve()] if explicit else
                  [Path(checkpoint_path).resolve().parent / "input.ply", Path(initial["ply_path"])])
    for path in candidates:
        if path.is_file():
            if _sha256(path) != initial["ply_sha256"]:
                raise ValueError(f"Checkpoint initialization PLY content changed: {path}")
            return str(path.resolve())
    raise FileNotFoundError("Checkpoint initialization PLY is missing; restore input.ply or pass the same --init_ply")
