"""Fail if the current environment is exposed to the eMudhra / emSign CA
incident (advisory 2026-10-20, https://discuss.python.org/t/109576).

Stdlib only: run with ``python -I scripts/check_ca_security.py``.
Exit code 1 when a hard requirement is not met.
"""

import importlib
import importlib.metadata
import re
import shutil
import subprocess
import sys

MIN_CPYTHON = {
    (3, 11): (3, 11, 18),
    (3, 12): (3, 12, 16),
    (3, 13): (3, 13, 18),
    (3, 14): (3, 14, 10),
    (3, 15): (3, 15, 2),
}
MIN_PIP = (26, 2, 2)
MIN_CERTIFI = (2026, 10, 19)
MIN_UV = (0, 13, 4)


def parse(version: str) -> tuple:
    return tuple(int(p) for p in re.findall(r"\d+", version)[:3])


def fmt(version: tuple) -> str:
    return ".".join(str(p) for p in version)


def dist_version(name: str):
    # Prefer the imported module: stale *.dist-info leftovers can make
    # importlib.metadata report an older version than the one in use.
    try:
        return importlib.import_module(name).__version__
    except (ImportError, AttributeError):
        pass
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> int:
    errors = []
    warnings = []

    current = sys.version_info[:3]
    floor = MIN_CPYTHON.get(current[:2])
    if floor is None:
        errors.append(
            f"Python {fmt(current)} is not a supported branch "
            f"(EOL or unknown): no CA fix available"
        )
    elif current < floor:
        errors.append(f"Python {fmt(current)} < {fmt(floor)}")

    pip_version = dist_version("pip")
    if pip_version and parse(pip_version) < MIN_PIP:
        errors.append(f"pip {pip_version} < {fmt(MIN_PIP)}")

    certifi_version = dist_version("certifi")
    if certifi_version is None:
        errors.append("certifi not installed")
    else:
        # Version floor is the advisory criterion: upstream certifi removes
        # emSign roots incrementally, so the bundle is not grepped here.
        if parse(certifi_version) < MIN_CERTIFI:
            errors.append(f"certifi {certifi_version} < {fmt(MIN_CERTIFI)}")

    uv = shutil.which("uv")
    if uv:
        out = subprocess.run(
            [uv, "--version"], capture_output=True, text=True, check=False
        ).stdout
        uv_version = parse(out)
        if uv_version and uv_version < MIN_UV:
            warnings.append(
                f"uv {fmt(uv_version)} < {fmt(MIN_UV)} on PATH ({uv}): "
                f"run 'UV_SYSTEM_CERTS=true uv self update'"
            )

    print(f"python={fmt(current)} pip={pip_version} certifi={certifi_version}")
    for msg in warnings:
        print(f"WARNING: {msg}")
    for msg in errors:
        print(f"ERROR: {msg}")
    if errors:
        print("CA security check FAILED")
        return 1
    print("CA security check OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
