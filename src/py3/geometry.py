"""Rotation conventions.

Single source of truth for how orientations are represented.

Conventions
-----------
* i3 ``.trf`` matrix ``M`` (row-major m11..m33) maps a direction in the
  *tomogram* frame onto the *reference* frame: ``M @ v_tomo = v_ref``.
  This is what ``pos2i3.py`` writes (membrane normal -> +z).
* RELION Euler angles (rot, tilt, psi) are ZYZ. RELION builds
  ``A = Euler_angles2matrix(rot, tilt, psi)`` and the particle is the
  reference actively rotated by ``A``. Hence the rotation that brings a
  particle onto the reference is ``A.T``.
* ``euler_to_matrix`` returns the intrinsic-ZYZ active matrix
  ``Rz(rot) @ Ry(tilt) @ Rz(psi)`` (identical to scipy
  ``Rotation.from_euler('ZYZ', ...)``), which equals RELION's ``A.T``.

Therefore ``M_i3 == euler_to_matrix(rot, tilt, psi)``, which is exactly what
the legacy ``trf2star.py`` did via ``scipy ... as_euler('ZYZ')``.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-8


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    z, o = np.zeros_like(a), np.ones_like(a)
    return np.stack([np.stack([c, -s, z], -1),
                     np.stack([s, c, z], -1),
                     np.stack([z, z, o], -1)], -2)


def _ry(a):
    c, s = np.cos(a), np.sin(a)
    z, o = np.zeros_like(a), np.ones_like(a)
    return np.stack([np.stack([c, z, s], -1),
                     np.stack([z, o, z], -1),
                     np.stack([-s, z, c], -1)], -2)


def euler_to_matrix(rot, tilt, psi) -> np.ndarray:
    """RELION (rot, tilt, psi) in degrees -> (..., 3, 3) matrix M (tomo -> reference)."""
    rot, tilt, psi = (np.deg2rad(np.asarray(a, dtype=float)) for a in (rot, tilt, psi))
    return _rz(rot) @ _ry(tilt) @ _rz(psi)


def matrix_to_euler(m) -> np.ndarray:
    """(..., 3, 3) matrix -> (..., 3) RELION (rot, tilt, psi) in degrees.

    tilt in [0, 180]; rot, psi in (-180, 180]. At gimbal lock psi is set to 0.
    """
    m = np.asarray(m, dtype=float)
    tilt = np.arccos(np.clip(m[..., 2, 2], -1.0, 1.0))
    sb = np.sin(tilt)
    rot = np.arctan2(m[..., 1, 2], m[..., 0, 2])
    psi = np.arctan2(m[..., 2, 1], -m[..., 2, 0])
    lock = sb < _EPS
    if np.any(lock):
        up = lock & (m[..., 2, 2] > 0)
        down = lock & (m[..., 2, 2] <= 0)
        rot = np.where(up, np.arctan2(m[..., 1, 0], m[..., 0, 0]), rot)
        rot = np.where(down, np.arctan2(-m[..., 1, 0], -m[..., 0, 0]), rot)
        psi = np.where(lock, 0.0, psi)
    return np.rad2deg(np.stack([rot, tilt, psi], -1))


def relion_A(rot, tilt, psi) -> np.ndarray:
    """RELION's own Euler_angles2matrix, transcribed literally (used for testing)."""
    a, b, g = (np.deg2rad(float(x)) for x in (rot, tilt, psi))
    ca, sa, cb, sb, cg, sg = np.cos(a), np.sin(a), np.cos(b), np.sin(b), np.cos(g), np.sin(g)
    cc, cs, sc, ss = cb * ca, cb * sa, sb * ca, sb * sa
    return np.array([[cg * cc - sg * sa, cg * cs + sg * ca, -cg * sb],
                     [-sg * cc - cg * sa, -sg * cs + cg * ca, sg * sb],
                     [sc, ss, cb]])


def is_rotation(m, tol=1e-4) -> np.ndarray:
    m = np.asarray(m, dtype=float)
    eye = np.eye(3)
    ortho = np.abs(np.swapaxes(m, -1, -2) @ m - eye).max(axis=(-1, -2)) < tol
    return ortho & (np.abs(np.linalg.det(m) - 1) < tol)


def frame_from_normal(normal, inplane_deg=0.0) -> np.ndarray:
    """Matrix M with M @ n = +z (membrane normal along reference z).

    Matches pos2i3.py: the in-plane reference is n x z mapped to +y, then an
    optional extra rotation of ``inplane_deg`` about the reference z axis.
    """
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n, axis=-1, keepdims=True)
    y = np.cross(n, [0.0, 0.0, 1.0])
    ylen = np.linalg.norm(y, axis=-1, keepdims=True)
    # normal parallel to z: in-plane direction is arbitrary; use x
    fallback = np.broadcast_to([1.0, 0.0, 0.0], y.shape)
    y = np.where(ylen > _EPS, y / np.maximum(ylen, _EPS), np.cross(n, fallback))
    y = y / np.linalg.norm(y, axis=-1, keepdims=True)
    x = np.cross(y, n)
    m = np.stack([x, y, n], axis=-2)  # rows
    return _rz(np.deg2rad(np.asarray(inplane_deg, dtype=float))) @ m
