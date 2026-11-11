from scipy.spatial.transform import Rotation
import numpy as np


def R_to_quat(R: np.ndarray) -> np.ndarray:
    rot = Rotation.from_matrix(R)
    return rot.as_quat()

def quat_to_R(quat: np.ndarray) -> np.ndarray:
    rot = Rotation.from_quat(quat)
    return rot.as_matrix()

def rot6d_to_R(r6: np.ndarray) -> np.ndarray:
    """
    Convert 6D rotation representation to a 3x3 rotation matrix.
    r6: shape (6,)
    """
    a1 = r6[:3]  # first column
    a2 = r6[3:]  # second column
    b1 = a1 / np.linalg.norm(a1)
    # remove projection of a2 onto b1
    b2 = a2 - np.dot(b1, a2) * b1
    b2 /= np.linalg.norm(b2)
    b3 = np.cross(b1, b2)

    return np.stack([b1, b2, b3], axis=1)

def R_to_rot6d(R: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to 6D rotation representation.
    """
    return np.concatenate([R[:, 0], R[:, 1]])

def quat_to_rot6d(quat: np.ndarray) -> np.ndarray:
    R = quat_to_R(quat)
    return R_to_rot6d(R)

def rot6d_to_quat(r6: np.ndarray) -> np.ndarray:
    R = rot6d_to_R(r6)
    return R_to_quat(R)
