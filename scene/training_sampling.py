"""Camera schedules whose randomness is independent of initialization and CUDA."""

import hashlib
from random import Random


def _camera_name(camera):
    return getattr(camera, "source_image_name", None) or camera.image_name


class CameraSamplingSchedule:
    def __init__(self, cameras, seed):
        self.cameras = list(cameras)
        self.ordered = sorted(self.cameras, key=_camera_name)
        if len({_camera_name(camera) for camera in self.ordered}) != len(self.ordered) or not self.ordered:
            raise ValueError("Camera sampling requires unique, nonempty real-camera names")
        self.real_rng = Random(seed)
        self.neighbor_rng = Random(f"{seed}:real_neighbors")
        self.stack = []
        self.real_hash = hashlib.sha256()
        self.neighbor_hash = hashlib.sha256()
        self.real_count = self.neighbor_count = 0

    def draw_real(self):
        if not self.stack:
            self.stack = self.ordered.copy()
        camera = self.stack.pop(self.real_rng.randrange(len(self.stack)))
        self.real_hash.update((_camera_name(camera) + "\n").encode())
        self.real_count += 1
        return camera

    def draw_neighbor(self, camera):
        neighbors = camera.nearest_id
        neighbor = None if not neighbors else self.cameras[neighbors[self.neighbor_rng.randrange(len(neighbors))]]
        self.neighbor_hash.update((_camera_name(camera) + "\0" + ("" if neighbor is None else _camera_name(neighbor)) + "\n").encode())
        self.neighbor_count += 1
        return neighbor

    def proof(self):
        return {
            "camera_sampling_policy": "sorted_names_private_seeded_cycle_v1",
            "real_camera_order_sha256": self.real_hash.hexdigest(),
            "real_camera_order_count": self.real_count,
            "neighbor_camera_order_sha256": self.neighbor_hash.hexdigest(),
            "neighbor_camera_order_count": self.neighbor_count,
        }
