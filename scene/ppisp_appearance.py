"""GGGS 的 PPISP 接口：帧身份、CHW/HWC 转换及独立优化状态。

camera_id 是 COLMAP 内参组，frame_idx 是训练 Camera.uid；不能把每个 pose
当作一台设备。首版固定关闭 controller，保持整段训练的 RGB→几何梯度。
"""

from dataclasses import asdict


PPISP_COMMIT = "a247368047651db2e03e6f97a36e4f5bebbe5c5f"


class PPISPAppearance:
    def __init__(self, train_cameras, device="cuda"):
        # 仅模式 4 加载 CUDA 扩展；模式 0–3 不依赖 PPISP 是否安装。
        try:
            import ppisp
        except ImportError as error:
            raise ImportError(
                "Appearance mode 4 requires PPISP. Run git submodule update --init "
                "submodules/ppisp, then pip install -e submodules/ppisp --no-build-isolation."
            ) from error

        cameras = sorted(train_cameras, key=lambda camera: camera.uid)
        if not cameras or [camera.uid for camera in cameras] != list(range(len(cameras))):
            raise ValueError("PPISP requires contiguous training Camera.uid values starting at zero")
        self.frame_keys = [
            getattr(camera, "source_image_name", None) or camera.image_name
            for camera in cameras
        ]
        if len(set(self.frame_keys)) != len(self.frame_keys):
            raise ValueError("PPISP training frame names must be unique")
        self.frame_index_by_name = {name: index for index, name in enumerate(self.frame_keys)}
        self.camera_ids = sorted({int(camera.colmap_id) for camera in cameras})
        self.camera_index_by_id = {key: index for index, key in enumerate(self.camera_ids)}
        self.frame_to_camera = [self.camera_index_by_id[int(camera.colmap_id)] for camera in cameras]

        self.config = ppisp.PPISPConfig(use_controller=False, controller_distillation=False)
        self.backend_version = getattr(ppisp, "__version__", None)
        self.model = ppisp.PPISP(
            num_cameras=len(self.camera_ids), num_frames=len(cameras), config=self.config
        ).to(device)
        self.optimizers = []
        self.schedulers = []
        self.completed_updates = 0

    def setup_optimizers(self, iterations):
        """使用官方优化器/调度，不混入 Gaussian 的 32/64 步累积日程。"""
        self.optimizers = self.model.create_optimizers()
        self.schedulers = self.model.create_schedulers(self.optimizers, iterations)
        self.completed_updates = 0

    def apply(self, image, frame_idx, camera_id=None):
        """已知训练帧用 learned ISP；frame=-1 仅用已知镜头的暗角/CRF。

        新视角不借用训练帧 uid 或根据 GT 拟合曝光。controller 已关闭，-1 的
        曝光/色彩为单位变换；这不是完全绕过 ISP，也不是完整新视角曝光预测。
        """
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError("PPISP expects a CHW RGB image")
        if frame_idx == -1:
            if camera_id is None or int(camera_id) not in self.camera_index_by_id:
                raise ValueError("PPISP novel views require a known COLMAP camera_id")
            camera_idx = self.camera_index_by_id[int(camera_id)]
        elif 0 <= frame_idx < len(self.frame_keys):
            camera_idx = self.frame_to_camera[frame_idx]
            if camera_id is not None and int(camera_id) != self.camera_ids[camera_idx]:
                raise ValueError("PPISP frame and camera identities disagree")
        else:
            raise ValueError(f"Unknown PPISP frame index: {frame_idx}")

        height, width = image.shape[1:]
        corrected = self.model(
            image.permute(1, 2, 0).contiguous(),
            pixel_coords=None,  # 官方实现使用 (x+.5, y+.5)，与整幅缩放后的像素网格一致。
            resolution=(width, height),
            camera_idx=camera_idx,
            frame_idx=frame_idx,
        )
        return corrected.permute(2, 0, 1).contiguous()

    def regularization_loss(self):
        """保留 ISP 参数的计算图；由训练器每次真实更新加入一次。"""
        return self.model.get_regularization_loss()

    def step(self):
        if not self.optimizers:
            raise RuntimeError("Call setup_optimizers before stepping PPISP")
        for optimizer in self.optimizers:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        for scheduler in self.schedulers:
            scheduler.step()
        self.completed_updates += 1

    def export_state(self):
        """与原生 PLY 配对保存 ISP；非线性/像素相关处理不能烘焙成一组 SH。"""
        return {
            "schema_version": 1,
            "dependency_commit": PPISP_COMMIT,
            "backend_version": self.backend_version,
            "config": asdict(self.config),
            "frame_keys": list(self.frame_keys),
            "camera_ids": list(self.camera_ids),
            "frame_to_camera": list(self.frame_to_camera),
            "model_state": self.model.state_dict(),
            "completed_updates": self.completed_updates,
        }

    def _validate_state(self, state):
        if state.get("schema_version") != 1 or state.get("dependency_commit") != PPISP_COMMIT:
            raise ValueError("Unsupported PPISP checkpoint version")
        for name in ("frame_keys", "camera_ids", "frame_to_camera"):
            if state.get(name) != getattr(self, name):
                raise ValueError(f"PPISP checkpoint identity changed: {name}")
        if state.get("config") != asdict(self.config):
            raise ValueError("PPISP checkpoint configuration differs from mode 4")
        if state.get("backend_version") != self.backend_version:
            raise ValueError("PPISP package version differs from the checkpoint")

    def load_export_state(self, state):
        self._validate_state(state)
        self.model.load_state_dict(state["model_state"])
        self.completed_updates = state["completed_updates"]

    def capture(self):
        if not self.optimizers:
            raise RuntimeError("PPISP optimizer state is unavailable for checkpointing")
        return {
            **self.export_state(),
            "optimizer_states": [optimizer.state_dict() for optimizer in self.optimizers],
            "scheduler_states": [scheduler.state_dict() for scheduler in self.schedulers],
        }

    def restore(self, state):
        if not self.optimizers:
            raise RuntimeError("Call setup_optimizers before restoring PPISP")
        self._validate_state(state)
        if len(state["optimizer_states"]) != len(self.optimizers) or len(state["scheduler_states"]) != len(self.schedulers):
            raise ValueError("PPISP optimizer/scheduler count differs from the checkpoint")
        self.load_export_state(state)
        for optimizer, saved in zip(self.optimizers, state["optimizer_states"]):
            optimizer.load_state_dict(saved)
        for scheduler, saved in zip(self.schedulers, state["scheduler_states"]):
            scheduler.load_state_dict(saved)
