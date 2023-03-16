import argparse

import open3d as o3d
from tqdm import tqdm

from common import *
import pyngp as ngp  # noqa


class Ingp:
    def __init__(self):
        self.testbed = ngp.Testbed()
        self.testbed.root_dir = ROOT_DIR
        self.testbed.nerf.sharpen = 0.0
        self.testbed.exposure = 0.0

    def load_scene(self, scene_dpath: Path, config_name: str = "base"):
        configs_dir = os.path.join(ROOT_DIR, "configs", "nerf")
        network = os.path.join(configs_dir, f"{config_name}.json")
        if not os.path.isabs(network):
            network = os.path.join(configs_dir, network)

        self.testbed.load_training_data(str(scene_dpath))
        self.testbed.reload_network_from_file(network)

        self.testbed.shall_train = True
        self.testbed.nerf.render_with_camera_distortion = True

    def train(self, n_steps: int, train_cam_poses: bool = False):
        self.testbed.nerf.training.optimize_extrinsics = train_cam_poses

        pbar = tqdm(range(n_steps), desc="Training with fixed poses")
        for _ in pbar:
            if self.testbed.frame():
                pbar.set_postfix(loss=self.testbed.loss)
            else:
                pbar.close()
                break

    def save_snapshot(self, fpath: Path):
        self.testbed.save_snapshot(str(fpath))

    def export_mesh(self, fpath: Path, resolution: int):
        self.testbed.compute_and_save_marching_cubes_mesh(
            str(fpath), (resolution,)*3)
        mesh = o3d.io.read_triangle_mesh(str(fpath))
        rotmat = np.array([
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
        ])
        mesh.rotate(rotmat, center=(0, 0, 0))
        o3d.io.write_triangle_mesh(str(fpath), mesh, compressed=True)

    def export_point_cloud(self, fpath: Path, resolution: int, mesh_fpath: Path | None = None):
        if mesh_fpath is None:
            mesh_fpath = fpath.with_suffix(".ply"), resolution
        if not mesh_fpath.is_file():
            self.export_mesh(mesh_fpath, resolution)
        mesh = o3d.io.read_triangle_mesh(str(mesh_fpath))
        xyz = np.asarray(mesh.vertices)
        rgb = np.asarray(mesh.vertex_colors)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(rgb)
        o3d.io.write_point_cloud(str(fpath), pcd, compressed=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace-dpath', type=Path, default=Path("/workspace"))
    parser.add_argument('--n-steps', type=int, default=100)
    parser.add_argument('--export-resolution', type=int, default=200)
    parser.add_argument('--export-mesh-fpath', type=Path, default=Path("/workspace/exported_mesh.ply"))
    parser.add_argument('--export-pcd-fpath', type=Path, default=Path("/workspace/exported_pcd.ply"))
    args = parser.parse_args()
    workspace = Path(args.workspace_dpath)
    n_steps = int(args.n_steps)
    export_resolution = int(args.export_resolution)
    export_mesh_fpath = Path(args.export_mesh_fpath)
    export_pcd_fpath = Path(args.export_pcd_fpath)

    ingp = Ingp()
    ingp.load_scene(workspace)
    ingp.train(n_steps)
    ingp.export_mesh(export_mesh_fpath, export_resolution)
    ingp.export_point_cloud(export_pcd_fpath, export_resolution, mesh_fpath=export_mesh_fpath)


def _test():
    workspace = Path(
        "/workspace/host/root/media/robovision-syno5-work/nucleus/"
        "0039_OCL3D_data/PoC2/pipeline_py/plant-nerf/nrot-003_ncam-003/plant_000/tmp"
    )

    sys.argv.clear()
    sys.argv.extend([
        "ingp",
        "--workspace-dpath", str(workspace),
        "--n-steps", str(100),
        "--export-resolution", str(200),
        "--export-mesh-fpath", "/workspace/host/home/Safe/Temp/exported_mesh.ply",
        "--export-pcd-fpath", "/workspace/host/home/Safe/Temp/exported_pcd.ply",
    ])


if __name__ == '__main__':
    # _test()
    main()
