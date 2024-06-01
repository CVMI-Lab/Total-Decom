import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import trimesh
from pyhocon import ConfigFactory

import utils.general as utils


def load_model(conf_path, checkpoint_path, device):
    conf = ConfigFactory.parse_file(conf_path)
    model = utils.get_class(conf.get_string('train.model_class'))(
        conf=conf.get_config('model')
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    if next(iter(state_dict)).startswith('module.'):
        state_dict = {key[7:]: value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


@torch.no_grad()
def export_vertex_features(model, mesh_path, output_path, device, chunk_size):
    mesh = trimesh.load(mesh_path, process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    if len(vertices) == 0:
        raise ValueError(f'foreground mesh has no vertices: {mesh_path}')

    feature_chunks = []
    for start in range(0, len(vertices), chunk_size):
        points = torch.from_numpy(vertices[start:start + chunk_size]).to(device)
        feature_chunks.append(model.feature_network(points).cpu())

    features = torch.cat(feature_chunks, dim=0)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(features, output_path)
    print(f'Wrote {features.shape[0]} vertex features to {output_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Export one learned Total-Decom feature vector per foreground-mesh vertex.'
    )
    parser.add_argument('--conf', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--foreground_mesh', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--chunk_size', type=int, default=100000)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available.')
    if args.chunk_size <= 0:
        raise ValueError('--chunk_size must be positive')

    model = load_model(args.conf, args.checkpoint, device)
    export_vertex_features(
        model,
        args.foreground_mesh,
        args.output,
        device,
        args.chunk_size,
    )


if __name__ == '__main__':
    main()
