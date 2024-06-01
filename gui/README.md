# Total-Decom interactive GUI

The GUI provides prompt-driven 3D scene decomposition. Navigate a reconstructed scene, click an object, grow the prompt into a 3D surface selection, and export the selected sub-mesh.

It combines Gaussian Splatting for real-time rendering, [Segment Anything](https://github.com/facebookresearch/segment-anything) for click-driven 2D masks, and feature-based surface growing for lifting the mask to the reconstructed mesh.

## Configure and launch

Install the dependencies described in the root [`README.md`](../README.md), then edit `configs/replica_mesh.yaml`:

| Key | Expected value |
| --- | --- |
| `data_dir` | MonoSDF-format dataset root containing `scanN/` |
| `scan_id` | `N` in the selected `scanN/` directory |
| `load` | Point-cloud PLY, or a Gaussian PLY when `load_ply: True` |
| `foreground_mesh` | Foreground surface used for region growing |
| `vertex_features` | PyTorch tensor with one row per foreground-mesh vertex |
| `sam_checkpoint` | SAM checkpoint path |
| `sam_model_type` | SAM registry name matching the checkpoint, such as `vit_h` |
| `device_index` | CUDA device index |

All relative paths are resolved from the directory where the command is launched. The example configuration assumes the GUI is launched from `gui/`:

```bash
python train.py --config configs/replica_mesh.yaml
```

The application validates all required paths and verifies that the vertex-feature tensor is aligned with the foreground mesh before opening the window. Generated component meshes and the current extracted mesh are written under `gui/tmp/`.

## Interface

The window has two panels:

- **Render View**: live scene viewport. Mouse and keyboard interaction happens here; click the panel before using shortcuts.
- **Control Panel**: rendering, initialization, and decomposition controls.

### Rendering modes

| Mode | Display |
| --- | --- |
| `image` | Gaussian Splatting RGB render |
| `mesh` | Shaded mesh and selected triangles |
| `depth` | Depth map |
| `alpha` | Coverage/opacity map |
| `layout` | Foreground silhouette |
| `decompose` | Interactive object selection view |

Selection is performed in `decompose` mode.

### Camera controls

| Action | Result |
| --- | --- |
| Left-drag | Orbit |
| Middle-drag | Pan |
| Scroll | Zoom |

## Decompose an object

1. Select `decompose` rendering mode.
2. Select **operation** mode so clicks update the selection. **view** mode leaves the selection unchanged while navigating.
3. Choose a selection strategy:
   - **seed vertices** sends positive and negative clicks to SAM, projects the resulting mask onto the mesh, and uses the projected vertices as region-growing seeds;
   - **connected component** selects a precomputed connected mesh component with one click.
4. Hold **Shift** to arm a fresh prompt sequence.
5. Left-click to add a positive point; right-click to add a negative point.
6. Press **Space** or click **grow** to expand the seed vertices over similar surface features.
7. Use **undo** or **reset** to refine the result.

The latest extracted sub-mesh is saved as `tmp/mesh_extracted.ply`.

## Shortcut reference

| Input | Result in `decompose` + `operation` |
| --- | --- |
| Shift | Arm a new decomposition selection |
| Left-click | Positive SAM prompt |
| Right-click | Negative SAM prompt |
| Space | Grow the current 3D selection |

## Notes

- A CUDA-capable NVIDIA GPU is required; rendering, rasterization, and SAM do not have a CPU fallback in this release.
- The first launch can be slow because foreground connected components and mesh adjacency are precomputed.
- `vertex_features` must be generated from the same reconstruction and in exactly the same vertex order as `foreground_mesh`.
- The GUI creates its temporary output directories automatically.
