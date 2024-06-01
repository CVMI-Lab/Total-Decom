# Third-party notices

This repository combines original Total-Decom code with code and interfaces derived from other research projects. Those components remain subject to their upstream terms.

## Bundled or adapted source

- **MonoSDF** — the reconstruction pipeline is derived from [autonomousvision/monosdf](https://github.com/autonomousvision/monosdf). Its MIT license is preserved in the root [`LICENSE`](LICENSE).
- **3D Gaussian Splatting** — parts of `gui/src/rendering/` and related Gaussian utilities are adapted from [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting). The upstream research-only license is included at [`THIRD_PARTY_LICENSES/3d-gaussian-splatting-LICENSE.md`](THIRD_PARTY_LICENSES/3d-gaussian-splatting-LICENSE.md).
- **PlenOctree spherical-harmonics utilities** — the BSD-style notice is retained in `gui/src/utils/sh_utils.py`.

## External dependencies

The following projects are installed separately and are not vendored here. Users must comply with their respective licenses:

- [Segment Anything](https://github.com/facebookresearch/segment-anything)
- [nvdiffrast](https://github.com/NVlabs/nvdiffrast)
- [Differential Gaussian Rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
- [simple-knn](https://gitlab.inria.fr/bkerbl/simple-knn)
- [Omnidata](https://github.com/EPFL-VILAB/omnidata)

The 3D Gaussian Splatting-derived GUI code is restricted to research/evaluation use under its upstream license. Do not describe the combined repository as permitting unrestricted commercial use without first replacing or relicensing those components.
