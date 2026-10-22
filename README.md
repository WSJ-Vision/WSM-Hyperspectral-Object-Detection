# Weighted Structured Spectral Adaptation Projection Mapper

This repository provides the research implementation of the Weighted Structured
Spectral Adaptation Projection Mapper (WSM) for detector-compatible
hyperspectral object detection. WSM maps a full-band tensor to a three-channel
representation while keeping the downstream detector interface unchanged.

## Included

- WSM, NNP, Fixed3, and learnable 1x1 spectral mappers.
- HOD3K 4x4 mosaic decoding and multi-page TIFF reading utilities.
- Dataset configuration templates for HOD3K, HOD1, and SZU_CAR.
- A minimal mapper smoke test.
- Public result summaries used in the manuscript.

## Data Availability and Repository Scope

This repository provides the implementation, configuration templates, and
public result summaries needed to inspect the WSM method. It does not
redistribute raw images, annotations, trained checkpoints, prediction files,
server logs, private filesystem paths, or access credentials.

HOD3K, HOD1, and SZU_CAR remain subject to the terms of their respective
releases. Users should obtain each dataset from its official distribution
channel and comply with the applicable license, access conditions, and citation
requirements. Before running a training job, set the local dataset locations in
the configuration templates and verify that the selected split and
preprocessing settings match the intended experimental protocol.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/smoke_test.py --channels 16
python scripts/smoke_test.py --channels 112
```

On Windows, activate the environment with `.venv\\Scripts\\Activate.ps1`.

## Using the Mapper

```python
import torch
from wsm_hsi import WSMMapper

mapper = WSMMapper(in_channels=112)
x = torch.rand(2, 112, 640, 640)
pseudo_image, auxiliary = mapper(x)
```

`pseudo_image` has shape `[B, 3, H, W]` and can be passed to a detector that
accepts three-channel inputs. During optimization, add `orth_loss` and
`band_entropy_loss` using the coefficients fixed by the relevant experiment
configuration.

## Protocol Notes

- HOD3K uses 16 decoded mosaic channels and the predefined clipping and
  normalization constant `tau=384`.
- HOD1 uses 96 TIFF pages converted from uint8 to float32 and divided by 255.
- SZU_CAR uses all 112 uint8 TIFF pages converted to float32 and divided by
  255. Its official release provides 417 training and 188 validation/evaluation
  images; it does not provide an independently released test split.
- Fixed3 indices are `[0, 7, 15]` for HOD3K, `[0, 47, 95]` for HOD1, and
  `[0, 55, 111]` for SZU_CAR. PCA3 must be fitted on training data only.

The primary WSM pathway does not include pseudo-image consistency (PISC). PISC
is a separately evaluated, training-only auxiliary control and is not used at
validation or inference.

## Reproducing Detector Runs

The mapper package is detector agnostic. To reproduce the reported detector
runs, connect a mapper to a compatible YOLO implementation, preserve the
dataset split and preprocessing protocol, and use the provided configuration
templates as the experimental record. The complete run-level result manifests
and large artifacts are not redistributed here because they include restricted
data references and nonportable infrastructure paths.

## Results

The HOD3K five-seed paired mAP50-95 summary used in the manuscript is included
in `results/hod3k_yolov8n_five_seed_map5095.csv`. Detailed seed-level results
are reported in the accompanying supplementary material.

## Citation

Add the final bibliographic citation and DOI after publication. Until then,
cite this repository by its tagged release version and commit hash.

## License

This project is released under the [MIT License](LICENSE).
