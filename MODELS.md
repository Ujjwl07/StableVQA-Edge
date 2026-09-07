# Model provenance and redistribution status

The five ONNX subgraphs in `onnx_models/` and `android/app/src/main/assets/` are
derived artifacts. They contain no weights trained by us: every parameter comes
from a published upstream checkpoint, converted to ONNX and, for some branches,
quantized to INT8. This file records where each one comes from and what its
upstream terms are, because those terms differ and one of them is unresolved.

## Where each subgraph comes from

| Subgraph | Upstream weights | Upstream licence |
|---|---|---|
| `flow_model_*` | NeuFlow v2 (`neufieldrobotics/NeuFlow_v2`) | Apache-2.0 |
| `backbone_*` | StableVQA checkpoint (Swin-T branch) | **none stated** |
| `deblur_net_*` | StableVQA checkpoint (Stripformer encoder) | **none stated** |
| `motion_analyzer_*` | StableVQA checkpoint (3D ResNet-18) | **none stated** |
| `quality_head_*` | StableVQA checkpoint (MLP head) | **none stated** |

Ancestry of the StableVQA checkpoint itself, for completeness:

- Swin Transformer — MIT (`microsoft/Swin-Transformer`)
- Stripformer — LICENSE file present in `pp00704831/Stripformer-ECCV-2022-`; terms not verified here
- RealBlur dataset, used to train the Stripformer encoder — CC BY 4.0 (`rimchang/RealBlur`)

## Getting the models

The five subgraphs are published as a release asset rather than committed, so a
clone stays small. Download and unpack them into the repository root:

```bash
curl -L -o models.zip https://github.com/Ujjwl07/StableVQA-Edge/releases/download/v1.0.0/stablevqa-edge-models-v1.0.0.zip
unzip -q models.zip && rm models.zip
```

This creates `onnx_models/` with the ten `.onnx` files. For the Android build,
copy the five `*_quant.onnx` files into
`android/app/src/main/assets/` before assembling the APK.

## The unresolved item

`github.com/QMME/StableVQA` publishes no LICENSE file and states no licence.
Under GitHub's terms, material published without a licence is all rights
reserved: it may be viewed and forked on GitHub, but no grant to redistribute
or to publish derivative works is given.

Four of the five subgraphs descend from that checkpoint. We therefore make no
licence claim over those four files and assert no grant to redistribute them.
They are included here only to make the measurements in the accompanying paper
reproducible. If the StableVQA authors object, we will remove them on request.

The flow subgraph is unaffected: NeuFlow v2 is Apache-2.0 and redistributable
with attribution.

## Reproducing the models instead of using ours

The export and calibration path is documented in `BUILD.md`. It takes the
upstream StableVQA checkpoint, obtained from the authors' own published link,
and produces the five ONNX files locally. Users who prefer not to rely on the
copies here can regenerate them.

## Attribution

    @inproceedings{kou2023stablevqa,
      title={Stablevqa: A deep no-reference quality assessment model for video stability},
      author={Kou, Tengchuan and Liu, Xiaohong and Sun, Wei and Jia, Jun and Min, Xiongkuo and Zhai, Guangtao and Liu, Ning},
      booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
      pages={1066--1076},
      year={2023}
    }

    @inproceedings{zhang2025neuflowv2,
      title={NeuFlow-V2: Push High-Efficiency Optical Flow To the Limit},
      author={Zhang, Zhiyong and Gupta, Aniket and Jiang, Huaizu and Singh, Hanumant},
      booktitle={2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
      pages={2479--2485},
      year={2025}
    }
