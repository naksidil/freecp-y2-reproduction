# Vocabulary sensitivity and selective class recovery in FreeCP

**Nakşidil Aktürk — YZM5110, Yıldız Technical University, 18 November 2026.**

This individual research project reproduces the SCLIP-based VOC21 result of [FreeCP (Chen et al., ICCV 2025)](https://arxiv.org/abs/2508.00557), documents unsuccessful extensions, and evaluates **Y2**: retaining a rejected class when its already-computed same-window CLIP probability is at least **0.50**. No weights are trained and no live LLM API is called.

Public repository: https://github.com/naksidil/freecp-y2-reproduction  
Submission tag: `submission-2026-11-18`

## Main results

All scores are mIoU percentages, calculated from summed confusion matrices. Differences are percentage points.

| Evaluation | FreeCP | + Y2 | Difference | Paired 95% bootstrap interval |
|---|---:|---:|---:|---|
| Disjoint train-derived confirmation300, K=21 | 63.047 | 63.204 | +0.158 | [−0.112, +0.663] |
| Disjoint confirmation300, K=124 (primary) | 59.828 | 60.117 | +0.289 | [−0.113, +0.844] |
| Previously inspected VOC val1449, K=21 | 65.763 | 66.024 | +0.260 | [+0.013, +0.507] |

The paper reports 65.8; our unmodified baseline rounds to the same value. **The independent confirmation intervals include zero.** The full validation benchmark had already been inspected during earlier experiments and is supplemental, not a fresh confirmatory test. These results do not establish robust superiority or SOTA. A matched-count random recovery control is included in confirmation300 (one fixed seed).

## Repository layout and first step

The final benchmark implementation (`y2_segmentor.py`, `run_y2_benchmark.py` and its helpers) is directly browsable in this repository. All eight historical stage folders, upload-ready Colab ZIPs, compact saved evidence and report sources are preserved with their directory structure in **`freecp-y2-source.zip`**. The archive contains source code and selected model outputs, not a dataset or checkpoint.

After cloning or downloading this repository, unpack the verified source bundle once:

```bash
python unpack_source.py
cd source_bundle
```

**All remaining relative commands and paths below are relative to `source_bundle`.** `unpack_source.py` verifies the archive SHA256 before extraction. The nested README contains the same experiment instructions. The report PDF and speaking notes are also provided at repository root for direct viewing.

## Quick verification — no GPU or dataset

Use a clean Python 3.10 or 3.11 environment. The analysis environment is separate from the legacy model environment.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-analysis.txt
python scripts/reproduce_results.py --quick
python -m unittest discover -s freecp_stage9 -p test_metrics.py -v
```

The quick check verifies the SHA256 of every selected archived source member, recalculates condition scores and paired point differences, and checks exact agreement of all 1449 baseline confusion matrices with the earlier baseline. The hand-computed metric tests include void labels, absent classes and invalid predictions.

Recompute the six confirmation comparisons and supplemental benchmark confidence interval using the **original 5000-draw paired-image bootstrap implementation and seed 51100813**:

```bash
python scripts/reproduce_results.py
python report/build/prepare_figures.py
python report/build/build_report.py
```

Outputs: `output/reanalysis/all_condition_scores.csv`, `output/reanalysis/verification.json`, report figures in `report/build`, and the Word report in `output/report`. PDF rendering requires an external DOCX-to-PDF renderer such as LibreOffice. Bootstrapping resamples the same image indices for both methods and aggregates confusion counts before mIoU; it does not average per-image IoUs. It estimates image-sampling uncertainty, not training-seed or cross-dataset uncertainty.

## Reproduce model inference in Colab

1. Select a T4 GPU. Upload/open `02_FreeCP_T4_Kurulum_ve_Pilot.ipynb` and execute its setup and three-image pilot cells. It creates `/content/freecp_t4/env_py310` and pins the upstream code and model dependencies. Use that Python for the model; do not substitute the Colab host's Python 3.13/Torch environment. The submission notebook replaces only the initial host Torch import with an `nvidia-smi` T4 check to avoid the observed host import error; the isolated CUDA/operator tests and experimental code are unchanged. This packaging change was syntax-checked, not rerun on a GPU during report preparation.
2. For confirmation300, append the exact launcher from `freecp_stage8/COLAB_BASLAT.txt`; upload `packages/08_FreeCP_Y2_Sabit_Dogrulama.zip` when requested. For val1449, use `freecp_stage9/COLAB_BASLAT.txt` and `packages/09_FreeCP_Y2_Final_Benchmark.zip`.
3. Keep downloaded result ZIPs. They contain run status, configs, manifests, per-image decisions, masks and analyses. A stopped run with zero conditions is **not a result**.

**Stages 03–07 do not need to be rerun to launch the final packages after successful 02 setup.** Their reviewed protocol evidence is bundled. Do not disable provenance guards or adjust thresholds in response to test results.

Alternatively, after 02 setup, place the unmodified stage folders under `/content` and run:

```bash
/content/freecp_t4/env_py310/bin/python /content/freecp_stage8/run_y2_confirmation.py
/content/freecp_t4/env_py310/bin/python /content/freecp_stage9/run_y2_benchmark.py --root /content/freecp_t4
```

These are full GPU experiments, not the CPU quick test. The saved val1449 run used about 0.28 seconds per image for the measured inference pipeline; setup, downloads, analysis and output writing add time. Comparable latency does not demonstrate a speedup. Exact bitwise reproduction can depend on CUDA and GPU libraries; archived baseline parity is specifically verified.

## Pinned implementation and data

- Upstream: [chenqi1126/FreeCP](https://github.com/chenqi1126/FreeCP), commit `6946aa7999ec5d57d1fe91968ba1ae35a1e53295`.
- CLIP ViT-B/16; SCLIP branch; Python 3.10.22; torch 1.13.1+cu117; torchvision 0.14.1+cu117; NumPy 1.26.4; mmcv 2.0.1; mmengine 0.8.4; mmsegmentation 1.1.1. Complete recorded freezes and runtime reports are under `results/stage*/`.
- Inference resize `(2048,448)` with preserved aspect ratio; sliding crop 384, stride 112; original RP threshold 0.15; AP overlap threshold 0.8. Checkpoint/tokenizer hashes and model provenance are retained in the notebook, stage source and report.
- Obtain [PASCAL VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) under its provider's terms, with the `VOCdevkit/VOC2012` directory layout. Dataset images, ground-truth masks and model checkpoints are not redistributed here. The setup notebook documents/downloads the original model assets and dataset access route. Follow the dataset/model provider's applicable terms.
- Initial50, development200 and confirmation300 are disjoint subsets of **VOC train**, separate from official val1449. Exact split lists and hashes are retained under `results/stage*/splits` and the manifests. Their sum leaves 914 train images unused.
- K=21 means 20 VOC foreground outputs plus background, with **46 text queries** (26 background queries). Expanded K=124 has 103 extra ADE-derived labels and **149 text queries**; ADE images are not used. Extra output predictions map to background only after the full argmax. Their presence is unannotated in VOC, not assumed false. Void label 255 is ignored.

## Experiment map

| Stage | Purpose | Status / interpretation |
|---|---|---|
| 02–03 | Environment, pilot and baseline | 65.763 on val1449 |
| 04–05 | RP threshold, cosine score, background query lists | Exploratory; loose RP and cosine did not generalize |
| 06 | PAMR, lower RP, selective second look, combination | No convincing gain on development200; PAMR added ~41% time |
| 07 | Two-vocabulary presence diagnostics and threshold selection | Y2 τ=0.50 selected by rescue precision/recall, not mIoU |
| 08 | Locked six-condition confirmation300 | Baseline, Y2 and matched-count random for each vocabulary |
| 09 | Locked two-condition supplemental val1449 benchmark | Baseline parity, Y2, bootstrap and failure examples |

RP cosine normalization increased absent-class retention from about 1.4% to 60.7%. This supports sensitivity to magnitude; it does **not** prove RP is only a confidence filter. In the second-look experiment, correct presence decisions could still restore incorrect spatial regions. Y1 synonym grouping, pure-confidence/shape ablations, GT-vocabulary inference and corruption tests were **not performed**.

## Files and evidence boundaries

- `freecp_stage2`…`freecp_stage9`: preserved tested source packages. Historical stages retain their original guards and launch instructions.
- `packages`: upload-ready stage ZIPs for Colab. Delivery edits only standardize launcher comments and notebook narrative; experimental Python source and archived numerical evidence remain unchanged.
- `results`: selected **unmodified** archived confusion matrices, exact splits/configs, summary/provenance JSON, environment freezes and four selected prediction PNGs. `source_members.json` maps each file to its source ZIP/member and hash.
- `report/build`: full-precision summaries, comparison tables, figure and report generation scripts, and references. The complete manuscript explains every experiment family, uncertainty, negative outcomes and implementation deviations.
- `scripts/reproduce_results.py`: compact CPU reanalysis using packaged matrices.

Large full prediction/log archives are intentionally omitted. Therefore, this compact repository recomputes the segmentation tables and final paired uncertainty, but presence-event and timing analyses are retained as archived summaries rather than regenerated from all raw logs. To rerun the original `analyze7.py`/`analyze8.py`/`analyze9.py`, use the **complete output ZIP from the corresponding full GPU run**, not this compact `results` directory. Report appendix E records the source archive SHA256 values.

## Attribution, licenses and AI assistance

See `THIRD_PARTY_NOTICES.md`. Upstream FreeCP and PAMR notices are preserved in the individual stage folders. No ownership over those implementations, datasets or checkpoints is claimed.

ChatGPT/Codex assisted with literature research, implementation, dependency troubleshooting, tests, result auditing, analysis, writing and presentation. A separate ChatGPT conversation supplied methodological criticism and suggestions; its model version was not recorded. The student ran the Colab experiments and supplied the archives and remains responsible for understanding, checking and defending the submitted work. No experiments were invented to fill a reporting gap.
