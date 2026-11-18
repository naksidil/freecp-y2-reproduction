# FreeCP reproduction and selective class recovery (Y2)

Nakşidil Aktürk — YZM5110, Yıldız Technical University.

**Publication status: source and evidence upload is pending. This initial repository is not yet the final submission.**

The prepared submission package includes the 28-page report, tested experiment code, exact splits and configurations, compact saved outputs, CPU reanalysis, and speaking notes. Its files must be uploaded before creating the final `submission-2026-11-18` tag.

## Research

Selected paper: [Training-Free Class Purification for Open-Vocabulary Semantic Segmentation](https://arxiv.org/abs/2508.00557), Chen et al., ICCV 2025.

The unchanged FreeCP baseline reproduces 65.763 mIoU on VOC val1449, matching the paper’s rounded 65.8. Earlier experiments include RP retuning, cosine RP, background-query changes, PAMR, and selective second look.

Y2 restores RP-rejected classes when an already-computed same-window CLIP probability is at least 0.50. On disjoint confirmation300, gains are +0.158 (K=21) and +0.289 (K=124); both paired 95% bootstrap intervals include zero. On the previously inspected val1449 benchmark, Y2 reaches 66.024 (+0.260). The latter is supplemental evidence, not a fresh independent test.

No model weights were trained. ChatGPT/Codex supported research, coding, tests, analysis and writing; a separate ChatGPT conversation supplied methodological criticism. The student ran the Colab experiments and remains responsible for the work.
