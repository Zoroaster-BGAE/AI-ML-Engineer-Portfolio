# Third-party code and attribution

This repository contains the author's own pipeline code and a synthetic-data generator. It deliberately does **not**
vendor upstream open-source code. Where the work builds on or adapts a published project, the project is credited here
and, where needed, fetched at run time into `third_party/` (git-ignored).

| Upstream | What it is | How this repo relates to it | Licence |
|---|---|---|---|
| **TOTEM**: TOkenized Time Series EMbeddings for General Time Series Analysis (Talukder, Yue, Gkioxari), arXiv:2402.16412 | Open-source VQ-VAE time-series tokenizer | Not copied. Any tokenizer work here is written separately from the paper. Upstream is fetched with `scripts/fetch_third_party.sh` if needed | **Verify upstream `LICENSE` before reuse** |
| **PyOMA** (operational modal analysis library) | Open-source OMA functions (SSI-COV, FDD, pLSCF) | Not included. Code that adapts upstream functions is withheld from this release | **Verify upstream `LICENSE` before reuse** |
| **SDyPy** | Structural-dynamics Python tools, referenced in a comment of upstream PyOMA | Not used here | Not applicable |

If upstream's licence requires it, derived files must carry upstream's copyright notice and be released under a
compatible licence. No file adapted from upstream is included in this release.
