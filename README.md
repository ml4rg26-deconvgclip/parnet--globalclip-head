# deconvgclip — Deconvolving total globalCLIP signal into per-RBP contributions

Course project (ml4rg26, "Idea 2"). This repository fine-tunes a pretrained
[ParNet](https://github.com/project-parnet) backbone to predict the **total globalCLIP
signal** from RNA sequence, then uses **Integrated Gradients (IG)** + **metamotif** to
recover the sequence motifs — and thereby the candidate RNA-binding proteins (RBPs) —
that drive that signal. See the shared report for the full write-up alongside Idea 1
(a reconstruction-layer approach combining Parnet's existing 223 eCLIP tracks).

---

## 1. Idea in one paragraph

ParNet was trained to predict 223 individual eCLIP tracks (one per RBP–cell-line pair).
Here we replace ParNet's 223-track output head with a **single-track `GlobalCLIPHead`**
that predicts the *total* CLIP signal (the sum of all RBP binding). We fine-tune this
model, evaluate how well it reproduces the measured signal (Pearson/Spearman on held-out
test data), and then interpret *which* sequence features it relies on via IG. The IG
attributions are fed to metamotif to extract informative subsequences, which are matched
against known RBP motif databases. The biological question: **which RBPs bind the target
RNA, as recovered purely from the total signal?**

Control (SMInput) modelling is intentionally **out of scope** for this project — we model
signal only. See "Outlook" below.

---

## 2. Repository layout

```
notebooks/globalclip-head/
  00_explore_globalclip_data.ipynb     # data inspection
  01_prepare_globalclip_dataset.ipynb  # outlier filtering → filtered dataset
  02_train_globalclip_head.ipynb       # fine-tuning pipeline
  03_integrated_gradients.ipynb        # IG + metamotif + RBP matching (incl. robustness checks)
  04_evaluate_runs.ipynb               # test-set Pearson/Spearman per run
  05_comparison_plots.ipynb            # all report/presentation figures + summary tables

notebooks/globalclip-head-clean/       # polished copies of the above (WIP, in progress)
notebooks/explore/                     # early exploratory notebooks (dataset comparison, etc.)
config/
  filepaths.yaml                       # central path management (data, models)
externals/                             # cloned deps (NOT committed — see setup)
setup_vm.sh                            # one-shot VM re-setup (on NAS, not git)
```

Results (models, figures, metamotif output) are written to
`results/globalclip-head/` on the NAS and are **not** committed to git. A full copy of
all models, the filtered dataset, all figures, and the complete per-run results tree is
also mirrored to shared storage at `/mnt/storage1/ml4rg26-deconvgclip/lhofer/` (see
Section 10).

---

## 3. Environment setup

The environment is managed with **pixi**. Two GPU variants exist in `pixi.toml`:
`parnet-dev-cu12` (preferred, faster, correct ipywidgets rendering) and `parnet-dev-cu11`
(fallback only).

> **Important:** on this Paperspace setup the home directory does not persist across VM
> restarts, so pixi, SSH keys, git identity, and the `externals/` repos must be
> re-established after each restart. The `setup_vm.sh` script automates most of this.

### First-time / post-restart setup

```bash
# 1. install pixi (if missing)
curl -fsSL https://pixi.sh/install.sh | bash
export PATH="$HOME/.pixi/bin:$PATH"

# 2. isolate the pixi cache (avoids cross-contamination between teammates)
echo 'export PIXI_CACHE_DIR="$HOME/.pixi-cache"' >> ~/.profile
source ~/.profile

# 3. git config
git config --global user.email "<you>"
git config --global user.name "<you>"
git config --global --add safe.directory '*'

# 4. install the environment (use --frozen to avoid re-resolving the lockfile)
cd <repo>
pixi install -e parnet-dev-cu12 --frozen

# 5. clone the externals (required before the env can be used)
#    all under the project-parnet org: https://github.com/project-parnet
git clone https://github.com/project-parnet/pylbsr.git externals/pylbsr
git clone https://github.com/project-parnet/parnet_analyses_libs.git externals/parnet_analyses_libs
git clone https://github.com/project-parnet/parnet_additional_utils.git externals/parnet_additional_utils
git clone https://github.com/project-parnet/metamotif.git externals/metamotif

# 6. install parnet_demo_utils (not in the pixi manifest)
#    source repo: https://github.com/project-parnet/parnet--demo--train-models
<repo>/.pixi/envs/parnet-dev-cu12/bin/pip install -e <path-to>/parnet--demo--train-models --no-deps
```

Point your Jupyter kernel / VS Code interpreter at:
`<repo>/.pixi/envs/parnet-dev-cu12/bin/python`

### Known gotchas

- **`GLIBCXX_3.4.29 not found`** when running the env's python from the terminal:
  `export LD_PRELOAD=$(find <repo>/.pixi/envs -name "libstdc++.so.6" | head -1)`
  (does not occur inside the notebook kernel).
- **cu11 vs cu12:** cu12 is noticeably faster and renders ipywidgets progress bars
  correctly; prefer it. cu11 is only a fallback.
- **`pixi install` "no compatible Python interpreter":** use `--frozen`.
- **`ParnetModelName`** must match an entry in `config/filepaths.yaml → models`.
- **Stale metamotif output:** the alignment script does not clear its output directory
  before writing. If you rerun `03_integrated_gradients.ipynb` on the same run with
  different `params_min_support` / `params_max_motifs`, old `motif-N.*` files can survive
  alongside new ones with inconsistent support counts. The notebook clears
  `aligned_motifs/` before each alignment run — if working from an older copy, add this
  manually or check file timestamps in that folder before trusting motif counts.

---

## 4. Data

- **Raw dataset:** provided globalCLIP `.pt` (lysate, noNHS), 39,052 train / 7,361 val /
  4,946 test windows, 600 nt, single track (`total_key="globalCLIP"`).
- **Filtered dataset** (produced by `01_prepare_globalclip_dataset.ipynb`): removes
  windows whose total signal exceeds a max threshold (1000). This removes ~1% of windows
  (extreme mapping-artefact outliers up to ~500k counts) and dramatically stabilises
  training. → 38,695 / 7,266 / 4,896 windows.

Note: the globalCLIP *signal* is identical across the provided dataset variants; only the
(unused) control differs.

---

## 5. Pipeline / how to run

1. **Prepare data** — run `01_prepare_globalclip_dataset.ipynb` once to create the
   filtered dataset. Registered in `filepaths.yaml` as `data_lysate_noNHS_filtered`.
2. **Train** — `02_train_globalclip_head.ipynb`. Key params (top of notebook):
   - `pretrained_model_name` — `PARNET_7M_0_0` or `PARNET_21M_5_0`
   - `params_finetuning_strategy` — `head` / `unfreeze_last_n_layers` / `full`
   - `params_unfreeze_last_layers_n`, `params_use_filtered_data`, `params_max_epochs`
   - Run folders are auto-named, e.g.
     `parnet.7m-0.0.unfreeze_last_5_layers.globalclip-lysate-noNHS.filtered.noctrl.ep50`
   - Saves best-checkpoint weights to `model.statedict.pt` in the run folder.
3. **Evaluate** — `04_evaluate_runs.ipynb`. Set `EVAL_RUN`, `PRETRAINED_MODEL`,
   and the matching test dataset; writes `test_evaluation.yaml` + per-sequence score
   arrays into the run folder.
4. **Interpret** — `03_integrated_gradients.ipynb`. Computes IG attributions on the test
   set, runs `metamotif search` + variable-length seed alignment to extract consensus
   motifs, and matches them against RBP databases. Writes to
   `<run>/integrated_gradients/`. Also contains the robustness checks described in
   Section 8 below (mCrossBase-only matching, PTBP1 vs. Idea 1's top RBPs, and the
   single-window prediction + attribution example).
5. **Plot** — `05_comparison_plots.ipynb` gathers all runs and produces the report/
   presentation figures + summary tables into `results/globalclip-head/figures/`.

### Model architecture note (7M vs 21M)

The two backbones output **different hidden dimensions** (7M → 512, 21M → 768). The head
uses `LazyConv1d`, so it auto-adapts; the notebooks derive the hidden dim from a real
backbone forward pass rather than hard-coding 512. When switching model size you only
change `pretrained_model_name` / `PRETRAINED_MODEL`.

---

## 6. Experiments (report structure)

| # | Comparison | Fixed config | Purpose |
|---|-----------|--------------|---------|
| 1 | Filtered vs raw dataset | unfreeze_2, 50 ep | justify filtering |
| 2 | Unfreeze depth (head / 2 / 5) | filtered, 50 ep | effect of fine-tuning depth |
| 3 | 7M vs 21M backbone | unfreeze_2, filtered, 50 ep | effect of model capacity |
| 4 | Motif recovery across models | head / unfreeze_2 / unfreeze_5 (7M) / unfreeze_2 (21M) | does a better model recover better motifs? |

---

## 7. Key findings (summary)

**Filtering is essential.** Raw data contains rare but extreme signal outliers
(up to ~525k counts vs a median of ~44) that cause large loss spikes and unstable
training; filtering (max signal 1000) yields smooth convergence and better test
correlation.

**Fine-tuning depth and model capacity both matter, and trade off against each other:**

| Config (filtered, ep50) | Val loss | Pearson | Spearman |
|---|---|---|---|
| 7M, head only | ~152 | 0.234 | 0.273 |
| 7M, unfreeze last 2 | 142.3 | 0.257 | 0.285 |
| 7M, unfreeze last 5 | 137.3 | 0.266 | 0.292 |
| 21M, unfreeze last 2 | 137.2 | 0.260 | 0.294 |

Notably, **21M + unfreeze_2 ≈ 7M + unfreeze_5** on val loss — a bigger backbone with
shallower fine-tuning reaches similar performance to a smaller backbone fine-tuned more
deeply. Interestingly, the *size* of the val-loss gap between configurations does not
translate proportionally into a Pearson/Spearman gap of similar size — correlation moves
only modestly across most configurations despite larger loss differences.

**Motif recovery (IG + metamotif, high-sensitivity settings — see §8):**

| Model | # consensus motifs | Notable RBP matches (PCC) |
|---|---|---|
| 7M, head | 3 (one large low-information motif) | TIA1/TIAL1, PCBP1/2 |
| 7M, unfreeze_2 | 3 | **PTBP1** (0.915, pyrimidine motif) |
| 7M, unfreeze_5 | 3 | HNRNPK/L, MBNL1, YBX1 |
| 21M, unfreeze_2 | 2 (one motif dominates 94.5% of k-mers) | **PTBP1/PTBP2/PUF60** (0.945, pyrimidine motif) |

All models recover biologically plausible pyrimidine/GU-rich binding motifs, matching
known RBPs (PTBP1/2, PCBP1/2, TIA1/TIAL1, MBNL1, HNRNPK/L). Low-complexity motifs
(poly-G/poly-C) additionally match many RBPs non-discriminatively — genuine shared
low-complexity binding preference rather than an artefact.

**Model quality and motif diversity do not simply track together:** the best-performing
model by loss/correlation (21M) recovers *fewer* distinct consensus motifs (2, one
dominant) than the 7M models (3 each) — a concentration effect worth discussing rather
than assuming "better model → richer motifs".

**Predictive power vs. biological interpretability is a genuine trade-off — arguably the
project's most important finding.** Across every comparison, the model that best predicts
the aggregate globalCLIP signal is consistently the one with the *least* per-protein
interpretability (fewest/most concentrated motifs, weakest self-matches). Our working
explanation: the more a model is pushed to fit the *overall* mixed globalCLIP signal, the
more resolution it has to give up from ParNet's original per-protein embeddings — fitting
the aggregate and preserving single-protein specificity appear to compete for the same
representational capacity. This pattern shows up independently in both this idea's
fine-tuned models and in Idea 1's self-match validation of ParNet's own frozen tracks
(see the shared report's Discussion), making it unlikely to be a fine-tuning artefact
specific to either approach. This trade-off would be a valuable direction for further,
more targeted investigation.

Exact final numbers and all figures: regenerate by running `05_comparison_plots.ipynb`
(produces `master_summary_table.csv`, `rbp_matches_across_models.csv`,
`trainable_params.csv`, and all figures in `results/globalclip-head/figures/`).

---

## 8. Robustness checks (added after supervisor feedback)

Three follow-up analyses live in `03_integrated_gradients.ipynb`, prompted by
supervisor questions during final review:

1. **Pretraining-bias check (mCrossBase-only matching).** Since ParNet was pretrained on
   eCLIP data for 223 specific RBPs, we repeated RBP matching restricted to mCrossBase
   alone (103 RBPs, a subset of the pretraining set) to check whether our motif–RBP
   matches simply reproduce the pretraining targets. Result: the top hit changes for
   nearly every motif under this restriction (results in
   `results/globalclip-head/mcross_only_rbp_matches.json`), and the matches obtained are
   *varied*, not a fixed repeated subset — evidence against simple pretraining-bias
   reproduction. Functionally related proteins (PCBP2, HNRNPK) persist across both
   conditions even where the exact top hit changes.

2. **PTBP1 vs. Idea 1's top-5 RBPs.** To check whether our recovered PTBP1 hit could be a
   false positive driven by generic pyrimidine-rich similarity, we directly correlated
   PTBP1's database motif against Idea 1's top-5 weighted RBPs (SND1, YBX3, PABPN1,
   SRSF1, LIN28B). All five correlate highly with PTBP1 (PCC 0.70–0.98) — meaning the two
   ideas' top candidates are not cleanly separable by this matching approach in either
   direction. Framed as a caution about RBP motif matching in general, not a caveat
   specific to either idea's results.

3. **Single-window case study.** Using the exact same 600 nt window highlighted in Idea
   1's example figure (`chr8:27610416-27611016`, raw signal sum 945), we ran our
   fine-tuned model's prediction and IG attribution on it directly, then cropped the
   attribution logo to the highest-attribution sub-region. The predicted profile closely
   tracks the observed signal at this window, and the highest-attribution region is
   C/T-rich, consistent with the pyrimidine-rich motif class recovered throughout this
   analysis (though not attributable to a single specific RBP with confidence).

---

## 9. metamotif parameters

Search + alignment sensitivity (set at the top of `03_integrated_gradients.ipynb`):

- `search.sig_p = 0.05` (relaxed from 0.01 — more sensitive k-mer detection)
- alignment `--min-support 20` (relaxed from 100)
- alignment `--max-motifs 15` (raised from 5)

Rationale (per supervisor): a large diversity of motifs is expected, so higher
sensitivity is preferred. Note that with these settings, all 7M configurations still
converged to 3 well-supported motifs and the 21M configuration to 2 — the parameters
raise the *ceiling*, not a forced count.

---

## 10. Shared storage & handover

A complete mirror of all outputs is kept at
`/mnt/storage1/ml4rg26-deconvgclip/lhofer/`, independent of the NAS workspace copy:

```
lhofer/
  models/       # all 9 trained model .statedict.pt files
  datasets/     # the filtered globalCLIP dataset
  figures/      # all report + presentation figures (.png) and summary tables (.csv)
  results/
    training/   # full per-run results tree (configs, checkpoints, metrics, IG/metamotif output)
    mcross_only_rbp_matches.json   # robustness check output (Section 8)
```

Code lives on GitHub at [ml4rg26-deconvgclip](https://github.com/ml4rg26-deconvgclip);
branch `dev` holds the merged, up-to-date state of this notebook set (a
`notebooks/globalclip-head-clean/` folder also exists with in-progress polished copies
of the same pipeline).

---

## 11. Outlook / not done

- **Control (SMInput) modelling.** The architecture is compatible with re-introducing a
  control branch (as in ParNet's `AdditiveMix` with `num_tasks=1`, `use_control=True`),
  but this was out of scope. Reintroducing it would let the model separate genuine RBP
  binding from background.
- **100-epoch runs** for the larger/deeper configurations (50 was sufficient for the
  configurations shown here; the 7M-unfreeze_5 and 21M runs already showed mild
  overfitting onset by epoch ~30–40, so longer runs are unlikely to help without
  regularisation).
- **Full-test-set motif comparison** at even higher sensitivity, and cross-model motif
  alignment by biological identity (rather than per-model rank) if a more direct
  motif-by-motif comparison across configurations is desired.
- **Hybrid model:** combining fine-tuning (this idea) with positional reconstruction
  (Idea 1) may capture the complementary strengths of both approaches — see the shared
  report's Discussion/Conclusion.
- **The predictive-power vs. interpretability trade-off** (Section 7) is worth
  investigating directly rather than only observing as a side effect — e.g. by
  explicitly regularising for per-protein motif specificity during fine-tuning, or by
  probing whether the trade-off is intrinsic to backbone capacity or an artefact of how
  a single aggregate output head is optimised.

---

## 12. Contacts

Idea 2 (this repo): globalCLIP-head deconvolution. Idea 1 (separate): a reconstruction
head combining the 223 existing tracks. Supervisor: Lambert Moyon.

All ParNet-ecosystem dependencies live under the
[`project-parnet`](https://github.com/project-parnet) GitHub org, including
`parnet--demo--train-models` (source of `parnet_demo_utils`), `pylbsr`, `metamotif`,
`parnet_additional_utils`, and `parnet_analyses_libs`.