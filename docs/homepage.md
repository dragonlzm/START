# START homepage

The homepage is a buildless static site at `index.html`. Styling and small progressive-enhancement interactions live in `assets/`; paper figures are reused from `images/`. No dependency installation or build step is required.

## Preview locally

From the repository root:

```bash
python3 -m http.server 8765 --bind 127.0.0.1
```

Open <http://127.0.0.1:8765/>. Opening `index.html` directly also works; clipboard buttons fall back to selecting text when the Clipboard API is unavailable.

## Publishing later

No publication or remote changes were made when this page was created. The README badge and canonical metadata use the intended address, <https://dragonlzm.github.io/START/>.

After committing and pushing the page, configure GitHub Pages to serve the repository root from the desired branch (normally `main`). The root `.nojekyll` marks this as a plain static site. The required files are `index.html`, `assets/`, and the four referenced images in `images/`. Keep relative paths intact so the site works under `/START/`.

## Section plan and content sources

| Section | Visitor question | Content and evidence |
| --- | --- | --- |
| Introduction | What does START contribute? | Paper title, authors, affiliations, grounding / code / QA framing from the repository and paper. |
| START-Dataset | What can I train on and download? | Released SFT/RL annotation and image counts, verified SFT composition, data-generation pipeline, raw and prepared data links. |
| CS-Bench | What does the benchmark measure? | Released question/image counts, two task families, a real grounding question, localization and QA metrics, evaluation links. |
| Method Overview | How is the data used? | Spatial grounding, textual code generation, and chart QA in SFT/RL; link to paper Section 3.1. |
| Performance | How does the released model perform? | Exact two-model release table from README, with absolute score improvements. |
| Qualitative Examples | What does supervision look like? | Original dataset figure plus a condensed, explicitly labeled released multi-round SFT annotation example. |
| Resources | Where do I get started? | Data, benchmark, raw code, checkpoint, repository, paper, configurable evaluation commands, training-code availability. |
| Citation | How should I cite or contact the authors? | README BibTeX and author contact. |

Primary sources:

- [Repository README](../README.md)
- [CS-Bench evaluation instructions](../eval/cs_bench/README.md)
- [CS-Bench evaluator](../eval/cs_bench/compute_the_stat_from_eval_file.py)
- [Released predictions](../predictions/cs_bench_predictions.json)
- [START paper](https://arxiv.org/html/2512.07186v1)
- [START-Dataset](https://huggingface.co/datasets/zhuomingliu/START-Dataset)
- [CS-Bench](https://huggingface.co/datasets/zhuomingliu/CS-Bench)

## Content distinctions to preserve

- Release statistics and results are reproduced resources, not the original paper's experimental configuration. CS-Bench has 600 unique images in the public release; the paper reports 613. Original figures are captioned accordingly.
- SFT and RL image counts are per split. Do not add them to claim a deduplicated image total.
- README's RL category rows account for only 30,662 of the stated 39,211 annotations. The homepage does not present that incomplete breakdown as a complete distribution. The SFT rows sum to 399,897 and are included.
- Performance scores are reported release scores, not newly recomputed evaluations. The original paper's `performance_figure.png` is not used as evidence for this reproduced checkpoint table.
- The multi-round conversation is a training annotation example, not a live demo or a new prediction.
- Training code remains listed as coming soon in the source README.

## Design and maintenance

The page follows the article structure of [MaLoW](https://dragonlzm.github.io/MaLoW/) and [DRIFT](https://dragonlzm.github.io/DRIFT/), with a teal palette and original HTML/CSS/JS implementation. The footer retains a design credit to the Clarity Template and Shikun Liu. No external fonts, UI libraries, analytics, or runtime services are required.

Update colors in `assets/site.css` under `:root`. Update page copy and tables in `index.html`, keeping sources and paper/release distinctions intact. If changing cached assets during local development, reload without cache or increment the script version query.
