# Chapter 2: Noisy Sparse Observation Enhancement

This directory keeps the Chapter 2 paper workflow focused on four representative enhancement algorithms only:

```text
knn_mean
median
graph_laplacian
amr_gtv
```

`noisy` is still computed internally as the non-enhancement baseline for Gain, but it is not counted as one of the four comparison algorithms. `amr_gtv` is the proposed method. It estimates point-wise observation confidence from noisy sparse observations, builds multi-scale kNN graphs, applies robust graph regularization, and uses an adaptive local-median prior for low-confidence points.

## Run Experiments

From the project root with the MMGN Python environment:

```bash
python -m experiments.chapter2.noiseEnhance.run_focused_experiments
```

The command writes focused CSV tables to `outputs/chapter2/metrics` and paper figures to `outputs/chapter2/figures`.

## Build Word

After the experiment outputs exist, build the integrated thesis chapter:

```bash
python docs/thesis/chapter2/tools/generate_chapter2_integrated_docx.py
```

If Word generation fails with a missing `docx` module, install it once:

```bash
conda install -n mmgn -c conda-forge python-docx
```

The generated chapter uses the same four-algorithm scope as the focused experiment script.

Generated Word files and all previous chapter versions are kept in `docs/thesis/chapter2/`.
For the rectified version, run `python docs/thesis/chapter2/tools/generate_chapter2_rectified_docx.py`.
