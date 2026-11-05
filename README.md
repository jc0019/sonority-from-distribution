# How Much Sonority Structure Do Phonotactic Language Models Learn from Distribution?

Code and data for the paper [*How Much Sonority Structure Do Phonotactic Language Models Learn from Distribution?*](https://aclanthology.org/2026.blackboxnlp-1.44/) (Jia Can Xu and Mario Giulianelli, University College London), presented as an oral at the 9th BlackboxNLP Workshop, EMNLP 2026.

`sonority_probe` trains LSTM and Transformer phonotactic language models under
four input conditions (`feat`, `emb`, `perm`, `rand`), probes the recoverability
of sonority structure across representation spaces and training, and evaluates
the models on held-out sonority contours, the Hayes and White (2013)
violator–control pairs and the Daland et al. (2011) acceptability judgements.
Every figure and table in the paper comes from the commands below.

## Layout

```
README.md, LICENSE, pyproject.toml
sonority_probe/
  config.py, vocab.py, data.py, features.py, model.py   data, input conditions, models
  train.py, training_summary.py                          training; Tab. 2
  evaluate.py, bigram.py, contour.py                     §5: Figs. 5-8, Tabs. 4-5
  probe.py, per_phoneme.py, hw_trajectory.py             §4 and appendices: Figs. 2-4, 9-14, Tabs. 3, 8, 9
  analysis.py                                            sonority ranks, probe fitting, surprisal
  figures.py                                             the 16 figures
data/derived/
  CMUcleaned.csv                       18,233 CMU entries (Daland et al. 2011 materials)
  {train,val,test}_split_seed999.csv   the 80/10/10 split used in the paper
  feat_noSyl.csv                       21 articulatory features per phoneme
  Hayeswhite.csv                       40 violator/control nonce-word pairs (Hayes & White 2013)
  human_acceptability.csv              96 nonce words with human judgements (Daland et al. 2011)
```

Trained models go to `runs/` and analysis outputs to `outputs/`; neither is part
of the repository (the 240 runs with all epoch checkpoints take about 12 GB).

## Setup

Python 3.10 or later. The paper's models were trained with Python 3.14.7 and
torch 2.14.0 on the CPU of an arm64 Mac, and its figures were drawn with
matplotlib 3.10.9 (other versions lay out the text a few pixels differently).
The versions of numpy, pandas, scipy and scikit-learn do not affect the numbers.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

The commands below are written for the repository root, where `runs/` and
`outputs/` are created. (`pip install -e .` installs the dependencies and the
package; with the dependencies already installed, the commands also run from
the repository root without installing.)

## Reproducing the paper

All analyses use seeds 101–120 (240 models: three architectures × four
conditions × twenty seeds); these are the defaults. Developmental analyses use
the untrained model, epochs 1–10 and epochs 15, 20, 25, 30, 35 and 40.

```bash
# 1. Training. Each run saves every epoch checkpoint; completed runs are skipped on rerun.
python -m sonority_probe.train --arch lstm
python -m sonority_probe.train --arch tf2
python -m sonority_probe.train --arch tf4
python -m sonority_probe.training_summary --runs-dir runs --out outputs/training   # Tab. 2

# 2. Behaviour on the nonce words (Figs. 6-7, Tabs. 4-5) and held-out contours (Figs. 5, 8)
python -m sonority_probe.evaluate --runs-dir runs --out outputs/eval
python -m sonority_probe.bigram --out outputs/eval
python -m sonority_probe.contour --runs-dir runs --out outputs/eval

# 3. Probes and RSA (Figs. 2-4, 9, 12-14, Tabs. 3, 8, 9); per-phoneme displacement (Fig. 10);
#    violator-control discrimination across training (Fig. 11)
python -m sonority_probe.probe --mode endpoint   --runs-dir runs --out outputs/probe
python -m sonority_probe.probe --mode trajectory --runs-dir runs --out outputs/probe
python -m sonority_probe.per_phoneme --runs-dir runs --out outputs/per_phoneme
python -m sonority_probe.hw_trajectory --runs-dir runs --out outputs/hw_trajectory

# 4. Figures, into outputs/figures (add --latex-dir <dir> to copy them into the paper's directory)
python -m sonority_probe.figures
```

The probes are the slow part (about a minute per run and mode on one core):
`probe` scores the runs in parallel (`--workers N`), writes one shard per run
and skips runs whose shard exists, so an interrupted batch can be resumed
(`--merge-only` rebuilds the CSVs from the shards). The other analyses take a
few minutes in total. The figure script needs all twenty seeds.

Training is deterministic for a given machine and number of CPU threads. The
LSTM results reproduce to about 1e-7 across platforms; the Transformer forward
pass varies at the 1e-6 level with the platform and the number of threads, so
its losses, surprisals and probe scores differ in the later decimals, with the
same best epochs and conclusions.

## Data sources

`CMUcleaned.csv`, `feat_noSyl.csv`, `Hayeswhite.csv` and
`human_acceptability.csv` derive from the materials of Daland et al. (2011)
and Hayes and White (2013), kindly shared by James White; the pronunciations
come from the CMU Pronouncing Dictionary (Weide, 1998). Please cite these
sources when using the data.

- Robert Daland, Bruce Hayes, James White, Marc Garellek, Andrea Davis and
  Ingrid Norrmann. 2011. Explaining sonority projection effects. *Phonology*
  28(2):197–234.
- Bruce Hayes and James White. 2013. Phonological naturalness and phonotactic
  learning. *Linguistic Inquiry* 44(1):45–75.
- R. L. Weide. 1998. The CMU pronouncing dictionary.

## Citation

```bibtex
@inproceedings{xu-giulianelli-2026-much,
    title     = "How Much Sonority Structure Do Phonotactic Language Models Learn from Distribution?",
    author    = "Xu, Jia Can and Giulianelli, Mario",
    booktitle = "Proceedings of the 9th {B}lackbox{NLP} Workshop: Analyzing and Interpreting Neural Networks for {NLP}",
    month     = oct,
    year      = "2026",
    address   = "Budapest, Hungary",
    publisher = "Association for Computational Linguistics",
    url       = "https://aclanthology.org/2026.blackboxnlp-1.44/",
    doi       = "10.18653/2026.blackboxnlp-1.44",
    pages     = "761--782"
}
```

## Licence

The code is released under the MIT licence (see `LICENSE`). The data files are
subject to their sources' terms as described above.
