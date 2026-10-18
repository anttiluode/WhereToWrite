# WhereToWrite

**An interactive microscope for WHERE a remembered value enters a frozen transformer.**

This repo isolates a single causal question suggested by [LiveMode](https://github.com/anttiluode/LiveMode) and [RealTimeReadWrite](https://github.com/anttiluode/RealTimeReadWrite):

> Does injecting a remembered property onto the **subject's own token** allow the model's existing computation to carry it to the answer, compared to inserting the same property directly at the **answer token**?

It **does not** claim to solve autonomous live learning. The writer uses a transparent exact dictionary; both the entity address and the value are supplied explicitly. The experiment varies **only injection location and direction**. The base Qwen weights remain frozen. There is no extra training, no hash addresses and no 30-minute J-lens computation in the default path.

## Measured result: 400 evaluations at gain 2, layer 16 (October 18, 2026)

**The privileged subject-token hypothesis is not supported at this setting.** See the [committed report](runs/site_gain2_n400.json) and [research ledger](docs/W1_GAIN2_400.md).

| Injection site | Four-color accuracy | Unrestricted exact-token accuracy |
|---|---:|---:|
| None | 31.25% | 9.25% |
| Subject name | 42.25% | 11.50% |
| Unrelated word `color` | **45.25%** | 9.25% |
| Final answer token | **100%** | **64%** |
| Random direction at subject | 30.50% | 8.50% |
| Wrong-color direction at subject | 22.50% | 6.25% |

The paired subject-minus-unrelated-site contrast is **−3.0 percentage points** (31 subject-only correct, 43 unrelated-token-only correct, 326 ties). There is no observed advantage for writing on the name token over the word `color`. Both are better than random-direction injection on this four-color diagnostic; the answer site gives strong direct steering, not native associative retrieval.

**Statistical correction:** the fixed-seed 400-evaluation schedule contains only **95 distinct** name × prompt × color cases and **305 duplicates**. The original reported normal 95% interval `[-0.0721, 0.0121]` relies on an independence assumption violated by these repeats; it is a historical exploratory statistic, *not* a defensible independent-trial interval. Only eight names and three phrasings were used. The result is about **logit-lens** directions at gain 2, not Naula's J-lens configuration.

A smaller balanced follow-up is now supported, testing each of the **96 distinct** cases exactly once and saving all per-case predictions:

```bat
python3.13 -m where_to_write.benchmark --design balanced --gain 2 --report runs/site_gain2_balanced96.json
```

The balanced grid improves coverage and auditability but is **not a new independent holdout**; its names, colors, and prompt templates were already tested. Avoid tuning on these cases and calling the result prospective.

## Windows GUI — start here

Requires Python 3.10+, Windows with Tkinter, and ideally an NVIDIA GPU. First run downloads Qwen 0.5B (~1 GB).

```bat
cd E:\DocsHouse\1018\WhereToWrite
python3.13 -m pip install -e ".[qwen]"
python3.13 -m where_to_write
```

1. Click **Load Qwen**. Model defaults to `Qwen/Qwen2.5-0.5B-Instruct`, **layer 16**, **gain 0.5** (selected from the *earlier RealTimeReadWrite dev sweep*, not optimized on this repo's held-out results).
2. Enter a name and color, then **WRITE / rewrite**. The fact changes *outside* the model.
3. Click **ASK + compare**. Watch six results: no intervention, correct memory injected at subject, correct memory injected at final answer, same pulse on another token (`color`), random pulse at subject, and **wrong color** injected at subject.
4. Rewrite Vera from yellow to blue; ask again. The prompt never includes the remembered color. The experiment executes fresh forward passes with the current value.

The bars are probabilities **normalized over just the four colors**, not actual full-vocabulary probabilities. A separate `strict` marker indicates whether the correct color token wins over *every* Qwen token. A model changing its answer is **not** evidence it reasoned correctly with the information.

## Short CLI benchmark

```bat
python3.13 -m where_to_write.benchmark --trials 12 --report runs/where_to_write.json
```

The benchmark runs 12 fixed-seed prompts using eight names and three supported phrasings. It measures strict full-vocabulary next-token accuracy, four-color choice, and margins for each matched control. **No numbers are claimed before a real Qwen run.** Use `--trials 24` to run longer. Do **not** tune gain against the reported test names and then claim it was held out.

## Optional J-lens comparison

RealTimeReadWrite's G4.2 measured **both J-lens and logit-lens steering at 100%** at the answer token. The default in this repo therefore deliberately uses the simpler gamma-weighted unembedding direction, with no J-lens dependency. If you have the **trusted, matching** `jlens_*.pt` cache from RealTimeReadWrite, use it optionally:

```bat
python3.13 -m where_to_write.benchmark --lens-cache "PATH_TO_YOUR_EXISTING_JLENS_CACHE.pt" --trials 12
```

**Check your actual cache filename**: the example path above is illustrative and the numbers after `128x` depend on the saved run's settings. Never download arbitrary `.pt` files for `torch.load`.

## Gate W1: exact causal site comparisons

- W1a: Answer-site correct-value intervention beats no memory (technical calibration; likely easy).
- W1b: Subject-site correct-value intervention exceeds both **random-direction** and **wrong-color** subject controls on the same questions.
- W1c: Subject-site intervention beats the identical pulse on the unrelated `color` token at a fixed layer/gain, establishing *site specificity*.
- W1d: Withheld from this gate: actual attention-head attribution; a downstream effect at subject is consistent with attention-mediated transmission but does not identify which head transports it.

Only W1a–c are scored in the batch; no thresholds were retrospectively registered. If W1b/W1c fail, retain the failure. Later experiments may ablate candidate attention heads under explicit new protocols.

## Mechanism

Qwen2.5-0.5B-Instruct, layer 16 after the transformer block. For a color `c` define a unit vector from the model's own unembedding row `u_c` and final-normalization weight `gamma`:

`v_c = normalize(gamma * u_c)`

At position `p` inject `h[layer, p] += gain * ||h[layer, p]|| * v_c`. Compare otherwise identical prompts. The positions are located via **fast-tokenizer character offsets** (including multi-token names); the injection is placed on the *last subtoken* of the name. The `color` token is a position-control. Random and wrong-color pulses use the same site and norm.

The exact dictionary uses negligible bytes and is a *stronger baseline* than the learned, interference-sensitive memories from LiveMode. This microscope tests causal routing, **not memory compression** or discovery of a new K-space.

## Sources and credit

- [RealTimeReadWrite](https://github.com/anttiluode/RealTimeReadWrite) G4.2: 100% answer-token J/logit steering, 56.7% subject-site effect (the empirical motivation).
- [LiveMode](https://github.com/anttiluode/LiveMode): trained-interface and continuous-memory gates.
- Gurnee et al., [*Verbalizable Representations Form a Global Workspace in Language Models*](https://arxiv.org/abs/2607.15495) (2026).

Run pure-Python checks with `python -m unittest discover -s tests -v` before downloading a model.


## Follow-up after Naula N2 (October 18, 2026)

Naula's 432-trial J-lens experiment found **56.7%** name-site accuracy versus **48.4%** at a different earlier token with gain 8. The name-site advantage is **still OPEN** and might reflect a generic large perturbation. WhereToWrite's default uses a **logit-lens** direction, not Naula's J-lens, so its performance cannot be inferred from Naula. N5 confirmed name decodability; N3 confirmed a nonmonotonic steering region; the proposed N6 slow-core/fast-residue dichotomy failed in its test.

The GUI gain slider now reaches **8**; keep **0.5** as the original reference. Gains from 2 to 8 inject **2–8 times the local activation norm**, not a subtle physiological pulse. Use the unrelated-word control as seriously as the subject site.

The CLI batch now evaluates the six arms on separate rows of **one frozen Qwen forward per question** to reduce repeated GPU overhead. A mock-transformer regression test checks agreement with the original six-pass code. The benchmark reports both strict lowercase-token accuracy and separately case-insensitive top-color correctness (e.g. ` Blue`), plus paired subject-versus-other-token wins and an **exploratory approximate paired 95% interval**. It is not an independently preregistered gate.

Try a 400-trial comparison when convenient (still may take minutes; don't tune on the same prompts and call the best result held-out):

```bat
python3.13 -m where_to_write.benchmark --gain 2 --trials 400 --report runs/site_gain2_n400.json
```

Optionally test gain 8 in a **separate report file**. The exact oracle dictionary still supplies entity and color; nothing here demonstrates autonomous memory or attention-head attribution.
