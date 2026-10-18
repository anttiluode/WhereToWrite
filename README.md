# WhereToWrite

**An interactive microscope for WHERE a remembered value enters a frozen transformer.**

This repo isolates a single causal question suggested by [LiveMode](https://github.com/anttiluode/LiveMode) and [RealTimeReadWrite](https://github.com/anttiluode/RealTimeReadWrite):

> Does injecting a remembered property onto the **subject's own token** allow the model's existing computation to carry it to the answer, compared to inserting the same property directly at the **answer token**?

It **does not** claim to solve autonomous live learning. The writer uses a transparent exact dictionary; both the entity address and the value are supplied explicitly. The experiment varies **only injection location and direction**. The base Qwen weights remain frozen. There is no extra training, no hash addresses and no 30-minute J-lens computation in the default path.

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
