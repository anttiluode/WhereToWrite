"""Short, fixed-seed, no-tuning batch for WhereToWrite.

Exact entity-color lookup is shared by all conditions. Does NOT test autonomous
memory, capacity, J-space, multi-hop reasoning, or information extraction.
"""
import argparse
import json
from pathlib import Path
import random

from .memory import COLORS
from .probe import QwenProbe, ARMS, PROMPTS

NAMES = ('Vera', 'Eero', 'Aino', 'John', 'Mary', 'David', 'Cleo', 'Xavier')


def run(probe, trials=12, gain=0.5, seed=20261018, out='runs/where_to_write.json'):
    rng = random.Random(seed)
    hits = {arm: {'choice': 0, 'strict': 0, 'sum_margin': 0.0} for arm in ARMS}
    examples = []
    for j in range(trials):
        name = NAMES[j % len(NAMES)]
        color = COLORS[rng.randrange(len(COLORS))]
        template = PROMPTS[(j // len(NAMES)) % len(PROMPTS)]
        result = probe.evaluate(name, color, template, gain)
        for arm, row in result.items():
            hits[arm]['choice'] += int(row.choice_correct)
            hits[arm]['strict'] += int(row.strict_correct)
            hits[arm]['sum_margin'] += row.true_margin
        examples.append({'name':name, 'color':color, 'choices':{a:r.choice for a,r in result.items()}})
        print(f'{j+1}/{trials} {name:>7} expected {color:<7}  subject={result["subject"].choice:<7}  answer={result["answer"].choice}', flush=True)
    summary = {'model':probe.model_name,'layer':probe.layer,'gain':gain,'seed':seed,
               'direction':probe.direction_kind,'n':trials,
               'conditions':{arm:{'choice_accuracy':round(v['choice']/trials,4),
                                  'strict_accuracy':round(v['strict']/trials,4),
                                  'mean_true_color_margin':round(v['sum_margin']/trials,4)}
                             for arm, v in hits.items()}, 'examples':examples,
               'disclosure':'Oracle dictionary/address/value; no learned native binding. Color-choice versus unrestricted next-token are different metrics.'}
    dest=Path(out);dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary['conditions'], indent=2))
    print(f'Saved {dest}')
    return summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model', default='Qwen/Qwen2.5-0.5B-Instruct')
    p.add_argument('--layer', type=int, default=16)
    p.add_argument('--gain', type=float, default=0.5)
    p.add_argument('--trials', type=int, default=12)
    p.add_argument('--lens-cache', default=None, help='OPTIONAL: trusted local RealTimeReadWrite J-lens cache, with matching Qwen weights')
    p.add_argument('--report',default='runs/where_to_write.json')
    a=p.parse_args()
    if a.trials < 1:
        p.error('--trials must be positive')
    probe=QwenProbe(a.model, layer=a.layer, j_lens_cache=a.lens_cache)
    run(probe, a.trials, a.gain, out=a.report)


if __name__=='__main__':
    main()
