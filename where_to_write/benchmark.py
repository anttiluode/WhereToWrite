"""Short, fixed-seed, no-tuning batch for WhereToWrite.

Exact entity-color lookup is shared by all conditions. Does NOT test autonomous
memory, capacity, J-space, multi-hop reasoning, or information extraction.
"""
import argparse
import json
from pathlib import Path
import random
import math

from .memory import COLORS
from .probe import QwenProbe, ARMS, PROMPTS

NAMES = ('Vera', 'Eero', 'Aino', 'John', 'Mary', 'David', 'Cleo', 'Xavier')


def case_schedule(trials=12, seed=20261018, design='legacy'):
    """Legacy repeated schedule or all 96 unique name/prompt/color cases."""
    rng = random.Random(seed)
    if design == 'balanced':
        cases = [(n, c, t) for n in NAMES for t in PROMPTS for c in COLORS]
        rng.shuffle(cases)
        return cases
    if design != 'legacy':
        raise ValueError('design must be legacy or balanced')
    if trials < 1:
        raise ValueError('trials must be positive')
    return [(NAMES[j % len(NAMES)], COLORS[rng.randrange(len(COLORS))],
             PROMPTS[(j // len(NAMES)) % len(PROMPTS)]) for j in range(trials)]


def run(probe, trials=12, gain=0.5, seed=20261018, out='runs/where_to_write.json',
        design='legacy'):
    cases = case_schedule(trials, seed, design)
    trials = len(cases)
    distinct_cases = len(set(cases))
    hits = {arm: {'choice': 0, 'strict': 0, 'semantic': 0, 'sum_margin': 0.0} for arm in ARMS}
    site_delta = []  # paired subject minus identical pulse on word color
    examples = []
    case_results = []
    for j, (name, color, template) in enumerate(cases):
        result = probe.evaluate(name, color, template, gain)
        for arm, row in result.items():
            hits[arm]['choice'] += int(row.choice_correct)
            hits[arm]['strict'] += int(row.strict_correct)
            hits[arm]['semantic'] += int(row.semantic_top_correct)
            hits[arm]['sum_margin'] += row.true_margin
        site_delta.append(int(result['subject'].choice_correct) - int(result['other_token'].choice_correct))
        if len(examples) < 24:
            examples.append({'name':name, 'color':color, 'choices':{a:r.choice for a,r in result.items()}})
        case_results.append({
            'name': name, 'color': color, 'template': template,
            'arms': {arm: {'choice': row.choice, 'choice_correct': row.choice_correct,
                           'strict_correct': row.strict_correct,
                           'semantic_top_correct': row.semantic_top_correct}
                     for arm, row in result.items()}
        })
        if trials <= 48 or (j+1) % 25 == 0 or j+1 == trials:
            print(f'{j+1}/{trials} {name:>7} expected {color:<7}  subject={result["subject"].choice:<7}  answer={result["answer"].choice}', flush=True)
    mean_delta = sum(site_delta) / trials
    sample_var = (sum((d - mean_delta)**2 for d in site_delta) / (trials - 1)) if trials > 1 else 0.0
    ci = [max(-1.0, mean_delta - 1.96 * math.sqrt(sample_var/trials)),
          min(1.0, mean_delta + 1.96 * math.sqrt(sample_var/trials))]
    summary = {'model':probe.model_name,'layer':probe.layer,'gain':gain,'seed':seed,
               'direction':probe.direction_kind,'n':trials,
               'design':design, 'distinct_cases':distinct_cases,
               'duplicate_cases':trials-distinct_cases,
               'conditions':{arm:{'choice_accuracy':round(v['choice']/trials,4),
                                  'strict_accuracy':round(v['strict']/trials,4),
                                  'semantic_top_color_accuracy':round(v['semantic']/trials,4),
                                  'mean_true_color_margin':round(v['sum_margin']/trials,4)}
                             for arm, v in hits.items()},
               'paired_site_comparison': {
                   'subject_minus_color_token_accuracy': round(mean_delta,4),
                   'approx_normal_95_ci': [round(x,4) for x in ci],
                   'subject_only_correct': site_delta.count(1),
                   'color_token_only_correct': site_delta.count(-1),
                   'ties': site_delta.count(0),
                   'metric': 'four_color_accuracy',
                   'note': ('Descriptive only: repeated cases violate the independent-observation '
                            'assumption behind this normal interval.' if distinct_cases < trials else
                            'Exploratory: cases are unique but share names/prompts; not a holdout.')},
               'examples':examples,'case_results':case_results,
               'disclosure':'Oracle dictionary/address/value; no learned native binding. Color-choice versus unrestricted next-token are different metrics.'}
    dest=Path(out);dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(f'Case design: {design}; {distinct_cases} unique of {trials} cases ({trials-distinct_cases} repeats).')
    print(json.dumps(summary['conditions'], indent=2))
    print('Subject versus same pulse at word color:', json.dumps(summary['paired_site_comparison']))
    print(f'Saved {dest}')
    return summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model', default='Qwen/Qwen2.5-0.5B-Instruct')
    p.add_argument('--layer', type=int, default=16)
    p.add_argument('--gain', type=float, default=0.5)
    p.add_argument('--trials', type=int, default=12, help='Applies to legacy repeated schedule only')
    p.add_argument('--design', choices=['legacy','balanced'], default='legacy',
                   help='balanced: all 96 unique name x phrase x color cases')
    p.add_argument('--lens-cache', default=None, help='OPTIONAL: trusted local RealTimeReadWrite J-lens cache, with matching Qwen weights')
    p.add_argument('--report',default='runs/where_to_write.json')
    a=p.parse_args()
    if a.trials < 1:
        p.error('--trials must be positive')
    if not 0 <= a.gain <= 8:
        p.error('--gain must lie between 0 and 8')
    probe=QwenProbe(a.model, layer=a.layer, j_lens_cache=a.lens_cache)
    run(probe, a.trials, a.gain, out=a.report, design=a.design)


if __name__=='__main__':
    main()
