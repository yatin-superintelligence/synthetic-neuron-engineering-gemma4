import json, os, sys, random, re
sys.path.insert(0, 'src')
src = open('experiments/scripts/exp088_sft_lora.py', encoding='utf-8').read()
E = {"__file__": os.path.abspath('experiments/scripts/exp088_sft_lora.py'), "__name__": "x"}
exec(compile(src.split('with Experiment("EXP-088"')[0], "exp088", "exec"), E)
s110 = open('experiments/scripts/exp110_success_distillation.py', encoding='utf-8').read()
ns = {}
exec(s110[s110.index('CAP, CUT ='):s110.index('with Experiment(EXPID')], ns)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('models/gemma-4-E2B-it-qat-unquantized')
for half in ('dev', 'test'):
    rows = [json.loads(l) for l in open(f'artifacts_local/EXP-110/success_{half}.jsonl', encoding='utf-8')]
    random.Random(1).shuffle(rows)
    n = bad = long_ = 0; lens = []; tg = []
    for r in rows[:40]:
        ids, lab = E['encode'](tok, ns['prepare'](r['messages']))
        n += 1
        if ids is None:
            bad += 1; continue
        lens.append(len(ids)); tg.append(sum(l != -100 for l in lab)); long_ += len(ids) > 16384
    lens.sort(); tg.sort()
    print(half, 'checked', n, '| render mismatch', bad, '| over 16384', long_, '| median tokens', lens[len(lens)//2] if lens else None, '| median target', tg[len(tg)//2] if tg else None)
