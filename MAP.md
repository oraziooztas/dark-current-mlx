# MAP - Dark Current MLX Pilot

Questa e' la root directory per il pilot benchmark in MLX per il paper "LLM Judges Have Dark Current".

## Obiettivo
Dimostrare la riproducibilita' locale della valutazione di giudici LLM open-source usando Apple Silicon (MLX), senza assumere una collaborazione formale finche' dataset/schema non sono approvati.

## Struttura
- `eval_pipeline.py`: entry point retro-compatibile (inietta `src/` nel path, nessuna install necessaria).
- `src/dark_current_mlx/`: package Python installabile.
- `src/dark_current_mlx/cli.py`: CLI `dark-current-mlx`.
- `src/dark_current_mlx/runners.py`: runner `mock` per smoke test e runner `mlx` lazy (API sampler mlx-lm>=0.31, allineamento eos dichiarato, usage stats per call).
- `src/dark_current_mlx/pipeline.py`: run A/B + B/A, JSONL writer, summary metrics + `usage_totals`.
- `src/dark_current_mlx/verdicts.py`: parser strict `[[A]]/[[B]]/[[Tie]]` + swap-consistency.
- `src/dark_current_mlx/{models,prompts,dataset}.py`: dataclass, template prompt, loader JSONL.
- `data/sample_pairwise.jsonl`: mini dataset locale per dry run.
- `data/minislice_v0.jsonl`: mini-slice 12 item in 3 condizioni (clear_gap/identical/paraphrase).
- `scripts/category_breakdown.py`: breakdown dark current per condizione dai run JSONL.
- `schema.json`: shape del record JSONL prodotto (incluso blocco `usage`).
- `tests/`: unit test offline (8), senza modelli o download.
- `docs/usami_results_note_template.md`: template della results note per gli autori.
- `docs/usami_results_note_2026-07-03.md`: bozza results note run 02-03/07 (NON inviata; /humanizer prima dell'invio).
- `results/`: output delle run. Gitignored TRANNE i 6 file delle run mini-slice v0 (3 judge × records+summary), committati come evidenza dei numeri nel README.

## Comandi

```bash
python3 eval_pipeline.py --runner mock --input data/sample_pairwise.jsonl
PYTHONPATH=src python3 -m unittest discover -s tests
```

Per run MLX reale:

```bash
uv pip install -e ".[mlx]"
python3 eval_pipeline.py --runner mlx --model mlx-community/Meta-Llama-3-8B-Instruct-4bit \
  --input data/minislice_v0.jsonl --max-tokens 512
python3 scripts/category_breakdown.py results/*_minislice_v0.jsonl
```

Modelli pilot in cache HF locale (4-bit mlx-community): Meta-Llama-3-8B-Instruct, Mistral-7B-Instruct-v0.3, Qwen2-7B-Instruct (~13GB totali).
