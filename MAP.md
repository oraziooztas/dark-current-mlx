# MAP - Dark Current MLX Pilot

Questa e' la root directory per il pilot benchmark in MLX per il paper "LLM Judges Have Dark Current".

## Obiettivo
Dimostrare la riproducibilita' locale della valutazione di giudici LLM open-source usando Apple Silicon (MLX), senza assumere una collaborazione formale finche' dataset/schema non sono approvati.

## Struttura
- `eval_pipeline.py`: entry point retro-compatibile (inietta `src/` nel path, nessuna install necessaria).
- `src/dark_current_mlx/`: package Python installabile.
- `src/dark_current_mlx/cli.py`: CLI `dark-current-mlx`.
- `src/dark_current_mlx/runners.py`: runner `mock` per smoke test e runner `mlx` lazy.
- `src/dark_current_mlx/pipeline.py`: run A/B + B/A, JSONL writer, summary metrics.
- `src/dark_current_mlx/verdicts.py`: parser strict `[[A]]/[[B]]/[[Tie]]` + swap-consistency.
- `src/dark_current_mlx/{models,prompts,dataset}.py`: dataclass, template prompt, loader JSONL.
- `data/sample_pairwise.jsonl`: mini dataset locale per dry run.
- `schema.json`: shape del record JSONL prodotto.
- `tests/`: unit test offline (8), senza modelli o download.
- `docs/usami_results_note_template.md`: template della results note per gli autori.
- `results/`: output locali generati dalle run (gitignored).

## Comandi

```bash
python3 eval_pipeline.py --runner mock --input data/sample_pairwise.jsonl
PYTHONPATH=src python3 -m unittest discover -s tests
```

Per run MLX reale:

```bash
uv pip install -e ".[mlx]"
python3 eval_pipeline.py --runner mlx --model mlx-community/Meta-Llama-3-8B-Instruct-4bit
```
