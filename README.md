# Bachelors Project

Dataset was found here -> https://github.com/garima2751/TF_CDN/blob/main/Data/cdn_pairs.csv
Paper that used that dataset -> https://arxiv.org/pdf/2301.03709

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -U pip
pip install pandas scikit-learn python-dotenv openai litellm
```

Create a `.env` file (or export env vars) with your API key:

```bash
OPENAI_API_KEY=...your key...
```

## 1) Split dataset (70/20/10 train/test/val)

Creates `train.csv`, `test.csv`, `val.csv`.

By default it also keeps all rows with the same `Text1` together in the same split (`--group-by text1`) to avoid leakage.
This dataset is often organized in blocks where `Text1` repeats many times.

```bash
python split_dataset.py --csv cdn_pairs.csv --train-out train.csv --test-out test.csv --val-out val.csv
```

If you want the classic row-level stratified split (may mix the same `Text1` across splits):

```bash
python split_dataset.py --csv cdn_pairs.csv --group-by none
```

## 2) Run LLM predictions

Script: `demo_code.py`

It reads `Text1`, `Text2`, `Class` from the input CSV, calls an LLM to predict the label, and appends results to a JSONL file.

By default, the JSONL stores `row_index` + labels/metadata (not the raw `Text1`/`Text2`) so files stay small.
You can still recover the texts later by looking up the row in the source CSV.

### OpenAI SDK backend (default)

```bash
python demo_code.py --csv train.csv --model gpt-4o --limit 10
```

### LiteLLM backend

```bash
python demo_code.py --backend litellm --csv train.csv --model gpt-4o --limit 10
```

Notes:

- For `--backend litellm`, if `--model` does not contain a provider prefix, it will be treated as `openai/<model>`.
- `--limit` is the number of rows processed from the start of the CSV.

### Use a config file (TOML)

Instead of passing many CLI flags, you can put defaults in a TOML config and override only what you need.

See `config.example.toml` for a ready-to-copy template.

Example config (either flat keys or a `[demo_code]` section are supported):

```toml
[demo_code]
backend = "litellm"
csv = "train.csv"
model = "gpt-4o"
prompt_name = "roses_v1"
pred_dir = "predictions"
limit = 10
```

Run:

```bash
python demo_code.py --config config.toml
```

CLI flags override config values:

```bash
python demo_code.py --config config.toml --limit 50
```

### Output files (separated by dataset/model/prompt)

By default, predictions are stored under `predictions/` using a filename that includes:
`<csv>__<backend>__<model>__<prompt_name>__<prompt_id>.jsonl`

Example:

```text
predictions/train.csv__openai__gpt-4o__roses_v1__<prompt_id>.jsonl
predictions/prompts/roses_v1__<prompt_id>.txt
```

You can override the output path if you want:

```bash
python demo_code.py --csv train.csv --out my_predictions.jsonl --limit 10
```

### Resume behavior (don’t re-call the API)

The script reads the output JSONL file on startup and skips any `row_index` values already present there.
This makes it safe to re-run the script without repeating API calls for already-processed rows.

## 3) Evaluate predictions (metrics)

Script: `evaluate_predictions.py`

It computes:

- accuracy
- precision
- recall
- f1-score

Example (one JSONL):

```bash
python evaluate_predictions.py --pred predictions/train.csv__openai__gpt-4o__roses_v1__<prompt_id>.jsonl
```

Example (glob multiple JSONL files):

```bash
python evaluate_predictions.py --pred "predictions/*.jsonl"
```

By default it uses `--average macro`. If your JSONL contains invalid/empty labels and you still want them included, use `--include-invalid`.

## 4) ML baseline (scikit-learn Decision Tree)

Script: `sklearn_decision_tree.py`

Train on `train.csv`, then write JSONL predictions for `val.csv` / `test.csv`:

```bash
python sklearn_decision_tree.py --train train.csv --eval val.csv --eval test.csv
```

You can control the Decision Tree complexity via `--max-depth` (default: 6):

```bash
python sklearn_decision_tree.py --train train.csv --eval val.csv --eval test.csv --max-depth 6
```

Evaluate the baseline predictions:

```bash
python evaluate_predictions.py --pred "predictions/val.csv__sklearn__decision_tree.jsonl"
python evaluate_predictions.py --pred "predictions/test.csv__sklearn__decision_tree.jsonl"
```

## ) Inspect mistakes (join back to CSV)

Because prediction JSONL files store `row_index` (not `Text1`/`Text2`), you can reconstruct the texts from the CSV when you want to inspect misclassifications.

Script: `inspect_errors.py`

Example:

```bash
python inspect_errors.py --pred "predictions/*.jsonl" --out errors.csv
```

By default it writes only mistakes (where `pred_label != true_label`). Use `--all` to include correct predictions too.

grit, random search