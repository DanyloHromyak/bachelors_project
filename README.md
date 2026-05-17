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

metrics: accuracy, precision, recall, f1-score
