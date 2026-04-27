## Setup (macOS)

### Prereqs
- Python **3.11** (this project uses wheels pinned for 3.11)
- Homebrew (recommended for installing Python 3.11)

### Create a virtualenv + install deps

```bash
cd [YOUR PROJECT FOLDER PATH]

brew install python@3.11
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt
```

### Quick sanity check (downloads model + verifies attention hook)

```bash
source .venv/bin/activate
python p2p_smoke_test.py
```

You should see output like:
- `OK: attention store counts: ...`
- `OK: num_att_layers: ...` (must be > 0)

### Running the notebook

- In your IDE, select the Python interpreter at `./.venv/bin/python` as the notebook kernel.
- Run `prompt-to-prompt_ldm.ipynb`.

