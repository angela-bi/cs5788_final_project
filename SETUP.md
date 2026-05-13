## Setup (macOS)

### Prereqs
- Python **3.11** (recommended; wheels are well tested) or **3.12–3.14** with updated `requirements.txt`
- Homebrew (optional; useful for installing Python 3.11)

### Create a virtualenv + install deps

```bash
cd [YOUR PROJECT FOLDER PATH]

brew install python@3.11
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate

python -m pip install -U pip
pip install -r requirements.txt
```

### If `pip` is broken (`No module named pip._vendor.rich...` or `No module named pip.__main__`)

The venv’s pip install can get corrupted. Reinstall pip **without** using the broken `pip` executable:

```bash
source .venv/bin/activate
curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
python /tmp/get-pip.py
python -m pip install -U pip
pip install -r requirements.txt
```

If problems persist, remove the venv and recreate it: `rm -rf .venv` then repeat the steps above.

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
- Open and run `cs5788_final_project.ipynb`.

