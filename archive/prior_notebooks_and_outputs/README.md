# Prior notebooks and outputs (bookkeeping)

This folder holds **bookkeeping** for work that lived in the repo root before cleanup.

- **`angela.ipynb`**, **`null_inversion_testing.ipynb`**, **`history.txt`**, and the **`out_*.png`** renders were **never committed**; they cannot be restored from this repository after terminal deletion. Use backups or Local History if you still need the exact files.
- **`prompt-to-prompt_ldm.ipynb`** *was* tracked; the last **`HEAD`** revision has been copied into `notebooks/` here.

## What was removed (not in git — cannot restore from this repo)

| Item | Notes |
|------|--------|
| `angela.ipynb` | Large notebook (embedded run outputs). Logic is reflected in `cs5788_final_project.ipynb` + `real_image_edit.py` / `p2p_controllers.py`. |
| `null_inversion_testing.ipynb` | Same as above. |
| `history.txt` | Scratch / log file. |
| `out_cat.png`, `out_cat_v2.png`, `out_cat_v3.png` | Saved experiment renders. |
| `out_cat_controlnet.png`, `out_cat_controlnetv2.png`, `out_cat_controlnetv3.png` | ControlNet-related runs. |
| `out_cat_watercolor.png` | Watercolor-style output. |
| `out_dog_v3.png` | Subject-edit output. |
| `_nb_extract/` | Temporary per-cell `.py` dumps from a one-off export script (not source of truth). |

## How you might still recover copies

1. **Time Machine** (or another backup): restore the files above into `notebooks/` or `outputs/` under this directory.
2. **Another clone or machine** that still has the old working tree.
3. **Cursor / VS Code Local History** (if enabled): right-click folder → *Local History* on a parent path; sometimes recovers deleted files.
4. **macOS Trash**: only if the deletes went through Finder (terminal `rm` usually bypasses Trash).

## Subfolders

- `notebooks/` — contains **`prompt-to-prompt_ldm.ipynb`** (from `git show HEAD:…`). Drop **`angela.ipynb`** / **`null_inversion_testing.ipynb`** here if you recover them from backup.
- `outputs/` — drop restored renders here; see `outputs/MANIFEST.txt` for the filenames that used to live in the repo root.

To refresh the LDM notebook from git after future commits:

```bash
git show HEAD:prompt-to-prompt_ldm.ipynb > archive/prior_notebooks_and_outputs/notebooks/prompt-to-prompt_ldm.ipynb
```
