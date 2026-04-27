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

### Real-photo stylized edits (img2img)

This repo now includes an img2img entry point in `ptp_utils.image2image_ldm_stable(...)` for doing **stylized edits** on a real input photo while applying Prompt-to-Prompt attention control.

At a high level:
- Load a Stable Diffusion pipeline (not the `CompVis/ldm-text2im-large-256` LDM pipeline).
- Pass your input `PIL.Image` as `init_image`.
- Use `strength` to control stylization (higher = more stylized / less faithful to the original).

#### Demo script

There is an example script you can run (after installing deps + activating your venv):

```bash
python sd_img2img_demo.py --image /path/to/photo.jpg --prompt "a watercolor painting of the same scene" --strength 0.8 --out out.png
```

#### ControlNet (composition-preserving stylization)

To make stylization less dependent on describing objects in the prompt, you can enable ControlNet (canny edges):

```bash
python sd_img2img_demo.py --controlnet canny --image /path/to/photo.jpg --prompt "make this watercolor" --out out_controlnet.png
```

