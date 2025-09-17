# Open LLM Explorer (Gradio)

Quick setup and run instructions for this demo.

Prerequisites
- Python 3.11+ (this project used Python 3.13 on macOS)
- Git (optional)

Create and activate a virtual environment (macOS / zsh):

```bash
cd /path/to/gradio
/opt/homebrew/bin/python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the app:

```bash
source .venv/bin/activate
python app.py
```

The app listens on http://127.0.0.1:7860 by default.

Hugging Face Inference API token
- You can paste a token into the password box in the UI for one-off requests.
- Or set the env var before launching the server:

```bash
export HF_TOKEN="hf_your_token_here"
python app.py
```

- Alternatively, log in with the Hugging Face CLI (saves credentials to your user):

```bash
source .venv/bin/activate
python -m huggingface_hub.cli login
# then follow the prompt to paste your token
```

Security note
- Do NOT commit your HF token to the repo. Use environment variables or the UI for transient usage.
