
import os
from datetime import datetime, timedelta
import random
from functools import partial
import gradio as gr
from huggingface_hub import InferenceClient
import threading
# Try to load .env variables if python-dotenv is installed. This lets users keep HF_TOKEN in .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # If python-dotenv isn't installed, we'll fall back to requiring the environment to be set.
    pass

css = """
gradio-app {
    background: none !important;
}
.md .container {
    border:1px solid #ccc; 
    border-radius:5px; 
    min-height:300px;
    color: #666;
    display: flex;
    justify-content: center;
    align-items: center;
    text-align: center;
    font-family: monospace;
    padding: 10px;
}
#hf_token_box {
    transition: height 1s ease-out, opacity 1s ease-out;
}
#hf_token_box.abc {
    height: 0;
    opacity: 0;
    overflow: hidden;
}
#generate_button {
    transition: background-color 1s ease-out, color 1s ease-out; border-color 1s ease-out;
}
#generate_button.changed {
    background: black !important;
    border-color: black !important; 
    color: white !important;
}
"""

js = """
function refresh() {
    const url = new URL(window.location);
    if (url.searchParams.get('__theme') === 'dark') {
        url.searchParams.set('__theme', 'light');
        window.location.href = url.href;
    }
}
"""

system_prompt = """
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
"""

code = """
```python
from huggingface_hub import InferenceClient
SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
PROMPT = "{PROMPT}"
MODEL_NAME = "meta-llama/Meta-Llama-3-70b-Instruct"  # or "driaforall/mem-agent" or "google/vaultgemma-1b"
messages = [
    {"role": "system", "content": SYSTEM_PROMPT}, 
    {"role": "user", "content": PROMPT}
]
client = InferenceClient(model=MODEL_NAME, token=HF_TOKEN)
for c in client.chat_completion(messages, max_tokens=200, stream=True):
    token = c.choices[0].delta.content
    print(token, end="")
```
"""

ip_requests = {}
ip_requests_lock = threading.Lock()

def allow_ip(request: gr.Request, show_error=True):
    ip = request.headers.get("X-Forwarded-For")
    now = datetime.now()
    window = timedelta(hours=24)
    with ip_requests_lock:
        if ip in ip_requests:
            ip_requests[ip] = [timestamp for timestamp in ip_requests[ip] if now - timestamp < window]
        if len(ip_requests.get(ip, [])) >= 15:
            raise gr.Error("Rate limit exceeded. Please try again tomorrow or use your Hugging Face Pro token.", visible=show_error)
        ip_requests.setdefault(ip, []).append(now)
        print("ip_requests", ip_requests)
    # Do not return a value here. Gradio expects pre-call functions used with `gr.on(..., fn=...)` to
    # either raise a gr.Error (to block the call) or return nothing. Returning a value triggers
    # the "returned too many output values" warning.

def inference(prompt, hf_token, model, model_name):
    # Here `model` is the model ID selected from the dropdown (e.g. 'gpt2')
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
    if hf_token is None or not hf_token.strip():
        hf_token = os.getenv("HF_TOKEN")
    # If no human-friendly model_name was provided, use the model id selected from the dropdown
    if not model_name:
        model_name = model
    # If no token is provided, yield a friendly message rather than letting the HF client raise
    # and crash the Gradio queue worker.
    if not hf_token:
        yield f"**`{model_name}`**\n\nNo Hugging Face API token provided. Paste a token in the UI, set the HF_TOKEN env var, or run `huggingface-cli login`."
        return

    try:
        client = InferenceClient(model=model, token=hf_token)
        tokens = f"**`{model_name}`**\n\n"
        # First try the chat_completion streaming API
        try:
            for completion in client.chat_completion(messages, max_tokens=200, stream=True):
                # Some streamed events may not have `delta.content` (e.g., role/meta events); guard access.
                try:
                    token = completion.choices[0].delta.content
                except Exception:
                    # skip events without text content
                    continue
                if token:
                    tokens += token
                    yield tokens
            return
        except Exception:
            # If chat_completion isn't supported for this model/provider, fall back to text_generation
            pass

        # Fallback: try text generation endpoint
        try:
            gen_resp = client.text_generation(messages[-1]["content"], max_new_tokens=200)
            # text_generation may return a dict with 'generated_text' or similar
            out_text = None
            if isinstance(gen_resp, dict):
                out_text = gen_resp.get("generated_text") or gen_resp.get("text")
            elif isinstance(gen_resp, list) and len(gen_resp) > 0:
                # sometimes a list of generations is returned
                first = gen_resp[0]
                if isinstance(first, dict):
                    out_text = first.get("generated_text") or first.get("text")
                else:
                    out_text = str(first)
            if out_text:
                yield tokens + out_text
                return
        except Exception:
            # if fallback fails, we'll handle below
            pass
    except ValueError as ve:
        # Common case: missing/invalid token or auth required
        yield f"**`{model_name}`**\n\nError: {ve}.\n\nMake sure your token has Inference API access or run `huggingface-cli login`."
    except StopIteration:
        # Provider selection failed inside the HF helper
        yield f"**`{model_name}`**\n\nError: no provider available for the requested model. Check model name and account access."
    except Exception as e:
        # Generic catch-all to avoid crashing the Gradio worker loop; show error to user.
        yield f"**`{model_name}`**\n\nUnexpected error contacting Hugging Face Inference API: {e}"

def random_prompt():
    return random.choice([
        "Give me 5 very different ways to say the following sentence: 'The quick brown fox jumps over the lazy dog.'",
        "Write a summary of the plot of the movie 'Inception' using only emojis.",
        "Write a sentence with the words 'serendipity', 'baguette', and 'C++'.",
        "Explain the concept of 'quantum entanglement' to a 5-year-old.",
        "Write a couplet about Python"
    ])

with gr.Blocks(css=css, theme="YTheme/Minecraft", js=js) as demo:
    gr.Markdown("<center><h1>🔮 Open LLM Explorer</h1></center>")
    gr.Markdown("Every LLM has its own personality! Type your prompt below and compare results from the 3 leading open models from the [Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) that are on the Hugging Face Inference API. You can sign up for [Hugging Face Pro](https://huggingface.co/pricing#pro) and get a token to avoid rate limits.")
    prompt = gr.Textbox(random_prompt, lines=2, max_lines=5, show_label=False, info="Type your prompt here.")
    hf_token_box = gr.Textbox(lines=1, placeholder="Your Hugging Face token (not required, but a HF Pro account avoids rate limits):", show_label=False, elem_id="hf_token_box", type="password")
    with gr.Group():
        with gr.Row():
            generate_btn = gr.Button(value="Generate", elem_id="generate_button", variant="primary", size="sm")
            code_btn = gr.Button(value="View Code", elem_id="code_button", variant="secondary", size="sm")

    # Compact curated hosted model options (smaller / medium models that are usually cheaper/faster).
    # These were chosen from models that previously showed Hosted Inference availability in scans.
    curated_models = [
        "meta-llama/Meta-Llama-3-8B",
        "meta-llama/Llama-3.1-8B",
        "google/gemma-2-9b",
        "astronomer/Llama-3-70B-Special-Tokens-Adjusted",
    ]

    # Model selectors (one per output column)
    with gr.Row():
        llama_model_sel = gr.Dropdown(curated_models, value="meta-llama/Meta-Llama-3-8B", label="Model for left panel")
        nous_model_sel = gr.Dropdown(curated_models, value="meta-llama/Llama-3.1-8B", label="Model for center panel")
        zephyr_model_sel = gr.Dropdown(curated_models, value="google/gemma-2-9b", label="Model for right panel")

    with gr.Row() as output_row:
        llama_output = gr.Markdown("<div class='container'>Llama 3-70B Instruct</div>", elem_classes=["md"], height=300)
        nous_output = gr.Markdown("<div class='container'>GPT-2 (fallback)</div>", elem_classes=["md"], height=300)
        zephyr_output = gr.Markdown("<div class='container'>BLOOM (fallback)</div>", elem_classes=["md"], height=300)

    with gr.Row(visible=False) as code_row:
        code_display = gr.Markdown(code, elem_classes=["md"], height=300)

    output_visible = gr.State(True)
    code_btn.click(
        lambda x: (not x, gr.Row(visible=not x), gr.Row(visible=x), "View Results" if x else "View Code"),
        output_visible,
        [output_visible, output_row, code_row, code_btn],
        api_name=False,
    )

    false = gr.State(False)

    gr.on(
        [prompt.submit, generate_btn.click],
        None,
        None, 
        None,
        api_name=False,
        js="""
            function disappear() {
            var element = document.getElementById("hf_token_box");
            var height = element.offsetHeight;
            var step = height / 30; // Adjust this value to change the speed of disappearance
            var padding_top = parseFloat(getComputedStyle(element).paddingTop); // Get initial padding
            var padding_bottom = parseFloat(getComputedStyle(element).paddingBottom); // Get initial padding
            var step_padding = padding_top / 30; // Adjust this value to change the speed of disappearance
            var interval = setInterval(function() {
                if (height > 0) {
                    height -= step;
                    element.style.height = height + "px";
                    padding_bottom -= step_padding;
                    element.style.paddingBottom = padding_bottom + "px";
                    console.log("height", height);
                } else {
                    clearInterval(interval);
                }
            }, 20); // Adjust this value to change the smoothness of the animation
            }
        """
    )    

    gr.on(
        [prompt.submit, generate_btn.click],
        allow_ip,
        false,
    ).success(
        # model and model_name will be provided from the dropdown when the event fires
        partial(inference, model_name=None),
        [prompt, hf_token_box, llama_model_sel],
        llama_output,
        show_progress="hidden",
        api_name=False
    )

    gr.on(
        [prompt.submit, generate_btn.click],
        allow_ip,
        false,
    ).success(
        partial(inference, model_name=None),
        [prompt, hf_token_box, nous_model_sel],
        nous_output,
        show_progress="hidden",
        api_name=False
    )

    gr.on(
        [prompt.submit, generate_btn.click],
        allow_ip,
    ).success(
        partial(inference, model_name=None),
        [prompt, hf_token_box, zephyr_model_sel],
        zephyr_output,
        show_progress="hidden",
        api_name=False
    )

    gr.on(
        triggers=[prompt.submit, generate_btn.click],
        fn=lambda x: (code.replace("{PROMPT}", x), True, gr.Row(visible=True), gr.Row(visible=False), "View Code"),
        inputs=[prompt],
        outputs=[code_display, output_visible, output_row, code_row, code_btn],
        api_name=False
    )


demo.launch(show_api=False , debug=True)



