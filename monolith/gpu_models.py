# This is a collection of GPU models used in the monolith, deployed onto Modal.

import modal

app = modal.App("lsb-ambient-research-accelerated-models")

MODEL_CACHE_PATH = "/root/models"
inference_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "accelerate",
    "huggingface_hub[hf_transfer]",
    "numpy",
    "torch",
    "transformers",
    "optimum-quanto",
).env(
    {"HF_HUB_CACHE": MODEL_CACHE_PATH, "HF_HUB_ENABLE_HF_TRANSFER": "1"}
)
model_cache = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)

ASR_MODEL_NAME = "openai/whisper-large-v3-turbo"
MODEL_REVISION = "main"

LLM_MODEL_NAME = "Qwen/Qwen3-8B"
MODEL_REVISION = "main"

def setup():
    from transformers import pipeline, QuantoConfig
    import torch
    asrpipe = pipeline("automatic-speech-recognition", model=ASR_MODEL_NAME, device="cuda", torch_dtype=torch.float16) # native dtype of model
    llmpipe = pipeline("text-generation", model=LLM_MODEL_NAME, device_map="auto", quantization_config=QuantoConfig(weights="float8"))
    mu6palette = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_"
    mu6palette_to_int = torch.zeros((128,), dtype=torch.float32) / 0.0 # all nans, for error detection
    for i, c in enumerate(mu6palette):
        mu6palette_to_int[c] = i

    return asrpipe, llmpipe, mu6palette_to_int


inference_image = inference_image.env(
    {"HF_HUB_CACHE": MODEL_CACHE_PATH, "HF_HUB_ENABLE_HF_TRANSFER": "1"}
)

@app.cls(
    gpu="l4",
    retries=3,
    volumes={MODEL_CACHE_PATH: model_cache},
    image=inference_image,
    max_containers=5,
)
class Inference:
    @modal.enter()
    def load_models(self):
        self.asrpipe, self.llmpipe, self.mu6palette_to_int = setup()
    
    @modal.method()
    def transcribe(self, mu_string):
        import torch
        mu_string = torch.tensor(list(mu_string), dtype=torch.int32)
        # now that we have a string of bytes, index into the palette for each byte
        mu_bytes = self.mu6palette_to_int[mu_string]
        mu_floats = ((mu_bytes / 63) * 2.0) - 1.0
        mu_floats = torch.sign(mu_floats) * (torch.exp(torch.abs(mu_floats) * torch.log1p(torch.tensor(63))) - 1.0) / 63

        if torch.any(torch.isnan(mu_floats)):
            return None

        return self.asrpipe(mu_floats.numpy(), return_timestamps=True)
    
    @modal.method()
    def summarize(self, text):
        return self.llmpipe(text, max_new_tokens=40000)




