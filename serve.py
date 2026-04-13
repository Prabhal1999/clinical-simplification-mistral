import modal

app = modal.App("clinical-simplifier")

image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "transformers",
        "bitsandbytes",
        "accelerate",
        "peft",
        "fastapi[standard]",
    )
)

@app.cls(
    image=image,
    gpu="T4",
    scaledown_window=300,
)
class Model:
    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        model_id = "prabhal/mistral-clinical-simplifier"

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
        )

    @modal.fastapi_endpoint(method="POST")
    def generate(self, request: dict):
        import torch

        text = request.get("text", "")
        prompt = (
            "### Instruction:\nSimplify the following clinical text for a patient.\n\n"
            f"### Input:\n{text}\n\n### Response:\n"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=250,
                temperature=0.4,
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True,
            )

        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        simplified = result.split("### Response:")[-1].strip()

        return {"output": simplified}