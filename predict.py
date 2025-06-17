from cog import BasePredictor, Input
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

class Predictor(BasePredictor):
    def setup(self):
        # Model cache location
        model_path = "model"
        model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

        # Download from Hugging Face if not already present
        if not os.path.exists(model_path) or not os.path.exists(os.path.join(model_path, "config.json")):
            print("⏬ Downloading model weights from Hugging Face...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                cache_dir=model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            # Save model to model_path for reuse
            self.tokenizer.save_pretrained(model_path)
            self.model.save_pretrained(model_path)
        else:
            print("✅ Using cached model in 'model/'")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )

        # Initialize history
        self.history = [
            {"role": "system", "content": "You are a friendly chatbot."}
        ]

    def count_tokens(self, messages):
        return sum(len(self.tokenizer.encode(msg["content"])) for msg in messages)

    def predict(self, prompt: str = Input(description="Your message")) -> str:
        self.history.append({"role": "user", "content": prompt})

        if self.count_tokens(self.history) > 1000:
            self.history = [self.history[0]] + self.history[-4:]

        full_prompt = self.tokenizer.apply_chat_template(
            self.history, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                repetition_penalty=1.2,
                eos_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        ).strip()

        self.history.append({"role": "assistant", "content": response})
        return response
