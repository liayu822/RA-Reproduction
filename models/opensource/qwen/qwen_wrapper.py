# models/opensource/qwen/qwen_wrapper.py

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import List, Dict

class QwenModelWrapper:
    def __init__(self, model_path: str, config: Dict = None):
        """
        初始化Qwen模型和分詞器
        """
        print(f"Initializing Qwen model from: {model_path}")
        self.model_path = model_path
        self.config = config if isinstance(config, dict) else {}

        # 檢測可用的設備 (CUDA GPU 或 CPU)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Target device set to: {self.device}")

        # 載入分詞器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )

        # --- 🔴 關鍵修正：採用更直接的方式將模型加載到 GPU ---
        print("Loading model to CPU first...")
        # 1. 先在 CPU 上以 bfloat16 格式加載模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        
        # 2. 然後，將整個模型強制移動到目標設備 (GPU)
        print(f"Moving model to {self.device}...")
        self.model.to(self.device)
        # ----------------------------------------------------
        
        self.model.eval() # 設置為評估模式
        print(f"Model successfully loaded and moved to {self.device}.")

    def chat(self, query: str, history: List[Dict[str, str]] = None) -> str:
        """
        與模型進行對話的核心方法
        """
        if history is None:
            messages = [{"role": "user", "content": query}]
        else:
            messages = history + [{"role": "user", "content": query}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = self.tokenizer([text], return_tensors="pt")

        # --- 🔴 同樣關鍵：確保輸入張量也被移動到與模型相同的設備 ---
        model_inputs = model_inputs.to(self.device)
        
        model_params = self.config.get("model_params", {})
        max_new_tokens = model_params.get("max_tokens", 1024)
        temperature = model_params.get("temperature", 0.6)
        top_p = model_params.get("top_p", 0.9)

        generated_ids = self.model.generate(
            model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p
        )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response.strip()

# 用於單獨測試的示例
if __name__ == '__main__':
    print("QwenModelWrapper is defined correctly. This file can be imported by other scripts.")