# models/closed-source/gpt4_wrapper.py
"""
一個用於與 OpenAI GPT 系列模型（如 GPT-4, GPT-4o）互動的封裝器。
此版本會從設定檔讀取參數，並包含錯誤重試機制。
"""

import os
import time
import logging
from openai import OpenAI
from typing import Dict, Any

class GPT4Wrapper:
    def __init__(self, config: Dict[str, Any]):
        """
        初始化 OpenAI API 客戶端。

        Args:
            config (Dict[str, Any]): 一個包含 'name', 'api_key', 'temperature', 'max_tokens' 等設定的字典。
        """
        self.model_name = config.get("name", "gpt-4o")
        self.api_key = config.get("api_key") or os.environ.get("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError("OpenAI API key is not provided. Please set it in config.yaml or as an environment variable OPENAI_API_KEY.")

        # 從 config 讀取模型參數，並設定與論文對齊的預設值
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 150) # 預設為影子模型的建議值
        
        self.client = OpenAI(api_key=self.api_key)
        logging.info(f"GPT4Wrapper initialized for model: {self.model_name} with temp={self.temperature}, max_tokens={self.max_tokens}")

    def chat(self, prompt: str, max_retries: int = 3, retry_delay: int = 5) -> str:
        """
        發送單個 prompt 給 GPT 模型並獲取回答。
        為了與您現有的 QwenWrapper 介面保持一致。

        Args:
            prompt (str): 要發送給模型的單個字符串提問。
            max_retries (int): 最大重試次數。
            retry_delay (int): 重試間隔秒數。

        Returns:
            str: 模型的文字回答。
        """
        for attempt in range(max_retries):
            try:
                # OpenAI 的對話 API 偏好接收一個 message 列表
                messages = [{"role": "user", "content": prompt}]
                
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=1.0,
                    frequency_penalty=0.0,
                    presence_penalty=0.0
                )
                response_text = completion.choices[0].message.content.strip()
                return response_text
                
            except Exception as e:
                logging.error(f"OpenAI API call failed on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    logging.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logging.error("Max retries reached. Returning an error message.")
                    return f"Error: API call failed after {max_retries} retries. Last error: {e}"
        
        return "" # 如果所有重試都失敗