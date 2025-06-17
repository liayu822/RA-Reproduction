# src/utils/logger.py

import os
import json
import logging
from datetime import datetime

def setup_logger(name, log_file, level=logging.INFO):
    """一個簡單的函式，用於設置和獲取一個 logger。"""
    handler = logging.FileHandler(log_file, encoding='utf-8')        
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers: # 避免重複添加 handler
        logger.addHandler(handler)
        
    return logger

class AttackLogger:
    """
    用於保存每一次攻擊詳細結果的類別。
    """
    def __init__(self, base_dir="results", config_dict=None, prefix="attack"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = os.path.join(base_dir, f"{prefix}_{timestamp}")
        os.makedirs(self.save_dir, exist_ok=True)

        if config_dict:
            with open(os.path.join(self.save_dir, "config.json"), "w", encoding="utf-8") as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)

        self.sample_count = 0
        print(f"AttackLogger initialized. Results will be saved in: {self.save_dir}")

    def save_sample(self, result: dict):
        """
        儲存單一樣本的攻擊結果。
        """
        filename = os.path.join(self.save_dir, f"result_{self.sample_count:04d}.json")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        self.sample_count += 1

    def save_summary(self, summary_data: dict):
        """
        儲存最終的匯總報告。
        """
        summary_path = os.path.join(self.save_dir, "summary_report.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        print(f"✅ Summary report saved to: {summary_path}")

    def get_save_dir(self):
        return self.save_dir