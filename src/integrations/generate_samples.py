"""
generate_samples.py - Utility to generate sample ATE logs for integration testing
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.integrations.ate_parser import ATEDataParser

def create_sample_files():
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'ate_samples'))
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Teradyne J750 ASCII Log
    txt_log = ATEDataParser.generate_sample_ate_log(n_parts=60, lot_id="LOT-ISRO-2026-X1")
    txt_path = os.path.join(out_dir, "teradyne_j750_lot_x1.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(txt_log)
    print(f"Created: {txt_path}")
    
    # 2. Chroma CSV Log
    parser = ATEDataParser()
    parsed = parser.parse_ascii_ate_datalog(txt_log)
    df = parsed['dataframe']
    csv_path = os.path.join(out_dir, "chroma_58158_lot_x1.csv")
    df.to_csv(csv_path, index=False)
    print(f"Created: {csv_path}")

if __name__ == "__main__":
    create_sample_files()
