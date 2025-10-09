import pandas as pd
import os
import re
import random
import json
from tqdm import tqdm
from PIL import Image
import ast
import math
from math import radians, sin, cos, sqrt, atan2
import google.generativeai as genai
import argparse
from PROMPT import SYSTEM_PROMPT, JSON_SCHEMA,SYSTEM_PROMPT_NO_JSON

def ensure_list_of_strings(value):
    """
    Ensure the input value is converted to a list of strings.
    Handles various input formats: string, list, or string representation of a list.
    """
    if value is None:
        return []
    
    if isinstance(value, list):
        # Already a list, ensure all elements are strings
        return [str(item) for item in value]
    
    if isinstance(value, str):
        # Try to parse as a list first
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
            else:
                # If it's not a list after parsing, treat as a single string
                return [value]
        except (ValueError, SyntaxError):
            # If parsing fails, treat as a single string
            return [value]
    
    # For any other type, convert to string and wrap in list
    return [str(value)]

def ensure_hint_format(hint):
    """Ensure hint is formatted as a list of strings"""
    return ensure_list_of_strings(hint)

def ensure_key_clues_format(key_clues):
    """Ensure key_clues is formatted as a list of strings"""
    return ensure_list_of_strings(key_clues)

def build_gemini_messages(hint: str, answer_type: str):
    """
    Builds messages compatible with Gemini API.
    answer_type ∈ {country|province|city|county|street|latlon}
    """
    user_text = (
        f"HINT: {hint}\n"
        f"ANSWER_TYPE: {answer_type}\n"
        + f"Try the best to find the location at the {answer_type} level. If requested 'latitude & longitude', try to find the exact coordinates and answer in the format as <answer>(latitude, longitude)</answer>. Response in English. Follow the following response format STRICTLY. Provide detailed reasoning process between the <think></think> tag. Give the final location answer in a fine-to-coarse manner in <answer></answer> tag."
    )
    
    return user_text

def extract_json(s: str) -> dict:
    """
    Extract first top-level JSON object from a string (tolerant to stray text),
    then parse with json.loads.
    """
    start = s.find("{")
    if start == -1:
        raise ValueError("No JSON object start found.")
    depth = 0
    for i, ch in enumerate(s[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                raw = s[start : i + 1]
                return json.loads(raw)
    raise ValueError("Unbalanced braces; could not extract JSON.")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gemini-2.5-flash", help="Gemini model to use for inference")
    parser.add_argument("--csv_path", type=str, default="./WhereBench/wherestreet/all.csv", help="Path to input csv file")
    parser.add_argument("--img_dir", type=str, default="./WhereBench/wherestreet/imgs", help="Directory containing images")
    parser.add_argument("--gemini_api_key", type=str, default=None, help="Gemini API key for inference")
    parser.add_argument("--output_csv", type=str, default="./results/json/gemini_2_5_flash_prediction_results.json", help="Path to save output results")
    parser.add_argument("--api_key", type=str, default=None, help="Gemini API key for inference")
    return parser.parse_args()

def initialize_gemini(api_key, model_name):
    """Initialize Gemini AI with the specified model"""
    genai.configure(api_key=api_key)
    
    # Configure safety settings to block none
    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE"
        }
    ]
    
    return genai.GenerativeModel(model_name, safety_settings=safety_settings, system_instruction=SYSTEM_PROMPT_NO_JSON)

def analyze_image_gemini(img_path, hint, answer_type, gemini_model):
    """Analyze image and get location prediction using Gemini
    
    return: predicted, response_text
    """

    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        return None, None
    
    
    # Load image
    image = Image.open(img_path)
    
    # Build prompt
    user_text = build_gemini_messages(hint, answer_type)
    
    # Generate response with Gemini
    response = gemini_model.generate_content([user_text, image])
    response_text = response.text.strip()
    

    answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', response_text, re.DOTALL)
    think_match = re.search(r'<think>\s*(.*?)\s*</think>', response_text, re.DOTALL)
    
    predicted = answer_match.group(1).strip() if answer_match else None
    thinking_process = think_match.group(1).strip() if think_match else None

    return predicted, thinking_process
   
    


def load_input_data(json_path):
    """Load input data from JSON file"""
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    # Convert JSON structure to DataFrame format for processing
    data_rows = []
    for bvid, info in json_data.items():
        row = {
            'BVID': bvid,
            'answer': info['ground_truth'],
            'answer_type': info['answer_type'],
            'key_clues': ensure_key_clues_format(info['key_clues']),
            'hint': ensure_hint_format(info.get('hint', []))
        }
        data_rows.append(row)
    
    return pd.DataFrame(data_rows), json_data

def load_existing_results(results_path):
    """Load existing results from JSON file"""
    if not os.path.exists(results_path):
        return {}, set()
    
    with open(results_path, 'r', encoding='utf-8') as f:
        existing_results = json.load(f)
    processed_bvids = set(existing_results.keys())
    return existing_results, processed_bvids

def save_results(results, results_path):
    """Save results to JSON file"""
    # Create results directory if it doesn't exist
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    args = parse_args()

    # Create results directory if it doesn't exist
    os.makedirs("./results/json", exist_ok=True)
    os.makedirs("./results/eval", exist_ok=True)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    # Initialize Gemini model
    gemini_model = initialize_gemini(args.api_key, args.model)
    print(f"Initialized Gemini model: {args.model}")
    
    # Load input data
    df = pd.read_csv(args.csv_path)
    print(f"Loaded {len(df)} rows")
    
    # Set up results file paths
    model_name_clean = args.model.replace("-", "_").replace(".", "_")
    
    if args.output_csv:
        results_path = args.output_csv
    else:
        results_path = f"./results/json/{model_name_clean}_prediction_results.json"
    
    # Load existing results
    existing_results, processed_bvids = load_existing_results(results_path)
    results = existing_results.copy()
    
    print(f"Found existing results with {len(processed_bvids)} processed BVIDs")
    
    failed_results = []
    count = 0
    
    for index, row in tqdm(df.iterrows()):
        bvid = row['BVID']
        
        # Skip if already processed
        if bvid in processed_bvids:
            continue
            
        answer_type = row['answer_type']
        hint = row['hint']
        key_clues = row['key_clues']
        img_path = os.path.join(args.img_dir,f"{bvid}.jpg")
    
        print(f"Processing {index}/{len(df)}: {str(bvid)}")
        predicted, evidence = analyze_image_gemini(img_path, hint, answer_type, gemini_model)
        print(f"Predicted: {predicted}, GT: {row['answer']}")

        if predicted:
            # Store in JSON format
            results[bvid] = {
                'ground_truth': row['answer'],
                'answer_type': answer_type,
                'key_clues': ensure_key_clues_format(row['key_clues']),
                'predicted': predicted,
                'response': evidence,
                'hint': ensure_hint_format(hint)
            }
            
            processed_bvids.add(bvid)
            
            # Save results incrementally after each prediction
            save_results(results, results_path)
            print(f"Saved result for BVID {bvid} to {results_path}")
        else:
            print(f"No prediction generated for BVID {bvid}")
        
    # Final save of results
    if results:
        save_results(results, results_path)
        print(f"Final results saved to {results_path}")
