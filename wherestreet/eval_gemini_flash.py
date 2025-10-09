import argparse
from typing import List, Union, Tuple
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

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from PROMPT import SYSTEM_PROMPT, JSON_SCHEMA, SYSTEM_PROMPT_NO_JSON

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for model inference.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Run evaluation with Qwen2.5-VL model for geolocation."
    )
    
    # Model configuration
    parser.add_argument(
        "--model_path",
        default="gemini_2_5_flash",
        type=str,
        help="Path to the Skywork model"
    )
    # CSV input parameters
    parser.add_argument("--csv_path", type=str, default="./WhereBench/wherestreet/all.csv", help="Path to input csv file")
    parser.add_argument(
        "--inf_path",
        type=str,
        default="./results/json/gemini/gemini_25_flash_prediction_results.json",
        help="Path to the INPUT inference file"
    )
    parser.add_argument(
        "--eval_path",
        type=str,
        default="./results/json/gemini/gemini_25_flash_eval_results.json",
        help="Path to the output evaluation file"
    )
    parser.add_argument(
        "--overall_path",
        type=str,
        default="./results/overall/gemini_25_flash.csv",
        help="Path to the output accuracy file"
    )
    # Evaluation parameters
    parser.add_argument(
        "--eval_model",
        type=str,
        default="gemini-2.5-pro",
        help="Gemini model to use for evaluation"
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="Gemini API key for evaluation"
    )
    
    return parser.parse_args()


def initialize_gemini(api_key: str, model_name: str):
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
    
    return genai.GenerativeModel(model_name, safety_settings=safety_settings)




def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the Haversine distance between two latitude/longitude points.
    
    Args:
        lat1, lon1: First point coordinates
        lat2, lon2: Second point coordinates
        
    Returns:
        Distance in kilometers
    """
    R = 6371.0  # Radius of the Earth in kilometers
    lat1_rad = radians(lat1)
    lon1_rad = radians(lon1)
    lat2_rad = radians(lat2)
    lon2_rad = radians(lon2)

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance


def answer_score(predicted: str, ground_truth: str, answer_type: str, hint: str, gemini_model) -> Tuple[float, float]:
    """Calculate score based on prediction accuracy.
    
    Args:
        predicted: Predicted location
        ground_truth: Ground truth location
        answer_type: Type of answer required
        hint: Hint text
        gemini_model: Gemini model for evaluation
        
    Returns:
        Tuple of (score, distance)
    """
    distance = -1
    
    if answer_type == 'latitude & longitude':
        # Parse coordinates from predicted string
        predicted_clean = predicted.strip().replace('(', '').replace(')', '')
        coord_matches = re.findall(r'[-+]?\d*\.?\d+', predicted_clean)
        
        # If no coordinates found, ask Gemini to convert to lat/lon format
        if len(coord_matches) < 2:
            conversion_prompt =  f"""Convert the following location to latitude and longitude coordinates.

            Location: "{predicted}"
            Hint: "{hint}"

            Please provide the coordinates in decimal degrees format as: <answer>latitude, longitude</answer>

            Example format: <answer>40.7128, -74.0060</answer>

            If the location is too vague or cannot be determined, provide the best estimate for the center of the most likely area.
            If the prediction is just a copy of the hint, return <answer>0.0</answer>.
            If prediction is not a valid location, or unable to get the coordinates, return <answer>0.0</answer>.

            Format your response as:
            <answer>latitude, longitude</answer>"""

            response = gemini_model.generate_content([conversion_prompt])
            answer_text = response.text.strip()
            
            # Extract coordinates from Gemini response
            coord_match = re.search(r'<answer>\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*</answer>', answer_text)
            if coord_match:
                latitude = float(coord_match.group(1))
                longitude = float(coord_match.group(2))
            else:
                return 0.0, distance
        else:
            latitude = float(coord_matches[0])
            longitude = float(coord_matches[1])
        
        # Parse ground truth coordinates
        ground_truth_clean = ground_truth.strip().replace('(', '').replace(')', '')
        gt_coord_matches = re.findall(r'[-+]?\d*\.?\d+', ground_truth_clean)
        if len(gt_coord_matches) >= 2:
            ground_truth_latitude = float(gt_coord_matches[0])
            ground_truth_longitude = float(gt_coord_matches[1])
            
            distance = calculate_distance(latitude, longitude, ground_truth_latitude, ground_truth_longitude)
            alpha = 1.3
            R = 0.1
            return math.exp(- (distance / R) ** alpha), distance
        else:
            return 0.0, distance
    else:
        # For text-based location answers, use Gemini to evaluate accuracy
        evaluation_prompt = f"""You are a strict geolocation evaluator. Compare a predicted location to a ground-truth location and return ONE accuracy score as a FLOAT in [0.0, 1.0].

            INPUTS
            - Predicted Location: "{predicted}"
            - Ground Truth Location: "{ground_truth}"
            - Granularity to Judge (answer_type): "{answer_type}"   # one of: country | province/state | county/district | city | town/subdistrict | street
            - Hint (reference only; do not copy): "{hint}"

            RULES

            1) Normalize & Parse
            - Case/diacritic-insensitive; ignore punctuation/extra whitespace; accept common aliases (e.g., “NYC” = “New York City”, “München” = “Munich”).
            - Use this ordered hierarchy (down→top): street > town/subdistrict > city > county/district> province/state > country.
            - Map obviously equivalent administrative terms across countries (e.g., borough/parish/district). Do not invent missing components.

            2) Define the SCORING PATH (denominator)
            - Let L_target be the level named by {answer_type}.
            - Determine a base level L_base:
                • If the Hint names a level L_hint that is **consistent with** the Ground Truth, then set L_base = one level **below** L_hint (i.e., make the Hint “free information” and exclude it from credit).  
                • Otherwise (no usable Hint), set L_base = country.
            - The scoring path is the **contiguous list of levels** from L_base **including** L_base up to and including L_target.  
                Denominator = number_of_levels_in_this_path (k ≥ 1).

            3) Compute MATCHED PREFIX COUNT (numerator)
            - Walk the path from L_base downward. Count how many consecutive levels match the Ground Truth before the first mismatch.  
            - A level “matches” if either:
                • The Predicted explicitly names the same unit as the Ground Truth at that level, OR  
                • The Predicted **omits** that level but correctly names any **finer (lower)** level under the same Ground Truth parent (implicit parent credit), with no contradicting tokens.
            - If the first level on the path (L_base) is wrong, the matched count is 0.

            4) Score = matched_count / denominator
            - Return the ratio as a float in [0.0, 1.0].  
            - Examples when {answer_type}=street and Hint says a province (e.g., “Guangdong”):
                • Correct city→county→town→street: 4/4 = **1.0**  
                • Correct city→county→town, wrong/missing street: 3/4 = **0.75**  
                • Correct city→county, wrong/missing town: 2/4 = **0.50**  
                • Correct city only, wrong/missing county: 1/4 = **0.25**  
                • Wrong city: 0/4 = **0.00**

            5) Anti-Cheating
            - If the Predicted string copies the Hint (or is trivially derived from it) **without adding any level at or below {answer_type}**, set score to **0.00**.

            Exception:
            If Hint is providing extra instructions(e.g "The image is taken in one of the following countries: 1. UK 2. Canada 3. USA 4. Mexico.""), do not penalize the score by repeating the location mentioned in the hint.

            OUTPUT (strict)
            Return only the float (≤3 decimals) inside this tag:  <answer>SCORE</answer>
            No other text.

            Examples:
            1.Ground truth: Beicheng Street, Zaoyang county, Xiangyang city, Hubei, China. Pred: Niushou Town, Xiangyang city, Hubei , China. answer-type: street. hint: The image is taken in China.
            Path: street→town/sub→county→city→Province = 5
            Match: county mismatch, but city match = 2
            The score is 2/5=0.4

            2. Ground truth: Fuxing Street, Yuexiu District, Guangzhou , Guangdong, China. Pred: Fuxing Street, Guangzhou, China. answer-type: street. hint: The image is taken in Guangdong.
            Path: street→town/sub→city = 3
            Match: street match (essential), city match = 3
            The score is 3/3=1.0

            3. Ground truth: USA. Pred: USA. answer-type: country. hint: The image is taken in one of the following countries: 1. UK 2. Canada 3. USA 4. Mexico.
            Path: country = 1
            Match: country match = 1
            The score is 1/1 = 1.0"""
       
        response = gemini_model.generate_content([evaluation_prompt])
        answer_text = response.text.strip()
        
        # Extract score from answer tags
        score_match = re.search(r'<answer>\s*([0-9]*\.?[0-9]+)\s*</answer>', answer_text)
        if score_match:
            gpt_score = float(score_match.group(1))
            return min(max(gpt_score, 0.0), 1.0), distance
        else:
            return 0.0, distance


def thinking_score(thinking_process: str, key_clues: str, gemini_model) -> float:
    """Use Gemini to evaluate the thinking process against key clues.
    
    Args:
        thinking_process: The reasoning text
        key_clues: Key clues to look for
        gemini_model: Gemini model for evaluation
        
    Returns:
        Thinking process score
    """
    score = 0.0
    total_score = 0
    
    # Parse key_clues if it's a string representation of a list
    if isinstance(key_clues, str):
        try:
            key_clues = ast.literal_eval(key_clues)
        except (ValueError, SyntaxError):
            key_clues = [key_clues]
    result = []
    for key_clue in key_clues:
        total_score += 1
        if not key_clue.strip():
            continue
        evaluation_prompt = f"""
        You are an expert evaluator of logical reasoning and evidence utilization.

        TASK
        Decide whether the Key Clue was actually USED within the Reasoning Process to advance or support the location inference.

        INPUT
        Key Clue: "{key_clue}"
        Reasoning Process: "{thinking_process}"

        DEFINITIONS
        - Mentioned: the clue (or a clear synonym) is referenced in the reasoning.
        - Used: the reasoning relies on the clue to narrow candidates, eliminate options, strengthen a hypothesis, or justify the final conclusion.
        - Dismissed: the clue is mentioned but explicitly rejected or not carried forward.
        - Misused: the clue is cited but interpreted incorrectly.

        ALLOWED EVIDENCE
        Judge ONLY from the provided Reasoning Process. Do NOT add facts from outside knowledge or the image itself. Do NOT judge whether the final answer is correct—only whether the clue was used.

        DECISION RULES
        Answer "Yes" ONLY if ALL are true:
        1) The clue (or a clear synonym/phrase) is mentioned or unmistakably referred to, AND
        2) The reasoning uses it to narrow, rule out, weigh options, or support the conclusion (an explicit causal link or justification).
        Otherwise answer "No", including these cases:
        - Mentioned as a guess, observation, or side note without narrowing/supporting.
        - Mentioned then dismissed or ignored.
        - Not mentioned at all (directly or via clear synonym).
        - Misunderstood or misused as evidence.
        - Ambiguous/uncertain whether it aided reasoning.

        OUTPUT INSTRUCTIONS
        Return:
        <answer>Yes/No</answer>
        <explanation>One brief sentence justifying the decision, quoting a short span from the Reasoning Process if possible.</explanation>

        CONSTRAINTS
        - Base your decision strictly on the Reasoning Process text above.
        - If in doubt, answer "No".
        - Keep the explanation to 1–2 sentences.

        NOW DECIDE.
        """
       
        response = gemini_model.generate_content([evaluation_prompt])
        answer_text = response.text.strip().lower()
        match = re.search(r'<answer>\s*(.*?)\s*</answer>', answer_text)
        if match and match.group(1).strip().lower() == "yes":
            score += 1
            result.append(1)
        else:
            result.append(0)
    
    return score / total_score if total_score > 0 else 0.0, result


def evaluate_prediction(
    predicted: str, 
    ground_truth: str, 
    answer_type: str, 
    key_clues: str, 
    thinking_process: str,
    hint: str, 
    gemini_model
) -> dict:
    """Evaluate both answer accuracy and thinking process quality.
    
    Args:
        predicted: Predicted location
        ground_truth: Ground truth location
        answer_type: Type of answer required
        key_clues: Key clues to evaluate
        thinking_process: Reasoning text
        hint: Hint text
        gemini_model: Gemini model for evaluation
        
    Returns:
        Dictionary with evaluation scores
    """
    ans_score, distance = answer_score(predicted, ground_truth, answer_type, hint, gemini_model)
    think_score, result = thinking_score(thinking_process, key_clues, gemini_model)
    
    return {
        'answer_score': ans_score,
        'thinking_score': think_score,
        'distance': distance,
        'thinking_result': result
    }




def calculate_distance_acc(list_of_distance: List[float], accuracy: float) -> float:
    """Calculate the accuracy of the distance.
    
    Args:
        list_of_distance: List of distances
        accuracy: Accuracy threshold in km
        
    Returns:
        Accuracy percentage
    """
    return sum(1 for d in list_of_distance if d <= accuracy and d > 0) / len(list_of_distance)


def load_data_from_json(json_path: str) -> pd.DataFrame:
    """Load prediction data from JSON file and convert to DataFrame.
    
    Args:
        json_path: Path to the JSON file
        
    Returns:
        pandas.DataFrame: Converted data
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    gt_path = "../all.csv"
    gt_df = pd.read_csv(gt_path)
    gt_dict = {}
    for index, row in gt_df.iterrows():
        gt_dict[row['BVID']] = ast.literal_eval(row['key_clues'])    # Convert JSON to list of dictionaries for DataFrame
    rows = []
    for bvid, content in data.items():
        row = {
            'BVID': bvid,
            'ground_truth': content['ground_truth'],
            'answer_type': content['answer_type'],
            'key_clues': gt_dict[bvid],
            'predicted': content['predicted'],
            'response': content['response'],
            'hint': content['hint']
        }
        rows.append(row)
    
    return pd.DataFrame(rows)


def save_results_to_json(results: List[dict], output_path: str) -> None:
    """Save evaluation results to JSON file.
    
    Args:
        results: List of evaluation result dictionaries
        output_path: Path to save the JSON file
    """
    # Convert list of results to JSON format with BVID as key
    json_data = {}
    for result in results:
        bvid = result['BVID']
        json_data[bvid] = {
            'ground_truth': result['ground_truth'],
            'answer_type': result['answer_type'],
            'key_clues': result['key_clues'],
            'predicted': result['predicted'],
            'response': result['response'],
            'answer_score': result['answer_score'],
            'thinking_score': result['thinking_score'],
            'distance': result['distance'],
            'thinking_result': result['thinking_result']
        }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)




def run_evaluation(args: argparse.Namespace) -> None:
    """Run evaluation on prediction results.
    
    Args:
        args: Parsed command line arguments
    """

    MODEL_NAME = args.eval_model
    try:
        assert args.api_key is not None, "API key must be provided"
        gemini_model = initialize_gemini(args.api_key, args.eval_model)
    except Exception as e:
        print(f"Error initializing Gemini model: {e}")
        exit()

    print(f"Using evaluation model: {MODEL_NAME}")
    # Determine input and output paths
    model_name_clean = args.model_path.split("/")[-1]
    model_name_clean = model_name_clean.replace("-", "_").replace(".", "_")

    # Load prediction results for evaluation gemini_2_5_flash_prediction_results
    if args.inf_path:
        input_path = args.inf_path
    else:
        input_path = f"./results/json/{model_name_clean}_prediction_results.json"
    print(f"Loading data from JSON: {input_path}")
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        exit()
    df = load_data_from_json(input_path)
    
    
    if args.eval_path:
        eval_path = args.eval_path
    else:
        eval_path = f"./results/json/gemini/{model_name_clean}_eval_results.json"
        
    
    print(f"Will save results to: {eval_path}")
    
    # Load existing eval results if they exist
    existing_eval_results = []
    processed_bvids = set()
    if os.path.exists(eval_path):
        
        with open(eval_path, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
        existing_eval_results = []
        for bvid, content in existing_data.items():
            result = {
                'BVID': bvid,
                'ground_truth': content['ground_truth'],
                'answer_type': content['answer_type'],
                'key_clues': content['key_clues'],
                'predicted': content['predicted'],
                'response': content['response'],
                'answer_score': content['answer_score'],
                'thinking_score': content['thinking_score'],
                'distance': content['distance'], 
                'thinking_result': content['thinking_result']
            }
            existing_eval_results.append(result)
            processed_bvids.add(bvid)
        
        print(f"Found existing eval results with {len(processed_bvids)} processed BVIDs")
    
    eval_results = existing_eval_results.copy()
    pred_answer_score = 0.0
    pred_thinking_score = 0.0
    pred_total_score = 0.0
    total_predictions = 0
    
    # Calculate scores from existing results
    for result in existing_eval_results:
        pred_answer_score += float(result['answer_score'])
        pred_thinking_score += float(result['thinking_score'])
        total_predictions += 1
    
    print(f"Starting evaluation. {len(df)} total items, {len(processed_bvids)} already processed")
    distance_list = []
    bili_count = 0
    yt_count = 0
    bili_answer_score = 0.0
    yt_answer_score = 0.0
    for _, row in tqdm(df.iterrows()):

        if row['BVID'] in processed_bvids:
            if row['BVID'].startswith('BV'):
                bili_count += 1
                bili_answer_score += float(row['answer_score'])
            else:
                yt_count += 1
                yt_answer_score += float(row['answer_score'])
            continue

        evaluation_result = evaluate_prediction(
            row['predicted'], 
            row['ground_truth'], 
            row['answer_type'], 
            row['key_clues'], 
            row['response'],
            row['hint'],    
            gemini_model
        )
        total_predictions += 1
        pred_answer_score += float(evaluation_result['answer_score'])
        pred_thinking_score += float(evaluation_result['thinking_score'])
        if evaluation_result['distance'] != -1: 
            distance_list.append(evaluation_result['distance'])
        # Add to results
        new_result = {
            'BVID': row['BVID'],
            'ground_truth': row['ground_truth'],
            'answer_type': row['answer_type'],
            'key_clues': row['key_clues'],
            'predicted': row['predicted'],
            'response': row['response'],
            'answer_score': evaluation_result['answer_score'],
            'thinking_score': evaluation_result['thinking_score'],
            'distance': evaluation_result['distance'],
            'thinking_result': evaluation_result['thinking_result']
        }
        if str(row['BVID']).startswith('BV'):
            bili_count += 1
            bili_answer_score += float(evaluation_result['answer_score'])
        else:
            yt_count += 1
            yt_answer_score += float(evaluation_result['answer_score'])
        eval_results.append(new_result)
        processed_bvids.add(row['BVID'])
        print(f"BVID: {row['BVID']}, predicted: {row['predicted']}, ground_truth: {row['ground_truth']},  hint: {row['hint']}")
        print(f"Answer_score: {evaluation_result['answer_score']}, thinking_score: {evaluation_result['thinking_score']}, distance: {evaluation_result['distance']}")
        
        # Save results incrementally after each evaluation
        
        save_results_to_json(eval_results, eval_path)
        
    

    if total_predictions > 0:

        print(f"\nFinal Results:")
        print(f"Total processed: {total_predictions}")
        print(f"Answer score: {pred_answer_score/total_predictions}")
        print(f"Thinking score: {pred_thinking_score/total_predictions}")
        if len(distance_list) > 0:
            print("Total samples with coordinates: ",len(distance_list))
            print(f"For distance, Acc@1km:{calculate_distance_acc(distance_list,1)}")
            print(f"For distance, Acc@5km:{calculate_distance_acc(distance_list,5)}")
            print(f"For distance, Acc@10km:{calculate_distance_acc(distance_list,10)}")
            print(f"For distance, Acc@20km:{calculate_distance_acc(distance_list,20)}")
            print(f"For distance, Acc@50km:{calculate_distance_acc(distance_list,50)}")
            print(f"For distance, Acc@100km:{calculate_distance_acc(distance_list,100)}")
            print(f"For distance, Acc@200km:{calculate_distance_acc(distance_list,200)}")

        
    else:
        print("No successful predictions made.")

    # Save final results
    
    save_results_to_json(eval_results, eval_path)
    print(f"Results saved to {eval_path}")
    
    #also save and calculate the final average accuracy
    #read in from the accuracy  
    # Read existing results if file exists, otherwise create new DataFrame
    if os.path.exists(args.overall_path):
        #wherecountry should have a score there already
        overall_df = pd.read_csv(args.overall_path)
        wherecountry_accuracy = overall_df['wherecountry'].mean() if 'wherecountry' in overall_df.columns else 0.0
    else:
        overall_df = pd.DataFrame(columns=['model_name', 'wherestreet', 'wherestreet-bili', 'wherestreet-YT', 'overall'])
    
    # Calculate accuracy
    
    bili_accuracy = bili_answer_score / bili_count if bili_count > 0 else 0.0
    yt_accuracy = yt_answer_score / yt_count if yt_count > 0 else 0.0
    wherecountry_count = 500 if wherecountry_accuracy != 0.0 else 0
    overall = (wherecountry_count*wherecountry_accuracy+ bili_count*bili_accuracy + yt_count*yt_accuracy)/(wherecountry_count+bili_count+yt_count)
    # Update or add row for this model
    model_basename = os.path.basename(args.model_path)
    if model_basename in overall_df['model_name'].values:
        overall_df.loc[overall_df['model_name'] == model_basename, 'wherecountry'] = wherecountry_accuracy
        overall_df.loc[overall_df['model_name'] == model_basename, 'wherestreet-bili'] = bili_accuracy
        overall_df.loc[overall_df['model_name'] == model_basename, 'wherestreet-YT'] = yt_accuracy
        overall_df.loc[overall_df['model_name'] == model_basename, 'overall'] = overall
    else:
        new_row = pd.DataFrame({
            'model_name': [model_basename],
            'wherecountry': [wherecountry_accuracy],
            'wherestreet-bili': [bili_accuracy],
            'wherestreet-YT': [yt_accuracy],
            'overall': [overall]
        })
        overall_df = pd.concat([overall_df, new_row], ignore_index=True)
    
    
    # Save updated results
    overall_df.to_csv(args.overall_path, index=False)
    print(f"Overall results saved to {args.overall_path}")



def main() -> None:
    """Main execution function."""
    args = parse_arguments()
    
    # Create results directory if it doesn't exist
    os.makedirs("./results", exist_ok=True)
    
    run_evaluation(args)

if __name__ == "__main__":
    main()
