import google.generativeai as genai
import base64
import pandas as pd
import os
import re
import random
from tqdm import tqdm
from PIL import Image
import ast

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gemini-2.5-flash', help='Model name to use')
    parser.add_argument('--csv_path', type=str, default='./WhereBench/wherecountry/wherecountry_500.csv', help='Path to input CSV file')
    parser.add_argument('--img_dir', type=str, default='./WhereBench/wherecountry/imgs/', help='Directory containing images')
    parser.add_argument('--api_key', type=str, default=None, help='API key for Gemini')
    parser.add_argument('--output_csv', type=str, default='./results/overall/gemini_25_flash.csv', help='Path to save output CSV file')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    model_name = args.model
    img_dir = args.img_dir
    output_path = args.output_csv
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Configure Gemini API
    try:
        assert args.api_key is not None, "API key must be provided"
        genai.configure(api_key=args.api_key)  # Replace with your actual API key

        # Initialize the model
        model = genai.GenerativeModel(model_name)  # or gemini-1.5-flash for faster responses
    except Exception as e:
        print(f"Error initializing Gemini API: {str(e)}")
        return

    # Read the CSV file
    csv_path = args.csv_path
    df = pd.read_csv(csv_path)
    
    def analyze_image(img_path, ground_truth_country, options):
        """Analyze image and get country prediction"""
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            return None
        
        try:
            # Load image using PIL
            image = Image.open(img_path)
            
            # Get neighboring countries for multiple choice
            neighboring_countries = ast.literal_eval(options)
            all_choices = [ground_truth_country] + neighboring_countries
            random.shuffle(all_choices)  # Randomize order
            
            choices_text = "\n".join([f"{i+1}. {country}" for i, country in enumerate(all_choices)])
            
            # Create the prompt
            prompt = f"""You are a geography expert. Analyze images and determine which country it is taken in based on visual clues like architecture, signage, landscape, vehicles, etc.

            Look at this image and determine which country it was taken in. Think first, then choose from the following options:

            {choices_text}

            Format strictly as: <think>...</think> <answer>...</answer>. Provide your answer as just the number (1, 2, 3, or 4) of your choice."""
            
            # Generate response
            response = model.generate_content([prompt, image])
            
            response_text = response.text.strip()
            
            # Parse the response to extract the choice number
            match = re.search(r'<answer>.*?(\b[1-4]\b).*?</answer>', response_text, re.DOTALL)
            if not match:
                # Fallback: look for any number 1-4 in the response
                match = re.search(r'\b([1-4])\b', response_text)
            
            if match:
                choice_num = int(match.group(1)) - 1  # Convert to 0-based index
                if 0 <= choice_num < len(all_choices):
                    predicted_country = all_choices[choice_num]
                    return predicted_country, all_choices, response_text
            
            return None, all_choices, response_text
            
        except Exception as e:
            print(f"Error processing image {img_path}: {str(e)}")
            return None, [], str(e)

    # Process each row and calculate accuracy
    correct_predictions = 0
    total_predictions = 0
    results = []
    count = 0

    for index, row in tqdm(df.iterrows()):
 
        pano_id = row['panoID']
        ground_truth_country = row['nation']
        options = row['options']
        
        # Assuming image path is constructed from panoId (adjust path as needed)
        img_path = os.path.join(img_dir,f"{pano_id}_panoramic.jpg")#f"/data1/zqian20/VisualSketchpad/tuxun/pano/qwen_failed/{pano_id}_panoramic.jpg"
        
        print(f"Processing {index+1}/{len(df)}: {pano_id}")
        
        predicted_country, choices, response_text = analyze_image(img_path, ground_truth_country, options)
        
        if predicted_country:
            is_correct = predicted_country == ground_truth_country
            if is_correct:
                correct_predictions += 1
            total_predictions += 1
            
            results.append({
                'panoId': pano_id,
                'ground_truth': ground_truth_country,
                'predicted': predicted_country,
                'choices': choices,
                'correct': is_correct,
                'response': response_text
            })
            
            print(f"Ground Truth: {ground_truth_country}")
            print(f"Predicted: {predicted_country}")
            print(f"Choices: {choices}")
            print("-" * 50)
        else:
            print(f"Failed to get prediction for {pano_id}")
            print("-" * 50)

    # Calculate and display accuracy
    if total_predictions > 0:
        accuracy = correct_predictions / total_predictions
        print(f"\nFinal Results:")
        print(f"Total processed: {total_predictions}")
        print(f"Correct predictions: {correct_predictions}")
        print(f"Accuracy: {accuracy:.2%}")
    else:
        accuracy = 0.0
        print("No successful predictions made.")

    # Save results to CSV
    results_df = pd.DataFrame(results)
    wherecountry_pred_path = "./results/gemini_25_flash_wherecountry_prediction_results.csv"
    os.makedirs(os.path.dirname(wherecountry_pred_path), exist_ok=True)
    results_df.to_csv(wherecountry_pred_path, index=False)
    print(f"Inference Results saved to {wherecountry_pred_path}")
    # Save final accuracy to CSV
    model_basename = os.path.basename(args.model)
    results_df = pd.DataFrame({
    'model_name': [model_basename],
    'wherecountry': [accuracy]
    })
    
    results_df.to_csv(output_path, index=False)
    print(f"\nAccuracy Results saved to {output_path}")


if __name__ == "__main__":
    main()