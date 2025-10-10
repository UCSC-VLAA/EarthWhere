echo "Inference Gemini-2.5-flash on WhereCountry ..."

export GEMINI_API_KEY="YOUR_GEMINI_API_KEY_HERE"

python ./wherecountry/eval_gemini_25_flash.py --model gemini-2.5-flash --csv_path "./wherecountry/wherecountry_500.csv" --img_dir "./wherecountry/imgs" --output_csv "./results/overall/gemini_25_flash.csv" --api_key $GEMINI_API_KEY

echo "Inference Gemini-2.5-flash on WhereStreet ..."

# export GEMINI_API_KEY="YOUR_GEMINI_API_HERE"

python ./wherestreet/inf_gemini_flash.py --model gemini-2.5-flash --csv_path "./wherestreet/all.csv" --img_dir "./wherestreet/imgs" --output_csv "./results/json/gemini_2_5_flash_prediction_results.json" --api_key $GEMINI_API_KEY

python ./wherestreet/eval_gemini_flash.py --model_path "gemini-2.5-flash" --overall_path "./results/overall/gemini_25_flash.csv" --inf_path "./results/json/gemini_2_5_flash_prediction_results.json" --eval_path "./results/eval/gemini_2_5_flash_eval_results.json" --eval_model "gemini-2.5-pro" --api_key $GEMINI_API_KEY