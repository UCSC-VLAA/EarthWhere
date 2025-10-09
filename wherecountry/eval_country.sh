echo "Inference Gemini-2.5-flash on WhereCountry ..."

export GEMINI_API_KEY="YOUR_GEMINI_API_KEY_HERE"

python ./wherecountry/eval_gemini_25_flash.py --model gemini-2.5-flash --csv_path "./wherecountry/wherecountry_500.csv" --img_dir "./wherecountry/imgs" --output_csv "./results/overall/gemini_25_flash.csv" --api_key $GEMINI_API_KEY
