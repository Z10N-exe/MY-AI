from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from flask_cors import CORS
import logging
from datetime import datetime
import pytz

app = Flask(__name__)
CORS(app, resources={r"/chat": {"origins": "http://localhost:8000"}})

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Define Hugging Face token directly
huggingface_token = "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"  # Replace with your actual Hugging Face token

# Load model and tokenizer
model_name = "google/gemma-2-2b-it"
try:
    logger.info(f"Loading tokenizer for {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=huggingface_token  # Use the hardcoded token
    )
    logger.info(f"Loading model {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=huggingface_token,  # Use the hardcoded token
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )
except Exception as e:
    logger.error(f"Error loading model or tokenizer: {e}")
    exit(1)

# Set pad_token_id
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
    logger.info("Set pad_token_id to eos_token_id")

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
model.to(device)
model.eval()  # Set model to evaluation mode


@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_input = data.get('message', '').strip()
        logger.info(f"Received message: {user_input}")

        if not user_input:
            logger.warning("No message provided")
            return jsonify({'error': 'No message provided'}), 400

        if len(user_input) > 1000:  # Basic input length validation
            logger.warning("Input too long")
            return jsonify({'error': 'Input exceeds maximum length of 1000 characters'}), 400

        # Format prompt using the model's chat template
        messages = [{"role": "user", "content": user_input}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate response
        logger.info("Generating response")
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            attention_mask=inputs["attention_mask"]
        )

        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()
        logger.info(f"Generated response: {response}")

        return jsonify({'response': response})
    except ValueError as ve:
        logger.error(f"Value error during generation: {ve}")
        return jsonify({'error': 'Invalid input or configuration'}), 400
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/health', methods=['GET'])
def health():
    logger.info("Health check requested")
    wat_time = datetime.now(pytz.timezone('Africa/Lagos')).strftime('%I:%M %p WAT, %B %d, %Y')
    return jsonify({'status': 'Server running', 'model': model_name, 'time': wat_time})


if __name__ == '__main__':
    logger.info("Starting Flask server")
    app.run(host='0.0.0.0', port=8000, debug=False)