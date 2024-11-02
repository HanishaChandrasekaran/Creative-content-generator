from flask import Flask, render_template, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)

# Load the model and tokenizer
model_name = "gpt2"  # Change this to your specific model name if needed
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    input_text = request.form['inputText']
    
    # Clear prompt
    prompt = f"Write a short story about {input_text}."

    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    output = model.generate(
        input_ids,
        max_length=150,  # Longer output for a story
        num_return_sequences=1,
        temperature=1.2,  # Increase for more creative responses
        top_k=50,
        top_p=0.95,
        do_sample=True
    )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    
    return jsonify({'outputText': generated_text})

if __name__ == '__main__':
    app.run(debug=True)
