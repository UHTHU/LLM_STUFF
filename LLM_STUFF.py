import json
import os
import subprocess
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset

def load_json_dataset(json_path):
    """Load and preprocess JSON dataset."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    texts = [entry['text'] for entry in data]
    return Dataset.from_dict({'text': texts})

def tokenize_dataset(dataset, tokenizer, max_length=512):
    """Tokenize the dataset for training."""
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=max_length)
    return dataset.map(tokenize_function, batched=True)

def train_model(base_model_name, dataset, output_dir, epochs=3):
    """Fine-tune the base model on the dataset."""
    model = AutoModelForCausalLM.from_pretrained(base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=4,
        save_strategy="epoch",
        logging_dir=f"{output_dir}/logs",
        logging_steps=100,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    
    trainer.train()
    
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return model, tokenizer

def convert_to_gguf(hf_model_dir, gguf_output_path):
    """Convert Hugging Face model to GGUF format using llama.cpp."""
    # Assuming llama.cpp is installed and convert.py is available
    llama_cpp_convert_script = "path/to/llama.cpp/convert.py"  # Update with actual path
    command = [
        "python",
        llama_cpp_convert_script,
        hf_model_dir,
        "--outfile",
        gguf_output_path
    ]
    try:
        subprocess.run(command, check=True)
        print(f"Successfully converted model to GGUF at {gguf_output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error during GGUF conversion: {e}")

def main():
    base_model_name = input("Enter the base model name or path (e.g., gpt2): ").strip()
    json_dataset_path = input("Enter the path to the JSON dataset (e.g., dataset.json): ").strip()
    output_dir = input("Enter the output directory for the fine-tuned model: ").strip()
    gguf_output_path = input("Enter the output path for the GGUF model (e.g., model.gguf): ").strip()
    
    os.makedirs(output_dir, exist_ok=True)
    
    dataset = load_json_dataset(json_dataset_path)
    
    model, tokenizer = train_model(base_model_name, dataset, output_dir)
    
    convert_to_gguf(output_dir, gguf_output_path)
    
    print(f"Fine-tuned model saved to {output_dir}")
    print(f"GGUF model saved to {gguf_output_path}")

if __name__ == "__main__":
    main()