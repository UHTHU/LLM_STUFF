import json
import os
import subprocess
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset

def validate_path(path):
    """Strip quotes and validate file or directory path."""
    path = path.strip().strip('"').strip("'")
    path = os.path.normpath(path)
    return path

def convert_gguf_to_hf(gguf_path, hf_output_dir, llama_cpp_convert_script):
    """Convert GGUF model to Hugging Face format using llama.cpp's convert-hf-to-gguf.py in reverse."""
    if not os.path.exists(llama_cpp_convert_script):
        raise FileNotFoundError(f"convert-hf-to-gguf.py script not found at: {llama_cpp_convert_script}")
    
    os.makedirs(hf_output_dir, exist_ok=True)
    
    command = [
        "python",
        llama_cpp_convert_script,
        "--outfile", hf_output_dir,
        gguf_path
    ]
    try:
        subprocess.run(command, check=True)
        print(f"Successfully converted GGUF to Hugging Face format at {hf_output_dir}")
        return hf_output_dir
    except subprocess.CalledProcessError as e:
        raise Exception(f"Error converting GGUF to Hugging Face format: {e}")

def load_json_dataset(json_path):
    """Load and preprocess JSONL dataset."""
    texts = []
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    if 'text' in entry:
                        texts.append(entry['text'])
                    else:
                        print(f"Warning: Skipping line without 'text' field: {line.strip()}")
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON line skipped: {e}")
        if not texts:
            raise ValueError("No valid text entries found in the dataset.")
        return Dataset.from_dict({'text': texts})
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file not found: {json_path}")
    except Exception as e:
        raise Exception(f"Error loading dataset: {e}")

def tokenize_dataset(dataset, tokenizer, max_length=512):
    """Tokenize the dataset for training."""
    # Ensure tokenizer has a padding token
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"Set pad_token to eos_token: {tokenizer.pad_token}")
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            print("Added new pad_token: [PAD]")
    
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=max_length)
    return dataset.map(tokenize_function, batched=True)

def train_model(base_model_path, dataset, output_dir, epochs=3):
    """Fine-tune the base model on the dataset."""
    try:
        model = AutoModelForCausalLM.from_pretrained(base_model_path)
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    except Exception as e:
        raise Exception(f"Error loading base model or tokenizer: {e}")
    
    # Update model if new tokens were added to tokenizer
    if tokenizer.pad_token == '[PAD]':
        model.resize_token_embeddings(len(tokenizer))
        print("Resized model embeddings to match tokenizer with new pad_token")
    
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

def convert_to_gguf(hf_model_dir, gguf_output_path, llama_cpp_convert_script):
    """Convert Hugging Face model to GGUF format using llama.cpp."""
    if not os.path.exists(llama_cpp_convert_script):
        raise FileNotFoundError(f"convert.py script not found at: {llama_cpp_convert_script}")
    
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
        raise Exception(f"Error during GGUF conversion: {e}")

def main():
    try:
        # Prompt user for paths
        base_model_input = input("Enter the base model name (e.g., gpt2) or path to GGUF/HF model: ").strip()
        json_dataset_path = input("Enter the path to the JSON dataset (e.g., dataset.jsonl): ").strip()
        output_dir = input("Enter the output directory for the fine-tuned model: ").strip()
        gguf_output_path = input("Enter the output path for the GGUF model (e.g., model.gguf): ").strip()
        llama_cpp_convert_script = input("Enter the path to llama.cpp's convert-hf-to-gguf.py script: ").strip()
        
        # Validate and normalize paths
        base_model_input = validate_path(base_model_input)
        json_dataset_path = validate_path(json_dataset_path)
        output_dir = validate_path(output_dir)
        gguf_output_path = validate_path(gguf_output_path)
        llama_cpp_convert_script = validate_path(llama_cpp_convert_script)
        
        # Check if base model is a GGUF file
        if base_model_input.endswith('.gguf'):
            print(f"Detected GGUF model at {base_model_input}. Converting to Hugging Face format...")
            hf_model_dir = os.path.join(os.path.dirname(base_model_input), "hf_temp_model")
            base_model_path = convert_gguf_to_hf(base_model_input, hf_model_dir, llama_cpp_convert_script)
        else:
            base_model_path = base_model_input
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Verify dataset file exists
        if not os.path.exists(json_dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {json_dataset_path}")
        
        # Load dataset
        dataset = load_json_dataset(json_dataset_path)
        
        # Train model
        model, tokenizer = train_model(base_model_path, dataset, output_dir)
        
        # Convert to GGUF
        convert_to_gguf(output_dir, gguf_output_path, llama_cpp_convert_script)
        
        print(f"Fine-tuned model saved to {output_dir}")
        print(f"GGUF model saved to {gguf_output_path}")
    
    except Exception as e:
        print(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()