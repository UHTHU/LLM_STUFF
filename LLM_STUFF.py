import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
from llama_cpp import convert_hf_to_gguf

def load_json_dataset(json_path):
    """Load and preprocess JSON dataset."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Assuming JSON has a list of entries with 'text' field
    texts = [entry['text'] for entry in data]
    return Dataset.from_dict({'text': texts})

def tokenize_dataset(dataset, tokenizer, max_length=512):
    """Tokenize the dataset for training."""
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=max_length)
    return dataset.map(tokenize_function, batched=True)

def train_model(base_model_name, dataset, output_dir, epochs=3):
    """Fine-tune the base model on the dataset."""
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    # Tokenize dataset
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=4,
        save_strategy="epoch",
        logging_dir=f"{output_dir}/logs",
        logging_steps=100,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    
    # Train the model
    trainer.train()
    
    # Save the fine-tuned model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return model, tokenizer

def convert_to_gguf(hf_model_dir, gguf_output_path):
    """Convert Hugging Face model to GGUF format."""
    convert_hf_to_gguf(hf_model_dir, gguf_output_path)

def main():
    # Prompt user for paths
    base_model_name = input("Enter the base model name or path (e.g., gpt2): ").strip()
    json_dataset_path = input("Enter the path to the JSON dataset (e.g., dataset.json): ").strip()
    output_dir = input("Enter the output directory for the fine-tuned model: ").strip()
    gguf_output_path = input("Enter the output path for the GGUF model (e.g., model.gguf): ").strip()
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    dataset = load_json_dataset(json_dataset_path)
    
    # Train model
    model, tokenizer = train_model(base_model_name, dataset, output_dir)
    
    # Convert to GGUF
    convert_to_gguf(output_dir, gguf_output_path)
    
    print(f"Fine-tuned model saved to {output_dir}")
    print(f"GGUF model saved to {gguf_output_path}")

if __name__ == "__main__":
    main()
