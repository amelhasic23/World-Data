from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import load_dataset
import os
import torch

# Load the model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Add pad token if not present
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Use a relative path for the CSV file
csv_file_name = "world-data-2023.csv"
csv_file_path = os.path.join(os.path.dirname(__file__), csv_file_name)

# Ensure the CSV file exists
if not os.path.exists(csv_file_path):
    raise FileNotFoundError(f"The file at {csv_file_path} does not exist.")

# Load the local dataset
dataset = load_dataset('csv', data_files={'train': csv_file_path})

# Verify the dataset columns
print("Dataset columns:", dataset['train'].column_names)

# Preprocess the dataset
def preprocess_function(examples):
    inputs = [f"summarize: {str(ex)}" for ex in examples.get("Country", [])]
    labels = [str(ex) for ex in examples.get("GDP", [])]

    # Tokenize inputs and labels with padding and attention mask
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True, padding="max_length", return_attention_mask=True)
    label_encodings = tokenizer(labels, max_length=128, truncation=True, padding="max_length", return_attention_mask=False)
    
    # Extract input_ids and attention_mask from model_inputs
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    label_ids = label_encodings["input_ids"]

    # Ensure input_ids and label_ids have the same length
    min_length = min(len(input_ids), len(label_ids))
    input_ids = input_ids[:min_length]
    label_ids = label_ids[:min_length]
    attention_mask = attention_mask[:min_length]

    # Update model_inputs with the corrected batch size and add attention_mask
    model_inputs["input_ids"] = input_ids
    model_inputs["labels"] = label_ids
    model_inputs["attention_mask"] = attention_mask

    return model_inputs

# Apply the preprocessing function to the entire dataset
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Verify that input_ids and labels have the same shape
for key in ["train"]:
    assert len(tokenized_datasets[key]["input_ids"]) == len(tokenized_datasets[key]["labels"]), \
        f"Batch size mismatch in {key} dataset"

# Set up the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Create the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
)


# Save the fine-tuned model
trainer.save_model("path_to_save_model")

# Load the fine-tuned model
fine_tuned_model = GPT2LMHeadModel.from_pretrained("path_to_save_model")

# Function to generate response from the fine-tuned model
def generate_response(text, max_length=150):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=1024)
    if inputs['input_ids'].size(1) == 0:  # Check for empty input_ids
        raise ValueError("Empty input_ids detected. Check your input text.")
    
    response_ids = fine_tuned_model.generate(
        inputs['input_ids'],
        attention_mask=inputs.get('attention_mask'),
        max_length=max_length,
        num_beams=2,  # Reduced beams for faster response
        temperature=0.7,  # Adjust temperature for more coherent responses
        top_p=0.9,  # Top-p sampling for more focused responses
        do_sample=True,  # Enable sampling
        early_stopping=True,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(response_ids[0], skip_special_tokens=True)
    return response

# Terminal interface for interactive conversation
def terminal_interface():
    print("Welcome! Type something and I will respond. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        try:
            response = generate_response(user_input)
            # Avoid repeating the same question
            if response.lower().strip() == user_input.lower().strip():
                print("Model: I don't have a new answer. Please ask a different question.")
            else:
                print("Model:", response)
        except ValueError as e:
            print(f"Error: {e}")

# Run the terminal interface
if __name__ == "__main__":
    terminal_interface()
