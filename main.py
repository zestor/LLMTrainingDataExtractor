import openai
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import torch

# Set OpenAI API key
openai.api_key = "your_openai_api_key"  # Replace with your OpenAI API key

# Load public datasets for generating prompts
def load_prompts():
    squad = load_dataset("squad")
    prompts = [item["question"] for item in squad["train"]]
    return prompts

# Load local LLM model and tokenizer
def load_local_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

# Generate response from local LLM
def generate_response(prompt, model, tokenizer, max_length=150):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Evaluate response using GPT-4o and supplement if necessary
def evaluate_and_supplement(prompt, local_response):
    evaluation_prompt = (
        f"Evaluate the quality of this response:\nPrompt: {prompt}\nResponse: {local_response}\n"
        "Rate on a scale of 1 (poor) to 10 (excellent), and explain why."
    )
    evaluation = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": evaluation_prompt}]
    )
    
    # Extract rating from GPT-4o's evaluation
    rating_text = evaluation["choices"][0]["message"]["content"]
    rating = int(rating_text.split()[0])  # Extract the first number as the rating

    if rating < 7:  # Threshold for low-quality responses
        supplement_prompt = f"Generate a better response to this prompt:\n{prompt}"
        supplement_response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": supplement_prompt}]
        )
        return supplement_response["choices"][0]["message"]["content"]
    
    return local_response

# Save dataset to file
def save_to_file(dataset, filename):
    with open(filename, "w") as file:
        for entry in dataset:
            file.write(f"Prompt: {entry['prompt']}\nResponse: {entry['response']}\n\n")

# Fine-tune the local LLM using PEFT (LoRA)
def fine_tune_with_peft(base_model_name, dataset_file):
    # Load tokenizer and base model
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)

    # Configure LoRA (Low-Rank Adaptation) for PEFT
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1
    )
    
    # Apply LoRA to the base model
    peft_model = get_peft_model(base_model, lora_config)

    # Load dataset for fine-tuning
    dataset = load_dataset("json", data_files={"train": dataset_file})

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./peft_fine_tuned_model",
        per_device_train_batch_size=8,
        num_train_epochs=3,
        save_steps=500,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=10,
        learning_rate=5e-5,
        fp16=True,
        push_to_hub=False  # Set to True if you want to upload to Hugging Face Hub
    )

    # Fine-tune the model using Trainer API
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=dataset["train"],
        tokenizer=tokenizer
    )
    
    trainer.train()

# Main program execution
if __name__ == "__main__":
    # Step 1: Load prompts from public datasets
    print("Loading prompts...")
    prompts = load_prompts()

    # Step 2: Load the local LLM and tokenizer
    print("Loading local LLM...")
    model_name = "your-local-model"  # Replace with your local model path or identifier
    tokenizer, model = load_local_model(model_name)

    # Step 3: Generate responses and evaluate/supplement them using GPT-4o
    print("Generating responses and evaluating...")
    final_dataset = []
    
    for prompt in prompts:
        try:
            local_response = generate_response(prompt, model, tokenizer)
            final_response = evaluate_and_supplement(prompt, local_response)
            final_dataset.append({"prompt": prompt, "response": final_response})
            print(f"Processed Prompt: {prompt}")
        
        except Exception as e:
            print(f"Error processing prompt: {prompt}. Error: {e}")

    # Step 4: Save the final dataset to a file
    print("Saving final dataset...")
    save_to_file(final_dataset, "final_dataset.jsonl")

    # Step 5: Fine-tune the local LLM using PEFT with adapter weights
    print("Fine-tuning the local LLM...")
    fine_tune_with_peft(model_name, "final_dataset.jsonl")

