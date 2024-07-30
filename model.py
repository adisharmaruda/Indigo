import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm

class QADataset(Dataset):
    def __init__(self, questions, answers, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.inputs = []
        
        for question, answer in zip(questions, answers):
            text = f"<|startoftext|>Question: {question} Answer: {answer}<|endoftext|>"
            encodings = tokenizer(text, truncation=True, max_length=self.max_length, padding="max_length")
            self.inputs.append({key: torch.tensor(val) for key, val in encodings.items()})

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        if idx >= len(self.inputs):
            raise IndexError(f"Index {idx} out of range for dataset with {len(self.inputs)} items")
        return self.inputs[idx]

def train_model(model, train_dataloader, val_dataloader, device, num_epochs=3):
    optimizer = AdamW(model.parameters(), lr=5e-5)
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Average training loss: {avg_train_loss}")
        
        # Validation
        model.eval()
        total_eval_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
                total_eval_loss += outputs.loss.item()
        
        avg_val_loss = total_eval_loss / len(val_dataloader)
        print(f"Validation Loss: {avg_val_loss}")
    
    return model

def main():
    # Load your dataset
    df = pd.read_json("hf://datasets/toughdata/quora-question-answer-dataset/Quora-QuAD.jsonl", lines=True)

    questions = df['question'].tolist()
    answers = df['answer'].tolist()

    # Split the data
    train_questions, val_questions, train_answers, val_answers = train_test_split(questions, answers, test_size=0.1, random_state=42)

    # Initialize tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    # Prepare datasets
    train_dataset = QADataset(train_questions, train_answers, tokenizer)
    val_dataset = QADataset(val_questions, val_answers, tokenizer)

    # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=2)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Train the model
    trained_model = train_model(model, train_dataloader, val_dataloader, device)

    # Save the model
    trained_model.save_pretrained('qa_gpt2_model')
    tokenizer.save_pretrained('qa_gpt2_model')

if __name__ == '__main__':
    main()

# Inference
def answer_question(question, model, tokenizer, max_length=200):
    input_text = f"<|startoftext|>Question: {question} Answer:"
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, 
                            no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.4)
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    answer = generated_text.split("Answer:")[-1].strip()
    return answer

# Example usage
model = GPT2LMHeadModel.from_pretrained('qa_gpt2_model')
tokenizer = GPT2Tokenizer.from_pretrained('qa_gpt2_model')

question = "What should writers know?"
answer = answer_question(question, model, tokenizer)
print(f"Question: {question}")
print(f"Answer: {answer}")
