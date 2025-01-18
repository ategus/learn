import torch
from transformers import BertTokenizer, BertForMaskedLM

# Set the device to run on (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the BERT model and tokenizer
model = BertForMaskedLM.from_pretrained('bert-base-uncased').to(device)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Set the prompt and maximum number of tokens to generate
prompt = "This is a simple backupscript in bash: "
max_tokens = 500

# Encode the prompt and convert it to a tensor
encoded_prompt = torch.tensor(tokenizer.encode(prompt, return_tensors='pt')).to(device)

# Generate text
output = model.generate(input_ids=encoded_prompt, max_length=max_tokens)

# Convert the output to a string and print it
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)

