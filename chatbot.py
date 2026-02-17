import torch
from training import GPTLanguageModel, decode, encode, device

# Load checkpoint
checkpoint = torch.load("checkpoint.pth", map_location=device)
model = GPTLanguageModel(vocab_size=checkpoint["config"]["vocab_size"])
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

def generate_text(prompt, max_new_tokens=512):
    # Encode input string to tensor
    input_ids = torch.tensor([encode(prompt)], dtype=torch.long).to(device)

    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=max_new_tokens)

    generated = decode(out[0].tolist())
    return generated


print('\n' + generate_text("yo what's up, now your name is chatbot"))