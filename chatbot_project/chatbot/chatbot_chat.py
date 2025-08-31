import torch
from chatbot_model import Encoder, Decoder, Seq2Seq, preprocess, word2idx, idx2word

# Load model
embed_size = 64
hidden_size = 128
encoder = Encoder(len(word2idx), embed_size, hidden_size)
decoder = Decoder(len(word2idx), embed_size, hidden_size)
model = Seq2Seq(encoder, decoder)
model.load_state_dict(torch.load("chatbot_model.pth"))
model.eval()

def predict(question):
    with torch.no_grad():
        tokens = [word2idx.get(w, word2idx['<unk>']) for w in preprocess(question)]
        input_tensor = torch.tensor(tokens).unsqueeze(0)

        hidden, cell = model.encoder(input_tensor)
        x = torch.tensor([word2idx['<sos>']])

        result = []
        for _ in range(20):
            output, hidden, cell = model.decoder(x, hidden, cell)
            best_guess = output.argmax(1).item()
            if idx2word[best_guess] == '<eos>':
                break
            result.append(idx2word[best_guess])
            x = torch.tensor([best_guess])

        return ' '.join(result)

# Chat loop
print("Bot: Hi! Ask me anything about your college.")
while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        print("Bot: Goodbye!")
        break
    response = predict(user_input)
    print("Bot:", response)
