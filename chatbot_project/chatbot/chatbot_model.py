import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import re
from collections import Counter
from textblob import TextBlob

def correct_spelling(text):
    return str(TextBlob(text).correct())

# Load and preprocess
def preprocess(text):
    text = correct_spelling(text.lower())
    text = re.sub(r'[^\w\s]', '', text)
    return text.split()

with open('college_data.json', 'r') as f:
    data = json.load(f)

# Build vocab
vocab = Counter()
for item in data:
    vocab.update(preprocess(item['question']))
    vocab.update(preprocess(item['answer']))

tokens = ['<pad>', '<sos>', '<eos>', '<unk>'] + sorted(vocab.keys())
word2idx = {w: i for i, w in enumerate(tokens)}
idx2word = {i: w for w, i in word2idx.items()}

# Dataset
class ChatDataset(Dataset):
    def __init__(self, data):
        self.data = []
        for item in data:
            q = [word2idx.get(w, word2idx['<unk>']) for w in preprocess(item['question'])]
            a = [word2idx['<sos>']] + [word2idx.get(w, word2idx['<unk>']) for w in preprocess(item['answer'])] + [word2idx['<eos>']]
            self.data.append((torch.tensor(q), torch.tensor(a)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    q, a = zip(*batch)
    q = pad_sequence(q, batch_first=True, padding_value=word2idx['<pad>'])
    a = pad_sequence(a, batch_first=True, padding_value=word2idx['<pad>'])
    return q, a

# Model
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)

    def forward(self, x):
        embed = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(embed)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden, cell):
        x = x.unsqueeze(1)
        embed = self.embedding(x)
        output, (hidden, cell) = self.lstm(embed, (hidden, cell))
        out = self.fc(output.squeeze(1))
        return out, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size, trg_len = trg.size()
        vocab_size = len(word2idx)
        outputs = torch.zeros(batch_size, trg_len, vocab_size)

        hidden, cell = self.encoder(src)
        x = trg[:, 0]

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(x, hidden, cell)
            outputs[:, t] = output
            best_guess = output.argmax(1)
            x = trg[:, t] if torch.rand(1).item() < teacher_forcing_ratio else best_guess
        return outputs
# Hyperparameters
embed_size = 64
hidden_size = 128
batch_size = 2
num_epochs = 300
learning_rate = 0.001

# Data and loaders
dataset = ChatDataset(data)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Model, Loss, Optimizer
encoder = Encoder(len(word2idx), embed_size, hidden_size)
decoder = Decoder(len(word2idx), embed_size, hidden_size)
model = Seq2Seq(encoder, decoder)

criterion = nn.CrossEntropyLoss(ignore_index=word2idx['<pad>'])
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    total_loss = 0
    for src, trg in loader:
        output = model(src, trg)

        output = output[:, 1:].reshape(-1, output.size(2))
        trg = trg[:, 1:].reshape(-1)

        loss = criterion(output, trg)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch+1) % 50 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(loader):.4f}")

# Save the model
torch.save(model.state_dict(), "chatbot_model.pth")
