from django.http import JsonResponse
from .chatbot_model import Encoder, Decoder, Seq2Seq, preprocess, word2idx, idx2word
import torch
from django.views.decorators.csrf import csrf_exempt
import json
from django.shortcuts import render

def chatbot_page(request):
    return render(request, 'chat.html')

# Load model once globally
embed_size = 64
hidden_size = 128

encoder = Encoder(len(word2idx), embed_size, hidden_size)
decoder = Decoder(len(word2idx), embed_size, hidden_size)
model = Seq2Seq(encoder, decoder)
model.load_state_dict(torch.load("chatbot/chatbot_model.pth", map_location=torch.device('cpu')))
model.eval()

def predict_response(text):
    print("User Input:", text)
    with torch.no_grad():
        tokens = [word2idx.get(w, word2idx['<unk>']) for w in preprocess(text)]
        print("Tokens:", tokens)

        input_tensor = torch.tensor(tokens).unsqueeze(0)

        hidden, cell = model.encoder(input_tensor)
        x = torch.tensor([word2idx['<sos>']])

        result = []
        for _ in range(20):
            output, hidden, cell = model.decoder(x, hidden, cell)
            best_guess = output.argmax(1).item()
            print("Predicted Token:", best_guess, "->", idx2word[best_guess])
            if idx2word[best_guess] == '<eos>':
                break
            result.append(idx2word[best_guess])
            x = torch.tensor([best_guess])

        final_reply = ' '.join(result)
        print("Bot Reply:", final_reply)
        return final_reply


@csrf_exempt
def chatbot_api(request):
    if request.method == "POST":
        data = json.loads(request.body)
        user_input = data.get("message", "")
        if user_input:
            bot_reply = predict_response(user_input)
            return JsonResponse({"reply": bot_reply})
        return JsonResponse({"reply": "Please type a message."})
