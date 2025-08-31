import os
import google.generativeai as genai

# --- IMPORTANT ---
# 1. Get your API key from Google AI Studio: https://aistudio.google.com/app/apikey
# 2. Paste your Google AI API key directly here.
api_key = os.getenv("GOOGLE_API_KEY") 

# This check now correctly looks for the placeholder text, not your actual key.
# This ensures it only triggers if you haven't replaced the key.
if not api_key or "YOUR_GOOGLE_AI_API_KEY_HERE" in api_key:
    print("---")
    print("Error: Google AI API key is missing!")
    print("Please get a key from Google AI Studio (https://aistudio.google.com/app/apikey)")
    print("and paste it into the 'api_key' variable in the script.")
    print("---")
    exit()

# Configure the client library with your API key
genai.configure(api_key=api_key)

# Create the model. 'gemini-1.5-flash' is a fast and versatile model.
model = genai.GenerativeModel('gemini-1.5-flash')

# Start a chat session to maintain the history of the conversation
chat = model.start_chat(history=[])

def chat_with_bot(prompt):
    """
    Sends a prompt to the Google Gemini API and returns the bot's response.
    """
    try:
        # Send the user's message to the chat session
        response = chat.send_message(prompt)
        return response.text
    except Exception as e:

        print(f"An error occurred: {e}")
        return "Sorry, I couldn't process that request."

# Main chat loop
if __name__ == "__main__":
    print("Chatbot is ready! Type 'exit' or 'quit' to end the session.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        reply = chat_with_bot(user_input)
        print("Bot:", reply)
