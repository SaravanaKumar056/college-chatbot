# chatbot/views.py
import os
import json
from pathlib import Path
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
import google.generativeai as genai

# --- Configuration ---
JSON_FILE_PATH = settings.BASE_DIR / 'chatbot/college_data.json'

try:
    with open(JSON_FILE_PATH, 'r') as f:
        college_data = json.load(f)
except FileNotFoundError:
    print(f"Warning: {JSON_FILE_PATH} not found. The chatbot will rely solely on the Gemini API.")
    college_data = []

# --- Helper Functions ---
def find_local_response(user_input):
    user_input_lower = user_input.lower()
    for item in college_data:
        for pattern in item['patterns']:
            if pattern.lower() in user_input_lower:
                return item['responses'][0]
    return None

def get_gemini_response(prompt):
    try:
        # Corrected the typo here
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("Error: GOOGLE_API_KEY environment variable not found.")
            return "Sorry, my connection to the AI brain is not configured correctly."

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # --- NEW, MORE FLEXIBLE PROMPT ---
        full_prompt = (
            "You are a helpful chatbot. Your primary role is to be an expert on Hindustan College of Arts and Science (HICAS) in Coimbatore, India. "
            "First, always try to answer questions based on HICAS. "
            "However, if the user asks a question that is clearly unrelated to the college (like about celebrities, general knowledge, or other topics), "
            "it is okay to answer that question using your general knowledge. "
            "Your personality should be friendly and helpful. "
            f"Here is the user's question: '{prompt}'"
        )
        
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        print(f"An error occurred with the Gemini API: {e}")
        return "Sorry, I am having a bit of trouble thinking right now. Please try again."

# --- Django Views ---
def chatbot_page(request):
    return render(request, 'chat.html')

@csrf_exempt
def chatbot_api(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            user_input = data.get("message", "")
            
            if not user_input:
                return JsonResponse({"reply": "Please type a message."})

            bot_reply = find_local_response(user_input)

            if not bot_reply:
                bot_reply = get_gemini_response(user_input)
            
            return JsonResponse({"reply": bot_reply})
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON in request"}, status=400)
    return JsonResponse({"error": "Only POST requests are accepted"}, status=405)

