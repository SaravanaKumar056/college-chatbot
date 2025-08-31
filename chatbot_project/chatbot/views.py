import os
import json
from pathlib import Path
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
import google.generativeai as genai

# --- Configuration ---

# Build path to the JSON file
BASE_DIR = Path(__file__).resolve().parent
JSON_FILE_PATH = BASE_DIR / 'college_data.json'

# Load the local college data once when the server starts
try:
    with open(JSON_FILE_PATH, 'r') as f:
        college_data = json.load(f)
except FileNotFoundError:
    print(f"Warning: {JSON_FILE_PATH} not found. The chatbot will rely solely on the Gemini API.")
    college_data = []

# --- Helper Functions ---

def find_local_response(user_input):
    """
    Searches for a simple, predefined response in the local JSON data.
    This is fast and free, ideal for common questions.
    """
    user_input_lower = user_input.lower()
    for item in college_data:
        for pattern in item['patterns']:
            # Use 'in' for broader matching
            if pattern.lower() in user_input_lower:
                # We found a match in our local data
                return item['responses'][0]
    return None # No local response found

def get_gemini_response(prompt):
    """
    Calls the Google Gemini API for more complex questions.
    """
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("Error: GOOGLE_API_KEY environment variable not found.")
            return "Sorry, my connection to the AI brain is not configured correctly."

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # --- IMPROVED PROMPT ---
        # This new prompt gives the AI more freedom to be helpful.
        full_prompt = (
            "You are a helpful and friendly chatbot for Hindustan College of Arts and Science (HICAS) in Coimbatore, India. "
            "Your goal is to answer user questions accurately and concisely. "
            "First, rely on the information in the provided `college_data.json`. "
            "If the question is about a general college topic (like courses, departments, or subjects) that isn't in the JSON, "
            "use your general knowledge to provide a typical answer for an Indian arts and science college. "
            "For example, if asked about a 'B.Sc. Physics' course, you should describe what that course typically entails. "
            "Do not just say the information is not in the JSON file. Be helpful. "
            f"Here is the user's question: '{prompt}'"
        )
        
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        print(f"An error occurred with the Gemini API: {e}")
        return "Sorry, I am having a bit of trouble thinking right now. Please try again."

# --- Django Views ---

def chatbot_page(request):
    """
    Renders the main HTML page for the chatbot.
    """
    return render(request, 'chat.html')

@csrf_exempt
def chatbot_api(request):
    """
    Handles the API requests from the frontend, gets a response, and returns it as JSON.
    """
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            user_input = data.get("message", "")
            
            if not user_input:
                return JsonResponse({"reply": "Please type a message."})

            # 1. First, try to find a simple answer in our local data.
            bot_reply = find_local_response(user_input)

            # 2. If no local answer is found, use the powerful Gemini API.
            if not bot_reply:
                bot_reply = get_gemini_response(user_input)
            
            return JsonResponse({"reply": bot_reply})

        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON in request"}, status=400)
        
    return JsonResponse({"error": "Only POST requests are accepted"}, status=405)
