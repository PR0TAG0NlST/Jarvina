import os
import json
import re
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, ConfigDict
from typing import Optional, Union
from datetime import datetime

# Import the new GeminiClient from the updated llm_client.py
from llm_client import GeminiClient

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Initialize FastAPI app and global client
gemini_client: Optional[GeminiClient] = None
jarvina_persona_data: Optional[dict] = None
custom_replies_map: dict = {}
CUSTOM_INSTRUCTIONS_FILE = "custom_instructions.json"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Helper function to normalize text for matching (lowercase, no punctuation, single spaces)
def normalize_text(text: str) -> str:
    """
    Converts text to lowercase, removes punctuation, and normalizes spaces.
    This ensures consistent matching regardless of user's input formatting.
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Function to load custom instructions from file
def load_custom_instructions_from_file() -> str:
    """Loads custom instructions from a JSON file."""
    if os.path.exists(CUSTOM_INSTRUCTIONS_FILE):
        try:
            with open(CUSTOM_INSTRUCTIONS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("instructions", "")
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding {CUSTOM_INSTRUCTIONS_FILE}: {e}")
            return ""
        except Exception as e:
            logging.error(f"Error loading {CUSTOM_INSTRUCTIONS_FILE}: {e}")
            return ""
    return ""

# Function to save custom instructions to file
def save_custom_instructions_to_file(instructions: str):
    """Saves custom instructions to a JSON file."""
    try:
        with open(CUSTOM_INSTRUCTIONS_FILE, 'w', encoding='utf-8') as f:
            json.dump({"instructions": instructions}, f, indent=4)
    except Exception as e:
        logging.error(f"Error saving to {CUSTOM_INSTRUCTIONS_FILE}: {e}")
        raise

# --- Lifespan Event Handler ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global gemini_client
    global jarvina_persona_data
    global custom_replies_map

    try:
        gemini_client = GeminiClient()
        print("GeminiClient initialized successfully.")
    except ValueError as e:
        print(f"Error during client initialization: {e}")
        gemini_client = None
        # Consider a more graceful failure if the key is missing

    # Load persona data and custom replies from memory.json
    memory_file_path = "memory.json"
    if os.path.exists(memory_file_path):
        try:
            with open(memory_file_path, 'r', encoding='utf-8') as f:
                memory_data = json.load(f)

            jarvina_persona_data = memory_data.get("persona_data")
            if jarvina_persona_data:
                print(f"Persona data loaded from {memory_file_path} successfully.")
            else:
                print(f"Warning: 'persona_data' not found in {memory_file_path}. Using default persona.")

            loaded_custom_replies = memory_data.get("custom_replies", [])
            for reply_entry in loaded_custom_replies:
                response = reply_entry.get("response")
                phrases = reply_entry.get("phrases", [])
                if response and phrases:
                    for phrase in phrases:
                        custom_replies_map[normalize_text(phrase)] = response
            if custom_replies_map:
                print(f"Custom replies loaded from {memory_file_path} successfully.")
            else:
                print(f"Warning: 'custom_replies' not found or empty in {memory_file_path}.")

        except json.JSONDecodeError as e:
            print(f"Error decoding memory.json: {e}. Persona data and custom replies will not be used.")
            jarvina_persona_data = None
            custom_replies_map = {}
        except Exception as e:
            print(f"Error loading memory.json: {e}. Persona data and custom replies will not be used.")
            jarvina_persona_data = None
            custom_replies_map = {}
    else:
        print(f"Warning: {memory_file_path} not found. Jarvina will use default persona and no custom replies.")
        jarvina_persona_data = None
        custom_replies_map = {}

    yield
    print("FastAPI application shutting down.")

# Pass the lifespan function to the FastAPI app constructor
app = FastAPI(lifespan=lifespan)

# Configure Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Pydantic model for the expected chat request body
class ChatRequest(BaseModel):
    messages: Optional[list[dict]] = None
    prompt: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None

    model_config = ConfigDict(extra='allow')

# Pydantic model for saving custom instructions
class CustomInstructionsRequest(BaseModel):
    instructions: str

# --- GENERAL CHAT ENDPOINT ---
@app.post("/api/chat/mistral/")
async def chat_endpoint(chat_request: ChatRequest):
    """
    Handles general chat requests, dynamically adjusting the system prompt
    and using the single Gemini model for all intents.
    """
    if gemini_client is None:
        raise HTTPException(status_code=500, detail="GeminiClient not initialized. Please check your API key.")

    logging.info(f"Received payload from frontend: {chat_request.model_dump_json(indent=2)}")

    messages_to_send = []
    user_prompt_content = ""

    # Check for custom replies first, this is a quick escape
    normalized_user_input = normalize_text(chat_request.prompt or "")
    if normalized_user_input in custom_replies_map:
        logging.info(f"Matched custom reply for normalized input '{normalized_user_input}'")
        return JSONResponse(content={"response": custom_replies_map[normalized_user_input]})

    # Handle direct commands like "current time"
    if "current time" in normalized_user_input or "what time is it" in normalized_user_input or "day date and time" in normalized_user_input:
        current_time_str = datetime.now().strftime('%I:%M:%S %p on %A, %B %d, %Y')
        return JSONResponse(content={"response": f"The current time in Delhi, India is {current_time_str}."})

    # Prepare initial system message content based on persona data and custom instructions
    system_content_parts = []
    emoji_constraint_message = "CRITICAL INSTRUCTION: ABSOLUTELY DO NOT use emojis, emoticons, or any decorative characters in your responses. Maintain a strictly formal, concise, and professional tone at all times, especially for technical or informational content. Any use of emojis will be considered a failure."
    system_content_parts.append(emoji_constraint_message)

    custom_instructions = load_custom_instructions_from_file()
    if custom_instructions:
        system_content_parts.append(f"User's Custom Instructions: {custom_instructions}")
        logging.info("Custom instructions loaded and added to system message.")

    if jarvina_persona_data:
        name = jarvina_persona_data.get("name", "AI Assistant")
        mode = jarvina_persona_data.get("mode", "a helpful companion")
        personalities = ", ".join(jarvina_persona_data.get("personalities", ["helpful", "friendly"]))
        voice_tone = jarvina_persona_data.get("voice_tone", {}).get("style", "natural")
        jarvina_rules = jarvina_persona_data.get("jarvina_rules", [])
        contextual_defaults = jarvina_persona_data.get("contextual_defaults", {})
        user_name = contextual_defaults.get("user_name", "user")
        motivation = contextual_defaults.get("motivation", "to assist you")

        system_content_parts.append(f"You are an AI named {name}. Your primary role is {mode}. You are {personalities}.")
        system_content_parts.append(f"Your voice tone is {voice_tone}.")
        system_content_parts.append(f"The user's name is {user_name}. Their motivation is: '{motivation}'.")

        if jarvina_rules:
            system_content_parts.append("Here are your specific rules:")
            for rule in jarvina_rules:
                system_content_parts.append(f"- {rule}")
    else:
        system_content_parts.append("You are Jarvina your personal AI assistant, designed to think alongside you, automate your world, and evolve with your ambition.")
    
    system_message_text = "\n".join(system_content_parts)

    # Reformat messages for Gemini API
    gemini_formatted_messages = []
    
    # Prepend system instructions to the first user message if chat history is empty
    if not chat_request.messages:
        # If the request comes with just a prompt, it's a new conversation
        full_user_prompt = f"{system_message_text}\n\n{chat_request.prompt}"
        gemini_formatted_messages.append({
            "role": "user",
            "parts": [{"text": full_user_prompt}]
        })
    else:
        # If there's chat history, process it
        for message in chat_request.messages:
            role = message.get("role")
            content = message.get("content")

            if role == "system":
                # Skip system role, as it's not supported in multi-turn conversation history
                # This logic assumes the very first message is handled correctly
                continue
            elif role == "assistant":
                # Convert "assistant" role to "model" for Gemini API compatibility
                gemini_formatted_messages.append({"role": "model", "parts": [{"text": content}]})
            elif role == "user":
                # Standard user message
                gemini_formatted_messages.append({"role": "user", "parts": [{"text": content}]})

    logging.info(f"Messages sent to LLM: {gemini_formatted_messages}")

    try:
        response_content = gemini_client.generate_content(
            messages=gemini_formatted_messages,
            temperature=0.7,
        )
        logging.info("Successfully generated response with the Gemini client.")
        return JSONResponse(content={"response": response_content})
    except Exception as e:
        logging.error(f"Error occurred during Gemini generation: {e}")
        raise HTTPException(status_code=500, detail=f"AI generation failed: {str(e)}")

# --- New Endpoints for Custom Instructions ---
@app.post("/api/save_custom_instructions")
async def save_custom_instructions(request: CustomInstructionsRequest):
    """Saves custom instructions received from the frontend."""
    try:
        save_custom_instructions_to_file(request.instructions)
        return JSONResponse(content={"message": "Custom instructions saved successfully!"})
    except Exception as e:
        logging.error(f"Failed to save custom instructions: {e}")
        raise HTTPException(status_code=500, detail="Failed to save custom instructions.")

@app.get("/api/load_custom_instructions")
async def load_custom_instructions_api():
    """Loads custom instructions and sends them to the frontend."""
    try:
        instructions = load_custom_instructions_from_file()
        return JSONResponse(content={"instructions": instructions})
    except Exception as e:
        logging.error(f"Failed to load custom instructions: {e}")
        raise HTTPException(status_code=500, detail="Failed to load custom instructions.")


# --- Existing Endpoints (for custom HTML forms) ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
