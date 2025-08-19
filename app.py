import os
import json
import re
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, ConfigDict
from openrouter_client import OpenRouterClient
from dotenv import load_dotenv
from typing import Optional, Union
from datetime import datetime
import logging

# Import DDGS from duckduckgo_search
from duckduckgo_search import DDGS

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
openrouter_client: Optional[OpenRouterClient] = None
jarvina_persona_data: Optional[dict] = None # Global variable to store loaded persona data
custom_replies_map: dict = {} # Global variable to store custom replies for quick lookup
CUSTOM_INSTRUCTIONS_FILE = "custom_instructions.json" # File to store custom instructions

# Define models for different use cases
FAIL_SAFE_MODEL = "mistralai/mistral-7b-instruct-v0.2"
EMOTIONAL_MODEL = "openai/gpt-4o"
ANALYSIS_MODEL = "anthropic/claude-3-sonnet"
CODING_MODEL = "google/gemini-pro"
GENERAL_CHAT_MODEL = "mistralai/mistral-7b-instruct-v0.2"
NEW_DEFAULT_MODEL = "openai/gpt-4o" # New default model as per your request

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Helper function to normalize text for matching (lowercase, no punctuation, single spaces)
def normalize_text(text: str) -> str:
    """
    Converts text to lowercase, removes punctuation, and normalizes spaces.
    This ensures consistent matching regardless of user's input formatting.
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation (keep alphanumeric and spaces)
    text = re.sub(r'\s+', ' ', text)    # Replace multiple spaces with a single space
    return text.strip()                 # Remove leading/trailing spaces

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
    global openrouter_client
    global jarvina_persona_data
    global custom_replies_map

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set. Please set it in your .env file.")
    openrouter_client = OpenRouterClient(api_key=api_key)
    print("OpenRouterClient initialized successfully.")

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

            # Process custom replies into a lookup map
            loaded_custom_replies = memory_data.get("custom_replies", [])
            for reply_entry in loaded_custom_replies:
                response = reply_entry.get("response")
                phrases = reply_entry.get("phrases", [])
                if response and phrases:
                    for phrase in phrases:
                        # Normalize the phrase from memory.json before storing as key
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
    max_tokens: Optional[int] = None # This field will now be ignored if present in chat_request

    model_config = ConfigDict(extra='allow')

# Pydantic model for saving custom instructions
class CustomInstructionsRequest(BaseModel):
    instructions: str

# Define the comprehensive fallback model list
# The order here is crucial for the fallback system
ALL_FALLBACK_MODELS = [
    "openai/gpt-4o",
    "google/gemini-pro",
    "anthropic/claude-3-sonnet",
    "meta-llama/llama-3-8b-instruct",
    "deepseek/deepseek-coder",
    "mistralai/mistral-7b-instruct-v0.2",
]

# --- GENERAL CHAT ENDPOINT FOR CLONE ---
@app.post("/api/chat/mistral/")
async def chat_endpoint(chat_request: ChatRequest):
    """
    Handles general chat requests from the clone, using dynamic model switching
    and token allocation based on intent and context.
    """
    if openrouter_client is None:
        raise HTTPException(status_code=500, detail="OpenRouterClient not initialized.")

    logging.info(f"Received payload from frontend: {chat_request.model_dump_json(indent=2)}")

    messages_to_send = []
    user_prompt_content = ""

    # Initialize frontend_system_message to None at the start of the function
    frontend_system_message = None 
    
    # --- Initialize system_content_parts here, before any conditional appends ---
    system_content_parts = [] 

    if chat_request.messages is not None and len(chat_request.messages) > 0:
        messages_to_send = chat_request.messages
        # Get the last user message to check for commands
        for msg in reversed(messages_to_send):
            if msg.get("role") == "user" and msg.get("content"):
                user_prompt_content = msg["content"]
                break
        # Find the frontend system message if it exists in the incoming messages
        frontend_system_message = next((msg for msg in messages_to_send if msg.get("role") == "system"), None)
    elif chat_request.prompt is not None and chat_request.prompt.strip() != "":
        user_prompt_content = chat_request.prompt
        messages_to_send = [{"role": "user", "content": chat_request.prompt}]
    else:
        raise HTTPException(status_code=400, detail="No valid 'messages' or 'prompt' field found in the request payload.")

    # Normalize user input for custom reply and direct command matching
    normalized_user_input = normalize_text(user_prompt_content)
    logging.info(f"Normalized user input: '{normalized_user_input}' (Original: '{user_prompt_content}')")

    # --- Extract explicit token/word request ---
    explicit_token_request = None
    match = re.search(r'(\d+)\s*(words|tokens)', normalized_user_input)
    if match:
        try:
            explicit_token_request = int(match.group(1))
            logging.info(f"Detected explicit token request: {explicit_token_request}")
        except ValueError:
            logging.warning(f"Could not parse explicit token request from '{match.group(1)}'")


    # --- Check for custom replies first from the loaded map ---
    if normalized_user_input in custom_replies_map:
        logging.info(f"Matched custom reply for normalized input '{normalized_user_input}'")
        return JSONResponse(content={"response": custom_replies_map[normalized_user_input]})

    # --- Direct Command Handling (e.g., Time and DuckDuckGo Search) ---
    if "current time" in normalized_user_input or "what time is it" in normalized_user_input or "day date and time" in normalized_user_input:
        current_time_str = datetime.now().strftime('%I:%M:%S %p on %A, %B %d, %Y')
        return JSONResponse(content={"response": f"The current time in Delhi, India is {current_time_str}."})
    
    # --- DuckDuckGo Search Integration ---
    if normalized_user_input.startswith("search for ") or normalized_user_input.startswith("what is ") or normalized_user_input.startswith("find out about "):
        query = normalized_user_input.replace("search for ", "").replace("what is ", "").replace("find out about ", "").strip()
        if query:
            logging.info(f"Performing DuckDuckGo search for: {query}")
            try:
                # Perform the search. You can adjust max_results as needed.
                search_results = DDGS().text(keywords=query, max_results=3)
                
                if search_results:
                    # Format the results for the AI. You might want to refine this.
                    formatted_results = "\n\n".join([f"Title: {r['title']}\nURL: {r['href']}\nSnippet: {r['body']}" for r in search_results])
                    
                    # Add search results to system_content_parts to inform the LLM
                    system_content_parts.append(f"I have performed a DuckDuckGo search for '{query}'. Here are the top results. Please use this information to answer the user's question, citing the source if appropriate:\n\n{formatted_results}")
                    
                    logging.info("DuckDuckGo search results integrated into system message.")
                else:
                    system_content_parts.append(f"I performed a DuckDuckGo search for '{query}' but found no results. Please try another query or rephrase.")
                    logging.warning(f"No DuckDuckGo search results for: {query}")
            except Exception as e:
                logging.error(f"Error during DuckDuckGo search: {e}")
                system_content_parts.append("I encountered an error while trying to perform the search. Please try again.")


    # --- Prepare initial system message content based on persona data ---
    # This block was moved here to ensure system_content_parts is always initialized
    # before any conditional appends.
    # The previous `system_content_parts = []` was removed from here.

    # Load custom instructions and add them to the system message
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
    
    # Add the explicit emoji constraint to the base system message (strengthened)
    emoji_constraint_message = "CRITICAL INSTRUCTION: ABSOLUTELY DO NOT use emojis, emoticons, or any decorative characters in your responses. Maintain a strictly formal, concise, and professional tone at all times, especially for technical or informational content. Any use of emojis will be considered a failure."
    system_content_parts.insert(0, emoji_constraint_message) # Insert at the beginning for higher priority

    # --- Heuristics for dynamic model and token adjustment (Intent Classification) ---
    # Calculate number of lines in user prompt (strip leading/trailing whitespace from each line)
    user_prompt_lines = len([line for line in user_prompt_content.split('\n') if line.strip()])
    if user_prompt_lines == 0 and user_prompt_content.strip() != "": # Handle single line inputs without newline
        user_prompt_lines = 1

    # Keywords for different intents (aligned with RTF document where possible)
    heard_only_keywords = ["i feel", "just needed", "don’t know why", "venting", "no need to reply", "just saying", "i just want to talk", "i need to get this off my chest"]
    reply_expected_keywords = ["what do you think", "can you tell", "why", "how", "should i", "what if", "explain", "describe", "what is", "tell me about", "elaborate", "discuss", "analyze", "provide details", "in depth", "give me information", "can you tell me", "can you explain", "define", "compare", "contrast", "implications", "effects", "impact", "significance", "describe your thoughts on", "what are your views on", "meaning of", "definition of"]
    
    emotional_keywords_general = ["feeling", "feel", "sad", "happy", "anxious", "stressed", "emotional", "how are you feeling", "depressed", "overwhelmed", "frustrated", "lonely", "joyful", "excited", "upset", "down", "confused", "worried", "scared", "angry", "hopeful", "grateful", "content", "my mood is"]
    
    analysis_keywords = [
        "analyze", "data", "report", "statistics", "trend", "chart", "graph", "metrics",
        "predict", "forecast", "correlation", "distribution", "summary", "breakdown"
    ]
    coding_keywords = [
        "code", "program", "script", "function", "bug", "error", "syntax", "develop",
        "implement", "debug", "algorithm", "language", "python", "javascript", "html", "css",
        "java", "c++", "react", "api", "framework", "library", "class", "method", "variable"
    ]

    # --- Determine Intent Flags based on RTF logic ---
    is_heard_only = any(keyword in normalized_user_input for keyword in heard_only_keywords)
    is_reply_expected = any(keyword in normalized_user_input for keyword in reply_expected_keywords) or user_prompt_content.endswith("?")
    is_emotional_general = any(keyword in normalized_user_input for keyword in emotional_keywords_general)
    is_analysis_query = any(keyword in normalized_user_input for keyword in analysis_keywords)
    is_coding_query = any(keyword in normalized_user_input for keyword in coding_keywords)
    is_search_query = normalized_user_input.startswith("search for ") or normalized_user_input.startswith("what is ") or normalized_user_input.startswith("find out about ")

    # Define the primary intent and initial model/temp/tokens
    # Default to 4000 tokens as per your request
    max_tokens_to_use = 4000
    model_to_use = NEW_DEFAULT_MODEL # Default model
    temperature_to_use = 0.7 # Default temperature

    # --- Intent Prioritization and Model/Token Assignment ---
    # Prioritize search queries if detected
    if is_search_query:
        model_to_use = NEW_DEFAULT_MODEL # Or a more factual model if preferred
        temperature_to_use = 0.4 # Less creative, more factual for search results
        logging.info("Detected search query. Using NEW_DEFAULT_MODEL for factual response based on search results.")
        system_content_parts.append("You have been provided with DuckDuckGo search results. Please synthesize this information to answer the user's question concisely and accurately. If no relevant information is found, state that you couldn't find a direct answer based on the search.")
    elif is_heard_only:
        model_to_use = NEW_DEFAULT_MODEL
        temperature_to_use = 0.9 # More creative/empathetic for pure listening
        logging.info("Detected 'HEARD_ONLY' intent. Using NEW_DEFAULT_MODEL for empathy.")
        system_content_parts.append("When responding to emotional expressions where the user primarily wants to be heard, prioritize active listening, empathy, and validation. Acknowledge their feelings without immediately offering advice or solutions. Your goal is to make the user feel heard and understood. Keep responses concise and supportive, aiming for a comprehensive response within the allocated token budget (up to 4000 tokens).")
    elif is_emotional_general and is_reply_expected: # "long yet emotional" from RTF, or emotional question
        model_to_use = NEW_DEFAULT_MODEL # Use emotional model for empathetic tone
        temperature_to_use = 0.7 # Balanced: empathetic but also factual/explanatory
        logging.info("Detected 'HYBRID' (emotional + reply expected) intent. Using NEW_DEFAULT_MODEL.")
        system_content_parts.append("When responding to questions that blend emotional expression with a request for explanation or insight, balance empathy and validation with providing clear, thoughtful, and detailed information. Acknowledge the user's feelings first, then offer the requested explanation or perspective in a supportive tone. Strive for a comprehensive and detailed response, utilizing the full allocated token budget (up to 4000 tokens).")
    elif is_emotional_general: # General emotional expression, not explicitly "heard only" or a question
        model_to_use = NEW_DEFAULT_MODEL
        temperature_to_use = 0.8 # More creative/empathetic
        logging.info("Detected general emotional query. Using NEW_DEFAULT_MODEL for empathy.")
        system_content_parts.append("When responding to emotional expressions, prioritize active listening, empathy and validation. Acknowledge the user's feelings before offering any advice or solutions. Your goal is to make the user feel heard and understood. You may offer gentle, supportive suggestions if appropriate after validation. Aim for a comprehensive response within the allocated token budget (up to 4000 tokens).")
    elif is_analysis_query:
        model_to_use = NEW_DEFAULT_MODEL
        temperature_to_use = 0.2 # More factual/less creative
        logging.info("Detected analysis query. Using NEW_DEFAULT_MODEL.")
        system_content_parts.append("Provide a comprehensive and detailed analysis, utilizing the full allocated token budget (up to 4000 tokens) to ensure thoroughness.")
    elif is_coding_query:
        model_to_use = NEW_DEFAULT_MODEL
        temperature_to_use = 0.1 # More precise/less creative
        logging.info("Detected coding query. Using NEW_DEFAULT_MODEL.")
        system_content_parts.append("Provide detailed code examples and explanations, utilizing the full allocated token budget (up to 4000 tokens) to ensure clarity and completeness.")
    elif is_reply_expected: # This covers general informational questions
        model_to_use = NEW_DEFAULT_MODEL # Informational queries use the new default (Mistral)
        temperature_to_use = 0.6 # Slightly less creative for factual info
        logging.info(f"Detected informational query. Using {model_to_use}.")
        system_content_parts.append("Provide a clear, structured, and comprehensive explanation. Utilize the full allocated token budget (up to 4000 tokens) to ensure thoroughness and depth in your response.")
    else: # Default for normal conversations, now also aiming for 4000 tokens
        model_to_use = NEW_DEFAULT_MODEL # General chat uses the new default (Mistral)
        temperature_to_use = 0.7
        logging.info(f"Detected general chat. Using {model_to_use} with 4000 max_tokens.")
        system_content_parts.append("Provide a comprehensive and detailed response, utilizing the full allocated token budget (up to 4000 tokens) if necessary to ensure thoroughness.")

    # Apply explicit token request override (if any)
    # This override happens AFTER intent-based model selection, ensuring user's explicit request takes precedence
    if explicit_token_request is not None:
        max_tokens_to_use = max(200, min(explicit_token_request, 4000)) # Max to 4000
        logging.info(f"Overriding max_tokens to explicit request: {max_tokens_to_use}")

    # Construct the final system message
    # If frontend_system_message exists, append our parts to it
    if frontend_system_message:
        # Append the dynamically generated system_content_parts to the frontend's system message
        frontend_system_message["content"] += "\n\n" + "\n".join(system_content_parts)
        # Ensure only one system message is at the beginning
        final_messages_for_llm = [frontend_system_message] + [msg for msg in messages_to_send if msg.get("role") != "system"]
    else:
        # Otherwise, create a new system message from our parts
        default_system_message = {
            "role": "system",
            "content": "\n".join(system_content_parts)
        }
        final_messages_for_llm = [default_system_message] + messages_to_send

    logging.info(f"Final Model: {model_to_use}, Temp: {temperature_to_use}, Max Tokens: {max_tokens_to_use}")
    logging.info(f"Messages sent to LLM: {final_messages_for_llm}")

    # Define the ordered list of models to try for this endpoint
    models_to_try = [model_to_use] + [m for m in ALL_FALLBACK_MODELS if m != model_to_use]

    for model in models_to_try:
        try:
            logging.info(f"Attempting to use model: {model}")
            response_content = openrouter_client.generate_text(
                model=model,
                messages=final_messages_for_llm,
                temperature=temperature_to_use,
                max_tokens=max_tokens_to_use,
            )
            logging.info(f"Successfully generated response with model: {model}")
            return JSONResponse(content={"response": response_content})
        except Exception as e:
            logging.error(f"Error occurred with model {model}: {e}")
            if model == models_to_try[-1]:
                # If this is the last model in the list and it failed, raise the final exception
                logging.error("All model fallbacks failed. Raising final exception.")
                raise HTTPException(status_code=500, detail=f"All AI generation models failed: {str(e)}")


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

@app.post("/emotional_talk", response_class=HTMLResponse)
async def emotional_talk_endpoint(request: Request, prompt: str = Form(...)):
    if openrouter_client is None:
        raise HTTPException(status_code=500, detail="OpenRouterClient not initialized.")
    messages = [{"role": "user", "content": prompt}]
    
    # Enhanced prompt for emotional talk to ensure continuous numbering
    # Added a clear header and emphasized the numbering format
    emotional_prompt_content = (
        f"I'm sorry to hear that you're feeling {prompt.lower().replace('i am feeling ', '').replace('i feel ', '')}. "
        f"It's important to remember that you're not alone and there are ways to feel more connected. "
        f"Here are a few suggestions that might help:\n\n"
        f"Please provide several distinct suggestions as a SINGLE, CONTINUOUS numbered list (1., 2., 3., etc.). " # Emphasized continuous numbering
        f"Ensure the suggestions are helpful, empathetic, and actionable. "
    )
    
    # Define the ordered list of models to try, starting with the intent-specific model
    models_to_try = [EMOTIONAL_MODEL] + [m for m in ALL_FALLBACK_MODELS if m != EMOTIONAL_MODEL]

    for model in models_to_try:
        try:
            logging.info(f"Attempting to use model: {model}")
            response_text = openrouter_client.generate_text(
                model=model,
                messages=[{"role": "user", "content": emotional_prompt_content}], # Use the enhanced prompt
                temperature=0.8,
                max_tokens=4000,
            )
            logging.info(f"Successfully generated response with model: {model}")
            return templates.TemplateResponse("index.html", {
                "request": request,
                "emotional_response": response_text,
                "emotional_prompt": prompt
            })
        except Exception as e:
            logging.error(f"Error occurred with model {model}: {e}")
            if model == models_to_try[-1]:
                logging.error("All model fallbacks failed. Raising final exception.")
                raise HTTPException(status_code=500, detail=f"All AI generation models failed: {str(e)}")

@app.post("/data_analysis", response_class=HTMLResponse)
async def data_analysis_endpoint(request: Request, prompt: str = Form(...)):
    if openrouter_client is None:
        raise HTTPException(status_code=500, detail="OpenRouterClient not initialized.")
    messages = [{"role": "user", "content": prompt}]
    
    # Define the ordered list of models to try, starting with the intent-specific model
    models_to_try = [ANALYSIS_MODEL] + [m for m in ALL_FALLBACK_MODELS if m != ANALYSIS_MODEL]

    for model in models_to_try:
        try:
            logging.info(f"Attempting to use model: {model}")
            response_text = openrouter_client.generate_text(
                model=model,
                messages=messages,
                temperature=0.2,
                max_tokens=4000,
            )
            logging.info(f"Successfully generated response with model: {model}")
            return templates.TemplateResponse("index.html", {
                "request": request,
                "analysis_response": response_text,
                "analysis_prompt": prompt
            })
        except Exception as e:
            logging.error(f"Error occurred with model {model}: {e}")
            if model == models_to_try[-1]:
                logging.error("All model fallbacks failed. Raising final exception.")
                raise HTTPException(status_code=500, detail=f"All AI generation models failed: {str(e)}")

@app.post("/coding_help", response_class=HTMLResponse)
async def coding_help_endpoint(request: Request, prompt: str = Form(...)):
    # Corrected '===' to 'is' for Python syntax
    if openrouter_client is None: 
        raise HTTPException(status_code=500, detail="OpenRouterClient not initialized.")
    messages = [{"role": "user", "content": prompt}]

    # Define the ordered list of models to try, starting with the intent-specific model
    models_to_try = [CODING_MODEL] + [m for m in ALL_FALLBACK_MODELS if m != CODING_MODEL]

    for model in models_to_try:
        try:
            logging.info(f"Attempting to use model: {model}")
            response_text = openrouter_client.generate_text(
                model=model,
                messages=messages,
                temperature=0.1,
                max_tokens=4000,
            )
            logging.info(f"Successfully generated response with model: {model}")
            return templates.TemplateResponse("index.html", {
                "request": request,
                "coding_response": response_text,
                "coding_prompt": prompt
            })
        except Exception as e:
            logging.error(f"Error occurred with model {model}: {e}")
            if model == models_to_try[-1]:
                logging.error("All model fallbacks failed. Raising final exception.")
                raise HTTPException(status_code=500, detail=f"All AI generation models failed: {str(e)}")
