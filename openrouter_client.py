import os
import requests
import json
import time
from dotenv import load_dotenv
from typing import Optional # Import Optional for clearer type hints

# Load environment variables from .env file
load_dotenv()

class OpenRouterClient:
    """
    A client for interacting with the OpenRouter API.
    """
    def __init__(self, api_key: str):
        """
        Initializes the OpenRouterClient with the given API key.

        Args:
            api_key (str): Your OpenRouter API key.
        """
        if not api_key:
            raise ValueError("API key cannot be empty.")
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def generate_text(self, model: str, messages: list, temperature: float = 0.7, max_tokens: int = 150, fallback_model: Optional[str] = None):
        """
        Generates text using the specified OpenRouter model, with an optional fallback.

        Args:
            model (str): The name of the primary model to use (e.g., "openai/gpt-3.5-turbo").
            messages (list): A list of message objects for the conversation.
                             Each message should be a dictionary with 'role' and 'content' keys.
                             Example: [{"role": "user", "content": "Hello!"}]
            temperature (float): Controls the randomness of the output. Higher values mean more random.
            max_tokens (int): The maximum number of tokens to generate.
            fallback_model (str, optional): The name of a model to use if the primary model fails. Defaults to None.

        Returns:
            str: The generated text, or an error message if the request fails.
        """
        current_model = model
        for attempt in range(2): # Try primary model, then fallback once if specified
            payload = {
                "model": current_model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            response = None # Initialize response to None to satisfy Pylance

            try:
                print(f"Attempting to use model: {current_model}")
                response = requests.post(self.base_url, headers=self.headers, data=json.dumps(payload))
                response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
                response_data = response.json()

                if response_data and response_data.get("choices"):
                    return response_data["choices"][0]["message"]["content"]
                else:
                    return f"Error: Unexpected response structure from {current_model} - {response_data}"

            except requests.exceptions.HTTPError as http_err:
                error_message = f"HTTP error occurred with {current_model}: {http_err} - {response.text if response else 'No response body'}"
                print(error_message)
            except requests.exceptions.ConnectionError as conn_err:
                error_message = f"Connection error occurred with {current_model}: {conn_err}"
                print(error_message)
            except requests.exceptions.Timeout as timeout_err:
                error_message = f"Timeout error occurred with {current_model}: {timeout_err}"
                print(error_message)
            except requests.exceptions.RequestException as req_err:
                # Catch all other requests exceptions
                error_message = f"An unexpected request error occurred with {current_model}: {req_err} - {response.text if response else 'No response body'}"
                print(error_message)
            except json.JSONDecodeError as json_err:
                error_message = f"JSON decoding error from {current_model}: {json_err} - Response text: {response.text if response else 'No response body'}"
                print(error_message)
            except Exception as e:
                # Catch any other unexpected errors
                error_message = f"An unhandled error occurred with {current_model}: {e}"
                print(error_message)


            # If an error occurred and a fallback model is defined and we haven't tried it yet
            if fallback_model and current_model == model:
                print(f"Primary model {model} failed. Attempting fallback to {fallback_model}...")
                current_model = fallback_model
                time.sleep(1) # Small delay before retrying with fallback
            else:
                # No fallback, or fallback already tried
                return f"Failed to generate text after trying {model} and potentially {fallback_model}."
        return "An unhandled error occurred after all attempts." # Should ideally not be reached
