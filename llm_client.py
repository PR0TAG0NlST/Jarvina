import os
import google.generativeai as genai

class GeminiClient:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY") # Corrected variable name
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        
        # Configure the client with the API key
        genai.configure(api_key=self.api_key)
        
        # Instantiate the correct model with a valid name
        self.model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest")

    def generate_content(self, messages: list, temperature: float = 0.7) -> str:
        """
        Generates content from the Gemini model using the provided messages.
        """
        try:
            # The generate_content method can take a list of message dictionaries
            # This is the correct way to pass a chat history to the model
            response = self.model.generate_content(
                contents=messages,
                generation_config=genai.types.GenerationConfig(temperature=temperature)
            )
            
            # Extract and return the text from the response
            return response.text

        except Exception as e:
            # Log the full exception for debugging
            print(f"An error occurred during content generation: {e}")
            raise RuntimeError(f"AI generation failed: {str(e)}")
