import os
from google import genai
from google.genai import types

class Gemini3Client:
    def __init__(self, api_key: str = None):
        self.client = genai.Client(api_key=api_key or os.environ.get("GOOGLE_API_KEY"))
        self.model_id = "gemini-3-pro-preview"  #

    def generate_strategy_content(
        self, 
        prompt: str, 
        thinking_level: str = "high", 
        enable_search: bool = False,
        history: list = None
    ):
        """
        Executes an agent call with Thought Signature persistence.
        """
        # Configure Search Grounding if requested
        tools = [types.Tool(google_search=types.GoogleSearch())] if enable_search else []

        # Configure Thinking Depth (High for Critic/Strategist)
        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level=thinking_level),
            tools=tools,
            temperature=1.0  # Optimized for Gemini 3 reasoning
        )

        # Standard SDK chat handling automatically persists thought signatures
        chat = self.client.chats.create(model=self.model_id, history=history or [], config=config)
        response = chat.send_message(prompt)

        # Extract Thought Signature for audit/logging (optional)
        thought_sig = None
        if response.candidates[0].content.parts:
            # Signatures are typically on the first function call or last text part
            thought_sig = getattr(response.candidates[0].content.parts[0], 'thought_signature', None)

        return {
            "text": response.text,
            "signature": thought_sig,
            "grounding": getattr(response.candidates[0], 'grounding_metadata', None),
            "history": chat.history # Pass this back to preserve state across agents
        }