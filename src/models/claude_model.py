"""
🌙 Moon Dev's Claude Model Implementation
Built with love by Moon Dev 🚀
"""

from anthropic import Anthropic
from termcolor import cprint
from .base_model import BaseModel, ModelResponse

class ClaudeModel(BaseModel):
    """Implementation for Anthropic's Claude models"""
    
    AVAILABLE_MODELS = {
        # Claude 4 Series (Latest Generation)
        "claude-opus-4-6": "Claude Opus 4.6 - Most powerful model",
        "claude-sonnet-4-6": "Claude Sonnet 4.6 - Latest balanced model",
        "claude-sonnet-4-5": "Claude Sonnet 4.5 - Strong reasoning, great for trading",
        "claude-haiku-4-5-20251001": "Claude Haiku 4.5 - Fast and efficient",

        # Legacy (deprecated, kept for compatibility)
        "claude-3-5-sonnet-latest": "Claude 3.5 Sonnet (deprecated)",
        "claude-3-5-haiku-latest": "Claude 3.5 Haiku (deprecated)",
    }

    def __init__(self, api_key: str, model_name: str = "claude-sonnet-4-6", **kwargs):
        self.model_name = model_name
        super().__init__(api_key, **kwargs)
    
    def initialize_client(self, **kwargs) -> None:
        """Initialize the Anthropic client"""
        try:
            self.client = Anthropic(api_key=self.api_key)
            cprint(f"✨ Initialized Claude model: {self.model_name}", "green")
        except Exception as e:
            cprint(f"❌ Failed to initialize Claude model: {str(e)}", "red")
            self.client = None
    
    def generate_response(self, 
        system_prompt: str,
        user_content: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> ModelResponse:
        """Generate a response using Claude"""
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_content}
                ]
            )
            
            return ModelResponse(
                content=response.content[0].text.strip(),
                raw_response=response,
                model_name=self.model_name,
                usage={"completion_tokens": response.usage.output_tokens}
            )
            
        except Exception as e:
            cprint(f"❌ Claude generation error: {str(e)}", "red")
            raise
    
    def is_available(self) -> bool:
        """Check if Claude is available"""
        return self.client is not None
    
    @property
    def model_type(self) -> str:
        return "claude" 