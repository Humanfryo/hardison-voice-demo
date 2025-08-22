import google.generativeai as genai
import os
from typing import Optional, AsyncGenerator
from vocode.streaming.agent.base_agent import BaseAgent, RespondAgent
from vocode.streaming.models.agent import AgentConfig, AgentType
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.utils import create_conversation_id

class GeminiAgentConfig(AgentConfig, type=AgentType.CHAT_GPT):  # Use existing type for compatibility
    model_name: str = "gemini-2.5-flash"
    temperature: float = 0.7
    max_tokens: int = 1000

class GeminiAgent(RespondAgent[GeminiAgentConfig]):
    def __init__(self, agent_config: GeminiAgentConfig):
        super().__init__(agent_config=agent_config)
        
        # Configure Gemini
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        self.model = genai.GenerativeModel(
            model_name=agent_config.model_name,
            generation_config=genai.types.GenerationConfig(
                temperature=agent_config.temperature,
                max_output_tokens=agent_config.max_tokens,
            )
        )
        
        # Initialize conversation history
        self.conversation_history = []
        
        # Add system prompt if provided
        if hasattr(agent_config, 'prompt_preamble') and agent_config.prompt_preamble:
            self.conversation_history.append({
                'role': 'user',
                'parts': [f"System: {agent_config.prompt_preamble}"]
            })
            self.conversation_history.append({
                'role': 'model', 
                'parts': ["I understand. I'm ready to help as Sarah from Hardison Heat - Air - Plumbing."]
            })

    async def respond(
        self,
        human_input: str,
        conversation_id: str,
        is_interrupt: bool = False,
    ) -> tuple[str, bool]:
        """Generate response using Google Gemini"""
        
        try:
            # Add human input to conversation history
            self.conversation_history.append({
                'role': 'user',
                'parts': [human_input]
            })
            
            # Generate response using Gemini
            chat = self.model.start_chat(history=self.conversation_history[:-1])
            response = await chat.send_message_async(human_input)
            
            # Extract response text
            response_text = response.text.strip()
            
            # Add response to conversation history
            self.conversation_history.append({
                'role': 'model',
                'parts': [response_text]
            })
            
            # Keep conversation history manageable (last 20 exchanges)
            if len(self.conversation_history) > 40:
                # Keep system prompt and last 20 exchanges
                system_messages = self.conversation_history[:2]
                recent_messages = self.conversation_history[-20:]
                self.conversation_history = system_messages + recent_messages
            
            return response_text, False
            
        except Exception as e:
            print(f"Error generating Gemini response: {e}")
            fallback_response = "I apologize, I'm having technical difficulties. Please call us directly at 512-677-5146 and we'll be happy to help you."
            return fallback_response, False

    def update_last_bot_message_on_cut_off(self, message: str):
        """Handle message cut-off scenarios"""
        if self.conversation_history and self.conversation_history[-1]['role'] == 'model':
            self.conversation_history[-1]['parts'] = [message]
