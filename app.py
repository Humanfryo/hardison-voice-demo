from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import os
import asyncio
import base64
import json
from elevenlabs import VoiceSettings, generate
import google.generativeai as genai
from dotenv import load_dotenv
from datetime import datetime
import logging

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hardison-voice-agent-secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HardisonVoiceAgent:
    def __init__(self):
        # Configure APIs
        try:
            genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
            self.model = genai.GenerativeModel('gemini-2.5-flash')
            logger.info("✅ Google AI configured successfully")
        except Exception as e:
            logger.error(f"❌ Google AI configuration failed: {e}")
            
        # Session management
        self.sessions = {}
        
    def get_system_prompt(self):
        """Enhanced system prompt for Hardison HVAC"""
        current_month = datetime.now().month
        season = 'winter' if current_month in [12, 1, 2] else 'summer' if current_month in [6, 7, 8] else 'spring/fall'
        
        return f"""You are Sarah, the friendly AI assistant for Hardison Heat - Air - Plumbing.

BUSINESS INFO:
- Owner: Jason Hardison (30+ years experience)
- Service Area: Post Falls, Idaho and surrounding areas
- Hours: 9 AM - 5 PM, Monday through Sunday
- Phone: 512-567-6370
- Current Season: {season}

YOUR PERSONALITY:
- Warm, professional, and empathetic
- Ask ONE question at a time
- Keep responses under 50 words
- Show empathy for emergencies
- Be efficient for routine requests

SERVICES:
1. HVAC (heating, cooling, AC repair/installation)
2. Plumbing (leaks, clogs, water heaters, pipes)
3. Emergency repairs (24/7 mindset)
4. Maintenance and inspections

CONVERSATION FLOW:
1. Greet warmly and identify the issue
2. Assess urgency (emergency vs routine)
3. Collect: name, phone, address, problem details
4. Suggest appointment times
5. Confirm details and next steps

EMERGENCY INDICATORS:
- No heat (winter) / No AC (summer)
- Water leaks or flooding
- Gas smells
- Frozen pipes
- Complete system failures

SAMPLE RESPONSES:
- Emergency: "Oh no! That sounds urgent. Let me get someone out to you today."
- Routine: "I'd be happy to schedule your maintenance. When works best?"
- Confirmation: "Perfect! I have you scheduled for [service] on [date] at [time]."

Remember: You represent a trusted local business with 30+ years of experience. Be professional, caring, and helpful."""

    async def get_ai_response(self, text, session_id):
        """Get response from Gemini with conversation context"""
        try:
            # Initialize session if new
            if session_id not in self.sessions:
                system_prompt = self.get_system_prompt()
                self.sessions[session_id] = [
                    {"role": "user", "parts": [f"System: {system_prompt}"]},
                    {"role": "model", "parts": ["I understand. I'm ready to help as Sarah from Hardison Heat-Air-Plumbing."]}
                ]
            
            # Add user message
            self.sessions[session_id].append({
                "role": "user", 
                "parts": [text]
            })
            
            # Generate response
            chat = self.model.start_chat(history=self.sessions[session_id][:-1])
            response = await chat.send_message_async(text)
            
            response_text = response.text.strip()
            
            # Add response to history
            self.sessions[session_id].append({
                "role": "model",
                "parts": [response_text]
            })
            
            # Keep conversation manageable
            if len(self.sessions[session_id]) > 20:
                system_msg = self.sessions[session_id][:2]
                recent_msgs = self.sessions[session_id][-10:]
                self.sessions[session_id] = system_msg + recent_msgs
            
            logger.info(f"Generated response: {response_text}")
            return response_text
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize for the technical difficulty. Please call us directly at 512-567-6370."

    def generate_speech(self, text):
        """Generate speech using ElevenLabs"""
        try:
            if not os.getenv('ELEVENLABS_API_KEY'):
                logger.warning("No ElevenLabs API key found")
                return None
                
            audio = generate(
                text=text,
                voice="Rachel",
                api_key=os.getenv('ELEVENLABS_API_KEY'),
                model="eleven_turbo_v2",
                voice_settings=VoiceSettings(
                    stability=0.75,
                    similarity_boost=0.75,
                    style=0.0,
                    use_speaker_boost=True
                )
            )
            
            return base64.b64encode(audio).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Speech generation error: {e}")
            return None

# Initialize the agent
agent = HardisonVoiceAgent()

@app.route('/')
def index():
    """Main demo page"""
    return render_template('demo.html')

@app.route('/health')
def health():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Hardison Voice Agent"}

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info("Client connected")
    emit('status', {'message': 'Connected to Sarah - Hardison Voice Agent'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info("Client disconnected")

@socketio.on('text_message')
def handle_text_message(data):
    """Handle text input from browser"""
    try:
        text = data['text']
        session_id = data.get('session_id', 'default')
        
        logger.info(f"Received message: {text}")
        
        # Process with AI
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        ai_response = loop.run_until_complete(
            agent.get_ai_response(text, session_id)
        )
        
        # Generate speech
        audio_response = agent.generate_speech(ai_response)
        
        # Send response
        emit('ai_response', {
            'text': ai_response,
            'audio': audio_response
        })
        
    except Exception as e:
        logger.error(f"Error handling message: {e}")
        emit('error', {'message': 'Sorry, I had a technical issue. Please try again or call 512-567-6370.'})

if __name__ == '__main__':
    logger.info("🚀 Starting Hardison Voice Agent...")
    
    # Check API keys
    if not os.getenv('GOOGLE_API_KEY'):
        logger.error("❌ Missing GOOGLE_API_KEY")
    if not os.getenv('ELEVENLABS_API_KEY'):
        logger.error("❌ Missing ELEVENLABS_API_KEY")
    
    # Get port from environment (Railway sets this)
    port = int(os.environ.get('PORT', 5000))
    socketio.run(app, debug=False, host='0.0.0.0', port=port)
