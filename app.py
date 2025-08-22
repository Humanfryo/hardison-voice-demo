from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import os
import openai
from dotenv import load_dotenv
import logging
from datetime import datetime

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hardison-voice-agent-secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenAIHardisonAgent:
    def __init__(self):
        self.openai_available = False
        self.conversations = {}  # Store conversation history by session
        
        # Initialize OpenAI
        try:
            self.client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            
            # Test the connection
            test_response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=10
            )
            
            self.openai_available = True
            logger.info("✅ OpenAI GPT-4o Mini connected successfully")
            
        except Exception as e:
            logger.error(f"❌ OpenAI setup failed: {e}")
            logger.info("Will use fallback responses")

    def get_system_prompt(self):
        """System prompt for Sarah, the HVAC assistant"""
        current_month = datetime.now().strftime("%B")
        
        return f"""You are Sarah, the professional and friendly AI assistant for Hardison Heat - Air - Plumbing.

BUSINESS INFORMATION:
- Company: Hardison Heat - Air - Plumbing
- Owner: Jason Hardison (30+ years of experience)
- Location: Post Falls, Idaho and surrounding areas
- Phone: 512-567-6370
- Hours: 9 AM to 5 PM, Monday through Sunday
- Current Month: {current_month}

SERVICES PROVIDED:
- HVAC: Heating, cooling, air conditioning, furnaces, heat pumps
- Plumbing: Leaks, repairs, water heaters, pipes, drains
- Emergency repairs (prioritize urgent situations)
- Maintenance and inspections
- New installations and replacements

YOUR PERSONALITY:
- Warm, professional, and empathetic
- Knowledgeable about HVAC and plumbing
- Patient with customers
- Always helpful and solution-oriented
- Keep responses conversational and under 60 words

CONVERSATION APPROACH:
1. Identify the customer's issue and urgency level
2. Show appropriate empathy (especially for emergencies)
3. Provide helpful next steps
4. Always include phone number for immediate help
5. Ask relevant follow-up questions when appropriate

EMERGENCY INDICATORS (immediate attention needed):
- No heat in winter / No AC in hot weather
- Water leaks, flooding, or burst pipes
- Gas smells or suspected gas leaks
- Complete system failures
- Frozen pipes

RESPONSE EXAMPLES:
Emergency: "Oh no, that sounds urgent! Please call us right away at 512-567-6370. We can often get someone out the same day for emergencies like this."

Routine: "I'd be happy to help you schedule that service! Please call us at 512-567-6370. We're available 9 AM - 5 PM, Monday through Sunday."

Remember: You represent a trusted local business with 30+ years of experience. Be professional, caring, and always provide clear next steps."""

    def get_response(self, text, session_id="default"):
        """Get response from OpenAI GPT-4o mini"""
        
        if not self.openai_available:
            return self.get_fallback_response(text)
        
        try:
            # Initialize conversation history for new sessions
            if session_id not in self.conversations:
                self.conversations[session_id] = [
                    {"role": "system", "content": self.get_system_prompt()},
                    {"role": "assistant", "content": "Hi! I'm Sarah from Hardison Heat, Air, and Plumbing. How can I help you today?"}
                ]
            
            # Add user message to conversation
            self.conversations[session_id].append({"role": "user", "content": text})
            
            # Get response from OpenAI
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=self.conversations[session_id],
                max_tokens=150,
                temperature=0.7,
                timeout=10  # 10 second timeout
            )
            
            ai_response = response.choices[0].message.content.strip()
            
            # Add AI response to conversation history
            self.conversations[session_id].append({"role": "assistant", "content": ai_response})
            
            # Keep conversation history manageable (last 10 exchanges)
            if len(self.conversations[session_id]) > 22:  # system + 10 exchanges
                system_msg = self.conversations[session_id][0]
                recent_msgs = self.conversations[session_id][-20:]
                self.conversations[session_id] = [system_msg] + recent_msgs
            
            logger.info(f"✅ OpenAI response: {ai_response[:50]}...")
            return ai_response
            
        except Exception as e:
            logger.error(f"❌ OpenAI error: {e}")
            return self.get_fallback_response(text)

    def get_fallback_response(self, text):
        """Smart fallback responses when OpenAI is unavailable"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['hello', 'hi', 'hey']):
            return "Hello! I'm Sarah from Hardison Heat, Air & Plumbing. How can I help you today? For immediate service, call us at 512-567-6370."
        
        elif any(word in text_lower for word in ['emergency', 'urgent', 'not working', 'broken', 'cold']):
            return "This sounds urgent! Please call us immediately at 512-567-6370 for emergency service. We prioritize urgent repairs and can often provide same-day service."
        
        elif any(word in text_lower for word in ['heat', 'heater', 'furnace', 'cold']):
            return "For heating issues, please call us right away at 512-567-6370. With 30+ years of experience, Jason and our team will get your heating system running properly."
        
        elif any(word in text_lower for word in ['ac', 'air conditioning', 'cooling', 'hot']):
            return "For air conditioning and cooling issues, call us at 512-567-6370. We handle all AC repairs, maintenance, and installations in Post Falls."
        
        elif any(word in text_lower for word in ['leak', 'water', 'plumbing', 'pipe']):
            return "For plumbing issues including leaks and repairs, call us at 512-567-6370. We handle all plumbing needs and emergency repairs."
        
        elif any(word in text_lower for word in ['maintenance', 'service', 'check']):
            return "Regular maintenance is important! Call us at 512-567-6370 to schedule service. We're open 9 AM - 5 PM, Monday through Sunday."
        
        else:
            return "I'm Sarah from Hardison Heat, Air & Plumbing. For all HVAC and plumbing needs, call us at 512-567-6370. We have 30+ years of experience serving Post Falls!"

    def generate_speech(self, text):
        """Generate speech using ElevenLabs (optional)"""
        try:
            if not os.getenv('ELEVENLABS_API_KEY'):
                return None
                
            from elevenlabs import VoiceSettings, generate
            
            audio = generate(
                text=text,
                voice="Rachel",
                api_key=os.getenv('ELEVENLABS_API_KEY'),
                model="eleven_turbo_v2",
                voice_settings=VoiceSettings(
                    stability=0.75,
                    similarity_boost=0.75
                )
            )
            
            import base64
            return base64.b64encode(audio).decode('utf-8')
            
        except Exception as e:
            logger.warning(f"Speech generation failed: {e}")
            return None

# Initialize agent
agent = OpenAIHardisonAgent()

@app.route('/')
def index():
    return render_template('demo.html')

@app.route('/health')
def health():
    return {
        "status": "healthy",
        "openai": agent.openai_available,
        "model": "gpt-4o-mini" if agent.openai_available else "fallback"
    }

@socketio.on('connect')
def handle_connect():
    logger.info("✅ Client connected")
    emit('status', {'message': 'Connected to Sarah - Hardison Voice Agent'})

@socketio.on('disconnect') 
def handle_disconnect():
    logger.info("Client disconnected")

@socketio.on('text_message')
def handle_text_message(data):
    """Handle text messages with OpenAI"""
    try:
        text = data.get('text', '').strip()
        session_id = data.get('session_id', 'default')
        
        if not text:
            emit('error', {'message': 'Please enter a message'})
            return
        
        logger.info(f"📨 Processing: {text}")
        
        # Get response from OpenAI or fallback
        response = agent.get_response(text, session_id)
        
        # Generate speech (optional)
        audio_response = agent.generate_speech(response)
        
        # Send response
        emit('ai_response', {
            'text': response,
            'audio': audio_response
        })
        
        logger.info(f"✅ Response sent successfully")
        
    except Exception as e:
        logger.error(f"❌ Handler error: {e}")
        emit('error', {
            'message': 'Sorry, please call us at 512-567-6370 for immediate help!'
        })

if __name__ == '__main__':
    logger.info("🚀 Starting OpenAI Hardison Voice Agent")
    logger.info(f"OpenAI Available: {agent.openai_available}")
    
    port = int(os.environ.get('PORT', 5000))
    socketio.run(app, debug=False, host='0.0.0.0', port=port)
