from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import os
import asyncio
import base64
from dotenv import load_dotenv
from datetime import datetime
import logging
import traceback

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
        self.sessions = {}
        self.google_ai_available = False
        self.elevenlabs_available = False
        
        # Test Google AI
        try:
            import google.generativeai as genai
            if os.getenv('GOOGLE_API_KEY'):
                genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
                self.model = genai.GenerativeModel('gemini-2.5-flash')
                self.google_ai_available = True
                logger.info("✅ Google AI configured successfully")
            else:
                logger.error("❌ No GOOGLE_API_KEY found")
        except Exception as e:
            logger.error(f"❌ Google AI setup failed: {e}")
            
        # Test ElevenLabs
        try:
            if os.getenv('ELEVENLABS_API_KEY'):
                self.elevenlabs_available = True
                logger.info("✅ ElevenLabs API key found")
            else:
                logger.error("❌ No ELEVENLABS_API_KEY found")
        except Exception as e:
            logger.error(f"❌ ElevenLabs setup failed: {e}")

    def get_fallback_response(self, text):
        """Generate fallback response when AI is unavailable"""
        text_lower = text.lower()
        
        # Emergency responses
        if any(word in text_lower for word in ['emergency', 'urgent', 'not working', 'broken', 'cold']):
            return "I understand this is urgent! Let me get you help right away. Please call us at 512-567-6370 for immediate emergency service. We'll have someone out to you today."
        
        # Routine service
        elif any(word in text_lower for word in ['maintenance', 'schedule', 'check', 'service']):
            return "I'd be happy to help schedule your service! Please call us at 512-567-6370 or visit our website. We're available 9 AM - 5 PM, Monday through Sunday."
        
        # HVAC related
        elif any(word in text_lower for word in ['heat', 'ac', 'air', 'furnace', 'hvac']):
            return "For HVAC services including heating and cooling, please call Jason at 512-567-6370. With 30+ years of experience, we'll get your system running perfectly!"
        
        # Plumbing related  
        elif any(word in text_lower for word in ['water', 'leak', 'pipe', 'plumbing', 'drain']):
            return "For plumbing services including leaks and repairs, please call us at 512-567-6370. We handle all plumbing needs in the Post Falls area!"
        
        # Default response
        else:
            return "Hi! I'm Sarah from Hardison Heat, Air & Plumbing. For all your HVAC and plumbing needs, please call us at 512-567-6370. We're here 9 AM - 5 PM, Monday through Sunday!"

    async def get_ai_response(self, text, session_id):
        """Get AI response with comprehensive error handling"""
        try:
            if not self.google_ai_available:
                logger.info("Using fallback response (Google AI not available)")
                return self.get_fallback_response(text)
            
            # Initialize session
            if session_id not in self.sessions:
                system_prompt = f"""You are Sarah, the friendly AI assistant for Hardison Heat - Air - Plumbing in Post Falls, Idaho.

BUSINESS INFO:
- Owner: Jason Hardison (30+ years experience)  
- Phone: 512-567-6370
- Hours: 9 AM - 5 PM, Monday through Sunday
- Services: HVAC, Plumbing, Emergency Repairs

YOUR ROLE:
- Be warm, professional, and empathetic
- Keep responses under 50 words
- Help schedule services and assess urgency
- Always provide phone number for immediate help

EMERGENCY INDICATORS: No heat/AC, water leaks, gas smells, frozen pipes

Current season: {datetime.now().strftime('%B')}"""

                self.sessions[session_id] = [
                    {"role": "user", "parts": [f"System: {system_prompt}"]},
                    {"role": "model", "parts": ["I'm ready to help as Sarah from Hardison Heat-Air-Plumbing."]}
                ]
            
            # Add user message
            self.sessions[session_id].append({"role": "user", "parts": [text]})
            
            # Generate response with timeout
            chat = self.model.start_chat(history=self.sessions[session_id][:-1])
            response = await asyncio.wait_for(
                chat.send_message_async(text), 
                timeout=10.0  # 10 second timeout
            )
            
            response_text = response.text.strip()
            
            # Add to history
            self.sessions[session_id].append({"role": "model", "parts": [response_text]})
            
            # Manage memory
            if len(self.sessions[session_id]) > 20:
                self.sessions[session_id] = self.sessions[session_id][:2] + self.sessions[session_id][-10:]
            
            logger.info(f"✅ AI response generated: {response_text[:50]}...")
            return response_text
            
        except asyncio.TimeoutError:
            logger.error("AI response timeout")
            return self.get_fallback_response(text)
        except Exception as e:
            logger.error(f"AI response error: {e}")
            logger.error(traceback.format_exc())
            return self.get_fallback_response(text)

    def generate_speech(self, text):
        """Generate speech with error handling"""
        try:
            if not self.elevenlabs_available:
                logger.info("No ElevenLabs available - text only response")
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
            
            logger.info("✅ Speech generated successfully")
            return base64.b64encode(audio).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Speech generation failed: {e}")
            return None

# Initialize agent
agent = HardisonVoiceAgent()

@app.route('/')
def index():
    return render_template('demo.html')

@app.route('/health')
def health():
    return {
        "status": "healthy", 
        "google_ai": agent.google_ai_available,
        "elevenlabs": agent.elevenlabs_available
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
    """Handle text messages with bulletproof error handling"""
    try:
        text = data.get('text', '').strip()
        session_id = data.get('session_id', 'default')
        
        if not text:
            emit('error', {'message': 'Please enter a message'})
            return
        
        logger.info(f"📨 Processing message: {text}")
        
        # Create new event loop for this request
        def run_async():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(agent.get_ai_response(text, session_id))
            finally:
                loop.close()
        
        # Get AI response
        ai_response = run_async()
        logger.info(f"✅ Generated response: {ai_response[:50]}...")
        
        # Generate speech (optional)
        audio_response = agent.generate_speech(ai_response)
        
        # Send response
        emit('ai_response', {
            'text': ai_response,
            'audio': audio_response
        })
        
        logger.info("✅ Response sent successfully")
        
    except Exception as e:
        logger.error(f"❌ Error in text_message handler: {e}")
        logger.error(traceback.format_exc())
        
        emit('error', {
            'message': 'Sorry, I had a technical issue. Please call us at 512-567-6370 for immediate help!'
        })

if __name__ == '__main__':
    logger.info("🚀 Starting Hardison Voice Agent (Debug Mode)")
    logger.info(f"Google AI Available: {agent.google_ai_available}")
    logger.info(f"ElevenLabs Available: {agent.elevenlabs_available}")
    
    port = int(os.environ.get('PORT', 5000))
    socketio.run(app, debug=False, host='0.0.0.0', port=port)
