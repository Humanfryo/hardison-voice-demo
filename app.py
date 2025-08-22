from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import os
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hardison-voice-agent-secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleHardisonAgent:
    def __init__(self):
        self.google_ai_available = False
        self.elevenlabs_available = False
        
        # Check Google AI
        try:
            import google.generativeai as genai
            if os.getenv('GOOGLE_API_KEY'):
                genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
                self.model = genai.GenerativeModel('gemini-2.5-flash')
                # Test with a simple request
                test_chat = self.model.start_chat()
                test_response = test_chat.send_message("Hello")
                self.google_ai_available = True
                logger.info("✅ Google AI working")
            else:
                logger.error("❌ No Google API key")
        except Exception as e:
            logger.error(f"❌ Google AI failed: {e}")
        
        # Check ElevenLabs
        if os.getenv('ELEVENLABS_API_KEY'):
            self.elevenlabs_available = True
            logger.info("✅ ElevenLabs key found")
        else:
            logger.error("❌ No ElevenLabs key")

    def get_response(self, text):
        """Get response - simple synchronous version"""
        if not self.google_ai_available:
            return self.get_fallback_response(text)
        
        try:
            # Simple synchronous chat
            chat = self.model.start_chat()
            response = chat.send_message(f"""You are Sarah from Hardison Heat-Air-Plumbing in Post Falls, Idaho (phone: 512-567-6370). 
            
Respond to this customer message in under 50 words with warmth and professionalism: {text}""")
            
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"AI error: {e}")
            return self.get_fallback_response(text)
    
    def get_fallback_response(self, text):
        """Smart fallback responses"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['emergency', 'urgent', 'not working', 'broken', 'cold', 'hot']):
            return "I understand this is urgent! Please call us immediately at 512-567-6370 for emergency service. We'll get someone out to you right away."
        
        elif any(word in text_lower for word in ['maintenance', 'schedule', 'check']):
            return "I'd be happy to help schedule your maintenance! Please call us at 512-567-6370. We're available 9 AM - 5 PM, Monday through Sunday."
        
        elif any(word in text_lower for word in ['heat', 'ac', 'air', 'furnace', 'hvac']):
            return "For HVAC services, call Jason at 512-567-6370. With 30+ years experience in Post Falls, we'll get your system running perfectly!"
        
        elif any(word in text_lower for word in ['water', 'leak', 'pipe', 'plumbing']):
            return "For plumbing services including leaks and repairs, call us at 512-567-6370. We handle all plumbing needs in Post Falls!"
        
        else:
            return "Hi! I'm Sarah from Hardison Heat, Air & Plumbing. For all your HVAC and plumbing needs, call us at 512-567-6370. We're here 9 AM - 5 PM, Monday-Sunday!"

# Initialize agent
agent = SimpleHardisonAgent()

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
    emit('status', {'message': 'Connected to Sarah'})

@socketio.on('text_message')
def handle_text_message(data):
    """Simple text message handler"""
    try:
        text = data.get('text', '').strip()
        
        if not text:
            emit('error', {'message': 'Please enter a message'})
            return
        
        logger.info(f"📨 Message: {text}")
        
        # Get response (synchronous - no async issues)
        response = agent.get_response(text)
        logger.info(f"✅ Response: {response[:50]}...")
        
        # Send back immediately
        emit('ai_response', {
            'text': response,
            'audio': None  # Skip audio for now
        })
        
    except Exception as e:
        logger.error(f"❌ Handler error: {e}")
        emit('error', {
            'message': 'Sorry, please call us at 512-567-6370 for immediate help!'
        })

if __name__ == '__main__':
    logger.info("🚀 Starting Minimal Hardison Agent")
    logger.info(f"Google AI: {agent.google_ai_available}")
    logger.info(f"ElevenLabs: {agent.elevenlabs_available}")
    
    port = int(os.environ.get('PORT', 5000))
    socketio.run(app, debug=False, host='0.0.0.0', port=port)
