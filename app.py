from flask import Flask, render_template, request, jsonify
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
socketio = SocketIO(app, cors_allowed_origins="*", logger=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductionHardisonVoiceAgent:
    def __init__(self):
        # Configure APIs
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        self.model = genai.GenerativeModel(
            'gemini-2.5-flash',
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=500,
                candidate_count=1,
            )
        )
        
        # Enhanced business context
        self.business_context = {
            'name': 'Hardison Heat - Air - Plumbing',
            'owner': 'Jason',
            'location': 'Post Falls, Idaho',
            'hours': '9 AM to 5 PM, Monday through Sunday',
            'phone': '512-567-6370',
            'specialties': {
                'hvac': ['heating', 'cooling', 'air conditioning', 'furnace', 'heat pump'],
                'plumbing': ['leaks', 'clogs', 'water heater', 'pipes', 'fixtures'],
                'emergency': ['no heat', 'no ac', 'water leak', 'gas smell', 'frozen pipes']
            }
        }
        
        # Session management for natural conversations
        self.sessions = {}
        self.conversation_memory = {}
        
    def get_enhanced_system_prompt(self):
        """Enhanced system prompt for natural conversations"""
        current_season = self.get_current_season()
        
        return f"""You are Sarah, the professional AI assistant for Hardison Heat - Air - Plumbing.

CORE IDENTITY:
- Warm, empathetic, and genuinely helpful
- 30+ years of HVAC/plumbing expertise through Jason's experience
- Local Post Falls, Idaho expert who understands regional climate needs
- Professional but conversational - like talking to a trusted neighbor

BUSINESS EXPERTISE:
- Owner: Jason Hardison (30+ years experience)
- Service Area: Post Falls, Idaho and surrounding areas
- Hours: 9 AM - 5 PM, Monday through Sunday  
- Phone: 512-567-6370
- Current Season: {current_season} (prioritize {self.get_seasonal_priorities(current_season)})

CONVERSATION STYLE:
- Listen actively and acknowledge customer emotions
- Ask clarifying questions naturally (one at a time)
- Use simple, clear language - avoid technical jargon unless needed
- Show empathy for emergency situations
- Be efficient but thorough for routine requests
- Remember previous conversation context

EMERGENCY DETECTION:
- No heat (winter priority) / No AC (summer priority)  
- Water leaks or flooding (immediate response)
- Gas smells (safety priority)
- Frozen pipes (winter concern)
- Complete system failures

CONVERSATION FLOW:
1. Greet warmly and identify the core issue
2. Assess urgency level and respond appropriately
3. Gather essential info: name, phone, address, problem details
4. Provide realistic timeframes and next steps
5. Confirm appointment details clearly
6. End with reassurance and clear expectations

RESPONSE GUIDELINES:
- Keep responses conversational and under 50 words
- Use natural transitions and acknowledgments
- Remember customer details throughout conversation
- Provide specific, actionable next steps
- Always end with confidence and reassurance

Remember: You represent a trusted local business with deep roots in the community. Every interaction should reflect professionalism, expertise, and genuine care for customer needs."""

    def get_current_season(self):
        """Determine current season for contextual responses"""
        month = datetime.now().month
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [6, 7, 8]:  
            return 'summer'
        else:
            return 'spring_fall'
    
    def get_seasonal_priorities(self, season):
        """Get seasonal emergency priorities"""
        priorities = {
            'winter': 'heating emergencies, frozen pipes, furnace issues',
            'summer': 'AC failures, cooling system problems', 
            'spring_fall': 'maintenance, water leaks, general repairs'
        }
        return priorities.get(season, 'general service needs')

    def initialize_conversation(self, session_id):
        """Initialize conversation with enhanced context"""
        if session_id not in self.sessions:
            system_prompt = self.get_enhanced_system_prompt()
            
            self.sessions[session_id] = [
                {"role": "user", "parts": [f"System: {system_prompt}"]},
                {"role": "model", "parts": ["I understand. I'm ready to help as Sarah from Hardison Heat, Air, and Plumbing. I'll provide warm, professional service with our 30+ years of expertise."]}
            ]
            
            self.conversation_memory[session_id] = {
                'customer_name': None,
                'phone': None,
                'address': None,
                'service_type': None,
                'urgency': None,
                'appointment_preferences': []
            }
        
        return self.sessions[session_id]

    async def process_conversation(self, text, session_id):
        """Enhanced conversation processing with context management"""
        try:
            logger.info(f"Processing message for session {session_id}: {text}")
            
            # Initialize conversation
            conversation_history = self.initialize_conversation(session_id)
            
            # Add user message
            conversation_history.append({
                "role": "user", 
                "parts": [text]
            })
            
            # Detect and store conversation elements
            self.extract_conversation_context(text, session_id)
            
            # Generate contextual response
            chat = self.model.start_chat(history=conversation_history[:-1])
            response = await chat.send_message_async(text)
            
            response_text = response.text.strip()
            
            # Add response to history
            conversation_history.append({
                "role": "model",
                "parts": [response_text]
            })
            
            # Manage conversation length
            self.manage_conversation_memory(session_id)
            
            logger.info(f"Generated response: {response_text}")
            return response_text
            
        except Exception as e:
            logger.error(f"Error in conversation processing: {e}")
            return "I apologize for the technical difficulty. Please call us directly at 512-567-6370 for immediate assistance."

    def extract_conversation_context(self, text, session_id):
        """Extract and store conversation context for personalization"""
        text_lower = text.lower()
        memory = self.conversation_memory[session_id]
        
        # Extract service type
        for service_type, keywords in self.business_context['specialties'].items():
            if any(keyword in text_lower for keyword in keywords):
                memory['service_type'] = service_type
                break
        
        # Detect urgency indicators
        emergency_words = ['emergency', 'urgent', 'immediately', 'asap', 'right now', 'not working', 'broken', 'flooding']
        if any(word in text_lower for word in emergency_words):
            memory['urgency'] = 'high'
        elif any(word in text_lower for word in ['maintenance', 'schedule', 'routine', 'check']):
            memory['urgency'] = 'routine'

    def manage_conversation_memory(self, session_id):
        """Keep conversation history manageable while preserving context"""
        conversation = self.sessions[session_id]
        
        if len(conversation) > 25:  # Increased from 20 for better context
            # Keep system prompt and recent exchanges
            system_messages = conversation[:2]
            recent_messages = conversation[-15:]  # Keep more recent context
            self.sessions[session_id] = system_messages + recent_messages

    def generate_speech_optimized(self, text):
        """Optimized speech generation for natural conversations"""
        try:
            logger.info(f"Generating speech for: {text[:50]}...")
            
            audio = generate(
                text=text,
                voice="Rachel",  # Professional, warm female voice
                api_key=os.getenv('ELEVENLABS_API_KEY'),
                model="eleven_turbo_v2",  # Fastest model for real-time conversation
                voice_settings=VoiceSettings(
                    stability=0.8,  # Slightly higher for consistency
                    similarity_boost=0.75,
                    style=0.1,  # Slight style for warmth
                    use_speaker_boost=True
                )
            )
            
            return base64.b64encode(audio).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Speech generation error: {e}")
            return None

# Initialize the enhanced agent
agent = ProductionHardisonVoiceAgent()

@app.route('/')
def index():
    """Main demo page"""
    return render_template('demo.html')

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "Hardison Production Voice Agent"})

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
    """Handle text input from browser (STT will be handled client-side)"""
    try:
        text = data['text']
        session_id = data.get('session_id', 'default')
        
        logger.info(f"Received text message: {text}")
        
        # Process with enhanced conversation management
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        ai_response = loop.run_until_complete(
            agent.process_conversation(text, session_id)
        )
        
        # Generate speech
        audio_response = agent.generate_speech_optimized(ai_response)
        
        # Send comprehensive response
        emit('ai_response', {
            'text': ai_response,
            'audio': audio_response,
            'session_context': agent.conversation_memory.get(session_id, {})
        })
        
    except Exception as e:
        logger.error(f"Error handling text message: {e}")
        emit('error', {'message': 'Sorry, I had a technical issue. Please try again or call 512-567-6370.'})

# Legacy audio handler (will be replaced by client-side STT)
@socketio.on('audio_data')
def handle_audio(data):
    """Legacy audio handler - recommend client-side STT instead"""
    emit('error', {'message': 'Please use text input or enable browser speech recognition'})

if __name__ == '__main__':
    logger.info("🚀 Starting Hardison Production Voice Agent...")
    logger.info("🌐 Optimized for natural conversation and reliability")
    
    # Verify API keys
    required_keys = ['GOOGLE_API_KEY', 'ELEVENLABS_API_KEY']
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    
    if missing_keys:
        logger.error(f"❌ Missing API keys: {', '.join(missing_keys)}")
    else:
        socketio.run(app, debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
