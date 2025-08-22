from flask import Flask, render_template, request, jsonify, send_file
from flask_socketio import SocketIO, emit
import os
import asyncio
import base64
import io
import wave
import json
from elevenlabs import ElevenLabs, VoiceSettings
import google.generativeai as genai
from dotenv import load_dotenv
from datetime import datetime
import speech_recognition as sr
import tempfile
import threading

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hardison-voice-agent-secret'
socketio = SocketIO(app, cors_allowed_origins="*")

class WebHardisonVoiceAgent:
    def __init__(self):
        # Configure APIs
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Initialize ElevenLabs client
        self.elevenlabs_client = ElevenLabs(api_key=os.getenv('ELEVENLABS_API_KEY'))
        
        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        
        # Business information
        self.business_info = {
            'name': 'Hardison Heat - Air - Plumbing',
            'owner': 'Jason',
            'location': 'Post Falls, Idaho',
            'hours': '9 AM to 5 PM, Monday through Sunday',
            'phone': '512-677-5146'
        }
        
        # Store conversation sessions
        self.sessions = {}
        
    def setup_system_prompt(self):
        """Set up the AI agent's personality and knowledge"""
        current_season = self.get_current_season()
        
        system_prompt = f"""
You are Sarah, the friendly booking assistant for Hardison Heat - Air - Plumbing in Post Falls, Idaho.

COMPANY INFO:
- Owner: Jason Hardison, 30+ years experience
- Service Area: Post Falls, Idaho and surrounding areas  
- Hours: 9 AM to 5 PM, Monday through Sunday
- Phone: 512-677-5146
- We offer financing options

CURRENT SEASON: {current_season}
PRIORITY EMERGENCIES: {"heating" if current_season == "winter" else "AC" if current_season == "summer" else "plumbing"}

SERVICES:
1. HVAC Service & Repair (heating, cooling, AC)
2. Plumbing Service & Repair (leaks, clogs, installations)  
3. New Equipment Installations
4. Preventive Maintenance
5. Emergency Repairs

YOUR PERSONALITY:
- Warm, professional, and helpful
- Ask ONE question at a time
- Use simple, clear language
- Be empathetic for emergencies
- Efficient for busy customers

CONVERSATION FLOW:
1. Greet and ask how you can help
2. Identify service needed and urgency
3. Collect: name, phone, address, problem description
4. Suggest appointment times
5. Confirm details
6. Provide next steps

EMERGENCY INDICATORS:
- No heat in winter / No AC in summer
- Water leaks or flooding
- Gas smell
- Frozen pipes
- No hot water

SAMPLE RESPONSES:
- Emergency: "Oh no! That sounds urgent. Let me get someone out to you today."
- Routine: "I'd be happy to schedule your maintenance. What day works best?"
- Confirmation: "Perfect! I have you scheduled for [service] on [date] at [time]."

Keep responses under 50 words. Be conversational, not robotic.
"""
        
        return [
            {"role": "user", "parts": [f"System: {system_prompt}"]},
            {"role": "model", "parts": ["I understand. I'm ready to help as Sarah from Hardison Heat-Air-Plumbing."]}
        ]
    
    def get_current_season(self):
        """Determine current season for prioritizing emergencies"""
        month = datetime.now().month
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [6, 7, 8]:  
            return 'summer'
        else:
            return 'spring_fall'
    
    def process_audio(self, audio_data, session_id):
        """Process audio data and return transcription"""
        try:
            # Convert base64 audio to wav file
            audio_bytes = base64.b64decode(audio_data)
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_file.write(audio_bytes)
                temp_file.flush()
                
                # Use speech recognition
                with sr.AudioFile(temp_file.name) as source:
                    audio = self.recognizer.record(source)
                    text = self.recognizer.recognize_google(audio)
                    
                # Clean up temp file
                os.unlink(temp_file.name)
                
                return text
                
        except Exception as e:
            print(f"Error processing audio: {e}")
            return None
    
    async def get_ai_response(self, user_input, session_id):
        """Get response from Gemini 2.5 Flash"""
        try:
            # Get or create session
            if session_id not in self.sessions:
                self.sessions[session_id] = self.setup_system_prompt()
            
            conversation_history = self.sessions[session_id]
            
            # Add user input to conversation history
            conversation_history.append({
                "role": "user", 
                "parts": [user_input]
            })
            
            # Generate response
            chat = self.model.start_chat(history=conversation_history[:-1])
            response = await chat.send_message_async(user_input)
            
            response_text = response.text.strip()
            
            # Add response to history
            conversation_history.append({
                "role": "model",
                "parts": [response_text]
            })
            
            # Keep conversation history manageable
            if len(conversation_history) > 20:
                system_msg = conversation_history[:2]
                recent_msgs = conversation_history[-10:]
                self.sessions[session_id] = system_msg + recent_msgs
            
            return response_text
            
        except Exception as e:
            print(f"AI Error: {e}")
            return "I apologize for the technical difficulty. Please call us directly at 512-677-5146."
    
    def generate_speech(self, text):
        """Convert text to speech using ElevenLabs"""
        try:
            # Generate audio using the new API
            audio = self.elevenlabs_client.text_to_speech.convert(
                text=text,
                voice_id="21m00Tcm4TlvDq8ikWAM",  # Rachel - professional female voice
                model_id="eleven_turbo_v2",
                voice_settings=VoiceSettings(
                    stability=0.75,
                    similarity_boost=0.75,
                    style=0.0,
                    use_speaker_boost=True
                )
            )
            
            # Convert to bytes and then base64 for web transmission
            audio_bytes = b"".join(audio)
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            return audio_base64
            
        except Exception as e:
            print(f"TTS Error: {e}")
            return None

# Initialize the agent
agent = WebHardisonVoiceAgent()

@app.route('/')
def index():
    """Main demo page"""
    return render_template('demo.html')

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "Hardison Voice Agent"})

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print("Client connected")
    emit('status', {'message': 'Connected to Hardison Voice Agent'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print("Client disconnected")

@socketio.on('audio_data')
def handle_audio(data):
    """Handle incoming audio data"""
    try:
        audio_data = data['audio']
        session_id = data.get('session_id', 'default')
        
        # Process speech to text
        text = agent.process_audio(audio_data, session_id)
        
        if text:
            emit('transcription', {'text': text})
            
            # Get AI response
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            ai_response = loop.run_until_complete(agent.get_ai_response(text, session_id))
            
            # Generate speech
            audio_response = agent.generate_speech(ai_response)
            
            # Send response back
            emit('ai_response', {
                'text': ai_response,
                'audio': audio_response
            })
        else:
            emit('error', {'message': 'Could not understand audio'})
            
    except Exception as e:
        print(f"Error handling audio: {e}")
        emit('error', {'message': 'Error processing audio'})

@socketio.on('text_input')
def handle_text(data):
    """Handle text input (for testing without microphone)"""
    try:
        text = data['text']
        session_id = data.get('session_id', 'default')
        
        # Get AI response
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        ai_response = loop.run_until_complete(agent.get_ai_response(text, session_id))
        
        # Generate speech
        audio_response = agent.generate_speech(ai_response)
        
        # Send response back
        emit('ai_response', {
            'text': ai_response,
            'audio': audio_response
        })
        
    except Exception as e:
        print(f"Error handling text: {e}")
        emit('error', {'message': 'Error processing text'})

if __name__ == '__main__':
    print("🚀 Starting Hardison Web Voice Agent...")
    print("🌐 Access at: http://localhost:5000")
    print("☁️  Ready for cloud deployment!")
    
    # Check API keys
    required_keys = ['GOOGLE_API_KEY', 'ELEVENLABS_API_KEY']
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    
    if missing_keys:
        print(f"❌ Missing API keys: {', '.join(missing_keys)}")
    else:
        socketio.run(app, debug=True, host='0.0.0.0', port=5000)
