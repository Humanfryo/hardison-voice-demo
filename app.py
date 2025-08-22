from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import os
import openai
from dotenv import load_dotenv
import logging
from datetime import datetime, timedelta
import traceback
import hashlib
import hmac
import json
import re
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from functools import lru_cache
import time

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', os.urandom(24).hex())
socketio = SocketIO(app, cors_allowed_origins=os.getenv('CORS_ORIGINS', '*').split(','))

# Rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# Configure production-ready logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hardison_agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Metrics collector
@dataclass
class Metrics:
    total_requests: int = 0
    successful_responses: int = 0
    fallback_responses: int = 0
    errors: int = 0
    avg_response_time: float = 0.0
    
    def to_dict(self):
        return asdict(self)

class HardisonVoiceAgent:
    def __init__(self):
        self.openai_available = False
        self.openai_error = None
        self.conversations = {}
        self.metrics = Metrics()
        self.max_conversation_length = 30
        self.api_key_status = "not_checked"
        
        # Business configuration
        self.business_config = {
            'name': 'Hardison Heat - Air - Plumbing',
            'owner': 'Jason Hardison',
            'phone': os.getenv('BUSINESS_PHONE', '512-567-6370'),
            'hours': '9 AM to 5 PM, Monday through Sunday',
            'location': 'Post Falls, Idaho',
            'services': ['HVAC', 'Plumbing', 'Emergency Repairs', 'Maintenance']
        }
        
        # Configurable AI parameters
        self.ai_config = {
            'model': os.getenv('OPENAI_MODEL', 'gpt-4o-mini'),
            'max_tokens': int(os.getenv('MAX_TOKENS', '150')),
            'temperature': float(os.getenv('TEMPERATURE', '0.7')),
            'timeout': int(os.getenv('API_TIMEOUT', '15'))
        }
        
        self._initialize_openai()
        
    def _initialize_openai(self):
        """Initialize OpenAI with proper validation"""
        logger.info("🔍 Initializing OpenAI...")
        
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            self.openai_error = "OPENAI_API_KEY not configured"
            logger.warning(f"⚠️ {self.openai_error}")
            return
        
        # More secure API key validation
        if not self._validate_api_key(api_key):
            self.openai_error = "Invalid API key format"
            logger.warning(f"⚠️ {self.openai_error}")
            return
        
        try:
            self.client = openai.OpenAI(api_key=api_key)
            self._test_api_connection()
        except Exception as e:
            self.openai_error = f"OpenAI initialization failed"
            logger.error(f"❌ {self.openai_error}: {str(e)}")
    
    def _validate_api_key(self, api_key: str) -> bool:
        """Validate API key format"""
        valid_prefixes = ('sk-', 'sk-proj-')
        return any(api_key.startswith(prefix) for prefix in valid_prefixes)
    
    def _test_api_connection(self):
        """Test API connection with minimal request"""
        try:
            test_response = self.client.chat.completions.create(
                model=self.ai_config['model'],
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5,
                timeout=5
            )
            self.openai_available = True
            self.api_key_status = "fully_working"
            logger.info("✅ OpenAI API connection verified")
        except Exception as e:
            self.openai_error = "API test failed"
            logger.error(f"❌ API test failed: {str(e)}")
    
    @lru_cache(maxsize=1)
    def get_system_prompt(self) -> str:
        """Generate system prompt with caching"""
        current_month = datetime.now().strftime("%B")
        return f"""You are Sarah, the professional AI assistant for {self.business_config['name']}.

COMPANY INFO:
- Owner: {self.business_config['owner']} (30+ years experience)
- Location: {self.business_config['location']}
- Phone: {self.business_config['phone']}
- Hours: {self.business_config['hours']}
- Services: {', '.join(self.business_config['services'])}
- Current Month: {current_month}

YOUR ROLE:
- Be warm, professional, and empathetic
- Keep responses under 60 words
- Always include phone number for urgent issues
- Prioritize emergency situations
- Provide clear next steps

EMERGENCY INDICATORS:
- No heat in winter / No AC in summer
- Water leaks or flooding
- Gas smells
- Complete system failures

Remember: You represent a trusted local business."""

    def sanitize_input(self, text: str) -> str:
        """Sanitize user input"""
        # Remove potentially harmful characters
        text = re.sub(r'[<>\"\'&]', '', text)
        # Limit length
        text = text[:500]
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text.strip()
    
    def get_response(self, text: str, session_id: str = "default") -> Dict:
        """Get response with metrics tracking"""
        start_time = time.time()
        self.metrics.total_requests += 1
        
        # Sanitize input
        text = self.sanitize_input(text)
        if not text:
            return {
                'text': "I didn't catch that. How can I help you?",
                'source': 'validation',
                'confidence': 1.0
            }
        
        # Check for profanity or inappropriate content
        if self._contains_inappropriate_content(text):
            return {
                'text': "I'm here to help with HVAC and plumbing needs. Please call us at " + self.business_config['phone'],
                'source': 'filter',
                'confidence': 1.0
            }
        
        # Try OpenAI first
        if self.openai_available:
            try:
                response = self._get_openai_response(text, session_id)
                self.metrics.successful_responses += 1
            except Exception as e:
                logger.error(f"OpenAI error: {str(e)}")
                response = self._get_enhanced_fallback(text)
                self.metrics.fallback_responses += 1
        else:
            response = self._get_enhanced_fallback(text)
            self.metrics.fallback_responses += 1
        
        # Track response time
        response_time = time.time() - start_time
        self.metrics.avg_response_time = (
            (self.metrics.avg_response_time * (self.metrics.total_requests - 1) + response_time) 
            / self.metrics.total_requests
        )
        
        response['response_time'] = response_time
        self._log_conversation(session_id, text, response['text'])
        
        return response
    
    def _contains_inappropriate_content(self, text: str) -> bool:
        """Basic inappropriate content filter"""
        # Add your inappropriate words list
        inappropriate_patterns = []  # Add patterns as needed
        text_lower = text.lower()
        return any(pattern in text_lower for pattern in inappropriate_patterns)
    
    def _get_openai_response(self, text: str, session_id: str) -> Dict:
        """Get OpenAI response with proper error handling"""
        # Initialize conversation if needed
        if session_id not in self.conversations:
            self.conversations[session_id] = {
                'history': [
                    {"role": "system", "content": self.get_system_prompt()},
                    {"role": "assistant", "content": "Hi! I'm Sarah from Hardison Heat, Air, and Plumbing. How can I help you today?"}
                ],
                'created_at': datetime.now(),
                'last_activity': datetime.now()
            }
        
        # Update last activity
        self.conversations[session_id]['last_activity'] = datetime.now()
        
        # Add user message
        self.conversations[session_id]['history'].append({"role": "user", "content": text})
        
        # Make API request
        response = self.client.chat.completions.create(
            model=self.ai_config['model'],
            messages=self.conversations[session_id]['history'],
            max_tokens=self.ai_config['max_tokens'],
            temperature=self.ai_config['temperature'],
            timeout=self.ai_config['timeout']
        )
        
        ai_response = response.choices[0].message.content.strip()
        
        # Add to history
        self.conversations[session_id]['history'].append({"role": "assistant", "content": ai_response})
        
        # Trim conversation if too long
        if len(self.conversations[session_id]['history']) > self.max_conversation_length:
            system_msg = self.conversations[session_id]['history'][0]
            recent_msgs = self.conversations[session_id]['history'][-20:]
            self.conversations[session_id]['history'] = [system_msg] + recent_msgs
        
        return {
            'text': ai_response,
            'source': 'openai',
            'confidence': 0.95
        }
    
    def _get_enhanced_fallback(self, text: str) -> Dict:
        """Enhanced fallback responses with intent detection"""
        text_lower = text.lower()
        
        # Intent detection with confidence scores
        intents = {
            'greeting': (['hello', 'hi', 'hey', 'good morning', 'good afternoon'], 0.9),
            'emergency': (['emergency', 'urgent', 'broken', 'flooding', 'no heat', 'no ac', 'leak'], 0.95),
            'heating': (['heat', 'heater', 'furnace', 'cold', 'warm'], 0.85),
            'cooling': (['ac', 'air conditioning', 'cooling', 'hot', 'air conditioner'], 0.85),
            'plumbing': (['leak', 'water', 'plumbing', 'pipe', 'drain', 'toilet'], 0.85),
            'maintenance': (['maintenance', 'service', 'check', 'inspection', 'tune-up'], 0.8),
            'pricing': (['cost', 'price', 'quote', 'estimate', 'how much'], 0.8),
            'hours': (['hours', 'open', 'closed', 'when', 'available'], 0.8)
        }
        
        detected_intent = None
        confidence = 0.5
        
        for intent, (keywords, intent_confidence) in intents.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_intent = intent
                confidence = intent_confidence
                break
        
        # Generate appropriate response based on intent
        responses = {
            'greeting': f"Hello! I'm Sarah from {self.business_config['name']}. How can I help you today? For immediate service, call {self.business_config['phone']}.",
            'emergency': f"This sounds urgent! Please call us immediately at {self.business_config['phone']} for emergency service. We prioritize urgent repairs.",
            'heating': f"For heating issues, please call {self.business_config['phone']}. With 30+ years of experience, we'll get your heating system running properly.",
            'cooling': f"For AC and cooling issues, call {self.business_config['phone']}. We handle all AC repairs and installations in {self.business_config['location']}.",
            'plumbing': f"For plumbing issues, call {self.business_config['phone']}. We handle all plumbing repairs and emergencies.",
            'maintenance': f"Regular maintenance is important! Call {self.business_config['phone']} to schedule service. We're open {self.business_config['hours']}.",
            'pricing': f"For pricing and estimates, please call {self.business_config['phone']}. We provide fair, upfront pricing with no hidden fees.",
            'hours': f"We're open {self.business_config['hours']}. Call {self.business_config['phone']} to schedule service or for emergencies."
        }
        
        response_text = responses.get(
            detected_intent,
            f"I'm Sarah from {self.business_config['name']}. For all HVAC and plumbing needs, call {self.business_config['phone']}."
        )
        
        return {
            'text': response_text,
            'source': 'fallback',
            'intent': detected_intent,
            'confidence': confidence
        }
    
    def _log_conversation(self, session_id: str, user_text: str, ai_response: str):
        """Log conversations for analytics"""
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'session_id': session_id,
                'user_text': user_text[:100],  # Truncate for privacy
                'response_preview': ai_response[:100],
                'source': 'openai' if self.openai_available else 'fallback'
            }
            # In production, send this to analytics service or database
            logger.info(f"Conversation: {json.dumps(log_entry)}")
        except Exception as e:
            logger.error(f"Failed to log conversation: {str(e)}")
    
    def cleanup_old_sessions(self):
        """Clean up inactive sessions"""
        cutoff_time = datetime.now() - timedelta(hours=1)
        sessions_to_remove = [
            sid for sid, data in self.conversations.items()
            if data['last_activity'] < cutoff_time
        ]
        for sid in sessions_to_remove:
            del self.conversations[sid]
        if sessions_to_remove:
            logger.info(f"Cleaned up {len(sessions_to_remove)} inactive sessions")

# Initialize agent
agent = HardisonVoiceAgent()

@app.route('/')
def index():
    return render_template('demo.html')

@app.route('/health')
@limiter.exempt
def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "openai": agent.openai_available,
        "model": agent.ai_config['model'],
        "metrics": agent.metrics.to_dict(),
        "active_sessions": len(agent.conversations)
    }

@app.route('/metrics')
@limiter.limit("10 per hour")
def metrics():
    """Metrics endpoint for monitoring"""
    return {
        "metrics": agent.metrics.to_dict(),
        "active_sessions": len(agent.conversations),
        "uptime": "calculated_separately"
    }

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    client_ip = request.remote_addr
    logger.info(f"✅ Client connected from {client_ip}")
    emit('status', {
        'message': 'Connected to Sarah - Hardison Voice Agent',
        'capabilities': {
            'ai_available': agent.openai_available,
            'fallback_available': True
        }
    })

@socketio.on('disconnect') 
def handle_disconnect():
    """Handle client disconnection"""
    logger.info("Client disconnected")

@socketio.on('text_message')
@limiter.limit("30 per minute")
def handle_text_message(data):
    """Handle text messages with rate limiting"""
    try:
        text = data.get('text', '').strip()
        session_id = data.get('session_id', 'default')
        
        # Get response
        response = agent.get_response(text, session_id)
        
        # Send response
        emit('ai_response', {
            'text': response['text'],
            'audio': None,
            'source': response['source'],
            'confidence': response.get('confidence', 0),
            'response_time': response.get('response_time', 0)
        })
        
        logger.info(f"✅ Response sent (source: {response['source']})")
        
    except Exception as e:
        agent.metrics.errors += 1
        logger.error(f"❌ Handler error: {str(e)}")
        emit('error', {
            'message': f'Sorry, please call us at {agent.business_config["phone"]} for immediate help!'
        })

# Background task to clean up old sessions
@socketio.on('cleanup_sessions')
def cleanup_sessions():
    """Periodic session cleanup"""
    agent.cleanup_old_sessions()

if __name__ == '__main__':
    logger.info("🚀 Starting Hardison Voice Agent (Production Mode)")
    logger.info(f"OpenAI Status: {'Available' if agent.openai_available else 'Using Fallback'}")
    
    port = int(os.environ.get('PORT', 5000))
    
    # Production settings
    socketio.run(
        app, 
        debug=False, 
        host='0.0.0.0', 
        port=port,
        use_reloader=False,
        log_output=True
    )
