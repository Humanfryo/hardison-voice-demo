from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import os
import openai
from dotenv import load_dotenv
import logging
from datetime import datetime
import traceback
import json
from typing import Dict, List, Optional, Tuple
from enum import Enum
import re

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'hardison-pro-voice-agent')
socketio = SocketIO(app, cors_allowed_origins="*")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ServiceType(Enum):
    """Service type categories"""
    EMERGENCY = "emergency"
    APPOINTMENT = "appointment"
    INFORMATION = "information"
    QUOTE = "quote"
    UNKNOWN = "unknown"

class UrgencyLevel(Enum):
    """Urgency levels for service requests"""
    CRITICAL = "critical"  # Gas leak, flooding
    HIGH = "high"  # No heat in winter, no AC in summer
    MEDIUM = "medium"  # No hot water, non-critical repairs
    LOW = "low"  # Maintenance, information requests

class HardisonProAgent:
    def __init__(self):
        self.openai_available = False
        self.openai_error = None
        self.conversations = {}
        self.api_key_status = "not_checked"
        
        # Load business configuration from JSON
        self.config = self.load_business_config()
        
        # Initialize OpenAI
        self._initialize_openai()
        
    def load_business_config(self) -> Dict:
        """Load the business configuration from JSON"""
        return {
            "company_name": "Hardison Pro",
            "tagline": "Your trusted local HVAC and plumbing experts",
            "location": "Post Falls, Idaho and surrounding areas",
            "experience": "Over 30 years",
            "owner": "Jason",
            "phone": os.getenv('BUSINESS_PHONE', '208-555-0100'),  # Update with actual number
            "hours": "Monday through Friday, 8 AM to 5 PM",
            "website": "hardisonpro.com",
            "services": {
                "hvac": [
                    "Furnace repair", "Furnace replacement", "AC installation",
                    "Heating system maintenance", "Ductwork", "Thermostat installation"
                ],
                "plumbing": [
                    "Pipe repair", "Drain cleaning", "Water heater installation",
                    "Fixture installation", "Emergency plumbing", "Remodel plumbing"
                ],
                "construction": [
                    "New construction HVAC", "New construction plumbing", "Remodel projects"
                ]
            },
            "emergency_keywords": {
                "critical": ["gas", "smell gas", "gas leak", "flooding", "flood", "burst pipe"],
                "high": ["no heat", "no cooling", "no ac", "frozen", "no air"],
                "medium": ["no hot water", "water heater", "leak", "dripping"]
            }
        }
    
    def _initialize_openai(self):
        """Initialize OpenAI with validation"""
        logger.info("🔧 Initializing OpenAI for Hardison Pro...")
        
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            self.openai_error = "No OPENAI_API_KEY found"
            logger.warning(f"⚠️ {self.openai_error}")
            return
        
        try:
            self.client = openai.OpenAI(api_key=api_key)
            # Test connection
            test_response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5,
                timeout=5
            )
            self.openai_available = True
            self.api_key_status = "fully_working"
            logger.info("✅ OpenAI connection established")
        except Exception as e:
            self.openai_error = f"OpenAI setup failed: {str(e)}"
            logger.error(f"❌ {self.openai_error}")
    
    def get_system_prompt(self) -> str:
        """Generate the Hardison Pro specific system prompt"""
        current_date = datetime.now().strftime("%B %d, %Y")
        current_season = self.get_current_season()
        
        return f"""You are a professional voice assistant for {self.config['company_name']}, {self.config['tagline']}.

COMPANY INFORMATION:
- Business: {self.config['company_name']}
- Owner: {self.config['owner']} (owner/operator with {self.config['experience']} experience)
- Location: {self.config['location']}
- Business Hours: {self.config['hours']}
- Emergency service available after hours
- Website: {self.config['website']}
- Today's Date: {current_date}
- Current Season: {current_season}

SERVICES PROVIDED:
HVAC Services: {', '.join(self.config['services']['hvac'])}
Plumbing Services: {', '.join(self.config['services']['plumbing'])}
Construction: {', '.join(self.config['services']['construction'])}

YOUR ROLE & PERSONALITY:
- Professional, friendly, and helpful voice assistant
- Clear, conversational, and efficient communication
- Represent a reliable, trustworthy, family-oriented local business
- Always empathetic to customer concerns, especially emergencies

CONVERSATION GUIDELINES:
1. GREETING: Start with "Thank you for calling {self.config['company_name']}, your trusted local HVAC and plumbing experts. How can I help you today?"
2. IDENTIFY NEEDS: Quickly determine if they need emergency service, appointment scheduling, service information, or a quote
3. GATHER INFO: Collect name, phone, address, issue details, preferred timing
4. URGENCY: Prioritize emergencies (no heat in winter, flooding, gas leaks)
5. SCHEDULING: Offer available appointments, confirm details
6. PRICING: Explain that accurate estimates require on-site evaluation, mention financing

EMERGENCY PROTOCOLS:
- Gas smell/leak: "For safety, please leave the area immediately and call the gas company. Once safe, we can help with repairs."
- Flooding: "Please shut off the main water valve if safely accessible. I'm scheduling emergency service right away."
- No heat (winter): "I understand you're without heat - that's a priority. Let me schedule same-day service."
- No AC (summer): "No AC in this heat is urgent. Let me get you scheduled for priority service."

RESPONSE STYLE:
- Keep responses under 75 words unless providing detailed information
- Be conversational but professional
- Show empathy for urgent situations
- Always provide clear next steps
- Include the business phone number when relevant

IMPORTANT:
- Financing is available - customers can get pre-qualified at {self.config['website']}
- All work comes with warranty protection
- Over 30 years serving the local community
- {self.config['owner']} personally ensures quality on every job"""
    
    def get_current_season(self) -> str:
        """Determine current season for context"""
        month = datetime.now().month
        if month in [12, 1, 2]:
            return "Winter (heating season - prioritize heat issues)"
        elif month in [6, 7, 8]:
            return "Summer (cooling season - prioritize AC issues)"
        elif month in [3, 4, 5]:
            return "Spring"
        else:
            return "Fall"
    
    def detect_service_type(self, text: str) -> Tuple[ServiceType, UrgencyLevel]:
        """Detect the type of service needed and urgency level"""
        text_lower = text.lower()
        
        # Check for emergencies first
        for level, keywords in self.config['emergency_keywords'].items():
            if any(keyword in text_lower for keyword in keywords):
                if level == "critical":
                    return ServiceType.EMERGENCY, UrgencyLevel.CRITICAL
                elif level == "high":
                    return ServiceType.EMERGENCY, UrgencyLevel.HIGH
                elif level == "medium":
                    return ServiceType.EMERGENCY, UrgencyLevel.MEDIUM
        
        # Check for appointment scheduling
        appointment_keywords = ['schedule', 'appointment', 'come out', 'visit', 'service call']
        if any(keyword in text_lower for keyword in appointment_keywords):
            return ServiceType.APPOINTMENT, UrgencyLevel.LOW
        
        # Check for quotes
        quote_keywords = ['quote', 'estimate', 'cost', 'price', 'how much']
        if any(keyword in text_lower for keyword in quote_keywords):
            return ServiceType.QUOTE, UrgencyLevel.LOW
        
        # Default to information request
        return ServiceType.INFORMATION, UrgencyLevel.LOW
    
    def extract_customer_info(self, text: str) -> Dict:
        """Extract customer information from text"""
        info = {}
        
        # Phone number extraction (basic pattern)
        phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        phone_match = re.search(phone_pattern, text)
        if phone_match:
            info['phone'] = phone_match.group()
        
        # Email extraction
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_match = re.search(email_pattern, text)
        if email_match:
            info['email'] = email_match.group()
        
        # Name extraction (simplified - looks for "my name is" or "I'm")
        name_patterns = [
            r"my name is ([A-Za-z]+)",
            r"i'm ([A-Za-z]+)",
            r"this is ([A-Za-z]+)"
        ]
        for pattern in name_patterns:
            match = re.search(pattern, text.lower())
            if match:
                info['name'] = match.group(1).capitalize()
                break
        
        return info
    
    def get_response(self, text: str, session_id: str = "default") -> str:
        """Get response for customer inquiry"""
        
        # Clean input
        text = text.strip()
        if not text:
            return "I didn't catch that. How can I help you today?"
        
        # Detect service type and urgency
        service_type, urgency = self.detect_service_type(text)
        
        # Extract customer information
        customer_info = self.extract_customer_info(text)
        
        # Store in session
        if session_id not in self.conversations:
            self.conversations[session_id] = {
                'history': [],
                'customer_info': {},
                'service_type': None,
                'urgency': None
            }
        
        # Update session info
        self.conversations[session_id]['customer_info'].update(customer_info)
        self.conversations[session_id]['service_type'] = service_type
        self.conversations[session_id]['urgency'] = urgency
        
        # Log the interaction
        logger.info(f"🔍 Service Type: {service_type.value}, Urgency: {urgency.value}")
        
        # Get appropriate response
        if self.openai_available:
            return self._get_ai_response(text, session_id, service_type, urgency)
        else:
            return self._get_smart_fallback(text, service_type, urgency)
    
    def _get_ai_response(self, text: str, session_id: str, 
                        service_type: ServiceType, urgency: UrgencyLevel) -> str:
        """Get AI-powered response"""
        try:
            # Initialize conversation if needed
            if not self.conversations[session_id]['history']:
                self.conversations[session_id]['history'] = [
                    {"role": "system", "content": self.get_system_prompt()},
                    {"role": "assistant", "content": f"Thank you for calling {self.config['company_name']}, your trusted local HVAC and plumbing experts. How can I help you today?"}
                ]
            
            # Add context about detected intent
            context = f"[Service Type: {service_type.value}, Urgency: {urgency.value}]"
            
            # Add user message with context
            self.conversations[session_id]['history'].append({
                "role": "user", 
                "content": f"{context} {text}"
            })
            
            # Get AI response
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=self.conversations[session_id]['history'],
                max_tokens=150,
                temperature=0.7,
                timeout=15
            )
            
            ai_response = response.choices[0].message.content.strip()
            
            # Store response
            self.conversations[session_id]['history'].append({
                "role": "assistant",
                "content": ai_response
            })
            
            # Trim history if too long
            if len(self.conversations[session_id]['history']) > 20:
                system_msg = self.conversations[session_id]['history'][0]
                recent = self.conversations[session_id]['history'][-18:]
                self.conversations[session_id]['history'] = [system_msg] + recent
            
            return ai_response
            
        except Exception as e:
            logger.error(f"AI response error: {str(e)}")
            return self._get_smart_fallback(text, service_type, urgency)
    
    def _get_smart_fallback(self, text: str, service_type: ServiceType, 
                           urgency: UrgencyLevel) -> str:
        """Smart fallback responses based on service type and urgency"""
        
        company = self.config['company_name']
        phone = self.config.get('phone', '208-555-0100')
        
        # Emergency responses by urgency
        if service_type == ServiceType.EMERGENCY:
            if urgency == UrgencyLevel.CRITICAL:
                return f"This is an emergency! For gas leaks, leave immediately and call 911. For flooding, shut off the main water valve if safe. Call us at {phone} for immediate emergency service."
            elif urgency == UrgencyLevel.HIGH:
                season = self.get_current_season()
                if "winter" in season.lower():
                    return f"No heat is our top priority! I'm marking this as urgent. Please call {phone} immediately for same-day emergency service. We'll get your heat restored quickly."
                else:
                    return f"No AC in this heat is urgent! Call {phone} right away for priority service. We'll get your cooling restored as soon as possible."
            else:
                return f"I understand this needs attention. Please call {phone} and we'll schedule service today or tomorrow. {company} is here to help!"
        
        # Appointment scheduling
        elif service_type == ServiceType.APPOINTMENT:
            return f"I'd be happy to schedule your service appointment. Please call {phone} to speak with our scheduling team. We're available {self.config['hours']}, with emergency service after hours."
        
        # Quote requests
        elif service_type == ServiceType.QUOTE:
            return f"For accurate pricing, we provide free on-site estimates. Call {phone} to schedule. We offer upfront pricing and financing options - you can pre-qualify at {self.config['website']}."
        
        # Information requests
        else:
            text_lower = text.lower()
            
            if any(word in text_lower for word in ['hour', 'open', 'closed']):
                return f"We're open {self.config['hours']} for regular service, with 24/7 emergency availability. Call {phone} anytime!"
            
            elif any(word in text_lower for word in ['financ', 'payment', 'afford']):
                return f"We offer flexible financing to make repairs affordable. Get pre-qualified instantly at {self.config['website']} or call {phone} to learn more."
            
            elif any(word in text_lower for word in ['warrant', 'guarantee']):
                return f"All our work comes with warranty protection. {self.config['owner']} personally ensures quality on every job. Call {phone} for details."
            
            else:
                return f"Thank you for calling {company}. With {self.config['experience']} serving {self.config['location']}, we're here for all your HVAC and plumbing needs. Call {phone} or visit {self.config['website']}."

# Initialize agent
agent = HardisonProAgent()

@app.route('/')
def index():
    return render_template('demo.html')

@app.route('/health')
def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "company": agent.config['company_name'],
        "openai_available": agent.openai_available,
        "api_status": agent.api_key_status,
        "active_sessions": len(agent.conversations)
    }

@app.route('/session/<session_id>')
def session_info(session_id):
    """Get session information for debugging"""
    if session_id in agent.conversations:
        session = agent.conversations[session_id]
        return {
            "session_id": session_id,
            "customer_info": session.get('customer_info', {}),
            "service_type": session.get('service_type', 'unknown'),
            "urgency": session.get('urgency', 'unknown'),
            "message_count": len(session.get('history', []))
        }
    return {"error": "Session not found"}, 404

@socketio.on('connect')
def handle_connect():
    logger.info(f"✅ Client connected - {agent.config['company_name']} Voice Agent")
    emit('status', {
        'message': f"Connected to {agent.config['company_name']} Voice Assistant",
        'greeting': f"Thank you for calling {agent.config['company_name']}!",
        'ai_available': agent.openai_available
    })

@socketio.on('disconnect') 
def handle_disconnect():
    logger.info("Client disconnected")

@socketio.on('text_message')
def handle_text_message(data):
    """Handle incoming text messages"""
    try:
        text = data.get('text', '').strip()
        session_id = data.get('session_id', 'default')
        
        if not text:
            emit('error', {'message': 'Please say or type something'})
            return
        
        logger.info(f"📝 Processing: {text[:50]}...")
        
        # Get response
        response = agent.get_response(text, session_id)
        
        # Get session info for context
        session = agent.conversations.get(session_id, {})
        
        # Send response with metadata
        emit('ai_response', {
            'text': response,
            'audio': None,
            'source': 'openai' if agent.openai_available else 'fallback',
            'service_type': session.get('service_type', ServiceType.UNKNOWN).value if session.get('service_type') else 'unknown',
            'urgency': session.get('urgency', UrgencyLevel.LOW).value if session.get('urgency') else 'low',
            'customer_info': session.get('customer_info', {})
        })
        
        logger.info(f"✅ Response sent for {session.get('service_type', 'unknown')}")
        
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        logger.error(traceback.format_exc())
        emit('error', {
            'message': f'Sorry for the trouble. Please call {agent.config.get("phone", "us")} for immediate assistance.'
        })

if __name__ == '__main__':
    logger.info(f"🏢 Starting {agent.config['company_name']} Voice Agent")
    logger.info(f"OpenAI: {'Connected' if agent.openai_available else 'Using Fallback'}")
    logger.info(f"Location: {agent.config['location']}")
    logger.info(f"Services: HVAC, Plumbing, Construction")
    
    port = int(os.environ.get('PORT', 5000))
    socketio.run(app, debug=False, host='0.0.0.0', port=port)
