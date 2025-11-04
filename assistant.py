"""
Medical Assistant Module for Early Disease Prediction and Patient Query Handling
This module integrates with Google's Gemini API for conversational medical assistance.
"""

import os
import logging
from typing import Dict, List, Optional
from datetime import datetime
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import with error handling
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
    logger.info("Google Generative AI library imported successfully")
except ImportError as e:
    GENAI_AVAILABLE = False
    logger.error(f"Failed to import google.generativeai: {e}")
    logger.error("Install with: pip install google-generativeai")

# Configure the API (use environment variable for security)
GEMINI_API_KEY ="AIzaSyA4p6sgBh-ICH63hWYnhwfCs3hW-ivtfPM"

if not GEMINI_API_KEY:
    logger.warning("⚠️ GEMINI_API_KEY not found in environment variables!")
    logger.warning("Please set it in your .env file or environment")
else:
    logger.info("✓ GEMINI_API_KEY found in environment")
    if GENAI_AVAILABLE:
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            logger.info("✓ Gemini API configured successfully")
        except Exception as e:
            logger.error(f"Failed to configure Gemini API: {e}")

# System prompt for the medical assistant
MEDICAL_ASSISTANT_PROMPT = """You are MediCare AI, an advanced medical assistant designed to help patients with early disease prediction, symptom analysis, and general health queries.

Your primary responsibilities:
1. **Early Disease Prediction**: Analyze symptoms described by patients and suggest possible conditions (always emphasize these are preliminary assessments, not diagnoses)
2. **Health Education**: Provide accurate, evidence-based information about diseases, symptoms, treatments, and preventive care
3. **Symptom Assessment**: Ask relevant follow-up questions to better understand patient conditions
4. **First Aid Guidance**: Provide immediate first aid instructions for emergencies
5. **Doctor Referral**: Identify severe or chronic conditions that require immediate professional medical attention

Critical Guidelines:
- **NEVER provide definitive diagnoses** - only suggest possibilities and emphasize the need for professional evaluation
- **Identify Red Flags**: Recognize severe symptoms like:
  * Chest pain, severe headache, difficulty breathing
  * Severe bleeding, loss of consciousness, stroke symptoms
  * Severe allergic reactions, high fever with confusion
  * Severe abdominal pain, signs of heart attack
  * Suicidal thoughts or severe mental health crisis
  
- **Emergency Response Protocol**: For severe/emergency cases:
  1. Immediately advise calling emergency services (911/local emergency number)
  2. Provide critical first aid steps while waiting for help
  3. Emphasize urgency and NOT waiting
  4. Recommend visiting ER/urgent care
  
- **Doctor Referral Criteria**: Recommend seeing a doctor if:
  * Symptoms persist for more than a few days
  * Symptoms worsen or new symptoms appear
  * Patient has chronic conditions or complex medical history
  * Symptoms suggest potential serious conditions
  * Patient is pregnant, elderly, or immunocompromised
  
- **First Aid Advice**: Provide clear, step-by-step first aid instructions for:
  * Minor cuts, burns, sprains
  * Common injuries and accidents
  * Basic emergency care techniques
  
- **Empathy and Clarity**: 
  * Use compassionate, reassuring language
  * Avoid medical jargon; explain terms simply
  * Be thorough but concise
  * Ask clarifying questions when needed
  
- **Limitations Acknowledgment**: 
  * Clearly state you're an AI assistant, not a doctor
  * Cannot perform physical examinations
  * Cannot prescribe medications
  * Cannot replace professional medical care
  
- **Privacy**: Respect patient confidentiality and handle information sensitively

Response Format:
- Start with empathy and acknowledgment of concern
- Analyze symptoms systematically
- Provide possible explanations (with appropriate caveats)
- Give actionable advice (first aid, self-care, when to see doctor)
- End with clear next steps

Remember: Your goal is to inform, guide, and ensure patient safety - not to replace healthcare professionals."""


class MedicalAssistant:
    """Medical Assistant chatbot using Google Gemini API"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Medical Assistant
        
        Args:
            api_key: Google Gemini API key (if not set via environment variable)
        """
        self.initialized = False
        self.model = None
        self.sessions: Dict[str, any] = {}
        
        # Check if library is available
        if not GENAI_AVAILABLE:
            logger.error("Google Generative AI library not available")
            raise ImportError("google-generativeai package not installed. Install with: pip install google-generativeai")
        
        # Configure API key
        api_key_to_use = api_key or GEMINI_API_KEY
        
        if not api_key_to_use:
            logger.error("No API key provided")
            raise ValueError(
                "GEMINI_API_KEY not found. Please:\n"
                "1. Create a .env file with: GEMINI_API_KEY=your_key_here\n"
                "2. Or set environment variable: export GEMINI_API_KEY=your_key_here\n"
                "3. Get API key from: https://makersuite.google.com/app/apikey"
            )
        
        try:
            genai.configure(api_key=api_key_to_use)
            logger.info("API configured successfully")
            
            # Initialize the model with safety settings
            self.model = genai.GenerativeModel(
                model_name="gemini-2.5-flash",
                generation_config={
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 2048,
                },
                safety_settings=[
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_NONE"
                    }
                ],
                system_instruction=MEDICAL_ASSISTANT_PROMPT
            )
            
            self.initialized = True
            logger.info("✓ Medical Assistant initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Medical Assistant: {e}")
            raise
    
    def create_session(self, session_id: str) -> Dict:
        """
        Create a new chat session
        
        Args:
            session_id: Unique identifier for the session
            
        Returns:
            Session information dictionary
        """
        if not self.initialized:
            return {
                "error": "Assistant not initialized",
                "message": "Medical assistant is not properly configured. Please check API key."
            }
        
        try:
            chat = self.model.start_chat(history=[])
            self.sessions[session_id] = {
                "chat": chat,
                "created_at": datetime.now().isoformat(),
                "message_count": 0
            }
            logger.info(f"Session created: {session_id}")
            return {
                "session_id": session_id,
                "status": "created",
                "message": "Medical assistant session started. How can I help you today?"
            }
        except Exception as e:
            logger.error(f"Error creating session: {e}")
            return {
                "error": str(e),
                "message": "Failed to create session. Please check API configuration."
            }
    
    def get_response(self, session_id: str, user_message: str) -> Dict:
        """
        Get response from the medical assistant
        
        Args:
            session_id: Session identifier
            user_message: User's message/query
            
        Returns:
            Response dictionary with assistant's reply
        """
        if not self.initialized:
            logger.error("Assistant not initialized")
            return {
                "error": "not_initialized",
                "message": "Medical assistant is not properly configured. Please check your GEMINI_API_KEY."
            }
        
        # Create session if it doesn't exist
        if session_id not in self.sessions:
            logger.info(f"Creating new session for: {session_id}")
            result = self.create_session(session_id)
            if "error" in result:
                return result
        
        session = self.sessions[session_id]
        chat = session["chat"]
        
        try:
            logger.info(f"Sending message to Gemini API for session: {session_id}")
            
            # Send message and get response
            response = chat.send_message(user_message)
            
            # Update session
            session["message_count"] += 1
            session["last_interaction"] = datetime.now().isoformat()
            
            # Analyze if response indicates emergency
            is_emergency = self._detect_emergency(response.text)
            
            logger.info(f"Response received successfully (emergency: {is_emergency})")
            
            return {
                "session_id": session_id,
                "user_message": user_message,
                "assistant_response": response.text,
                "is_emergency": is_emergency,
                "timestamp": datetime.now().isoformat(),
                "message_count": session["message_count"]
            }
            
        except Exception as e:
            logger.error(f"Error getting response: {type(e).__name__}: {str(e)}")
            
            # Provide more specific error messages
            error_msg = str(e)
            if "API_KEY_INVALID" in error_msg or "invalid api key" in error_msg.lower():
                message = "Invalid API key. Please check your GEMINI_API_KEY configuration."
            elif "quota" in error_msg.lower():
                message = "API quota exceeded. Please check your Google Cloud billing or wait for quota reset."
            elif "permission" in error_msg.lower():
                message = "Permission denied. Please ensure your API key has access to Gemini API."
            else:
                message = f"API Error: {error_msg}"
            
            return {
                "session_id": session_id,
                "error": error_msg,
                "message": message
            }
    
    def _detect_emergency(self, response_text: str) -> bool:
        """
        Detect if the response indicates an emergency situation
        
        Args:
            response_text: Assistant's response text
            
        Returns:
            Boolean indicating if it's an emergency
        """
        emergency_keywords = [
            "emergency", "911", "urgent care", "immediately",
            "call an ambulance", "go to the ER", "emergency room",
            "life-threatening", "seek immediate", "critical"
        ]
        
        response_lower = response_text.lower()
        return any(keyword in response_lower for keyword in emergency_keywords)
    
    def get_session_history(self, session_id: str) -> Dict:
        """
        Get conversation history for a session
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session history dictionary
        """
        if session_id not in self.sessions:
            return {"error": "Session not found"}
        
        session = self.sessions[session_id]
        chat = session["chat"]
        
        # Convert history to readable format
        history = []
        for message in chat.history:
            history.append({
                "role": message.role,
                "content": message.parts[0].text if message.parts else ""
            })
        
        return {
            "session_id": session_id,
            "created_at": session["created_at"],
            "message_count": session["message_count"],
            "history": history
        }
    
    def end_session(self, session_id: str) -> Dict:
        """
        End a chat session
        
        Args:
            session_id: Session identifier
            
        Returns:
            Status dictionary
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Session ended: {session_id}")
            return {"session_id": session_id, "status": "ended"}
        return {"error": "Session not found"}
    
    def clear_all_sessions(self) -> Dict:
        """Clear all active sessions"""
        count = len(self.sessions)
        self.sessions.clear()
        logger.info(f"Cleared {count} sessions")
        return {"message": f"Cleared {count} sessions"}


# Initialize global assistant instance with error handling
medical_assistant = None

try:
    medical_assistant = MedicalAssistant()
    logger.info("✓ Global medical assistant instance created")
except Exception as e:
    logger.error(f"✗ Failed to initialize medical assistant: {e}")
    logger.error("The API will not function until this is resolved!")


# Convenience functions for FastAPI integration
def create_chat_session(session_id: str) -> Dict:
    """Create a new chat session"""
    if medical_assistant is None:
        return {
            "error": "assistant_not_initialized",
            "message": "Medical assistant failed to initialize. Check server logs and API key configuration."
        }
    return medical_assistant.create_session(session_id)


def get_medical_response(session_id: str, message: str) -> Dict:
    """Get response from medical assistant"""
    if medical_assistant is None:
        return {
            "error": "assistant_not_initialized",
            "message": "Medical assistant failed to initialize. Check server logs and API key configuration."
        }
    return medical_assistant.get_response(session_id, message)


def get_chat_history(session_id: str) -> Dict:
    """Get conversation history"""
    if medical_assistant is None:
        return {"error": "Assistant not initialized"}
    return medical_assistant.get_session_history(session_id)


def end_chat_session(session_id: str) -> Dict:
    """End a chat session"""
    if medical_assistant is None:
        return {"error": "Assistant not initialized"}
    return medical_assistant.end_session(session_id)


# Test function
def test_configuration():
    """Test if the medical assistant is properly configured"""
    logger.info("=" * 60)
    logger.info("CONFIGURATION TEST")
    logger.info("=" * 60)
    
    # Check API key
    if not GEMINI_API_KEY:
        logger.error("✗ GEMINI_API_KEY not found in environment")
        return False
    else:
        logger.info(f"✓ GEMINI_API_KEY found (length: {len(GEMINI_API_KEY)})")
    
    # Check library
    if not GENAI_AVAILABLE:
        logger.error("✗ google-generativeai library not available")
        return False
    else:
        logger.info("✓ google-generativeai library imported")
    
    # Check assistant
    if medical_assistant is None:
        logger.error("✗ Medical assistant not initialized")
        return False
    else:
        logger.info("✓ Medical assistant initialized")
    
    logger.info("=" * 60)
    return True


# Example usage and testing
if __name__ == "__main__":
    # Run configuration test
    if not test_configuration():
        logger.error("Configuration test failed! Fix the issues above.")
        exit(1)
    
    # Test the assistant
    test_session_id = "test_session_001"
    
    # Create session
    print("\n" + "="*60)
    print("Creating session...")
    print("="*60)
    result = create_chat_session(test_session_id)
    print(json.dumps(result, indent=2))
    
    if "error" in result:
        print("\n✗ Failed to create session. Check configuration.")
        exit(1)
    
    # Test query
    test_query = "I have a persistent headache and mild fever for 2 days"
    print(f"\n{'='*60}")
    print(f"Query: {test_query}")
    print(f"{'='*60}")
    response = get_medical_response(test_session_id, test_query)
    
    if "error" in response:
        print(f"\n✗ Error: {response.get('message')}")
        print(f"Details: {response.get('error')}")
    else:
        print(f"\nResponse: {response['assistant_response']}")
        if response.get('is_emergency'):
            print("\n⚠️ EMERGENCY DETECTED")
    
    # End session
    end_chat_session(test_session_id)