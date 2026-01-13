# -*- coding: utf-8 -*-
"""
Smart PHQ-8 Detection & Empathetic Response System

This script manages the complete workflow:
1. PHQ-8 Detection: Analyzes user input for depression indicators using the detection model
2. Empathetic Response: Generates empathetic responses based on PHQ-8 analysis using the empathy model
3. RoBERTa Empathy Checker: Validates and enhances the response using RoBERTa classification model

Model Paths (auto-detected):
- Detection: /content/drive/MyDrive/CBT_Counsellor_Project/models/final_model (Colab)
  or ./detect (local)
- Empathy: /content/drive/MyDrive/CBT_Counsellor_Project/models/final_model_empathetic (Colab)
  or ./empathy (local)
- RoBERTa: /content/drive/MyDrive/CBT_Counsellor_Project/models/roberta_empathy_final (Colab)
  or ./models/roberta_empathy_final (local)

Both Mistral models are LoRA adapters fine-tuned on Mistral-7B-Instruct-v0.3 base model.
RoBERTa model is a sequence classification model for empathy detection.

Workflow:
User Input → Detection Model (PHQ-8 Analysis) → Empathy Model (Empathetic Response) → RoBERTa Checker (Validation & Enhancement)

Usage:
    # Interactive mode (default):
    python smart_phq8_empathy_system.py
    
    # Or explicitly:
    python smart_phq8_empathy_system.py --interactive
    
    # Single query:
    python smart_phq8_empathy_system.py --input "I've been feeling really down lately"
    
    # Custom model paths:
    python smart_phq8_empathy_system.py --detect-path /path/to/model --empathy-path /path/to/model --roberta-path /path/to/model
    
    # API mode (localhost only):
    python smart_phq8_empathy_system.py --api --port 8000
    
    # API mode with ngrok (optional):
    python smart_phq8_empathy_system.py --api --port 8000 --ngrok --ngrok-token YOUR_TOKEN
"""

import os
import sys
import json
import argparse
import torch
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import re

# Check if running in Colab
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# Install required packages if in Colab
if IN_COLAB:
    print("Running in Google Colab environment")
    print("Checking dependencies...")
    
    import subprocess
    required_packages = [
        ('transformers', '>=4.35.0'),
        ('peft', '>=0.7.0'),
        ('torch', '>=2.0.0'),
        ('nest_asyncio', None),  # For Jupyter/Colab async compatibility
    ]
    
    for package_name, version in required_packages:
        try:
            __import__(package_name)
        except ImportError:
            print(f"Installing {package_name}...")
            if version:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", 
                    f"{package_name}{version}", "-q", "--upgrade"
                ])
            else:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", 
                    package_name, "-q", "--upgrade"
                ])
    
    # Apply nest_asyncio for Colab/Jupyter compatibility
    try:
        import nest_asyncio
        nest_asyncio.apply()
    except ImportError:
        pass  # Will be handled later if needed

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    logging,
    RobertaTokenizer,
    RobertaForSequenceClassification,
)
from peft import PeftModel

# Try to import Gemini for rewriting non-empathetic responses
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None

# Try to import FastAPI for API endpoints
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    FastAPI = None
    HTTPException = None
    CORSMiddleware = None
    BaseModel = None

# Try to import ngrok
try:
    import pyngrok
    from pyngrok import ngrok
    NGROK_AVAILABLE = True
except ImportError:
    NGROK_AVAILABLE = False
    ngrok = None
    pyngrok = None

# Suppress warnings
logging.set_verbosity(logging.ERROR)

# Model paths
BASE_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

# Default model paths - prioritize Google Drive if in Colab, otherwise use local paths
def get_default_detect_path():
    """Get default detection model path with fallback options"""
    # Primary: Google Drive path (Colab)
    colab_path = "/content/drive/MyDrive/CBT_Counsellor_Project/models/final_model"
    if IN_COLAB and os.path.exists(colab_path):
        return colab_path
    # Fallback: local detect folder
    if os.path.exists("./detect"):
        return "./detect"
    # Return Colab path as default (will error if not found, prompting user)
    return colab_path if IN_COLAB else "./detect"

def get_default_empathy_path():
    """Get default empathy model path with fallback options"""
    # Primary: Google Drive path (Colab)
    colab_path = "/content/drive/MyDrive/CBT_Counsellor_Project/models/final_model_empathetic"
    if IN_COLAB and os.path.exists(colab_path):
        return colab_path
    # Fallback: local empathy folder
    if os.path.exists("./empathy"):
        return "./empathy"
    # Return Colab path as default (will error if not found, prompting user)
    return colab_path if IN_COLAB else "./empathy"

def get_default_roberta_path():
    """Get default RoBERTa model path with fallback options"""
    # Primary: Google Drive path (Colab)
    colab_path = "/content/drive/MyDrive/CBT_Counsellor_Project/models/roberta_empathy_final"
    if IN_COLAB and os.path.exists(colab_path):
        return colab_path
    # Fallback: local paths
    for path in ["./models/roberta_empathy_final", "./roberta_empathy_final"]:
        if os.path.exists(path):
            return path
    # Return Colab path as default (will error if not found, prompting user)
    return colab_path if IN_COLAB else "./models/roberta_empathy_final"

DEFAULT_DETECT_PATH = get_default_detect_path()
DEFAULT_EMPATHY_PATH = get_default_empathy_path()
DEFAULT_ROBERTA_PATH = get_default_roberta_path()

# PHQ-8 Questions Reference
PHQ8_QUESTIONS = {
    "interest_pleasure": "Over the last 2 weeks, how often have you been bothered by little interest or pleasure in doing things?",
    "mood": "Over the last 2 weeks, how often have you been bothered by feeling down, depressed, or hopeless?",
    "sleep": "Over the last 2 weeks, how often have you been bothered by trouble falling or staying asleep, or sleeping too much?",
    "energy": "Over the last 2 weeks, how often have you been bothered by feeling tired or having little energy?",
    "appetite": "Over the last 2 weeks, how often have you been bothered by poor appetite or overeating?",
    "self_worth": "Over the last 2 weeks, how often have you been bothered by feeling bad about yourself - or that you are a failure or have let yourself or your family down?",
    "concentration": "Over the last 2 weeks, how often have you been bothered by trouble concentrating on things, such as reading the newspaper or watching television?",
    "psychomotor": "Over the last 2 weeks, how often have you been bothered by moving or speaking so slowly that other people could have noticed, or the opposite - being so fidgety or restless that you have been moving around a lot more than usual?"
}


class PHQ8DetectionModel:
    """Manages PHQ-8 depression detection using the model from detect folder"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        
    def load(self):
        """Load the PHQ-8 detection model (LoRA adapter or full model)"""
        print(f"\n{'='*80}")
        print("Loading PHQ-8 Detection Model")
        print(f"{'='*80}")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Detection model directory not found: {self.model_path}\n"
                f"Please ensure the model exists or specify --detect-path"
            )
        
        # Check if it's a LoRA adapter or full model
        is_lora_adapter = os.path.exists(os.path.join(self.model_path, "adapter_config.json"))
        is_full_model = os.path.exists(os.path.join(self.model_path, "config.json"))
        
        if not (is_lora_adapter or is_full_model):
            raise FileNotFoundError(
                f"Invalid detection model directory: {self.model_path}\n"
                f"Expected either LoRA adapter (adapter_config.json) or full model (config.json) files."
            )
        
        device_map = "auto" if torch.cuda.is_available() else "cpu"
        print(f"Device: {device_map}")
        
        if is_lora_adapter:
            print(f"1. Detected LoRA adapter at: {self.model_path}")
            print(f"2. Loading base model: {BASE_MODEL_NAME}")
            
            # Load tokenizer from adapter directory
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, 
                trust_remote_code=True
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "right"
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_NAME,
                device_map=device_map,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True,
            )
            
            # Load LoRA adapter
            print("3. Loading LoRA adapter...")
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
            self.model = self.model.merge_and_unload()
            self.model.eval()
        else:
            print(f"1. Detected full model at: {self.model_path}")
            print("2. Loading full model...")
            
            # Load tokenizer and model directly
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "right"
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map=device_map,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True,
            )
            self.model.eval()
        
        # Create pipeline for easier inference
        print("3. Creating pipeline...")
        self.pipeline = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=400,
            temperature=0.3,
            do_sample=True,
            return_full_text=False,
        )
        
        print("   PHQ-8 Detection model loaded successfully!")
        return self
    
    def detect_phq8(self, user_input: str, conversation_context: Optional[str] = None) -> Dict:
        """
        Analyze user input for PHQ-8 depression indicators
        
        Args:
            user_input: Current user message/input
            conversation_context: Optional previous conversation context
            
        Returns:
            Dictionary with PHQ-8 assessment including severity, score, symptoms
        """
        # Build conversation context for analysis
        if conversation_context:
            full_context = f"{conversation_context}\n\nUser: {user_input}"
        else:
            full_context = f"User: {user_input}"
        
        # Create detection prompt
        detection_prompt = f"""Analyze the following conversation for depression indicators using PHQ-8 assessment.
Extract PHQ-8 symptoms even from indirect responses and general conversation.

Conversation:
{full_context}

Identify:
1. PHQ-8 symptoms present in the user's responses (even if not explicitly mentioned)
2. PHQ-8 scores for each detected symptom (0-3)
3. Overall depression assessment

PHQ-8 Questions Reference:
{json.dumps(PHQ8_QUESTIONS, indent=2)}

Note: The user may not directly answer PHQ-8 questions. Extract symptoms from their general responses and context.
Provide your analysis in the following format:

PHQ-8 Symptoms Detected:
- [topic]: Score [0-3] (brief explanation)

Total PHQ-8 Score: [X]/24
Depression Assessment: [Minimal/Mild/Moderate/Moderately Severe/Severe]"""
        
        # Query detection model
        formatted_prompt = f"[INST] {detection_prompt} [/INST]"
        
        try:
            response = self.pipeline(formatted_prompt, truncation=True)
            detection_result = response[0]['generated_text'].strip()
            
            # Parse the detection result
            depression_analysis = self._parse_detection_result(detection_result, user_input)
            
            return depression_analysis
            
        except Exception as e:
            print(f"⚠️ Error during PHQ-8 detection: {e}")
            # Return fallback assessment
            return {
                "severity": "Unknown",
                "phq8_score": None,
                "detected_symptoms": [],
                "analysis": f"Error during analysis: {str(e)}",
                "raw_response": detection_result if 'detection_result' in locals() else None,
                "confidence": 0.0
            }
    
    def _parse_detection_result(self, response: str, user_input: str) -> Dict:
        """Parse detection model response into structured format"""
        # Extract PHQ-8 score
        score_match = re.search(r'Total PHQ-8 Score:\s*(\d+)', response, re.IGNORECASE)
        phq8_score = int(score_match.group(1)) if score_match else None
        
        # Extract severity level
        severity_match = re.search(
            r'Depression Assessment:\s*(Minimal|Mild|Moderate|Moderately Severe|Severe)',
            response, 
            re.IGNORECASE
        )
        severity = severity_match.group(1) if severity_match else "Unknown"
        
        # Extract detected symptoms
        detected_symptoms = []
        symptom_pattern = r'-\s*([^:]+):\s*Score\s*(\d+)'
        matches = re.finditer(symptom_pattern, response, re.IGNORECASE)
        
        for match in matches:
            topic = match.group(1).strip()
            score = int(match.group(2))
            detected_symptoms.append({
                "topic": topic.lower().replace(" ", "_"),
                "score": score,
                "description": match.group(0)
            })
        
        # Determine confidence based on response quality
        confidence = 0.7 if phq8_score is not None and detected_symptoms else 0.5
        
        return {
            "severity": severity,
            "phq8_score": phq8_score,
            "detected_symptoms": detected_symptoms,
            "analysis": response,
            "raw_response": response,
            "confidence": confidence,
            "user_input": user_input
        }


class EmpathyResponseModel:
    """Manages empathetic response generation using the model from empathy folder"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        
    def load(self):
        """Load the empathetic response model (LoRA adapter or full model)"""
        print(f"\n{'='*80}")
        print("Loading Empathetic Response Model")
        print(f"{'='*80}")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Empathy model directory not found: {self.model_path}\n"
                f"Please ensure the model exists or specify --empathy-path"
            )
        
        # Check if it's a LoRA adapter or full model
        is_lora_adapter = os.path.exists(os.path.join(self.model_path, "adapter_config.json"))
        is_full_model = os.path.exists(os.path.join(self.model_path, "config.json"))
        
        if not (is_lora_adapter or is_full_model):
            raise FileNotFoundError(
                f"Invalid empathy model directory: {self.model_path}\n"
                f"Expected either LoRA adapter (adapter_config.json) or full model (config.json) files."
            )
        
        device_map = "auto" if torch.cuda.is_available() else "cpu"
        print(f"Device: {device_map}")
        
        if is_lora_adapter:
            print(f"1. Detected LoRA adapter at: {self.model_path}")
            print(f"2. Loading base model: {BASE_MODEL_NAME}")
            
            # Load tokenizer from adapter directory
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "right"
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_NAME,
                device_map=device_map,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True,
            )
            
            # Load LoRA adapter
            print("3. Loading LoRA adapter...")
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
            self.model = self.model.merge_and_unload()
            self.model.eval()
        else:
            print(f"1. Detected full model at: {self.model_path}")
            print("2. Loading full model...")
            
            # Load tokenizer and model directly
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "right"
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map=device_map,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True,
            )
            self.model.eval()
        
        # Create pipeline
        print("3. Creating pipeline...")
        self.pipeline = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=300,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            return_full_text=False,
        )
        
        print("   Empathy model loaded successfully!")
        return self
    
    def generate_response(
        self, 
        user_input: str, 
        depression_analysis: Dict,
        conversation_history: Optional[List[Dict]] = None
    ) -> str:
        """
        Generate empathetic response based on user input and PHQ-8 analysis
        
        Args:
            user_input: Current user message
            depression_analysis: PHQ-8 detection analysis from detection model
            conversation_history: Optional list of previous messages in format [{"role": "user", "content": "..."}, ...]
            
        Returns:
            Empathetic response text
        """
        # Build symptoms description
        if depression_analysis.get("detected_symptoms"):
            symptoms_list = []
            for symptom in depression_analysis["detected_symptoms"]:
                topic_name = symptom.get("topic", "").replace("_", " ").title()
                score = symptom.get("score", 0)
                symptoms_list.append(f"- {topic_name}: Score {score}/3")
            symptoms_desc = "\n".join(symptoms_list)
        else:
            symptoms_desc = "- No specific PHQ-8 symptoms detected (general mental health concerns)"
        
        # Build conversation context if available
        conversation_context = ""
        if conversation_history:
            recent_history = conversation_history[-3:]  # Last 3 exchanges
            context_parts = []
            for msg in recent_history:
                role = msg.get("role", "user").title()
                content = msg.get("content", "")
                context_parts.append(f"{role}: {content}")
            conversation_context = "\n".join(context_parts) + "\n\n"
        
        # Determine tone and approach based on severity
        severity = depression_analysis.get("severity", "Unknown")
        phq8_score = depression_analysis.get("phq8_score", 0) or 0
        
        # Customize prompt based on severity level
        severity_guidance = self._get_severity_guidance(severity, phq8_score)
        
        # Create empathetic response prompt
        empathy_prompt = f"""You are an empathetic and compassionate mental health counselor named Ellie.
You are having a therapeutic conversation with someone who is sharing their feelings and experiences.

{conversation_context}User's Current Statement:
{user_input}

Depression Assessment (PHQ-8):
- Severity Level: {severity} Depression
- PHQ-8 Score: {phq8_score}/24
- Detected Symptoms:
{symptoms_desc}

{severity_guidance}

Please provide a supportive, empathetic, and helpful response that:
1. Acknowledges and validates their feelings
2. Shows genuine understanding and care
3. Offers appropriate support and guidance
4. Maintains a warm, professional, and non-judgmental tone
5. Encourages further conversation if helpful

Keep your response conversational, natural, and focused on their needs. Do NOT diagnose or provide medical advice."""
        
        # Query empathy model
        formatted_prompt = f"[INST] {empathy_prompt} [/INST]"
        
        try:
            response = self.pipeline(formatted_prompt, truncation=True)
            empathetic_response = response[0]['generated_text'].strip()
            
            # Clean up response (remove any remaining prompt artifacts)
            if "[/INST]" in empathetic_response:
                empathetic_response = empathetic_response.split("[/INST]")[-1].strip()
            
            return empathetic_response
            
        except Exception as e:
            print(f"⚠️ Error during empathetic response generation: {e}")
            return f"I'm sorry, I'm having trouble processing that right now. Please know that I'm here to listen and support you. Would you like to share more about what you're experiencing?"
    
    def _get_severity_guidance(self, severity: str, phq8_score: int) -> str:
        """Get guidance text based on depression severity"""
        guidance_templates = {
            "Minimal": """Guidance for Minimal Depression (PHQ-8 Score: 0-4):
- The person shows minimal signs of depression
- Focus on supportive listening and encouragement
- Validate their feelings while maintaining positivity
- Offer gentle suggestions for maintaining mental well-being
- Keep the tone light but still empathetic""",
            
            "Mild": """Guidance for Mild Depression (PHQ-8 Score: 5-9):
- The person shows mild symptoms of depression
- Provide supportive listening and gentle encouragement
- Validate their feelings and normalize their experience
- Suggest coping strategies and self-care practices
- Offer hope and reassurance while being genuine""",
            
            "Moderate": """Guidance for Moderate Depression (PHQ-8 Score: 10-14):
- The person shows moderate symptoms of depression
- Use a warm, supportive, and understanding tone
- Acknowledge the difficulty of what they're experiencing
- Provide validation and emotional support
- Gently suggest professional support and coping strategies
- Avoid minimizing their experience""",
            
            "Moderately Severe": """Guidance for Moderately Severe Depression (PHQ-8 Score: 15-19):
- The person shows moderately severe symptoms of depression
- Use a compassionate, gentle, and supportive approach
- Acknowledge the severity of what they're experiencing
- Provide strong validation and emotional support
- Recommend professional mental health support
- Express genuine care and concern
- Be careful not to overwhelm them with suggestions""",
            
            "Severe": """Guidance for Severe Depression (PHQ-8 Score: 20-24):
- The person shows severe symptoms of depression
- Use the utmost compassion, care, and support
- Acknowledge the significant challenges they're facing
- Provide strong emotional validation and support
- Strongly recommend professional mental health support
- Express deep care and let them know they're not alone
- Be gentle and avoid overwhelming them
- If they express suicidal thoughts, encourage immediate professional help"""
        }
        
        return guidance_templates.get(severity, """Guidance:
- Provide empathetic and supportive listening
- Validate their feelings and experiences
- Offer appropriate support and guidance
- Maintain a warm, professional, and caring tone""")


class RoBERTaEmpathyChecker:
    """Manages RoBERTa-based empathy classification and response enhancement"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.gemini_model = None
        
    def load(self):
        """Load the RoBERTa empathy classification model"""
        print(f"\n{'='*80}")
        print("Loading RoBERTa Empathy Checker Model")
        print(f"{'='*80}")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"RoBERTa model directory not found: {self.model_path}\n"
                f"Please ensure the model exists or specify --roberta-path"
            )
        
        print(f"1. Loading RoBERTa model from: {self.model_path}")
        
        try:
            self.tokenizer = RobertaTokenizer.from_pretrained(
                self.model_path,
                local_files_only=True
            )
            self.model = RobertaForSequenceClassification.from_pretrained(
                self.model_path,
                local_files_only=True
            )
            self.model.eval()
            print("   RoBERTa model loaded successfully!")
        except Exception as e:
            raise FileNotFoundError(
                f"Error loading RoBERTa model from {self.model_path}: {e}\n"
                f"Please ensure the model files are present and valid."
            )
        
        # Initialize Gemini for rewriting if available
        if GEMINI_AVAILABLE:
            try:
                api_key = os.getenv("GEMINI_API_KEY", "AIzaSyDXCBFZKsNhbHGJ3rjbPRv4VnjSQmVWlA0")
                genai.configure(api_key=api_key)
                self.gemini_model = genai.GenerativeModel("models/gemini-2.0-flash")
                print("   Gemini model initialized for response rewriting")
            except Exception as e:
                print(f"   Warning: Could not initialize Gemini: {e}")
                print("   Response rewriting will be skipped if classification fails")
        
        return self
    
    def check_and_enhance(self, text: str, silent: bool = False) -> Dict:
        """
        Check if text is empathetic and rewrite if needed
        
        Args:
            text: Text to check and potentially enhance
            silent: If True, suppress console output
            
        Returns:
            Dictionary with classification results and enhanced text
        """
        if not self.tokenizer or not self.model:
            return {
                "input": text,
                "label": "unknown",
                "confidence": 0.0,
                "rewritten": text,
                "was_rewritten": False
            }
        
        try:
            # Classify empathy
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                pred = torch.argmax(probs, dim=1).item()
                confidence = probs[0][pred].item()
                label = "empathetic" if pred == 1 else "non_empathetic"
            
            rewritten = text
            was_rewritten = False
            
            # Rewrite if not empathetic and Gemini is available
            if label == "non_empathetic" and self.gemini_model:
                if not silent:
                    print(f"   ⚠️  Response classified as non-empathetic (confidence: {confidence:.2f})")
                    print("   → Rewriting response to be more empathetic...")
                
                prompt = f"""
You are a highly empathetic CBT counselor.
Rewrite this message so it sounds warm, understanding, and emotionally validating.
Keep it concise (1–2 sentences), avoid clichés, and sound naturally human.
Do not list multiple options — return only one rewritten version that sounds like a human therapist responding with empathy.

Sentence: "{text}"
Rewritten:
"""
                try:
                    response = self.gemini_model.generate_content(prompt)
                    rewritten = response.text.strip()
                    was_rewritten = True
                    if not silent:
                        print("   ✓ Response rewritten successfully")
                except Exception as e:
                    if not silent:
                        print(f"   ⚠️  Error rewriting response: {e}")
                    rewritten = text
            elif not silent and label == "empathetic":
                print(f"   ✓ Response classified as empathetic (confidence: {confidence:.2f})")
            
            return {
                "input": text,
                "label": label,
                "confidence": round(confidence, 4),
                "rewritten": rewritten,
                "was_rewritten": was_rewritten
            }
            
        except Exception as e:
            if not silent:
                print(f"⚠️ Error during RoBERTa empathy check: {e}")
            return {
                "input": text,
                "label": "unknown",
                "confidence": 0.0,
                "rewritten": text,
                "was_rewritten": False
            }


class SmartPHQ8EmpathySystem:
    """Main system that coordinates PHQ-8 detection, empathetic response generation, and RoBERTa validation"""
    
    def __init__(self, detect_path: str = DEFAULT_DETECT_PATH, empathy_path: str = DEFAULT_EMPATHY_PATH, roberta_path: str = DEFAULT_ROBERTA_PATH):
        self.detect_path = detect_path
        self.empathy_path = empathy_path
        self.roberta_path = roberta_path
        self.detection_model = None
        self.empathy_model = None
        self.roberta_checker = None
        self.conversation_history = []
        
    def initialize(self, skip_roberta_if_missing=False):
        """Load all models"""
        print("\n" + "="*80)
        print("Initializing Smart PHQ-8 & Empathy System")
        print("="*80)
        
        # Load detection model
        self.detection_model = PHQ8DetectionModel(self.detect_path).load()
        
        # Load empathy model
        self.empathy_model = EmpathyResponseModel(self.empathy_path).load()
        
        # Load RoBERTa checker (optional if skip_roberta_if_missing is True)
        roberta_loaded = False
        try:
            self.roberta_checker = RoBERTaEmpathyChecker(self.roberta_path).load()
            roberta_loaded = True
        except FileNotFoundError as e:
            if skip_roberta_if_missing:
                print(f"\n⚠️  Warning: RoBERTa model not found, skipping (optional for /chat endpoint)")
                print(f"   Path: {self.roberta_path}")
                self.roberta_checker = None
                roberta_loaded = False
            else:
                raise
        
        print("\n" + "="*80)
        print("System Initialized Successfully!")
        print("="*80)
        print(f"  - Detection Model: Loaded from {self.detect_path}")
        print(f"  - Empathy Model: Loaded from {self.empathy_path}")
        if roberta_loaded:
            print(f"  - RoBERTa Checker: Loaded from {self.roberta_path}")
        else:
            print(f"  - RoBERTa Checker: Not loaded (optional)")
        
        if torch.cuda.is_available():
            print(f"  - GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
        else:
            print(f"  - Device: CPU")
        
    def process_user_input(self, user_input: str, show_detection: bool = True, silent: bool = False) -> Dict:
        """
        Process user input through both models
        
        Args:
            user_input: User's message/input
            show_detection: Whether to show detection analysis details
            silent: If True, suppress all console output (for API mode)
            
        Returns:
            Dictionary with detection analysis and empathetic response
        """
        if not self.detection_model or not self.empathy_model or not self.roberta_checker:
            raise RuntimeError("Models not initialized. Call initialize() first.")
        
        if not silent:
            print("\n" + "="*80)
            print("Processing User Input")
            print("="*80)
            print(f"\nUser: {user_input}\n")
        
        # Step 1: Detect PHQ-8 depression indicators
        if not silent:
            print("[Step 1] Analyzing PHQ-8 depression indicators...")
        depression_analysis = self.detection_model.detect_phq8(
            user_input,
            self._format_conversation_history()
        )
        
        if not silent and show_detection:
            print(f"\n  ✓ Severity: {depression_analysis.get('severity', 'Unknown')}")
            print(f"  ✓ PHQ-8 Score: {depression_analysis.get('phq8_score', 'N/A')}/24")
            if depression_analysis.get('detected_symptoms'):
                print(f"  ✓ Detected Symptoms: {len(depression_analysis['detected_symptoms'])}")
        
        # Step 2: Generate empathetic response
        if not silent:
            print("\n[Step 2] Generating empathetic response...")
        empathetic_response = self.empathy_model.generate_response(
            user_input,
            depression_analysis,
            self.conversation_history
        )
        
        # Step 3: Check and enhance with RoBERTa
        if not silent:
            print("\n[Step 3] Validating empathy with RoBERTa checker...")
        empathy_check = self.roberta_checker.check_and_enhance(empathetic_response, silent=silent)
        final_response = empathy_check["rewritten"]
        
        # Update conversation history with final response
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": final_response})
        
        # Prepare result
        result = {
            "user_input": user_input,
            "depression_analysis": depression_analysis,
            "empathetic_response": empathetic_response,
            "roberta_check": empathy_check,
            "final_response": final_response,
            "timestamp": datetime.now().isoformat()
        }
        
        if not silent:
            print("\n" + "="*80)
            print("Final Response")
            print("="*80)
            if empathy_check["was_rewritten"]:
                print(f"\n📝 Original Response: {empathetic_response}")
                print(f"\n✨ Enhanced Response:")
            print(f"\nEllie: {final_response}\n")
        
        return result
    
    def _format_conversation_history(self) -> str:
        """Format conversation history for context"""
        if not self.conversation_history:
            return ""
        
        formatted = []
        for msg in self.conversation_history[-5:]:  # Last 5 messages
            role = msg.get("role", "user").title()
            content = msg.get("content", "")
            formatted.append(f"{role}: {content}")
        
        return "\n".join(formatted)


# ============================================================================
# API MODE - FastAPI Endpoints
# ============================================================================

if FASTAPI_AVAILABLE:
    # Create FastAPI app
    app = FastAPI(
        title="Smart PHQ-8 & Empathy System API",
        description="API for PHQ-8 depression detection, empathetic response generation, and CBT chatbot",
        version="1.0.0"
    )
    
    # Add CORS middleware for frontend integration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, specify your frontend domain
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Global system instance (will be initialized in main)
    api_system: Optional[SmartPHQ8EmpathySystem] = None
    
    # Pydantic models for API requests/responses
    class UserInputRequest(BaseModel):
        user_input: str
        session_id: Optional[str] = None
        show_detection: bool = False
    
    class ConversationHistoryRequest(BaseModel):
        session_id: Optional[str] = None
    
    class ClearHistoryRequest(BaseModel):
        session_id: Optional[str] = None
    
    class HealthResponse(BaseModel):
        status: str
        models_loaded: bool
        detection_model: Optional[str] = None
        empathy_model: Optional[str] = None
        roberta_model: Optional[str] = None
    
    @app.get("/", tags=["General"])
    async def root():
        """Root endpoint"""
        return {
            "message": "Smart PHQ-8 & Empathy System API",
            "version": "1.0.0",
            "endpoints": {
                "phq8_process": "/api/process",
                "cbt_chat": "/chat",
                "health": "/api/health",
                "history": "/api/history",
                "clear": "/api/clear",
                "cbt_sessions": "/api/cbt/sessions",
                "docs": "/docs"
            }
        }
    
    @app.get("/api/health", response_model=HealthResponse, tags=["System"])
    async def health_check():
        """Check system health and model status"""
        if api_system is None:
            raise HTTPException(status_code=503, detail="System not initialized")
        
        models_loaded = (
            api_system.detection_model is not None and
            api_system.empathy_model is not None and
            api_system.roberta_checker is not None
        )
        
        return HealthResponse(
            status="healthy" if models_loaded else "initializing",
            models_loaded=models_loaded,
            detection_model=api_system.detect_path if models_loaded else None,
            empathy_model=api_system.empathy_path if models_loaded else None,
            roberta_model=api_system.roberta_path if models_loaded else None
        )
    
    @app.post("/api/process", tags=["Conversation"])
    async def process_user_input(request: UserInputRequest):
        """
        Process user input through the complete pipeline:
        1. PHQ-8 Detection
        2. Empathetic Response Generation
        3. RoBERTa Validation & Enhancement
        
        Note: Requires PHQ-8 models to be initialized
        """
        if api_system is None or api_system.detection_model is None:
            raise HTTPException(
                status_code=503, 
                detail="PHQ-8 system not initialized. Models may not be loaded. Use /chat endpoint for CBT counseling."
            )
        
        try:
            # Process silently (no console output) for API mode
            result = api_system.process_user_input(
                request.user_input,
                show_detection=False,  # Don't show detection details in API
                silent=True  # Suppress all console output
            )
            return {
                "success": True,
                "data": result
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing input: {str(e)}")
    
    @app.get("/api/history", tags=["Conversation"])
    async def get_conversation_history(session_id: Optional[str] = None):
        """Get conversation history"""
        if api_system is None:
            raise HTTPException(status_code=503, detail="System not initialized")
        
        return {
            "success": True,
            "data": {
                "conversation_history": api_system.conversation_history,
                "message_count": len(api_system.conversation_history)
            }
        }
    
    @app.post("/api/clear", tags=["Conversation"])
    async def clear_conversation_history(request: ClearHistoryRequest):
        """Clear conversation history"""
        if api_system is None:
            raise HTTPException(status_code=503, detail="System not initialized")
        
        api_system.conversation_history = []
        return {
            "success": True,
            "message": "Conversation history cleared"
        }
    
    @app.get("/api/analysis", tags=["Analysis"])
    async def get_last_analysis():
        """Get PHQ-8 analysis for the last user message"""
        if api_system is None:
            raise HTTPException(status_code=503, detail="System not initialized")
        
        if not api_system.conversation_history:
            raise HTTPException(status_code=404, detail="No conversation history found")
        
        # Find last user message
        last_user_msg = None
        for msg in reversed(api_system.conversation_history):
            if msg.get("role") == "user":
                last_user_msg = msg.get("content")
                break
        
        if not last_user_msg:
            raise HTTPException(status_code=404, detail="No user message found in history")
        
        try:
            result = api_system.process_user_input(last_user_msg, show_detection=True)
            return {
                "success": True,
                "data": {
                    "depression_analysis": result["depression_analysis"],
                    "user_input": last_user_msg
                }
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error getting analysis: {str(e)}")
    
    # ============================================================================
    # CBT CHATBOT ENDPOINTS
    # ============================================================================
    
    # CBT Session phases and PHQ-8 questions
    sessions_phases = {
        1: ["start", "mood", "thoughts", "challenge", "coping", "summary"],
        2: ["mood_review", "thoughts_review", "link_feelings", "summary"],
        3: ["identify_distortion", "reframe_thought", "reflection", "summary"],
        4: ["explore_actions", "choose_activity", "reflect_mood", "summary"],
        5: ["identify_values", "link_activities", "motivation", "summary"],
        6: ["inner_critic", "self_kindness", "calming_exercise", "summary"],
        7: ["review_progress", "identify_helpful_strategies", "adjust_goals", "summary"],
        8: ["reframe_thoughts", "challenge_beliefs", "balanced_view", "summary"],
        9: ["strengths_review", "early_warnings", "prevention_plan", "summary"],
        10: ["maintenance_habits", "social_support", "connection_reflection", "summary"],
        11: ["self_coaching", "confidence_building", "future_visualization", "summary"],
        12: ["journey_reflection", "celebrate_growth", "farewell", "summary"]
    }
    
    phq8_questions_by_session = {
        1: [
            "Over the past couple of weeks, have you found yourself losing interest or enjoyment in things you usually like?",
            "Lately, have you felt more down or hopeless than usual?",
            "Have you had any trouble sleeping — like falling asleep, staying asleep, or sleeping too much?",
            "Do you find yourself feeling low on energy or easily tired these days?",
            "How has your appetite been — eating less or more than normal?",
            "Have you been hard on yourself, or felt like you've let yourself or others down recently?",
            "Do you notice it's been harder to focus on things like reading, TV, or conversations?",
            "Have you been moving or speaking slower than usual, or feeling unusually restless?"
        ],
        4: [
            "In the last two weeks, have you noticed less enjoyment in your usual activities?",
            "Have feelings of sadness or hopelessness been showing up lately?",
            "How's your sleep been — getting too little or too much rest?",
            "Have your energy levels been lower than normal?",
            "Has your appetite changed — maybe eating too much or too little?",
            "Have you caught yourself being overly self-critical or feeling like a failure?",
            "Is it difficult to concentrate on daily tasks or hobbies recently?",
            "Have you been feeling slowed down, or maybe restless inside your body?"
        ],
        8: [
            "Recently, have you felt less interested in things that normally make you happy?",
            "Have you had moments of feeling sad or hopeless more often than not?",
            "How have your sleep patterns been — any struggles with falling asleep or oversleeping?",
            "Do you often feel low on motivation or energy during the day?",
            "Have your eating habits changed at all these past two weeks?",
            "Have you been feeling guilty, worthless, or disappointed in yourself?",
            "Has your ability to focus on reading or conversations been affected?",
            "Have others mentioned that you seem more restless or slowed down than usual?"
        ],
        12: [
            "Over the last couple of weeks, have you been enjoying your activities as much as before?",
            "Have feelings of sadness or emptiness been visiting you recently?",
            "How's your sleep quality been — are you getting enough rest?",
            "Have you been feeling tired or lacking energy most days?",
            "Has your appetite or eating pattern changed in any way?",
            "Have you been judging yourself harshly or feeling like you're falling short?",
            "Do you find it hard to stay focused or pay attention lately?",
            "Have you felt physically slowed down or more fidgety than usual?"
        ]
    }
    
    # Initialize Gemini model for CBT chatbot
    cbt_gemini_model = None
    if GEMINI_AVAILABLE:
        try:
            api_key = os.getenv("GEMINI_API_KEY", "AIzaSyDXCBFZKsNhbHGJ3rjbPRv4VnjSQmVWlA0")
            if api_key:
                genai.configure(api_key=api_key)
                cbt_gemini_model = genai.GenerativeModel("models/gemini-2.0-flash")
            else:
                print("Warning: GEMINI_API_KEY not set. CBT chatbot will use fallback.")
        except Exception as e:
            print(f"Warning: Could not initialize Gemini for CBT: {e}")
    
    def build_cbt_prompt(user_input, phase, history, session_num):
        """Build CBT counseling prompt"""
        base = """
You are a professional CBT (Cognitive Behavioral Therapy) counselor.
- Respond in 3–5 sentences.
- Be empathetic, calm, and supportive.
- Ask short reflective questions to explore feelings and thoughts.
- Focus on understanding emotions, not fixing them quickly.
- Avoid giving long advice or multiple solutions at once.
"""
        
        tasks = {
            1: {
                "start": "Comfort the user and give a brief CBT introduction. Then ask how they feel today.",
                "mood": "Ask user to rate their mood 1–10.",
                "thoughts": "Explore what thoughts or events brought them here.",
                "challenge": "Gently help reframe negative thoughts.",
                "coping": "Suggest one small positive action.",
                "summary": "Summarize and close warmly."
            },
            2: {
                "mood_review": "Ask how their week has been emotionally and explore mood changes.",
                "thoughts_review": "Explore recurring thoughts or triggers.",
                "link_feelings": "Help link thoughts with feelings.",
                "summary": "Summarize and close warmly."
            },
            3: {
                "identify_distortion": "Help user identify a cognitive distortion.",
                "reframe_thought": "Guide them to reframe it gently.",
                "reflection": "Ask what they notice about their thinking.",
                "summary": "Summarize and close warmly."
            },
            4: {
                "explore_actions": "Discuss link between actions and emotions.",
                "choose_activity": "Encourage one small positive activity.",
                "reflect_mood": "Ask how action impacts mood.",
                "summary": "Summarize and close warmly."
            },
            5: {
                "identify_values": "Identify what truly matters to the user.",
                "link_activities": "Connect activities with core values.",
                "motivation": "Ask what motivates them to act consistently.",
                "summary": "Summarize and close warmly."
            },
            6: {
                "inner_critic": "Discuss inner critic and self-kindness.",
                "self_kindness": "Encourage gentle self-compassion.",
                "calming_exercise": "Suggest one simple grounding strategy.",
                "summary": "Summarize and close warmly."
            },
            7: {
                "review_progress": "Ask how their mood and coping have progressed.",
                "identify_helpful_strategies": "Explore strategies that helped most.",
                "adjust_goals": "Ask if they want to modify any goals.",
                "summary": "Summarize and close warmly."
            },
            8: {
                "reframe_thoughts": "Strengthen realistic thinking.",
                "challenge_beliefs": "Gently question negative beliefs.",
                "balanced_view": "Encourage seeing both sides.",
                "summary": "Summarize and close warmly."
            },
            9: {
                "strengths_review": "Ask about key coping strengths.",
                "early_warnings": "Identify early warning signs for relapse.",
                "prevention_plan": "Help plan small prevention steps.",
                "summary": "Summarize and close warmly."
            },
            10: {
                "maintenance_habits": "Discuss habits for maintaining progress.",
                "social_support": "Ask about sources of support.",
                "connection_reflection": "Encourage reflection on social connections.",
                "summary": "Summarize and close warmly."
            },
            11: {
                "self_coaching": "Teach self-coaching questions.",
                "confidence_building": "Build confidence in independent CBT use.",
                "future_visualization": "Encourage imagining a calm, stable future self.",
                "summary": "Summarize and close warmly."
            },
            12: {
                "journey_reflection": "Reflect on the whole journey.",
                "celebrate_growth": "Celebrate achievements and learning.",
                "farewell": "End with warm farewell and encouragement.",
                "summary": "Summarize and close warmly."
            }
        }
        
        task = tasks.get(session_num, {}).get(phase, "Summarize and close warmly.")
        short_history = "\n".join(history.split("\n")[-4:]) if history else ""
        
        prompt = f"""{base}

Conversation so far:

{short_history}

Your current goal: {task}

User: {user_input}

Counselor:"""
        
        return prompt
    
    class ChatRequest(BaseModel):
        user_message: str
        session_number: int = 1
        session_phase: str = "start"
        chat_history: str = ""
    
    @app.post("/chat", tags=["CBT Chatbot"])
    async def cbt_chat(req: ChatRequest):
        """
        CBT Chatbot endpoint - Provides CBT counseling responses
        
        Sessions: 1-12 (each with different phases)
        Occasionally inserts PHQ-8 questions during sessions 1, 4, 8, 12
        
        This endpoint works independently and doesn't require PHQ-8 models.
        """
        try:
            # Occasionally insert PHQ-8 questions
            if req.session_number in phq8_questions_by_session:
                questions = phq8_questions_by_session[req.session_number]
                import random
                if random.random() < 0.25:  # 25% chance to ask PHQ-8 question
                    return {
                        "success": True,
                        "reply": random.choice(questions),
                        "is_phq8_question": True,
                        "session_number": req.session_number,
                        "session_phase": req.session_phase
                    }
            
            # Build prompt
            prompt = build_cbt_prompt(
                req.user_message,
                req.session_phase,
                req.chat_history,
                req.session_number
            )
            
            # Generate response using Gemini
            if cbt_gemini_model:
                try:
                    response = cbt_gemini_model.generate_content(prompt)
                    reply = response.text.strip() if response and getattr(response, "text", None) else "I'm sorry, could you please repeat that?"
                except Exception as e:
                    # Suppress error output in API mode
                    reply = "I'm here to listen and support you. Could you tell me more about what you're experiencing?"
            else:
                # Fallback response if Gemini not available
                reply = "I'm here to listen and support you. Could you tell me more about what you're experiencing?"
            
            return {
                "success": True,
                "reply": reply,
                "is_phq8_question": False,
                "session_number": req.session_number,
                "session_phase": req.session_phase
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error in CBT chat: {str(e)}")
    
    @app.get("/api/cbt/sessions", tags=["CBT Chatbot"])
    async def get_cbt_sessions():
        """Get available CBT sessions and phases"""
        return {
            "success": True,
            "data": {
                "sessions": sessions_phases,
                "phq8_sessions": list(phq8_questions_by_session.keys())
            }
        }
    
    @app.get("/api/cbt/phq8-questions/{session_number}", tags=["CBT Chatbot"])
    async def get_phq8_questions(session_number: int):
        """Get PHQ-8 questions for a specific session"""
        if session_number not in phq8_questions_by_session:
            raise HTTPException(
                status_code=404,
                detail=f"PHQ-8 questions not available for session {session_number}"
            )
        
        return {
            "success": True,
            "data": {
                "session_number": session_number,
                "questions": phq8_questions_by_session[session_number]
            }
        }


def run_interactive_mode(system: SmartPHQ8EmpathySystem):
    """Run interactive conversation mode"""
    print("\n" + "="*80)
    print("Interactive Conversation Mode")
    print("="*80)
    print("\nYou're now chatting with Ellie, an empathetic mental health counselor.")
    print("Type 'quit', 'exit', or 'bye' to end the conversation.")
    print("Type 'history' to see conversation history.")
    print("Type 'clear' to clear conversation history.")
    print("Type 'analysis' to see detailed PHQ-8 analysis.\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\nEllie: Thank you for sharing with me today. Take care of yourself, and remember I'm here if you need to talk again. ❤️\n")
                break
            
            if user_input.lower() == 'history':
                print("\nConversation History:")
                for i, msg in enumerate(system.conversation_history[-10:], 1):
                    role = msg.get("role", "user").title()
                    content = msg.get("content", "")[:100]  # First 100 chars
                    print(f"  {i}. {role}: {content}...")
                print()
                continue
            
            if user_input.lower() == 'clear':
                system.conversation_history = []
                print("\n✓ Conversation history cleared\n")
                continue
            
            if user_input.lower() == 'analysis':
                if system.conversation_history:
                    last_user_msg = None
                    for msg in reversed(system.conversation_history):
                        if msg.get("role") == "user":
                            last_user_msg = msg.get("content")
                            break
                    if last_user_msg:
                        print("\n📊 PHQ-8 Analysis:")
                        result = system.process_user_input(last_user_msg, show_detection=True)
                        print(f"\nDetailed Analysis:\n{result['depression_analysis'].get('analysis', 'N/A')}\n")
                    else:
                        print("\nNo previous user message found.\n")
                else:
                    print("\nNo conversation history yet.\n")
                continue
            
            # Process input
            result = system.process_user_input(user_input, show_detection=False)
            
        except KeyboardInterrupt:
            print("\n\nEllie: I understand. Feel free to come back anytime you want to talk. Take care. ❤️\n")
            break
        except Exception as e:
            print(f"\n⚠️ Error: {e}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Smart PHQ-8 Detection & Empathetic Response System"
    )
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Run in interactive conversation mode'
    )
    parser.add_argument(
        '--input', '-in',
        type=str,
        help='Single user input to process'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Save results to JSON file'
    )
    parser.add_argument(
        '--detect-path',
        type=str,
        default=None,
        help='Path to detection model (default: auto-detect from Google Drive or ./detect)'
    )
    parser.add_argument(
        '--empathy-path',
        type=str,
        default=None,
        help='Path to empathy model (default: auto-detect from Google Drive or ./empathy)'
    )
    parser.add_argument(
        '--roberta-path',
        type=str,
        default=None,
        help='Path to RoBERTa model (default: auto-detect from Google Drive or ./models/roberta_empathy_final)'
    )
    parser.add_argument(
        '--api',
        action='store_true',
        help='Run as API server instead of interactive mode'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='Port for API server (default: 8000)'
    )
    parser.add_argument(
        '--ngrok',
        action='store_true',
        help='Expose API using ngrok tunnel (optional, requires ngrok authtoken)'
    )
    parser.add_argument(
        '--no-ngrok',
        action='store_true',
        help='Disable ngrok tunnel (use localhost only)'
    )
    parser.add_argument(
        '--ngrok-token',
        type=str,
        default=None,
        help='Ngrok authtoken (can also be set via NGROK_AUTHTOKEN env var)'
    )
    
    # In Colab, Jupyter kernel passes extra arguments that we need to ignore
    if IN_COLAB:
        args, unknown = parser.parse_known_args()
        if unknown:
            print(f"Note: Ignoring unknown arguments from kernel launcher: {unknown}")
    else:
        args = parser.parse_args()
    
    # Determine model paths (use provided paths or auto-detect)
    if args.detect_path:
        detect_path = args.detect_path
    else:
        detect_path = get_default_detect_path()
        print(f"Using detection model path: {detect_path}")
    
    if args.empathy_path:
        empathy_path = args.empathy_path
    else:
        empathy_path = get_default_empathy_path()
        print(f"Using empathy model path: {empathy_path}")
    
    if args.roberta_path:
        roberta_path = args.roberta_path
    else:
        roberta_path = get_default_roberta_path()
        print(f"Using RoBERTa model path: {roberta_path}")
    
    # Verify paths exist
    if not os.path.exists(detect_path):
        print(f"\n⚠️  Warning: Detection model path not found: {detect_path}")
        print("Trying alternative paths...")
        alt_detect_paths = [
            "/content/drive/MyDrive/CBT_Counsellor_Project/models/final_model",
            "./detect",
            "./models/final_model"
        ]
        found = False
        for alt_path in alt_detect_paths:
            if os.path.exists(alt_path):
                detect_path = alt_path
                print(f"✓ Found detection model at: {detect_path}")
                found = True
                break
        if not found:
            raise FileNotFoundError(
                f"Detection model not found. Tried: {detect_path} and alternatives.\n"
                f"Please specify --detect-path with the correct path."
            )
    
    if not os.path.exists(empathy_path):
        print(f"\n⚠️  Warning: Empathy model path not found: {empathy_path}")
        print("Trying alternative paths...")
        alt_empathy_paths = [
            "/content/drive/MyDrive/CBT_Counsellor_Project/models/final_model_empathetic",
            "./empathy",
            "./models/final_model_empathetic"
        ]
        found = False
        for alt_path in alt_empathy_paths:
            if os.path.exists(alt_path):
                empathy_path = alt_path
                print(f"✓ Found empathy model at: {empathy_path}")
                found = True
                break
        if not found:
            raise FileNotFoundError(
                f"Empathy model not found. Tried: {empathy_path} and alternatives.\n"
                f"Please specify --empathy-path with the correct path."
            )
    
    if not os.path.exists(roberta_path):
        print(f"\n⚠️  Warning: RoBERTa model path not found: {roberta_path}")
        print("Trying alternative paths...")
        alt_roberta_paths = [
            "/content/drive/MyDrive/CBT_Counsellor_Project/models/roberta_empathy_final",
            "./models/roberta_empathy_final",
            "./roberta_empathy_final"
        ]
        found = False
        for alt_path in alt_roberta_paths:
            if os.path.exists(alt_path):
                roberta_path = alt_path
                print(f"✓ Found RoBERTa model at: {roberta_path}")
                found = True
                break
        if not found:
            # In API mode, RoBERTa is optional (only needed for /api/process, not /chat)
            if args.api:
                print(f"⚠️  RoBERTa model not found. Continuing in API mode.")
                print(f"   /chat endpoint will work without RoBERTa model.")
                print(f"   /api/process endpoint requires all models.")
                # Set a dummy path to avoid errors, but it won't be loaded
                roberta_path = "./models/roberta_empathy_final"
            else:
                raise FileNotFoundError(
                    f"RoBERTa model not found. Tried: {roberta_path} and alternatives.\n"
                    f"Please specify --roberta-path with the correct path."
                )
    
    # Initialize system (models will be loaded in API mode if needed)
    system = SmartPHQ8EmpathySystem(
        detect_path=detect_path,
        empathy_path=empathy_path,
        roberta_path=roberta_path
    )
    
    # Determine run mode - Interactive mode is default
    # Default to interactive mode if no specific mode is requested
    # Check if args exists (might not be defined if parse_args failed)
    if 'args' not in locals():
        # Create a simple args object for interactive mode
        class Args:
            api = False
            interactive = True
            input = None
            port = 8000
            ngrok = False
            no_ngrok = False
            ngrok_token = None
            detect_path = None
            empathy_path = None
            roberta_path = None
        args = Args()
    elif not args.api and not args.interactive and not args.input:
        args.interactive = True  # Default to interactive mode
    
    # Run based on mode - Interactive mode is default
    # Only initialize PHQ-8 models if needed (not for CBT-only mode)
    initialize_phq8_models = True  # Always try to initialize
    
    # For non-API modes, initialize models now
    if args.interactive or args.input:
        try:
            system.initialize()
        except FileNotFoundError as e:
            print("\n" + "="*80)
            print("❌ MODEL SETUP ERROR")
            print("="*80)
            print(str(e))
            print("\n" + "="*80)
            sys.exit(1)
        except Exception as e:
            print(f"\nError initializing system: {e}")
            print("\nTroubleshooting:")
            print("1. Ensure detection model is in ./detect folder")
            print("2. Ensure empathy model is in ./empathy folder")
            print("3. Verify models are LoRA adapters trained on Mistral-7B-Instruct-v0.3")
            print("4. Check that you have sufficient GPU memory (~14GB)")
            sys.exit(1)
    
    if args.api:
        # API Mode (default) - All communication through API endpoints
        if not FASTAPI_AVAILABLE:
            print("\n❌ FastAPI is not available. Please install it:")
            print("   pip install fastapi uvicorn")
            sys.exit(1)
        
        # Only initialize PHQ-8 system if needed (for /api/process endpoint)
        # CBT chatbot (/chat) doesn't need PHQ-8 models
        if initialize_phq8_models:
            try:
                # In API mode, allow skipping RoBERTa if missing (only needed for /api/process)
                system.initialize(skip_roberta_if_missing=True)
                # Set global system instance for API endpoints
                if FASTAPI_AVAILABLE:
                    globals()['api_system'] = system
                print("\nPHQ-8 models loaded successfully. All endpoints available.")
            except FileNotFoundError as e:
                print("\n" + "="*80)
                print("MODEL SETUP WARNING (API Mode)")
                print("="*80)
                print(str(e))
                print("\n" + "="*80)
                print("Note: PHQ-8 models not loaded. CBT chatbot (/chat) will still work.")
                print("   To use PHQ-8 detection (/api/process), ensure models are available.")
                print("   Continuing with API server for /chat endpoint...")
                print("="*80)
                # Don't set api_system, so /api/process will return 503
            except Exception as e:
                print(f"\n⚠️  Warning: Error initializing PHQ-8 models: {e}")
                print("   CBT chatbot (/chat) will still work without PHQ-8 models.")
                print("   Continuing with API server...")
        else:
            print("\n⚠️  Skipping PHQ-8 model initialization (CBT-only mode)")
            print("   Use /chat endpoint for CBT counseling")
            print("   Use /api/process endpoint for PHQ-8 detection (requires models)")
        
        import uvicorn
        
        # Setup ngrok - Only use if explicitly enabled
        ngrok_url = None
        # Use ngrok only if explicitly enabled with --ngrok flag
        use_ngrok = args.ngrok
        ngrok_module = None  # Will hold the ngrok module if available
        
        # Check ngrok availability (use global variable)
        global NGROK_AVAILABLE
        ngrok_available = NGROK_AVAILABLE
        
        if use_ngrok:
            if not ngrok_available:
                print("\n⚠️  Warning: pyngrok is not available. Installing...")
                import subprocess
                try:
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install", 
                        "pyngrok", "-q", "--upgrade"
                    ])
                    import pyngrok
                    from pyngrok import ngrok as ngrok_module
                    ngrok_available = True
                    NGROK_AVAILABLE = True  # Update global
                except Exception as e:
                    print(f"❌ Failed to install pyngrok: {e}")
                    print("   Continuing without ngrok...")
                    use_ngrok = False
            else:
                # Import ngrok if already available
                try:
                    from pyngrok import ngrok as ngrok_module
                except ImportError:
                    ngrok_available = False
                    use_ngrok = False
            
            if ngrok_available and ngrok_module:
                try:
                    # Set ngrok authtoken (prioritize env var, then args)
                    ngrok_token = os.getenv("NGROK_AUTHTOKEN") or args.ngrok_token
                    if ngrok_token:
                        print(f"   Setting ngrok authtoken...")
                        ngrok_module.set_auth_token(ngrok_token)
                        print(f"   ✓ Ngrok authtoken configured")
                    else:
                        print(f"   ⚠️  No ngrok authtoken provided. Using default (may have limitations)")
                    
                    # Create ngrok tunnel
                    print(f"\n{'='*80}")
                    print("🌐 Creating ngrok tunnel...")
                    print(f"{'='*80}")
                    public_url = ngrok_module.connect(args.port)
                    ngrok_url = public_url.public_url
                    print(f"\n✅ Ngrok tunnel created successfully!")
                    print(f"  🌐 Public URL: {ngrok_url}")
                    print(f"  🏠 Local URL: http://localhost:{args.port}")
                    print(f"\n  📚 API Documentation: {ngrok_url}/docs")
                    print(f"  💬 CBT Chat Endpoint: POST {ngrok_url}/chat")
                    print(f"  🧠 PHQ-8 Process Endpoint: POST {ngrok_url}/api/process")
                    print(f"  ❤️  Health Check: GET {ngrok_url}/api/health")
                    print(f"\n  Example curl command:")
                    print(f'  curl -X POST "{ngrok_url}/chat" \\')
                    print(f'    -H "Content-Type: application/json" \\')
                    print(f'    -d \'{{"user_message": "Hello", "session_number": 1, "session_phase": "start", "chat_history": ""}}\'')
                    print(f"\n{'='*80}\n")
                except Exception as e:
                    print(f"⚠️  Warning: Failed to create ngrok tunnel: {e}")
                    import traceback
                    traceback.print_exc()
                    print("   Continuing without ngrok...")
                    use_ngrok = False
        
        print(f"\n{'='*80}")
        print("API Server Starting")
        print(f"{'='*80}")
        print(f"   Local URL: http://localhost:{args.port}")
        if ngrok_url:
            print(f"   Public URL: {ngrok_url}")
        else:
            print(f"   No ngrok tunnel (use --ngrok flag)")
        print(f"\n   API Docs: http://localhost:{args.port}/docs")
        print(f"   CBT Chat: POST http://localhost:{args.port}/chat")
        print(f"   PHQ-8 Process: POST http://localhost:{args.port}/api/process")
        print(f"\n  Press Ctrl+C to stop the server")
        print(f"{'='*80}\n")
        
        # Suppress verbose output during API requests
        import logging
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
        
        # Check if we're in an async environment (Jupyter/Colab)
        try:
            import asyncio
            loop = asyncio.get_running_loop()
            in_async_env = True
        except RuntimeError:
            in_async_env = False
        
        # Run uvicorn server
        if in_async_env:
            # In Jupyter/Colab, run uvicorn in a separate thread with its own event loop
            print("⚠️  Running in async environment (Jupyter/Colab)")
            print("   Starting uvicorn server in background thread...")
            
            import threading
            import time
            
            def run_server():
                """Run uvicorn server in a separate thread with new event loop"""
                # Create a new event loop for this thread
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                
                config = uvicorn.Config(
                    app,
                    host="0.0.0.0",
                    port=args.port,
                    log_level="warning"
                )
                server = uvicorn.Server(config)
                
                # Run server in the new event loop
                try:
                    new_loop.run_until_complete(server.serve())
                finally:
                    new_loop.close()
            
            # Start server in background thread
            server_thread = threading.Thread(target=run_server, daemon=True)
            server_thread.start()
            
            # Wait a moment for server to start
            time.sleep(3)
            
            print(f"\n✓ Server started in background thread")
            print(f"  Server is running on port {args.port}")
            if ngrok_url:
                print(f"  Public URL: {ngrok_url}")
            print(f"\n  Server will continue running in the background.")
            print(f"  Press Ctrl+C to stop (or interrupt the kernel)")
            
            # Keep the main thread alive
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n\nShutting down server...")
                if ngrok_url and ngrok_module:
                    try:
                        ngrok_module.kill()
                        print("✓ Ngrok tunnel closed")
                    except:
                        pass
        else:
            # Standard environment - use uvicorn.run normally
            try:
                uvicorn.run(
                    app,
                    host="0.0.0.0",
                    port=args.port,
                    log_level="warning"  # Reduced logging
                )
            except KeyboardInterrupt:
                print("\n\nShutting down server...")
                if ngrok_url and ngrok_module:
                    try:
                        ngrok_module.kill()
                        print("✓ Ngrok tunnel closed")
                    except:
                        pass
    
    elif args.interactive:
        # Interactive mode (optional, for testing only)
        system.initialize()
        run_interactive_mode(system)
    elif args.input:
        # Single query mode
        system.initialize()
        result = system.process_user_input(args.input)
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\n✓ Results saved to {args.output}")


if __name__ == "__main__":
    main()

