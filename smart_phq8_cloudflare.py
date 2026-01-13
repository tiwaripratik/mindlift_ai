# -*- coding: utf-8 -*-

"""
Smart PHQ-8 Detection & Empathetic Response System with Cloudflare Tunnel

FastAPI server with Cloudflare Tunnel for stable public URL access in Google Colab.

Usage in Google Colab:
    !python smart_phq8_cloudflare.py

The script will:
1. Install dependencies
2. Start FastAPI server immediately (without transformers)
3. Create Cloudflare Tunnel
4. Load transformers/models only when needed (lazy loading)

Endpoints:
- /chat - CBT Chatbot (works immediately, uses Gemini)
- /api/process - Full PHQ-8 pipeline (loads models on first use)
- /api/health - Health check
- /docs - API documentation
"""

import os
import sys
import json
import subprocess
import time
import threading
import asyncio
from typing import Dict, List, Optional
from datetime import datetime
import re

# Check if running in Colab
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# ============================================================================
# INSTALLATION
# ============================================================================

if IN_COLAB:
    print("🔧 Installing dependencies...")
    print("   This may take a few minutes...")
    try:
        packages = [
            "fastapi",
            "uvicorn[standard]",
            "pydantic",
            "nest_asyncio",
            "google-generativeai",
        ]
        
        for pkg in packages:
            try:
                print(f"   Installing {pkg}...")
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "-q", "--upgrade", pkg],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=300
                )
            except Exception as e:
                print(f"   ⚠️  {pkg} failed: {e}")
        
        print("✅ Core dependencies installed")
        time.sleep(1)
    except Exception as e:
        print(f"⚠️  Installation error: {e}")

# ============================================================================
# CORE IMPORTS (Required for server)
# ============================================================================

print("📦 Loading core packages...")

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
    print("   ✅ FastAPI loaded")
except Exception as e:
    print(f"❌ FastAPI failed: {e}")
    sys.exit(1)

try:
    import nest_asyncio
    nest_asyncio.apply()
    print("   ✅ nest_asyncio loaded")
except:
    pass

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
    print("   ✅ Google Generative AI loaded")
except:
    GEMINI_AVAILABLE = False
    print("   ⚠️  Google Generative AI not available")

print("✅ Core packages ready - Server can start immediately\n")

# ============================================================================
# LAZY LOADING FOR TRANSFORMERS
# ============================================================================

# Global variables for lazy-loaded transformers
_transformers_loaded = False
_transformers_available = False
_transformers_lock = threading.Lock()

def load_transformers_lazy():
    """Load transformers only when needed"""
    global _transformers_loaded, _transformers_available
    
    with _transformers_lock:
        if _transformers_loaded:
            return _transformers_available
        
        _transformers_loaded = True
        print("\n🔄 Loading transformers (lazy load)...")
        sys.stdout.flush()
        
        try:
            # Install transformers if not available
            try:
                import transformers
            except ImportError:
                print("   Installing transformers...")
                sys.stdout.flush()
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "-q", "transformers>=4.35.0", "peft>=0.7.0", "torch>=2.0.0"],
                    timeout=600
                )
                time.sleep(2)
            
            # Import transformers components
            print("   → Importing PyTorch...")
            sys.stdout.flush()
            import torch
            print(f"   ✅ PyTorch loaded")
            sys.stdout.flush()
            
            # Check GPU availability
            if torch.cuda.is_available():
                print(f"   🎮 GPU detected: {torch.cuda.get_device_name(0)}")
                print(f"   💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
                sys.stdout.flush()
                # Warm up GPU
                try:
                    torch.cuda.empty_cache()
                    # Small tensor operation to initialize CUDA
                    _ = torch.zeros(1).cuda()
                    print("   ✅ GPU initialized")
                    sys.stdout.flush()
                except Exception as e:
                    print(f"   ⚠️  GPU initialization warning: {e}")
                    sys.stdout.flush()
            else:
                print("   ℹ️  Using CPU (no GPU available)")
                sys.stdout.flush()
            
            print("   → Importing transformers components...")
            sys.stdout.flush()
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from transformers.utils import logging as transformers_logging
            from peft import PeftModel
            
            # Try to import pipeline - this is where crashes happen
            print("   → Importing pipeline...")
            sys.stdout.flush()
            try:
                from transformers.pipelines import pipeline
            except:
                from transformers import pipeline
            
            # Store in globals
            globals()['torch'] = torch
            globals()['AutoModelForCausalLM'] = AutoModelForCausalLM
            globals()['AutoTokenizer'] = AutoTokenizer
            globals()['pipeline'] = pipeline
            globals()['transformers_logging'] = transformers_logging
            globals()['PeftModel'] = PeftModel
            
            transformers_logging.set_verbosity(transformers_logging.ERROR)
            
            _transformers_available = True
            print("✅ Transformers loaded successfully")
            sys.stdout.flush()
            return True
            
        except Exception as e:
            print(f"⚠️  Transformers failed to load: {type(e).__name__}: {str(e)[:100]}")
            sys.stdout.flush()
            import traceback
            traceback.print_exc()
            _transformers_available = False
            return False

# ============================================================================
# CLOUDFLARE TUNNEL
# ============================================================================

class CloudflareTunnel:
    def __init__(self, port: int = 8000):
        self.port = port
        self.process = None
        self.public_url = None
        self.cloudflared_path = None
        
    def install_cloudflared(self):
        print("\n📥 Installing cloudflared...")
        try:
            result = subprocess.run(["cloudflared", "--version"], capture_output=True, timeout=5)
            if result.returncode == 0:
                print("✅ Cloudflared already installed")
                self.cloudflared_path = "cloudflared"
                return True
        except:
            pass
        
        try:
            cloudflared_url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
            if subprocess.run(["which", "wget"], capture_output=True).returncode == 0:
                subprocess.run(["wget", "-q", cloudflared_url, "-O", "cloudflared"], check=True)
            else:
                subprocess.run(["curl", "-L", cloudflared_url, "-o", "cloudflared"], check=True)
            subprocess.run(["chmod", "+x", "cloudflared"], check=True)
            self.cloudflared_path = "./cloudflared"
            print("✅ Cloudflared installed")
            return True
        except Exception as e:
            print(f"❌ Failed to install cloudflared: {e}")
            return False
    
    def start_tunnel(self):
        if not self.cloudflared_path:
            if not self.install_cloudflared():
                return None
        
        print(f"\n🌐 Starting Cloudflare Tunnel on port {self.port}...")
        try:
            cmd = [self.cloudflared_path, "tunnel", "--url", f"http://localhost:{self.port}"]
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            output_lines = []
            url_found = False
            
            def read_output():
                nonlocal url_found
                while self.process.poll() is None:
                    line = self.process.stdout.readline()
                    if line:
                        line = line.strip()
                        output_lines.append(line)
                        print(f"   {line}")
                        url_match = re.search(r'https://[a-zA-Z0-9-]+\.trycloudflare\.com', line)
                        if url_match:
                            self.public_url = url_match.group(0)
                            url_found = True
                            return
            
            read_thread = threading.Thread(target=read_output, daemon=True)
            read_thread.start()
            
            for _ in range(30):
                if self.process.poll() is not None:
                    return None
                if url_found:
                    break
                time.sleep(0.5)
            
            if url_found and self.public_url:
                print(f"\n✅ Cloudflare Tunnel created!")
                print(f"🌐 Public URL: {self.public_url}")
                return self.public_url
            else:
                return "https://tunnel-active-check-logs"
        except Exception as e:
            print(f"❌ Failed to start tunnel: {e}")
            return None
    
    def stop_tunnel(self):
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except:
                try:
                    self.process.kill()
                except:
                    pass

# ============================================================================
# MODEL PATHS
# ============================================================================

BASE_MODEL_DIR = "/content/drive/MyDrive/CBT_Counsellor_Project/models"

def get_default_detect_path():
    paths = [
        os.path.join(BASE_MODEL_DIR, "final_model"),
        "/content/drive/MyDrive/CBT_Counsellor_Project/models/final_model",
        "./detect",
        "./models/final_model"
    ]
    for p in paths:
        if os.path.exists(p):
            return p
    return paths[0] if IN_COLAB else "./detect"

def get_default_empathy_path():
    paths = [
        os.path.join(BASE_MODEL_DIR, "final_model_empathetic"),
        "/content/drive/MyDrive/CBT_Counsellor_Project/models/final_model_empathetic",
        "./empathy",
        "./models/final_model_empathetic"
    ]
    for p in paths:
        if os.path.exists(p):
            return p
    return paths[0] if IN_COLAB else "./empathy"

def get_default_roberta_path():
    paths = [
        os.path.join(BASE_MODEL_DIR, "roberta_empathy_final"),
        "/content/drive/MyDrive/CBT_Counsellor_Project/models/roberta_empathy_final",
        "./models/roberta_empathy_final"
    ]
    for p in paths:
        if os.path.exists(p):
            return p
    return paths[0] if IN_COLAB else "./models/roberta_empathy_final"

# ============================================================================
# MODEL CLASSES (Lazy loaded)
# ============================================================================

BASE_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

class PHQ8DetectionModel:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.pipeline = None

    def load(self):
        if not load_transformers_lazy():
            raise ImportError("Transformers not available")
        
        print(f"\n{'='*80}")
        print("Loading PHQ-8 Detection Model")
        print(f"{'='*80}")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        torch = globals().get('torch')
        AutoModelForCausalLM = globals().get('AutoModelForCausalLM')
        AutoTokenizer = globals().get('AutoTokenizer')
        pipeline = globals().get('pipeline')
        
        is_lora = os.path.exists(os.path.join(self.model_path, "adapter_config.json"))
        device_map = "auto" if torch.cuda.is_available() else "cpu"
        
        if is_lora:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_NAME,
                device_map=device_map,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True,
            )
            PeftModel = globals().get('PeftModel')
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
            self.model = self.model.merge_and_unload()
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map=device_map,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True,
            )
        
        self.model.eval()
        self.pipeline = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=400,
            temperature=0.3,
            do_sample=True,
            return_full_text=False,
        )
        
        print("✅ Detection model loaded")
        return self

    def detect_phq8(self, user_input: str, context: Optional[str] = None) -> Dict:
        full_context = f"{context}\n\nUser: {user_input}" if context else f"User: {user_input}"
        prompt = f"""Analyze for PHQ-8 depression indicators.

Conversation:
{full_context}

Provide PHQ-8 assessment with:
- Detected symptoms (0-3 each)
- Total score /24
- Severity level

PHQ-8 Symptoms Detected:
- [symptom]: Score [0-3]

Total PHQ-8 Score: [X]/24
Depression Assessment: [Minimal/Mild/Moderate/Moderately Severe/Severe]"""
        formatted_prompt = f"[INST] {prompt} [/INST]"
        
        try:
            response = self.pipeline(formatted_prompt, truncation=True)
            result = response[0]['generated_text'].strip()
            return self._parse_result(result, user_input)
        except Exception as e:
            return {
                "severity": "Unknown",
                "phq8_score": None,
                "detected_symptoms": [],
                "analysis": f"Error: {e}",
                "confidence": 0.0
            }
    
    def _parse_result(self, response: str, user_input: str) -> Dict:
        score_match = re.search(r'Total PHQ-8 Score:\s*(\d+)', response, re.IGNORECASE)
        phq8_score = int(score_match.group(1)) if score_match else None
        severity_match = re.search(
            r'Depression Assessment:\s*(Minimal|Mild|Moderate|Moderately Severe|Severe)',
            response, re.IGNORECASE
        )
        severity = severity_match.group(1) if severity_match else "Unknown"
        symptoms = []
        for match in re.finditer(r'-\s*([^:]+):\s*Score\s*(\d+)', response, re.IGNORECASE):
            symptoms.append({
                "topic": match.group(1).strip().lower().replace(" ", "_"),
                "score": int(match.group(2)),
                "description": match.group(0)
            })
        return {
            "severity": severity,
            "phq8_score": phq8_score,
            "detected_symptoms": symptoms,
            "analysis": response,
            "confidence": 0.7 if phq8_score and symptoms else 0.5,
            "user_input": user_input
        }

class EmpathyResponseModel:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.pipeline = None

    def load(self):
        if not load_transformers_lazy():
            raise ImportError("Transformers not available")
        
        print(f"\n{'='*80}")
        print("Loading Empathy Model")
        print(f"{'='*80}")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        torch = globals().get('torch')
        AutoModelForCausalLM = globals().get('AutoModelForCausalLM')
        AutoTokenizer = globals().get('AutoTokenizer')
        pipeline = globals().get('pipeline')
        
        is_lora = os.path.exists(os.path.join(self.model_path, "adapter_config.json"))
        device_map = "auto" if torch.cuda.is_available() else "cpu"
        
        if is_lora:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_NAME,
                device_map=device_map,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True,
            )
            PeftModel = globals().get('PeftModel')
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
            self.model = self.model.merge_and_unload()
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map=device_map,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True,
            )
        
        self.model.eval()
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
        
        print("✅ Empathy model loaded")
        return self

    def generate_response(self, user_input: str, depression_analysis: Dict, history: Optional[List[Dict]] = None) -> str:
        symptoms_desc = "\n".join([
            f"- {s.get('topic', '').replace('_', ' ').title()}: Score {s.get('score', 0)}/3"
            for s in depression_analysis.get("detected_symptoms", [])
        ]) or "- No specific symptoms detected"
        
        severity = depression_analysis.get("severity", "Unknown")
        score = depression_analysis.get("phq8_score", 0) or 0
        
        prompt = f"""You are Ellie, an empathetic mental health counselor.

User: {user_input}

Assessment:
- Severity: {severity}
- PHQ-8 Score: {score}/24
- Symptoms:
{symptoms_desc}

Provide a warm, supportive response (2-3 sentences). Validate feelings and offer guidance.

Response:"""
        formatted_prompt = f"[INST] {prompt} [/INST]"
        
        try:
            response = self.pipeline(formatted_prompt, truncation=True)
            return response[0]['generated_text'].strip()
        except Exception as e:
            return "I'm here to listen and support you. Would you like to share more about what you're experiencing?"

class SmartPHQ8EmpathySystem:
    def __init__(self, detect_path: str, empathy_path: str, roberta_path: str):
        self.detect_path = detect_path
        self.empathy_path = empathy_path
        self.roberta_path = roberta_path
        self.detection_model = None
        self.empathy_model = None
        self.conversation_history = []
        self._initialized = False

    def initialize(self):
        if self._initialized:
            return
        
        print("\n" + "="*80)
        print("🚀 Initializing Smart PHQ-8 System")
        print("="*80)
        
        self.detection_model = PHQ8DetectionModel(self.detect_path).load()
        self.empathy_model = EmpathyResponseModel(self.empathy_path).load()
        self._initialized = True
        print("\n✅ System initialized")

    def process_user_input(self, user_input: str) -> Dict:
        if not self._initialized:
            self.initialize()
        
        depression_analysis = self.detection_model.detect_phq8(user_input)
        empathetic_response = self.empathy_model.generate_response(user_input, depression_analysis, self.conversation_history)
        
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": empathetic_response})
        
        return {
            "user_input": user_input,
            "depression_analysis": depression_analysis,
            "empathetic_response": empathetic_response,
            "final_response": empathetic_response,
            "timestamp": datetime.now().isoformat()
        }

# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="Smart PHQ-8 & Empathy System API",
    description="CBT Chatbot & PHQ-8 Depression Detection API",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global system instance (lazy loaded)
api_system: Optional[SmartPHQ8EmpathySystem] = None
_system_lock = threading.Lock()

def get_system():
    """Get or create system instance"""
    global api_system
    with _system_lock:
        if api_system is None:
            detect_path = get_default_detect_path()
            empathy_path = get_default_empathy_path()
            roberta_path = get_default_roberta_path()
            api_system = SmartPHQ8EmpathySystem(detect_path, empathy_path, roberta_path)
        return api_system

# CBT Configuration
sessions_phases = {
    1: ["start", "mood", "thoughts", "challenge", "coping", "summary"],
    2: ["mood_review", "thoughts_review", "link_feelings", "summary"],
}

# Initialize Gemini
cbt_gemini_model = None
if GEMINI_AVAILABLE:
    try:
        api_key = os.getenv("GEMINI_API_KEY", "AIzaSyDXCBFZKsNhbHGJ3rjbPRv4VnjSQmVWlA0")
        if api_key:
            genai.configure(api_key=api_key)
            cbt_gemini_model = genai.GenerativeModel("models/gemini-2.0-flash")
    except:
        pass

def build_cbt_prompt(user_input, phase, history, session_num):
    base = """You are a professional CBT counselor. Respond in 2-4 sentences.
Be empathetic, calm, supportive. Ask reflective questions."""
    tasks = {
        1: {"start": "Welcome warmly. Ask how they feel today.",
            "mood": "Ask mood rating 1-10.",
            "thoughts": "Explore thoughts/events.",
            "challenge": "Gently reframe negative thoughts.",
            "coping": "Suggest one positive action.",
            "summary": "Summarize warmly."},
    }
    task = tasks.get(session_num, {}).get(phase, "Continue supportively.")
    short_history = "\n".join(history.split("\n")[-4:]) if history else ""
    return f"""{base}

{short_history}

Goal: {task}

User: {user_input}

Counselor:"""

# Pydantic Models
class ChatRequest(BaseModel):
    user_message: str
    session_number: int = 1
    session_phase: str = "start"
    chat_history: str = ""

class UserInputRequest(BaseModel):
    user_input: str
    session_id: Optional[str] = None
    show_detection: bool = False

class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    gpu_available: bool
    cloudflare_url: Optional[str] = None

# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "Smart PHQ-8 & Empathy System API",
        "version": "3.0.0",
        "endpoints": {
            "cbt_chat": "/chat",
            "phq8_process": "/api/process",
            "load_models": "/api/load-models",
            "status": "/api/status",
            "health": "/api/health",
            "docs": "/docs"
        }
    }

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    models_loaded = False
    gpu_available = False
    
    try:
        if load_transformers_lazy():
            torch = globals().get('torch')
            gpu_available = torch and torch.cuda.is_available()
        models_loaded = api_system is not None and api_system._initialized if api_system else False
    except:
        pass
    
    return HealthResponse(
        status="healthy" if models_loaded else "ready",
        models_loaded=models_loaded,
        gpu_available=gpu_available,
        cloudflare_url=getattr(app.state, 'cloudflare_url', None)
    )

@app.post("/chat")
async def cbt_chat(req: ChatRequest):
    """CBT Chatbot - Works immediately without models"""
    try:
        prompt = build_cbt_prompt(req.user_message, req.session_phase, req.chat_history, req.session_number)
        
        if cbt_gemini_model:
            try:
                response = cbt_gemini_model.generate_content(prompt)
                reply = response.text.strip() if response and hasattr(response, "text") else "I'm here to listen."
            except:
                reply = "I'm here to support you. Could you tell me more?"
        else:
            reply = "I'm here to listen and support you."
        
        return {
            "success": True,
            "reply": reply,
            "session_number": req.session_number,
            "session_phase": req.session_phase
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/process")
async def process_user_input(request: UserInputRequest):
    """Full PHQ-8 Pipeline - Loads models on first use"""
    try:
        system = get_system()
        system.initialize()  # This will load transformers and models
        result = system.process_user_input(request.user_input)
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"PHQ-8 models not available: {str(e)}. Use /chat for CBT counseling."
        )

@app.get("/api/history")
async def get_conversation_history():
    if api_system is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    return {
        "success": True,
        "data": {
            "conversation_history": api_system.conversation_history,
            "message_count": len(api_system.conversation_history)
        }
    }

@app.post("/api/clear")
async def clear_conversation_history():
    if api_system is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    api_system.conversation_history = []
    return {"success": True, "message": "History cleared"}

@app.post("/api/load-models")
async def load_models():
    """Pre-load models and initialize GPU"""
    try:
        print("\n🔄 Pre-loading models...")
        sys.stdout.flush()
        system = get_system()
        system.initialize()
        
        # Get GPU status
        gpu_status = "Not available"
        gpu_name = None
        gpu_memory = None
        
        try:
            if load_transformers_lazy():
                torch = globals().get('torch')
                if torch and torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_memory = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
                    gpu_status = "Available"
        except:
            pass
        
        return {
            "success": True,
            "message": "Models loaded successfully",
            "gpu_status": gpu_status,
            "gpu_name": gpu_name,
            "gpu_memory": gpu_memory,
            "models_loaded": system._initialized
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load models: {str(e)}")

@app.get("/api/status")
async def get_status():
    """Get system status including GPU and models"""
    status = {
        "server": "running",
        "transformers_loaded": _transformers_loaded,
        "transformers_available": _transformers_available,
        "models_loaded": False,
        "gpu_available": False,
        "gpu_name": None,
        "gpu_memory": None
    }
    
    try:
        if load_transformers_lazy():
            torch = globals().get('torch')
            if torch and torch.cuda.is_available():
                status["gpu_available"] = True
                status["gpu_name"] = torch.cuda.get_device_name(0)
                status["gpu_memory"] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
    except:
        pass
    
    if api_system:
        status["models_loaded"] = api_system._initialized
    
    return {"success": True, "data": status}

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*80)
    print("🚀 Smart PHQ-8 System with Cloudflare Tunnel")
    print("="*80)
    
    PORT = 8000
    
    print(f"\n📁 Model Paths:")
    print(f"  Detection: {get_default_detect_path()}")
    print(f"  Empathy: {get_default_empathy_path()}")
    print(f"  RoBERTa: {get_default_roberta_path()}")
    
    # Check if we should auto-load models
    AUTO_LOAD_MODELS = os.getenv("AUTO_LOAD_MODELS", "true").lower() == "true"
    
    if AUTO_LOAD_MODELS:
        print("\n💡 Models will auto-load in background after server starts")
        print("   /chat endpoint works immediately without transformers\n")
    else:
        print("\n💡 Transformers will load lazily when /api/process is first called")
        print("   /chat endpoint works immediately without transformers\n")
    
    # Start Cloudflare Tunnel
    tunnel = CloudflareTunnel(PORT)
    public_url = tunnel.start_tunnel()
    
    if public_url:
        app.state.cloudflare_url = public_url
        print(f"\n{'='*80}")
        print("✅ SERVER READY")
        print(f"{'='*80}")
        print(f"\n🌐 Public URL: {public_url}")
        print(f"🏠 Local URL: http://localhost:{PORT}")
        print(f"\n📚 API Docs: {public_url}/docs")
        print(f"💬 CBT Chat: POST {public_url}/chat")
        print(f"🧠 PHQ-8 Process: POST {public_url}/api/process")
        print(f"❤️  Health Check: GET {public_url}/api/health")
        print(f"{'='*80}\n")
    else:
        print(f"\n🏠 Local URL: http://localhost:{PORT}")
        print(f"📚 Docs: http://localhost:{PORT}/docs")
    
    # Start server
    try:
        if IN_COLAB:
            print("🔧 Starting server in Google Colab...")
            
            server_running = threading.Event()
            server_error = None
            
            def run_server():
                nonlocal server_error
                try:
                    import nest_asyncio
                    nest_asyncio.apply()
                    
                    config = uvicorn.Config(
                        app,
                        host="0.0.0.0",
                        port=PORT,
                        log_level="warning",
                        loop="asyncio"
                    )
                    server = uvicorn.Server(config)
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    server_running.set()
                    loop.run_until_complete(server.serve())
                except Exception as e:
                    server_error = e
                    print(f"❌ Server error: {e}")
                    server_running.set()
            
            server_thread = threading.Thread(target=run_server, daemon=True)
            server_thread.start()
            
            if server_running.wait(timeout=10):
                if server_error:
                    print(f"\n❌ Server failed: {server_error}")
                else:
                    print("\n✅ Server running in background")
                    print("   Keep this cell running")
                    
                    # Auto-load models in background if enabled
                    if AUTO_LOAD_MODELS:
                        def load_models_background():
                            """Load models in background thread"""
                            time.sleep(3)  # Wait for server to be fully ready
                            print("\n" + "="*80)
                            print("🔄 Auto-loading models in background...")
                            print("="*80)
                            try:
                                system = get_system()
                                system.initialize()
                                print("\n✅ Models loaded successfully!")
                                
                                # Show GPU status
                                try:
                                    if load_transformers_lazy():
                                        torch = globals().get('torch')
                                        if torch and torch.cuda.is_available():
                                            print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
                                            print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
                                except:
                                    pass
                                
                                print("="*80 + "\n")
                            except Exception as e:
                                print(f"\n⚠️  Model loading failed: {e}")
                                print("   You can still use /chat endpoint")
                                print("   Or call POST /api/load-models to retry")
                                print("="*80 + "\n")
                        
                        model_thread = threading.Thread(target=load_models_background, daemon=True)
                        model_thread.start()
                        print("   🔄 Models will load in background...")
            else:
                print("\n⚠️  Server startup timeout")
            
            try:
                while True:
                    time.sleep(1)
                    if not server_thread.is_alive() and server_running.is_set():
                        if server_error:
                            print(f"\n⚠️  Server died: {server_error}")
                        break
            except KeyboardInterrupt:
                print("\n\n🛑 Shutting down...")
                tunnel.stop_tunnel()
        else:
            uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down...")
        tunnel.stop_tunnel()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        tunnel.stop_tunnel()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Interrupted")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        if not IN_COLAB:
            sys.exit(1)
