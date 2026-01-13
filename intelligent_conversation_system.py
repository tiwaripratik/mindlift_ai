# -*- coding: utf-8 -*-
"""
Intelligent Conversation System - Depression Detection + Empathetic Response Generation

This script connects two fine-tuned models:
1. Depression Detection Model (from ./detect folder) - Analyzes user input for PHQ-8 depression indicators
2. Empathetic Response Model (from ./empathy folder) - Generates empathetic responses based on depression level

Workflow:
1. User provides input/response
2. Detection model analyzes input for depression symptoms (PHQ-8)
3. System determines depression severity level
4. Empathy model generates empathetic response tailored to depression level and user input
5. Returns both detection analysis and empathetic response

Usage:
    python intelligent_conversation_system.py
    
    # Interactive mode:
    python intelligent_conversation_system.py --interactive
    
    # Single query:
    python intelligent_conversation_system.py --input "I've been feeling really down lately"
"""

import os
import sys
import json
import argparse
import torch
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Utilities
def ensure_model_directory(model_path: str, friendly_name: str) -> str:
    """Ensure model directory exists; if not, try to auto-extract from zip in common locations.

    Looks for archives like:
      - detect.zip / empathy.zip
      - {friendly_name}.zip
      - model.zip variants uploaded to Colab (/content)

    Returns the resolved model directory path (may be under /content after extraction).
    """
    if os.path.exists(model_path) and os.path.isdir(model_path):
        return model_path

    # Define model-specific zip names
    model_zip_mappings = {
        "detect": [
            "detect.zip",
            "final_model (1).zip",
            "final_model(1).zip",
            "final_model_1.zip",
            "final_model.zip",
        ],
        "empathy": [
            "empathy.zip",
            "final_model_empathetic.zip",
            "final_model_empathy.zip",
            "empathetic.zip",
        ],
    }
    
    # Get model-specific zip names, plus generic ones
    possible_zip_names = model_zip_mappings.get(friendly_name.lower(), [])
    possible_zip_names.extend([
        f"{friendly_name}.zip",
        f"{os.path.basename(model_path)}.zip",
        "model.zip",
        "adapter.zip",
    ])
    
    # Remove duplicates while preserving order
    seen = set()
    unique_zip_names = []
    for name in possible_zip_names:
        if name not in seen:
            seen.add(name)
            unique_zip_names.append(name)
    possible_zip_names = unique_zip_names

    # Search locations - prioritize Colab content directory
    possible_locations = [
        "/content" if os.path.exists("/content") else None,  # Colab content directory first
        ".",
        os.getcwd(),
        "/content/drive/MyDrive" if os.path.exists("/content/drive/MyDrive") else None,
    ]
    possible_locations = [p for p in possible_locations if p]
    
    # Also check current directory and subdirectories more thoroughly
    if IN_COLAB:
        # In Colab, files are often in /content
        if "/content" not in possible_locations and os.path.exists("/content"):
            possible_locations.insert(0, "/content")

    # First, check if model directory already exists (maybe it was extracted before)
    if os.path.exists(model_path) and os.path.isdir(model_path):
        if os.path.exists(os.path.join(model_path, "adapter_config.json")):
            print(f"   ✓ Model directory already exists: {model_path}")
            return model_path
    
    # Check if there's a directory with the same name (without .zip extension)
    # Sometimes files get uploaded as directories or extracted already
    for base in possible_locations:
        # Check for directory with exact name
        dir_path = os.path.join(base, friendly_name)
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            if os.path.exists(os.path.join(dir_path, "adapter_config.json")):
                print(f"   ✓ Found model directory: {dir_path}")
                return dir_path
        
        # Check if zip file is actually a directory (sometimes happens in Colab)
        zip_path = os.path.join(base, f"{friendly_name}.zip")
        if os.path.exists(zip_path) and os.path.isdir(zip_path):
            print(f"   → {zip_path} is actually a directory, not a ZIP file")
            if os.path.exists(os.path.join(zip_path, "adapter_config.json")):
                print(f"   ✓ Using directory directly: {zip_path}")
                return zip_path
    
    # Try to find and extract a zip archive
    print(f"\n   Searching for {friendly_name} model archive...")
    print(f"   Looking for: {', '.join(possible_zip_names[:5])}...")
    
    def check_file_type(file_path):
        """Check what type of file this is"""
        if not os.path.exists(file_path):
            return None
        if os.path.isdir(file_path):
            return "directory"
        
        # Check file signature/magic bytes
        try:
            with open(file_path, 'rb') as f:
                header = f.read(4)
                if header[:2] == b'PK':
                    return "zip"
                elif header[:2] == b'\x1f\x8b':  # gzip
                    return "gzip"
                elif header == b'BZh9':  # bzip2
                    return "bzip2"
                elif header[:2] == b'\x1f\x8b' or header == b'ustar':  # tar
                    return "tar"
                else:
                    # Check if it's a tar.gz by reading more
                    f.seek(0)
                    header = f.read(10)
                    if header[:2] == b'\x1f\x8b':
                        return "tar.gz"
        except Exception:
            pass
        
        # Check extension
        if file_path.endswith('.zip'):
            return "zip_extension"
        elif file_path.endswith('.tar.gz') or file_path.endswith('.tgz'):
            return "tar.gz"
        elif file_path.endswith('.tar'):
            return "tar"
        
        return "unknown"
    
    def extract_archive(archive_path, extract_dir):
        """Extract archive using appropriate method"""
        file_type = check_file_type(archive_path)
        
        # Double-check if it's actually a directory (sometimes os.path.isdir fails)
        try:
            if os.path.isdir(archive_path):
                print(f"   ⚠️  File is actually a directory, not an archive")
                # Check if it's already a model directory
                if os.path.exists(os.path.join(archive_path, "adapter_config.json")):
                    print(f"   ✓ Directory contains model files, using directly")
                    return archive_path
                return False
        except Exception:
            pass
        
        # Check file size - if it's 0 or very small, it's likely corrupted
        try:
            file_size = os.path.getsize(archive_path)
            if file_size == 0:
                print(f"   ⚠️  File is empty (0 bytes)")
                return False
            if file_size < 100:  # Very small files are likely not valid archives
                print(f"   ⚠️  File is too small ({file_size} bytes) to be a valid archive")
                return False
        except Exception:
            pass
        
        if file_type == "zip" or file_type == "zip_extension":
            try:
                import zipfile
                # Try different zipfile modes for corrupted files
                zip_modes = [
                    ('r', 'standard read'),
                    ('r', 'with allowZip64'),
                ]
                
                for mode, desc in zip_modes:
                    try:
                        # Try standard mode first
                        if 'allowZip64' in desc:
                            zf = zipfile.ZipFile(archive_path, 'r', allowZip64=True)
                        else:
                            zf = zipfile.ZipFile(archive_path, 'r')
                        
                        # Test if zip is valid by trying to read file list
                        file_list = zf.namelist()
                        if not file_list:
                            zf.close()
                            print(f"   ⚠️  ZIP archive is empty (no files inside)")
                            return False
                        print(f"   → Archive contains {len(file_list)} files ({desc})")
                        # Extract
                        zf.extractall(extract_dir)
                        zf.close()
                        return True
                    except zipfile.BadZipFile:
                        if 'allowZip64' not in desc:
                            continue  # Try next mode
                        raise
                    except Exception as e:
                        if 'allowZip64' not in desc:
                            continue  # Try next mode
                        raise
                
                # If we get here, all modes failed
                raise zipfile.BadZipFile("All extraction modes failed")
            except zipfile.BadZipFile as e:
                print(f"   ⚠️  File is not a valid ZIP archive: {e}")
                
                # Try alternative extraction methods for corrupted ZIPs
                print(f"   → Attempting alternative extraction methods...")
                
                # Method 1: Try using 7z or unzip command line tools if available
                try:
                    import subprocess
                    extract_cmd = None
                    
                    # Try unzip command
                    try:
                        result = subprocess.run(
                            ['unzip', '-t', archive_path],
                            capture_output=True,
                            timeout=10
                        )
                        if result.returncode == 0:
                            print(f"   → unzip command can read the file, trying extraction...")
                            subprocess.run(
                                ['unzip', '-o', archive_path, '-d', extract_dir],
                                check=True,
                                timeout=60
                            )
                            print(f"   ✓ Extracted using unzip command")
                            return True
                    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                        pass
                    
                    # Try 7z command
                    try:
                        result = subprocess.run(
                            ['7z', 't', archive_path],
                            capture_output=True,
                            timeout=10
                        )
                        if result.returncode == 0:
                            print(f"   → 7z command can read the file, trying extraction...")
                            subprocess.run(
                                ['7z', 'x', archive_path, f'-o{extract_dir}', '-y'],
                                check=True,
                                timeout=60
                            )
                            print(f"   ✓ Extracted using 7z command")
                            return True
                    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                        pass
                        
                except Exception as ex:
                    print(f"   → Alternative extraction methods failed: {ex}")
                
                # Method 2: Check if it might be a directory with .zip extension
                try:
                    dir_candidates = [
                        archive_path.rstrip('.zip'),
                        archive_path.replace('.zip', ''),
                        os.path.join(os.path.dirname(archive_path), friendly_name),
                    ]
                    for alt_path in dir_candidates:
                        if os.path.exists(alt_path) and os.path.isdir(alt_path):
                            if os.path.exists(os.path.join(alt_path, "adapter_config.json")):
                                print(f"   → Found matching directory: {alt_path}")
                                return alt_path
                except Exception:
                    pass
                
                return False
            except Exception as e:
                print(f"   ⚠️  Error extracting ZIP: {e}")
                return False
        
        elif file_type == "tar.gz" or file_type == "gzip":
            try:
                import tarfile
                with tarfile.open(archive_path, 'r:gz') as tf:
                    tf.extractall(extract_dir)
                return True
            except Exception as e:
                print(f"   ⚠️  Error extracting tar.gz: {e}")
                return False
        
        elif file_type == "tar":
            try:
                import tarfile
                with tarfile.open(archive_path, 'r') as tf:
                    tf.extractall(extract_dir)
                return True
            except Exception as e:
                print(f"   ⚠️  Error extracting tar: {e}")
                return False
        
        else:
            print(f"   ⚠️  Unknown file type: {file_type}")
            print(f"   File might be corrupted or in unsupported format")
            return False
    
    for base in possible_locations:
        # First, try exact name matches
        for zip_name in possible_zip_names:
            zip_path = os.path.join(base, zip_name)
            if os.path.exists(zip_path):
                file_type = check_file_type(zip_path)
                print(f"   Found potential archive: {zip_path} (type: {file_type})")
                
                if file_type == "directory":
                    # It's already a directory, check if it's the model directory
                    if os.path.exists(os.path.join(zip_path, "adapter_config.json")):
                        print(f"   ✓ Using directory as model: {zip_path}")
                        return zip_path
                    continue
                
                print("   Extracting...")
                extract_dir = base if base != "." else os.getcwd()
                
                if extract_archive(zip_path, extract_dir):
                    print(f"   ✓ Extracted successfully")
                    
                    # After extraction, try to locate the model folder
                    candidates = []
                    for entry in os.listdir(extract_dir):
                        full = os.path.join(extract_dir, entry)
                        if os.path.isdir(full):
                            # Heuristics: likely model dir contains tokenizer files or adapter_config.json
                            if os.path.exists(os.path.join(full, "adapter_config.json")) or \
                               os.path.exists(os.path.join(full, "tokenizer.json")) or \
                               os.path.exists(os.path.join(full, "tokenizer.model")):
                                candidates.append(full)
                    
                    # Prefer exact folder name if present
                    for cand in candidates:
                        if os.path.basename(cand).lower() in [os.path.basename(model_path).lower(), friendly_name.lower()]:
                            print(f"   ✓ Using extracted model directory: {cand}")
                            return cand
                    if candidates:
                        print(f"   ✓ Using extracted model directory: {candidates[0]}")
                        return candidates[0]
                    # If nothing matched, but intended path now exists, use it
                    if os.path.exists(model_path) and os.path.isdir(model_path):
                        print(f"   ✓ Using model directory: {model_path}")
                        return model_path
        
        # Also search for any archive files that might contain the model
        # Include model-specific patterns
        model_patterns = {
            "detect": ["final_model", "detect"],
            "empathy": ["final_model_empathetic", "empathetic", "empathy"],
        }
        search_patterns = model_patterns.get(friendly_name.lower(), [friendly_name])
        
        try:
            if os.path.exists(base):
                for entry in os.listdir(base):
                    # Check for various archive formats
                    entry_lower = entry.lower()
                    matches_pattern = any(pattern.lower() in entry_lower for pattern in search_patterns)
                    is_archive = entry.endswith(('.zip', '.tar.gz', '.tgz', '.tar'))
                    
                    if is_archive and matches_pattern:
                        zip_path = os.path.join(base, entry)
                        if os.path.isfile(zip_path) and entry not in possible_zip_names:
                            file_type = check_file_type(zip_path)
                            print(f"   Found potential archive: {zip_path} (type: {file_type})")
                            print("   Extracting...")
                            
                            extract_dir = base if base != "." else os.getcwd()
                            if extract_archive(zip_path, extract_dir):
                                print(f"   ✓ Extracted successfully")
                                
                                # Check if model directory now exists
                                if os.path.exists(model_path) and os.path.isdir(model_path):
                                    print(f"   ✓ Using model directory: {model_path}")
                                    return model_path
                                
                                # Look for model directories
                                for entry in os.listdir(extract_dir):
                                    full = os.path.join(extract_dir, entry)
                                    if os.path.isdir(full):
                                        if os.path.exists(os.path.join(full, "adapter_config.json")):
                                            print(f"   ✓ Using extracted model directory: {full}")
                                            return full
        except Exception:
            pass  # Skip if can't list directory

    # If still missing, provide helpful error with available files
    print(f"\n   ⚠️  Model directory not found: {model_path}")
    print(f"   Searched for archives: {', '.join(possible_zip_names)}")
    
    # List available zip files and check their status
    available_zips = []
    zip_status = {}
    for base in possible_locations:
        try:
            if os.path.exists(base):
                for entry in os.listdir(base):
                    if entry.endswith('.zip') or friendly_name.lower() in entry.lower():
                        zip_path = os.path.join(base, entry)
                        file_type = check_file_type(zip_path)
                        available_zips.append(zip_path)
                        zip_status[zip_path] = file_type
        except Exception:
            pass
    
    if available_zips:
        print(f"\n   Files found (checking status):")
        for zip_file in available_zips[:10]:  # Show first 10
            file_type = zip_status.get(zip_file, "unknown")
            
            # Get file size for diagnostics
            try:
                file_size = os.path.getsize(zip_file)
                size_mb = file_size / (1024 * 1024)
                size_str = f"{size_mb:.2f} MB" if size_mb > 1 else f"{file_size / 1024:.2f} KB"
            except Exception:
                size_str = "unknown size"
            
            # Check if it's a valid zip
            is_valid = False
            if file_type in ["zip", "zip_extension"]:
                try:
                    import zipfile
                    with zipfile.ZipFile(zip_file, 'r') as zf:
                        zf.testzip()  # Test zip integrity
                        file_count = len(zf.namelist())
                    is_valid = True
                    status_icon = "✓"
                    print(f"     {status_icon} {zip_file} (type: {file_type}, size: {size_str}, files: {file_count})")
                except zipfile.BadZipFile:
                    status_icon = "⚠️"
                    print(f"     {status_icon} {zip_file} (type: {file_type}, size: {size_str}) - CORRUPTED or INVALID ZIP")
                    print(f"        → This file appears corrupted. Please re-upload a valid archive.")
                except Exception as e:
                    status_icon = "⚠️"
                    print(f"     {status_icon} {zip_file} (type: {file_type}, size: {size_str}) - Error: {e}")
            else:
                status_icon = "✓" if file_type in ["tar.gz", "tar"] else "⚠️"
                print(f"     {status_icon} {zip_file} (type: {file_type}, size: {size_str})")
            
            # If it's a directory, suggest renaming
            if file_type == "directory":
                print(f"        → This appears to be a directory, not an archive file")
                print(f"        → Try: import shutil; shutil.make_archive('{friendly_name}', 'zip', '{zip_file}')")
        
        if len(available_zips) > 10:
            print(f"     ... and {len(available_zips) - 10} more")
    
    # Check if model directory might already exist with different name
    print(f"\n   Checking if model directory exists with different name...")
    found_dirs = []
    for base in possible_locations:
        try:
            if os.path.exists(base):
                for entry in os.listdir(base):
                    full_path = os.path.join(base, entry)
                    if os.path.isdir(full_path):
                        if os.path.exists(os.path.join(full_path, "adapter_config.json")):
                            # Check if this directory name matches our model
                            if friendly_name.lower() in entry.lower() or \
                               os.path.basename(model_path).lower() in entry.lower():
                                print(f"   ✓ Found matching model directory: {full_path}")
                                print(f"   → Using this directory directly")
                                return full_path
                            found_dirs.append(full_path)
        except Exception:
            pass
    
    if found_dirs:
        print(f"\n   Found {len(found_dirs)} model directory(ies) but name doesn't match:")
        for dir_path in found_dirs[:3]:
            print(f"     - {dir_path}")
        print(f"   → You can use: --{friendly_name}-path {found_dirs[0]}")
    
    # Provide detailed help for fixing corrupted files
    help_message = f"""
Model directory not found: {model_path}
Model name: {friendly_name}

Troubleshooting Corrupted ZIP Files:
====================================

The ZIP files appear to be corrupted or invalid. Here's how to fix this:

OPTION 1: Check if files exist and list them
---------------------------------------------
Run this in Colab to see what files are available:
   !ls -lh /content/*.zip
   !ls -lh ./*.zip
   !find /content -name "*final_model*" -o -name "*empathetic*" 2>/dev/null | head -20
   
If files exist but have different names, you can:
   - Rename them: !mv "/content/final_model (1).zip" "/content/detect.zip"
   - Or specify path: --detection-path /content/final_model\ \(1\).zip

OPTION 2: Extract using command line tools (If files are corrupted)
--------------------------------------------------------------------
If ZIP files are corrupted, try extracting manually in Colab:

   !unzip -o "/content/final_model (1).zip" -d ./detect
   !unzip -o "/content/final_model_empathetic.zip" -d ./empathy
   
Or try 7z if available:
   !7z x "/content/final_model (1).zip" -o./detect -y
   !7z x "/content/final_model_empathetic.zip" -o./empathy -y

OPTION 3: Use Google Drive (More Reliable)
-------------------------------------------
1. Upload ZIP files to Google Drive
2. Mount Drive in Colab:
   from google.colab import drive
   drive.mount('/content/drive')
   
3. Copy files:
   !cp /content/drive/MyDrive/detect.zip /content/
   !cp /content/drive/MyDrive/empathy.zip /content/

OPTION 4: Direct Directory Path (If folders exist)
---------------------------------------------------
If model folders already exist (not zipped), specify the path:
  --detection-path /path/to/detect/folder
  --empathy-path /path/to/empathy/folder

Current Status:
- Files found but appear corrupted
- File sizes: 5.00 MB (reasonable size, but structure is invalid)
- Suggestion: Re-upload files or use Google Drive for more reliable transfer
"""
    
    raise FileNotFoundError(help_message)

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
    ]
    
    for package_name, version in required_packages:
        try:
            __import__(package_name)
        except ImportError:
            print(f"Installing {package_name}{version}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                f"{package_name}{version}", "-q", "--upgrade"
            ])

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    logging,
)
from peft import PeftModel

# Suppress warnings
logging.set_verbosity(logging.ERROR)

# Model paths
BASE_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
DETECTION_MODEL_PATH = "./detect"
EMPATHY_MODEL_PATH = "./empathy"

# PHQ-8 Questions Reference (for detection prompts)
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


class DepressionDetectionModel:
    """Wrapper for the depression detection model"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        
    def load(self):
        """Load the depression detection model"""
        print(f"\n{'='*80}")
        print("Loading Depression Detection Model")
        print(f"{'='*80}")

        # Ensure model directory exists (auto-extract zip if needed)
        self.model_path = ensure_model_directory(self.model_path, friendly_name="detect")
        
        print(f"1. Loading base model: {BASE_MODEL_NAME}")
        print(f"2. Loading detection adapter from: {self.model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, 
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        # Load base model
        device_map = "auto" if torch.cuda.is_available() else "cpu"
        print(f"3. Device: {device_map}")
        
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            device_map=device_map,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        )
        
        # Load LoRA adapter
        print("4. Loading LoRA adapter...")
        self.model = PeftModel.from_pretrained(base_model, self.model_path)
        self.model = self.model.merge_and_unload()
        self.model.eval()
        
        # Create pipeline for easier inference
        self.pipeline = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=400,
            temperature=0.3,
            do_sample=True,
            return_full_text=False,
        )
        
        print("   ✓ Detection model loaded successfully!")
        return self
    
    def detect_depression(self, user_input: str, conversation_context: Optional[str] = None) -> Dict:
        """
        Analyze user input for depression indicators using PHQ-8 assessment
        
        Args:
            user_input: Current user message/input
            conversation_context: Optional previous conversation context
            
        Returns:
            Dictionary with depression assessment including severity, PHQ-8 score, symptoms
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
            print(f"⚠️ Error during depression detection: {e}")
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
        import re
        
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


class EmpathyModel:
    """Wrapper for the empathetic response generation model"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        
    def load(self):
        """Load the empathetic response model"""
        print(f"\n{'='*80}")
        print("Loading Empathetic Response Model")
        print(f"{'='*80}")

        # Ensure model directory exists (auto-extract zip if needed)
        self.model_path = ensure_model_directory(self.model_path, friendly_name="empathy")
        
        print(f"1. Loading base model: {BASE_MODEL_NAME}")
        print(f"2. Loading empathy adapter from: {self.model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        # Load base model
        device_map = "auto" if torch.cuda.is_available() else "cpu"
        print(f"3. Device: {device_map}")
        
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            device_map=device_map,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        )
        
        # Load LoRA adapter
        print("4. Loading LoRA adapter...")
        self.model = PeftModel.from_pretrained(base_model, self.model_path)
        self.model = self.model.merge_and_unload()
        self.model.eval()
        
        # Create pipeline
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
        
        print("   ✓ Empathy model loaded successfully!")
        return self
    
    def generate_response(
        self, 
        user_input: str, 
        depression_analysis: Dict,
        conversation_history: Optional[List[Dict]] = None
    ) -> str:
        """
        Generate empathetic response based on user input and depression analysis
        
        Args:
            user_input: Current user message
            depression_analysis: Depression detection analysis from detection model
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


class IntelligentConversationSystem:
    """Main system that connects both models"""
    
    def __init__(self):
        self.detection_model = None
        self.empathy_model = None
        self.conversation_history = []
        
    def initialize(self):
        """Load both models"""
        print("\n" + "="*80)
        print("Initializing Intelligent Conversation System")
        print("="*80)
        
        # Load detection model
        self.detection_model = DepressionDetectionModel(DETECTION_MODEL_PATH).load()
        
        # Load empathy model
        self.empathy_model = EmpathyModel(EMPATHY_MODEL_PATH).load()
        
        print("\n" + "="*80)
        print("✓ System Initialized Successfully!")
        print("="*80)
        print(f"  - Detection Model: Loaded from {DETECTION_MODEL_PATH}")
        print(f"  - Empathy Model: Loaded from {EMPATHY_MODEL_PATH}")
        
        if torch.cuda.is_available():
            print(f"  - GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
        else:
            print(f"  - Device: CPU")
        
    def process_user_input(self, user_input: str, show_detection: bool = True) -> Dict:
        """
        Process user input through both models
        
        Args:
            user_input: User's message/input
            show_detection: Whether to show detection analysis details
            
        Returns:
            Dictionary with detection analysis and empathetic response
        """
        if not self.detection_model or not self.empathy_model:
            raise RuntimeError("Models not initialized. Call initialize() first.")
        
        print("\n" + "="*80)
        print("Processing User Input")
        print("="*80)
        print(f"\nUser: {user_input}\n")
        
        # Step 1: Detect depression
        print("[Step 1] Analyzing depression indicators...")
        depression_analysis = self.detection_model.detect_depression(
            user_input,
            self._format_conversation_history()
        )
        
        if show_detection:
            print(f"\n  ✓ Severity: {depression_analysis.get('severity', 'Unknown')}")
            print(f"  ✓ PHQ-8 Score: {depression_analysis.get('phq8_score', 'N/A')}/24")
            if depression_analysis.get('detected_symptoms'):
                print(f"  ✓ Detected Symptoms: {len(depression_analysis['detected_symptoms'])}")
        
        # Step 2: Generate empathetic response
        print("\n[Step 2] Generating empathetic response...")
        empathetic_response = self.empathy_model.generate_response(
            user_input,
            depression_analysis,
            self.conversation_history
        )
        
        # Update conversation history
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": empathetic_response})
        
        # Prepare result
        result = {
            "user_input": user_input,
            "depression_analysis": depression_analysis,
            "empathetic_response": empathetic_response,
            "timestamp": datetime.now().isoformat()
        }
        
        print("\n" + "="*80)
        print("Response")
        print("="*80)
        print(f"\nEllie: {empathetic_response}\n")
        
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


def run_interactive_mode(system: IntelligentConversationSystem):
    """Run interactive conversation mode"""
    print("\n" + "="*80)
    print("Interactive Conversation Mode")
    print("="*80)
    print("\nYou're now chatting with Ellie, an empathetic mental health counselor.")
    print("Type 'quit', 'exit', or 'bye' to end the conversation.")
    print("Type 'history' to see conversation history.")
    print("Type 'clear' to clear conversation history.\n")
    
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
            
            # Process input
            result = system.process_user_input(user_input, show_detection=False)
            
        except KeyboardInterrupt:
            print("\n\nEllie: I understand. Feel free to come back anytime you want to talk. Take care. ❤️\n")
            break
        except Exception as e:
            print(f"\n⚠️ Error: {e}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Intelligent Conversation System - Depression Detection + Empathetic Response"
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
        '--detection-path',
        type=str,
        default="./detect",
        help='Path to detection model (default: ./detect)'
    )
    parser.add_argument(
        '--empathy-path',
        type=str,
        default="./empathy",
        help='Path to empathy model (default: ./empathy)'
    )
    
    # In Colab, Jupyter kernel passes extra arguments that we need to ignore
    # Parse arguments, ignoring unknown ones that might come from kernel launcher
    if IN_COLAB:
        args, unknown = parser.parse_known_args()
        if unknown:
            print(f"Note: Ignoring unknown arguments from kernel launcher: {unknown}")
    else:
        args = parser.parse_args()
    
    # Get model paths and ensure they exist (auto-extract zip if needed)
    detection_path = args.detection_path
    empathy_path = args.empathy_path
    
    # Ensure model directories exist (auto-extract zip if needed)
    try:
        detection_path = ensure_model_directory(detection_path, friendly_name="detect")
        empathy_path = ensure_model_directory(empathy_path, friendly_name="empathy")
    except FileNotFoundError as e:
        print("\n" + "="*80)
        print("❌ MODEL SETUP ERROR")
        print("="*80)
        print(str(e))
        print("\n" + "="*80)
        
        if IN_COLAB:
            print("\n📋 QUICK SETUP FOR COLAB:")
            print("="*80)
            print("Run these commands in separate cells:")
            print("\n1. Upload detect.zip:")
            print("   from google.colab import files")
            print("   files.upload()  # Select detect.zip")
            print("\n2. Upload empathy.zip:")
            print("   from google.colab import files")
            print("   files.upload()  # Select empathy.zip")
            print("\n3. Then run this script again:")
            print("   !python intelligent_conversation_system.py --interactive")
            print("="*80)
        
        # In IPython/Colab, return instead of sys.exit to avoid traceback issues
        if IN_COLAB:
            return
        else:
            sys.exit(1)
    
    # Update global paths
    global DETECTION_MODEL_PATH, EMPATHY_MODEL_PATH
    DETECTION_MODEL_PATH = detection_path
    EMPATHY_MODEL_PATH = empathy_path
    
    # Initialize system
    system = IntelligentConversationSystem()
    
    try:
        system.initialize()
    except Exception as e:
        print(f"\n❌ Error initializing system: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure detection model is in ./detect folder")
        print("2. Ensure empathy model is in ./empathy folder")
        print("3. Verify models are trained and saved correctly")
        sys.exit(1)
    
    # Run based on mode
    if args.interactive:
        run_interactive_mode(system)
    elif args.input:
        result = system.process_user_input(args.input)
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\n✓ Results saved to {args.output}")
    else:
        # Default: interactive mode
        run_interactive_mode(system)


if __name__ == "__main__":
    main()

