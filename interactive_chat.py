# -*- coding: utf-8 -*-
"""
Interactive Chat Test for Fine-tuned Mistral 7B PHQ-8 Depression Detection Model
COLAB-COMPATIBLE

This script simulates a conversation where you (the user) chat with the model,
and the model analyzes your responses for depression indicators using PHQ-8.

Usage:
    # In Colab:
    !python interactive_chat.py
    
    # Or locally:
    python interactive_chat.py

During the chat:
    - Type your responses naturally
    - Type 'exit' or 'quit' to end the conversation
    - Type 'assessment' to see current depression assessment
    - Type 'help' for more commands
"""

import os
import sys
import json
import torch
from typing import List, Dict, Optional
from datetime import datetime

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
            import subprocess
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                f"{package_name}{version}", "-q", "--upgrade"
            ])
            print(f"✓ Installed {package_name}")

# Import after checking/installing dependencies
from transformers import AutoModelForCausalLM, AutoTokenizer

# Disable bitsandbytes by default
USE_4BIT = False

# Model paths
BASE_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
MODEL_PATH = "./final_model"

# PHQ-8 Questions for reference
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


class DepressionChatBot:
    """Interactive chatbot for depression assessment"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.conversation_history = []
        self.phq8_scores = {}
        self.turn_count = 0
        
    def format_assessment_prompt(self, conversation_context: str) -> str:
        """Format prompt for depression assessment"""
        phq8_ref = json.dumps({k: v for k, v in PHQ8_QUESTIONS.items()}, indent=2)
        
        return f"""Analyze the following conversation for depression indicators using PHQ-8 assessment:
Conversation: {conversation_context}

Identify:
1. PHQ-8 symptoms present in the user's responses
2. PHQ-8 scores for each detected symptom (0-3)
3. Overall depression assessment

PHQ-8 Questions Reference:
{phq8_ref}"""
    
    def format_question_response_prompt(self, question: str, user_response: str, context: str) -> str:
        """Format prompt for analyzing a specific question-response pair"""
        phq8_ref = json.dumps({k: v for k, v in PHQ8_QUESTIONS.items()}, indent=2)
        
        return f"""Analyze the following conversation and identify:
1. Which PHQ-8 question is being asked (if any)
2. Extract the PHQ-8 answer score (0-3) from the user's response
3. Assess depression risk based on PHQ-8 score

Conversation: {context}

Interviewer: {question}
User: {user_response}

PHQ-8 Questions Reference:
{phq8_ref}"""
    
    def generate_response(self, prompt: str, max_length: int = 512, temperature: float = 0.7) -> str:
        """Generate response from the model"""
        formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
        
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "### Response:\n" in generated_text:
            response = generated_text.split("### Response:\n")[-1].strip()
        else:
            response = generated_text[len(formatted_prompt):].strip()
        
        return response
    
    def get_assessment(self) -> Dict:
        """Get current depression assessment"""
        if not self.conversation_history:
            return {"message": "No conversation yet. Start chatting!"}
        
        # Build conversation context
        context = "\n".join([
            f"{turn['speaker']}: {turn['text']}"
            for turn in self.conversation_history[-10:]  # Last 10 turns
        ])
        
        prompt = self.format_assessment_prompt(context)
        assessment = self.generate_response(prompt, max_length=400)
        
        return {
            "assessment": assessment,
            "turns": len(self.conversation_history),
            "conversation": context
        }
    
    def analyze_response(self, question: str, user_response: str) -> Dict:
        """Analyze a user response to a question"""
        # Add to conversation history
        self.conversation_history.append({
            "speaker": "Interviewer",
            "text": question,
            "turn": self.turn_count
        })
        self.conversation_history.append({
            "speaker": "User",
            "text": user_response,
            "turn": self.turn_count + 1
        })
        self.turn_count += 2
        
        # Build context
        context = "\n".join([
            f"{turn['speaker']}: {turn['text']}"
            for turn in self.conversation_history[-6:]  # Last 6 turns
        ])
        
        # Analyze
        prompt = self.format_question_response_prompt(question, user_response, context)
        analysis = self.generate_response(prompt, max_length=400)
        
        return {
            "analysis": analysis,
            "question": question,
            "response": user_response
        }
    
    def save_conversation(self, filename: str = "chat_history.json"):
        """Save conversation history"""
        data = {
            "timestamp": datetime.now().isoformat(),
            "conversation": self.conversation_history,
            "total_turns": len(self.conversation_history)
        }
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Conversation saved to {filename}")


def load_model():
    """Load the fine-tuned model"""
    print("=" * 80)
    print("Loading Fine-tuned Model for Interactive Chat")
    print("=" * 80)
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model directory not found: {MODEL_PATH}\n"
            "Please ensure the model has been trained and saved to ./final_model"
        )
    
    print(f"\n1. Loading tokenizer from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    print(f"2. Loading base model: {BASE_MODEL_NAME}...")
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    print(f"   Device: {device_map}")
    
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        device_map=device_map,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    
    print("3. Loading LoRA adapter...")
    from peft import PeftModel
    model = PeftModel.from_pretrained(base_model, MODEL_PATH)
    model = model.merge_and_unload()
    
    print("✓ Model loaded successfully!\n")
    
    return model, tokenizer


def print_help():
    """Print help message"""
    print("\n" + "=" * 80)
    print("COMMANDS:")
    print("=" * 80)
    print("  Type your response naturally - the model will analyze it")
    print("  'assessment' - Get current depression assessment")
    print("  'help'       - Show this help message")
    print("  'save'       - Save conversation history")
    print("  'exit'/'quit' - End the conversation")
    print("=" * 80 + "\n")


def main():
    """Main interactive chat function"""
    print("=" * 80)
    print("Interactive Depression Assessment Chat")
    print("=" * 80)
    print("\nThis is an interactive chat to test the fine-tuned model.")
    print("You are the user, and I (the model) will ask questions and analyze your responses.")
    print("\nType 'help' for commands, 'exit' to quit.\n")
    
    # Load model
    try:
        model, tokenizer = load_model()
        chatbot = DepressionChatBot(model, tokenizer)
    except Exception as e:
        print(f"\n❌ Failed to load model: {e}")
        return
    
    # Sample questions to ask
    sample_questions = [
        "Hi! How have you been feeling lately?",
        "Over the last 2 weeks, have you been bothered by little interest or pleasure in doing things?",
        "How have you been sleeping?",
        "Have you been feeling tired or having little energy?",
        "How's your appetite been?",
        "Have you been feeling down, depressed, or hopeless?",
        "How's your ability to concentrate been?",
        "Have you been feeling bad about yourself recently?",
    ]
    
    question_index = 0
    
    print("\n" + "=" * 80)
    print("CONVERSATION STARTED")
    print("=" * 80 + "\n")
    
    # Start conversation
    print("🤖 Model: Hello! I'm here to help assess how you've been feeling. Let's start with a simple question.\n")
    
    # Ask first question
    current_question = sample_questions[0]
    print(f"🤖 Model: {current_question}\n")
    
    while True:
        try:
            # Get user input
            user_input = input("👤 You: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("\n🤖 Model: Thank you for chatting. Take care!\n")
                break
            
            elif user_input.lower() == 'help':
                print_help()
                continue
            
            elif user_input.lower() == 'assessment':
                print("\n🤖 Model: Let me analyze our conversation so far...\n")
                assessment = chatbot.get_assessment()
                print("=" * 80)
                print("DEPRESSION ASSESSMENT:")
                print("=" * 80)
                print(assessment.get("assessment", "Assessment unavailable"))
                print("=" * 80 + "\n")
                continue
            
            elif user_input.lower() == 'save':
                chatbot.save_conversation()
                continue
            
            # Analyze user response
            print("\n🤖 Model: Analyzing your response...\n")
            analysis = chatbot.analyze_response(current_question, user_input)
            
            print("=" * 80)
            print("ANALYSIS:")
            print("=" * 80)
            print(analysis["analysis"])
            print("=" * 80 + "\n")
            
            # Ask next question
            question_index += 1
            if question_index < len(sample_questions):
                current_question = sample_questions[question_index]
                print(f"🤖 Model: {current_question}\n")
            else:
                # Get final assessment
                print("🤖 Model: Thank you for answering those questions. Let me provide a final assessment...\n")
                final_assessment = chatbot.get_assessment()
                print("=" * 80)
                print("FINAL DEPRESSION ASSESSMENT:")
                print("=" * 80)
                print(final_assessment.get("assessment", "Assessment unavailable"))
                print("=" * 80 + "\n")
                
                # Ask if they want to continue
                continue_chat = input("Would you like to continue chatting? (yes/no): ").strip().lower()
                if continue_chat not in ['yes', 'y']:
                    print("\n🤖 Model: Thank you for chatting. Take care!\n")
                    break
                else:
                    # Reset question index for more questions
                    question_index = 0
                    current_question = sample_questions[0]
                    print(f"\n🤖 Model: {current_question}\n")
            
        except KeyboardInterrupt:
            print("\n\n🤖 Model: Conversation interrupted. Goodbye!\n")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Continuing conversation...\n")
    
    # Save conversation
    try:
        save = input("\n💾 Save conversation history? (yes/no): ").strip().lower()
        if save in ['yes', 'y']:
            chatbot.save_conversation()
    except:
        pass
    
    print("\n" + "=" * 80)
    print("Chat session ended")
    print("=" * 80)


if __name__ == "__main__":
    main()

