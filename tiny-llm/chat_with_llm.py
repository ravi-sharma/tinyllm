#!/usr/bin/env python3
"""
Interactive Chat with Tiny LLM
Chat with your trained tiny transformer model
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tiny_llm import TinyLLMService
import requests

def get_api_key():
    """Get the API key from the observability platform"""
    try:
        response = requests.get("http://localhost:8001/api/v1/projects")
        if response.status_code == 200:
            projects = response.json()
            if projects:
                return projects[0]['api_key']
    except Exception as e:
        print(f"⚠️ Could not get API key: {e}")
    return None

def main():
    print("🤖 Tiny LLM Interactive Chat")
    print("=" * 40)
    
    model_path = "models/tiny_llm_trained.pt"
    
    # Check if trained model exists
    if not os.path.exists(model_path):
        print(f"❌ Trained model not found at {model_path}")
        print("Please run train_tiny_llm.py first to train the model!")
        return
    
    # Get API key for monitoring
    api_key = get_api_key()
    if api_key:
        print(f"✅ Connected to observability platform")
        print("📊 All conversations will be monitored and logged")
    else:
        print("⚠️ Running without observability monitoring")
    
    # Load the trained model
    print(f"🔄 Loading trained model from {model_path}...")
    service = TinyLLMService(model_path=model_path, observability_api_key=api_key)
    print("✅ Model loaded successfully!")
    
    print("\n" + "=" * 40)
    print("💬 Start chatting with your Tiny LLM!")
    print("Commands:")
    print("  - Type your message and press Enter")
    print("  - Type 'quit' or 'exit' to end the chat")
    print("  - Type 'help' for generation settings")
    print("=" * 40)
    
    # Generation settings
    max_length = 50
    temperature = 0.8
    top_k = 50
    
    while True:
        try:
            # Get user input
            user_input = input("\n🧑 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye! Thanks for chatting with Tiny LLM!")
                break
            
            if user_input.lower() == 'help':
                print(f"\n⚙️ Current settings:")
                print(f"   Max length: {max_length} tokens")
                print(f"   Temperature: {temperature}")
                print(f"   Top-k: {top_k}")
                print("\n💡 Tips:")
                print("   - Lower temperature (0.3-0.7) = more focused responses")
                print("   - Higher temperature (0.8-1.2) = more creative responses")
                print("   - Try short prompts for better results")
                continue
            
            if not user_input:
                print("Please enter a message or 'quit' to exit.")
                continue
            
            # Generate response
            print("🤖 Tiny LLM: ", end="", flush=True)
            
            result = service.generate_with_monitoring(
                prompt=user_input,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k
            )
            
            # Extract just the generated part (remove the original prompt)
            generated = result['response']
            if generated.lower().startswith(user_input.lower()):
                generated = generated[len(user_input):].strip()
            
            print(generated)
            
            # Show metrics
            print(f"📊 [{result['total_tokens']} tokens, {result['latency_ms']}ms]")
            
        except KeyboardInterrupt:
            print("\n\n👋 Chat interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again with a different prompt.")

if __name__ == "__main__":
    main()
