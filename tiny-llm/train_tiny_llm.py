#!/usr/bin/env python3
"""
Training Script for Tiny LLM
Train a tiny transformer model and integrate with observability platform
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tiny_llm import TinyLLMService
import requests
import time

def get_sample_training_data():
    """Get sample training data for the model"""
    return [
        # General knowledge
        "The Earth orbits around the Sun once every year.",
        "Water boils at 100 degrees Celsius at sea level.",
        "The human brain contains billions of neurons.",
        "Plants convert sunlight into energy through photosynthesis.",
        "The speed of light is approximately 300,000 kilometers per second.",
        
        # Technology
        "Artificial intelligence is changing how we work and live.",
        "Machine learning algorithms can identify patterns in data.",
        "Neural networks are inspired by the human brain.",
        "Deep learning uses multiple layers to process information.",
        "Natural language processing helps computers understand text.",
        "Computer vision enables machines to see and interpret images.",
        "Robotics combines mechanical engineering with artificial intelligence.",
        
        # Programming
        "Python is a popular programming language for data science.",
        "JavaScript runs in web browsers and on servers.",
        "SQL is used to query databases and retrieve information.",
        "Git helps developers manage code versions and collaborate.",
        "APIs allow different software applications to communicate.",
        "Debugging is the process of finding and fixing errors in code.",
        
        # Math and Science
        "Mathematics is the language of science and engineering.",
        "Statistics help us understand and analyze data patterns.",
        "Physics explains how the universe works at different scales.",
        "Chemistry studies the composition and behavior of matter.",
        "Biology explores life and living organisms.",
        "The scientific method involves observation, hypothesis, and testing.",
        
        # Communication
        "Clear communication is essential for successful collaboration.",
        "Active listening helps build better relationships.",
        "Writing skills are important in both personal and professional contexts.",
        "Public speaking can be improved with practice and preparation.",
        "Body language often conveys more than spoken words.",
        
        # Learning
        "Learning is a lifelong process that never ends.",
        "Practice and repetition help build new skills.",
        "Making mistakes is a natural part of the learning process.",
        "Teaching others helps reinforce your own understanding.",
        "Curiosity drives exploration and discovery.",
        
        # Simple conversations
        "Hello, how are you doing today?",
        "Thank you for your help with this project.",
        "What time does the meeting start tomorrow?",
        "I hope you have a wonderful weekend.",
        "Could you please explain that concept again?",
        "Let me know if you need any assistance.",
        
        # Questions and answers
        "What is machine learning? Machine learning is a method of teaching computers to learn from data.",
        "How do computers work? Computers process information using binary code and electronic circuits.",
        "Why is exercise important? Exercise helps maintain physical health and mental well-being.",
        "Where does rain come from? Rain forms when water vapor in clouds condenses and falls to Earth.",
        "When was the internet invented? The internet was developed gradually, with key milestones in the 1960s-1990s.",
    ]

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
    print("🚂 Tiny LLM Training Script")
    print("=" * 40)
    
    # Check if observability platform is running
    api_key = get_api_key()
    if api_key:
        print(f"✅ Connected to observability platform")
        print(f"🔑 API Key: {api_key[:20]}...")
    else:
        print("⚠️ Observability platform not available - training without monitoring")
        api_key = None
    
    # Get training data
    training_texts = get_sample_training_data()
    print(f"📚 Loaded {len(training_texts)} training examples")
    
    # Initialize the LLM service
    service = TinyLLMService(observability_api_key=api_key)
    
    # Train the model
    print("\n🏋️ Starting training...")
    service.train(
        training_texts=training_texts,
        epochs=10,  # More epochs for better results
        batch_size=4
    )
    
    # Save the trained model
    model_path = "models/tiny_llm_trained.pt"
    os.makedirs("models", exist_ok=True)
    service.save_model(model_path)
    
    # Test the model with various prompts
    test_prompts = [
        "The Earth",
        "Machine learning",
        "Python is",
        "What is",
        "How do computers",
        "Hello, how",
        "Learning is"
    ]
    
    print("\n🎯 Testing the trained model:")
    print("-" * 40)
    
    for prompt in test_prompts:
        result = service.generate_with_monitoring(
            prompt=prompt,
            max_length=30,
            temperature=0.8,
            top_k=50
        )
        
        print(f"Prompt: '{prompt}'")
        print(f"Generated: '{result['response']}'")
        print(f"Tokens: {result['total_tokens']}, Latency: {result['latency_ms']}ms")
        print()
        
        # Small delay between generations
        time.sleep(0.5)
    
    print("🎉 Training and testing completed!")
    print(f"💾 Model saved to: {model_path}")
    
    if api_key:
        print("\n📊 Check the observability dashboard at http://localhost:3000")
        print("   You should see the tiny LLM requests in the analytics!")

if __name__ == "__main__":
    main()
