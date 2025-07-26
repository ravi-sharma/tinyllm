#!/usr/bin/env python3
"""
Tiny LLM Implementation
A lightweight transformer-based language model for the observability platform
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
import re
import math
import time
from typing import List, Dict, Optional, Tuple
import requests
from pathlib import Path

class TinyLLMConfig:
    """Configuration for the Tiny LLM"""
    def __init__(
        self,
        vocab_size: int = 10000,
        max_seq_length: int = 256,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        dropout: float = 0.1,
        pad_token_id: int = 0,
    ):
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.pad_token_id = pad_token_id

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism"""
    
    def __init__(self, config: TinyLLMConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_k = config.d_model // config.n_heads
        
        self.w_q = nn.Linear(config.d_model, config.d_model)
        self.w_k = nn.Linear(config.d_model, config.d_model)
        self.w_v = nn.Linear(config.d_model, config.d_model)
        self.w_o = nn.Linear(config.d_model, config.d_model)
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        
        # Linear projections
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            # Expand mask to match scores dimensions if needed
            if mask.dim() == 4:  # [batch, 1, seq_len, seq_len]
                mask = mask.expand(batch_size, self.n_heads, seq_len, seq_len)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        return self.w_o(attn_output)

class FeedForward(nn.Module):
    """Position-wise feed-forward network"""
    
    def __init__(self, config: TinyLLMConfig):
        super().__init__()
        self.linear1 = nn.Linear(config.d_model, config.d_ff)
        self.linear2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerBlock(nn.Module):
    """Single transformer block"""
    
    def __init__(self, config: TinyLLMConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection
        attn_output = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class SimpleTokenizer:
    """Simple tokenizer for the tiny LLM"""
    
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.word_to_id = {}
        self.id_to_word = {}
        self.word_counts = {}
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
        
        # Initialize special tokens
        self.word_to_id = {
            self.pad_token: 0,
            self.unk_token: 1,
            self.bos_token: 2,
            self.eos_token: 3,
        }
        self.id_to_word = {v: k for k, v in self.word_to_id.items()}
        
    def build_vocab(self, texts: List[str]):
        """Build vocabulary from training texts"""
        print(f"Building vocabulary from {len(texts)} texts...")
        
        # Count words
        for text in texts:
            words = self._tokenize_text(text)
            for word in words:
                self.word_counts[word] = self.word_counts.get(word, 0) + 1
        
        # Select most frequent words
        sorted_words = sorted(self.word_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Add words to vocabulary (reserve space for special tokens)
        current_id = len(self.word_to_id)
        for word, count in sorted_words:
            if current_id >= self.vocab_size:
                break
            if word not in self.word_to_id:
                self.word_to_id[word] = current_id
                self.id_to_word[current_id] = word
                current_id += 1
        
        print(f"Built vocabulary with {len(self.word_to_id)} tokens")
        
    def _tokenize_text(self, text: str) -> List[str]:
        """Simple tokenization"""
        # Convert to lowercase and split on whitespace and punctuation
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.split()
        
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs"""
        words = self._tokenize_text(text)
        
        token_ids = []
        if add_special_tokens:
            token_ids.append(self.word_to_id[self.bos_token])
            
        for word in words:
            token_id = self.word_to_id.get(word, self.word_to_id[self.unk_token])
            token_ids.append(token_id)
            
        if add_special_tokens:
            token_ids.append(self.word_to_id[self.eos_token])
            
        return token_ids
        
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text"""
        words = []
        for token_id in token_ids:
            if token_id in self.id_to_word:
                word = self.id_to_word[token_id]
                if skip_special_tokens and word in [self.pad_token, self.bos_token, self.eos_token]:
                    continue
                if word == self.unk_token and skip_special_tokens:
                    continue
                words.append(word)
                
        return ' '.join(words)

class TinyLLM(nn.Module):
    """Tiny Language Model implementation"""
    
    def __init__(self, config: TinyLLMConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_length, config.d_model)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        # Output layer
        self.layer_norm = nn.LayerNorm(config.d_model)
        self.output_projection = nn.Linear(config.d_model, config.vocab_size)
        
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len = input_ids.size()
        
        # Create position IDs
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len)
        
        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        
        hidden_states = self.dropout(token_embeds + position_embeds)
        
        # Create causal mask
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device))
        if attention_mask is not None:
            # Expand attention mask to match causal mask dimensions
            expanded_attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)  # [batch, 1, 1, seq_len]
            expanded_attention_mask = expanded_attention_mask.expand(batch_size, 1, seq_len, seq_len)
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len)
            causal_mask = causal_mask * expanded_attention_mask
        else:
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len)
            
        # Transformer blocks
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, causal_mask)
            
        # Output
        hidden_states = self.layer_norm(hidden_states)
        logits = self.output_projection(hidden_states)
        
        return logits
        
    def generate(
        self, 
        tokenizer: SimpleTokenizer, 
        prompt: str, 
        max_length: int = 100, 
        temperature: float = 1.0,
        top_k: int = 50
    ) -> str:
        """Generate text from a prompt"""
        self.eval()
        
        # Encode the prompt
        input_ids = tokenizer.encode(prompt, add_special_tokens=True)
        input_tensor = torch.tensor([input_ids], dtype=torch.long)
        
        generated_ids = input_ids.copy()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Get model predictions
                logits = self.forward(input_tensor)
                next_token_logits = logits[0, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits[top_k_indices] = top_k_logits
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token_id = torch.multinomial(probs, 1).item()
                
                # Stop if EOS token
                if next_token_id == tokenizer.word_to_id[tokenizer.eos_token]:
                    break
                    
                generated_ids.append(next_token_id)
                
                # Update input tensor
                input_tensor = torch.tensor([generated_ids], dtype=torch.long)
                
                # Stop if max sequence length reached
                if len(generated_ids) >= self.config.max_seq_length:
                    break
        
        # Decode generated text
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        return generated_text

class TinyLLMService:
    """Service for integrating Tiny LLM with the observability platform"""
    
    def __init__(self, model_path: Optional[str] = None, observability_api_key: str = None):
        self.config = TinyLLMConfig()
        self.tokenizer = SimpleTokenizer(vocab_size=self.config.vocab_size)
        self.model = TinyLLM(self.config)
        self.observability_api_key = observability_api_key
        self.observability_url = "http://localhost:8001/api/v1"
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
            
    def train(self, training_texts: List[str], epochs: int = 5, batch_size: int = 8):
        """Train the model on provided texts"""
        print(f"🚂 Training Tiny LLM on {len(training_texts)} texts...")
        
        # Build vocabulary
        self.tokenizer.build_vocab(training_texts)
        
        # Prepare training data
        dataset = TextDataset(training_texts, self.tokenizer, self.config.max_seq_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Setup training
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.word_to_id[self.tokenizer.pad_token])
        
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, (input_ids, attention_mask) in enumerate(dataloader):
                optimizer.zero_grad()
                
                # Forward pass
                logits = self.model(input_ids, attention_mask)
                
                # Calculate loss (predict next token)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                
                loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{epochs} completed. Average Loss: {avg_loss:.4f}")
            
        print("✅ Training completed!")
        
    def generate_with_monitoring(self, prompt: str, **kwargs) -> Dict:
        """Generate text and send metrics to observability platform"""
        start_time = time.time()
        
        try:
            # Generate text
            generated_text = self.model.generate(self.tokenizer, prompt, **kwargs)
            end_time = time.time()
            
            # Calculate metrics
            latency_ms = int((end_time - start_time) * 1000)
            input_tokens = len(self.tokenizer.encode(prompt))
            output_tokens = len(self.tokenizer.encode(generated_text)) - input_tokens
            total_tokens = input_tokens + output_tokens
            
            # Prepare result
            result = {
                "prompt": prompt,
                "response": generated_text,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "latency_ms": latency_ms,
                "status": "success"
            }
            
            # Send to observability platform
            if self.observability_api_key:
                self._send_to_observability(result)
                
            return result
            
        except Exception as e:
            end_time = time.time()
            latency_ms = int((end_time - start_time) * 1000)
            
            result = {
                "prompt": prompt,
                "response": "",
                "input_tokens": len(self.tokenizer.encode(prompt)),
                "output_tokens": 0,
                "total_tokens": len(self.tokenizer.encode(prompt)),
                "latency_ms": latency_ms,
                "status": "error",
                "error": str(e)
            }
            
            if self.observability_api_key:
                self._send_to_observability(result)
                
            return result
    
    def _send_to_observability(self, result: Dict):
        """Send metrics to the observability platform"""
        try:
            payload = {
                "project_api_key": self.observability_api_key,
                "provider_name": "tiny-llm",
                "model_name": "tiny-transformer",
                "prompt": result["prompt"],
                "response": result["response"],
                "completion_tokens": result["output_tokens"],
                "prompt_tokens": result["input_tokens"],
                "total_tokens": result["total_tokens"],
                "latency_ms": result["latency_ms"],
                "cost_usd": 0.0,  # Our local model is free!
                "status": result["status"],
                "error_message": result.get("error"),
                "metadata": {"model_type": "tiny-llm-transformer"}
            }
            
            response = requests.post(
                f"{self.observability_url}/ingest/llm",
                json=payload,
                timeout=5
            )
            
            if response.status_code == 200:
                print("📊 Metrics sent to observability platform")
            else:
                print(f"⚠️ Failed to send metrics: {response.status_code}")
                
        except Exception as e:
            print(f"⚠️ Error sending metrics: {e}")
    
    def save_model(self, path: str):
        """Save model and tokenizer"""
        save_data = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config.__dict__,
            'tokenizer_word_to_id': self.tokenizer.word_to_id,
            'tokenizer_id_to_word': self.tokenizer.id_to_word,
        }
        torch.save(save_data, path)
        print(f"✅ Model saved to {path}")
        
    def load_model(self, path: str):
        """Load model and tokenizer"""
        save_data = torch.load(path, map_location='cpu')
        
        # Restore config
        for key, value in save_data['config'].items():
            setattr(self.config, key, value)
            
        # Restore tokenizer
        self.tokenizer.word_to_id = save_data['tokenizer_word_to_id']
        self.tokenizer.id_to_word = save_data['tokenizer_id_to_word']
        
        # Restore model
        self.model = TinyLLM(self.config)
        self.model.load_state_dict(save_data['model_state_dict'])
        self.model.eval()
        
        print(f"✅ Model loaded from {path}")

class TextDataset(Dataset):
    """Dataset for training the language model"""
    
    def __init__(self, texts: List[str], tokenizer: SimpleTokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        for text in texts:
            token_ids = tokenizer.encode(text, add_special_tokens=True)
            if len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
            
            # Pad to max_length
            attention_mask = [1] * len(token_ids) + [0] * (max_length - len(token_ids))
            token_ids = token_ids + [tokenizer.word_to_id[tokenizer.pad_token]] * (max_length - len(token_ids))
            
            self.examples.append({
                'input_ids': torch.tensor(token_ids, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
            })
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]['input_ids'], self.examples[idx]['attention_mask']

if __name__ == "__main__":
    # Example usage
    print("🤖 Tiny LLM Example")
    
    # Sample training data
    training_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world.",
        "Machine learning models can learn from data.",
        "Natural language processing enables computers to understand text.",
        "Deep learning uses neural networks with multiple layers.",
    ]
    
    # Initialize service
    service = TinyLLMService()
    
    # Train the model
    service.train(training_texts, epochs=2)
    
    # Generate text
    result = service.generate_with_monitoring("The quick brown", max_length=20)
    print(f"\n🎯 Generated: {result['response']}")
    print(f"📊 Tokens: {result['total_tokens']}, Latency: {result['latency_ms']}ms")
    
    # Save model
    service.save_model("models/tiny_llm.pt")
