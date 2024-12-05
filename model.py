import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from tqdm import tqdm
from utils import *
import sys
from peft import LoraConfig, get_peft_model, TaskType

class PromptTuningGPT2(nn.Module):
    def __init__(self, model, config, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.n_tokens = config["prompt_tuning"]["num_tokens"]
        self.max_length = config.get("max_length", 1024)  # GPT2's default max length == 1024
        self.device = next(model.parameters()).device
        self.initialize_soft_prompt(config["prompt_tuning"]["soft_prompt"])
        
    def initialize_soft_prompt(self, init_text):
        # Tokenize initialization text
        init_tokens = self.tokenizer(init_text, return_tensors="pt")
        init_token_ids = init_tokens.input_ids[0].to(self.device)
        
        # Print tokenization info
        print(f"Initializing soft prompt with text: {init_text}")
        print(f"Tokenized into: {self.tokenizer.convert_ids_to_tokens(init_token_ids)}")
        print(f"Number of tokens: {len(init_token_ids)}")
        
        # Get embeddings for all tokens
        init_embeddings = self.model.transformer.wte(init_token_ids)
        print(f"Initial embeddings shape: {init_embeddings.shape}")
        
        # Handle padding or truncation
        if init_embeddings.size(0) < self.n_tokens:
            pad_length = self.n_tokens - init_embeddings.size(0)
            print(f"Padding with {pad_length} random embeddings")
            
            random_embeddings = torch.randn(
                pad_length,
                init_embeddings.size(1),
                device=self.device
            ) * 0.02
            random_embeddings = random_embeddings.to(init_embeddings.device)
            init_embeddings = torch.cat([init_embeddings, random_embeddings], dim=0)
        elif init_embeddings.size(0) > self.n_tokens:
            print(f"Truncating {init_embeddings.size(0) - self.n_tokens} tokens")
            init_embeddings = init_embeddings[:self.n_tokens]
            
        print(f"Final soft prompt shape: {init_embeddings.shape}")
        self.soft_prompt = nn.Parameter(init_embeddings.unsqueeze(0))
    
    def get_soft_prompt_embeds(self, batch_size):
        """
        Get the learned soft prompt embeddings
        """
        soft_prompt = self.soft_prompt.repeat(batch_size, 1, 1)
        return soft_prompt
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Calculate how much to truncate from the input
        input_truncate_size = self.max_length - self.n_tokens
        
        # Truncate input_ids and attention_mask
        input_ids = input_ids[:, :input_truncate_size]
        if attention_mask is not None:
            attention_mask = attention_mask[:, :input_truncate_size]
        
        # Get input embeddings
        inputs_embeds = self.model.transformer.wte(input_ids)
        
        # Expand soft prompt for batch size
        soft_prompt = self.soft_prompt.expand(batch_size, -1, -1)
        
        # Prepend soft prompt to input embeddings
        inputs_embeds = torch.cat([soft_prompt, inputs_embeds], dim=1)
        
        # Create position IDs
        position_ids = torch.arange(0, inputs_embeds.shape[1], dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Adjust attention mask
        if attention_mask is not None:
            prompt_attention = torch.ones(batch_size, self.n_tokens, device=device)
            attention_mask = torch.cat([prompt_attention, attention_mask], dim=1)
        
        # Adjust labels if provided
        if labels is not None:
            # Truncate labels to match input size
            labels = labels[:, :input_truncate_size]
            # Create new labels tensor with soft prompt prefix
            new_labels = torch.full(
                (batch_size, self.n_tokens + labels.shape[1]),
                -100,  # Ignore index for loss calculation
                device=device
            )
            new_labels[:, self.n_tokens:] = labels
            labels = new_labels
        
        # Forward pass through model
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
        )

        return outputs


class LastLayerGPT2(nn.Module):
    def __init__(self, model: GPT2LMHeadModel, config):
        super().__init__()
        self.model = model
        self.max_length = config.get("max_length", 1024) # max length of GPT-2 model

        # freeze all parameters initially
        for param in self.model.parameters():
            param.requires_grad = False

        # # unfreeze the last layer (e.g., pre-classifier and classifier)
        # for param in self.model.pre_classifier.parameters():
        #     param.requires_grad = True

        # for param in self.model.classifier.parameters():
        #     param.requires_grad = True

        # unfreeze the last transformer block
        for param in self.model.transformer.h[-1].parameters():
            param.requires_grad = True

        # unfreeze the language model head (which acts as the classifier)
        for param in self.model.lm_head.parameters():
            param.requires_grad = True

        # optionally unfreeze layer norm parameters
        for param in self.model.transformer.ln_f.parameters():
            param.requires_grad = True

        # print trainable parameter info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"Percentage of trainable parameters: {100 * trainable_params / total_params:.2f}%")

    def forward(self, input_ids, attention_mask=None, labels=None):
        # truncate input to max length if necessary
        if input_ids.size(1) > self.max_length:
            input_ids = input_ids[:, :self.max_length]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :self.max_length]
            if labels is not None:
                labels = labels[:, :self.max_length]

        # forward pass through the model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        return outputs

class LoraGPT(nn.Module):
    def __init__(self, model: GPT2LMHeadModel, config):
        super().__init__()
        self.max_length = config.get("max_length", 1024)

        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config["lora"]["rank"],
            lora_alpha=config["lora"]["alpha"],
            lora_dropout=config["lora"]["dropout"],
            bias="none",
            # Target more modules for better fine-tuning
            target_modules=["c_attn", "c_proj", "c_fc"],
            inference_mode=False
        )

        # Create PEFT model
        self.model = get_peft_model(model, lora_config)
        
        # Freeze all parameters except LoRA
        for param in self.model.parameters():
            param.requires_grad = False
        for name, param in self.model.named_parameters():
            if "lora" in name:
                param.requires_grad = True
        
        # Print parameter info
        self._print_trainable_parameters()

    def _print_trainable_parameters(self):
        """
        Print trainable parameter statistics
        """
        trainable_params = 0
        all_params = 0
        for _, param in self.model.named_parameters():
            num_params = param.numel()
            all_params += num_params
            if param.requires_grad:
                trainable_params += num_params
        print(
            f"trainable params: {trainable_params:,d} || "
            f"all params: {all_params:,d} || "
            f"trainable%: {100 * trainable_params / all_params:.2f}%"
        )

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # Handle input truncation
        if input_ids is not None and input_ids.size(1) > self.max_length:
            input_ids = input_ids[:, :self.max_length]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :self.max_length]
            if labels is not None:
                labels = labels[:, :self.max_length]

        # Forward pass through model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )

        # print(f"outputs: {outputs.logits.shape}")

        # sys.exit()
        return outputs

    def generate(self, *args, **kwargs):
        """
        Wrapper for model.generate() method
        """
        return self.model.generate(*args, **kwargs)
    


