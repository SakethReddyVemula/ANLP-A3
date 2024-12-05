import os
import sys
import argparse
from tqdm import tqdm
from itertools import count
import wandb
import yaml
from utils import *
import pandas as pd
from omegaconf import OmegaConf

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from torch.optim import Adam
from transformers import logging as hf_logging
from torch.nn.utils import clip_grad_norm_

from dataset import CNN_dataset
from model import PromptTuningGPT2, LastLayerGPT2, LoraGPT

import time
from rouge_score import rouge_scorer
import torch.cuda
from utils import FLOPsCounter, get_gpu_memory_usage, count_parameters, calculate_flops, RobustLossTracker

# Set up wandb key
os.environ["WANDB_API_KEY"] = ""

def setup_dataset(config, tokenizer, device):
    train_df = config["paths"]["train_df"]
    val_df = config["paths"]["val_df"]
    test_df = config["paths"]["test_df"]

    print(f"loading dataset from csv files")
    train_df = pd.read_csv(train_df).drop(columns=["id"]).sample(frac=config["fraction_of_data"], random_state=config["random_state"]).reset_index(drop=True)
    val_df = pd.read_csv(val_df).drop(columns=["id"]).sample(frac=config["fraction_of_data"], random_state=config["random_state"]).reset_index(drop=True)
    test_df = pd.read_csv(test_df).drop(columns=["id"]).sample(frac=config["fraction_of_data"], random_state=config["random_state"]).reset_index(drop=True)

    train_dataset = CNN_dataset(
        dataframe = train_df,
        config = config,
        tokenizer = tokenizer
    )

    val_dataset = CNN_dataset(
        dataframe = val_df,
        config = config,
        tokenizer = tokenizer
    )

    test_dataset = CNN_dataset(
        dataframe = test_df,
        config = config,
        tokenizer = tokenizer
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config["validation"]["batch_size"],
        shuffle=True
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config["validation"]["batch_size"],
        shuffle=True
    )

    print(f"loading dataset, dataloader completed")
    return train_dataset, val_dataset, test_dataset, train_dataloader, val_dataloader, test_dataloader

def train_epoch(model, dataloader: DataLoader, optimizer, device, config, flops_counter):
    model.train()
    progress_bar = tqdm(dataloader, desc="Training")
    
    # Initialize robust loss tracker
    window_size = config["training"].get("loss_window_size", 100)
    plateau_threshold = config["training"].get("plateau_threshold", 0.01)
    plateau_patience = config["training"].get("plateau_patience", 3)
    loss_tracker = RobustLossTracker(window_size=window_size, plateau_threshold=plateau_threshold)
    
    # Plateau detection counters
    plateau_counter = 0
    
    # Statistics for logging
    running_loss = 0.0
    running_count = 0

    # start counting FLOPs
    flops_counter.reset()
    flops_counter.start_counting(model)
    
    for batch_idx, (article, highlight, _) in enumerate(progress_bar):
        try:
            input_ids = article["input_ids"].squeeze(1).to(device)
            attention_mask = article["attention_mask"].squeeze(1).to(device)
            labels = highlight["input_ids"].squeeze(1).to(device)

            # Forward pass
            output = model(
                input_ids,
                attention_mask,
                labels
            )

            loss = output.loss
            current_loss = loss.item()
            
            # Update running statistics
            running_loss += current_loss
            running_count += 1
            avg_loss = running_loss / running_count
            
            # Check for plateau using the robust tracker
            is_plateau = loss_tracker.add_loss(current_loss)
            
            if is_plateau:
                plateau_counter += 1
                if plateau_counter >= plateau_patience:
                    print(f"\nEarly stopping within epoch at batch {batch_idx}")
                    print(f"Detected stable plateau in training loss for {plateau_patience} consecutive windows")
                    break
            else:
                plateau_counter = 0

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(parameters=model.parameters(), max_norm=config["training"]["max_grad_norm"])
            optimizer.step()

            # Update progress bar
            progress_bar.set_postfix({
                "loss": current_loss,
                "avg_loss": avg_loss,
                "plateau_counter": plateau_counter
            })

            if config["wandb_logging"] == True:
                wandb.log({
                    "training/batch_loss": current_loss,
                    "training/average_loss": avg_loss,
                    "training/plateau_counter": plateau_counter
                })

        except RuntimeError as e:
            print(f"\nSkipping batch {batch_idx} due to runtime error: {str(e)}")
            continue
    
    flops_counter.stop_counting()
    epochs_flops = flops_counter.total_flops

    if config["wandb_logging"]:
        wandb.log({
            "training/epoch_flops": epochs_flops
        })

    return running_loss / (batch_idx + 1), epochs_flops

def validate(model: GPT2LMHeadModel, dataloader: DataLoader, device, config):
    model.eval()

    total_loss = 0.0

    with torch.no_grad():
        for article, highlight, _ in tqdm(dataloader, desc="Validation"):
            input_ids = article["input_ids"].squeeze(1).to(device)
            attention_mask = article["attention_mask"].squeeze(1).to(device)
            labels = highlight["input_ids"].squeeze(1).to(device)

            output = model(
                input_ids,
                attention_mask,
                labels
            )

            loss = output.loss
            total_loss += loss.item()

    return total_loss / len(dataloader)

def generate_summary(model, tokenizer, article, device, config):
    """Generate a summary for a single article using the fine-tuned model"""
    model.eval()
    with torch.no_grad():
        input_ids = article["input_ids"].squeeze(1).to(device)
        attention_mask = article["attention_mask"].squeeze(1).to(device)

        # Calculate available space for generation
        if config["method"] == "prompt_tuning":
            n_prompt_tokens = config["prompt_tuning"]["num_tokens"]
            desired_input_length = 1024 - config["generation"]["max_new_tokens"] - n_prompt_tokens
        elif config["method"] == "last_layer":
            desired_input_length = 1024 - config["generation"]["max_new_tokens"]
        elif config["method"] == "lora":
            desired_input_length = 1024 - config["generation"]["max_new_tokens"]
        
        # Truncate input to leave room for generation and prompt tokens
        input_ids = input_ids[:, :desired_input_length]
        attention_mask = attention_mask[:, :desired_input_length]

        if config["method"] == "prompt_tuning":
            generated_model = model.model
        elif config["method"] == "lora":
            generated_model = model.model
            generated_model.eval()
        elif config["method"] == "last_layer":
            generated_model = model.model
            generated_model.eval()
        
        outputs = generated_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=config["generation"]["max_new_tokens"],  # Use max_new_tokens instead of max_length
            num_beams=config["generation"]["num_beams"],
            no_repeat_ngram_size=config["generation"]["no_repeat_ngram_size"],
            length_penalty=config["generation"]["length_penalty"],
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id,
            min_length=input_ids.size(1) + 1
        )

        # print(f"output: {outputs[0][input_ids.size(1):]}")

        # Decode only the generated part (excluding the input)
        generated_summary = tokenizer.decode(
            outputs[0][input_ids.size(1):],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        # print(f"generated_summary length: {len(generated_summary.split())}")
        # print(f"generated summary: {generated_summary}")

        return generated_summary

def evaluate(config, model_path, test_dataloader, tokenizer, device, flops_counter):
    """Evaluate the model on the test set and calculate ROUGE scores"""
    print("Starting evaluation...")
    
    # Initialize ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Load the trained model
    if config["method"] == "prompt_tuning":
        model = PromptTuningGPT2(
            GPT2LMHeadModel.from_pretrained(config["model_name"]).to(device),
            config,
            tokenizer
        )
    elif config["method"] == "last_layer":
        model = LastLayerGPT2(
            GPT2LMHeadModel.from_pretrained(config["model_name"]).to(device),
            config
        )
    elif config["method"] == "lora":
        model = LoraGPT(
            GPT2LMHeadModel.from_pretrained(config["model_name"]).to(device),
            config
        )

    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    
    # Initialize score accumulators
    rouge_scores = {
        'rouge1': {'precision': 0, 'recall': 0, 'fmeasure': 0},
        'rouge2': {'precision': 0, 'recall': 0, 'fmeasure': 0},
        'rougeL': {'precision': 0, 'recall': 0, 'fmeasure': 0}
    }
    
    total_samples = 0
    generated_summaries = []
    reference_summaries = []
    
    print("Generating summaries and calculating ROUGE scores...")
    for batch_idx, (article, highlight, _) in enumerate(tqdm(test_dataloader)):
        try:
            # Generate summary
            generated_summary = generate_summary(model, tokenizer, article, device, config)
            
            # Get reference summary
            reference_summary = tokenizer.decode(
                highlight["input_ids"].squeeze(1)[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            # Calculate ROUGE scores
            scores = scorer.score(reference_summary, generated_summary)
            
            # Accumulate scores
            for metric in rouge_scores.keys():
                rouge_scores[metric]['precision'] += scores[metric].precision
                rouge_scores[metric]['recall'] += scores[metric].recall
                rouge_scores[metric]['fmeasure'] += scores[metric].fmeasure
            
            total_samples += 1
            
            # Store summaries for later analysis
            generated_summaries.append(generated_summary)
            reference_summaries.append(reference_summary)
            
            # Log example summaries periodically
            if batch_idx % 100 == 0:
                print("\nExample at batch", batch_idx)
                print("Generated:", generated_summary[:200] + "...")
                print("Reference:", reference_summary[:200] + "...")
                
        except Exception as e:
            print(f"Error processing batch {batch_idx}: {str(e)}")
            continue
    
    # Calculate average scores
    for metric in rouge_scores.keys():
        for score_type in rouge_scores[metric].keys():
            rouge_scores[metric][score_type] /= total_samples
    
    # Print results
    print("\nEvaluation Results:")
    print("-" * 50)
    for metric in rouge_scores.keys():
        print(f"\n{metric}:")
        for score_type, score in rouge_scores[metric].items():
            print(f"{score_type}: {score:.4f}")
    
    # Log to wandb if enabled
    if config["wandb_logging"]:
        wandb.log({
            "test/rouge1_f": rouge_scores['rouge1']['fmeasure'],
            "test/rouge2_f": rouge_scores['rouge2']['fmeasure'],
            "test/rougeL_f": rouge_scores['rougeL']['fmeasure'],
            "test/samples": total_samples
        })
        
        # Log example predictions
        example_table = wandb.Table(columns=["Generated Summary", "Reference Summary"])
        for gen, ref in zip(generated_summaries[:10], reference_summaries[:10]):
            example_table.add_data(gen, ref)
        wandb.log({"test/example_predictions": example_table})
    
    return rouge_scores, generated_summaries, reference_summaries

def train_prompt_tuning(base_model: GPT2LMHeadModel, config, tokenizer: GPT2TokenizerFast, train_dataloader, val_dataloader, device, flops_counter, total_training_flops):
    print(f"Starting Prompt Tuning")
    
    model = PromptTuningGPT2(base_model, config, tokenizer)
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=config["prompt_tuning"]["lr"]
    )

    total_epochs = config["training"]["num_epochs"]
    best_val_loss = float('inf')
    
    # Early stopping parameters
    patience = config["training"].get("early_stopping_patience", 3)  # Default patience of 3 epochs
    min_delta = config["training"].get("early_stopping_min_delta", 1e-4)  # Minimum change to qualify as an improvement
    patience_counter = 0
    
    for epoch in range(total_epochs):
        print(f"Epoch: {epoch + 1}/{total_epochs}")
        train_loss, epoch_flops = train_epoch(model, train_dataloader, optimizer, device, config, flops_counter)
        val_loss = validate(model, val_dataloader, device, config)

        total_training_flops += epoch_flops

        print(f"Train_Loss: {train_loss:.4f}")
        print(f"Validation_Loss: {val_loss:.4f}")

        if config["wandb_logging"] == True:
            wandb.log({
                "epoch": epoch,
                "train/epoch_loss": train_loss,
                "validation/loss": val_loss,
                "early_stopping/patience_counter": patience_counter
            })

        # Check if validation loss improved
        if val_loss < (best_val_loss - min_delta):
            best_val_loss = val_loss
            patience_counter = 0
            
            if config["do_save_model_checkpoints"] == True:
                torch.save(model.state_dict(), config["prompt_tuning"]["model_save_path"])
                print(f"Saved current best checkpoint")
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience counter: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

        print(f"-"*90)    
    return model, total_training_flops


def train_last_layer(base_model: GPT2LMHeadModel, config, tokenizer: GPT2TokenizerFast, train_dataloader: DataLoader, val_dataloader: DataLoader, device, flops_counter, total_training_flops):
    print(f"Starting Last Layer Fine-Tuning...")

    # initialize model with frozen layers except last layer
    model = LastLayerGPT2(
        base_model,
        config
    )

    model = model.to(device)

    optimizer = torch.optim.AdamW(
        # [p for p in model.parameters() if p.requires_grad],
        params=model.parameters(),
        lr=config["last_layer"]["lr"]
    )

    total_epochs = config["training"]["num_epochs"]
    best_val_loss = float('inf')

    patience = config["training"].get("early_stopping_patience", 3)
    min_delta = config["training"].get("early_min_delta", 1e-4)

    patience_counter = 0

    for epoch in range(total_epochs):
        print(f"Epoch: {epoch + 1}/{total_epochs}")

        model.train()

        train_loss, epoch_flops = train_epoch(model, train_dataloader, optimizer, device=device, config=config, flops_counter=flops_counter)
        val_loss = validate(model, val_dataloader, device, config)

        total_training_flops += epoch_flops

        print(f"Train loss: {train_loss:.4f}")
        print(f"Validation loss: {val_loss:.4f}")

        if config["wandb_logging"]:
            wandb.log({
                "epoch": epoch,
                "train/epoch_loss": train_loss,
                "validation/loss": val_loss,
                "early_stopping/patience_counter": patience_counter
            })

        if val_loss < (best_val_loss - min_delta):
            best_val_loss = val_loss
            patience_counter = 0

            if config["do_save_model_checkpoints"]:
                torch.save(model.state_dict(), config["last_layer"]["model_save_path"])
                print(f"Saved current best checkpoint")
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience Counter: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

        print(f"-"*90)

    return model, total_training_flops

def train_lora(base_model: GPT2LMHeadModel, config, tokenizer: GPT2TokenizerFast, train_dataloader: DataLoader, val_dataloader: DataLoader, device, flops_counter, total_training_flops):
    lora_model = LoraGPT(
        base_model,
        config
    )
    
    optimizer = torch.optim.AdamW(
        # filter(lambda p: p.requires_grad, lora_model.parameters()),
        params=model.parameters(),
        lr = config["lora"]["lr"]
    )

    best_val_loss = float("inf")
    patience = config["training"].get("early_stopping_patience", 3)
    min_delta = config["training"].get("early_min_delta", 1e-4)

    patience_counter = 0
    total_epochs = config["training"]["num_epochs"]

    for epoch in range(total_epochs):
        print(f"\nEpoch: {epoch + 1}/{total_epochs}")
        
        train_loss, epoch_flops = train_epoch(
            model=lora_model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            device=device,
            config=config,
            flops_counter=flops_counter
        )

        val_loss = validate(
            model=lora_model,
            dataloader=val_dataloader,
            device=device,
            config=config
        )

        total_training_flops += epoch_flops

        print(f"Train loss: {train_loss:.4f}")
        print(f"Validation loss: {val_loss:.4f}")

        if config["wandb_logging"]:
            wandb.log({
                "epoch": epoch,
                "train/epoch_loss": train_loss,
                "validation/loss": val_loss,
                "early_stopping/patience_counter": patience_counter
            })

        if val_loss < (best_val_loss - min_delta):
            best_val_loss = val_loss
            patience_counter = 0

            if config["do_save_model_checkpoints"]:
                torch.save(lora_model.state_dict(), config["lora"]["model_save_path"])
                print(f"Saved current best checkpoint")
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience Counter: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

        print(f"-"*90)

    # Final evaluation on test set
    test_loss = validate(lora_model, test_dataloader, device, config)
    print(f"\nFinal Test Loss: {test_loss:.4f}")
    
    if config["wandb_logging"]:
        wandb.log({"test_loss": test_loss})
    
    return lora_model, total_training_flops



if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    hf_logging.set_verbosity_error()

    config = OmegaConf.load("config/config.yaml")
    overrides = OmegaConf.from_cli()
    config = OmegaConf.merge(config, overrides)
    OmegaConf.to_container(config)

    config = OmegaConf.to_container(config, resolve=True)
    
    # PromptTuning / Last Layer / LoRA
    method = config['method']
    
    # setup the device for GPU acceleration
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"using device: CUDA")
    else:
        device = torch.device("cpu")
        print(f"using device: CPU")

    # initialize the run in wandb
    if config["wandb_logging"] == True:
        wandb.init(
            id = wandb.util.generate_id(),
            project = config["wandb_project"],
            entity = config["entity"],
            config = config
        )

    # Load the model
    model = GPT2LMHeadModel.from_pretrained(config["model_name"])
    model = model.to(device)
    
    # Load the GPT-2 FastTokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained(config["model_name"])
    
    # print(f"eos_token: {tokenizer.eos_token}\teos_token_id: {tokenizer.eos_token_id}")
    # print(f"pad_token: {tokenizer.pad_token}\tpad_token_id: {tokenizer.pad_token_id}")

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # get dataset, load dataloaders
    train_dataset, val_dataset, test_dataset, train_dataloader, val_dataloader, test_dataloader = setup_dataset(config, tokenizer, device)

    # Freeze all the parameters of the pretrained model
    print(f"Freezing all the parameters initially")
    for param in model.parameters():
        param.requires_grad = False

    if config["method"] == "prompt_tuning":
        if config["do_training"]:
            train_start_time = time.time()
            flops_counter = FLOPsCounter()
            total_training_flops = 0

            model, total_training_flops = train_prompt_tuning(model, config, tokenizer, train_dataloader, val_dataloader, device, flops_counter, total_training_flops)
            
            training_time = time.time() - train_start_time

            if config["wandb_logging"]:
                wandb.log({
                    "metrics/training_time_seconds": training_time,
                    "metrics/total_training_flops": total_training_flops
                })

        if config["do_evaluation"]:
            flops_counter = FLOPsCounter()
            rouge_scores, generated_summaries, reference_summaries = evaluate(
                config=config,
                model_path=config["prompt_tuning"]["model_save_path"],
                test_dataloader=test_dataloader,
                tokenizer=tokenizer,
                device=device,
                flops_counter=flops_counter
            )
    elif config["method"] == "last_layer":
        if config["do_training"]:
            train_start_time = time.time()
            flops_counter = FLOPsCounter()
            total_training_flops = 0

            model, total_training_flops = train_last_layer(model, config, tokenizer, train_dataloader, val_dataloader, device, flops_counter, total_training_flops)

            training_time = time.time() - train_start_time

            if config["wandb_logging"]:
                wandb.log({
                    "metrics/training_time_seconds": training_time,
                    "metrics/total_training_flops": total_training_flops
                })

        if config["do_evaluation"]:
            flops_counter = FLOPsCounter()
            rouge_scores, generated_summaries, reference_summaries = evaluate(
                config=config,
                model_path=config["last_layer"]["model_save_path"],
                test_dataloader=test_dataloader,
                tokenizer=tokenizer,
                device=device,
                flops_counter=flops_counter
            )
    elif config["method"] == "lora":
        if config["do_training"] == True:
            train_start_time = time.time()
            flops_counter = FLOPsCounter()
            total_training_flops = 0

            model, total_training_flops = train_lora(model, config, tokenizer, train_dataloader, val_dataloader, device, flops_counter, total_training_flops)
            
            training_time = time.time() - train_start_time

            if config["wandb_logging"]:
                wandb.log({
                    "metrics/training_time_seconds": training_time,
                    "metrics/total_training_flops": total_training_flops
                })

        if config["do_evaluation"]:
            flops_counter = FLOPsCounter()
            rouge_scores, generated_summaries, reference_summaries = evaluate(
                config=config,
                model_path=config["lora"]["model_save_path"],
                test_dataloader=test_dataloader,
                tokenizer=tokenizer,
                device=device,
                flops_counter=flops_counter
            )

