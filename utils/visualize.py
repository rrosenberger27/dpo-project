import matplotlib.pyplot as plt

def plot_sft_metrics(step_losses, eval_loss, eval_perplexity):
    """Plots SFT training loss curve and displays evaluation metrics."""
    plt.figure(figsize=(10, 5))
    plt.plot(step_losses, label="Training Loss", color="blue", alpha=0.7)
    
    # Add a horizontal line for the final evaluation loss
    plt.axhline(y=eval_loss, color="red", linestyle="--", label=f"Eval Loss ({eval_loss:.4f})")
    
    plt.title(f"SFT Training Curve (Final PPL: {eval_perplexity:.2f})")
    plt.xlabel("Training Steps")
    plt.ylabel("Cross Entropy Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_dpo_metrics(step_losses, step_rewards, eval_accuracy, eval_loss):
    """Plots DPO loss and the critical Reward Margin curve."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: DPO Loss
    ax1.plot(step_losses, label="DPO Loss", color="purple", alpha=0.7)
    ax1.axhline(y=eval_loss, color="red", linestyle="--", label=f"Eval Loss ({eval_loss:.4f})")
    ax1.set_title("DPO Loss over Steps")
    ax1.set_xlabel("Training Steps")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Reward Margin (The most important DPO metric)
    ax2.plot(step_rewards, label="Reward Margin (Chosen - Rejected)", color="green", alpha=0.7)
    ax2.axhline(y=0, color="black", linestyle="-") # Baseline 0
    ax2.set_title(f"DPO Reward Margin (Final Test Acc: {eval_accuracy:.1%})")
    ax2.set_xlabel("Training Steps")
    ax2.set_ylabel("Reward Margin")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.show()