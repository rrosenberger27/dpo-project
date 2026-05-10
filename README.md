# Final Project DPO

> Finetuning pretrained model using supervised finetuning and direct policy optimization

## Project Structure

```
final-project/
├── data/
│   ├── __init__.py
│   └── data_loader.py         # Code to fetch 'sft' and 'prefs' splits from UltraFeedback
├── models/
│   ├── __init__.py
│   └── model_builder.py       # Logic to load Qwen2-0.5B and handle the reference model
├── training/
│   ├── __init__.py
│   ├── sft_loop.py            # Supervised Fine-Tuning logic
│   └── dpo_loop.py            # DPO logic (managing policy vs. reference model)
├── utils/
│   ├── __init__.py
│   ├── memory.py              # Any memory optimization (gradient checkpointing, LoRA)
│   └── visualize.py           # Code to plot the SFT and DPO loss curves
├── notebooks/
│   └── colab_runner.ipynb     # Notebook to run on Collab/some GPU
├── main.py                    # Entry point to run locally
└── requirements.txt
```

---

**Note:**

- Two saved folders for the SFT adapters and DPO adapters.
- Commands in the notebook for merging the SFT adapters with the base model (for then using DPO adapters with), but merged model stored in drive for now.
