GPT2 Reproduce by Andrej Karpathy

### Before we start
1. Install required packages:
```
pip install -r requirements.txt
```

2. Get ready for the pretraining dataset(requires an hour or less):
```
python fineweb.py
```

### Train the GPT2 model
To run the DDP (assuming you have 8 GPUs in your machine)
```
./run_train_gpt2.sh
```

or if you have problem with torchrun command, run
```
python -m torch.distributed.run --nproc_per_node=8 train_gpt2.py
```

or for a non-DDP launch, run 
```
python train_gpt2.py
```