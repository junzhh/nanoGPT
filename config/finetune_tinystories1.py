out_dir = 'out-tinystories-data'
init_from = 'resume'  # Ensure checkpoint exists
eval_interval = 5
eval_iters = 40
wandb_log = True  # Enable logging if you want visual tracking
wandb_project = 'tinystories'
wandb_run_name = 'ft-' + str(time.time())

dataset = 'tinystories'
always_save_checkpoint = True  # Save checkpoints regardless of validation loss

block_size = 256
batch_size = 1
gradient_accumulation_steps = 32
max_iters = 1000  # Increase for meaningful fine-tuning

learning_rate = 3e-5
decay_lr = False

precision = 'bfloat16'  # Enable mixed precision if supported
