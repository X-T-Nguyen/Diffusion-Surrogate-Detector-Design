import os
from DiffusionFreeGuidence.TrainCondition import train, eval

def main(model_config=None):
    modelConfig = {
        "state": "eval", 
        "epoch": 100,
        "batch_size": 25,  # batch_size for eval = 25 (maximum for A100 GPU)
        "T": 500,
        "channel": 128,
        "channel_mult": [1, 2, 2, 2],
        "num_res_blocks": 2, 
        "dropout": 0.15,
        "lr": 1e-4,
        "multiplier": 2.5,
        "beta_1": 1e-4,
        "beta_T": 0.028,
        "img_size": 128,
        "grad_clip": 1.,
        "device": "cuda:0",
        "w": 1.8, 
        "save_weight_dir": "./CheckpointsCondition_large_batch/",
        "training_load_weight": None,
        "test_load_weight": "ckpt_40.pt",
        "sampled_dir": "./image_results/",
        "sampledNoisyImgName": "NoisyGuidenceImgs.png",
        "sampledImgName": "SampledGuidenceImgs.png",
        "nrow": 5
    }

    if model_config is not None:
        modelConfig = model_config

    # Get the absolute path of the checkpoint directory
    checkpoint_dir = os.path.abspath(modelConfig["save_weight_dir"])

    # Get the list of checkpoint files
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('ckpt_') and f.endswith('.pt')]

    # Loop through each checkpoint file
    for checkpoint in checkpoint_files:
        print(f"Evaluating with checkpoint: {checkpoint}")
        
        # Update the modelConfig to load the specific checkpoint
        modelConfig["test_load_weight"] = os.path.join(checkpoint_dir, checkpoint)

        # Modify the names based on the checkpoint number
        checkpoint_number = checkpoint.split('_')[1].split('.')[0]
        modelConfig["sampledNoisyImgName"] = f"NoisyGuidenceImgs_{checkpoint_number}.png"
        modelConfig["sampledImgName"] = f"SampledGuidenceImgs_{checkpoint_number}.png"

        # Perform evaluation for this checkpoint
        eval(modelConfig)

if __name__ == '__main__':
    main()
