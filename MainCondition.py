from DiffusionFreeGuidence.TrainCondition import train, eval


def main(model_config=None):
    modelConfig = {
        #"state": "train", 
        "state": "eval", 
        "epoch": 300,
        "batch_size": 25, # batch_size for eval = 25 (maximum for A100 GPU)
        "T": 500,
        "channel": 128,
        "channel_mult": [1, 2, 2, 2], # [1, 2, 2]
        "num_res_blocks": 2, 
        "dropout": 0.15,
        "lr": 1e-4, #1e-4,
        "multiplier": 2, #2.5
        "beta_1": 1e-4,
        "beta_T": 0.028,
        "img_size": 128,
        "grad_clip": 3., #1.
        "device": "cuda:0",
        "w": 1.8, 
        "save_weight_dir": "./CheckpointsCondition_large_batch/",
        "training_load_weight": None,
        "test_load_weight": "ckpt_300.pt",
        "sampled_dir": "./Test_img_condition/",
        "sampledNoisyImgName": "NoisyGuidenceImgs.png",
        "sampledImgName": "SampledGuidenceImgs.png",
        "nrow": 5
    }
    if model_config is not None:
        modelConfig = model_config
    if modelConfig["state"] == "train":
        train(modelConfig)
    else:
        eval(modelConfig)


if __name__ == '__main__':
    main()
