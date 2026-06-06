dataset_config = dict(
    type = 'CDD',
    #data_root = '/mnt/f/shujuji_daima/小测试用/LEVIR-CD',
    data_root = 'data/CDD256',
    train_mode = dict(
        transform = dict(
            #RandomSizeAndCrop = {"size": 256, "crop_nopad": False},
            RandomHorizontallyFlip = {"p": 0.5},
            RandomVerticalFlip = {"p": 0.5},
            RandomGaussianBlur = None,
        ),
        loader = dict(
            batch_size = 8,
            num_workers = 4,
            pin_memory=True,
            shuffle = True,
            drop_last = True
        ),
    ),
    
    val_mode = dict(
        transform = dict(
            # RandomSizeAndCrop = {"size": 256, "crop_nopad": False},
        ),
        loader = dict(
            batch_size = 8,
            num_workers = 4,
            pin_memory=True,
            shuffle = False,
            drop_last = False
        )
    ),

    test_mode = dict(
        transform = dict(
            # RandomSizeAndCrop = {"size": 256, "crop_nopad": False},
        ),
        loader = dict(
            batch_size = 8,
            num_workers = 4,
            pin_memory=True,
            shuffle = False,
            drop_last = False
        )
    ),
)
