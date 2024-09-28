conf = {
    "data": {
        'dataset_path': "./GaitData",
        'resolution': '64',
        'dataset': 'GaitData',
    },
    "model": {
        'hidden_dim': 128,
        'lr': 1e-4,
        'hard_or_full_trip': 'full',
        'batch_size': (4, 4),
        'restore_iter': 0,
        'total_iter': 2000,
        'margin': 0.2,
        'num_workers': 0,
        'frame_num': 30,
        'model_name': 'GaitSet', 
        'test_load_iter': 2000,
    },
}
