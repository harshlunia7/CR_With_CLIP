config = {
    'exp_name' : 'supervised_train_cr_clip',
    'epoch' : 10000,
    'lr' : 1,
    'batch' : 128,
    'val_frequency' : 100,
    'test_only' : False,
    'resume' : '',
    'train_dataset' : '/ssd_scratch/cvit/rafaelgetto/cr_clip_sup_english_chinese_max_35_train_data',
    'test_dataset': '/ssd_scratch/cvit/rafaelgetto/cr_clip_sup_english_chinese_max_35_test_data',
    'schedule_frequency' : 10,
    'imageH' : 32,
    'imageW' : 128,
    'encoder' : 'vit',
    'decoder' : 'transformer',
    'encoder_checkpoint_path': '/home2/rafaelgetto/DiG_str_multilingual_baseline/experiment_outputs/ssl_pretraining/ssl_pretrain_multi/english_and_chinese/5_Million_samples/checkpoint-9.pth',
    'alpha_path' : './CCR-CLIP/data/char_english_chinese.txt',
    'radical_path': './CCR-CLIP/data/radical_alphabet_27533_benchmark.txt',
    'decompose_path': './CCR-CLIP/data/decompose_27533_benchmark.txt',
    'radical_model': './CCR-CLIP/history/pre_train_clip_multi_lingual/epoch_20_cr_clip.pth',
    'stn': False,
    'constrain': False,
    'char_len' : 60,
}
