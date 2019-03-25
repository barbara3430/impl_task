params = dict()

params['epochs'] = 24
params['train_file'] = 'train-v1.1.json'
params['dev_file'] = 'dev-v1.1.json'
params['dim_embeddings'] = 100
params['train_batch_size'] = 50
params['dev_batch_size'] = 16
params['cuda'] = True
params['char_out_size'] = 100
params['out_channel_dims'] = 100
params['filter_heights'] = 5
params['highway_network_layers'] = 2
params['lstm_hidden_size'] = 100
params['dropout'] = 0.2
params['optimizer'] = 'adadelta'
params['weight_decay'] = 0.999
params['highway_network_layers'] = 2
params['max_ctx_len'] = 400
params['data_path'] = './data/SQuAD/'
# params['dataset_file'] = './data/SQuAD/dev-v1.1.json'
params['prediction_file'] = 'pred.out'

