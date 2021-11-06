import warnings


class DefaultConfig(object):

    trainCsvPath = '/media/adnub/bf2cc213-0d25-4fa4-b89c-09791af2b29a/project/data_set/CheXpert-v1.0-small/train.csv'
    validCsvPath = '/media/adnub/bf2cc213-0d25-4fa4-b89c-09791af2b29a/project/data_set/CheXpert-v1.0-small/valid.csv'
    testCsvPath = '/media/adnub/bf2cc213-0d25-4fa4-b89c-09791af2b29a/project/data_set/CheXpert-v1.0-small/valid.csv'
    adj_file = 'get_gcn_x_adj/train_adj_14.pkl'

    # todo: baseline
    data_root = '/media/adnub/bf2cc213-0d25-4fa4-b89c-09791af2b29a/project/data_set'

    to0 = False
    if to0 == True:
        train_data_list = 'data/to_1/my_trainSet_14.csv'
        valid_data_list = 'data/to_1/my_validSet_14.csv'
        test_data_list = 'data/to_1/my_testSet_14.csv'
    elif to0 == False:
        train_data_list = 'data/my_trainSet_14.csv'
        valid_data_list = 'data/my_validSet_14.csv'
        test_data_list = 'data/my_testSet_14.csv'

    classes = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion',
               'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
               'Pleural Other', 'Fracture', 'Support Devices']
    classes_draw = ['NoFi', 'EnCa', 'Card', 'Opac', 'Lesi',
                    'Edem', 'Cons', 'Pneu1', 'Atel', 'Pneu2', 'Effu',
                    'PleO', 'Frac', 'SuDe']
    classes_draw_method = ['ResNet-101', 'GL-MLL', 'ML-GCN', 'RethinkNet', 'CheXGCN', 'ADD-GCN', 'CNN-RNN']

    color_style = ['blue', 'yellow', 'orange', 'gray', 'yellowgreen', 'deepskyblue', 'purple',
                   'chocolate', 'hotpink', 'green', 'red', 'turquoise', 'gold', 'violet']
    color_method_style = ['blue', 'red', 'green', 'orange', 'pink', 'gray', 'yellow']

    # TODO: baseline parameters
    batch_size = 64
    lr = 0.0001  # initial learning rate
    betas = (0.9, 0.999)
    eps = 1e-08
    weight_decay = 1e-5

    cam = True
    draw_roc = True

    def parse(self, kwargs):
        """
         update Config through kwargs
        """
        if kwargs:
            for k in kwargs:
                if not hasattr(self, k):
                    warnings.warn("Warning: opt has not attribut %s" % k)
                setattr(self, k, kwargs[k])


opt = DefaultConfig()
