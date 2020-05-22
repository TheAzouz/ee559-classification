from Models import *

print('************** Baseline Net ************** ')


model_baseline, error_baseline, train_error_baseline, test_error_baseline = train_Net_model('BaselineNet',
                                                                    lr=0.1 ,verbose = False)

print('************** Fully Connected Net ************** ')


model_fully, error_fully, train_error_fully, test_error_fully = train_Net_model('FullyConnectedNet',
                                                                    lr=0.1 ,verbose = False)

print('************** Res Net ************** ')

model_resnet, error_resnet, train_error_resnet, test_error_resnet = train_Net_model('ResNet',
                                                                    lr=0.1 ,verbose = False)

print('**************** CNN ***************')
print('Weight Sharing = ''False'', Auxilary Loss = ''False''')

model_cnn0, error_cnn0, train_error_cnn0, test_error_cnn0 = train_Net_model('CNN',
                                                                weight_sharing = False, auxilary_loss = False,
                                                                lr=0.1 ,verbose = False)

print('Weight Sharing = ''True'', Auxilary Loss = ''False''')
model_cnn1, error_cnn1, train_error_cnn1, test_error_cnn1 = train_Net_model('CNN',
                                                                weight_sharing = True, auxilary_loss = False,
                                                                lr=0.1 ,verbose = False)

print('Weight Sharing = ''False'', Auxilary Loss = ''True''')
model_cnn2, error_cnn2, train_error_cnn2, test_error_cnn2 = train_Net_model('CNN',
                                                                weight_sharing = False, auxilary_loss = True,
                                                                lr=0.1 ,verbose = False)

print('Weight Sharing = ''True'', Auxilary Loss = ''True''')
model_cnn3, error_cnn3, train_error_cnn3, test_error_cnn3 = train_Net_model('CNN',
                                                                weight_sharing = True, auxilary_loss = True,
                                                                lr=0.1 ,verbose = False)

print('**************** LeNet ***************')
print('Weight Sharing = ''False'', Auxilary Loss = ''False''')
model_lenet0, error_lenet0, train_error_lenet0, test_error_lenet0 = train_Net_model('LeNet',
                                                                weight_sharing = False, auxilary_loss = False,
                                                                lr=0.1 ,verbose = False)

print('Weight Sharing = ''True'', Auxilary Loss = ''False''')
model_lenet1, error_lenet1, train_error_lenet1, test_error_lenet1 = train_Net_model('LeNet',
                                                                weight_sharing = True, auxilary_loss = False,
                                                                lr=0.1 ,verbose = False)

print('Weight Sharing = ''False'', Auxilary Loss = ''True''')
model_lenet2, error_lenet2, train_error_lenet2, test_error_lenet2 = train_Net_model('LeNet',
                                                                weight_sharing = False, auxilary_loss = True,
                                                                lr=0.1 ,verbose = False)

print('Weight Sharing = ''True'', Auxilary Loss = ''True''')
model_lenet3, error_lenet3, train_error_lenet3, test_error_lenet3 = train_Net_model('LeNet',
                                                                weight_sharing = True, auxilary_loss = True,
                                                                lr=0.1 ,verbose = False)