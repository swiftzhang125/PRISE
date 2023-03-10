# star-convex loss
dataset_name = 'GoogleMap'
rho = 0.5
mu = 1
lambda_loss = 5
option = 'option1'

# model training
lr = 1e-5
if 'Google' in dataset_name:
    sample_noise = 4
    train_stage1 = 10
    train_stage2 = 10
    train_stage3 = 10
    batch_size = 4
    load_epoch_regression = [100, 100, 80]
elif 'COCO' in dataset_name:
    sample_noise = 2
    train_stage1 = 3
    train_stage2 = 3
    train_stage3 = 3
    batch_size = 4
    load_epoch_regression = [50, 50, 40]

else:
    assert 1 == 2, 'Not implement yet!'
path = f'../results/{dataset_name}/rho{rho}_mu{mu}_l{lambda_loss}_nsample{sample_noise}'
num_iters = 30