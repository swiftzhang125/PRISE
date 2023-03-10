import pandas as pd
import numpy as np
import sys
sys.path.append('../')
from data_read import *
from net import *
import numpy as np
import os
import settings
from train_utils import initial_motion, gt_motion_rs, calculate_feature_map, compute_ssim, gt_motion_rs_random_noisy, construct_matrix

rho = settings.rho
mu = settings.mu
lambda_loss = settings.lambda_loss
batch_size = settings.batch_size
lr = settings.lr

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10000)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)


load_path = f'{settings.path}/training'+'/level_one/'
save_path = f'{settings.path}/training'+'/level_two/'
path_split = save_path.split('/')
make_path = '../'
for i in range(1, len(path_split)):
    make_path = make_path + path_split[i] + '/'
    if not(os.path.exists(make_path)):
        os.mkdir(make_path)

level_one_input = ResNet_first_input()
level_one_template = ResNet_first_template()
level_two_input = ResNet_second_input()
level_two_template = ResNet_second_template()

level_one_input.load_weights(load_path + 'epoch_'+str(settings.train_stage1)+"input_full")
level_one_template.load_weights(load_path + 'epoch_'+str(settings.train_stage1)+"template_full")
  
initial_matrix = initial_motion(batch_size)
LK_layer_two = Lucas_Kanade_layer(batch_size=batch_size,height_template=64,width_template=64,num_channels=1)

dfhistory = pd.DataFrame(columns=['epoch', 'convex_loss', 'equ1_loss', 'equ2_loss', 'equ3_loss', 'ssim_loss', 'total_loss'])
for current_epoch in range(settings.train_stage2):
    if settings.dataset_name=='MSCOCO':
        data_loader_caller=data_loader_MSCOCO('train')
    if settings.dataset_name=='GoogleMap':
        data_loader_caller=data_loader_GoogleMap('train')
    if settings.dataset_name=='GoogleEarth':
        data_loader_caller=data_loader_GoogleEarth('train')
        
    optimizer = tf.keras.optimizers.Adam(lr=lr,beta_1=0.9)
    print("Starting epoch " + str(current_epoch+1))
    print("Learning rate is " + str(lr)) 

    loss_total = 0.0
    equ1_loss_total = 0.0
    equ2_loss_total = 0.0
    equ3_loss_total = 0.0
    ssim_loss_total = 0.0
    convex_loss_total = 0.0
    ssim_loss_total = 0.0
    count = 0
    for iters in range(100000000):
        input_img,u_list,v_list,template_img = data_loader_caller.data_read_batch(batch_size=batch_size)
        count += 1

        if len(np.shape(input_img))<2:
          level_two_input.save_weights(save_path +'epoch_'+str(1+current_epoch)+"input_full")
          level_two_template.save_weights(save_path +'epoch_'+str(1+current_epoch)+"template_full")
          break

        input_feature_one = level_one_input.call(input_img,training=False)
        template_feature_one = level_one_template.call(template_img,training=False)
  

        gt_matrix = gt_motion_rs(u_list,v_list,batch_size=batch_size)
        gt_matrix_two = construct_matrix(gt_matrix,scale_factor=0.5,batch_size=batch_size)

        with tf.GradientTape() as tape:
            input_feature = level_two_input.call(input_feature_one)
            template_feature = level_two_template.call(template_feature_one)

            input_feature_map_two = calculate_feature_map(input_feature)
            template_feature_map_two = calculate_feature_map(template_feature)

            input_warped_to_template_two=LK_layer_two.projective_inverse_warp(input_feature_map_two, gt_matrix_two)

            ssim_middle_two = tf.reduce_mean(compute_ssim(template_feature_map_two,input_warped_to_template_two))
            ssim_middle = ssim_middle_two

            for nn in range(settings.sample_noise):
                lambda_two=(np.random.rand(8)-0.5)/6.0
                for mm in range(len(lambda_two)):
                  if lambda_two[mm]>0 and lambda_two[mm]<0.02:
                    lambda_two[mm]=0.02
                  if lambda_two[mm]<0 and lambda_two[mm]>-0.02:
                    lambda_two[mm]=-0.02
                
                noise_two = gt_motion_rs_random_noisy(u_list,v_list,batch_size=batch_size,lambda_noisy=lambda_two)

                lambda_noisy_labels_two_plus =LK_layer_two.projective_inverse_warp(input_feature_map_two, gt_matrix_two + rho * noise_two)
                h_lambda_noisy_labels_two_plus = tf.reduce_mean(compute_ssim(template_feature_map_two, lambda_noisy_labels_two_plus))

                lambda_noisy_labels_two_mins = LK_layer_two.projective_inverse_warp(input_feature_map_two, gt_matrix_two - rho * noise_two)
                h_lambda_noisy_labels_two_mins = tf.reduce_mean(compute_ssim(template_feature_map_two,lambda_noisy_labels_two_mins))

                noisy_labels_two_plus = LK_layer_two.projective_inverse_warp(input_feature_map_two, gt_matrix_two + noise_two)
                h_noisy_labels_two_plus = tf.reduce_mean(compute_ssim(template_feature_map_two,noisy_labels_two_plus))

                noisy_labels_two_mins = LK_layer_two.projective_inverse_warp(input_feature_map_two, gt_matrix_two - noise_two)
                h_noisy_labels_two_mins = tf.reduce_mean(compute_ssim(template_feature_map_two,noisy_labels_two_mins))
                
                if settings.option == 'option1':
                    # equ1
                    equ1_plus= tf.math.maximum(ssim_middle_two - h_lambda_noisy_labels_two_plus + mu * (rho**2) * np.sum(lambda_two**2) / 2 ,0)
                    equ1_mins= tf.math.maximum(ssim_middle_two - h_lambda_noisy_labels_two_mins + mu * (rho**2) * np.sum(lambda_two**2) / 2 ,0)
                elif settings.option == 'option2':
                    # equ1
                    equ1_plus= tf.math.maximum(ssim_middle_two - h_lambda_noisy_labels_two_plus, 0)
                    equ1_mins= tf.math.maximum(ssim_middle_two - h_lambda_noisy_labels_two_mins, 0)
                # equ2
                equ2_plus = tf.math.maximum(ssim_middle_two + mu * np.sum(lambda_two**2) / 2 - h_noisy_labels_two_plus,0)
                equ2_mins = tf.math.maximum(ssim_middle_two + mu * np.sum(lambda_two**2) / 2 - h_noisy_labels_two_mins,0)
                # equ3
                equ3_plus = tf.math.maximum(h_lambda_noisy_labels_two_plus - (1 - rho) * ssim_middle_two - rho * h_noisy_labels_two_plus + mu * rho * (1 - rho) * np.sum(lambda_two**2) / 2 ,0)
                equ3_mins = tf.math.maximum(h_lambda_noisy_labels_two_mins - (1 - rho) * ssim_middle_two - rho * h_noisy_labels_two_mins + mu * rho * (1 - rho) * np.sum(lambda_two**2) / 2 ,0)
         
                if nn==0:
                    convex_loss = equ1_mins + equ1_plus + equ2_mins + equ2_plus + equ3_mins + equ3_plus
                    equ1_loss = equ1_mins + equ1_plus
                    equ2_loss = equ2_mins + equ2_plus
                    equ3_loss = equ3_mins + equ3_plus
                else:
                    convex_loss = convex_loss + equ1_mins + equ1_plus + equ2_mins + equ2_plus + equ3_mins + equ3_plus
                    equ1_loss = equ1_loss + equ1_mins + equ1_plus
                    equ2_loss = equ2_loss + equ2_mins + equ2_plus
                    equ3_loss = equ3_loss + equ3_mins + equ3_plus

            total_loss = ssim_middle + lambda_loss * convex_loss

            convex_loss_total += convex_loss
            equ1_loss_total += equ1_loss
            equ2_loss_total += equ2_loss
            equ3_loss_total += equ3_loss
            ssim_loss_total += ssim_middle
            loss_total += total_loss

 
        all_parameters=level_two_template.trainable_variables+level_two_input.trainable_variables
           
        grads = tape.gradient(total_loss, all_parameters)
        grads = [tf.clip_by_value(i,-0.1,0.1) for i in grads]
        optimizer.apply_gradients(zip(grads, all_parameters))

        
        input_img = None
        u_list = None
        v_list = None
        template_img = None

        input_feature_map = None
        template_feature_map = None
        input_warped_to_template=None
        input_warped_to_template_left_1=None
        input_warped_to_template_left_2=None
        input_warped_to_template_right_1=None
        input_warped_to_template_right_2=None

    # print('current_epoch', current_epoch)
    print('total_loss', loss_total)
    info = (int(current_epoch+1), float(convex_loss_total) / count, float(equ1_loss_total) / count, float(equ2_loss_total) / count, float(equ3_loss_total) / count, float(ssim_loss_total) / count, float(loss_total) / count)
    dfhistory.loc[current_epoch] = info
    dfhistory.to_csv(f'{settings.path}/training/dfhistory_level2.csv', index=False)
 
    loss_total=0.0
    convex_loss_total=0.0
    equ1_loss_total=0.0
    equ2_loss_total=0.0
    equ3_loss_total=0.0
    ssim_loss_total=0.0

    level_two_input.save_weights(save_path +'epoch_'+str(1+current_epoch)+"input_"+str(iters))
    level_two_template.save_weights(save_path +'epoch_'+str(1+current_epoch)+"template_"+str(iters))

       




