import sys
sys.path.append('../')
from data_read import *
from net import *
import numpy as np
import os
import pandas as pd
import settings
from test_utils import initial_motion, gt_motion_rs, calculate_feature_map, compute_ssim, gt_motion_rs_random_noisy, construct_matrix, construct_matrix_regression, average_cornner_error 

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=9000)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)


load_path_one= f'{settings.path}/training'+'/level_one/'
load_path_two= f'{settings.path}/training'+'/level_two/'
load_path_three= f'{settings.path}/training'+'/level_three/'

level_one_input=ResNet_first_input()
level_one_template=ResNet_first_template()
level_two_input=ResNet_second_input()
level_two_template=ResNet_second_template()
level_three_input=ResNet_third_input()
level_three_template=ResNet_third_template()

level_one_input.load_weights(load_path_one + 'epoch_'+str(settings.train_stage1)+"input_full")
level_one_template.load_weights(load_path_one + 'epoch_'+str(settings.train_stage1)+"template_full")
level_two_input.load_weights(load_path_two + 'epoch_'+str(settings.train_stage2)+"input_full")
level_two_template.load_weights(load_path_two  + 'epoch_'+str(settings.train_stage2)+"template_full")
level_three_input.load_weights(load_path_three + 'epoch_'+str(settings.train_stage3)+"input_full")
level_three_template.load_weights(load_path_three  + 'epoch_'+str(settings.train_stage3)+"template_full")

save_path_one='./checkpoints/' + settings.dataset_name + '/regression_stage_1/'
save_path_two='./checkpoints/' + settings.dataset_name + '/regression_stage_2/'
save_path_three='./checkpoints/' + settings.dataset_name + '/regression_stage_3/'
regression_network_one=Net_first()
regression_network_one.load_weights(save_path_one + 'epoch_'+str(settings.load_epoch_regression[0]))
regression_network_two=Net_second()
regression_network_two.load_weights(save_path_two + 'epoch_'+str(settings.load_epoch_regression[1]))
regression_network_three=Net_third()
regression_network_three.load_weights(save_path_three + 'epoch_'+str(settings.load_epoch_regression[2]))

LK_layer_one=Lucas_Kanade_layer(batch_size=1,height_template=128,width_template=128,num_channels=1)
LK_layer_two=Lucas_Kanade_layer(batch_size=1,height_template=64,width_template=64,num_channels=1)
LK_layer_three=Lucas_Kanade_layer(batch_size=1,height_template=32,width_template=32,num_channels=1)

LK_layer_regression=Lucas_Kanade_layer(batch_size=1,height_template=192,width_template=192,num_channels=3)

if settings.dataset_name=='MSCOCO':
    data_loader_caller=data_loader_MSCOCO('val')
if settings.dataset_name=='GoogleMap':
    data_loader_caller=data_loader_GoogleMap('val')
if settings.dataset_name=='GoogleEarth':
    data_loader_caller=data_loader_GoogleEarth('val')  

dfhistory = pd.DataFrame(columns=['index', 'corner_error'], dtype=np.float16)
for iters in range(10000000):
    input_img,u_list,v_list,template_img=data_loader_caller.data_read_batch(batch_size=1)
    if len(np.shape(input_img))<2:
        break

    input_img_grey=tf.image.rgb_to_grayscale(input_img)
    template_img_new=tf.image.pad_to_bounding_box(template_img, 32, 32, 192, 192)  
    template_img_grey=tf.image.rgb_to_grayscale(template_img_new)
    network_input=tf.concat([template_img_grey,input_img_grey],axis=-1)
    
    homography_vector_one=regression_network_one.call(network_input,training=False)

    matrix_one=construct_matrix_regression(1,homography_vector_one)
    template_img_new=LK_layer_regression.projective_inverse_warp(tf.dtypes.cast(template_img,tf.float32), matrix_one)
    template_img_grey=tf.image.rgb_to_grayscale(template_img_new) 
    network_input=tf.concat([template_img_grey,input_img_grey],axis=-1)
    homography_vector_two=regression_network_two.call(network_input,training=False)
    matrix_two=construct_matrix_regression(1,homography_vector_one,homography_vector_two)
    template_img_new=LK_layer_regression.projective_inverse_warp(tf.dtypes.cast(template_img,tf.float32), matrix_two)
    template_img_grey=tf.image.rgb_to_grayscale(template_img_new)  
    network_input=tf.concat([template_img_grey,input_img_grey],axis=-1)
    homography_vector_three=regression_network_three.call(network_input,training=False)

    extra=tf.ones((1,1))
    initial_matrix=tf.concat([homography_vector_three,extra],axis=-1)
    initial_matrix=tf.reshape(initial_matrix,[1,3,3])
    initial_matrix=np.dot(initial_matrix[0,:,:], np.linalg.inv(matrix_two[0,:,:]))
    initial_matrix=np.expand_dims(initial_matrix,axis=0)
    cornner_error_pre=average_cornner_error(1,initial_matrix,u_list,v_list,top_left_u=0,top_left_v=0,bottom_right_u=127,bottom_right_v=127)
    initial_matrix=construct_matrix(initial_matrix,scale_factor=0.25,batch_size=1)

    input_feature_one=level_one_input.call(input_img,training=False)
    template_feature_one=level_one_template.call(template_img,training=False)
    input_feature_two=level_two_input.call(input_feature_one,training=False)
    template_feature_two=level_two_template.call(template_feature_one,training=False)
    input_feature_three=level_three_input.call(input_feature_two,training=False)
    template_feature_three=level_three_template.call(template_feature_two,training=False)
                
    input_feature_map_one=calculate_feature_map(input_feature_one)
    template_feature_map_one=calculate_feature_map(template_feature_one)
    input_feature_map_two=calculate_feature_map(input_feature_two)
    template_feature_map_two=calculate_feature_map(template_feature_two)
    input_feature_map_three=calculate_feature_map(input_feature_three)
    template_feature_map_three=calculate_feature_map(template_feature_three)


    cornner_error_previous=0.0
    updated_matrix=initial_matrix
    if True:
        for j in range(settings.num_iters):
            try:
                updated_matrix=LK_layer_three.update_matrix(template_feature_map_three,input_feature_map_three,updated_matrix)
                updated_matrix_back=construct_matrix(updated_matrix,scale_factor=4.0,batch_size=1)
                cornner_error=average_cornner_error(1,updated_matrix_back,u_list,v_list,top_left_u=0,top_left_v=0,bottom_right_u=127,bottom_right_v=127)
                if np.abs(cornner_error-cornner_error_previous)<1.0:
                    break
                cornner_error_previous=cornner_error
            except:
                print ('s')

        cornner_error_previous=0.0
        updated_matrix=construct_matrix(updated_matrix,scale_factor=2.0,batch_size=1)
        for j in range(settings.num_iters):
            try:
                updated_matrix=LK_layer_two.update_matrix(template_feature_map_two,input_feature_map_two,updated_matrix)
                updated_matrix_back=construct_matrix(updated_matrix,scale_factor=2.0,batch_size=1)
                cornner_error=average_cornner_error(1,updated_matrix_back,u_list,v_list,top_left_u=0,top_left_v=0,bottom_right_u=127,bottom_right_v=127)
                if np.abs(cornner_error-cornner_error_previous)<0.1:
                    break
                cornner_error_previous=cornner_error
            except:
                print ('s')

        cornner_error_previous=0.0
        updated_matrix=construct_matrix(updated_matrix,scale_factor=2.0,batch_size=1)
        for j in range(settings.num_iters):
            try:
                updated_matrix=LK_layer_one.update_matrix(template_feature_map_one,input_feature_map_one,updated_matrix)
                cornner_error=average_cornner_error(1,updated_matrix,u_list,v_list,top_left_u=0,top_left_v=0,bottom_right_u=127,bottom_right_v=127)
                if np.abs(cornner_error-cornner_error_previous)<0.01:
                    break
                cornner_error_previous=cornner_error
            except:
                print ('s')


    predicted_matrix=updated_matrix
    cornner_error=average_cornner_error(1,predicted_matrix,u_list,v_list,top_left_u=0,top_left_v=0,bottom_right_u=127,bottom_right_v=127)

    if np.abs(np.float(cornner_error)-np.float(cornner_error_pre))>25.0:
        cornner_error=min(np.float(cornner_error),np.float(cornner_error_pre))
    
    print (iters)
    print (cornner_error)
    info = (int(iters)+1, np.float(cornner_error))
    dfhistory.loc[iters] = info
    dfhistory.to_csv(f'{settings.path}/training/test.csv', index=False)

            
            

               




