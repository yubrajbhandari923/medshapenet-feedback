from glob import glob
import numpy as np
import os
import tensorflow.compat.v1 as tf
import SimpleITK as sitk
import nibabel
import nibabel.processing
import argparse
from scipy.ndimage import zoom
tf.disable_v2_behavior()


import logging  
# logging.getLogger('tensorflow').setLevel(logging.ERROR)


logging.basicConfig(level=logging.INFO)

class auto_encoder(object):
    def __init__(self, sess, phase='train'):
        self.sess           = sess
        self.phase          = phase
        
        self.batch_size     = 1
        # self.input_size = (512, 256, 256)
        self.input_size = (128, 128, 128)
        # self.input_size = (256, 256, 256) # YUBRJA fixed *
        self.inputI_size    = 256
        self.inputI_chn     = 1
        self.output_chn     = 12

        self.lr             = 0.0001
        self.beta1          = 0.3
        self.epoch          = 500
        self.model_name     = 'n2.model'
        self.save_intval    = 10

        self.code_test = True

        self.build_model()
        self.chkpoint_dir   = f"/scratch/railabs/yb107/genPhantom/MSN_Training_Data/checkpoints/{self.model_name}/"
        # self.chkpoint_dir = "/home/yb107/cvit-work/genPhantom/medshapenet-feedback/benchmark_dataset/ckpt"
        self.train_data_dir = "/scratch/railabs/yb107/genPhantom/MSN_Training_Data/train/incomplete/" 
        self.train_label_dir = "/scratch/railabs/yb107/genPhantom/MSN_Training_Data/train/complete/"

        self.valid_data_dir = "/scratch/railabs/yb107/genPhantom/MSN_Training_Data/validation/incomplete/"
        self.valid_label_dir = "/scratch/railabs/yb107/genPhantom/MSN_Training_Data/validation/complete/"
 
        # self.test_label_dir="./Dtest4/dataset/0_ground_truth/lung/"
        self.test_data_dir = "/scratch/railabs/yb107/genPhantom/MSN_Training_Data/test/incomplete/" # YUBRJA fixed *
        # self.test_data_dir = "/scratch/railabs/yb107/genPhantom/MSN_Training_Data/test_one_case/incomplete/" # YUBRJA fixed *
        self.test_label_dir="/scratch/railabs/yb107/genPhantom/MSN_Training_Data/test/complete/"
        # self.test_label_dir="/scratch/railabs/yb107/genPhantom/MSN_Training_Data/test_one_case/complete/"
        
        self.curr_epoch = -1

        self.ckpt_name = 'ckpt'

        if self.code_test:
            self.epoch = 3
            self.save_intval = 1



    @property
    def save_output_dir(self):
        if self.phase == 'train':
            return f"/scratch/railabs/yb107/genPhantom/MSN_Training_Data/inferences/{self.model_name}/epoch_{self.curr_epoch}/output_multiclass/"
        # return f"/scratch/railabs/yb107/genPhantom/MSN_Training_Data/test_one_case/inferences/{self.ckpt_name}/output_multiclass/"
        return f"/scratch/railabs/yb107/genPhantom/MSN_Training_Data/inferences/{self.ckpt_name}/output_multiclass/"
        
    @property
    def save_residual_dir(self):
        return self.save_output_dir + "residuals/"

    def dice_loss_fun(self, pred, input_gt):
        input_gt = tf.one_hot(input_gt, 12)
        dice = 0
        for i in range(12):
            inse = tf.reduce_mean(pred[:, :, :, :, i]*input_gt[:, :, :, :, i])
            l = tf.reduce_sum(pred[:, :, :, :, i]*pred[:, :, :, :, i])
            r = tf.reduce_sum(input_gt[:, :, :, :, i] * input_gt[:, :, :, :, i])
            dice = dice + 2*inse/(l+r)
        return -dice



    def conv3d(self,input, output_chn, kernel_size, stride, use_bias=False, name='conv'):
        return tf.layers.conv3d(inputs=input, filters=output_chn, kernel_size=kernel_size, strides=stride,
                                padding='same', data_format='channels_last',
                                kernel_initializer=tf.truncated_normal_initializer(0.0, 0.01),use_bias=use_bias, name=name)


    def conv_bn_relu(self,input, output_chn, kernel_size, stride, use_bias, is_training, name):
        with tf.variable_scope(name):
            conv = self.conv3d(input, output_chn, kernel_size, stride, use_bias, name='conv')
            relu = tf.nn.relu(conv, name='relu')
        return relu



    def Deconv3d(self,input, output_chn, name):
        batch, in_depth, in_height, in_width, in_channels = [int(d) for d in input.get_shape()]
        filter = tf.get_variable(name+"/filter", shape=[4, 4, 4, output_chn, in_channels], dtype=tf.float32,
                                 initializer=tf.random_normal_initializer(0, 0.01))
        conv = tf.nn.conv3d_transpose(value=input, filter=filter, output_shape=[batch, in_depth * 2, in_height * 2, in_width * 2, output_chn],
                                      strides=[1, 2, 2, 2, 1], padding="SAME", name=name)
        return conv



    def deconv_bn_relu(self,input, output_chn, is_training, name):
        with tf.variable_scope(name):
            conv = self.Deconv3d(input, output_chn, name='deconv')
            relu = tf.nn.relu(conv, name='relu')
        return relu




    def build_model(self):
        logging.info('building patch based model...')       
        # self.input_I = tf.placeholder(dtype=tf.float32, shape=[self.batch_size,self.inputI_size,self.inputI_size,128, self.inputI_chn], name='inputI')
        # self.input_gt = tf.placeholder(dtype=tf.int64, shape=[self.batch_size,self.inputI_size,self.inputI_size,128,1], name='target')
        
        self.input_I = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.input_size[0], self.input_size[1],self.input_size[2], self.inputI_chn], name='inputI')
        self.input_gt = tf.placeholder(dtype=tf.int64, shape=[self.batch_size,self.input_size[0], self.input_size[1],self.input_size[2],1], name='target')

        self.soft_prob , self.task0_label = self.encoder_decoder(self.input_I)
        self.main_dice_loss = self.dice_loss_fun(self.soft_prob, self.input_gt[:,:,:,:,0])
        logging.info(f"Main Dice Loss: {self.main_dice_loss.numpy()}")
        self.dice_loss=200000000*self.main_dice_loss
        self.Loss = self.dice_loss
        logging.info(f"Loss: {self.Loss.numpy()}")
        self.saver = tf.train.Saver()
        exit()



    def encoder_decoder(self, inputI):
        phase_flag = (self.phase=='train')
        conv1_1 = self.conv3d(input=inputI, output_chn=64, kernel_size=3, stride=2, use_bias=True, name='conv1')
        conv1_relu = tf.nn.relu(conv1_1, name='conv1_relu')
        conv2_1 = self.conv3d(input=conv1_relu, output_chn=128, kernel_size=3, stride=2, use_bias=True, name='conv2')
        conv2_relu = tf.nn.relu(conv2_1, name='conv2_relu')
        conv3_1 = self.conv3d(input=conv2_relu, output_chn= 256, kernel_size=3, stride=2, use_bias=True, name='conv3a')
        conv3_relu = tf.nn.relu(conv3_1, name='conv3_1_relu')
        conv4_1 = self.conv3d(input=conv3_relu, output_chn=512, kernel_size=3, stride=2, use_bias=True, name='conv4a')
        conv4_relu = tf.nn.relu(conv4_1, name='conv4_1_relu')
        conv5_1 = self.conv3d(input=conv4_relu, output_chn=512, kernel_size=3, stride=1, use_bias=True, name='conv5a')
        conv5_relu = tf.nn.relu(conv5_1, name='conv5_1_relu')
        feature= self.conv_bn_relu(input=conv5_relu, output_chn=256, kernel_size=3, stride=1, use_bias=True, is_training=phase_flag, name='conv6_1')
        deconv1_1 = self.deconv_bn_relu(input=feature, output_chn=256, is_training=phase_flag, name='deconv1_1')
        deconv1_2 = self.conv_bn_relu(input=deconv1_1, output_chn=128, kernel_size=3, stride=1, use_bias=True, is_training=phase_flag, name='deconv1_2')
        deconv2_1 = self.deconv_bn_relu(input=deconv1_2, output_chn=128, is_training=phase_flag, name='deconv2_1')
        deconv2_2 = self.conv_bn_relu(input=deconv2_1, output_chn=64, kernel_size=3,stride=1, use_bias=True, is_training=phase_flag, name='deconv2_2')
        deconv3_1 = self.deconv_bn_relu(input=deconv2_2, output_chn=64, is_training=phase_flag, name='deconv3_1')
        deconv3_2 = self.conv_bn_relu(input=deconv3_1, output_chn=64, kernel_size=3, stride=1, use_bias=True, is_training=phase_flag, name='deconv3_2')
        deconv4_1 = self.deconv_bn_relu(input=deconv3_2, output_chn=32, is_training=phase_flag, name='deconv4_1')
        deconv4_2 = self.conv_bn_relu(input=deconv4_1, output_chn=32, kernel_size=3, stride=1, use_bias=True, is_training=phase_flag, name='deconv4_2')
        pred_prob1 = self.conv_bn_relu(input=deconv4_2, output_chn=self.output_chn, kernel_size=3, stride=1, use_bias=True, is_training=phase_flag, name='pred_prob1')
        pred_prob = self.conv3d(input=pred_prob1, output_chn=self.output_chn, kernel_size=3, stride=1, use_bias=True, name='pred_prob')
        pred_prob2 = self.conv3d(input=pred_prob, output_chn=self.output_chn, kernel_size=3, stride=1, use_bias=True, name='pred_prob2')
        pred_prob3 = self.conv3d(input=pred_prob2, output_chn=self.output_chn, kernel_size=3, stride=1, use_bias=True, name='pred_prob3')
        soft_prob=tf.nn.softmax(pred_prob3,name='task_0')
        task0_label=tf.argmax(soft_prob,axis=4,name='argmax0')
        return  soft_prob, task0_label


    def train(self):
        u_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=self.beta1).minimize(self.Loss)
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        counter=1

        train_label_list=glob('{}/*.nii.gz'.format(self.train_label_dir))
        # train_label_list=os.listdir(self.train_label_dir)

        for epoch in np.arange(self.epoch):            
            self.curr_epoch = epoch
            logging.info("=========================")
            logging.info(f'epoch:{epoch}' )

            if self.code_test:
                logging.info(f"Code Testing Mode: {self.code_test}")

                train_label_list = train_label_list[:10]

            for j in range(len(train_label_list)):
                labelImg=sitk.ReadImage(train_label_list[j])
                labelNpy=sitk.GetArrayFromImage(labelImg)
                labelNpy_resized=zoom(labelNpy,(self.input_size[0]/labelNpy.shape[0],self.input_size[1]/labelNpy.shape[1],self.input_size[2]/labelNpy.shape[2]),order=0, mode='constant')
                
                labelNpy_resized=np.expand_dims(np.expand_dims(labelNpy_resized,axis=0),axis=4) 
                # name=train_label_list[j][-len('_full.nii.gz')-len('s0556'):-len('_full.nii.gz')] # Name is just s0556
                
                
                name = train_label_list[j].replace('.nii.gz', '').split('/')[-1]
                

                # for k in range(10):
                files =glob(f"{self.train_data_dir}{name}/*")
                
                for data_dir in files:
                    logging.info(f"Training file: {data_dir}")
                        
                    # data_dir=self.train_data_dir+str(name)+'/'+str(name)+'_%d'%k+'.nii.gz' # Changed a . 
                    trainImg=sitk.ReadImage(data_dir)
                    trainNpy=sitk.GetArrayFromImage(trainImg)
                    
                    trainNpy_resized=zoom(trainNpy,(self.input_size[0]/trainNpy.shape[0],self.input_size[1]/trainNpy.shape[1],self.input_size[2]/trainNpy.shape[2]),order=0, mode='constant')
                    trainNpy_resized=np.expand_dims(np.expand_dims(trainNpy_resized,axis=0),axis=4) 
                    
                    logging.info(f"TrainNpy Shape: {trainNpy_resized.shape}")

                    _, cur_train_loss = self.sess.run([u_optimizer, self.Loss], feed_dict={self.input_I: trainNpy_resized, self.input_gt: labelNpy_resized})

                    train_output0 = self.sess.run(self.task0_label, feed_dict={self.input_I: trainNpy_resized})
                    
                    logging.info('sum for current training whole: %.8f, pred whole:  %.8f'%(np.sum(labelNpy_resized),np.sum(train_output0)))        
                logging.info(f'current training loss:{cur_train_loss}')
           
            counter+=1
            if np.mod(counter, self.save_intval) == 0:
                self.save_chkpoint(self.chkpoint_dir, self.model_name, counter)
                self.valid()

        self.save_chkpoint(self.chkpoint_dir, self.model_name, counter)

    def valid(self):
        valid_label_list=glob('{}/*.nii.gz'.format(self.valid_label_dir))
        
        if self.code_test:
            valid_label_list = valid_label_list[:10]

        total_valid_loss=0
        for j in range(len(valid_label_list)):
            labelImg=sitk.ReadImage(valid_label_list[j])
            labelNpy=sitk.GetArrayFromImage(labelImg)
            labelNpy_resized=zoom(labelNpy,(self.input_size[0]/labelNpy.shape[0],self.input_size[1]/labelNpy.shape[1],self.input_size[2]/labelNpy.shape[2]),order=0, mode='constant')
            labelNpy_resized=np.expand_dims(np.expand_dims(labelNpy_resized,axis=0),axis=4) 
            
            name = valid_label_list[j].replace('.nii.gz', '').split('/')[-1]
            
            files =glob(f"{self.valid_data_dir}{name}/*")
            valid_loss =0 
            
            for data_dir in files:
                
                validImg=sitk.ReadImage(data_dir)
                validNpy=sitk.GetArrayFromImage(validImg)
                validNpy_resized=zoom(validNpy,(self.input_size[0]/validNpy.shape[0],self.input_size[1]/validNpy.shape[1],self.input_size[2]/validNpy.shape[2]),order=0, mode='constant')
                validNpy_resized=np.expand_dims(np.expand_dims(validNpy_resized,axis=0),axis=4) 
                
                cur_valid_loss = self.sess.run(self.Loss, feed_dict={self.input_I: validNpy_resized, self.input_gt: labelNpy_resized})
                # valid_output0 = self.sess.run(self.task0_label, feed_dict={self.input_I: validNpy_resized})
                # logging.info('sum for current validation whole: %.8f, pred whole:  %.8f'%(np.sum(labelNpy_resized),np.sum(valid_output0)))        
                valid_loss+=cur_valid_loss
            
            logging.info(f'current validation loss for {os.path.basename(valid_label_list[j])} :{valid_loss/len(files)}')
            
            total_valid_loss+= valid_loss/len(files)
        
        logging.info(f'Total Average validation loss for Epoch {self.curr_epoch} :{total_valid_loss/len(valid_label_list)}')


    def test(self):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        if self.load_chkpoint(self.chkpoint_dir):
            logging.info(" *****Successfully load the checkpoint**********")
        else:
            logging.info("*******Fail to load the checkpoint***************")
        
        test_list=glob('{}/*/*.nii.gz'.format(self.test_data_dir))

        if self.code_test:
            test_list = test_list[:10]

        for i in range(len(test_list)):

            ### input                  
            test_img=sitk.ReadImage(test_list[i])
            test_input = sitk.GetArrayFromImage(test_img)
            # test_input_resized_ = zoom(test_input,(256/test_input.shape[0],256/test_input.shape[1],128/test_input.shape[2]),order=0, mode='constant')
            test_input_resized_ = zoom(test_input,(self.input_size[0]/test_input.shape[0],self.input_size[1]/test_input.shape[1],self.input_size[2]/test_input.shape[2]),order=0, mode='constant') # Yubraj: fixed 256 
            # logging.info(f"Test Input: {test_input.shape}")
            test_input_resized_[test_input_resized_>12]=0
            test_input_resized_[test_input_resized_<0]=0
            # logging.info(f'test_input_resized_ {np.unique(test_input_resized_)}')
            test_input_resized=np.expand_dims(np.expand_dims(test_input_resized_,axis=0),axis=4)


            ## prediction
            test_output = self.sess.run(self.task0_label, feed_dict={self.input_I: test_input_resized})
            # logging.info(f'Output Shape: {test_output.shape}')
            # logging.info(f'Predicted Unique: {np.unique(test_output)}')


            ## output
            os.makedirs(self.save_output_dir, exist_ok=True)
            os.makedirs(self.save_residual_dir, exist_ok=True)

            # filename=self.save_output_dir+test_list[i][-7-len('s0332_0'):-7]+'.nii.gz'
            filename=self.save_output_dir + os.path.basename(test_list[i])
            resize2original=1

            if resize2original:
                # logging.info('resizing predictions...')
                # logging.info(f"Test Output: {test_output}")
                # test_output=zoom(test_output[0],(test_input.shape[0]/256,test_input.shape[1]/256,test_input.shape[2]/128),order=0, mode='constant')
                # test_output=zoom(test_output[0],(test_input.shape[0]/128,test_input.shape[1]/128,test_input.shape[2]/128),order=0, mode='constant') # Yubraj: fixed 256
                
                test_output = test_output[0]
                logging.info(test_output.shape)

                test_output[test_output>12]=0
                test_output[test_output<0]=0
                test_pred=sitk.GetImageFromArray(test_output.astype('int32'))
                # test_pred.CopyInformation(test_img)
                sitk.WriteImage(test_pred,filename)

            else:
                logging.info('resizing input...')
                #test_img_downsampled=self.downsamplePatient(test_img,test_input.shape[0]/256,test_input.shape[1]/256,test_input.shape[2]/128)
                logging.info('resizing done...')

                input_img = nibabel.load(test_list[i])

                voxel_size=input_img.header.get_zooms()
                # voxel_size_new=[voxel_size[0]*(test_input.shape[0]/256),voxel_size[1]*(test_input.shape[1]/256),voxel_size[2]*(test_input.shape[2]/128)]
                voxel_size_new=[voxel_size[0]*(test_input.shape[0]/self.input_size[0]),voxel_size[1]*(test_input.shape[1]/self.input_size[1]),voxel_size[2]*(test_input.shape[2]/self.input_size[2])]
                resampled_img = nibabel.processing.resample_to_output(input_img, voxel_size_new)
                # filename_img=self.save_output_dir+test_list[i][-7-len('s0332_0'):-7]+'_org'+'.nii.gz'
                filename_img=self.save_output_dir + os.path.basename(test_list[i])
                nibabel.save(resampled_img, filename_img)


                test_pred=sitk.GetImageFromArray(test_output[0].astype('int32'))
                sitk.WriteImage(test_pred,filename)


            
            #filename_res=self.save_residual_dir+test_list[i][-7-len('s0332_0'):-7]+'.nii.gz'
            #res_output=test_output-test_input
            #res_output_img=sitk.GetImageFromArray(res_output.astype('int32'))
            #res_output_img.CopyInformation(test_img)
            #sitk.WriteImage(res_output_img,filename_res)



    def save_chkpoint(self, checkpoint_dir, model_name, step):
        logging.info(" [*] Saving checkpoint to {}...".format(checkpoint_dir))
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)



    def load_chkpoint(self, checkpoint_dir):
        logging.info(" [*] Reading checkpoint...")
        logging.info('########################################################')
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.ckpt_name = ckpt_name
            logging.info('##############ckpt_name:{}'.format(ckpt_name))
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False







if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase")
    args = parser.parse_args()

    sess1 = tf.compat.v1.Session()
    with sess1.as_default():
        with sess1.graph.as_default():
            model = auto_encoder(sess1, phase=args.phase)
            total_parameters = 0
            for variable in tf.trainable_variables():
                shape = variable.get_shape()
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                total_parameters += variable_parameters
            logging.info(f'trainable params:{total_parameters}')

    if args.phase == "train":
        logging.info('training model...')
        model.train()
    if args.phase == "test":
        logging.info('testing model...')
        model.test()
