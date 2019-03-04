"""Visual Attention Based OCR Model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random, time, os, shutil, math, sys, logging
import numpy as np
from six.moves import xrange
from PIL import Image
import tensorflow as tf
from pypylon import pylon
import cv2

from .cnn import CNN
from .seq2seq_model import Seq2SeqModel
from data_util.data_gen import DataGen

try:
    import distance
    distance_loaded = True
except ImportError:
    distance_loaded = False

def my_visual(word):
    rendering = ''.join([chr(c - 13 + 97) if c - 13 + 97 > 96 else chr(c - 3 + 48) for c in word])
    return rendering


class Model(object):

    def __init__(self,
            phase,
            visualize,
            data_path,
            data_base_dir,
            output_dir,
            batch_size,
            initial_learning_rate,
            num_epoch,
            steps_per_checkpoint,
            target_vocab_size, 
            model_dir, 
            target_embedding_size,
            attn_num_hidden, 
            attn_num_layers,
            clip_gradients,
            max_gradient_norm,
            session,
            load_model,
            gpu_id,
            use_gru,
            evaluate=False,
            valid_target_length=float('inf'),
            reg_val = 0 ):
            
        gpu_device_id = '/gpu:' + str(gpu_id)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        logging.info('loading data')
        # load data
        if phase == 'train':
            self.s_gen = DataGen(
                data_base_dir, data_path, valid_target_len=valid_target_length, evaluate=False)
        else:
            batch_size = 1
            self.s_gen = DataGen(
                data_base_dir, data_path, evaluate=True)


        #logging.info('valid_target_length: %s' %(str(valid_target_length)))
        logging.info('phase: %s' % phase)
        logging.info('model_dir: %s' % (model_dir))
        logging.info('load_model: %s' % (load_model))
        logging.info('output_dir: %s' % (output_dir))
        logging.info('steps_per_checkpoint: %d' % (steps_per_checkpoint))
        logging.info('batch_size: %d' %(batch_size))
        logging.info('num_epoch: %d' %num_epoch)
        logging.info('learning_rate: %d' % initial_learning_rate)
        logging.info('reg_val: %d' % (reg_val))
        logging.info('max_gradient_norm: %f' % max_gradient_norm)
        logging.info('clip_gradients: %s' % clip_gradients)
        logging.info('valid_target_length %f' %valid_target_length)
        logging.info('target_vocab_size: %d' %target_vocab_size)
        logging.info('target_embedding_size: %f' % target_embedding_size)
        logging.info('attn_num_hidden: %d' % attn_num_hidden)
        logging.info('attn_num_layers: %d' % attn_num_layers)
        logging.info('visualize: %s' % visualize)

        buckets = self.s_gen.bucket_specs
        logging.info('buckets')
        logging.info(buckets)
        if use_gru:
            logging.info('ues GRU in the decoder.')

        # variables
        self.img_data = tf.placeholder(tf.float32, shape=(None, 1, 32, None), name='img_data')
        self.zero_paddings = tf.placeholder(tf.float32, shape=(None, None, 512), name='zero_paddings')
        
        self.decoder_inputs = []
        self.encoder_masks = []
        self.target_weights = []
        for i in xrange(int(buckets[-1][0] + 1)):
            self.encoder_masks.append(tf.placeholder(tf.float32, shape=[None, 1],
                                                    name="encoder_mask{0}".format(i)))
        for i in xrange(buckets[-1][1] + 1):
            self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                    name="decoder{0}".format(i)))
            self.target_weights.append(tf.placeholder(tf.float32, shape=[None],
                                                    name="weight{0}".format(i)))
      
        self.reg_val = reg_val
        self.sess = session
        self.evaluate = evaluate
        self.steps_per_checkpoint = steps_per_checkpoint 
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.buckets = buckets
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.global_step = tf.Variable(0, trainable=False)
        self.valid_target_length = valid_target_length
        self.phase = phase
        self.visualize = visualize
        self.learning_rate = initial_learning_rate
        self.clip_gradients = clip_gradients
       
        if phase == 'train':
            self.forward_only = False
        elif phase == 'test':
            self.forward_only = True
        else:
            assert False, phase

        with tf.device(gpu_device_id):
            cnn_model = CNN(self.img_data, True) #(not self.forward_only))
            self.conv_output = cnn_model.tf_output()
            self.concat_conv_output = tf.concat(axis=1, values=[self.conv_output, self.zero_paddings])
            self.perm_conv_output = tf.transpose(self.concat_conv_output, perm=[1, 0, 2])

        with tf.device(gpu_device_id):
            self.attention_decoder_model = Seq2SeqModel(
                encoder_masks = self.encoder_masks,
                encoder_inputs_tensor = self.perm_conv_output, 
                decoder_inputs = self.decoder_inputs,
                target_weights = self.target_weights,
                target_vocab_size = target_vocab_size, 
                buckets = buckets,
                target_embedding_size = target_embedding_size,
                attn_num_layers = attn_num_layers,
                attn_num_hidden = attn_num_hidden,
                forward_only = self.forward_only,
                use_gru = use_gru)

        if not self.forward_only:
            self.updates = []
            self.summaries_by_bucket = []
            with tf.device(gpu_device_id):
                params = tf.trainable_variables()
                # Gradients and SGD update operation for training the model.
                opt = tf.train.AdadeltaOptimizer(learning_rate=initial_learning_rate)
                for b in xrange(len(buckets)):
                    if self.reg_val > 0:
                        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                        logging.info('Adding %s regularization losses', len(reg_losses))
                        logging.debug('REGULARIZATION_LOSSES: %s', reg_losses)
                        loss_op = self.reg_val * tf.reduce_sum(reg_losses) + self.attention_decoder_model.losses[b]
                    else:
                        loss_op = self.attention_decoder_model.losses[b]

                    gradients, params = zip(*opt.compute_gradients(loss_op, params))
                    if self.clip_gradients:
                        gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)
                    # Add summaries for loss, variables, gradients, gradient norms and total gradient norm.
                    summaries = []
                    summaries.append(tf.summary.scalar("loss", loss_op))
                    summaries.append(tf.summary.scalar("total_gradient_norm", tf.global_norm(gradients)))
                    all_summaries = tf.summary.merge(summaries)
                    self.summaries_by_bucket.append(all_summaries)
                    # update op - apply gradients
                    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                    with tf.control_dependencies(update_ops):
                        self.updates.append(opt.apply_gradients(zip(gradients, params), global_step=self.global_step))

        self.saver_all = tf.train.Saver(tf.all_variables())

        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and load_model:
            logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            #self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            self.saver_all.restore(self.sess, ckpt.model_checkpoint_path)
            '''all_variable_name = [v.name for v in tf.global_variables()]
            print(all_variable_name)'''
            # frozen = tf.graph_util.convert_variables_to_constants(self.sess, self.sess.graph_def,
            #                   ['embedding_attention_decoder/attention_decoder/AttnOutputProjection/bias:0'])
            # graph_io.write_graph(frozen, '.', 'attention_ocr.pb', as_text=False)
        else:
            logging.info("Created model with fresh parameters.")
            self.sess.run(tf.initialize_all_variables())
        #self.sess.run(init_new_vars_op)

        # train or test as specified by phase
    def launch(self):
        camera = 0
        if camera:
            # basler init
            # conecting to the first available camera
            camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
            # Grabing Continusely (video) with minimal delay
            camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            converter = pylon.ImageFormatConverter()
            # converting to opencv bgr format
            converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

            cv2.namedWindow('show', cv2.WINDOW_NORMAL)
            cv2.namedWindow('bin', cv2.WINDOW_NORMAL)

            # writer = tf.summary.FileWriter(self.model_dir, self.sess.graph)
            while camera.IsGrabbing():
                grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                if grabResult.GrabSucceeded():
                    # Access the image data
                    __image = converter.Convert(grabResult)
                    image_RGB = __image.GetArray()
                    image_GRAY = cv2.cvtColor(image_RGB, cv2.COLOR_BGR2GRAY)
                    _1, image_BIN = cv2.threshold(image_GRAY, 117, 255, cv2.THRESH_BINARY)
                    _1, contours, _2 = cv2.findContours(image_BIN, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for c in contours:
                        area = cv2.contourArea(c)
                        if area > 5000 and area < 15000:
                            rect = cv2.boundingRect(c)  # 返回左上角坐标以及矩形高度和宽度
                            roi = image_GRAY[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]

                            batch = self.s_gen.gen(self.batch_size, roi)
                            # Get a batch and make a step.
                            start_time = time.time()
                            bucket_id = batch['bucket_id']
                            img_data = batch['data']
                            zero_paddings = batch['zero_paddings']
                            decoder_inputs = batch['decoder_inputs']
                            target_weights = batch['target_weights']
                            encoder_masks = batch['encoder_mask']
                            file_list = batch['filenames']
                            real_len = batch['real_len']
                            start_time = time.time()
                            _, step_loss, step_logits, step_attns = self.step(encoder_masks, img_data, zero_paddings,
                                                                              decoder_inputs, target_weights, bucket_id,
                                                                              self.forward_only)
                            #print(step_logits)
                            infer_time = (time.time() - start_time)
                            print("infer_time:",round(infer_time*1000,1),'ms')
                            step_outputs = [b for b in np.array(
                                [np.argmax(logit, axis=1).tolist() for logit in step_logits]).transpose()]
                            step_output = step_outputs[0]
                            output_valid = []
                            for j in range(len(step_output)):
                                s1 = step_output[j]
                                if s1 == 2:
                                    break
                                else:
                                    output_valid.append(s1)
                            pred_str = my_visual(output_valid)
                            #print('pred:', pred_str)
                            step_time = time.time() - start_time
                            print("step_time:", round(step_time * 1000, 1),'ms')
                            #print("infer time:%s ms,step time:%s ms" %(round(infer_time * 1000, 1), round(step_time * 1000, 1)))
                            cv2.putText(image_RGB, pred_str, (rect[0], rect[1]), cv2.FONT_HERSHEY_COMPLEX,
                                        fontScale=1.2, color=(255, 0, 0), thickness=2)
                            #cv2.putText(image_RGB, pred_str, (rect[0], rect[1]), cv2.FONT_HERSHEY_COMPLEX,
                            #            fontScale=2.5, color=(255,0,0), thickness=3)
                            #roi = cv2.resize(roi, (int(roi.shape[1]/roi.shape[0]*32),32))
                            #cv2.imshow('roi', roi)
                            #print(step_output)
                    cv2.imshow('bin', image_BIN)
                    cv2.imshow('show', image_RGB)
                    k = cv2.waitKey(1)
                    if k == ord('q'):
                        break
                    elif k == ord('c'):
                        cv2.imwrite('result.png', image_RGB)
                grabResult.Release()
            # Releasing the resource
            camera.StopGrabbing()
        else:
            image = cv2.imread('/home/xuxin/simple_data/crnn/num3_4_200/Test/0.jpg', 0)
            batch = self.s_gen.gen(self.batch_size, image)
            # Get a batch and make a step.
            bucket_id = batch['bucket_id']
            img_data = batch['data']
            zero_paddings = batch['zero_paddings']
            decoder_inputs = batch['decoder_inputs']
            target_weights = batch['target_weights']
            encoder_masks = batch['encoder_mask']
            #file_list = batch['filenames']
            #real_len = batch['real_len']

            _, step_loss, step_logits, step_attns = self.step(encoder_masks, img_data, zero_paddings,
                                                              decoder_inputs, target_weights, bucket_id,
                                                              self.forward_only)

            step_outputs = [b for b in np.array(
                [np.argmax(logit, axis=1).tolist() for logit in step_logits]).transpose()]
            step_output = step_outputs[0]
            output_valid = []
            for j in range(len(step_output)):
                s1 = step_output[j]
                if s1 == 2:
                    break
                else:
                    output_valid.append(s1)
            pred_str = my_visual(output_valid)
            print('pred:', pred_str)


    # step, read one batch, generate gradients
    def step(self, encoder_masks, img_data, zero_paddings, decoder_inputs, target_weights,
               bucket_id, forward_only):
        # Check if the sizes match.
        encoder_size, decoder_size = self.buckets[bucket_id]
        if len(decoder_inputs) != decoder_size:
            raise ValueError("Decoder length must be equal to the one in bucket,"
                    " %d != %d." % (len(decoder_inputs), decoder_size))
        if len(target_weights) != decoder_size:
            raise ValueError("Weights length must be equal to the one in bucket,"
                    " %d != %d." % (len(target_weights), decoder_size))
        
        # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
        input_feed = {}
        input_feed[self.img_data.name] = img_data
        input_feed[self.zero_paddings.name] = zero_paddings
        for l in xrange(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.target_weights[l].name] = target_weights[l]
        for l in xrange(int(encoder_size)):
            try:
                input_feed[self.encoder_masks[l].name] = encoder_masks[l]
            except Exception as e:
                pass
                #ipdb.set_trace()
    
        # Since our targets are decoder inputs shifted by one, we need one more.
        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)
    
        # Output feed: depends on whether we do a backward step or not.
        if not forward_only:
            output_feed  = [self.updates[bucket_id],  # Update Op that does SGD.
                    #self.gradient_norms[bucket_id],  # Gradient norm.
                    self.attention_decoder_model.losses[bucket_id],
                             self.summaries_by_bucket[bucket_id]]
            for l in xrange(decoder_size):  # Output logits.
                output_feed.append(self.attention_decoder_model.outputs[bucket_id][l])
        else:
            output_feed = [self.attention_decoder_model.losses[bucket_id]]  # Loss for this batch.
            for l in xrange(decoder_size):  # Output logits.
                output_feed.append(self.attention_decoder_model.outputs[bucket_id][l])
            if self.visualize:
                output_feed += self.attention_decoder_model.attention_weights_histories[bucket_id]
        # for i in output_feed:
        #     print(i)
        outputs = self.sess.run(output_feed, input_feed)

        if not forward_only:
            return outputs[2], outputs[1], outputs[3:(3+self.buckets[bucket_id][1])], None  # Gradient norm summary, loss, no outputs, no attentions.
        else:
            return None, outputs[0], outputs[1:(1+self.buckets[bucket_id][1])], outputs[(1+self.buckets[bucket_id][1]):]  # No gradient norm, loss, outputs, attentions.


