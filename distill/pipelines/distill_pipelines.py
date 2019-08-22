import tensorflow as tf
import numpy as np
from distill.data_util.prep_sst import SST
from distill.data_util.vocab import PretrainedVocab
from distill.common.distill_util import get_single_state_rsa_distill_loss, get_logit_distill_loss, \
  get_single_state_uniform_rsa_loss, get_biased_single_state_rsa_distill_loss, get_rep_sim
from distill.pipelines.seq2seq import Seq2SeqTrainer


class Distiller(object):
  def __init__(self, config, student_model, teacher_model):
    self.config = config

    self.student = student_model
    self.teacher = teacher_model

  def build_train_graph(self):
    raise NotImplementedError()

  def train(self):
    g = tf.Graph()
    with g.as_default():
      update_op, distill_logit_op, distill_rep_op, student_update_op, scaffold  = self.build_train_graph()
      ops = []
      if self.config.train_teacher:
        ops.append(update_op)
      if self.config.train_student:
        ops.append(student_update_op)
#      if self.config.distill_rep:
#        ops.append(distill_rep_op)
#      if self.config.distill_logit:
#        ops.append(distill_logit_op)
        
      with tf.train.MonitoredTrainingSession(checkpoint_dir=self.config.save_dir, scaffold=scaffold, save_summaries_steps=1000, log_step_count_steps=500) as sess:
        for _ in np.arange(self.config.training_iterations):
            sess.run(ops)
            
class SSTDistiller(Distiller):
  def __init__(self, config, student_model, teacher_model):
    super(SSTDistiller, self).__init__(config, student_model, teacher_model)

    self.sst = SST("data/sst")
    self.config.vocab_size = len(self.sst.vocab)
    self.student.hparams.vocab_size = self.config.vocab_size
    self.teacher.hparams.vocab_size = self.config.vocab_size

    self.vocab = PretrainedVocab(self.config.data_path, self.config.pretrained_embedding_path,
                                 self.config.embedding_dim)
    self.pretrained_word_embeddings, self.word2id = self.vocab.get_word_embeddings()



  def get_train_op(self, loss, params, start_learning_rate, base_learning_rate, warmup_steps, scope=""):
    # add training op
    with tf.variable_scope(scope):
      self.global_step = tf.train.get_or_create_global_step()

      loss_l2 = tf.add_n([tf.nn.l2_loss(p) for p in params]) * self.config.l2_rate

      loss += loss_l2

      slope = (base_learning_rate - start_learning_rate) / warmup_steps
      warmup_rate = slope * tf.cast(self.global_step,
                                    tf.float32) + start_learning_rate

      decay_learning_rate = tf.train.exponential_decay(base_learning_rate, self.global_step,
                                                 1000, 0.96, staircase=True)
      learning_rate = tf.where(self.global_step < warmup_steps, warmup_rate,
                               decay_learning_rate)


      opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
      grads_and_vars = opt.compute_gradients(loss, params)
      gradients, variables = zip(*grads_and_vars)
      self.gradient_norm = tf.global_norm(gradients)
      clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5)
      self.param_norm = tf.global_norm(params)

      # Include batch norm mean and variance in gradient descent updates
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
        # Fetch self.updates to apply gradients to all trainable parameters.
        updates = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

    return updates, learning_rate

  def get_data_itaratoes(self):
    with tf.device('/cpu:0'):
      dataset = tf.data.TFRecordDataset(SST.get_tfrecord_path("data/sst", mode="train"))
      dataset = dataset.map(SST.parse_full_sst_tree_examples)
      dataset = dataset.padded_batch(self.config.batch_size, padded_shapes=SST.get_padded_shapes(), drop_remainder=True)
      dataset = dataset.shuffle(buffer_size=1000)
      dataset = dataset.repeat()
      iterator = dataset.make_initializable_iterator()
    
      dev_dataset = tf.data.TFRecordDataset(SST.get_tfrecord_path("data/sst", mode="dev"))
      dev_dataset = dev_dataset.map(SST.parse_full_sst_tree_examples)
      dev_dataset = dev_dataset.shuffle(buffer_size=1101)
      dev_dataset = dev_dataset.repeat()
      dev_dataset = dev_dataset.padded_batch(1101, padded_shapes=SST.get_padded_shapes(),
                                               drop_remainder=True)
      dev_iterator = dev_dataset.make_initializable_iterator()
    
      test_dataset = tf.data.TFRecordDataset(SST.get_tfrecord_path("data/sst", mode="test"))
      test_dataset = test_dataset.map(SST.parse_full_sst_tree_examples)
      test_dataset = test_dataset.shuffle(buffer_size=2210)
      test_dataset = test_dataset.repeat()
      test_dataset = test_dataset.padded_batch(2210, padded_shapes=SST.get_padded_shapes(),
                                                 drop_remainder=True)
      test_iterator = test_dataset.make_initializable_iterator()
    
      return iterator, dev_iterator, test_iterator

  def build_train_graph(self):
    self.student.build_graph(self.pretrained_word_embeddings)
    self.teacher.build_graph(self.pretrained_word_embeddings)

    train_iterator, dev_iterator, test_iterator = self.get_data_itaratoes()

    train_examples = train_iterator.get_next()
    dev_examples = dev_iterator.get_next()
    test_examples =  test_iterator.get_next()

    student_train_output_dic = self.student.apply(train_examples)
    teacher_train_output_dic = self.teacher.apply(train_examples)

    student_dev_output_dic = self.student.apply(dev_examples)
    teacher_dev_output_dic = self.teacher.apply(dev_examples)

    student_test_output_dic = self.student.apply(test_examples)
    teacher_test_output_dic = self.teacher.apply(test_examples)

    tf.summary.scalar("loss", student_train_output_dic[self.config.loss_type], family="student_train")
    tf.summary.scalar("accuracy", student_train_output_dic["root_accuracy"], family="student_train")

    tf.summary.scalar("loss", student_dev_output_dic[self.config.loss_type], family="student_dev")
    tf.summary.scalar("accuracy", student_dev_output_dic["root_accuracy"], family="student_dev")

    tf.summary.scalar("loss", student_test_output_dic[self.config.loss_type], family="student_test")
    tf.summary.scalar("accuracy", student_test_output_dic["root_accuracy"], family="student_test")

    tf.summary.scalar("loss", teacher_train_output_dic[self.config.loss_type], family="teacher_train")
    tf.summary.scalar("accuracy", teacher_train_output_dic["root_accuracy"], family="teacher_train")

    tf.summary.scalar("loss", teacher_dev_output_dic[self.config.loss_type], family="teacher_dev")
    tf.summary.scalar("accuracy", teacher_dev_output_dic["root_accuracy"], family="teacher_dev")

    tf.summary.scalar("loss", teacher_test_output_dic[self.config.loss_type], family="teacher_test")
    tf.summary.scalar("accuracy", teacher_test_output_dic["root_accuracy"], family="teacher_test")


    update_op, teacher_learning_rate = self.get_train_op(teacher_train_output_dic[self.config.loss_type],
                                                 teacher_train_output_dic["trainable_vars"],
                                                 start_learning_rate=self.teacher.hparams.learning_rate/20,
                                                 base_learning_rate=self.teacher.hparams.learning_rate,#0.001, 
                                                 warmup_steps=1000,
                                                 scope="main")

    distill_loss = get_logit_distill_loss(student_train_output_dic['logits'],teacher_train_output_dic['logits'])
    tf.summary.scalar("distill loss", distill_loss, family="student_train")

    distill_op, distill_learning_rate = self.get_train_op(distill_loss, student_train_output_dic["trainable_vars"],
                                                  start_learning_rate=self.student.hparams.learning_rate/20,
                                                  base_learning_rate=self.student.hparams.learning_rate,#0.001, 
                                                  warmup_steps=10000,
                                                  scope="distill")

    tf.summary.scalar("learning_rate", teacher_learning_rate, family="teacher_train")
    tf.summary.scalar("learning_rate", distill_learning_rate, family="student_train")



    scaffold = tf.train.Scaffold(local_init_op=tf.group(tf.local_variables_initializer(),
                                                        train_iterator.initializer,
                                                        dev_iterator.initializer,
                                                        test_iterator.initializer))

    return update_op,distill_op, scaffold

class SSTRepDistiller(SSTDistiller):
  def __init__(self, config, student_model, teacher_model):
    super(SSTRepDistiller, self).__init__(config, student_model, teacher_model)

  def build_train_graph(self):
    self.student.build_graph(self.pretrained_word_embeddings)
    self.teacher.build_graph(self.pretrained_word_embeddings)

    train_iterator, dev_iterator, test_iterator = self.get_data_itaratoes()

    train_examples = train_iterator.get_next()
    dev_examples = dev_iterator.get_next()
    test_examples =  test_iterator.get_next()

    student_train_output_dic = self.student.apply(train_examples)
    teacher_train_output_dic = self.teacher.apply(train_examples)

    student_dev_output_dic = self.student.apply(dev_examples)
    teacher_dev_output_dic = self.teacher.apply(dev_examples)

    student_test_output_dic = self.student.apply(test_examples)
    teacher_test_output_dic = self.teacher.apply(test_examples)

    tf.summary.scalar("loss", student_train_output_dic[self.config.loss_type], family="student_train")
    tf.summary.scalar("accuracy", student_train_output_dic["root_accuracy"], family="student_train")

    tf.summary.scalar("loss", student_dev_output_dic[self.config.loss_type], family="student_dev")
    tf.summary.scalar("accuracy", student_dev_output_dic["root_accuracy"], family="student_dev")

    tf.summary.scalar("loss", student_test_output_dic[self.config.loss_type], family="student_test")
    tf.summary.scalar("accuracy", student_test_output_dic["root_accuracy"], family="student_test")

    tf.summary.scalar("loss", teacher_train_output_dic[self.config.loss_type], family="teacher_train")
    tf.summary.scalar("accuracy", teacher_train_output_dic["root_accuracy"], family="teacher_train")

    tf.summary.scalar("loss", teacher_dev_output_dic[self.config.loss_type], family="teacher_dev")
    tf.summary.scalar("accuracy", teacher_dev_output_dic["root_accuracy"], family="teacher_dev")

    tf.summary.scalar("loss", teacher_test_output_dic[self.config.loss_type], family="teacher_test")
    tf.summary.scalar("accuracy", teacher_test_output_dic["root_accuracy"], family="teacher_test")

    distill_rep_loss = get_single_state_rsa_distill_loss(student_train_output_dic['sents_reps'],
                                                     teacher_train_output_dic['sents_reps'],
                                                     mode=self.config.rep_loss_mode)
    distill_logit_loss = get_logit_distill_loss(student_train_output_dic['logits'],
                                                     teacher_train_output_dic['logits'])

    tf.summary.scalar("distill_rep_loss", distill_rep_loss, family="student_train")
    tf.summary.scalar("distill_logit_loss", distill_logit_loss, family="student_train")


    teacher_update_op, teacher_learning_rate = self.get_train_op(teacher_train_output_dic[self.config.loss_type],
                                                         teacher_train_output_dic["trainable_vars"],
                                                         start_learning_rate=0.0005,
                                                         base_learning_rate=0.001, warmup_steps=1000,
                                                         scope="teacher_main")

    distill_rep_op, distill_rep_learning_rate = self.get_train_op(distill_rep_loss, student_train_output_dic["trainable_vars"],
                                                          start_learning_rate=0.0001,
                                                          base_learning_rate=0.001, warmup_steps=10000,
                                                          scope="distill_rep")

    distill_logit_op, distill_logit_learning_rate = self.get_train_op(distill_logit_loss,
                                                                  student_train_output_dic["trainable_vars"],
                                                                  start_learning_rate=0.0001,
                                                                  base_learning_rate=0.001, warmup_steps=10000,
                                                                  scope="distill_logit")

    student_update_op, student_learning_rate = self.get_train_op(student_train_output_dic[self.config.loss_type], student_train_output_dic["trainable_vars"],
                                                          start_learning_rate=0.00005,
                                                          base_learning_rate=0.001, warmup_steps=1000,
                                                          scope="student_main")




    scaffold = tf.train.Scaffold(local_init_op=tf.group(tf.local_variables_initializer(),
                                                        train_iterator.initializer,
                                                        dev_iterator.initializer,
                                                        test_iterator.initializer))

    return teacher_update_op, distill_rep_op, distill_logit_op, student_update_op, scaffold

class Seq2SeqDistiller(Distiller):

  def __init__(self, config, student_model, teacher_model, trainer):
    super(Seq2SeqDistiller, self).__init__(config, student_model, teacher_model)
    self.trainer = trainer

  def apply_model(self, model, train_examples, dev_examples, test_examples, name_tag="", softmax_temperature=1.0):
    train_output_dic = model.apply(train_examples, target_length=self.trainer.task.target_length, is_train=True)
    dev_output_dic = model.apply(dev_examples, target_length=self.trainer.task.target_length, is_train=False)
    #test_output_dic = model.apply(test_examples, target_length=self.trainer.task.target_length, is_train=False)

    train_loss = self.trainer.compute_loss(train_output_dic['logits'],
                                                   train_output_dic['targets'], softmax_temperature=softmax_temperature)
    dev_loss = self.trainer.compute_loss(dev_output_dic['logits'],
                                                 dev_output_dic['targets'], softmax_temperature=softmax_temperature)
    #test_loss = self.trainer.compute_loss(test_output_dic['logits'],
    #                                              test_output_dic['targets'], softmax_temperature=softmax_temperature)

    train_output_dic['loss'] = train_loss
    tf.summary.scalar("loss", train_loss, family=name_tag+"_train")
    tf.summary.scalar("loss", dev_loss, family=name_tag+"_dev")
    #tf.summary.scalar("loss", test_loss, family=name_tag+"_test")

    self.trainer.add_metric_summaries(train_output_dic['logits'],
                                      train_output_dic['targets'], name_tag+"_train")
    self.trainer.add_metric_summaries(dev_output_dic['logits'],
                                      dev_output_dic['targets'], name_tag+"_dev")
    #self.trainer.add_metric_summaries(test_output_dic['logits'],
    #                                  test_output_dic['targets'], name_tag+"_test")

    return train_output_dic, dev_output_dic, None

  def build_train_graph(self):


    train_iterator, dev_iterator, test_iterator = self.get_train_data_itaratoes()
    teacher_train_examples, student_train_examples = train_iterator.get_next()
    teacher_dev_examples, student_dev_examples = dev_iterator.get_next()
    #teacher_test_examples, student_test_examples = test_iterator.get_next()


    self.teacher.create_vars(reuse=False)
    self.student.create_vars(reuse=False)

    teacher_train_output_dic, teacher_dev_output_dic, teacher_test_output_dic = \
    self.apply_model(self.teacher, teacher_train_examples, teacher_dev_examples, None, "teacher", softmax_temperature=self.config.teacher_temp)
    student_train_output_dic, student_dev_output_dic, student_test_output_dic = \
    self.apply_model(self.student, student_train_examples, student_dev_examples, None, "student", softmax_temperature=self.config.student_temp)

    # Compute mean distance between representations
    distill_rep_loss = get_single_state_rsa_distill_loss(student_train_output_dic['outputs'],
                                                     teacher_train_output_dic['outputs'],
                                                     mode=self.config.rep_loss_mode)

    # Compute mean distance between representations
    general_bias_distill_rep_loss = get_biased_single_state_rsa_distill_loss(student_train_output_dic['outputs'],
                                                         teacher_train_output_dic['outputs'],
                                                         mode=self.config.rep_loss_mode, bias="general")
    local_bias_distill_rep_loss = get_biased_single_state_rsa_distill_loss(student_train_output_dic['outputs'],
                                                                             teacher_train_output_dic['outputs'],
                                                                             mode=self.config.rep_loss_mode, bias="local")
                                                     
    # Compute logit distill loss
    distill_logit_loss = get_logit_distill_loss(student_train_output_dic['logits'],
                                                     teacher_train_output_dic['logits'], 
                                                     softmax_temperature=self.config.distill_temp, 
                                                     stop_grad_for_teacher=not self.config.learn_to_teach)
                                                     
    # Compute uniform rep loss
    teacher_uniform_rep_loss = get_single_state_uniform_rsa_loss(teacher_train_output_dic['outputs'],
                                                         mode=self.config.rep_loss_mode)         
    student_uniform_rep_loss = get_single_state_uniform_rsa_loss(student_train_output_dic['outputs'],
                                                         mode=self.config.rep_loss_mode)
    embedding_rep_loss = get_single_state_rsa_distill_loss(self.student.input_embedding_layer.shared_weights,
                                                       self.teacher.input_embedding_layer.shared_weights,
                                                       mode=self.config.rep_loss_mode)

    tf.summary.scalar("embedding_rep_loss", embedding_rep_loss, family="student_train")
    tf.summary.scalar("distill_rep_loss", distill_rep_loss, family="student_train")
    tf.summary.scalar("distill_logit_loss", distill_logit_loss, family="student_train")
    tf.summary.scalar("general_bias_distill_rep_loss", general_bias_distill_rep_loss, family="student_train")
    tf.summary.scalar("local_bias_distill_rep_loss", local_bias_distill_rep_loss, family="student_train")
    tf.summary.scalar("uniform_rep_loss", student_uniform_rep_loss, family="student_train")
    tf.summary.scalar("uniform_rep_loss", teacher_uniform_rep_loss, family="teacher_train")
    
    
    dev_distill_rep_loss = get_single_state_rsa_distill_loss(student_dev_output_dic['outputs'],
                                                     teacher_dev_output_dic['outputs'],
                                                     mode=self.config.rep_loss_mode)


    dev_distill_logit_loss = get_logit_distill_loss(student_dev_output_dic['logits'],
                                                     teacher_dev_output_dic['logits'], softmax_temperature=self.config.distill_temp)
    dev_uniform_rep_loss = get_single_state_uniform_rsa_loss(student_dev_output_dic['outputs'],
                                                         mode=self.config.rep_loss_mode) 
    teacher_dev_uniform_rep_loss = get_single_state_uniform_rsa_loss(teacher_dev_output_dic['outputs'],
                                                         mode=self.config.rep_loss_mode)                                                      
                                                     
    tf.summary.scalar("distill_rep_loss", dev_distill_rep_loss, family="student_dev")
    tf.summary.scalar("distill_logit_loss", dev_distill_logit_loss, family="student_dev")
    tf.summary.scalar("uniform_rep_loss", dev_uniform_rep_loss, family="student_dev")
    tf.summary.scalar("uniform_rep_loss", teacher_dev_uniform_rep_loss, family="teacher_dev")


    distill_params = student_train_output_dic["trainable_vars"]
    if self.config.learn_to_teach:
        distill_params + teacher_train_output_dic["trainable_vars"]
        
    teacher_update_op, teacher_learning_rate = self.trainer.get_train_op(teacher_train_output_dic['loss'],
                                                         teacher_train_output_dic["trainable_vars"],
                                                         start_learning_rate=0.000,
                                                         base_learning_rate=self.teacher.hparams.learning_rate, warmup_steps=1000,
                                                         l2_rate=self.trainer.config.l2_rate,
                                                         scope="teacher")

#    distill_rep_op, distill_rep_learning_rate = self.trainer.get_train_op(distill_rep_loss, 
#                                                          distill_params,
#                                                          start_learning_rate=0.0,
#                                                          base_learning_rate=self.config.distill_learning_rate, warmup_steps=10000,
#                                                          l2_rate=0.0,
#                                                          scope="distill_rep")

#    distill_logit_op, distill_logit_learning_rate = self.trainer.get_train_op(distill_logit_loss,
#                                                                  distill_params,
#                                                                  start_learning_rate=0.00,
#                                                                  base_learning_rate=self.config.distill_learning_rate, warmup_steps=10000,
#                                                                  l2_rate=0.0,
#                                                                  scope="distill_logit")

    student_loss = self.config.data_weight * student_train_output_dic['loss'] + self.config.distill_logits_weight * distill_logit_loss
    student_update_op, student_learning_rate = self.trainer.get_train_op(student_loss, student_train_output_dic["trainable_vars"],
                                                          start_learning_rate=0.000,
                                                          base_learning_rate=self.student.hparams.learning_rate, warmup_steps=10000,
                                                          l2_rate=self.trainer.config.l2_rate,
                                                          scope="student")


    tf.logging.info("student variables")
    tf.logging.info(student_train_output_dic["trainable_vars"])
    
    tf.logging.info("teacher variables")
    tf.logging.info(teacher_train_output_dic["trainable_vars"])
    
    tf.summary.scalar("learning_rate", teacher_learning_rate, family="teacher_train")
    #tf.summary.scalar("distill_logit_learning_rate", distill_logit_learning_rate, family="student_train")

    scaffold = tf.train.Scaffold(local_init_op=tf.group(tf.local_variables_initializer(),
                                                        train_iterator.initializer,
                                                        dev_iterator.initializer,
                                                        test_iterator.initializer))

    return teacher_update_op, None, None, student_update_op, scaffold

  def get_train_data_itaratoes(self):
    dataset = tf.data.TFRecordDataset(self.trainer.task.get_tfrecord_path(mode="train"))
    dataset = dataset.map(self.trainer.task.parse_examples)
    dataset = dataset.padded_batch(self.trainer.config.batch_size, padded_shapes=self.trainer.task.get_padded_shapes())
    dataset = dataset.map((lambda x1,x2,x3,x4: ((x1,x2,x3,x4),(x1,x2,x3,x4))))
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.repeat()
    train_iterator = dataset.make_initializable_iterator()

    dataset = tf.data.TFRecordDataset(self.trainer.task.get_tfrecord_path(mode="dev"))
    dataset = dataset.map(self.trainer.task.parse_examples)
    dataset = dataset.padded_batch(self.trainer.config.batch_size, padded_shapes=self.trainer.task.get_padded_shapes())
    dataset = dataset.map((lambda x1,x2,x3,x4: ((x1,x2,x3,x4),(x1,x2,x3,x4))))
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.repeat()
    dev_iterator = dataset.make_initializable_iterator()

    dataset = tf.data.TFRecordDataset(self.trainer.task.get_tfrecord_path(mode="test"))
    dataset = dataset.map(self.trainer.task.parse_examples)
    dataset = dataset.padded_batch(self.trainer.config.batch_size, padded_shapes=self.trainer.task.get_padded_shapes())
    dataset = dataset.map((lambda x1,x2,x3,x4: ((x1,x2,x3,x4),(x1,x2,x3,x4))))
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.repeat()
    test_iterator = dataset.make_initializable_iterator()

    return train_iterator, dev_iterator, test_iterator


class Seq2SeqParallel(Seq2SeqDistiller):

  def __init__(self, config, student_model, teacher_model, trainer):
    super(Seq2SeqParallel, self).__init__(config, student_model, teacher_model, trainer)
    self.trainer = trainer

  def apply_model(self, model, train_examples, dev_examples, test_examples, name_tag="", softmax_temperature=1.0):
    train_output_dic = model.apply(train_examples, target_length=self.trainer.task.target_length, is_train=True)
    dev_output_dic = model.apply(dev_examples, target_length=self.trainer.task.target_length, is_train=False)
    # test_output_dic = model.apply(test_examples, target_length=self.trainer.task.target_length, is_train=False)

    train_loss = self.trainer.compute_loss(train_output_dic['logits'],
                                           train_output_dic['targets'], softmax_temperature=softmax_temperature)
    dev_loss = self.trainer.compute_loss(dev_output_dic['logits'],
                                         dev_output_dic['targets'], softmax_temperature=softmax_temperature)
    # test_loss = self.trainer.compute_loss(test_output_dic['logits'],
    #                                              test_output_dic['targets'], softmax_temperature=softmax_temperature)

    train_output_dic['loss'] = train_loss
    tf.summary.scalar("loss", train_loss, family=name_tag + "_train")
    tf.summary.scalar("loss", dev_loss, family=name_tag + "_dev")
    # tf.summary.scalar("loss", test_loss, family=name_tag+"_test")

    self.trainer.add_metric_summaries(train_output_dic['logits'],
                                      train_output_dic['targets'], name_tag + "_train")
    self.trainer.add_metric_summaries(dev_output_dic['logits'],
                                      dev_output_dic['targets'], name_tag + "_dev")
    # self.trainer.add_metric_summaries(test_output_dic['logits'],
    #                                  test_output_dic['targets'], name_tag+"_test")

    return train_output_dic, dev_output_dic, None

  def build_train_graph(self):
    train_iterator, dev_iterator, test_iterator = self.get_train_data_itaratoes()
    teacher_train_examples, student_train_examples = train_iterator.get_next()
    teacher_dev_examples, student_dev_examples = dev_iterator.get_next()

    self.teacher.create_vars(reuse=False)
    self.student.create_vars(reuse=False)

    teacher_train_output_dic, teacher_dev_output_dic, teacher_test_output_dic = \
      self.apply_model(self.teacher, teacher_train_examples, teacher_dev_examples, None, "teacher",
                       softmax_temperature=self.config.teacher_temp)
    student_train_output_dic, student_dev_output_dic, student_test_output_dic = \
      self.apply_model(self.student, student_train_examples, student_dev_examples, None, "student",
                       softmax_temperature=self.config.student_temp)

    # Compute mean distance between representations
    distill_rep_loss = get_single_state_rsa_distill_loss(student_train_output_dic['outputs'],
                                                         teacher_train_output_dic['outputs'],
                                                         mode=self.config.rep_loss_mode)

    # Compute mean distance between representations
    # general_bias_distill_rep_loss = get_biased_single_state_rsa_distill_loss(student_train_output_dic['outputs'],
    #                                                                          teacher_train_output_dic['outputs'],
    #                                                                          mode=self.config.rep_loss_mode,
    #                                                                          bias="general")
    local_bias_distill_rep_loss = get_biased_single_state_rsa_distill_loss(student_train_output_dic['outputs'],
                                                                           teacher_train_output_dic['outputs'],
                                                                           mode=self.config.rep_loss_mode, bias="local")

    # Compute logit distill loss
    distill_logit_loss = get_logit_distill_loss(student_train_output_dic['logits'],
                                                teacher_train_output_dic['logits'],
                                                softmax_temperature=self.config.distill_temp,
                                                stop_grad_for_teacher=not self.config.learn_to_teach)

    # Compute uniform rep loss
    # teacher_uniform_rep_loss = get_single_state_uniform_rsa_loss(teacher_train_output_dic['outputs'],
    #                                                              mode=self.config.rep_loss_mode)
    # student_uniform_rep_loss = get_single_state_uniform_rsa_loss(student_train_output_dic['outputs'],
    #                                                              mode=self.config.rep_loss_mode)
    embedding_rep_loss = get_single_state_rsa_distill_loss(self.student.input_embedding_layer.shared_weights,
                                                            self.teacher.input_embedding_layer.shared_weights,
                                                            mode=self.config.rep_loss_mode)

    rep_degree_sim =  get_rep_sim(student_train_output_dic['outputs'],
                                  teacher_train_output_dic['outputs'],mode="degree",topk=None)
    rep_rank_sim = get_rep_sim(student_train_output_dic['outputs'],
                                  teacher_train_output_dic['outputs'],mode="rank",topk=None)
    rep_std_sim = get_rep_sim(student_train_output_dic['outputs'],
                                  teacher_train_output_dic['outputs'],mode="std",topk=None)
    rep_std_sim_top10 = get_rep_sim(student_train_output_dic['outputs'],
                                  teacher_train_output_dic['outputs'],mode="std",topk=20)

    embedding_degree_sim = get_rep_sim(self.student.input_embedding_layer.shared_weights,
                                  self.teacher.input_embedding_layer.shared_weights,mode="degree",topk=None)
    embedding_rank_sim = get_rep_sim(self.student.input_embedding_layer.shared_weights,
                                  self.teacher.input_embedding_layer.shared_weights,mode="rank",topk=None)
    embedding_std_sim = get_rep_sim(self.student.input_embedding_layer.shared_weights,
                                  self.teacher.input_embedding_layer.shared_weights,mode="std",topk=None)

    tf.summary.scalar("rep_degree_sim", rep_degree_sim, family="train_reps")
    tf.summary.scalar("rep_rank_sim", rep_rank_sim, family="train_reps")
    tf.summary.scalar("rep_std_sim", rep_std_sim, family="train_reps")
    tf.summary.scalar("rep_std_sim_top10", rep_std_sim_top10, family="train_reps")
    tf.summary.scalar("embedding_degree_sim", embedding_degree_sim, family="train_reps")
    tf.summary.scalar("embedding_rank_sim", embedding_rank_sim, family="train_reps")
    tf.summary.scalar("embedding_std_sim", embedding_std_sim, family="train_reps")


    tf.summary.scalar("embedding_rep_loss", embedding_rep_loss, family="student_train")
    tf.summary.scalar("distill_rep_loss", distill_rep_loss, family="student_train")
    tf.summary.scalar("distill_logit_loss", distill_logit_loss, family="student_train")
    #tf.summary.scalar("general_bias_distill_rep_loss", general_bias_distill_rep_loss, family="student_train")
    tf.summary.scalar("local_bias_distill_rep_loss", local_bias_distill_rep_loss, family="student_train")
    #tf.summary.scalar("uniform_rep_loss", student_uniform_rep_loss, family="student_train")
    #tf.summary.scalar("uniform_rep_loss", teacher_uniform_rep_loss, family="teacher_train")

    dev_distill_rep_loss = get_single_state_rsa_distill_loss(student_dev_output_dic['outputs'],
                                                             teacher_dev_output_dic['outputs'],
                                                             mode=self.config.rep_loss_mode)

    dev_distill_logit_loss = get_logit_distill_loss(student_dev_output_dic['logits'],
                                                    teacher_dev_output_dic['logits'],
                                                    softmax_temperature=self.config.distill_temp)
    dev_uniform_rep_loss = get_single_state_uniform_rsa_loss(student_dev_output_dic['outputs'],
                                                             mode=self.config.rep_loss_mode)
    teacher_dev_uniform_rep_loss = get_single_state_uniform_rsa_loss(teacher_dev_output_dic['outputs'],
                                                                     mode=self.config.rep_loss_mode)

    tf.summary.scalar("distill_rep_loss", dev_distill_rep_loss, family="student_dev")
    tf.summary.scalar("distill_logit_loss", dev_distill_logit_loss, family="student_dev")
    tf.summary.scalar("uniform_rep_loss", dev_uniform_rep_loss, family="student_dev")
    tf.summary.scalar("uniform_rep_loss", teacher_dev_uniform_rep_loss, family="teacher_dev")

    distill_params = student_train_output_dic["trainable_vars"]
    if self.config.learn_to_teach:
      distill_params + teacher_train_output_dic["trainable_vars"]

    teacher_update_op, teacher_learning_rate = self.trainer.get_train_op(teacher_train_output_dic['loss'],
                                                                         teacher_train_output_dic["trainable_vars"],
                                                                         start_learning_rate=0.000,
                                                                         base_learning_rate=self.teacher.hparams.learning_rate,
                                                                         warmup_steps=1000,
                                                                         l2_rate=self.trainer.config.l2_rate,
                                                                         scope="teacher")

    student_loss = self.config.data_weight * student_train_output_dic[
      'loss'] + self.config.distill_logits_weight * distill_logit_loss
    student_update_op, student_learning_rate = self.trainer.get_train_op(student_loss,
                                                                         student_train_output_dic["trainable_vars"],
                                                                         start_learning_rate=0.000,
                                                                         base_learning_rate=self.student.hparams.learning_rate,
                                                                         warmup_steps=1000,
                                                                         l2_rate=self.trainer.config.l2_rate,
                                                                         scope="student")

    tf.summary.scalar("learning_rate", teacher_learning_rate, family="teacher_train")

    scaffold = tf.train.Scaffold(local_init_op=tf.group(tf.local_variables_initializer(),
                                                        train_iterator.initializer,
                                                        dev_iterator.initializer,
                                                        test_iterator.initializer))

    return teacher_update_op, None, None, student_update_op, scaffold

