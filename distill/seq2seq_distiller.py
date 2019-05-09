import tensorflow as tf

from distill.common.hparams import TransformerHparam, LSTMHparam
import os

from distill.data_util.prep_algorithmic import AlgorithmicIdentityDecimal40, AlgorithmicIdentityBinary40, \
  AlgorithmicAdditionDecimal40, AlgorithmicMultiplicationDecimal40, AlgorithmicSortProblem, AlgorithmicReverseProblem
from distill.data_util.prep_arithmatic import Arithmatic
from distill.data_util.prep_ptb import PTB
from distill.data_util.prep_sst import SST
from distill.data_util.prep_wsj_parsing import ParseWSJ
from distill.models.lstm_seq2seq import LSTMSeq2Seq, BidiLSTMSeq2Seq
from distill.models.transformer import Transformer, UniversalTransformer, EncodingTransformer, \
  EncodingUniversalTransformer
from distill.pipelines.distill_pipelines import Seq2SeqDistiller
from distill.pipelines.seq2seq import Seq2SeqTrainer

tf.logging.set_verbosity(tf.logging.INFO)

tf.app.flags.DEFINE_string("exp_name", "distill", "")
tf.app.flags.DEFINE_string("task_name", "identity_binary", "")
tf.app.flags.DEFINE_string("log_dir", "logs", "")
tf.app.flags.DEFINE_string("save_dir", None, "")

tf.app.flags.DEFINE_string("model_type", "transformer2transformer", "")
tf.app.flags.DEFINE_string("teacher_model", "transformer", "")
tf.app.flags.DEFINE_string("student_model", "transformer", "")

tf.app.flags.DEFINE_boolean("train_teacher", True, "")
tf.app.flags.DEFINE_boolean("train_student", False, "")
tf.app.flags.DEFINE_boolean("distill_rep", False, "")
tf.app.flags.DEFINE_boolean("distill_logit", True, "")

tf.app.flags.DEFINE_boolean("pretrain_teacher", True, "")
tf.app.flags.DEFINE_integer("teacher_pretraining_iters", 100, "")
tf.app.flags.DEFINE_string("rep_loss_mode", 'dot_product', "representation loss type (squared,softmax_cross_ent,sigmoid_cross_ent")

tf.app.flags.DEFINE_string("model", "transformer", "transformer | utransformer | lstm | bilstm")
tf.app.flags.DEFINE_string("teacher_encoder_attention_dir", "top_down", "top_down | bottom_up")
tf.app.flags.DEFINE_string("student_encoder_attention_dir", "top_down", "top_down | bottom_up")


tf.app.flags.DEFINE_integer("teacher_hidden_dim", 300, "")
tf.app.flags.DEFINE_integer("teacher_encoder_depth", 2, "")
tf.app.flags.DEFINE_integer("teacher_decoder_depth", 1, "")
tf.app.flags.DEFINE_integer("student_hidden_dim", 300, "")
tf.app.flags.DEFINE_integer("student_encoder_depth", 2, "")
tf.app.flags.DEFINE_integer("student_decoder_depth", 1, "")

tf.app.flags.DEFINE_integer("input_dim", None, "")
tf.app.flags.DEFINE_integer("output_dim", 1, "")
tf.app.flags.DEFINE_integer("number_of_heads", 4, "")
tf.app.flags.DEFINE_integer("ff_filter_size", 512, "")
tf.app.flags.DEFINE_float("initializer_gain", 1.0, "")
tf.app.flags.DEFINE_float("teacher_label_smoothing", 0.1, "")
tf.app.flags.DEFINE_float("student_label_smoothing", 0.1, "")

tf.app.flags.DEFINE_boolean('teacher_train_embeddings', True, " False | True")
tf.app.flags.DEFINE_boolean('student_train_embeddings', True, " False | True")

tf.app.flags.DEFINE_string('teacher_sent_rep_mode', "final", "none | final | all")
tf.app.flags.DEFINE_string('student_sent_rep_mode', "final", "none | final | all")

tf.app.flags.DEFINE_string('teacher_attention_mechanism', None, 'attention_mechanism')
tf.app.flags.DEFINE_string('student_attention_mechanism', None, 'attention_mechanism')


tf.app.flags.DEFINE_float("teacher_input_dropout_keep_prob", 0.75, "")
tf.app.flags.DEFINE_float("teacher_hidden_dropout_keep_prob", 0.5, "")
tf.app.flags.DEFINE_float("student_input_dropout_keep_prob", 0.75, "")
tf.app.flags.DEFINE_float("student_hidden_dropout_keep_prob", 0.5, "")


tf.app.flags.DEFINE_float("teacher_learning_rate", 0.001, "")
tf.app.flags.DEFINE_float("student_learning_rate", 0.001, "")

tf.app.flags.DEFINE_boolean("decay_learning_rate", True, "True | False")
tf.app.flags.DEFINE_float("l2_rate", 0.001, "")


tf.app.flags.DEFINE_integer("batch_size", 32, "")
tf.app.flags.DEFINE_integer("training_iterations", 60000, "")

tf.app.flags.DEFINE_integer("vocab_size", 3, "")
tf.app.flags.DEFINE_integer("embedding_dim", 300, "embeddings dim")

tf.app.flags.DEFINE_string("data_path", "./data", "data path")

hparams = tf.app.flags.FLAGS


if __name__ == '__main__':

  Models = {"lstm": LSTMSeq2Seq,
            "bilstm": BidiLSTMSeq2Seq,
            "transformer": Transformer,
            "utransformer": UniversalTransformer,
            "enc_transformer": EncodingTransformer,
            "enc_utransformer": EncodingUniversalTransformer}


  tasks = {'identity': AlgorithmicIdentityDecimal40('data/alg'),
           'identity_binary': AlgorithmicIdentityBinary40('data/alg'),
           'addition': AlgorithmicAdditionDecimal40('data/alg'),
           'multiplication': AlgorithmicMultiplicationDecimal40('data/alg'),
           'sort': AlgorithmicSortProblem('data/alg'),
           'reverse': AlgorithmicReverseProblem('data/alg'),
           'arithmatic': Arithmatic('data/arithmatic'),
           'sst': SST(data_path="data/sst/",
                 add_subtrees=False,
                 pretrained=True),
           'ptb_lm': PTB('data/ptb'),
           'wsj_parse': ParseWSJ('data/wsj')}

  hparams.vocab_size = tasks[hparams.task_name].vocab_length
  hparams.output_dim = len(tasks[hparams.task_name].target_vocab)

  PARAM_TYPES = {"lstm": LSTMHparam,
            "bilstm": LSTMHparam,
            "transformer": TransformerHparam,
            "utransformer": TransformerHparam,
            "enc_transformer": TransformerHparam,
            "enc_utransformer": TransformerHparam}

  teacher_params = PARAM_TYPES[hparams.teacher_model](input_dim=hparams.input_dim,
                                                      output_dim=hparams.output_dim,
                                                      hidden_dim=hparams.teacher_hidden_dim,
                                                      encoder_depth=hparams.teacher_encoder_depth,
                                                      decoder_depth=hparams.teacher_decoder_depth,
                                                      number_of_heads=2,
                                                      ff_filter_size=512,
                                                      initializer_gain=hparams.initializer_gain,
                                                      batch_size=hparams.batch_size,
                                                      input_dropout_keep_prob=hparams.teacher_input_dropout_keep_prob,
                                                      hidden_dropout_keep_prob=hparams.teacher_hidden_dropout_keep_prob,
                                                      vocab_size=hparams.vocab_size,
                                                      label_smoothing=hparams.teacher_label_smoothing,
                                                      encoder_self_attention_dir=hparams.teacher_encoder_attention_dir,
                                                      decoder_self_attention_dir="top_down",
                                                      decoder_cross_attention_dir="top_down",
                                                      train_embeddings=hparams.teacher_train_embeddings,
                                                      attention_mechanism=None,
                                                      sent_rep_mode=hparams.teacher_sent_rep_mode,
                                                      embedding_dim=300,
                                                      learning_rate=hparams.teacher_learning_rate
                                                      )

  student_params = PARAM_TYPES[hparams.teacher_model](input_dim=hparams.input_dim,
                                                      output_dim=hparams.output_dim,
                                                      hidden_dim=hparams.teacher_hidden_dim,
                                                      encoder_depth=hparams.teacher_encoder_depth,
                                                      decoder_depth=hparams.teacher_decoder_depth,
                                                      number_of_heads=2,
                                                      ff_filter_size=512,
                                                      initializer_gain=hparams.initializer_gain,
                                                      batch_size=hparams.batch_size,
                                                      input_dropout_keep_prob=hparams.teacher_input_dropout_keep_prob,
                                                      hidden_dropout_keep_prob=hparams.teacher_hidden_dropout_keep_prob,
                                                      vocab_size=hparams.vocab_size,
                                                      label_smoothing=hparams.teacher_label_smoothing,
                                                      encoder_self_attention_dir=hparams.teacher_encoder_attention_dir,
                                                      decoder_self_attention_dir="top_down",
                                                      decoder_cross_attention_dir="top_down",
                                                      train_embeddings=hparams.teacher_train_embeddings,
                                                      attention_mechanism=None,
                                                      sent_rep_mode=hparams.teacher_sent_rep_mode,
                                                      embedding_dim=300,
                                                      learning_rate=hparams.teacher_learning_rate
                                                      )


  hparams.model_type = '_'.join([hparams.teacher_model,'to',hparams.student_model])

  if hparams.save_dir is None:
    hparams.save_dir = os.path.join(hparams.log_dir, hparams.task_name, '_'.join(
      [hparams.rep_loss_mode, hparams.model_type, 'std_depth' + str(hparams.student_encoder_depth),
       'teacher_depth' + str(hparams.teacher_encoder_depth), 'std_hidden_dim' + str(hparams.student_hidden_dim),
       'teacher_hidden_dim' + str(hparams.teacher_hidden_dim), hparams.exp_name]))

  student_model = Models[hparams.student_model](student_params,
                                task=tasks[hparams.task_name],
                                scope=hparams.student_model+"_student")
  teacher_model = Models[hparams.teacher_model](teacher_params,
                                        task=tasks[hparams.task_name],
                                        scope=hparams.teacher_model+"_teacher")

  trainer = Seq2SeqTrainer(hparams, teacher_model, tasks[hparams.task_name])
  distiller = Seq2SeqDistiller(hparams, student_model, teacher_model, trainer)
  distiller.train()
