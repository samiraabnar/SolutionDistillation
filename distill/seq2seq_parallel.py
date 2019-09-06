import tensorflow as tf

from distill.common.hparams import TransformerHparam, LSTMHparam
import os

from distill.data_util.prep_algorithmic import AlgorithmicIdentityDecimal40, AlgorithmicIdentityBinary40, \
  AlgorithmicAdditionDecimal40, AlgorithmicMultiplicationDecimal40, AlgorithmicSortProblem, AlgorithmicReverseProblem
from distill.data_util.prep_arithmatic import Arithmatic, ArithmaticSameLength, ArithmaticSimple, \
  ArithmaticSimpleCurriculumLength, \
  ArithmaticSimpleSameLength10, ArithmaticSimpleSameLength10Depth6, ArithmaticSimpleSameLength10Depth2, \
  ArithmaticSimpleSameLength10Depth4, \
  ArithmaticSimpleSameLength21Depth2Normal, ArithmaticSimpleSameLength201Depth2Normal, \
  ArithmaticSimpleSameLength21Depth2NormalBiLing, ArithmaticSimpleMissingLength21Depth2NormalBiLing
from distill.data_util.prep_imdb import IMDB
from distill.data_util.prep_ptb import PTB
from distill.data_util.prep_sst import SST
from distill.data_util.prep_trec6 import CharTrec6, Trec6
from distill.data_util.prep_wsj_parsing import ParseWSJ
from distill.models.lm_lstm import LmLSTM
from distill.models.lstm_seq2seq import LSTMSeq2Seq, BidiLSTMSeq2Seq
from distill.models.transformer import Transformer, UniversalTransformer, EncodingTransformer, \
  EncodingUniversalTransformer, DecodingUniversalTransformer, DecodingTransformer
from distill.pipelines.distill_pipelines import Seq2SeqParallel
from distill.pipelines.seq2seq import Seq2SeqTrainer

tf.logging.set_verbosity(tf.logging.INFO)

tf.app.flags.DEFINE_string("exp_name", "distill", "")
tf.app.flags.DEFINE_string("task_name", "identity_binary", "")
tf.app.flags.DEFINE_string("log_dir", "logs", "")
tf.app.flags.DEFINE_string("data_dir", "data", "")

tf.app.flags.DEFINE_string("save_dir", None, "")

tf.app.flags.DEFINE_string("model_type", "lstm2transformer", "")
tf.app.flags.DEFINE_string("teacher_model", "lstm", "")
tf.app.flags.DEFINE_string("student_model", "enc_utransformer", "")

tf.app.flags.DEFINE_boolean("train_teacher", True, "")
tf.app.flags.DEFINE_boolean("train_student", True, "")
tf.app.flags.DEFINE_boolean("distill_rep", False, "")
tf.app.flags.DEFINE_boolean("distill_logit", True, "")

tf.app.flags.DEFINE_boolean("pretrain_teacher", True, "")
tf.app.flags.DEFINE_integer("teacher_pretraining_iters", 100, "")
tf.app.flags.DEFINE_string("rep_loss_mode", 'dot_product', "representation loss type (squared,softmax_cross_ent,sigmoid_cross_ent")

tf.app.flags.DEFINE_string("model", "enc_utransformer", "transformer | utransformer | lstm | bilstm")
tf.app.flags.DEFINE_string("teacher_encoder_attention_dir", "top_down", "top_down | bottom_up")
tf.app.flags.DEFINE_string("student_encoder_attention_dir", "top_down", "top_down | bottom_up")


tf.app.flags.DEFINE_integer("teacher_hidden_dim", 256, "")
tf.app.flags.DEFINE_integer("teacher_encoder_depth", 1, "")
tf.app.flags.DEFINE_integer("teacher_decoder_depth", 1, "")
tf.app.flags.DEFINE_integer("student_hidden_dim", 128, "")
tf.app.flags.DEFINE_integer("student_encoder_depth", 4, "")
tf.app.flags.DEFINE_integer("student_decoder_depth", 1, "")

tf.app.flags.DEFINE_integer("input_dim", None, "")
tf.app.flags.DEFINE_integer("output_dim", 1, "")
tf.app.flags.DEFINE_integer("number_of_heads", 4, "")
tf.app.flags.DEFINE_integer("ff_filter_size", 512, "")
tf.app.flags.DEFINE_float("initializer_gain", 1.0, "")
tf.app.flags.DEFINE_float("teacher_label_smoothing", 0.0001, "")
tf.app.flags.DEFINE_float("student_label_smoothing", 0.0001, "")

tf.app.flags.DEFINE_boolean('teacher_train_embeddings', True, " False | True")
tf.app.flags.DEFINE_boolean('student_train_embeddings', True, " False | True")

tf.app.flags.DEFINE_string('teacher_sent_rep_mode', "final", "none | final | all")
tf.app.flags.DEFINE_string('student_sent_rep_mode', "final", "none | final | all")

tf.app.flags.DEFINE_string('teacher_attention_mechanism', None, 'attention_mechanism')
tf.app.flags.DEFINE_string('student_attention_mechanism', None, 'attention_mechanism')


tf.app.flags.DEFINE_float("teacher_input_dropout_keep_prob", 0.8, "")
tf.app.flags.DEFINE_float("teacher_hidden_dropout_keep_prob", 0.9, "")
tf.app.flags.DEFINE_float("student_input_dropout_keep_prob", 0.75, "")
tf.app.flags.DEFINE_float("student_hidden_dropout_keep_prob", 0.5, "")
tf.app.flags.DEFINE_float("teacher_attention_dropout_keepprob", 1.0, "")
tf.app.flags.DEFINE_float("teacher_relu_dropout_keepprob", 1.0, "")
tf.app.flags.DEFINE_float("teacher_postprocess_dropout_keepprob", 1.0, "")
tf.app.flags.DEFINE_float("student_attention_dropout_keepprob", 1.0, "")
tf.app.flags.DEFINE_float("student_relu_dropout_keepprob", 1.0, "")
tf.app.flags.DEFINE_float("student_postprocess_dropout_keepprob", 1.0, "")

tf.app.flags.DEFINE_float("teacher_learning_rate", 0.001, "")
tf.app.flags.DEFINE_float("student_learning_rate", 0.001, "")
tf.app.flags.DEFINE_float("distill_learning_rate", 0.001, "")
tf.app.flags.DEFINE_float("data_weight", 0.00, "")
tf.app.flags.DEFINE_float("distill_logits_weight", 1.00, "")

tf.app.flags.DEFINE_float("distill_temp", 1, "")
tf.app.flags.DEFINE_float("teacher_temp", 1, "")
tf.app.flags.DEFINE_float("student_temp", 1, "")
tf.app.flags.DEFINE_boolean("learn_to_teach", False, "")



tf.app.flags.DEFINE_boolean("decay_learning_rate", True, "True | False")
tf.app.flags.DEFINE_float("l2_rate", 0.0001, "")


tf.app.flags.DEFINE_integer("batch_size", 32, "")
tf.app.flags.DEFINE_integer("training_iterations", 300000, "")

tf.app.flags.DEFINE_integer("vocab_size", 3, "")
tf.app.flags.DEFINE_integer("embedding_dim", 128, "embeddings dim")

tf.app.flags.DEFINE_string("data_path", "../data", "data path")

hparams = tf.app.flags.FLAGS


if __name__ == '__main__':

  Models = {
            "lm_lstm": LmLSTM,
            "lstm": LSTMSeq2Seq,
            "bilstm": BidiLSTMSeq2Seq,
            "transformer": Transformer,
            "utransformer": UniversalTransformer,
            "enc_transformer": EncodingTransformer,
            "enc_utransformer": EncodingUniversalTransformer,
            "dec_utransformer": DecodingUniversalTransformer,
            "dec_transformer": DecodingTransformer}


  tasks = {'identity': AlgorithmicIdentityDecimal40(os.path.join(hparams.data_dir,'alg')),
           'identity_binary': AlgorithmicIdentityBinary40(os.path.join(hparams.data_dir,'alg')),
           'addition': AlgorithmicAdditionDecimal40(os.path.join(hparams.data_dir,'alg')),
           'multiplication': AlgorithmicMultiplicationDecimal40(os.path.join(hparams.data_dir,'alg')),
           'sort': AlgorithmicSortProblem(os.path.join(hparams.data_dir,'alg')),
           'reverse': AlgorithmicReverseProblem(os.path.join(hparams.data_dir,'alg')),
           'arithmatic': Arithmatic(os.path.join(hparams.data_dir,'arithmatic')),
           'arithmatic_simple': ArithmaticSimple(os.path.join(hparams.data_dir,'arithmatic_simple')),
           'arithmatic_samelength': ArithmaticSameLength(os.path.join(hparams.data_dir,'arithmatic_samelength')),
           'arithmatic_simple_curriculum_length': ArithmaticSimpleCurriculumLength(os.path.join(hparams.data_dir,'arithmatic_simple_curriculum_length')),
           'arithmatic_simple_samelength': ArithmaticSameLength(os.path.join(hparams.data_dir,'arithmatic_samelength')),
           'arithmatic_simple_samelength10': ArithmaticSimpleSameLength10(os.path.join(hparams.data_dir,'arithmatic_simple_samelength10')),
           'arithmatic_simple_samelength10_depth6': ArithmaticSimpleSameLength10Depth6(os.path.join(hparams.data_dir,'arithmatic_simple_samelength10_depth6')),
           'arithmatic_simple_samelength10_depth4': ArithmaticSimpleSameLength10Depth4(os.path.join(hparams.data_dir,'arithmatic_simple_samelength10_depth4')),
           'arithmatic_simple_samelength10_depth2': ArithmaticSimpleSameLength10Depth2(os.path.join(hparams.data_dir,'arithmatic_simple_samelength10_depth2')),
           'arithmatic_simple_samelength21_depth2_normal': ArithmaticSimpleSameLength21Depth2Normal(os.path.join(hparams.data_dir,'arithmatic_simple_samelength21_depth2_normal')),
           'arithmatic_simple_samelength21_depth2_normal_biling': ArithmaticSimpleSameLength21Depth2NormalBiLing(
             os.path.join(hparams.data_dir, 'arithmatic_simple_samelength21_depth2_normal_biling')),
           'arithmatic_simple_samelength201_depth2_normal': ArithmaticSimpleSameLength201Depth2Normal(
             os.path.join(hparams.data_dir, 'arithmatic_simple_samelength201_depth2_normal')),
           'arithmatic_simple_missinglength21_depth2_normal_biling': ArithmaticSimpleMissingLength21Depth2NormalBiLing(
             os.path.join(hparams.data_dir, 'arithmatic_simple_missinglength21_depth2_normal_biling')),
           'sst': SST(data_path=os.path.join(hparams.data_dir,"sst/"),
                 add_subtrees=False,
                 pretrained=True),
           'ptb_lm': PTB(os.path.join(hparams.data_dir,'ptb')),
           'wsj_parse': ParseWSJ(os.path.join(hparams.data_dir,'wsj')),
           'imdb': IMDB(data_path=os.path.join(hparams.data_dir,"imdb"),
                        pretrained=True),
            'char_trec': CharTrec6(os.path.join(hparams.data_dir,"char_trec6"), build_vocab=False)
           }

  hparams.vocab_size = tasks[hparams.task_name].vocab_length
  hparams.output_dim = len(tasks[hparams.task_name].target_vocab)

  PARAM_TYPES = {
            "lm_lstm": LSTMHparam,
            "lstm": LSTMHparam,
            "bilstm": LSTMHparam,
            "transformer": TransformerHparam,
            "utransformer": TransformerHparam,
            "enc_transformer": TransformerHparam,
            "enc_utransformer": TransformerHparam}

  CLS_TOKEN = {
              "lm_lstm": False,
              "lstm": False,
              "bilstm": False,
              "transformer": False,
              "utransformer": False,
              "enc_transformer": True,
              "enc_utransformer": True}

  teacher_params = PARAM_TYPES[hparams.teacher_model](input_dim=hparams.input_dim,
                                                      output_dim=hparams.output_dim,
                                                      hidden_dim=hparams.teacher_hidden_dim,
                                                      encoder_depth=hparams.teacher_encoder_depth,
                                                      decoder_depth=hparams.teacher_decoder_depth,
                                                      number_of_heads=4,
                                                      ff_filter_size=hparams.teacher_hidden_dim*4,
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
                                                      embedding_dim=hparams.teacher_hidden_dim,
                                                      learning_rate=hparams.teacher_learning_rate,
                                                      cls_token=CLS_TOKEN[hparams.teacher_model],
                                                      attention_dropout_keepprob=hparams.teacher_hidden_dropout_keep_prob,
                                                      relu_dropout_keepprob=hparams.teacher_relu_dropout_keepprob,
                                                      postprocess_dropout_keepprob=hparams.teacher_postprocess_dropout_keepprob,
                                                      )

  student_params = PARAM_TYPES[hparams.student_model](input_dim=hparams.input_dim,
                                                      output_dim=hparams.output_dim,
                                                      hidden_dim=hparams.student_hidden_dim,
                                                      encoder_depth=hparams.student_encoder_depth,
                                                      decoder_depth=hparams.student_decoder_depth,
                                                      number_of_heads=4,
                                                      ff_filter_size=hparams.student_hidden_dim*4,
                                                      initializer_gain=hparams.initializer_gain,
                                                      batch_size=hparams.batch_size,
                                                      input_dropout_keep_prob=hparams.student_input_dropout_keep_prob,
                                                      hidden_dropout_keep_prob=hparams.student_hidden_dropout_keep_prob,
                                                      vocab_size=hparams.vocab_size,
                                                      label_smoothing=hparams.student_label_smoothing,
                                                      encoder_self_attention_dir=hparams.student_encoder_attention_dir,
                                                      decoder_self_attention_dir="top_down",
                                                      decoder_cross_attention_dir="top_down",
                                                      train_embeddings=hparams.student_train_embeddings,
                                                      attention_mechanism=None,
                                                      sent_rep_mode=hparams.student_sent_rep_mode,
                                                      embedding_dim=hparams.student_hidden_dim,
                                                      learning_rate=hparams.student_learning_rate,
                                                      cls_token = CLS_TOKEN[hparams.student_model],
                                                      attention_dropout_keepprob=hparams.student_hidden_dropout_keep_prob,
                                                      relu_dropout_keepprob=hparams.student_relu_dropout_keepprob,
                                                      postprocess_dropout_keepprob=hparams.student_postprocess_dropout_keepprob,
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
  distiller = Seq2SeqParallel(hparams, student_model, teacher_model, trainer)
  distiller.train()
