import tensorflow as tf

from distill.common.hparams import TransformerHparam, LSTMHparam
import os

from distill.data_util.prep_algorithmic import AlgorithmicIdentityDecimal40, AlgorithmicIdentityBinary40, \
  AlgorithmicAdditionDecimal40, AlgorithmicMultiplicationDecimal40, AlgorithmicSortProblem, AlgorithmicReverseProblem
from distill.data_util.prep_arithmatic import Arithmatic
from distill.data_util.prep_sst import SST
from distill.models.lstm_seq2seq import LSTMSeq2Seq, BidiLSTMSeq2Seq
from distill.models.transformer import Transformer, UniversalTransformer
from distill.pipelines.distill_pipelines import Seq2SeqDistiller
from distill.pipelines.seq2seq import Seq2SeqTrainer

tf.logging.set_verbosity(tf.logging.INFO)

tf.app.flags.DEFINE_string("exp_name", "distill", "")
tf.app.flags.DEFINE_string("task_name", "identity_binary", "")
tf.app.flags.DEFINE_string("log_dir", "logs", "")
tf.app.flags.DEFINE_string("save_dir", None, "")

tf.app.flags.DEFINE_string("teacher_model", "transformer", "")
tf.app.flags.DEFINE_boolean("train_teacher", True, "")
tf.app.flags.DEFINE_boolean("train_student", False, "")
tf.app.flags.DEFINE_boolean("distill_rep", False, "")
tf.app.flags.DEFINE_boolean("distill_logit", True, "")

tf.app.flags.DEFINE_string("student_model", "lstm", "")
tf.app.flags.DEFINE_boolean("pretrain_teacher", True, "")
tf.app.flags.DEFINE_integer("teacher_pretraining_iters", 100, "")
tf.app.flags.DEFINE_string("rep_loss_mode", 'dot_product', "representation loss type (squared,softmax_cross_ent,sigmoid_cross_ent")


tf.app.flags.DEFINE_string("model_type", "", "")
tf.app.flags.DEFINE_integer("student_hidden_dim", 128, "")
tf.app.flags.DEFINE_integer("student_depth", 2, "")
tf.app.flags.DEFINE_integer("teacher_hidden_dim", 128, "")
tf.app.flags.DEFINE_integer("teacher_depth", 2, "")
tf.app.flags.DEFINE_integer("input_dim", None, "")
tf.app.flags.DEFINE_integer("output_dim", 1, "")
tf.app.flags.DEFINE_string("student_attention_mechanism", None, "")
tf.app.flags.DEFINE_string("teacher_attention_mechanism", None, "")
tf.app.flags.DEFINE_string("sent_rep_mode", 'final', "")
tf.app.flags.DEFINE_integer("number_of_heads", 4, "")
tf.app.flags.DEFINE_integer("ff_filter_size", 512, "")
tf.app.flags.DEFINE_float("initializer_gain", 1.0, "")
tf.app.flags.DEFINE_float("label_smoothing", 0.1, "")

tf.app.flags.DEFINE_string("loss_type", "root_loss", "")
tf.app.flags.DEFINE_float("input_dropout_keep_prob", 0.75, "")
tf.app.flags.DEFINE_float("hidden_dropout_keep_prob", 0.25, "")

tf.app.flags.DEFINE_float("learning_rate", 0.00001, "")
tf.app.flags.DEFINE_float("l2_rate", 0.001, "")

tf.app.flags.DEFINE_integer("batch_size", 32, "")
tf.app.flags.DEFINE_integer("training_iterations", 30000, "")

tf.app.flags.DEFINE_integer("vocab_size", 8000, "")
tf.app.flags.DEFINE_integer("embedding_dim", 300, "embeddings dim")


tf.app.flags.DEFINE_string("pretrained_embedding_path", "data/sst/filtered_glove.txt", "pretrained embedding path")
tf.app.flags.DEFINE_string("data_path", "./data", "data path")


hparams = tf.app.flags.FLAGS


if __name__ == '__main__':


  Models = {"lstm": LSTMSeq2Seq,
            "bilstm": BidiLSTMSeq2Seq,
            "transformer": Transformer,
            "utransformer": UniversalTransformer}


  tasks = {'identity': AlgorithmicIdentityDecimal40('data/alg'),
           'identity_binary': AlgorithmicIdentityBinary40('data/alg'),
           'addition': AlgorithmicAdditionDecimal40('data/alg'),
           'multiplication': AlgorithmicMultiplicationDecimal40('data/alg'),
           'sort': AlgorithmicSortProblem('data/alg'),
           'reverse': AlgorithmicReverseProblem('data/alg'),
           'arithmatic': Arithmatic('data/arithmatic'),
           'sst': SST(data_path="data/sst/",
                 add_subtrees=True,
                 pretrained=True,
                 pretrained_path="data/sst/filtered_glove.txt",
                 embedding_size=300)}

  hparams.vocab_size = tasks[hparams.task_name].vocab_length
  hparams.output_dim = len(tasks[hparams.task_name].target_vocab)

  transformer_params = TransformerHparam(input_dim=hparams.input_dim,
                                         hidden_dim=hparams.teacher_hidden_dim,
                                         output_dim=hparams.output_dim,
                                         depth=hparams.teacher_depth,
                                         number_of_heads=4,
                                         ff_filter_size=512,
                                         initializer_gain=hparams.initializer_gain,
                                         batch_size=hparams.batch_size,
                                         pretrained_embedding_path=hparams.pretrained_embedding_path,
                                         input_dropout_keep_prob=hparams.input_dropout_keep_prob,
                                         hidden_dropout_keep_prob=0.75,
                                         vocab_size=hparams.vocab_size,
                                         label_smoothing=hparams.label_smoothing
                                         )

  lstm_params = LSTMHparam(input_dim=hparams.input_dim,
                           hidden_dim=hparams.student_hidden_dim,
                           output_dim=hparams.output_dim,
                           depth=hparams.student_depth,
                           number_of_heads=hparams.number_of_heads,
                           ff_filter_size=hparams.ff_filter_size,
                           initializer_gain=hparams.initializer_gain,
                           batch_size=hparams.batch_size,
                           pretrained_embedding_path=hparams.pretrained_embedding_path,
                           input_dropout_keep_prob=hparams.input_dropout_keep_prob,
                           hidden_dropout_keep_prob=hparams.hidden_dropout_keep_prob,
                           vocab_size=hparams.vocab_size,
                           label_smoothing=hparams.label_smoothing,
                           attention_mechanism=None,
                           sent_rep_mode=hparams.sent_rep_mode,
                           embedding_dim=hparams.vocab_size / 2 if hparams.vocab_size < 100 else 100
                           )


  model_params = {"transformer": transformer_params,
                  "utransformer": transformer_params,
                  "lstm": lstm_params,
                  "bilstm": lstm_params}


  hparams.model_type ='_'.join([hparams.teacher_model,'to',hparams.student_model])
  if hparams.save_dir is None:
    hparams.save_dir = os.path.join(hparams.log_dir, hparams.task_name, '_'.join(
      [hparams.rep_loss_mode, hparams.model_type, hparams.loss_type, 'std_depth' + str(hparams.student_depth),
       'teacher_depth' + str(hparams.teacher_depth), 'std_hidden_dim' + str(hparams.student_hidden_dim),
       'teacher_hidden_dim' + str(hparams.teacher_hidden_dim), hparams.exp_name]))

  student_model = Models[hparams.student_model](model_params[hparams.student_model],
                                task= tasks[hparams.task_name],
                                scope=hparams.student_model+"_student")
  teacher_model = Models[hparams.teacher_model](model_params[hparams.teacher_model],
                                        task=tasks[hparams.task_name],
                                        scope=hparams.teacher_model+"_teacher")

  trainer = Seq2SeqTrainer(hparams, teacher_model, tasks[hparams.task_name])
  distiller = Seq2SeqDistiller(hparams, student_model, teacher_model, trainer)
  distiller.train()
