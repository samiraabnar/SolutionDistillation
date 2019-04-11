from distill.pipelines.basic_evaluator import Evaluator
from distill.pipelines.seq2seq import Seq2SeqTrainer


class AlgorithmicTrainer(Seq2SeqTrainer):
  def __init__(self, config, model_obj, task):
    super(AlgorithmicTrainer, self).__init__(config, model_obj, task)



class AlgorithmicEvaluator(Evaluator):
  def __init__(self, config, model_obj, task):
    super(AlgorithmicEvaluator, self).__init__(config, model_obj,task)



