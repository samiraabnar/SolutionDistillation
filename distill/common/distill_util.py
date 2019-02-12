import tensorflow as tf

def get_logit_distill_loss(student_logits, teacher_logits, softmax_temperature=1.0):
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                logits=student_logits / softmax_temperature, labels=teacher_logits))

  loss = tf.square(softmax_temperature) * loss

  return loss