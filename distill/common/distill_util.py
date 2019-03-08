import tensorflow as tf

def get_logit_distill_loss(student_logits, teacher_logits, softmax_temperature=1.0):
  teacher_logits = tf.stop_gradient(teacher_logits)
  loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                                logits=student_logits / softmax_temperature, labels=tf.nn.sigmoid(teacher_logits)))

  loss = tf.square(softmax_temperature) * loss

  return loss


def get_single_state_rsa_distill_loss(student_states, teacher_states):
  pass