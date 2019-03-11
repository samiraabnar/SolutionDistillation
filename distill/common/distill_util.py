import tensorflow as tf

def get_logit_distill_loss(student_logits, teacher_logits, softmax_temperature=1.0):
  teacher_logits = tf.stop_gradient(teacher_logits)
  loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                                logits=student_logits / softmax_temperature, labels=tf.nn.sigmoid(teacher_logits)))

  loss = tf.square(softmax_temperature) * loss

  return loss


def get_single_state_rsa_distill_loss(student_states, teacher_states):
  teacher_states = tf.stop_gradient(teacher_states)

  teacher_rsm = squared_dist_rsm(teacher_states,teacher_states)
  student_rsm = squared_dist_rsm(student_states, student_states)

  rsa_score = tf.reduce_mean(squared_dist_rsm(teacher_rsm,student_rsm))

  return rsa_score




def squared_dist_rsm(A, B):
  assert A.shape.as_list() == B.shape.as_list()

  row_norms_A = tf.reduce_sum(tf.square(A), axis=1)
  row_norms_A = tf.reshape(row_norms_A, [-1, 1])  # Column vector.

  row_norms_B = tf.reduce_sum(tf.square(B), axis=1)
  row_norms_B = tf.reshape(row_norms_B, [1, -1])  # Row vector.

  return row_norms_A - 2 * tf.matmul(A, tf.transpose(B)) + row_norms_B


if __name__ == '__main__':
    a = tf.constant([[1,1],[2,2],[3,3]])
    b = tf.constant([[1, 2, 3], [2, 3, 4], [3, 4, 5]])

    d_a = squared_dist(a,a)
    d_b = squared_dist(b,b)
    with tf.Session() as sess:
      print(sess.run(d_a))
      print(sess.run(d_b))
