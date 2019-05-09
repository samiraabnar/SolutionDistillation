import tensorflow as tf

def get_logit_distill_loss(student_logits, teacher_logits, softmax_temperature=1.0):
  teacher_logits = tf.stop_gradient(teacher_logits)
  loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                                logits=student_logits / softmax_temperature, labels=tf.nn.sigmoid(teacher_logits)))

  loss = tf.square(softmax_temperature) * loss

  return loss


def get_single_state_rsa_distill_loss(student_states, teacher_states, mode='dot_product'):
  teacher_states = tf.stop_gradient(teacher_states)

  tf.logging.info('state shapes')
  tf.logging.info(teacher_states)
  tf.logging.info(student_states)

  teacher_rsm = dot_product_sim(teacher_states,teacher_states)
  student_rsm = dot_product_sim(student_states, student_states)

  tf.logging.info('dist shapes')
  tf.logging.info(teacher_rsm)
  tf.logging.info(student_rsm)


  if mode == 'squared':
    rsa_score = tf.reduce_mean(squared_dist_rsm(student_rsm,teacher_rsm))
  elif mode == 'softmax_cross_ent':
    rsa_score = tf.reduce_mean(sigmoid_cross_entropy_rsa(student_rsm, teacher_rsm))
  elif mode == 'dot_product':
    rsa_score = tf.reduce_mean(dot_product_sim(student_rsm, teacher_rsm))
  else:
    rsa_score = tf.reduce_mean(dot_product_sim(student_rsm, teacher_rsm))

  return rsa_score



def sigmoid_cross_entropy_rsa(d_a, d_b):
  rsa = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=d_a, labels=tf.nn.sigmoid(d_b)))
  return rsa

def softmax_cross_entropy_rsa(d_a, d_b):
  rsa = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    logits=d_a, labels=tf.nn.softmax(d_b)))
  return rsa

def squared_dist_rsm(a, b):
  assert a.shape.as_list() == b.shape.as_list()

  a = tf.reshape(a, [-1, tf.shape(a)[-1]])
  b = tf.reshape(b, [-1, tf.shape(b)[-1]])

  row_norms_A = tf.reduce_sum(tf.square(a), axis=1)
  row_norms_A = tf.reshape(row_norms_A, [-1, 1])  # Column vector.

  row_norms_B = tf.reduce_sum(tf.square(b), axis=1)
  row_norms_B = tf.reshape(row_norms_B, [1, -1])  # Row vector.

  return row_norms_A - 2 * tf.matmul(a, tf.transpose(b)) + row_norms_B


def dot_product_sim(a, b):
  a = tf.reshape(a, [-1, tf.shape(a)[-1]])
  b = tf.reshape(b, [-1, tf.shape(b)[-1]])

  a_norm = tf.expand_dims(tf.norm(a, axis=-1), -1)
  a = a / a_norm

  b_norm = tf.expand_dims(tf.norm(b, axis=-1), -1)
  b = b / b_norm

  sim_mat = tf.matmul(a, b,
                     transpose_b=True  # transpose second matrix
                     )

  return sim_mat

if __name__ == '__main__':
    a = tf.constant([[[1,1],[2,7],[4,3]],[[5,1],[2,20],[13,3]]], dtype=tf.float32)
    b = tf.constant([[[1, 2, 3], [2, 3, 1], [3, 0, 5]],[[3, 2, 3], [2, 3, 4], [3, 4, 5]]], dtype=tf.float32)

    rsa_1 = get_single_state_rsa_distill_loss(a,a)
    rsa_2 = get_single_state_rsa_distill_loss(a,b)
    rsa_3 = get_single_state_rsa_distill_loss(a,a, mode='squared')
    rsa_4 = get_single_state_rsa_distill_loss(a,b, mode='squared')

    #tf.logging.info('dist shapes')
    #tf.logging.info(d_a)
    #tf.logging.info(d_b)

    #rsa = squared_dist_rsm(d_a, d_b)

    with tf.Session() as sess:
      print(sess.run(rsa_1))
      print(sess.run(rsa_2))

      print(sess.run(rsa_3))
      print(sess.run(rsa_4))


