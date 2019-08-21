import tensorflow as tf

def get_logit_distill_loss(student_logits, teacher_logits, softmax_temperature=1.0, stop_grad_for_teacher=True):
  if stop_grad_for_teacher:
    teacher_logits_copy = tf.stop_gradient(teacher_logits)
  else:
    teacher_logits_copy = teacher_logits
    
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                                logits=student_logits / softmax_temperature, labels=tf.nn.softmax(teacher_logits_copy / softmax_temperature)))

  loss = tf.square(softmax_temperature) * loss

  return loss


def get_single_state_rsa_distill_loss(student_states, teacher_states, mode='dot_product', train_teacher=False):
  if not train_teacher:
    teacher_states = tf.stop_gradient(teacher_states)

  #tf.logging.info('state shapes')
  #tf.logging.info(teacher_states)
  #tf.logging.info(student_states)

  teacher_rsm = dot_product_sim(teacher_states,teacher_states, pair_wise=True)
  student_rsm = dot_product_sim(student_states, student_states, pair_wise=True)

  #tf.logging.info('dist shapes')
  #tf.logging.info(teacher_rsm)
  #tf.logging.info(student_rsm)


  if mode == 'squared':
    rsa_score = tf.reduce_mean(squared_dist_rsm(student_rsm,teacher_rsm))
  elif mode == 'softmax_cross_ent':
    rsa_score = tf.reduce_mean(sigmoid_cross_entropy_rsa(student_rsm, teacher_rsm))
  elif mode == 'dot_product':
    rsa_score = 1.0 - tf.reduce_mean(dot_product_sim(student_rsm, teacher_rsm, pair_wise=False))
  else:
    rsa_score = 1.0 - tf.reduce_mean(dot_product_sim(student_rsm, teacher_rsm, pair_wise=False))

  return rsa_score


def get_single_state_uniform_rsa_loss(student_states, mode='dot_product'):

  student_rsm = dot_product_sim(student_states, student_states,pair_wise=True)
  teacher_rsm = tf.ones_like(student_rsm)



  if mode == 'squared':
    rsa_score = tf.reduce_mean(squared_dist_rsm(student_rsm,teacher_rsm))
  elif mode == 'softmax_cross_ent':
    rsa_score = tf.reduce_mean(sigmoid_cross_entropy_rsa(student_rsm, teacher_rsm))
  elif mode == 'dot_product':
    rsa_score = 1.0 - tf.reduce_mean(dot_product_sim(student_rsm, teacher_rsm, pair_wise=False))
  else:
    rsa_score = 1.0 - tf.reduce_mean(dot_product_sim(student_rsm, teacher_rsm, pair_wise=False))

  return rsa_score

def get_biased_single_state_rsa_distill_loss(student_states, teacher_states, mode='dot_product', bias="general"):

  teacher_rsm = dot_product_sim(teacher_states, teacher_states, pair_wise=True)
  student_rsm = dot_product_sim(student_states, student_states, pair_wise=True)

  if mode == 'squared':
    rsa_score = tf.reduce_mean(squared_dist_rsm(student_rsm,teacher_rsm))
  elif mode == 'softmax_cross_ent':
    rsa_score = tf.reduce_mean(sigmoid_cross_entropy_rsa(student_rsm, teacher_rsm))
  elif mode == 'dot_product':
    rsa_score = 1.0 - tf.reduce_mean(dot_product_sim(reweight(student_rsm, bias), reweight(teacher_rsm, bias), pair_wise=False))
  else:
    rsa_score = 1.0 - tf.reduce_mean(dot_product_sim(reweight(student_rsm, bias), reweight(teacher_rsm, bias), pair_wise=False))

  return rsa_score


def dot_product(a, b, pair_wise=False):
  a = tf.reshape(a, [-1, tf.shape(a)[-1]])
  b = tf.reshape(b, [-1, tf.shape(b)[-1]])

  a = tf.cast(a, dtype=tf.float32)
  b = tf.cast(b, dtype=tf.float32)

  if pair_wise:
    sum_ab = tf.matmul(a, b, transpose_b=True)
    sum_aa = tf.reduce_sum(tf.multiply(a,a), axis=-1)
    sum_bb = tf.reduce_sum(tf.multiply(b, b), axis=-1)

    return sum_ab / (tf.sqrt(sum_aa) * tf.sqrt(sum_bb))
  else:
    sum_ab = tf.reduce_sum(tf.multiply(a,b), axis=-1)
    sum_aa = tf.reduce_sum(tf.multiply(a,a), axis=-1)
    sum_bb = tf.reduce_sum(tf.multiply(b,b), axis=-1)

    return tf.reduce_mean(sum_ab / tf.maximum(tf.sqrt(sum_aa) * tf.sqrt(sum_bb),0.000000000001))



def get_rep_sim(student_state, teacher_states, mode, topk=None, similarity_fn=dot_product):
  a = similarity_fn(student_state,student_state, pair_wise=True)
  b = similarity_fn(teacher_states,teacher_states, pair_wise=True)

  _, ar = tf.nn.top_k(a, tf.shape(a)[0])
  _, br = tf.nn.top_k(b, tf.shape(b)[0])

  if topk is not None:
    a = a * tf.cast(ar > tf.shape(ar)[0] - topk, tf.float32)
    b = b * tf.cast(br > tf.shape(br)[0] - topk, tf.float32)

  if mode == "rank":
    return similarity_fn(ar,br)
  elif mode == "degree":
    ad = tf.reduce_sum(a, axis=-1)
    bd = tf.reduce_sum(b, axis=-1)
    return similarity_fn(ad, bd)

  else:
    return similarity_fn(a, b)


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


def dot_product_sim(a, b, pair_wise=True):
  a = tf.reshape(a, [-1, tf.shape(a)[-1]])
  b = tf.reshape(b, [-1, tf.shape(b)[-1]])

  a_norm = tf.expand_dims(tf.norm(a, axis=-1), -1)
  a = a / (a_norm+0.000000001)

  b_norm = tf.expand_dims(tf.norm(b, axis=-1), -1)
  b = b / (b_norm+0.000000001)


  if pair_wise:
    sim_mat = tf.matmul(a, b, transpose_b=True)
  else:
    sim_mat = tf.reduce_sum(tf.multiply(a, b), axis=-1)

  return sim_mat

def reweight(dists, direction="general"):
  # It should be applied before normalizing!
  if direction == "general": #the closer they are: the bigger they will be!
    return 1.0 / tf.math.pow(dists+ 0.000000001, 2)
  else: #local -> the more distant the points, the much more distant they will be!
    return tf.math.pow(dists, 2)


if __name__ == '__main__':
    a = tf.constant([[[1,1],[2,7],[4,3]],[[5,1],[2,20],[13,3]]], dtype=tf.float32)
    b = tf.constant([[[1, 2, 3], [2, 3, 1], [3, 0, 5]],[[3, 2, 3], [2, 3, 4], [3, 4, 5]]], dtype=tf.float32)

    rsa_1 = get_biased_single_state_rsa_distill_loss(a,a)
    rsa_2 = get_single_state_rsa_distill_loss(a,b)
    rsa_3 = get_single_state_rsa_distill_loss(a,a, mode='squared')
    rsa_4 = get_single_state_rsa_distill_loss(a,b, mode='squared')

    a_sim = dot_product(a,a, pair_wise=True)
    b_sim = dot_product(b,b, pair_wise=True)

    rsa_5 = get_rep_sim(a,b,mode="degree",topk=3)
    rsa_55 = get_rep_sim(a,a,mode="degree",topk=3)

    rsa_6 = get_rep_sim(a,b,mode="rank")
    rsa_66 = get_rep_sim(a,a,mode="rank")

    rsa_7 = get_rep_sim(a, b, mode="std")
    rsa_77 = get_rep_sim(a, a, mode="std")


    #tf.logging.info('dist shapes')
    #tf.logging.info(d_a)
    #tf.logging.info(d_b)

    #rsa = squared_dist_rsm(d_a, d_b)

    with tf.Session() as sess:
      # print(sess.run(rsa_1))
      # print(sess.run(rsa_2))
      #
      # print(sess.run(rsa_3))
      # print(sess.run(rsa_4))

      print(sess.run(a_sim))
      print(sess.run(b_sim))

      print(sess.run(rsa_5))
      print(sess.run(rsa_55))

      print(sess.run(rsa_6))
      print(sess.run(rsa_66))

      print(sess.run(rsa_7))
      print(sess.run(rsa_77))


