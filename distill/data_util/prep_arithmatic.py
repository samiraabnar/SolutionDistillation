import numpy as np
from random import choices, uniform, randint, sample
from distill.data_util.trees import Tree

def minus(x, y):
  return x - y

def plus(x, y):
  return x + y


def binary_math_tree_generator(length, numbers, ops):
  if length == 1:
    return str(np.random.choice(numbers))
  else:
    left_length = np.random.randint(1,length)
    right_length = length - left_length
    left_child = binary_math_tree_generator(left_length, numbers, ops)
    right_child = binary_math_tree_generator(right_length, numbers, ops)

    op = np.random.choice(ops)

    exp = '('+left_child + op + right_child + ')'
    print(exp, '= ', eval(exp))
    return exp


def first_draft():
    max_length = 10
    min_length = 2

    numbers = [1,2,3,4,5,6,7,8,9,10]
    numbers_ids = [0,1,2,3,4,5,6,7,8,9]
    input_probs = [0.1]*10
    operations = [minus, plus]
    operations_codes = ['-','+']
    operations_ids = [0, 1]
    operation_probs = [0.5]*2

    length = randint(min_length, max_length)
    print("Length:", length)
    print("Random input:", choices(numbers, input_probs))
    print("Probability of adding open param: ", uniform(0,1))
    print("Random operation", choices(operations_codes, operation_probs))

    number_of_examples = 2000
    examples = []
    labels = []
    trees = []
    while len(examples) < number_of_examples:
      example = []
      open_params = 0
      length = randint(min_length, max_length)
      numb_ids = np.random.choice(numbers_ids, length, replace=True)
      op_ids = np.random.choice(operations_ids, length-1, replace=True)
      ops_codes = np.asarray(operations_codes)[op_ids]
      ops = np.asarray(operations)[op_ids]
      numbs = np.asarray(numbers)[numb_ids]
      for i in np.arange(length-1):
        opened = False
        if uniform(0,1) > 0.5: #open para?
          example.append('(')
          open_params +=1
          opened = True

        answer = 0
        example.append(str(numbs[i]))
        if not opened:
          while open_params > 0 and uniform(0,1) > 0.5: #close para?
            example.append(')')
            open_params -= 1

        example.append(ops_codes[i])

      example.append(str(numbs[length-1]))

      while open_params > 0:  # close all remaining open paras
        example.append(')')
        open_params -= 1


      if example not in examples:
        examples.append(example)
        labels.append(eval(' '.join(example)))
        trees.append(Tree(' '.join(example)))
        print(examples[-1], labels[-1], trees[-1].get_words())



if __name__ == '__main__':
  binary_math_tree_generator(5, [0,1,2,3,4,5,6,7,8,9], ['-','+'])

