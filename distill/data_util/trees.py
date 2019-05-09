"""Classes and methods to deal with trees.
"""
class Node:  # a node in the tree
  def __init__(self, label, word=None):
    self.label = label
    self.word = word
    self.parent = None  # reference to parent
    self.left = None  # reference to left child
    self.right = None  # reference to right child
    # true if I am a leaf (could have probably derived this from if I have
    # a word)
    self.isLeaf = False
    # true if we have finished performing fowardprop on this node (note,
    # there are many ways to implement the recursion.. some might not
    # require this flag)

  def __str__(self):
    if self.isLeaf:
      return '[{0}:{1}]'.format(self.word, self.label)
    return '({0} <- [{1}:{2}] -> {3})'.format(self.left, self.word, self.label, self.right)


class Tree:

  def __init__(self, treeString, openChar='(', closeChar=')'):
    tokens = []
    self.open = '('
    self.close = ')'
    for toks in treeString.strip().split():
      tokens += list(toks)
    self.root = self.parse(tokens)
    # get list of labels as obtained through a post-order traversal
    self.labels = get_labels(self.root)
    self.num_words = len(self.labels)
    self.tree_string = treeString

  def parse(self, tokens, parent=None):
    assert tokens[0] == self.open, "Malformed tree"
    assert tokens[-1] == self.close, "Malformed tree"

    split = 2  # position after open and label
    countOpen = countClose = 0

    if tokens[split] == self.open:
      countOpen += 1
      split += 1
    # Find where left child and right child split
    while countOpen != countClose:
      if tokens[split] == self.open:
        countOpen += 1
      if tokens[split] == self.close:
        countClose += 1
      split += 1

    # New node
    node = Node(int(tokens[1]))  # zero index labels

    node.parent = parent

    # leaf Node
    if countOpen == 0:
      node.word = ''.join(tokens[2:-1]).lower()  # lower case?
      node.isLeaf = True
      return node

    node.left = self.parse(tokens[2:split], parent=node)
    node.right = self.parse(tokens[split:-1], parent=node)

    return node

  def get_words(self):
    leaves = getLeaves(self.root)
    words = [node.word for node in leaves]
    return words


def get_subtrees(node, openChar='(', closeChar=')'):
  if node.isLeaf:
    return [Tree(' '.join([openChar, str(node.label), node.word, closeChar]))]

  left_sub_trees = []
  if node.left is not None:
    left_sub_trees = get_subtrees(node.left)

  write_sub_trees = []
  if node.right is not None:
    write_sub_trees = get_subtrees(node.right)

  return [Tree(get_tree_string(node))] + left_sub_trees + write_sub_trees


def leftTraverse(node, nodeFn=None, args=None):
  """
  Recursive function traverses tree
  from left to right.
  Calls nodeFn at each node
  """
  if node is None:
    return
  leftTraverse(node.left, nodeFn, args)
  leftTraverse(node.right, nodeFn, args)
  nodeFn(node, args)


def getLeaves(node):
  if node is None:
    return []
  if node.isLeaf:
    return [node]
  else:
    return getLeaves(node.left) + getLeaves(node.right)


def get_tree_string(node, openChar='(', closeChar=')'):
  if node is None:
    return ''
  if node.isLeaf:
    return ' '.join([openChar, str(node.label), node.word, closeChar])
  else:
    return ' '.join([openChar, str(node.label), get_tree_string(node.left), get_tree_string(node.right), closeChar])

def get_labels(node):
  if node is None:
    return []
  return get_labels(node.left) + get_labels(node.right) + [node.label]


if __name__ == '__main__':
  tree = Tree('(3 (3 (2 If) (3 (2 you) (3 (2 sometimes) (2 (2 like) (3 (2 to) (3 (3 (2 go) (2 (2 to) (2 (2 the) (2 movies)))) (3 (2 to) (3 (2 have) (4 fun))))))))) (2 (2 ,) (2 (2 Wasabi) (3 (3 (2 is) (2 (2 a) (2 (3 good) (2 (2 place) (2 (2 to) (2 start)))))) (2 .)))))')
  sub_trees = get_subtrees(tree.root)

  for sub_tree in sub_trees:
    print(sub_tree.tree_string)