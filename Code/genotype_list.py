from typing import List, Text

class ArchitectureDecoder:
  def str2lists(arch_str: Text) -> List[tuple]:
    """
    This function shows how to read the string-based architecture encoding.

    :param
      arch_str: the input is a string indicates the architecture topology, such as
      |nor_conv_1x1~0|+|none~0|none~1|+|none~0|none~1|skip_connect~2|
    :return: a list of tuple, contains multiple (op, input_node_index) pairs.

    :usage
      arch = ArchitectureDecoder.str2lists( '|nor_conv_1x1~0|+|none~0|none~1|+|none~0|none~1|skip_connect~2|' )
      print ('there are {:} nodes in this arch'.format(len(arch)+1)) # arch is a list
      for i, node in enumerate(arch):
        print('the {:}-th node is the sum of these {:} nodes with op: {:}'.format(i+1, len(node), node))
    """
    node_strs = arch_str.split('+')
    genotypes = []
    for i, node_str in enumerate(node_strs):
      inputs = list(filter(lambda x: x != '', node_str.split('|')))
      for xinput in inputs: assert len(xinput.split('~')) == 2, 'invalid input length : {:}'.format(xinput)
      inputs = (xi.split('~') for xi in inputs)
      input_infos = tuple((op, int(IDX)) for (op, IDX) in inputs)
      genotypes.append( input_infos )
    return genotypes

# arch_str = "|nor_conv_1x1~0|+|none~0|none~1|+|none~0|none~1|skip_connect~2|"
# genotype = ArchitectureDecoder.str2lists(arch_str)
# print(genotype)