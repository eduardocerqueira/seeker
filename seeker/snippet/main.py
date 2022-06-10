#date: 2022-06-10T17:16:06Z
#url: https://api.github.com/gists/202bbf58be85ec9647fbed117e9cccad
#owner: https://api.github.com/users/makslevental

import torch.jit
from torch.utils.cpp_extension import load_inline

source = '''
#include "torch/csrc/jit/ir/irparser.h"
#include "torch/csrc/jit/passes/common_subexpression_elimination.h"
#include "torch/csrc/jit/passes/utils/subgraph_utils.h"
using namespace torch::jit;

std::shared_ptr<Graph> getSubgraph(Node* n) {
  return n->g(attr::Subgraph);
}

std::pair<std::shared_ptr<Graph>, std::shared_ptr<Graph>> splitGraph(
    std::shared_ptr<Graph> graph,
    int split_num) {
  auto it = graph->nodes().begin();
  auto end = graph->nodes().end();

  Node* subgraph1 =
      SubgraphUtils::createSingletonSubgraph(*it, prim::DifferentiableGraph);
  it = ++subgraph1->iterator();

  int i = 1;
  for (; it != end;) {
    if (i > split_num - 1)
      break;
    i++;

    SubgraphUtils::mergeNodeIntoSubgraph(*it, subgraph1);
    it = ++subgraph1->iterator();
  }

  Node* subgraph2 =
      SubgraphUtils::createSingletonSubgraph(*it, prim::DifferentiableGraph);
  it = ++subgraph2->iterator();

  for (; it != end;) {
    SubgraphUtils::mergeNodeIntoSubgraph(*it, subgraph2);
    it = ++subgraph2->iterator();
  }

  return std::make_pair(getSubgraph(subgraph1), getSubgraph(subgraph2));
}
'''

module = load_inline(name='inline_extension',
                         cpp_sources=[source],
                         functions=['splitGraph'])


graph_str = """
graph(%a : Tensor, %b : Tensor, %c : Tensor):
  %x : Tensor = aten::sigmoid(%a)
  %y : Tensor = aten::mul(%a, %b)
  %p : Tensor = aten::div(%c, %b)
  %q1 : Tensor = aten::mul(%p, %a)
  %q2 : Tensor = aten::tanh(%q1)
  %q3 : Tensor = aten::tanh(%q2)
  %q4 : Tensor = aten::tanh(%q3)
  %q5 : Tensor = aten::hardsigmoid(%q4)
  return (%x, %y, %q5)
"""

graph = torch._C.parse_ir(graph_str)
print("*** graph before\n", graph)

subgraph1, subgraph2 = module.splitGraph(graph, 2)
print("*** subgraph1\n", subgraph1)
print("*** subgraph2\n", subgraph2)

print("*** graph after\n", graph)
