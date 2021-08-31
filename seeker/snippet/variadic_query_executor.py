#date: 2021-08-31T02:07:37Z
#url: https://api.github.com/gists/d930e0dd697582e5a7ad606a3b4e7853
#owner: https://api.github.com/users/KiddoZhu

import torch
from torch import nn
from torch import autograd
from torch.nn import functional as F

from torchdrug import core, utils
from torchdrug.layers import functional
from torchdrug.core import Registry as R


@R.register("framework.QueryExecutor")
class QueryExecutor(nn.Module, core.Configurable):

    any = 1 << 25
    projection = 1 << 26
    intersection = 1 << 27
    union = 1 << 28
    negation = 1 << 29
    stop = 1 << 30

    max_stack_size = 2

    def __init__(self, model, symbolic_model, t_norm="product", any_mode="mean", remove_inverse=False):
        super(QueryExecutor, self).__init__()
        self.model = model
        self.symbolic_model = symbolic_model
        self.t_norm = t_norm
        self.any_mode = any_mode
        self.remove_inverse = remove_inverse

    def conjunction(self, x, y):
        if self.t_norm == "godel":
            return torch.min(x, y)
        elif self.t_norm == "product":
            return x * y
        elif self.t_norm == "lukasiewicz":
            return (x + y - 1).clamp(min=0)
        elif self.t_norm == "rw":
            return x * y
        else:
            raise ValueError("Unknown t-norm `%s`" % self.t_norm)

    def disjunction(self, x, y):
        if self.t_norm == "godel":
            return torch.max(x, y)
        elif self.t_norm == "product":
            return x + y - x * y
        elif self.t_norm == "lukasiewicz":
            return (x + y).clamp(max=1)
        elif self.t_norm == "rw":
            return x + y
        else:
            raise ValueError("Unknown t-norm `%s`" % self.t_norm)

    def forward(self, graph, query, all_loss=None, metric=None):
        if all_loss is not None and self.symbolic_model is not None:
            # training, drop ground truth edges according to the symbolic model
            t_norm = self.t_norm
            self.t_norm = "rw"
            query_i2u = query.clone()
            query_i2u[query == -self.intersection] = -self.union
            output = self.execute(self.symbolic_model, graph, query_i2u, separate_grad=True, all_loss=all_loss, metric=metric)
            prob = output["probability"]

            step_graphs = output["step_graphs"]
            edge_weights = [graph.edge_weight for graph in step_graphs]
            edge_grads = autograd.grad(prob, edge_weights, prob)
            edge_masks = [grad == 0 for grad in edge_grads]
            if self.remove_inverse:
                node_in, node_out, relation = graph.edge_list.t()
                key = (node_in * graph.num_node + node_out) * graph.num_relation + relation
                inverse_key = (node_out * graph.num_node + node_in) * graph.num_relation + (relation ^ 1)
                order = key.argsort()
                inverse_order = inverse_key.argsort()
                assert (key[order] == inverse_key[inverse_order]).all()
                index2inverse = torch.zeros(graph.num_edge, dtype=torch.long, device=self.device)
                index2inverse[order] = inverse_order
                new_edge_masks = []
                for edge_mask in edge_masks:
                    inverse_mask = functional.as_mask(index2inverse[edge_mask], graph.num_edge)
                    new_edge_mask = edge_mask & inverse_mask
                    new_edge_masks.append(new_edge_mask)
                edge_masks = new_edge_masks
            self.t_norm = t_norm
        else:
            edge_masks = None

        output = self.execute(self.model, graph, query, edge_masks=edge_masks, all_loss=all_loss, metric=metric)
        return output["probability"]

    def execute(self, model, graph, query, edge_masks=None, separate_grad=False, all_loss=None, metric=None):
        batch_size = len(query)
        stop = torch.ones(batch_size, 1, dtype=torch.long, device=self.device) * self.stop
        query = torch.cat([query, stop], dim=-1)

        self.stack = torch.zeros(batch_size, self.max_stack_size, graph.num_node, device=self.device)
        # stack pointer
        self.SP = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        # instruction pointer
        self.IP = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        step_graphs = []
        i = 0
        while True:
            token = query.gather(1, self.IP.unsqueeze(-1)).squeeze(-1)
            if (token == self.stop).all():
                break
            is_stop = token == self.stop
            is_operand = ~is_stop & (token >= 0)
            is_negation = ~is_operand & ~is_stop & (-token & self.negation > 0)
            is_intersection = ~is_operand & ~is_stop & (-token & self.intersection > 0)
            is_union = ~is_operand & ~is_stop & (-token & self.union > 0)
            is_projection = ~is_stop & (-token & self.projection > 0)
            if is_operand.any():
                self.apply_operand(is_operand, token, graph.num_node)
            if is_negation.any():
                self.apply_negation(is_negation)
            if is_intersection.any():
                self.apply_intersection(is_intersection)
            if is_union.any():
                self.apply_union(is_union)
            # apply projection if there are no other operations
            # this maximize the efficiency for projection
            if not (is_operand | is_negation | is_intersection | is_union).any() and is_projection.any():
                if separate_grad:
                    step_graph = graph.clone().requires_grad_()
                else:
                    step_graph = graph
                if edge_masks:
                    metric["step %d #edge masked" % i] = (~edge_masks[i]).sum().float()
                    step_graph = step_graph.edge_mask(edge_masks[i])
                    i += 1
                self.apply_projection(is_projection, token, model, step_graph, all_loss=all_loss, metric=metric)
                step_graphs.append(step_graph)

        if (self.SP > 1).any():
            raise ValueError("More operands than expected")
        prob = self.stack[:, 0]

        if metric is not None:
            num_projection = ((query < 0) & (-query & self.projection > 0)).sum(dim=-1)
            metric["#projection"] = num_projection.float().mean()

        return {
            "probability": prob,
            "step_graphs": step_graphs,
        }

    def push(self, mask, value):
        assert (self.SP[mask] < self.max_stack_size).all()
        self.stack[mask, self.SP[mask]] = value
        self.SP[mask] += 1

    def pop(self, mask):
        assert (self.SP[mask] > 0).all()
        self.SP[mask] -= 1
        return self.stack[mask, self.SP[mask]]

    def apply_operand(self, mask, token, num_node):
        h_index = token[mask]
        if self.any_mode == "sum":
            h_prob = torch.ones(*h_index.shape, num_node, device=self.device)
        elif self.any_mode == "mean":
            h_prob = torch.ones(*h_index.shape, num_node, device=self.device) / num_node
        else:
            raise ValueError("Unknown any mode `%s`" % self.any_mode)
        is_specific = h_index != self.any
        h_prob[is_specific] = functional.one_hot(h_index[is_specific], num_node)
        self.push(mask, h_prob)
        self.IP[mask] += 1

    def apply_negation(self, mask):
        x = self.pop(mask)
        self.push(mask, 1 - x)
        self.IP[mask] += 1

    def apply_intersection(self, mask):
        y = self.pop(mask)
        x = self.pop(mask)
        self.push(mask, self.conjunction(x, y))
        self.IP[mask] += 1

    def apply_union(self, mask):
        y = self.pop(mask)
        x = self.pop(mask)
        self.push(mask, self.disjunction(x, y))
        self.IP[mask] += 1

    def apply_projection(self, mask, token, model, graph, all_loss=None, metric=None):
        r_index = -token[mask] & ~self.projection
        x = self.pop(mask)
        x = model(graph, x, r_index, all_loss=all_loss, metric=metric)
        if model is not self.symbolic_model:
            x = F.sigmoid(x)
        self.push(mask, x)
        self.IP[mask] += 1