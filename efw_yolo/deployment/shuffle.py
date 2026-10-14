"""Conservative channel-shuffle rewrites derived from the historical surgery script.

Recognize only static two-group Reshape/Transpose/Reshape shuffles. Absorb a
right-branch permutation only into a single dense Conv with constant weights;
all other recognized shuffles become an exact channel Gather.
"""
from __future__ import annotations

import copy
import numpy as np
import onnx
from onnx import helper, numpy_helper


def _attributes(node):
    return {a.name: helper.get_attribute_value(a) for a in node.attribute}


def _sort_nodes(graph):
    available = {x.name for x in graph.input} | {x.name for x in graph.initializer}
    pending, ordered = list(graph.node), []
    while pending:
        ready = [node for node in pending if all(not name or name in available for name in node.input)]
        if not ready:
            raise ValueError("Rewrite left a cycle or missing graph input")
        ready_outputs = {name for node in ready for name in node.output}
        ordered.extend(ready)
        available.update(ready_outputs)
        pending = [node for node in pending if not set(node.output).intersection(ready_outputs)]
    del graph.node[:]
    graph.node.extend(ordered)


def optimize_graph(original):
    model = copy.deepcopy(original)
    onnx.checker.check_model(model)
    nodes = list(model.graph.node)
    producers = {name: node for node in nodes for name in node.output}
    consumers = {}
    for node in nodes:
        for name in node.input:
            consumers.setdefault(name, []).append(node)
    constants = {x.name: numpy_helper.to_array(x) for x in model.graph.initializer}
    for node in nodes:
        if node.op_type == "Constant" and "value" in _attributes(node):
            constants[node.output[0]] = numpy_helper.to_array(_attributes(node)["value"])
    graph_outputs = {x.name for x in model.graph.output}
    occupied = set(producers) | set(constants) | {x.name for x in model.graph.input}
    removed, additions, new_initializers = set(), [], []
    details = []

    def unique(base):
        name = base
        index = 1
        while name in occupied:
            name = f"{base}_{index}"
            index += 1
        occupied.add(name)
        return name

    def gather(source, output, indices):
        name = unique("efw_shuffle_indices")
        new_initializers.append(numpy_helper.from_array(np.asarray(indices, dtype=np.int64), name=name))
        additions.append(helper.make_node("Gather", [source, name], [output], axis=1, name=unique("efw_shuffle_gather")))

    def half_slice(node, channels):
        if node.op_type != "Slice" or len(node.input) < 4:
            return None
        values = [constants.get(name) for name in node.input[1:]]
        if any(value is None for value in values) or any(value.size != 1 for value in values):
            return None
        start, end, axis = [int(value.item()) for value in values[:3]]
        step = int(values[3].item()) if len(values) > 3 else 1
        if axis != 1 or step != 1:
            return None
        if start == 0 and end == channels:
            return "left"
        if start == channels and end >= 2 * channels:
            return "right"
        return None

    for trans in nodes:
        if trans.op_type != "Transpose" or _attributes(trans).get("perm") != [0, 2, 1, 3, 4]:
            continue
        before = producers.get(trans.input[0])
        successors = consumers.get(trans.output[0], [])
        if before is None or before.op_type != "Reshape" or len(successors) != 1 or successors[0].op_type != "Reshape":
            continue
        after = successors[0]
        if len(consumers.get(before.output[0], [])) != 1 or before.output[0] in graph_outputs or trans.output[0] in graph_outputs:
            continue
        target = constants.get(before.input[1])
        target_after = constants.get(after.input[1])
        if target is None or target_after is None or len(target) != 5 or int(target[1]) != 2:
            continue
        channels = int(target[2])
        if channels <= 0 or channels % 2 or len(target_after) != 4:
            continue
        # Static shapes or ONNX's inferred channel dimension only; preserve N/H/W exactly.
        if list(target_after)[0] != target[0] or list(target_after)[2:] != list(target)[3:] or int(target_after[1]) not in (-1, 2 * channels):
            continue
        source = before.input[0]
        following = consumers.get(after.output[0], [])
        halves = {half_slice(node, channels): node for node in following}
        absorbed = False
        if len(following) == 2 and set(halves) == {"left", "right"} and after.output[0] not in graph_outputs:
            left, right = halves["left"], halves["right"]
            right_users = consumers.get(right.output[0], [])
            if len(right_users) == 1 and right_users[0].op_type == "Conv" and right.output[0] not in graph_outputs:
                conv = right_users[0]
                weight = constants.get(conv.input[1])
                if _attributes(conv).get("group", 1) == 1 and weight is not None and weight.ndim == 4 and weight.shape[1] == channels:
                    # W @ interleave(R) == W[:, deinterleave] @ R. Copy weights to preserve sharing.
                    permutation = list(range(0, channels, 2)) + list(range(1, channels, 2))
                    weight_name = unique("efw_shuffle_absorbed_weight")
                    new_initializers.append(numpy_helper.from_array(weight[:, permutation].copy(), name=weight_name))
                    conv.input[1] = weight_name
                    left_indices = [j for i in range(channels // 2) for j in (i, channels + i)]
                    right_indices = list(range(channels // 2, channels)) + list(range(3 * channels // 2, 2 * channels))
                    gather(source, left.output[0], left_indices)
                    gather(source, right.output[0], right_indices)
                    removed.update([left.output[0], right.output[0]])
                    absorbed = True
        if not absorbed:
            indices = [j for i in range(channels) for j in (i, channels + i)]
            gather(source, after.output[0], indices)
        removed.update([before.output[0], trans.output[0], after.output[0]])
        details.append({"transpose": trans.name, "channels": 2 * channels,
                        "strategy": "partial_right_conv_absorption" if absorbed else "exact_gather",
                        "added_gathers": 2 if absorbed else 1})
    kept = [node for node in nodes if not set(node.output).intersection(removed)]
    del model.graph.node[:]
    model.graph.node.extend(kept + additions)
    model.graph.initializer.extend(new_initializers)
    _sort_nodes(model.graph)
    # Drop stale intermediate shape annotations for removed tensors; outputs retain their public names.
    value_info = [info for info in model.graph.value_info if info.name not in removed]
    del model.graph.value_info[:]
    model.graph.value_info.extend(value_info)
    onnx.checker.check_model(model)
    report = {"matched": len(details), "recognized": len(details),
              "absorbed_partial": sum(d["strategy"] == "partial_right_conv_absorption" for d in details),
              "gather_fallback": sum(d["strategy"] == "exact_gather" for d in details), "rewrites": details,
              "added_gather_operators": sum(d["added_gathers"] for d in details),
              "total_gather_operators": sum(n.op_type == "Gather" for n in model.graph.node),
              "semantics": "Partial absorption moves only the right-branch permutation into dense Conv weights; both branch selections still execute Gather.",
              "remaining_shuffle_transposes": sum(n.op_type == "Transpose" and _attributes(n).get("perm") == [0, 2, 1, 3, 4] for n in model.graph.node)}
    return model, report
