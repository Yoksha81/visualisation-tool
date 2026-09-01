"""Računa raspored i iscrtava graf pomoću NetworkX-a i Matplotlib-a."""
import matplotlib
matplotlib.use('TkAgg')
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch as LegendPatch
from matplotlib.path import Path
import re

MIN_OPERATION_WIDTH = 1.8
MIN_MODULE_WIDTH = 2.2
OPERATION_HEIGHT = 0.9
MODULE_HEIGHT = 1.2
CHAR_WIDTH = 0.13
HORIZONTAL_TEXT_PADDING = 0.8
NODE_STYLE_INPUT = {'facecolor': '#DDF4E4', 'edgecolor': '#2E6B45'}
NODE_STYLE_OUTPUT = {'facecolor': '#F7DEDE', 'edgecolor': '#8A3D3D'}
NODE_STYLE_EXTERNAL = {'facecolor': '#E9E2F7', 'edgecolor': '#66528A'}
NODE_STYLE_EXPANDABLE_MODULE = {'facecolor': '#D9EAF7', 'edgecolor': '#2F5F7D'}
NODE_STYLE_TERMINAL_MODULE = {'facecolor': '#E9ECEF', 'edgecolor': '#5F6368'}
NODE_STYLE_OPERATION = {'facecolor': '#FFF0C9', 'edgecolor': '#8A681D'}
EDGE_COLOR = '#3F444A'
GROUP_COLOR = '#6B7280'

def napravi_nx_graf(visible_graph):

    DG = nx.DiGraph()

    for node_id, node in visible_graph.nodes.items():
        DG.add_node(node_id)

    for edge in visible_graph.edges:
        DG.add_edge(edge.source, edge.target)

    return DG

def roditelj_grupe(group_id):

    if '.' in group_id:
        return group_id.rsplit('.', 1)[0]

    return 'ROOT'

def razdvoji_grupe_po_visini(nx_graph, visible_graph, layer_of, group_gap_layers=2):
    """Razdvaja velike otvorene sibling module po vertikali da se njihovi čvorovi manje preklapaju."""

    if not visible_graph.groups:
        return layer_of

    topological_nodes = list(nx.topological_sort(nx_graph))
    groups_by_parent = {}

    for group_id in visible_graph.groups:

        if group_id == 'ROOT':
            continue

        parent_id = roditelj_grupe(group_id)
        groups_by_parent.setdefault(parent_id, []).append(group_id)

    parent_ids = sorted(groups_by_parent, key=lambda parent_id: 0 if parent_id == 'ROOT' else parent_id.count('.') + 1)

    for parent_id in parent_ids:

        sibling_ids = groups_by_parent[parent_id]

        if len(sibling_ids) <= 1:
            continue

        sibling_node_sets = {group_id: set(visible_graph.groups[group_id].nodes) for group_id in sibling_ids}
        node_to_group = {}

        for group_id, node_ids in sibling_node_sets.items():
            for node_id in node_ids:
                node_to_group[node_id] = group_id

        group_graph = nx.DiGraph()

        for group_id in sibling_ids:
            group_graph.add_node(group_id)

        for edge in visible_graph.edges:
            source_group = node_to_group.get(edge.source)
            target_group = node_to_group.get(edge.target)

            if source_group is not None and target_group is not None and (source_group != target_group):
                group_graph.add_edge(source_group, target_group)

        if group_graph.number_of_edges() == 0:
            continue

        if not nx.is_directed_acyclic_graph(group_graph):
            continue

        for group_id in nx.topological_sort(group_graph):
            predecessors = list(group_graph.predecessors(group_id))

            if not predecessors:
                continue

            current_nodes = sibling_node_sets[group_id]

            if not current_nodes:
                continue

            required_start = max((max((layer_of[node_id] for node_id in sibling_node_sets[predecessor_id])) for predecessor_id in predecessors if sibling_node_sets[predecessor_id])) + group_gap_layers
            current_start = min((layer_of[node_id] for node_id in current_nodes))
            shift = max(0, required_start - current_start)

            if shift == 0:
                continue

            for node_id in current_nodes:
                layer_of[node_id] += shift

        for node_id in topological_nodes:
            predecessors = list(nx_graph.predecessors(node_id))

            if not predecessors:
                continue

            required_layer = max((layer_of[predecessor_id] for predecessor_id in predecessors)) + 1

            if layer_of[node_id] < required_layer:
                layer_of[node_id] = required_layer

    for node_id, layer in layer_of.items():
        nx_graph.nodes[node_id]['layer'] = layer

    return layer_of

def izracunaj_raspored(nx_graph, visible_graph, vertical_gap=2.5, horizontal_gap=4.0, node_gap=0.8):
    """Računa položaje čvorova po topološkim slojevima DAG-a."""

    if not nx.is_directed_acyclic_graph(nx_graph):
        raise ValueError('Visible graph nije DAG.')

    generations = list(nx.topological_generations(nx_graph))
    layer_of = {}

    for layer, nodes in enumerate(generations):
        for node in nodes:
            layer_of[node] = layer
            nx_graph.nodes[node]['layer'] = layer

    layer_of = razdvoji_grupe_po_visini(nx_graph, visible_graph, layer_of)
    distance_to_end = {}

    for node in reversed(list(nx.topological_sort(nx_graph))):
        successors = list(nx_graph.successors(node))
        if not successors:
            distance_to_end[node] = 0
        else:
            distance_to_end[node] = 1 + max((distance_to_end[successor] for successor in successors))

    x_positions = {}
    sources = [node for node in nx_graph.nodes if nx_graph.in_degree(node) == 0]

    for index, source in enumerate(sources):
        x_positions[source] = (index - (len(sources) - 1) / 2) * horizontal_gap

    for node in nx.topological_sort(nx_graph):
        predecessors = list(nx_graph.predecessors(node))

        if len(predecessors) > 1:
            main_predecessor = max(predecessors, key=lambda predecessor: layer_of[predecessor])
            x_positions[node] = x_positions[main_predecessor]
        elif node not in x_positions and len(predecessors) == 1:
            x_positions[node] = x_positions[predecessors[0]]

        successors = list(nx_graph.successors(node))

        if len(successors) > 1:
            sorted_successors = sorted(successors, key=lambda successor: distance_to_end[successor], reverse=True)
            main_successor = sorted_successors[0]
            x_positions[main_successor] = x_positions[node]
            side_successors = sorted_successors[1:]

            for index, successor in enumerate(side_successors):
                direction = 1 if index % 2 == 0 else -1
                multiplier = index // 2 + 1
                offset = direction * multiplier * horizontal_gap
                x_positions[successor] = x_positions[node] + offset

        elif len(successors) == 1:
            successor = successors[0]
            if nx_graph.in_degree(successor) == 1:
                if successor not in x_positions:
                    x_positions[successor] = x_positions[node]

    pos = {}

    for node in nx_graph.nodes:
        x = x_positions.get(node, 0)
        y = -layer_of[node] * vertical_gap
        pos[node] = (x, y)

    pos = razmakni_cvorove_u_sloju(nx_graph, visible_graph, pos, node_gap=node_gap)

    return pos

def glavni_prethodnik(nx_graph, target):

    predecessors = list(nx_graph.predecessors(target))

    if not predecessors:
        return None

    return max(predecessors, key=lambda node: nx_graph.nodes[node]['layer'])

def x_za_skip_granu(visible_graph, pos, source_y, target_y, source_x, margin=0.8):

    min_y = min(source_y, target_y)
    max_y = max(source_y, target_y)
    left = float('inf')
    right = float('-inf')

    for node_id, node in visible_graph.nodes.items():
        x, y = pos[node_id]
        if min_y <= y <= max_y:
            width, _ = dimenzije_cvora(node)
            left = min(left, x - width / 2)
            right = max(right, x + width / 2)

    if left == float('inf'):
        return source_x + 3.0

    left_lane = left - margin
    right_lane = right + margin

    if abs(source_x - left_lane) < abs(source_x - right_lane):
        return left_lane

    return right_lane

def rasporedi_skip_grane(nx_graph, visible_graph, pos):
    """Dodeljuje bočne trake skip granama koje bi se inače preklapale."""

    skip_edges = []

    for edge in visible_graph.edges:
        target_in_degree = nx_graph.in_degree(edge.target)

        if target_in_degree <= 1:
            continue

        main_predecessor = glavni_prethodnik(nx_graph, edge.target)

        if edge.source == main_predecessor:
            continue

        source_x, source_y = pos[edge.source]
        target_x, target_y = pos[edge.target]

        if abs(source_x - target_x) >= 1e-09:
            continue

        top_y = max(source_y, target_y)
        bottom_y = min(source_y, target_y)
        span = top_y - bottom_y
        skip_edges.append({'key': (edge.source, edge.target), 'top': top_y, 'bottom': bottom_y, 'span': span, 'source_x': source_x})

    skip_edges.sort(key=lambda item: item['span'])
    lanes = {}
    occupied_intervals = []

    for skip in skip_edges:
        lane_index = 0

        while True:

            if lane_index == len(occupied_intervals):
                occupied_intervals.append([])

            conflict = False

            for existing_bottom, existing_top in occupied_intervals[lane_index]:
                overlaps = not (skip['top'] < existing_bottom or skip['bottom'] > existing_top)

                if overlaps:
                    conflict = True
                    break

            if not conflict:
                occupied_intervals[lane_index].append((skip['bottom'], skip['top']))
                lanes[skip['key']] = lane_index
                break

            lane_index += 1

    return lanes

def dimenzije_cvora(node):

    label = formatiraj_oznaku_cvora(node)
    longest_line = max((len(line) for line in label.split('\n')))
    estimated_width = longest_line * CHAR_WIDTH + HORIZONTAL_TEXT_PADDING

    if node.node_kind == 'module':
        width = max(MIN_MODULE_WIDTH, estimated_width)
        height = MODULE_HEIGHT
    else:
        width = max(MIN_OPERATION_WIDTH, estimated_width)
        height = OPERATION_HEIGHT

    return (width, height)

def vizuelna_uloga_cvora(node):
    """Određuje boju čvora: ulaz, izlaz, modul, operacija ili runtime čvor."""

    if node.node_kind == 'operation':
        if node.subtype == 'placeholder':
            return 'input'
        if node.subtype == 'output':
            return 'output'
        if node.subtype == 'external':
            return 'external'
        if node.subtype in ('call_function', 'call_method', 'get_attr'):
            return 'operation'
        return 'terminal_module'

    if node.node_kind == 'module':
        if node.expandable:
            return 'expandable_module'
        return 'terminal_module'

    return 'operation'

def stil_cvora(node):

    role = vizuelna_uloga_cvora(node)

    return {'input': NODE_STYLE_INPUT, 'output': NODE_STYLE_OUTPUT, 'external': NODE_STYLE_EXTERNAL, 'expandable_module': NODE_STYLE_EXPANDABLE_MODULE, 'terminal_module': NODE_STYLE_TERMINAL_MODULE, 'operation': NODE_STYLE_OPERATION}[role]

def iscrtaj_legendu(ax, visible_graph):
    """Prikazuje samo tipove čvorova koji postoje u trenutnom grafu."""

    present_roles = {vizuelna_uloga_cvora(node) for node in visible_graph.nodes.values()}
    specs = [('input', 'Input', NODE_STYLE_INPUT), ('output', 'Output', NODE_STYLE_OUTPUT), ('external', 'Runtime / external', NODE_STYLE_EXTERNAL), ('expandable_module', 'Expandable module', NODE_STYLE_EXPANDABLE_MODULE), ('terminal_module', 'Terminal module', NODE_STYLE_TERMINAL_MODULE), ('operation', 'Operation', NODE_STYLE_OPERATION)]
    handles = []

    for role, label, style in specs:
        if role not in present_roles:
            continue
        handles.append(LegendPatch(facecolor=style['facecolor'], edgecolor=style['edgecolor'], label=label))

    if not handles:
        return

    legend = ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.015), ncol=min(len(handles), 6), frameon=False, fontsize=7, handlelength=1.4, columnspacing=1.2)
    legend.set_zorder(20)

def iscrtaj_cvorove(ax, visible_graph, pos):

    node_patches = {}

    for node_id, node in visible_graph.nodes.items():
        x, y = pos[node_id]
        width, height = dimenzije_cvora(node)
        label = formatiraj_oznaku_cvora(node)
        style = stil_cvora(node)
        box = FancyBboxPatch((x - width / 2, y - height / 2), width, height, boxstyle='round,pad=0.08', linewidth=1.5, facecolor=style['facecolor'], edgecolor=style['edgecolor'], zorder=2)
        ax.add_patch(box)
        ax.text(x, y, label, ha='center', va='center', fontsize=8, zorder=3)
        node_patches[node_id] = box

    return node_patches

def dubina_grupe(group_id):

    if group_id == 'ROOT':
        return 0

    return group_id.count('.') + 1

def formatiraj_oznaku_grupe(group):

    local_name = group.id.rsplit('.', 1)[-1]
    if local_name.isdigit():
        return f'{group.module_type} {local_name}'

    return f'{local_name} [{group.module_type}]'

def formatiraj_oznaku_modula(node):

    if node.id == 'ROOT':
        return node.subtype

    local_name = node.label

    if local_name.isdigit():
        return f'{node.subtype} {local_name}'

    if local_name == node.subtype:
        return local_name

    return f'{local_name}\n[{node.subtype}]'

OPERATION_DISPLAY_NAMES = {'getitem': 'GetItem', 'getattr': 'GetAttr', 'setitem': 'SetItem', 'add': 'Add', 'sub': 'Sub', 'mul': 'Mul', 'matmul': 'MatMul', 'truediv': 'TrueDiv', 'floordiv': 'FloorDiv', 'eq': 'Eq', 'ne': 'Ne', 'lt': 'Lt', 'le': 'Le', 'gt': 'Gt', 'ge': 'Ge', 'cat': 'Cat', 'stack': 'Stack', '_assert': 'Assert'}

def ime_operacije_za_prikaz(name):

    if name in OPERATION_DISPLAY_NAMES:
        return OPERATION_DISPLAY_NAMES[name]

    parts = [part for part in name.strip('_').split('_') if part]

    if parts:
        return ''.join((part[0].upper() + part[1:] for part in parts))

    return name

def normalizuj_oznaku_operacije(label):

    match = re.fullmatch('<built-in function ([^>]+)>', label)

    if match:
        function_name = match.group(1)
        return ime_operacije_za_prikaz(function_name)

    match = re.fullmatch('<built-in method ([^\\s]+).*?>', label)

    if match:
        method_name = match.group(1)
        return ime_operacije_za_prikaz(method_name)

    match = re.fullmatch('<function ([^\\s]+) at 0x[0-9A-Fa-f]+>', label)

    if match:
        function_name = match.group(1)
        function_name = function_name.rsplit('.', 1)[-1]
        return ime_operacije_za_prikaz(function_name)

    if label.startswith('aten.'):
        parts = label.split('.')

        if len(parts) >= 2:
            operation_name = parts[1]
            return ime_operacije_za_prikaz(operation_name)

    if '.' in label and ' ' not in label:
        name = label.rsplit('.', 1)[-1]
        if name:
            return ime_operacije_za_prikaz(name)

    return label

def formatiraj_oznaku_cvora(node):

    if node.node_kind == 'module':
        return formatiraj_oznaku_modula(node)

    return normalizuj_oznaku_operacije(node.label)

def razmakni_cvorove_u_sloju(nx_graph, visible_graph, pos, node_gap=0.8):
    """Pomeranjem po x osi sprečava preklapanje čvorova u istom sloju."""

    nodes_by_layer = {}

    for node_id in nx_graph.nodes:
        layer = nx_graph.nodes[node_id]['layer']
        nodes_by_layer.setdefault(layer, []).append(node_id)

    for layer, node_ids in nodes_by_layer.items():

        if len(node_ids) <= 1:
            continue

        node_ids.sort(key=lambda node_id: pos[node_id][0])
        original_x = {node_id: pos[node_id][0] for node_id in node_ids}
        new_x = {}
        first_node = node_ids[0]
        new_x[first_node] = original_x[first_node]

        for index in range(1, len(node_ids)):
            previous_id = node_ids[index - 1]
            current_id = node_ids[index]
            previous_node = visible_graph.nodes[previous_id]
            current_node = visible_graph.nodes[current_id]
            previous_width, _ = dimenzije_cvora(previous_node)
            current_width, _ = dimenzije_cvora(current_node)
            minimum_distance = previous_width / 2 + node_gap + current_width / 2
            minimum_x = new_x[previous_id] + minimum_distance
            new_x[current_id] = max(original_x[current_id], minimum_x)

        original_center = (min(original_x.values()) + max(original_x.values())) / 2
        new_center = (min(new_x.values()) + max(new_x.values())) / 2
        center_shift = original_center - new_center

        for node_id in node_ids:
            _, y = pos[node_id]
            pos[node_id] = (new_x[node_id] + center_shift, y)

    return pos

def iscrtaj_okvire_grupa(ax, visible_graph, pos):

    if not visible_graph.groups:
        return None

    max_depth = max((dubina_grupe(group.id) for group in visible_graph.groups.values()))
    global_left = float('inf')
    global_right = float('-inf')
    global_bottom = float('inf')
    global_top = float('-inf')
    groups = sorted(visible_graph.groups.values(), key=lambda group: dubina_grupe(group.id))

    for group in groups:

        if group.id == 'ROOT':
            continue

        if not group.nodes:
            continue

        left = float('inf')
        right = float('-inf')
        bottom = float('inf')
        top = float('-inf')

        for node_id in group.nodes:
            node = visible_graph.nodes[node_id]
            x, y = pos[node_id]
            width, height = dimenzije_cvora(node)
            left = min(left, x - width / 2)
            right = max(right, x + width / 2)
            bottom = min(bottom, y - height / 2)
            top = max(top, y + height / 2)

        depth = dubina_grupe(group.id)
        nesting_extra = (max_depth - depth) * 0.45
        horizontal_padding = 0.6 + nesting_extra
        bottom_padding = 0.6 + nesting_extra
        top_padding = 1.0 + nesting_extra
        left -= horizontal_padding
        right += horizontal_padding
        bottom -= bottom_padding
        top += top_padding
        outline = FancyBboxPatch((left, bottom), right - left, top - bottom, boxstyle='round,pad=0.10', fill=False, linestyle='--', linewidth=1.2, edgecolor=GROUP_COLOR, zorder=0)
        ax.add_patch(outline)
        label = formatiraj_oznaku_grupe(group)
        ax.text(left + 0.25, top - 0.25, label, ha='left', va='top', fontsize=8, zorder=4, bbox={'facecolor': ax.get_facecolor(), 'edgecolor': 'none', 'pad': 1.5})
        global_left = min(global_left, left)
        global_right = max(global_right, right)
        global_bottom = min(global_bottom, bottom)
        global_top = max(global_top, top)

    return (global_left, global_right, global_bottom, global_top)

def granice_prikaza(visible_graph, pos, group_bounds=None):

    left = float('inf')
    right = float('-inf')
    bottom = float('inf')
    top = float('-inf')
    max_node_width = 0

    for node_id, node in visible_graph.nodes.items():
        x, y = pos[node_id]
        width, height = dimenzije_cvora(node)
        max_node_width = max(max_node_width, width)
        left = min(left, x - width / 2)
        right = max(right, x + width / 2)
        bottom = min(bottom, y - height / 2)
        top = max(top, y + height / 2)

    if group_bounds is not None:
        group_left, group_right, group_bottom, group_top = group_bounds
        left = min(left, group_left)
        right = max(right, group_right)
        bottom = min(bottom, group_bottom)
        top = max(top, group_top)

    content_width = right - left
    minimum_view_width = max_node_width * 4

    if content_width < minimum_view_width:
        center_x = (left + right) / 2
        half_width = minimum_view_width / 2
        left = center_x - half_width
        right = center_x + half_width

    horizontal_padding = max_node_width * 0.5
    vertical_padding = 1.0
    left -= horizontal_padding
    right += horizontal_padding
    bottom -= vertical_padding
    top += vertical_padding

    return (left, right, bottom, top)

def iscrtaj_graf(ax, visible_graph):

    ax.clear()
    nx_graph = napravi_nx_graf(visible_graph)
    pos = izracunaj_raspored(nx_graph, visible_graph)
    group_bounds = iscrtaj_okvire_grupa(ax, visible_graph, pos)
    iscrtaj_grane(ax, visible_graph, pos, nx_graph)
    node_patches = iscrtaj_cvorove(ax, visible_graph, pos)

    left, right, bottom, top = granice_prikaza(visible_graph, pos, group_bounds)

    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
    ax.set_axis_off()

    iscrtaj_legendu(ax, visible_graph)

    return node_patches

def iscrtaj_grane(ax, visible_graph, pos, nx_graph):
    """Crta obične, merge i skip veze između vidljivih čvorova."""

    skip_lanes = rasporedi_skip_grane(nx_graph, visible_graph, pos)

    for edge in visible_graph.edges:
        source_node = visible_graph.nodes[edge.source]
        target_node = visible_graph.nodes[edge.target]
        source_x, source_y = pos[edge.source]
        target_x, target_y = pos[edge.target]
        source_width, source_height = dimenzije_cvora(source_node)
        target_width, target_height = dimenzije_cvora(target_node)
        source_out_degree = nx_graph.out_degree(edge.source)
        target_in_degree = nx_graph.in_degree(edge.target)
        is_merge_edge = target_in_degree > 1
        main_predecessor = None

        if is_merge_edge:
            main_predecessor = glavni_prethodnik(nx_graph, edge.target)

        is_skip_merge_edge = is_merge_edge and edge.source != main_predecessor

        if is_skip_merge_edge:
            start_x = source_x
            start_y = source_y - source_height / 2
            end_y = target_y

            if abs(source_x - target_x) < 1e-09:
                edge_key = (edge.source, edge.target)
                lane_index = skip_lanes.get(edge_key, 0)
                base_lane_x = x_za_skip_granu(visible_graph, pos, source_y, target_y, source_x)
                LANE_GAP = 0.8
                if base_lane_x > source_x:
                    lane_x = base_lane_x + lane_index * LANE_GAP
                else:
                    lane_x = base_lane_x - lane_index * LANE_GAP
                branch_y = start_y - 0.4
                if lane_x < target_x:
                    end_x = target_x - target_width / 2
                else:
                    end_x = target_x + target_width / 2
                vertices = [(start_x, start_y), (start_x, branch_y), (lane_x, branch_y), (lane_x, end_y), (end_x, end_y)]
            else:
                if source_x < target_x:
                    end_x = target_x - target_width / 2
                else:
                    end_x = target_x + target_width / 2
                vertices = [(start_x, start_y), (start_x, end_y), (end_x, end_y)]

            codes = [Path.MOVETO] + [Path.LINETO] * (len(vertices) - 1)
            path = Path(vertices, codes)
            arrow = FancyArrowPatch(path=path, arrowstyle='-|>', mutation_scale=14, linewidth=1.3, color=EDGE_COLOR, shrinkA=0, shrinkB=0, zorder=1)
            ax.add_patch(arrow)

        elif abs(source_x - target_x) < 1e-09:
            start = (source_x, source_y - source_height / 2)
            end = (target_x, target_y + target_height / 2)
            arrow = FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=14, linewidth=1.3, color=EDGE_COLOR, shrinkA=0, shrinkB=0, zorder=1)
            ax.add_patch(arrow)

        elif is_merge_edge:
            start_x = source_x
            start_y = source_y - source_height / 2

            if source_x < target_x:
                end_x = target_x - target_width / 2
            else:
                end_x = target_x + target_width / 2

            end_y = target_y
            vertices = [(start_x, start_y), (start_x, end_y), (end_x, end_y)]
            codes = [Path.MOVETO, Path.LINETO, Path.LINETO]
            path = Path(vertices, codes)
            arrow = FancyArrowPatch(path=path, arrowstyle='-|>', mutation_scale=14, linewidth=1.3, color=EDGE_COLOR, shrinkA=0, shrinkB=0, zorder=1)
            ax.add_patch(arrow)

        elif source_out_degree > 1:
            start_x = source_x
            start_y = source_y - source_height / 2
            end_x = target_x
            end_y = target_y + target_height / 2
            branch_y = start_y - 0.4
            vertices = [(start_x, start_y), (start_x, branch_y), (end_x, branch_y), (end_x, end_y)]
            codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO]
            path = Path(vertices, codes)
            arrow = FancyArrowPatch(path=path, arrowstyle='-|>', mutation_scale=14, linewidth=1.3, color=EDGE_COLOR, shrinkA=0, shrinkB=0, zorder=1)
            ax.add_patch(arrow)

        else:
            if target_x > source_x:
                start_x = source_x + source_width / 2
                start_y = source_y
                end_x = target_x - target_width / 2
                end_y = target_y
            else:
                start_x = source_x - source_width / 2
                start_y = source_y
                end_x = target_x + target_width / 2
                end_y = target_y

            mid_x = (start_x + end_x) / 2
            vertices = [(start_x, start_y), (mid_x, start_y), (mid_x, end_y), (end_x, end_y)]
            codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO]
            path = Path(vertices, codes)
            arrow = FancyArrowPatch(path=path, arrowstyle='-|>', mutation_scale=14, linewidth=1.3, color=EDGE_COLOR, shrinkA=0, shrinkB=0, zorder=1)
            ax.add_patch(arrow)

def prikazi_graf(visible_graph):

    fig, ax = plt.subplots(figsize=(10, 12))
    iscrtaj_graf(ax, visible_graph)
    plt.tight_layout()
    plt.show()

    return (fig, ax)
