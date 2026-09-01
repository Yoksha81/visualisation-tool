"""Pretvara kompletan ModelGraph u graf koji trenutno treba prikazati."""
from model_visualizer.graph import GraphEdge, VisibleGraph, VisibleNode, VisibleGroup

def nadji_vidljivog_predstavnika(operation, graph, expanded_modules):

    if operation.op in ('placeholder', 'output'):
        return operation.id

    chain = []
    current = operation.scope

    while current is not None:
        chain.append(current)
        current = graph.modules[current].parent

    chain.reverse()

    for module_path in chain:
        if module_path not in expanded_modules:
            return module_path

    return operation.id

def napravi_vidljive_grane(graph, expanded_modules):

    visible_edges = []
    seen_edges = set()

    for edge in graph.edges:
        source_operation = graph.operations[edge.source]
        target_operation = graph.operations[edge.target]
        visible_source = nadji_vidljivog_predstavnika(source_operation, graph, expanded_modules)
        visible_target = nadji_vidljivog_predstavnika(target_operation, graph, expanded_modules)
        if visible_source == visible_target:
            continue
        edge_key = (visible_source, visible_target)
        if edge_key not in seen_edges:
            seen_edges.add(edge_key)
            visible_edges.append(GraphEdge(source=visible_source, target=visible_target))

    return visible_edges

def nadji_vidljive_cvorove(graph, expanded_modules):

    visible_nodes = []

    for operation in graph.operations.values():
        visible_node = nadji_vidljivog_predstavnika(operation, graph, expanded_modules)
        if visible_node not in visible_nodes:
            visible_nodes.append(visible_node)

    return visible_nodes

def oznaka_operacije(operation, graph=None):

    if operation.op == 'placeholder':
        if graph is not None:
            placeholder_count = sum((1 for graph_operation in graph.operations.values() if graph_operation.op == 'placeholder'))
            if placeholder_count > 1:
                return str(operation.target)
        return 'Input'

    if operation.op == 'output':
        return 'Output'

    if operation.op == 'call_module':
        return operation.module_type

    if operation.op == 'call_function':

        target = operation.target

        if 'flatten' in target:
            return 'Flatten'

        if 'add' in target:
            return 'Add'
        return target

    if operation.op == 'call_method':
        return str(operation.target).capitalize()

    return str(operation.target)

def je_standardni_torch_modul(module):
    """
    True samo za klase definisane u torch.nn paketu.
    """

    source = module.module_source

    return source == 'torch.nn' or source.startswith('torch.nn.')

def je_export_wrapper(operation, graph):
    """Prepoznaje torch.export bookkeeping operacije koje nisu korisne za arhitektonski prikaz."""

    target = str(operation.target)

    if operation.op == 'get_attr' and target.startswith('submod_'):
        return True

    if target == 'wrap_with_set_grad_enabled':
        return True

    if 'getitem' in target.lower():
        for input_id in operation.inputs:
            input_operation = graph.operations.get(input_id)
            if input_operation is not None and str(input_operation.target) == 'wrap_with_set_grad_enabled':
                return True

    return False

def ima_smislene_operacije(module, graph):
    """
    Custom leaf modul je vredno otvarati samo ako bar jedna njegova
    direktna operacija predstavlja stvarni computational detalj,
    a ne samo torch.export wrapper/bookkeeping lanac.
    """

    for operation_id in module.operations:
        operation = graph.operations.get(operation_id)
        if operation is None:
            continue
        if not je_export_wrapper(operation, graph):
            return True

    return False

def modul_moze_da_se_otvori(module, graph):
    """Modul je otvarljiv ako ima podmodule ili smislen sadržaj koji nije samo export bookkeeping."""

    if module.children:
        return True

    if not module.operations:
        return False

    if je_standardni_torch_modul(module):
        return False

    return ima_smislene_operacije(module, graph)

def napravi_vidljivi_cvor(node_id, graph):

    if node_id in graph.operations:
        operation = graph.operations[node_id]
        visible_id = operation.id
        node_kind = 'operation'
        subtype = operation.module_type if operation.op == 'call_module' else operation.op
        expandable = False
        label = oznaka_operacije(operation, graph)
        return VisibleNode(id=visible_id, label=label, node_kind=node_kind, subtype=subtype, expandable=expandable)
    elif node_id in graph.modules:
        module = graph.modules[node_id]
        visible_id = module.path
        node_kind = 'module'
        subtype = module.module_type
        expandable = modul_moze_da_se_otvori(module, graph)
        label = module.path.rsplit('.', 1)[-1]
        if module.path == 'ROOT':
            label = module.module_type
        else:
            label = module.path.rsplit('.', 1)[-1]
        return VisibleNode(id=visible_id, label=label, node_kind=node_kind, subtype=subtype, expandable=expandable)

def nadji_usmereni_ciklus(node_ids, edges):
    """Vraća jedan ciklus u vidljivom grafu ili None ako je graf DAG."""

    adjacency = {node_id: [] for node_id in node_ids}

    for edge in edges:
        if edge.source in adjacency and edge.target in adjacency:
            adjacency[edge.source].append(edge.target)

    state = {node_id: 0 for node_id in node_ids}
    stack = []
    stack_index = {}

    def dfs(node_id):

        state[node_id] = 1
        stack_index[node_id] = len(stack)
        stack.append(node_id)

        for successor in adjacency[node_id]:
            if state[successor] == 0:
                cycle = dfs(successor)
                if cycle is not None:
                    return cycle
            elif state[successor] == 1:
                start_index = stack_index[successor]
                return stack[start_index:].copy()

        stack.pop()
        stack_index.pop(node_id, None)
        state[node_id] = 2
        return None

    for node_id in node_ids:
        if state[node_id] != 0:
            continue
        cycle = dfs(node_id)
        if cycle is not None:
            return cycle

    return None

def moze_prinudno_otvaranje(module):
    """
    Širi kriterijum koji se koristi samo za uklanjanje veštačkih
    ciklusa nastalih collapsing-om.

    Standardni torch.nn leaf moduli se normalno ne nude korisniku
    za ručni expand. Međutim, ako se ISTI modul pozove više puta u
    forward()-u (npr. isti GELU između više Linear slojeva u MLP-u),
    collapsing svih tih poziva u jedan vidljivi čvor može napraviti:

        GELU -> Linear -> GELU

    odnosno lažni ciklus.

    U toj situaciji je bezbedno automatski otvoriti taj leaf do
    njegovih pojedinačnih operation occurrence-a, jer originalni
    computation graph ostaje nepromenjen i DAG.
    """
    return bool(module.children or module.operations)

def obezbedi_dag_prikaz(graph, expanded_modules):
    """Po potrebi automatski otvara module čije bi sažimanje napravilo veštački ciklus."""

    effective_expanded = set(expanded_modules)
    max_iterations = len(graph.modules) + 1

    for _ in range(max_iterations):
        visible_node_ids = nadji_vidljive_cvorove(graph, effective_expanded)
        visible_edges = napravi_vidljive_grane(graph, effective_expanded)
        cycle = nadji_usmereni_ciklus(visible_node_ids, visible_edges)

        if cycle is None:
            return (effective_expanded, visible_node_ids, visible_edges)

        candidates = []

        for node_id in cycle:
            if node_id not in graph.modules:
                continue
            if node_id in effective_expanded:
                continue
            module = graph.modules[node_id]
            if moze_prinudno_otvaranje(module):
                candidates.append(node_id)

        if not candidates:
            raise ValueError(f'Visible graph sadrži ciklus koji nije moguće ukloniti automatskim otvaranjem collapsed modula. Cycle nodes: {cycle}')

        module_to_expand = max(candidates, key=lambda module_id: (len(graph.modules[module_id].operations), 0 if module_id == 'ROOT' else module_id.count('.') + 1))
        effective_expanded.add(module_to_expand)

    raise RuntimeError('Nije moguće napraviti cycle-safe VisibleGraph.')

def napravi_vidljivi_graf(graph, expanded_modules):
    """Pravi graf koji odgovara trenutnom stanju expand/collapse prikaza."""

    effective_expanded, visible_node_ids, visible_edges = obezbedi_dag_prikaz(graph, expanded_modules)
    visible_nodes = {}

    for node_id in visible_node_ids:
        visible_node = napravi_vidljivi_cvor(node_id, graph)
        visible_nodes[node_id] = visible_node

    groups = {}

    for module_path in expanded_modules:
        if module_path not in graph.modules:
            continue
        module = graph.modules[module_path]
        group_nodes = []
        for node_id in visible_nodes:
            if cvor_pripada_modulu(node_id, module_path, graph):
                group_nodes.append(node_id)
        if group_nodes:
            if module_path == 'ROOT':
                label = module.module_type
            else:
                label = module_path.rsplit('.', 1)[-1]
            groups[module_path] = VisibleGroup(id=module_path, label=label, module_type=module.module_type, nodes=group_nodes)

    return VisibleGraph(nodes=visible_nodes, edges=visible_edges, groups=groups)

def cvor_pripada_modulu(node_id, module_path, graph):

    if node_id in graph.operations:
        operation = graph.operations[node_id]
        if operation.op in ('placeholder', 'output'):
            return False
        current = operation.scope
    elif node_id in graph.modules:
        current = node_id
    else:
        return False
    while current is not None:
        if current == module_path:
            return True
        current = graph.modules[current].parent

    return False
