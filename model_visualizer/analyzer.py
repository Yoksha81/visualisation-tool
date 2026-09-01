"""Analiza PyTorch modela pomoću torch.fx i torch.export backenda."""
from model_visualizer.graph import OperationNode, GraphEdge, ModuleBlock, ModelGraph
from model_visualizer.runtime import normalizuj_ulaze
import torch
import torch.nn as nn
from torch.fx import symbolic_trace

def nadji_opseg_cvora(node):

    scope = 'ROOT'

    if node.op == 'call_module':
        target = str(node.target)
        if '.' in target:
            scope = target.rsplit('.', 1)[0]
    elif node.op in ('call_function', 'call_method'):
        stack = node.meta.get('nn_module_stack')

        if stack:
            stack_scope = list(stack.values())[-1][0]
            scope = stack_scope if stack_scope else 'ROOT'

    return scope

def napravi_blokove_modula(model):

    modules = {}

    for name, module in model.named_modules():

        if name == '':
            path = 'ROOT'
            parent = None
        else:
            path = name

            if '.' in name:
                parent = name.rsplit('.', 1)[0]
            else:
                parent = 'ROOT'

        modules[path] = ModuleBlock(path=path, module_type=type(module).__name__, parent=parent, module_source=type(module).__module__)

    for path, module_block in modules.items():

        if module_block.parent is not None:
            modules[module_block.parent].children.append(path)

    return modules

def napravi_fx_graf(model, traced_model):

    operations = {}
    modules = napravi_blokove_modula(model)
    edges = []

    for node in traced_model.graph.nodes:

        node_id = node.name
        op = node.op
        target = str(node.target)
        scope = nadji_opseg_cvora(node)
        inputs = [input_node.name for input_node in node.all_input_nodes]

        if node.op == 'call_module':
            submodule = traced_model.get_submodule(str(node.target))
            module_type = type(submodule).__name__
        else:
            module_type = None

        operation = OperationNode(id=node_id, op=op, target=target, scope=scope, inputs=inputs, module_type=module_type)
        operations[node_id] = operation

        if scope in modules:
            modules[scope].operations.append(node_id)

        for input_id in inputs:
            edges.append(GraphEdge(source=input_id, target=node_id))

    return ModelGraph(operations=operations, modules=modules, edges=edges)

def nadji_korisnicke_ulaze_exporta(exported_program):

    user_inputs = set()

    for input_spec in exported_program.graph_signature.input_specs:

        if input_spec.kind.name != 'USER_INPUT':
            continue

        argument = input_spec.arg

        if hasattr(argument, 'name'):
            user_inputs.add(argument.name)

    return user_inputs

def mapiraj_export_parametre_na_module(exported_program, modules):
    """Povezuje parametre i buffere iz torch.export grafa sa modulima kojima pripadaju."""

    lifted_input_modules = {}

    for input_spec in exported_program.graph_signature.input_specs:

        kind_name = input_spec.kind.name

        if kind_name not in ('PARAMETER', 'BUFFER'):
            continue

        argument = input_spec.arg
        target = input_spec.target

        if not hasattr(argument, 'name') or not isinstance(target, str):
            continue

        candidate = target

        while candidate:

            if candidate in modules:
                lifted_input_modules[argument.name] = candidate
                break
            if '.' not in candidate:
                break

            candidate = candidate.rsplit('.', 1)[0]

    return lifted_input_modules

def propagiraj_poreklo_modula(graph_nodes, lifted_input_modules):
    """Propagira informaciju o poreklu parametara kroz export graf. Koristi se samo za određivanje scope-a."""

    provenance = {}

    for node in graph_nodes:

        if node.name in lifted_input_modules:
            provenance[node.name] = {lifted_input_modules[node.name]}
            continue

        node_provenance = set()

        for input_node in node.all_input_nodes:
            node_provenance.update(provenance.get(input_node.name, set()))

        provenance[node.name] = node_provenance

    return provenance

def nadji_opseg_export_cvora(node, modules, module_provenance=None):
    """Određuje kom modulu pripada export čvor. Ako metadata nije dovoljna, koristi poreklo parametara."""

    stack = node.meta.get('nn_module_stack')

    if stack:
        for stack_entry in reversed(list(stack.values())):

            module_path = stack_entry[0]

            if not module_path:
                module_path = 'ROOT'

            if module_path in modules and module_path != 'ROOT':
                return module_path

    if module_provenance is not None:

        candidates = {module_path for module_path in module_provenance.get(node.name, set()) if module_path in modules and module_path != 'ROOT'}

        if len(candidates) == 1:
            return next(iter(candidates))

    return 'ROOT'

def napravi_export_graf(model, exported_program):

    operations = {}
    modules = napravi_blokove_modula(model)
    edges = []

    user_inputs = nadji_korisnicke_ulaze_exporta(exported_program)
    graph_nodes = list(exported_program.graph.nodes)
    lifted_input_modules = mapiraj_export_parametre_na_module(exported_program, modules)
    module_provenance = propagiraj_poreklo_modula(graph_nodes, lifted_input_modules)

    for node in graph_nodes:

        if node.op == 'placeholder':

            if node.name not in user_inputs:
                continue
            scope = 'ROOT'

        elif node.op == 'output':
            scope = 'ROOT'

        else:
            scope = nadji_opseg_export_cvora(node, modules, module_provenance=module_provenance)

        operation = OperationNode(id=node.name, op=node.op, target=str(node.target), scope=scope, inputs=[], module_type=None)
        operations[node.name] = operation

        if scope in modules:
            modules[scope].operations.append(node.name)

    for node in graph_nodes:

        if node.name not in operations:
            continue

        valid_inputs = []

        for input_node in node.all_input_nodes:
            if input_node.name not in operations:
                continue

            valid_inputs.append(input_node.name)
            edges.append(GraphEdge(source=input_node.name, target=node.name))

        operations[node.name].inputs = valid_inputs

    return ModelGraph(operations=operations, modules=modules, edges=edges)

NON_ARCHITECTURAL_OPERATION_NAMES = {'to', '_to_copy', '_assert', '_assert_async', '_assert_scalar', '_assert_tensor_metadata'}

def osnovno_ime_operacije(operation):

    target = str(operation.target)

    if target.startswith('aten.'):
        parts = target.split('.')
        if len(parts) >= 2:
            return parts[1]

    if operation.op == 'call_method':
        return target

    if target.startswith('<function '):
        function_part = target[len('<function '):].split(' at ', 1)[0]
        return function_part.rsplit('.', 1)[-1]

    if target.startswith('<built-in function '):
        return target[len('<built-in function '):].rstrip('>')

    return target

def treba_sakriti_operaciju(operation):

    if operation.op in ('placeholder', 'output'):
        return False

    return osnovno_ime_operacije(operation) in NON_ARCHITECTURAL_OPERATION_NAMES

def pojednostavi_graf(graph):
    """Uklanja pomoćne export operacije, premošćava njihove veze i odbacuje mrtve grane koje ne vode do izlaza."""

    hidden_ids = {operation_id for operation_id, operation in graph.operations.items() if treba_sakriti_operaciju(operation)}

    if not hidden_ids:
        return graph

    predecessors = {operation_id: [] for operation_id in graph.operations}

    for edge in graph.edges:
        if edge.target in predecessors:
            predecessors[edge.target].append(edge.source)

    resolved_cache = {}

    def resolve_visible_predecessors(operation_id, visiting=None):

        if operation_id not in hidden_ids:
            return [operation_id]

        if operation_id in resolved_cache:
            return resolved_cache[operation_id]

        if visiting is None:
            visiting = set()

        if operation_id in visiting:
            return []

        visiting = set(visiting)
        visiting.add(operation_id)
        resolved = []

        for predecessor_id in predecessors.get(operation_id, []):
            for source_id in resolve_visible_predecessors(predecessor_id, visiting):
                if source_id not in resolved:
                    resolved.append(source_id)

        resolved_cache[operation_id] = resolved
        return resolved

    retained_operations = {operation_id: operation for operation_id, operation in graph.operations.items() if operation_id not in hidden_ids}
    new_edges = []
    seen_edges = set()

    for operation_id, operation in retained_operations.items():

        new_inputs = []

        for input_id in operation.inputs:
            for resolved_source in resolve_visible_predecessors(input_id):
                if resolved_source not in retained_operations:
                    continue
                if resolved_source == operation_id:
                    continue
                if resolved_source not in new_inputs:
                    new_inputs.append(resolved_source)
                edge_key = (resolved_source, operation_id)
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    new_edges.append(GraphEdge(source=resolved_source, target=operation_id))

        operation.inputs = new_inputs

    output_ids = [operation_id for operation_id, operation in retained_operations.items() if operation.op == 'output']

    if output_ids:
        reverse_adjacency = {operation_id: [] for operation_id in retained_operations}

        for edge in new_edges:
            reverse_adjacency[edge.target].append(edge.source)

        live_ids = set(output_ids)
        stack = list(output_ids)

        while stack:
            current = stack.pop()
            for predecessor_id in reverse_adjacency.get(current, []):
                if predecessor_id not in live_ids:
                    live_ids.add(predecessor_id)
                    stack.append(predecessor_id)

        retained_operations = {operation_id: operation for operation_id, operation in retained_operations.items() if operation_id in live_ids}
        new_edges = [edge for edge in new_edges if edge.source in retained_operations and edge.target in retained_operations]
        incoming = {operation_id: [] for operation_id in retained_operations}

        for edge in new_edges:
            incoming[edge.target].append(edge.source)

        for operation_id, operation in retained_operations.items():
            operation.inputs = incoming[operation_id]

    for module in graph.modules.values():
        module.operations = [operation_id for operation_id in module.operations if operation_id in retained_operations]

    return ModelGraph(operations=retained_operations, modules=graph.modules, edges=new_edges)

def analyze_model(model, example_inputs=None, example_kwargs=None):
    """Analizira PyTorch model. Prvo pokušava torch.fx, a ako to ne uspe koristi torch.export kada su dati primeri ulaza."""

    if not isinstance(model, nn.Module):
        raise TypeError('Očekivan je PyTorch model tipa torch.nn.Module.')

    try:
        traced_model = symbolic_trace(model)
    except Exception as symbolic_error:
        if example_inputs is None and (not example_kwargs):
            raise RuntimeError('Model nije moguće analizirati pomoću torch.fx.symbolic_trace(). Za ovaj model prosledite reprezentativne example_inputs i/ili example_kwargs kako bi alat mogao da pokuša torch.export.export().') from symbolic_error
        export_args, export_kwargs = normalizuj_ulaze(example_inputs, example_kwargs)

        try:
            exported_program = torch.export.export(model, export_args, kwargs=export_kwargs)
        except Exception as export_error:
            raise RuntimeError('Model nije moguće analizirati ni pomoću torch.fx.symbolic_trace() ni pomoću torch.export.export(). Proverite da li su example input-i kompatibilni sa modelom.') from export_error

        graph = napravi_export_graf(model, exported_program)

        return pojednostavi_graf(graph)

    graph = napravi_fx_graf(model, traced_model)

    return pojednostavi_graf(graph)
