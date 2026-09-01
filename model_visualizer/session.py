"""Jedna interaktivna sesija za složene modele kao što je SAM2."""
from dataclasses import dataclass, field
from typing import Callable
import torch.nn as nn
from model_visualizer.analyzer import analyze_model
from model_visualizer.graph import GraphEdge, VisibleGraph, VisibleNode
from model_visualizer.runtime import proveri_ulaze_modela
from model_visualizer.view import napravi_vidljivi_graf
from model_visualizer.renderer import iscrtaj_graf
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

@dataclass
class ComponentSpec:
    """Podešavanja za analizu jedne komponente složenog modela."""

    example_inputs: object = None
    example_kwargs: dict | None = None
    wrapper: Callable[[nn.Module], nn.Module] | nn.Module | None = None
    initial_expanded_modules: set[str] = field(default_factory=lambda: {'ROOT'})
    validate_inputs: bool = True

@dataclass
class OverviewNodeSpec:
    """Opis jednog čvora u high-level pregledu složenog modela."""

    label: str
    component_name: str | None = None
    subtype: str | None = None

@dataclass
class OverviewSpec:
    """Opis high-level grafa: čvorovi, veze i opcioni naslov."""

    nodes: dict[str, OverviewNodeSpec]
    edges: list[tuple[str, str]]
    title: str | None = None

def napravi_pregled_komponenti(model, overview_spec=None):
    """Pravi podrazumevani containment pregled ili korisnički zadat high-level dataflow."""

    if not isinstance(model, nn.Module):
        raise TypeError('Očekivan je PyTorch model tipa torch.nn.Module.')

    components = dict(model.named_children())

    if overview_spec is None:

        if not components:
            raise ValueError('Model nema direktne child module-e za composite prikaz.')

        root_id = 'ROOT'
        nodes = {root_id: VisibleNode(id=root_id, label=type(model).__name__, node_kind='module', subtype=type(model).__name__, expandable=False)}
        edges = []
        click_targets = {}

        for component_name, component in components.items():
            nodes[component_name] = VisibleNode(id=component_name, label=component_name, node_kind='module', subtype=type(component).__name__, expandable=True)
            click_targets[component_name] = component_name
            edges.append(GraphEdge(source=root_id, target=component_name))

        graph = VisibleGraph(nodes=nodes, edges=edges, groups={})
        title = f'{type(model).__name__} — strukturni pregled\nLevi klik: otvori komponentu | Strelice predstavljaju containment, ne dataflow'

        return (graph, click_targets, title)
    if not isinstance(overview_spec, OverviewSpec):
        raise TypeError('overview_spec mora biti OverviewSpec ili None.')

    if not overview_spec.nodes:
        raise ValueError('OverviewSpec mora sadržati bar jedan čvor.')

    nodes = {}
    click_targets = {}

    for node_id, node_spec in overview_spec.nodes.items():

        if not isinstance(node_spec, OverviewNodeSpec):
            raise TypeError(f"Overview node '{node_id}' mora biti OverviewNodeSpec.")

        if node_spec.component_name is not None:

            if node_spec.component_name not in components:
                raise ValueError(f"Overview node '{node_id}' referencira nepostojeći direktni child modul '{node_spec.component_name}'.")

            component = components[node_spec.component_name]
            subtype = node_spec.subtype or type(component).__name__
            nodes[node_id] = VisibleNode(id=node_id, label=node_spec.label, node_kind='module', subtype=subtype, expandable=True)
            click_targets[node_id] = node_spec.component_name
        else:
            nodes[node_id] = VisibleNode(id=node_id, label=node_spec.label, node_kind='operation', subtype=node_spec.subtype or 'external', expandable=False)

    edges = []

    for source, target in overview_spec.edges:
        if source not in nodes:
            raise ValueError(f"Overview edge koristi nepostojeći source '{source}'.")
        if target not in nodes:
            raise ValueError(f"Overview edge koristi nepostojeći target '{target}'.")
        edges.append(GraphEdge(source=source, target=target))

    graph = VisibleGraph(nodes=nodes, edges=edges, groups={})
    title = overview_spec.title or f'{type(model).__name__} — high-level dataflow'

    return (graph, click_targets, title)

class CompositeVisualizer:
    """Interaktivni prikaz složenog modela: overview i lazy analiza pojedinačnih komponenti."""

    def __init__(self, model, component_specs=None, *, overview_spec=None, validate_inputs=True, figsize=(12, 12)):

        if not isinstance(model, nn.Module):
            raise TypeError('Očekivan je PyTorch model tipa torch.nn.Module.')

        self.model = model
        self.validate_inputs = validate_inputs
        self.components = dict(model.named_children())

        if not self.components:
            raise ValueError('Model nema direktne child module-e. Za ovakav model koristite običan visualize(model).')

        if component_specs is None:
            component_specs = {}

        unknown_components = set(component_specs) - set(self.components)

        if unknown_components:
            unknown_text = ', '.join(sorted(unknown_components))
            raise ValueError(f'component_specs sadrži putanje koje nisu direktni child moduli modela: {unknown_text}')

        self.component_specs = dict(component_specs)

        for component_name, spec in self.component_specs.items():
            if not isinstance(spec, ComponentSpec):
                raise TypeError(f"Specifikacija za '{component_name}' mora biti ComponentSpec.")

        self.overview_graph, self.overview_click_targets, self.overview_title = napravi_pregled_komponenti(model, overview_spec=overview_spec)
        self.graph_cache = {}
        self.target_module_cache = {}
        self.expanded_cache = {}
        self.mode = 'overview'
        self.current_component = None
        self.current_graph = None
        self.expanded_modules = set()
        self.visible_graph = None
        self.node_patches = {}
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.fig.subplots_adjust(top=0.9, bottom=0.1)

        try:
            self.fig.canvas.manager.set_window_title('VizTool')
        except Exception:
            pass

        self.back_ax = self.fig.add_axes([0.02, 0.94, 0.11, 0.035])
        self.back_button = Button(self.back_ax, '← Overview', color='#F3F4F6', hovercolor='#E5E7EB')
        self.back_button.label.set_fontsize(9)
        self.back_button.on_clicked(self.klik_na_nazad)
        self.fig.canvas.mpl_connect('button_press_event', self.obradi_klik)
        self.fig.canvas.mpl_connect('key_press_event', self.obradi_taster)
        self.osvezi_pregled()

    def specifikacija_komponente(self, component_name):
        return self.component_specs.get(component_name, ComponentSpec())

    def napravi_ciljni_modul(self, component_name, spec):

        if component_name in self.target_module_cache:
            return self.target_module_cache[component_name]

        component = self.components[component_name]

        if spec.wrapper is None:
            target_module = component
        elif isinstance(spec.wrapper, nn.Module):
            target_module = spec.wrapper
        elif callable(spec.wrapper):
            target_module = spec.wrapper(component)
        else:
            raise TypeError(f"wrapper za '{component_name}' mora biti nn.Module, callable ili None.")

        if not isinstance(target_module, nn.Module):
            raise TypeError(f"wrapper za '{component_name}' nije vratio nn.Module.")

        target_module.train(component.training)
        self.target_module_cache[component_name] = target_module

        return target_module

    def analiziraj_komponentu(self, component_name):

        if component_name in self.graph_cache:
            return self.graph_cache[component_name]

        spec = self.specifikacija_komponente(component_name)
        target_module = self.napravi_ciljni_modul(component_name, spec)
        has_inputs = spec.example_inputs is not None or bool(spec.example_kwargs)
        should_validate = self.validate_inputs and spec.validate_inputs and has_inputs

        if should_validate:
            proveri_ulaze_modela(target_module, spec.example_inputs, spec.example_kwargs)

        graph = analyze_model(target_module, example_inputs=spec.example_inputs, example_kwargs=spec.example_kwargs)
        self.graph_cache[component_name] = graph

        return graph

    def osvezi_pregled(self, error_message=None):

        self.mode = 'overview'
        self.current_component = None
        self.current_graph = None
        self.back_ax.set_visible(False)
        self.visible_graph = self.overview_graph
        self.node_patches = iscrtaj_graf(self.ax, self.visible_graph)
        title = self.overview_title

        if error_message:
            title += '\nGreška pri analizi komponente: ' + error_message

        self.ax.set_title(title, fontsize=10)

        try:
            self.fig.canvas.manager.set_window_title(f'VizTool — {type(self.model).__name__}')
        except Exception:
            pass

        self.fig.canvas.draw_idle()

    def osvezi_komponentu(self):

        self.back_ax.set_visible(True)
        self.visible_graph = napravi_vidljivi_graf(self.current_graph, self.expanded_modules)
        self.node_patches = iscrtaj_graf(self.ax, self.visible_graph)
        self.ax.set_title(f'{self.current_component} — computation graph\nLevi klik: expand | Desni klik: collapse | Backspace/Esc: overview', fontsize=10)

        try:
            self.fig.canvas.manager.set_window_title(f'VizTool — {self.current_component}')
        except Exception:
            pass

        self.fig.canvas.draw_idle()

    def otvori_komponentu(self, component_name):

        if component_name not in self.components:
            return

        self.ax.set_title(f'Analiziram komponentu: {component_name} ...', fontsize=10)
        self.fig.canvas.draw_idle()

        try:
            self.fig.canvas.flush_events()
        except Exception:
            pass

        try:
            graph = self.analiziraj_komponentu(component_name)
        except Exception as error:
            message = f'{type(error).__name__}: {error}'
            print(f"\nNeuspešna analiza komponente '{component_name}':\n{message}\n")
            self.osvezi_pregled(error_message=message)
            return

        self.mode = 'component'
        self.current_component = component_name
        self.current_graph = graph

        if component_name in self.expanded_cache:
            self.expanded_modules = set(self.expanded_cache[component_name])
        else:
            spec = self.specifikacija_komponente(component_name)
            self.expanded_modules = set(spec.initial_expanded_modules)

        self.osvezi_komponentu()

    def nazad_na_pregled(self):

        if self.mode != 'component':
            return

        if self.current_component is not None:
            self.expanded_cache[self.current_component] = set(self.expanded_modules)

        self.osvezi_pregled()

    def klik_na_nazad(self, _event):
        self.nazad_na_pregled()

    def obradi_taster(self, event):

        if event.key in ('escape', 'backspace', 'b'):
            self.nazad_na_pregled()

    def nadji_kliknuti_cvor(self, event):

        if event.inaxes != self.ax:
            return None
        for node_id, patch in self.node_patches.items():
            contains, _ = patch.contains(event)
            if contains:
                return node_id

        return None

    def obradi_klik(self, event):

        node_id = self.nadji_kliknuti_cvor(event)

        if node_id is None:
            return

        if self.mode == 'overview':
            if event.button == 1:
                component_name = self.overview_click_targets.get(node_id)
                if component_name is not None:
                    self.otvori_komponentu(component_name)
            return

        if event.button == 1:
            node = self.visible_graph.nodes[node_id]
            if node.node_kind == 'module' and node.expandable:
                self.expanded_modules.add(node_id)
                self.osvezi_komponentu()
        elif event.button == 3:
            module_to_collapse = self.nadji_najblizi_otvoren_modul(node_id)
            if module_to_collapse is not None:
                self.zatvori_modul(module_to_collapse)
                self.osvezi_komponentu()

    def nadji_najblizi_otvoren_modul(self, node_id):

        graph = self.current_graph

        if node_id in graph.operations:
            current = graph.operations[node_id].scope
        elif node_id in graph.modules:
            current = graph.modules[node_id].parent
        else:
            return None

        while current is not None:
            if current in self.expanded_modules:
                return current
            current = graph.modules[current].parent

        return None

    def zatvori_modul(self, module_path):

        modules_to_remove = set()

        for expanded_module in self.expanded_modules:
            if self.je_isti_ili_potomak(expanded_module, module_path):
                modules_to_remove.add(expanded_module)

        self.expanded_modules.difference_update(modules_to_remove)

    def je_isti_ili_potomak(self, module_path, possible_ancestor):

        graph = self.current_graph
        current = module_path

        while current is not None:
            if current == possible_ancestor:
                return True
            current = graph.modules[current].parent

        return False

    def show(self):
        plt.show()
