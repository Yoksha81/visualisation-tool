"""Expand/collapse interakcija za običan ModelGraph."""
import matplotlib.pyplot as plt
from model_visualizer.view import napravi_vidljivi_graf
from model_visualizer.renderer import iscrtaj_graf

class InteractiveVisualizer:

    def __init__(self, graph, expanded_modules=None):

        self.graph = graph

        if expanded_modules is None:
            self.expanded_modules = set()
        else:
            self.expanded_modules = set(expanded_modules)

        self.visible_graph = None
        self.node_patches = {}
        self.fig, self.ax = plt.subplots(figsize=(10, 12))
        self.fig.subplots_adjust(top=0.9, bottom=0.1)

        try:
            self.fig.canvas.manager.set_window_title('VizTool')
        except Exception:
            pass

        self.fig.canvas.mpl_connect('button_press_event', self.obradi_klik)
        self.osvezi_prikaz()

    def osvezi_prikaz(self):

        self.visible_graph = napravi_vidljivi_graf(self.graph, self.expanded_modules)
        self.node_patches = iscrtaj_graf(self.ax, self.visible_graph)
        root_type = self.graph.modules['ROOT'].module_type if 'ROOT' in self.graph.modules else 'PyTorch model'
        self.ax.set_title(f'{root_type} — computation graph\nLevi klik: expand | Desni klik: collapse', fontsize=10)
        self.fig.canvas.draw_idle()

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

        if event.button == 1:
            node = self.visible_graph.nodes[node_id]

            if node.node_kind == 'module' and node.expandable:
                self.expanded_modules.add(node_id)
                self.osvezi_prikaz()

        elif event.button == 3:
            module_to_collapse = self.nadji_najblizi_otvoren_modul(node_id)

            if module_to_collapse is not None:
                self.zatvori_modul(module_to_collapse)
                self.osvezi_prikaz()

    def nadji_najblizi_otvoren_modul(self, node_id):

        if node_id in self.graph.operations:
            current = self.graph.operations[node_id].scope
        elif node_id in self.graph.modules:
            current = self.graph.modules[node_id].parent
        else:
            return None

        while current is not None:
            if current in self.expanded_modules:
                return current
            current = self.graph.modules[current].parent

        return None

    def zatvori_modul(self, module_path):

        modules_to_remove = set()

        for expanded_module in self.expanded_modules:
            if self.je_isti_ili_potomak(expanded_module, module_path):
                modules_to_remove.add(expanded_module)
        self.expanded_modules.difference_update(modules_to_remove)

    def je_isti_ili_potomak(self, module_path, possible_ancestor):

        current = module_path

        while current is not None:
            if current == possible_ancestor:
                return True
            current = self.graph.modules[current].parent

        return False

    def show(self):

        try:
            self.fig.tight_layout(rect=(0.02, 0.07, 0.98, 0.94))
        except Exception:
            pass

        plt.show()
