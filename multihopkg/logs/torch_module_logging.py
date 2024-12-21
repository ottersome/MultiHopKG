from typing import Dict, DefaultDict, Set
 
from torch import nn
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from multihopkg.logging import setup_logger

########################################
# Graph Changes Logger
########################################

class ModuleSupervisor():

    def __init__(self, module: nn.Module):
        self.module = module
        self.connections = {}

        # Save the initial grad values
        self.initial_grads = {}
        self.last_module = None
        self.module_stats = {}
        self.module_names: Dict[nn.Module, str] = {}
        self._last_module = None

        self.logger = setup_logger(__class__.__name__) 

        self._register_hooks()

    def _register_hooks(self):
        for name, module in self.module.named_modules():
            if len(list(module.children())) == 0 and hasattr(module, "weight"):  # leaf modules only
                self.module_names[module] = name
                self.connections[module] = set()
                print(f"Wrote set for module {module._get_name()} with object hash {id(module)}")
                module.register_forward_hook(self._forward_hook)
                module.register_backward_hook(self._backward_hook)


    def _forward_hook(self, module: nn.Module, input, output):
        """
        Tracks/Calculates the forward pass connections
        """
        def get_module_name(m):
            # Find the full path of the module in the model
            for name, mod in self.module.named_modules():  # Assuming self.model is your model
                if mod is m:
                    return name
            return m._get_name()  # Fallback to class name if not found

        if self._last_module is not None:
            last_name = get_module_name(self._last_module)
            current_name = get_module_name(module)
            self.logger.debug(f"For module {last_name} we are adding {current_name} to the connections set")
            self.connections[self._last_module].add(module)
        self._last_module = module

    def _backward_hook(self, module, grad_input, grad_output):
        """
        Stores the gradient statistics for each module
        """

        if module not in self.module_stats:
            self.module_stats[module] = []

        # Calculate statistics for the gradient
        grad = grad_output[0]
        if grad is not None:
            mean = grad.mean().item()
            var = grad.var().item()
            self.module_stats[module].append((mean, var))
        

    def dump_visual_dag(self, destination_path: str, figsize=(10, 10),  cmap='RdYlBu'):
        """
        Visualize the model DAG with gradient information
        """
        G = nx.DiGraph()

        # Create nodes (same as before)
        for module in self.module_stats:
            name = self.module_names[module]
            stats = np.array(self.module_stats[module])
            mean_grad = np.mean(stats[:, 0])
            var_grad = np.mean(stats[:, 1])

            norm = mcolors.Normalize(vmin=-1, vmax=1)
            color = plt.cm.get_cmap(cmap)(norm(mean_grad))
            size = 1000 * (1 + np.log(1 + abs(var_grad)))

            G.add_node(name, color=color, size=size)

        # Add edges based on module structure
        for module in self.module_stats:
            name = self.module_names[module]
            for child in module.children():
                if child in self.module_names:
                    child_name = self.module_names[child]
                    G.add_edge(name, child_name)

        plt.figure(figsize=figsize)
        pos = nx.spring_layout(G)

        nx.draw(G, pos,
                node_color=[G.nodes[node]['color'] for node in G.nodes()],
                node_size=[G.nodes[node]['size'] for node in G.nodes()],
                with_labels=True,
                font_size=8,
                font_weight='bold')
        plt.title('Model DAG with Gradient Information')
        plt.savefig(destination_path)
        plt.close()
