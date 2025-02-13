import os
from typing import Dict, List
 
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

    def __init__(self, modules: Dict[str, nn.Module]):
        self.connections = {}
        if not isinstance(modules, dict):
            raise ValueError("The modules must be a dictionary of modules")
        self.modules: Dict[str, nn.Module] = modules

        # Save the initial grad values
        self.initial_grads = {}
        self.last_module = None
        self.module_stats = {}
        self.module_names: Dict[nn.Module, str] = {}
        self._last_module = None

        self.logger = setup_logger(__class__.__name__) 

        self._register_hooks()

    def _register_hooks(self):
        for k, indep_module in self.modules.items():
            for name, module in indep_module.named_modules():
                if len(list(module.children())) == 0 and hasattr(module, "weight"):  # leaf modules only
                    self.module_names[module] = k + "." + name
                    self.connections[module] = set()
                    self.logger.debug(f"Wrote set for module {module._get_name()} with object hash {id(module)}")
                    module.register_forward_hook(self._forward_hook)
                    module.register_backward_hook(self._backward_hook)


    def _forward_hook(self, module: nn.Module, input, output):
        """
        Tracks/Calculates the forward pass connections
        """

        if self._last_module is not None:
            last_name = self.module_names[self._last_module]
            current_name = self.module_names[module]
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
        

    def dump_visual_dag(self, destination_path: str, figsize=(40, 200),  cmap='RdYlBu'):
        """
        Visualize the model DAG with gradient information
        """
        G = nx.DiGraph()

        # Ensure destiantion exists
        destination_dir = os.path.dirname(destination_path)
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)

        self.logger.debug(f"Dropping {len(self.module_stats)} modules")
        # Create nodes (same as before)
        for module in self.module_stats:
            name = self.module_names[module]
            stats = np.array(self.module_stats[module])
            mean_grad = np.mean(stats[:, 0])
            var_grad = np.mean(stats[:, 1])

            norm = mcolors.Normalize(vmin=-1, vmax=1)
            self.logger.debug(f"The values of mean_grad and var_grad are {mean_grad} and {var_grad}")
            color = plt.cm.get_cmap(cmap)(norm(mean_grad))
            self.logger.debug(f"Color looks like {color}")
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
        plt.tight_layout()
        # pos = nx.spring_layout(G)
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot', args='-Grankdir=LR') # 還不錯

        nx.draw(G, pos,
                node_color=[G.nodes[node]['color'] for node in G.nodes()],
                node_size=[G.nodes[node]['size'] for node in G.nodes()],
                with_labels=True,
                font_size=8,
                font_weight='bold')
        plt.title('Model DAG with Gradient Information')
        plt.savefig(destination_path)
        plt.close()
