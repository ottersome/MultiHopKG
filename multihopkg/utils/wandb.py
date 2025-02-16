import torch
from torch.jit import export_opnames
import wandb
import torch.nn as nn
from typing import Dict, List, Optional,Set, DefaultDict, Tuple
import numpy as np

def return_most_global(name1: str, name2: str) -> str:
    if len(name1) > len(name2):
        larger = name1
        smaller = name2
    else:
        smaller = name1
        larger = name2
    
    names = larger.split(".")
    for i in range(len(names)):
        name = ".".join(names[:i+1])
        print(f"-Checking {name} vs {smaller}" )
        if name == smaller:
            print(f"--returning {name}")
            return name
    return ""
    
def fix_namespace_duplicates(names_list: List[str]) -> Set[str]:
    new_names_list = []
    for potential_new_name in names_list:
        print(f"Inspecting {potential_new_name}")
        found=False
        for i,added_names in enumerate(new_names_list): 
            print(f"--/On {added_names} out of {new_names_list}")
            if return_most_global(added_names, potential_new_name) != "":
                if len(potential_new_name) < len(added_names) and potential_new_name != added_names:
                    print(f"--#Replacing {new_names_list[i]} for {potential_new_name}")
                    new_names_list[i] = potential_new_name
                found=True
        if not found:
            new_names_list.append(potential_new_name)
            print(f"--*Adding {potential_new_name}")
    return set(new_names_list)

def find_parent_namespace(available_names_spaces: List[str], name: str) -> str:
    scopes = name.split(".")
    inspecting = ""
    for _ in range(len(scopes)):
        inspecting += "." + scopes.pop(0)
        if inspecting in available_names_spaces:
            return inspecting

    raise ValueError(f"Could not find parent namespace for {name} in {available_names_spaces}")

def histogram_all_modules(
    modules: Dict[str, nn.Module],
    provided_filter: Optional[List[str]] = None,
    filter_aggregates: bool = False,
    num_buckets: int = 20,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Will sned all histograms about model to wandb
    Params:
        - modules: Which modules to log
        - Optional[filter]: If set, it will whitelist which specific tensors will be plotted
        - filter_aggregates: If set it will aggregate child tensors under its parent filter name. 
    """
    # Ensure we clean the filters
    filter: Optional[List[str]] = None
    if provided_filter:
        filter = list(fix_namespace_duplicates(provided_filter))

    child_to_parent_mapping = DefaultDict(list)

    # Where we store our tensors with the weight information.
    distributions_to_aggregate: Dict[str, List[np.ndarray]] = DefaultDict(list)

    for module_name, module in modules.items():
        if not isinstance(module, nn.Module):
            continue

        for param_name, param in module.named_parameters():
            # Determine if this module should be aggregated
            if not filter: 
                report_key = param_name
            else:
                report_key = find_parent_namespace(filter, param_name)

            distributions_to_aggregate[report_key].append(param.detach().cpu().numpy())

    # Now aggregate into buckets
    distributions_to_report: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for report_key, distribution in distributions_to_report.items():
        meep = np.histogram(distribution, bins=num_buckets)
        distributions_to_report[report_key] = meep

    return distributions_to_report

        # if aggregate and aggregated_params:
        #     aggregated_params = torch.cat([torch.tensor(p).flatten() for p in aggregated_params])
        #     wandb.log(
        #         {
        #             f"histogram_{module_name}_aggregated": wandb.Histogram(
        #                 aggregated_params.numpy()
        #             ),
        #             f"{module_name}_mean": aggregated_params.mean().item(),
        #             f"{module_name}_std": aggregated_params.std().item(),
        #         }
        #     )

        # for buffer_name, buffer in module.named_buffers(recurse=not aggregate):
        #     full_name = f"{module_name}.{buffer_name}"
        #     if aggregate:
        #         aggregated_params.append(buffer.detach().cpu().numpy())
        #     elif filter is None or any(f in full_name for f in filter):
        #         wandb.log(
        #             {
        #                 f"histogram_{full_name}": wandb.Histogram(
        #                     buffer.detach().cpu().numpy()
        #                 )
        #             }
        #         )
        #
        # If the module was meant to be aggregated, compute statistics

