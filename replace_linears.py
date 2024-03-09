from torch import nn
from .bitnet import BitLinear

activation_layers = [nn.SiLU, nn.ReLU, nn.GELU]

def replace_linears_in_hf(model, parent=None):
    """
    Replaces all instances of nn.Linear in the given model with BitLinear.
    If a Linear layer is immediately followed by a specified activation layer, sets flg_before_linear to False.
    refers: https://github.com/kyegomez/BitNet/blob/d32fb9b8d83028d9571bfb213d8c5e4e7b915e42/bitnet/replace_hf.py#L6

    Parameters:
        model (nn.Module): The model to modify.
        parent (nn.Module): The parent module of the current module being processed.
    """
    children = list(model.named_children())
    for i, (name, module) in enumerate(children):
        if isinstance(module, nn.Linear):
            # Check if the next module is in the specified activation layers
            next_module_is_activation = (
                i + 1 < len(children) and any(isinstance(children[i + 1][1], layer) for layer in activation_layers)
            )
            # Replace the nn.Linear with BitLinear
            setattr(
                model,
                name,
                BitLinear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    bias=module.bias is not None,
                    flg_before_linear=not next_module_is_activation,
                ),
            )
        else:
            # Recursively apply to child modules
            replace_linears_in_hf(module, parent=model)