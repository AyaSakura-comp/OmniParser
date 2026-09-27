"""Isolated CPU candidate; never modifies the source weights or live service."""
import copy
import torch

def prepare(model):
    """FP32 freeze/fold first, then BF16 frozen floating constants.

    Uses TorchScript graph APIs: pin the PyTorch version and revalidate on upgrade.
    Caller must supply BF16 input; original model stays FP32 and unchanged.
    """
    result = torch.jit.freeze(copy.deepcopy(model).eval())
    def convert(graph):
        for node in graph.nodes():
            if (node.kind() == 'prim::Constant' and node.hasAttribute('value')
                    and node.kindOf('value') == 't'):
                value = node.t('value')
                if value.is_floating_point():
                    node.t_('value', value.to(dtype=torch.bfloat16))
            for block in node.blocks():
                convert(block)
    convert(result.graph)
    return result


class CPUForward:
    """Fixed-resolution BF16/oneDNN path; convert heads back to FP32 for NMS.

    Call torch.jit.enable_onednn_fusion(True) once in an isolated CPU process.
    Warm up six predictions before recording steady-state latency. Resolution
    changes require a fresh process to avoid profiling-plan cross-shape effects.
    """
    def __init__(self, model, image_size):
        self.model = prepare(model)
        self.image_size = image_size

    def __call__(self, x):
        if x.device.type != 'cpu' or tuple(x.shape) != (1, 3, self.image_size, self.image_size):
            raise ValueError('Expected a fixed-size batch-one CPU RGB tensor')
        with torch.inference_mode(), torch.jit.optimized_execution(True):
            outputs = self.model(x.to(dtype=torch.bfloat16))
        if isinstance(outputs, torch.Tensor):
            return outputs.float()
        return [output.float() for output in outputs]
