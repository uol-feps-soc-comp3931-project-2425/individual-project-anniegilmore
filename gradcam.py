import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None
        self.hook_handles = []

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output
            output.retain_grad()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_backward_hook(backward_hook))

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output['level'].argmax(dim=1).item()

        self.model.zero_grad()
        output['level'][0, class_idx].backward()

        grads = self.gradients
        acts = self.activations

        pooled_grads = torch.mean(grads, dim=[0, 2, 3])
        for i in range(acts.shape[1]):
            acts[0, i, :, :] *= pooled_grads[i]

        heatmap = acts[0].mean(dim=0).cpu().detach().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= heatmap.max() + 1e-6  # avoid divide by zero

        return heatmap

    def overlay(self, heatmap, original_img):
        if isinstance(original_img, torch.Tensor):
            original_img = original_img.squeeze().permute(1, 2, 0).cpu().numpy()
            original_img = np.uint8(255 * original_img)
            
        original_np = np.array(original_img)

        heatmap_resized = cv2.resize(heatmap, (original_np.shape[1], original_np.shape[0]))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        superimposed = np.uint8(0.4 * heatmap_colored + 0.6 * original_np)

        return superimposed

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()
