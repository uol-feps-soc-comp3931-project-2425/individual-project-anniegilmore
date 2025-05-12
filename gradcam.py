import torch
import numpy as np
import cv2
from model_architecture import DiabeticRetinopathyNet
from torch.nn import Sequential
from cv2.typing import MatLike


class GradCAM:
    """
    Generates Grad-CAM heatmaps by hooking into forward and backward passes of given layer in DiabeticRetinopathyNet
    """

    def __init__(self, model: DiabeticRetinopathyNet, target_layer: Sequential):
        """

        Initialises GradCAM object with DiabeticRetinopathyNet model and the layer to be investigated
        Args:
            model (DiabeticRetinopathyNet): model used to make prediction
            target_layer (Sequential): layer whose gradients and activations are used to generate the heatmap

        """
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None
        self.hook_handles = []

        self._register_hooks()

    def _register_hooks(self):
        """
        Registers forward and backward hooks on the target layer to capture
        activations and gradients needed for computing Grad-CAM.
        """

        def forward_hook(module, input, output):
            """
            Stores forward activation values of the specified layer and maintains
            gradients for backward computation.

            Args:
                module (nn.Module): The target model (DiabeticRetinopathyNet) being hooked.
                input (Tuple): Input data to the model.
                output (Tensor): Output of the model.
            """
            self.activations = output
            output.retain_grad()

        def backward_hook(module, grad_input, grad_output):
            """
            Stores gradient values of the model output with respect to the activations.

            Args:
                module (nn.Module): The target model being hooked.
                grad_input (Tuple): Gradients with respect to the input data.
                grad_output (Tuple): Gradients with respect to the output data.
            """
            self.gradients = grad_output[0]

        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(
            self.target_layer.register_backward_hook(backward_hook)
        )

    def generate_heatmap(self, input_tensor: torch.Tensor) -> np.ndarray:
        """
        Computes Grad-CAM heatmap for a given input image to be predicted by the model.

        Args:
            input_tensor (torch.Tensor): Image tensor to be predicted.

        Returns:
            np.ndarray: Heatmap array highlighting important regions of image in clasification.
        """
        self.model.eval()
        output = self.model(input_tensor)
        predicted_level = output["level"].argmax(dim=1).item()
        self.model.zero_grad()
        output["level"][0, predicted_level].backward()
        gradients = self.gradients
        activations = self.activations
        pooled_gradients: torch.Tensor = torch.mean(gradients, dim=[0, 2, 3])
        for row_val in range(activations.shape[1]):
            activations[0, row_val, :, :] *= pooled_gradients[row_val]
        heatmap = activations[0].mean(dim=0).cpu().detach().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= heatmap.max() + 1e-6
        return heatmap

    def overlay_heatmap(self, heatmap: np.ndarray, original_image: MatLike) -> np.uint8:
        """
        Overlays a Grad-CAM heatmap onto the original image.

        Args:
            heatmap (np.ndarray): 2D Grad-CAM heatmap.
            original_image (MatLike): The original image that has been classified.

        Returns:
            np.uint8: The heatmap overlayed on the original image.
        """
        original_image_array: np.ndarray = np.array(original_image)
        heatmap_resized: MatLike = cv2.resize(
            heatmap, (original_image_array.shape[1], original_image_array.shape[0])
        )
        heatmap_colored: MatLike = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
        )
        overlayed_heatmap: np.uint8 = np.uint8(
            0.4 * heatmap_colored + 0.6 * original_image_array
        )
        return overlayed_heatmap

    def remove_hooks(self):
        """
        Removes all registered hooks from the model to free up resources.
        """
        for handle in self.hook_handles:
            handle.remove()
