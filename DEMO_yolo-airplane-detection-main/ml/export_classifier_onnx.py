"""
Экспорт классификатора в ONNX.
"""

import os
import torch
import torch.onnx
from torchvision import models


def export_classifier(pt_path: str, onnx_path: str, img_size: int = 224):
    """Экспорт классификатора в ONNX."""
    if not os.path.exists(pt_path):
        print(f"Warning: {pt_path} not found")
        return False
    
    # Загрузить чекпоинт
    ckpt = torch.load(pt_path, map_location='cpu', weights_only=False)
    num_classes = ckpt['num_classes']
    
    # Создать модель
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    # Экспорт в ONNX
    dummy_input = torch.randn(1, 3, img_size, img_size)
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=12,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
    )
    
    print(f"Classifier exported to {onnx_path}")
    print(f"  Classes: {num_classes}")
    print(f"  Input shape: (1, 3, {img_size}, {img_size})")
    return True


if __name__ == "__main__":
    export_classifier(
        pt_path="/weights/classifier.pth",
        onnx_path="/weights/classifier.onnx",
        img_size=224,
    )
