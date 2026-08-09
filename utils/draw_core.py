import numpy as np
from mediapipe.tasks.python.vision import drawing_utils

def get_scale_factors(image_shape):
    height, width = image_shape[:2]
    line_scale = max(2, int(width / 300))
    dot_scale = max(1, int(width / 1000))
    return line_scale, dot_scale

def scale_connection_spec(spec, line_scale):
    if isinstance(spec, dict):
        return {k: scale_connection_spec(v, line_scale) for k, v in spec.items()}
    elif isinstance(spec, drawing_utils.DrawingSpec):
        return drawing_utils.DrawingSpec(
            color=spec.color,
            thickness=spec.thickness * line_scale,
            circle_radius=spec.circle_radius
        )
    return spec

def scale_landmark_spec(spec, dot_scale):
    if isinstance(spec, dict):
        return {k: scale_landmark_spec(v, dot_scale) for k, v in spec.items()}
    elif isinstance(spec, drawing_utils.DrawingSpec):
        new_thickness = spec.thickness * dot_scale if spec.thickness > 0 else spec.thickness
        return drawing_utils.DrawingSpec(
            color=spec.color,
            thickness=new_thickness,
            circle_radius=spec.circle_radius * dot_scale
        )
    return spec
