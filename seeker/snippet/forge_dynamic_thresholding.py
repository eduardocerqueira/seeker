#date: 2024-03-19T17:04:23Z
#url: https://api.github.com/gists/058c7ea5ca03b3a28aab6b76b0287816
#owner: https://api.github.com/users/pekochan069

from functools import partial
from random import choice
from typing import Any
import sys
import traceback
import gradio as gr

from modules import scripts, script_callbacks
from lib_dynamic_thresholding.dynthres_core import DynThresh
from lib_dynamic_thresholding.dynthres import DynamicThresholdingNode

opDynamicThresholdingNode = DynamicThresholdingNode().patch


class DynamicThresholdingForForge(scripts.Script):
    sorting_priority = 11

    def title(self):
        return "DynamicThresholding (CFG-Fix) Integrated"

    def show(self, is_img2img):
        # make this extension visible in both txt2img and img2img tab.
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            dynthresh_enabled = gr.Checkbox(label='Enabled', value=False)
            dynthresh_mimic_scale = gr.Slider(label='Mimic Scale', minimum=0.0, maximum=100.0, step=0.5, value=7.0)
            dynthresh_threshold_percentile = gr.Slider(label='Threshold Percentile', minimum=0.0, maximum=1.0, step=0.01,
                                             value=1.0)
            dynthresh_mimic_mode = gr.Radio(label='Mimic Mode',
                                  choices=['Constant', 'Linear Down', 'Cosine Down', 'Half Cosine Down', 'Linear Up',
                                           'Cosine Up', 'Half Cosine Up', 'Power Up', 'Power Down', 'Linear Repeating',
                                           'Cosine Repeating', 'Sawtooth'], value='Constant')
            dynthresh_mimic_scale_min = gr.Slider(label='Mimic Scale Min', minimum=0.0, maximum=100.0, step=0.5, value=0.0)
            cfg_mode = gr.Radio(label='Cfg Mode',
                                choices=['Constant', 'Linear Down', 'Cosine Down', 'Half Cosine Down', 'Linear Up',
                                         'Cosine Up', 'Half Cosine Up', 'Power Up', 'Power Down', 'Linear Repeating',
                                         'Cosine Repeating', 'Sawtooth'], value='Constant')
            dynthresh_cfg_scale_min = gr.Slider(label='Cfg Scale Min', minimum=0.0, maximum=100.0, step=0.5, value=0.0)
            dynthresh_sched_val = gr.Slider(label='Sched Val', minimum=0.0, maximum=100.0, step=0.01, value=1.0)
            dynthresh_separate_feature_channels = gr.Radio(label='Separate Feature Channels', choices=['enable', 'disable'],
                                                 value='enable')
            dynthresh_scaling_startpoint = gr.Radio(label='Scaling Startpoint', choices=['MEAN', 'ZERO'], value='MEAN')
            dynthresh_variability_measure = gr.Radio(label='Variability Measure', choices=['AD', 'STD'], value='AD')
            dynthresh_interpolate_phi = gr.Slider(label='Interpolate Phi', minimum=0.0, maximum=1.0, step=0.01, value=1.0)

        return dynthresh_enabled, dynthresh_mimic_scale, dynthresh_threshold_percentile, dynthresh_mimic_mode, dynthresh_mimic_scale_min, cfg_mode, dynthresh_cfg_scale_min, \
            dynthresh_sched_val, dynthresh_separate_feature_channels, dynthresh_scaling_startpoint, dynthresh_variability_measure, dynthresh_interpolate_phi

    def process_before_every_sampling(self, p, *script_args, **kwargs):
        # This will be called before every sampling.
        # If you use highres fix, this will be called twice.

        dynthresh_enabled, dynthresh_mimic_scale, dynthresh_threshold_percentile, dynthresh_mimic_mode, dynthresh_mimic_scale_min, dynthresh_cfg_mode, dynthresh_cfg_scale_min, dynthresh_sched_val, dynthresh_separate_feature_channels, dynthresh_scaling_startpoint, dynthresh_variability_measure, dynthresh_interpolate_phi = script_args

        xyz = getattr(p, "_dynthresh_xyz", {})
        if "dynthresh_enabled" in xyz:
            dynthresh_enabled = xyz["dynthresh_enabled"] == "True"
        if "dynthresh_mimic_scale" in xyz:
            dynthresh_mimic_scale = float(xyz["dynthresh_mimic_scale"])
        if "dynthresh_threshold_percentile" in xyz:
            dynthresh_threshold_percentile = float(xyz["dynthresh_threshold_percentile"])
        if "dynthresh_mimic_mode" in xyz:
            dynthresh_mimic_mode = xyz["dynthresh_mimic_mode"]
        if "dynthresh_mimic_scale_min" in xyz:
            dynthresh_mimic_scale_min = float(xyz["dynthresh_mimic_scale_min"])
        if "dynthresh_cfg_mode" in xyz:
            dynthresh_cfg_mode = xyz["dynthresh_cfg_mode"]
        if "dynthresh_cfg_scale_min" in xyz:
            dynthresh_cfg_scale_min = float(xyz["dynthresh_cfg_scale_min"])
        if "dynthresh_sched_val" in xyz:
            dynthresh_sched_val = float(xyz["dynthresh_sched_val"])
        if "dynthresh_separate_feature_channels" in xyz:
            dynthresh_separate_feature_channels = xyz["dynthresh_separate_feature_channels"]
        if "dynthresh_scaling_startpoint" in xyz:
            dynthresh_scaling_startpoint = xyz["dynthresh_scaling_startpoint"]
        if "dynthresh_variability_measure" in xyz:
            dynthresh_variability_measure = xyz["dynthresh_variability_measure"]
        if "dynthresh_interpolate_phi" in xyz:
            dynthresh_interpolate_phi = float(xyz["dynthresh_interpolate_phi"])

        if not dynthresh_enabled:
            return

        unet = p.sd_model.forge_objects.unet

        unet = opDynamicThresholdingNode(unet, dynthresh_mimic_scale, dynthresh_threshold_percentile, dynthresh_mimic_mode, dynthresh_mimic_scale_min,
                                         dynthresh_cfg_mode, dynthresh_cfg_scale_min, dynthresh_sched_val, dynthresh_separate_feature_channels,
                                         dynthresh_scaling_startpoint, dynthresh_variability_measure, dynthresh_interpolate_phi)[0]

        p.sd_model.forge_objects.unet = unet

        # Below codes will add some logs to the texts below the image outputs on UI.
        # The extra_generation_params does not influence results.
        p.extra_generation_params.update(dict(
            dynthres_enabled=dynthresh_enabled,
            dynthres_mimic_scale=dynthresh_mimic_scale,
            dynthres_threshold_percentile=dynthresh_threshold_percentile,
            dynthres_mimic_mode=dynthresh_mimic_mode,
            dynthres_mimic_scale_min=dynthresh_mimic_scale_min,
            dynthres_cfg_mode=dynthresh_cfg_mode,
            dynthres_cfg_scale_min=dynthresh_cfg_scale_min,
            dynthres_sched_val=dynthresh_sched_val,
            dynthres_separate_feature_channels=dynthresh_separate_feature_channels,
            dynthres_scaling_startpoint=dynthresh_scaling_startpoint,
            dynthres_variability_measure=dynthresh_variability_measure,
            dynthres_interpolate_phi=dynthresh_interpolate_phi,
        ))

        return

def set_value(p, x: Any, xs: Any, *, field: str):
    if not hasattr(p, "_dynthresh_xyz"):
        p._dynthresh_xyz = {}
    p._dynthresh_xyz[field] = x

def make_axis_on_xyz_grid():
    xyz_grid = None
    for script in scripts.scripts_data:
        if script.script_class.__module__ == "xyz_grid.py":
            xyz_grid = script.module
            break
        
    if xyz_grid is None:
        return

    axis = [
        xyz_grid.AxisOption(
            "DynThresh Enabled",
            str,
            partial(set_value, field="dynthresh_enabled"),
            choices=lambda: ["True", "False"],
        ),
        xyz_grid.AxisOption(
            "DynThresh Mimic Scale",
            float,
            partial(set_value, field="dynthresh_mimic_scale"),
        ),
        xyz_grid.AxisOption(
            "DynThresh Threshold Percentile",
            float,
            partial(set_value, field="dynthresh_threshold_percentile"),
        ),
        xyz_grid.AxisOption(
            "DynThresh Mimic Mode",
            str,
            partial(set_value, field="dynthresh_mimic_mode"),
            choices=lambda: DynThresh.Modes,
        ),
        xyz_grid.AxisOption(
            "DynThresh Mimic Scale Min",
            float,
            partial(set_value, field="dynthresh_mimic_scale_min"),
        ),
        xyz_grid.AxisOption(
            "DynThresh Cfg Mode",
            str,
            partial(set_value, field="dynthresh_cfg_mode"),
            choices=lambda: DynThresh.Modes,
        ),
        xyz_grid.AxisOption(
            "DynThresh Cfg Scale Min",
            float,
            partial(set_value, field="dynthresh_cfg_scale_min"),
        ),
        xyz_grid.AxisOption(
            "DynThresh Sched Val",
            float,
            partial(set_value, field="dynthresh_sched_val"),
        ),
        xyz_grid.AxisOption(
            "DynThresh Separate Feature Channels",
            str,
            partial(set_value, field="dynthresh_separate_feature_channels"),
            choices=lambda: ["enable", "disable"],
        ),
        xyz_grid.AxisOption(
            "DynThresh Scaling Startpoint",
            str,
            partial(set_value, field="dynthresh_scaling_startpoint"),
            choices=lambda: DynThresh.Startpoints,
        ),
        xyz_grid.AxisOption(
            "DynThresh Variability Measure",
            str,
            partial(set_value, field="dynthresh_variability_measure"),
            choices=lambda: DynThresh.Variabilities,
        ),
        xyz_grid.AxisOption(
            "DynThresh Interpolate Phi",
            float,
            partial(set_value, field="dynthresh_interpolate_phi"),
        ),
    ]
    
    if not any(x.label.startswith("DynThresh") for x in xyz_grid.axis_options):
        xyz_grid.axis_options.extend(axis)
        
def on_before_ui():
    try:
        make_axis_on_xyz_grid()
    except Exception as e:
        traceback.print_exc()
        print(f"[-] Dynamic Thresholding xyz_grid error:\n{e}", file=sys.stderr)

script_callbacks.on_before_ui(on_before_ui)