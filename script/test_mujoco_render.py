import os
import mujoco
import numpy as np

# Set GL backend to EGL
os.environ["MUJOCO_GL"] = "egl"

# Create a simple model
xml = """
<mujoco>
  <worldbody>
    <geom type="plane" size="5 5 0.1" rgba="0.9 0 0 1"/>
    <body pos="0 0 1">
      <joint type="free"/>
      <geom type="sphere" size="0.1" rgba="0 0.9 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""
try:
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)

    # Step simulation and render
    mujoco.mj_step(model, data)
    renderer.update_scene(data)
    pixels = renderer.render()

    print(f"Successfully rendered image with shape {pixels.shape}")
    print("MuJoCo EGL rendering is WORKING.")
except Exception as e:
    print(f"MuJoCo EGL rendering FAILED: {e}")
    import traceback

    traceback.print_exc()
