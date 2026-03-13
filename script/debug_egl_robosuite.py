import os
import ctypes

# Force EGL
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["MUJOCO_GL"] = "egl"

from mujoco.egl import egl_ext as EGL
import OpenGL.error


def debug_egl():
    print("Querying EGL devices...")
    try:
        devices = EGL.eglQueryDevicesEXT()
        print(f"Found {len(devices)} EGL devices.")
    except Exception as e:
        print(f"eglQueryDevicesEXT failed: {e}")
        return

    for i, dev in enumerate(devices):
        print(f"\nTesting Device {i}:")
        try:
            # Query device extensions to see if we can get a vendor string
            # EGL_DEVICE_EXTENSIONS = 0x3229 (not exported directly in all wrappers, use raw if needed)
            pass
        except:
            pass

        display = EGL.eglGetPlatformDisplayEXT(EGL.EGL_PLATFORM_DEVICE_EXT, dev, None)
        if display == EGL.EGL_NO_DISPLAY:
            print(f"  eglGetPlatformDisplayEXT returned NO_DISPLAY for device {i}")
            continue

        err = EGL.eglGetError()
        if err != EGL.EGL_SUCCESS:
            print(f"  eglGetPlatformDisplayEXT error: {err}")
            continue

        initialized = EGL.eglInitialize(display, None, None)
        if not initialized:
            print(f"  eglInitialize failed for device {i}")
            continue

        print(f"  Device {i} initialized successfully.")

        # Try to use same attributes as robosuite but simplified
        EGL_ATTRIBUTES = (
            EGL.EGL_RED_SIZE,
            8,
            EGL.EGL_GREEN_SIZE,
            8,
            EGL.EGL_BLUE_SIZE,
            8,
            EGL.EGL_ALPHA_SIZE,
            8,
            EGL.EGL_DEPTH_SIZE,
            24,
            # EGL.EGL_STENCIL_SIZE, 8,  # Try without stencil
            EGL.EGL_COLOR_BUFFER_TYPE,
            EGL.EGL_RGB_BUFFER,
            EGL.EGL_SURFACE_TYPE,
            EGL.EGL_PBUFFER_BIT,
            EGL.EGL_RENDERABLE_TYPE,
            EGL.EGL_OPENGL_BIT,
            EGL.EGL_NONE,
        )

        # Try to create a context
        EGL.eglBindAPI(EGL.EGL_OPENGL_API)

        EGL.eglReleaseThread()
        config = EGL.EGLConfig()
        num_configs = ctypes.c_long()
        EGL.eglChooseConfig(
            display, EGL_ATTRIBUTES, ctypes.byref(config), 1, num_configs
        )

        if num_configs.value < 1:
            print(f"  eglChooseConfig failed for device {i}")
            continue

        context = EGL.eglCreateContext(display, config, EGL.EGL_NO_CONTEXT, None)
        if context == EGL.EGL_NO_CONTEXT:
            err = EGL.eglGetError()
            print(f"  eglCreateContext FAILED for device {i} with error: {hex(err)}")
        else:
            print(f"  eglCreateContext SUCCESS for device {i}")
            EGL.eglDestroyContext(display, context)

        EGL.eglTerminate(display)


if __name__ == "__main__":
    debug_egl()
