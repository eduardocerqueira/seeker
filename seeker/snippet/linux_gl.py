#date: 2022-02-10T17:06:21Z
#url: https://api.github.com/gists/05f13182d26b70ceb6c2df739cbb3edd
#owner: https://api.github.com/users/larsoner

from ctypes import CDLL, c_int, byref
import faulthandler
import vtk as _vtk
faulthandler.enable()
renderWindow = _vtk.vtkRenderWindow()
logger = _vtk.vtkLogger
logger.Init()
logger.SetStderrVerbosity(_vtk.vtkLogger.VERBOSITY_MAX)
renderWindow.DebugOn()
renderWindow.InitializeFromCurrentContext()
renderWindow.SetOffScreenRendering(True)
# Eventually we should add more special cases here, but let's at least catch
# the linux-with-no-display and linux-with-old-opengl cases
# adapted from https://github.com/Kitware/VTK/blob/master/Rendering/OpenGL2/vtkXOpenGLRenderWindow.cxx
if isinstance(renderWindow, _vtk.vtkXOpenGLRenderWindow):
    # No display
    xlib = CDLL('libX11.so.6')
    assert xlib is not None
    display = xlib.XOpenDisplay()
    if not display:
        raise RuntimeError('Could not initialize display')
    # Bad OpenGL
    glx = CDLL('libGL.so')
    assert glx is not None
    attributes = [
        0x8010,  # GLX_DRAWABLE_TYPE
        0x00000001,  # GLX_WINDOW_BIT
        0x8011,  # GLX_RENDER_TYPE
        0x00000001,  # GLX_RGBA_BIT
        8,  # GLX_RED_SIZE
        1,
        9,  # GLX_GREEN_SIZE
        1,
        10,  # GLX_BLUE_SIZE
        1,
        12,  # GLX_DEPTH_SIZE
        1,
        11,  # GLX_ALPHA_SIZE
        1,
        0,
    ]
    attributes = (c_int * len(attributes))(*attributes)
    tmp = c_int()
    fb = glx.glXChooseFBConfig(display, xlib.XDefaultScreen(display), attributes, byref(tmp))
    if fb:
        xlib.XFree(fb)
    if tmp.value <= 0:
        raise RuntimeError('no decent framebuffer')
    print('Usable')