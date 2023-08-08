//date: 2023-08-08T17:02:42Z
//url: https://api.github.com/gists/440ff3e517d221b4cca22acf42efe2ab
//owner: https://api.github.com/users/MairwunNx

package org.rubber.dwm.internal;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
import java.util.function.IntSupplier;
import java.util.logging.Logger;

import static java.lang.foreign.MemorySegment.NULL;
import static java.lang.foreign.ValueLayout.*;
import static java.nio.charset.StandardCharsets.UTF_16LE;
import static org.rubber.dwm.internal.Natives.*;
// todo: WM_EXITSIZEMOVE for optimize resize, like in minecraft
public class Zygote {

    private static final Logger logger = Logger.getLogger("base");
    private static final int PAINT_COLOR = 0x001E97BA;  // In RGB format
    private static final Set<Long> keysPressed = new HashSet<>();
    private static int windowWidth;
    private static int windowHeight;

    // todo: add support AdjustWindowRectExForDpi
    public static void main(String[] args) throws InterruptedException {
        boolean isImmediateRendering = Arrays.asList(args).contains("--ImdtRnr");

        Thread.ofVirtual().name("window-loop-thread").uncaughtExceptionHandler((t, e) -> logger.severe("An error occurred from thread %s with exception %s".formatted(t, e))).start(() -> {
            try (final var arena = Arena.ofAuto()) {
                final var fnt = arena.allocateArray(JAVA_BYTE, ("Segoe UI" + "\0").getBytes(UTF_16LE));
                final var hInstance = arena.allocate(HMODULE);
                requireSuccess(() -> GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, NULL, hInstance));
                final var hInstanceVal = hInstance.get(HINSTANCE, 0);
                logger.info("Hmodule: %d".formatted(hInstanceVal.get(JAVA_LONG, 0)));

                final var fakeWndProc = WNDPROC.allocate((hWnd, uMsg, wParam, lParam) -> switch (uMsg) {
                    case WM_CREATE, WM_DESTROY -> 0;
                    default -> DefWindowProcW(hWnd, uMsg, wParam, lParam);
                }, arena);

                final var wndProc = WNDPROC.allocate((hWnd, uMsg, wParam, lParam) -> switch (uMsg) {
                    case WM_DESTROY -> { // After closing
                        PostQuitMessage(0);
                        yield 0;
                    }
                    case WM_NCCREATE -> { // Window Message - Non-Client Create
                        logger.info("A WM_NCCREATE message received, window is being created.");
                        // here you can modify styles or attributes that affect the non-client area
                        yield DefWindowProcW(hWnd, uMsg, wParam, lParam);
                    }
                    case WM_CREATE -> {
                        logger.info("A WM_CREATE message received, window has been successfully created.");
                        // Initialization of resources, etc. goes here.
                        yield DefWindowProcW(hWnd, uMsg, wParam, lParam);
                    }
                    case WM_QUIT -> {
                        logger.info("A WM_QUIT message received, application is terminating.");
                        System.exit(0);
                        yield 0;
                    }
                    case WM_SIZE -> {
                        switch ((int) wParam) {
                            case SIZE_RESTORED -> logger.info("Window was restored.");
                            case SIZE_MINIMIZED -> logger.info("Window was minimized.");
                            case SIZE_MAXIMIZED -> logger.info("Window was maximized.");
                            default -> logger.info("Window was resized.");
                        }

                        windowWidth = LOWORD(lParam);
                        windowHeight = HIWORD(lParam);
                        glViewport(0, 0, windowWidth, windowHeight);

                        yield DefWindowProcW(hWnd, uMsg, wParam, lParam);
                    }
                    case WM_MOVE -> {
                        int xPos = LOWORD(lParam);
                        int yPos = HIWORD(lParam);
                        logger.info("Window moved. Position: " + xPos + ", " + yPos);
                        yield DefWindowProcW(hWnd, uMsg, wParam, lParam);
                    }
                    case WM_ACTIVATE -> {
                        logger.info("Window activation state changed.");
                        yield DefWindowProcW(hWnd, uMsg, wParam, lParam);
                    }
                    case WM_MOUSEMOVE -> {
                        int xPos = LOWORD(lParam);
                        int yPos = HIWORD(lParam);
                        logger.info("Mouse moved. Position: " + xPos + ", " + yPos);
                        yield 0;
                    }
                    case WM_LBUTTONDOWN -> {
                        logger.info("Left mouse button pressed.");
                        yield DefWindowProcW(hWnd, uMsg, wParam, lParam);
                    }
                    case WM_LBUTTONUP -> {
                        logger.info("Left mouse button released.");
                        yield DefWindowProcW(hWnd, uMsg, wParam, lParam);
                    }
                    case WM_RBUTTONDOWN -> {
                        logger.info("Right mouse button pressed.");
                        yield DefWindowProcW(hWnd, uMsg, wParam, lParam);
                    }
                    case WM_RBUTTONUP -> {
                        logger.info("Right mouse button released.");
                        yield DefWindowProcW(hWnd, uMsg, wParam, lParam);
                    }
                    case WM_MOUSEHOVER -> {
                        logger.info("Mouse hovered over the window.");
                        yield DefWindowProcW(hWnd, uMsg, wParam, lParam);
                    }
                    case WM_KEYDOWN -> {
                        keysPressed.add(wParam);
                        logger.info("Key pressed. KeyCode: " + wParam + ". Currently pressed: " + keysPressed);
                        yield 0;
                    }
                    case WM_KEYUP -> {
                        keysPressed.remove(wParam);
                        logger.info("Key released. KeyCode: " + wParam + ". Currently pressed: " + keysPressed);
                        yield 0;
                    }
                    case WM_CHAR -> {
                        logger.info("A character key was pressed.");
                        yield DefWindowProcW(hWnd, uMsg, wParam, lParam);
                    }
                    case WM_MOUSEWHEEL -> {
                        int zDelta = (short) (wParam >>> 16);  // This does sign-extension
                        if (zDelta > 0) {
                            logger.info("Mouse wheel scrolled up");
                        } else if (zDelta < 0) {
                            logger.info("Mouse wheel scrolled down");
                        }
                        yield 0;
                    }
                    case WM_DROPFILES -> {
                        final var hwndUninterpreted = HDROP__.ofAddress(MemorySegment.ofAddress(wParam), arena);
                        logger.info("hdrop address: %d".formatted(hwndUninterpreted.address()));

                        int fileCount = DragQueryFileW(hwndUninterpreted, 0xFFFFFFFF, NULL, 0);
                        logger.info("Dropped file count: %d".formatted(fileCount));
                        for (int i = 0; i < fileCount; i++) {
                            int fileNameLength = DragQueryFileW(hwndUninterpreted, i, NULL, 0);
                            logger.info("Dropped file name buffer length: %d".formatted(fileNameLength));
                            arena.allocateArray(JAVA_CHAR, fileNameLength + 1);
//                char[] fileNameBuffer = new char[fileNameLength + 1];
                            final var chararr = arena.allocateArray(JAVA_CHAR, fileNameLength + 1);
                            DragQueryFileW(hwndUninterpreted, i, chararr, fileNameLength + 1);
                            String fileName = new String(chararr.toArray(JAVA_CHAR)).trim();
                            logger.info("File dropped: " + fileName);
                        }
                        DragFinish(hwndUninterpreted);
                        yield 0;
                    }
                    case WM_CONTEXTMENU -> {
                        int xPos = LOWORD(lParam);
                        int yPos = HIWORD(lParam);
                        logger.info("Context menu requested. Position: " + xPos + ", " + yPos);
                        yield 0;
                    }
                    case WM_CLOSE -> { // on wnd close button
                        logger.info("Window is closing.");
                        DestroyWindow(hWnd);
                        yield 0;
                    }
                    case WM_PAINT -> {
                        final var ps = PAINTSTRUCT.allocate(arena);
                        final var hdc = BeginPaint(hWnd, ps);
                        final var rect = RECT.allocate(arena);
                        GetClientRect(hWnd, rect);
                        final var hbrush = CreateSolidBrush(PAINT_COLOR);
                        FillRect(hdc, rect, hbrush);
                        DeleteObject(hbrush);
                        SwapBuffers(hWnd);
                        EndPaint(hWnd, ps);

//              Paint paint = new Paint().setColor(0xff4286f4);
//              canvas.drawRect(Rect.makeXYWH(10, 10, 100, 50), paint);

                        yield 0;
                    }
                    case WM_ERASEBKGND -> {
                        logger.info("Window background must be erased.");
                        yield DefWindowProcW(hWnd, uMsg, wParam, lParam);
                    }
                    case WM_WINDOWPOSCHANGED -> {
                        final var hwndpos = WINDOWPOS.ofAddress(MemorySegment.ofAddress(lParam), arena);
                        final var x = WINDOWPOS.x$get(hwndpos);
                        final var y = WINDOWPOS.y$get(hwndpos);
                        final var cx = WINDOWPOS.cx$get(hwndpos);
                        final var cy = WINDOWPOS.cy$get(hwndpos);

                        logger.info("Window size, position, or Z order has changed. ( x = %d , y = %d , cx = %d , cy = %d)".formatted(x, y, cx, cy));
                        yield DefWindowProcW(hWnd, uMsg, wParam, lParam);
                    }
                    case WM_SETFOCUS -> {
                        logger.info("Window has gained keyboard focus.");
                        yield DefWindowProcW(hWnd, uMsg, wParam, lParam);
                    }
                    case WM_KILLFOCUS -> {
                        logger.info("Window has lost keyboard focus.");
                        yield DefWindowProcW(hWnd, uMsg, wParam, lParam);
                    }
                    case WM_ENABLE -> {
                        logger.info("Window enable state changed.");
                        yield DefWindowProcW(hWnd, uMsg, wParam, lParam);
                    }
                    case WM_SETTEXT -> {
                        logger.info("A WM_SETTEXT message received");
                        yield DefWindowProcW(hWnd, uMsg, wParam, lParam);
                    }
                    case WM_GETTEXT -> {
                        logger.info("A WM_GETTEXT message received");
                        yield DefWindowProcW(hWnd, uMsg, wParam, lParam);
                    }
                    case 0x031D -> {
                        logger.info("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Clipboard changed!");
                        yield 0;
                    }
                    default -> DefWindowProcW(hWnd, uMsg, wParam, lParam);
                }, arena);

                final var fakeclsname = wstr(arena, "glfakewndcls");
                final var fakehwndclass = WNDCLASSEXW.allocate(arena);
                WNDCLASSEXW.cbSize$set(fakehwndclass, (int) WNDCLASSEXW.sizeof());
                WNDCLASSEXW.style$set(fakehwndclass, CS_VREDRAW | CS_HREDRAW | CS_OWNDC | CS_DBLCLKS);
                WNDCLASSEXW.lpfnWndProc$set(fakehwndclass, fakeWndProc);
                WNDCLASSEXW.hInstance$set(fakehwndclass, hInstance);
                WNDCLASSEXW.lpszClassName$set(fakehwndclass, fakeclsname);

                RegisterClassExW(fakehwndclass);

                final var fakehwnd = CreateWindowExW(0, fakeclsname, wstr(arena, "glfakewindow"), WS_OVERLAPPEDWINDOW, CW_USEDEFAULT(), CW_USEDEFAULT(), CW_USEDEFAULT(), CW_USEDEFAULT(), NULL, NULL, hInstance, NULL);

                final var fakehdc = GetDC(fakehwnd);

                final var pfd = PIXELFORMATDESCRIPTOR.allocate(arena);
                PIXELFORMATDESCRIPTOR.nSize$set(pfd, (short) PIXELFORMATDESCRIPTOR.sizeof());
                PIXELFORMATDESCRIPTOR.nVersion$set(pfd, (short) 1);
                PIXELFORMATDESCRIPTOR.dwFlags$set(pfd, PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER);
                PIXELFORMATDESCRIPTOR.iPixelType$set(pfd, (byte) PFD_TYPE_RGBA);
                PIXELFORMATDESCRIPTOR.cColorBits$set(pfd, (byte) 32);
                PIXELFORMATDESCRIPTOR.cRedBits$set(pfd, (byte) 0);
                PIXELFORMATDESCRIPTOR.cRedShift$set(pfd, (byte) 0);
                PIXELFORMATDESCRIPTOR.cGreenBits$set(pfd, (byte) 0);
                PIXELFORMATDESCRIPTOR.cGreenShift$set(pfd, (byte) 0);
                PIXELFORMATDESCRIPTOR.cBlueBits$set(pfd, (byte) 0);
                PIXELFORMATDESCRIPTOR.cBlueShift$set(pfd, (byte) 0);
                PIXELFORMATDESCRIPTOR.cAlphaBits$set(pfd, (byte) 0);
                PIXELFORMATDESCRIPTOR.cAlphaShift$set(pfd, (byte) 0);
                PIXELFORMATDESCRIPTOR.cAccumBits$set(pfd, (byte) 0);
                PIXELFORMATDESCRIPTOR.cAccumRedBits$set(pfd, (byte) 0);
                PIXELFORMATDESCRIPTOR.cAccumGreenBits$set(pfd, (byte) 0);
                PIXELFORMATDESCRIPTOR.cAccumBlueBits$set(pfd, (byte) 0);
                PIXELFORMATDESCRIPTOR.cAccumAlphaBits$set(pfd, (byte) 0);
                PIXELFORMATDESCRIPTOR.cDepthBits$set(pfd, (byte) 24);
                PIXELFORMATDESCRIPTOR.cStencilBits$set(pfd, (byte) 8);
                PIXELFORMATDESCRIPTOR.cAuxBuffers$set(pfd, (byte) 0);
                PIXELFORMATDESCRIPTOR.iLayerType$set(pfd, (byte) PFD_MAIN_PLANE);
                PIXELFORMATDESCRIPTOR.bReserved$set(pfd, (byte) 0);
                PIXELFORMATDESCRIPTOR.dwLayerMask$set(pfd, 0);
                PIXELFORMATDESCRIPTOR.dwVisibleMask$set(pfd, 0);
                PIXELFORMATDESCRIPTOR.dwDamageMask$set(pfd, 0);

                final var pixfmt = ChoosePixelFormat(fakehdc, pfd);

                SetPixelFormat(fakehdc, pixfmt, pfd);

                final var fakehglrc = wglCreateContext(fakehdc);
                wglMakeCurrent(fakehdc, fakehglrc);


                final var hwglChoosePixelFormatARB = wglGetProcAddress(arena.allocateUtf8String("wglChoosePixelFormatARB"));
                final var hwglCreateContextAttribsARB = wglGetProcAddress(arena.allocateUtf8String("wglCreateContextAttribsARB"));
                final var hwglSwapIntervalEXT = wglGetProcAddress(arena.allocateUtf8String("wglSwapIntervalEXT"));

                final var wglChoosePixelFormatARB = PFNWGLCHOOSEPIXELFORMATARBPROC.ofAddress(hwglChoosePixelFormatARB, arena);
                final var wglCreateContextAttribsARB = PFNWGLCREATECONTEXTATTRIBSARBPROC.ofAddress(hwglCreateContextAttribsARB, arena);
                final var wglSwapIntervalEXT = PFNWGLSWAPINTERVALEXTPROC.ofAddress(hwglSwapIntervalEXT, arena);

                // Now normal window!

                final var title = wstr(arena, "hello");
                final var clsName = wstr(arena, "clsnametest");
                final var wndclass = WNDCLASSEXW.allocate(arena);

                WNDCLASSEXW.cbSize$set(wndclass, (int) WNDCLASSEXW.sizeof());
                WNDCLASSEXW.style$set(wndclass, CS_VREDRAW | CS_HREDRAW | CS_OWNDC);
                WNDCLASSEXW.lpfnWndProc$set(wndclass, wndProc);
                WNDCLASSEXW.hInstance$set(wndclass, hInstance);
                WNDCLASSEXW.lpszClassName$set(wndclass, clsName);
                WNDCLASSEXW.hCursor$set(wndclass, LoadImageW(NULL, wstr(arena, "#32512"), IMAGE_CURSOR, 0, 0, LR_SHARED));

                requireSuccess(() -> RegisterClassExW(wndclass));

                final var hwnd = CreateWindowExW(0, clsName, title, WS_OVERLAPPEDWINDOW, CW_USEDEFAULT(), CW_USEDEFAULT(), CW_USEDEFAULT(), CW_USEDEFAULT(), NULL, NULL, hInstance, NULL);
                final var hdc = GetDC(hwnd);

                final var pixelFormatID = arena.allocate(JAVA_INT);
                final var numFormats = arena.allocate(JAVA_INT);
                final var pixelAttribs = arena.allocateArray(
                        JAVA_INT,
                        WGL_DRAW_TO_WINDOW_ARB, GL_TRUE,
                        WGL_SUPPORT_OPENGL_ARB, GL_TRUE,
                        WGL_DOUBLE_BUFFER_ARB, GL_TRUE,
                        WGL_PIXEL_TYPE_ARB, WGL_TYPE_RGBA_ARB,
                        WGL_ACCELERATION_ARB, WGL_FULL_ACCELERATION_ARB,
                        WGL_COLOR_BITS_ARB, 32,
                        WGL_ALPHA_BITS_ARB, 8,
                        WGL_DEPTH_BITS_ARB, 24,
                        WGL_STENCIL_BITS_ARB, 8,
                        WGL_SAMPLE_BUFFERS_ARB, GL_TRUE,
                        WGL_SAMPLES_ARB, 4,
                        0
                );
                final var status = wglChoosePixelFormatARB.apply(hdc, pixelAttribs, NULL, 1, pixelFormatID, numFormats);
                final var newpfd = PIXELFORMATDESCRIPTOR.allocate(arena);
                DescribePixelFormat(hdc, pixelFormatID.get(JAVA_INT, 0), (int) newpfd.byteSize(), newpfd);
                SetPixelFormat(hdc, pixelFormatID.get(JAVA_INT, 0), newpfd);

                int major_min = 4, minor_min = 6;
                final var contextAttribs = arena.allocateArray(
                        JAVA_INT,
                        WGL_CONTEXT_MAJOR_VERSION_ARB, major_min,
                        WGL_CONTEXT_MINOR_VERSION_ARB, minor_min,
                        WGL_CONTEXT_LAYER_PLANE_ARB, 0,
                        WGL_CONTEXT_FLAGS_ARB, WGL_CONTEXT_FORWARD_COMPATIBLE_BIT_ARB | WGL_CONTEXT_DEBUG_BIT_ARB,
                        WGL_CONTEXT_PROFILE_MASK_ARB, WGL_CONTEXT_CORE_PROFILE_BIT_ARB,
                        0
                );
                final var hglrc = wglCreateContextAttribsARB.apply(hdc, NULL, contextAttribs);
                wglMakeCurrent(NULL, NULL);
                wglDeleteContext(fakehglrc);
                ReleaseDC(fakehwnd, fakehdc);
                DestroyWindow(fakehwnd);
                wglMakeCurrent(hdc, hglrc);

                DragAcceptFiles(hwnd, 1);
                SetClipboardViewer(hwnd);

                final var glver = glGetString(GL_VERSION).getUtf8String(0);
                System.out.println("glver = " + glver);

                glEnable(GL_DEBUG_OUTPUT);
                // if vsync:
                wglSwapIntervalEXT.apply(1);

                glEnable(GL_DEPTH_TEST);
                glEnable(GL_BLEND);
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
                glEnable(GL_SCISSOR_TEST);
//                glScissor(x, y, width, height); for clipping scrollable area

                glTexParameteri(GL_TEXTURE_2D(), GL_TEXTURE_MIN_FILTER, GL_LINEAR); // or GL_NEAREST
                glTexParameteri(GL_TEXTURE_2D(), GL_TEXTURE_MAG_FILTER, GL_LINEAR); // or GL_NEAREST

                glTexParameteri(GL_TEXTURE_2D(), GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                glTexParameteri(GL_TEXTURE_2D(), GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

                glViewport(0, 0, windowWidth, windowHeight);

                final var rect = RECT.allocate(arena);
                RECT.left$set(rect, 0);
                RECT.top$set(rect, 0);
                RECT.right$set(rect, windowWidth);
                RECT.bottom$set(rect, windowHeight);

                long style = GetWindowLongW(hwnd, GWL_STYLE());
                int dpi = GetDpiForWindow(hwnd);
                if (AdjustWindowRectExForDpi(rect, (int) style, 0, 0, dpi) == 0) {
                    logger.severe("Failed to adjust window rect: %d".formatted(GetLastError()));
                    return;
                }

                int adjustedWidth = RECT.right$get(rect) - RECT.left$get(rect);
                int adjustedHeight = RECT.bottom$get(rect) - RECT.top$get(rect);

                windowWidth = adjustedWidth;
                windowHeight = adjustedHeight;

                SetWindowPos(hwnd, NULL, CW_USEDEFAULT(), CW_USEDEFAULT(), adjustedWidth, adjustedHeight, SWP_NOZORDER | SWP_NOACTIVATE);

                ShowWindow(hwnd, SW_SHOWNORMAL);
                SetForegroundWindow(hwnd);
                UpdateWindow(hwnd);

                if (isImmediateRendering) {
                    logger.info("Using immediate rendering (used --ImdtRnr arg)");
                    // Use PeekMessageW for immediate rendering.
                    final var msg = MSG.allocate(arena);
                    while (true) {
                        if (PeekMessageW(msg, hwnd, 0, 0, 1) != 0) {
                            TranslateMessage(msg);
                            DispatchMessageW(msg);
                        }

                        // OpenGL render calls
                        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);  // Clear the buffers
                        glClearColor(1.0f, 0.0f, 1.0f, 1.0f);  // Set background color to black (can be any color)
                        SwapBuffers(hdc);  // Swap front and back buffers to display rendered content
                    }
                } else {
                    logger.info("Using lazy rendering");
                    // Use GetMessageW for standard message loop.
                    final var msg = MSG.allocate(arena);
                    int getMessageResult;
                    while ((getMessageResult = GetMessageW(msg, hwnd, 0, 0)) != 0) {
                        if (getMessageResult == -1) {
                            logger.severe("GetMessageW error: %d".formatted(GetLastError()));
                            break;
                        }
                        TranslateMessage(msg);
                        DispatchMessageW(msg);

                        // OpenGL render calls
                        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);  // Clear the buffers
                        glClearColor(1.0f, 1.0f, 0.0f, 1.0f);  // Set background color to black (can be any color)
                        SwapBuffers(hdc);  // Swap front and back buffers to display rendered content
                    }
                }

            }
        }).join();
    }

    // todo: set progress on icon in tray
    // todo: working with app icon
    private static MemorySegment makeResourceName(Arena arena, String name) {
        return arena.allocate(C_POINTER, MemorySegment.ofAddress(Long.parseLong(name.substring(1))));
    }

    private static MemorySegment wstr(Arena arena, String name) {
        return arena.allocateArray(JAVA_BYTE, (name + "\0").getBytes(UTF_16LE));
    }

    static int LOWORD(long value) {
        return (int) value & 0xFFFF;
    }

    static int HIWORD(long value) {
        return (int) ((value >> 16) & 0xFFFF);
    }

    public static void requireSuccess(IntSupplier f) {
        if (f.getAsInt() == 0) {
            final var errorCode = GetLastError();
            final var msg = "An error occurred while working with native winapi call, error code: %d".formatted(errorCode);
            logger.severe(msg);
            throw new RuntimeException(msg);
        }
    }

}
