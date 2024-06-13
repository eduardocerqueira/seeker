#date: 2024-06-13T16:56:11Z
#url: https://api.github.com/gists/69905bdcfe1541e4f85c40eba36fee53
#owner: https://api.github.com/users/d-orm

import os
import sys

import pygame
import zengl

os.environ['SDL_WINDOWS_DPI_AWARENESS'] = 'permonitorv2'

from imgui.integrations.pygame import PygameRenderer
import imgui

pygame.init()
pygame.display.set_mode((1280, 720), flags=pygame.OPENGL | pygame.DOUBLEBUF, vsync=True)

ctx = zengl.context()

show_custom_window = True
size = pygame.display.get_window_size()
image = ctx.image(size, 'rgba8unorm', samples=4)

imgui.create_context()
impl = PygameRenderer()
io = imgui.get_io()
io.display_size = size

def create_pipeline():
    print("creating pipeline")
    return ctx.pipeline(
        vertex_shader='''
            #version 330 core

            out vec3 v_color;

            vec2 vertices[3] = vec2[](
                vec2(0.0, 0.8),
                vec2(-0.6, -0.8),
                vec2(0.6, -0.8)
            );

            vec3 colors[3] = vec3[](
                vec3(1.0, 0.0, 0.0),
                vec3(0.0, 1.0, 0.0),
                vec3(0.0, 0.0, 1.0)
            );

            void main() {
                gl_Position = vec4(vertices[gl_VertexID], 0.0, 1.0);
                v_color = colors[gl_VertexID];
            }
        ''',
        fragment_shader='''
            #version 330 core

            in vec3 v_color;

            layout (location = 0) out vec4 out_color;

            void main() {
                out_color = vec4(v_color, 1.0);
                out_color.rgb = pow(out_color.rgb, vec3(1.0 / 2.2));
            }
        ''',
        framebuffer=[image],
        topology='triangles',
        vertex_count=3,
    )

pipeline = create_pipeline()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                pipeline = create_pipeline()

        impl.process_event(event)
    impl.process_inputs()
    
    ctx.new_frame()
    image.clear()
    pipeline.render()
    image.blit()
    ctx.end_frame()

    imgui.new_frame()

    if imgui.begin_main_menu_bar():
        if imgui.begin_menu("File", True):

            clicked_quit, selected_quit = imgui.menu_item(
                "Quit", "Cmd+Q", False, True
            )

            if clicked_quit:
                sys.exit(0)

            imgui.end_menu()
        imgui.end_main_menu_bar()

    imgui.show_test_window()

    if show_custom_window:
        is_expand, show_custom_window = imgui.begin("Custom window", True)
        if is_expand:
            imgui.text("Bar")
            imgui.text_colored("Eggs", 0.2, 1.0, 0.0)
        imgui.end()

    imgui.render()
    impl.render(imgui.get_draw_data())

    pygame.display.flip()