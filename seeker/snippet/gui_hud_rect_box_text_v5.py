#date: 2025-08-13T16:51:22Z
#url: https://api.github.com/gists/e911324f76a087c3694aba9c6f2ea413
#owner: https://api.github.com/users/EncodeTheCode

# message_rect_opengl_blur_with_custom_font.py
"""
Lean + memory-optimized version of the blurred message box demo.
- Reuses shared blur/draw programs (no per-message shader compile).
- Frees intermediate surfaces after uploading to GPU.
- Smaller, clearer code while keeping full functionality (custom font, padding, per-side overrides, box_opacity, blur_radius, fades, spawn on E).
"""
import pygame, time, ctypes
from pygame.locals import *
from OpenGL.GL import *

WINDOW_SIZE=(900,640);DEFAULT_FONT_SIZE=24
PADDING=(4,4,4,4) # left,top,right,bottom

VS='''#version 120
attribute vec2 a_pos;attribute vec2 a_uv;varying vec2 v_uv;void main(){v_uv=a_uv;gl_Position=vec4(a_pos,0.0,1.0);}'''
FS_BLUR='''#version 120
uniform sampler2D u_texture;uniform vec2 u_texelSize;uniform vec2 u_dir;varying vec2 v_uv;float g(float x,float s){return exp(-(x*x)/(2.0*s*s));}void main(){float s=4.0;vec4 sum=vec4(0.0);float ws=0.0;for(int i=-4;i<=4;++i){float w=g(float(i),s);sum+=texture2D(u_texture,v_uv+u_dir*(float(i)*u_texelSize))*w;ws+=w;}gl_FragColor=sum/ws;}'''
FS_DRAW='''#version 120
uniform sampler2D u_texture;uniform float u_alpha;varying vec2 v_uv;void main(){vec4 c=texture2D(u_texture,v_uv);c.a*=u_alpha;gl_FragColor=c;}'''

def compile(src, t):
    """Compile a GLSL shader from source and return its shader handle."""
    s = glCreateShader(t)
    glShaderSource(s, src)
    glCompileShader(s)
    if glGetShaderiv(s, GL_COMPILE_STATUS) != GL_TRUE:
        raise RuntimeError(glGetShaderInfoLog(s).decode())
    return s

def link(vs, fs):
    """Link a vertex and fragment shader into a program and return program id."""
    vs_s = compile(vs, GL_VERTEX_SHADER)
    fs_s = compile(fs, GL_FRAGMENT_SHADER)
    p = glCreateProgram()
    glAttachShader(p, vs_s)
    glAttachShader(p, fs_s)
    glLinkProgram(p)
    if glGetProgramiv(p, GL_LINK_STATUS) != GL_TRUE:
        raise RuntimeError(glGetProgramInfoLog(p).decode())
    glDeleteShader(vs_s)
    glDeleteShader(fs_s)
    return p

class GLTex:
    def __init__(self,w,h,surf=None):
        self.w=int(w);self.h=int(h);self.id=glGenTextures(1);glBindTexture(GL_TEXTURE_2D,self.id);glPixelStorei(GL_UNPACK_ALIGNMENT,1);glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
        if surf is None:glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,self.w,self.h,0,GL_RGBA,GL_UNSIGNED_BYTE,None)
        else:glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,self.w,self.h,0,GL_RGBA,GL_UNSIGNED_BYTE,pygame.image.tostring(surf,'RGBA',True));glBindTexture(GL_TEXTURE_2D,0)
    def bind(self):glBindTexture(GL_TEXTURE_2D,self.id)
    def delete(self):
        try:glDeleteTextures([self.id])
        except:pass

class FBO:
    def __init__(self,w,h):
        self.w=int(w);self.h=int(h);self.tex=GLTex(self.w,self.h);self.fbo=glGenFramebuffers(1);glBindFramebuffer(GL_FRAMEBUFFER,self.fbo);glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,self.tex.id,0);
        if glCheckFramebufferStatus(GL_FRAMEBUFFER)!=GL_FRAMEBUFFER_COMPLETE:raise RuntimeError('FBO incomplete')
        glBindFramebuffer(GL_FRAMEBUFFER,0)
    def bind(self):glBindFramebuffer(GL_FRAMEBUFFER,self.fbo);glViewport(0,0,self.w,self.h)
    def unbind(self,sw,sh):glBindFramebuffer(GL_FRAMEBUFFER,0);glViewport(0,0,sw,sh)
    def delete(self):
        try:glDeleteFramebuffers([self.fbo])
        except:pass;self.tex.delete()

def norm_pad(p):
    """Normalize padding input into a 4-tuple: (left, top, right, bottom).

    Accepts: int, (x,y), or (l,t,r,b).
    """
    if isinstance(p, (tuple, list)):
        if len(p) == 2:
            lx = int(p[0]); ty = int(p[1]); return (lx, ty, lx, ty)
        if len(p) == 4:
            return tuple(int(x) for x in p)
        lx = int(p[0]); ty = int(p[1]) if len(p) > 1 else lx; return (lx, ty, lx, ty)
    v = int(p); return (v, v, v, v)
    v=int(p);return(v,v,v,v)

# draw textured rect (pixel coords top-left origin)
def draw_rect(program, tex, x, y, w, h, sw, sh, alpha=1.0):
    """Draw a textured quad in pixel coordinates (top-left origin).

    program: GLSL program to use (expects a_pos, a_uv, u_texture, u_alpha)
    tex: GLTex or texture-like with .bind() and w/h
    (x,y,w,h): rectangle in pixels (top-left origin)
    (sw,sh): screen width/height in pixels
    alpha: multiply texture alpha
    """
    x0 = (x / sw) * 2 - 1; x1 = ((x + w) / sw) * 2 - 1; y0 = 1 - (y / sh) * 2; y1 = 1 - ((y + h) / sh) * 2
    verts = (ctypes.c_float * 16)(x0, y1, 0.0, 0.0, x1, y1, 1.0, 0.0, x1, y0, 1.0, 1.0, x0, y0, 0.0, 1.0)
    stride = ctypes.sizeof(ctypes.c_float) * 4
    glUseProgram(program)
    a = glGetUniformLocation(program, 'u_alpha')
    if a != -1:
        glUniform1f(a, alpha)
    t = glGetUniformLocation(program, 'u_texture')
    if t != -1:
        glUniform1i(t, 0)
    pos = glGetAttribLocation(program, 'a_pos'); uv = glGetAttribLocation(program, 'a_uv')
    if pos != -1:
        glEnableVertexAttribArray(pos); glVertexAttribPointer(pos, 2, GL_FLOAT, GL_FALSE, stride, ctypes.cast(verts, ctypes.c_void_p))
    if uv != -1:
        glEnableVertexAttribArray(uv); glVertexAttribPointer(uv, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(ctypes.addressof(verts) + 8))
    glActiveTexture(GL_TEXTURE0); tex.bind(); glEnable(GL_TEXTURE_2D); glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glDrawArrays(GL_TRIANGLE_FAN, 0, 4)
    glDisable(GL_BLEND); glDisable(GL_TEXTURE_2D)
    if pos != -1: glDisableVertexAttribArray(pos)
    if uv != -1: glDisableVertexAttribArray(uv)
    glUseProgram(0)

class Message:
    def __init__(self,text,pos,font,fg,bg,fade_in,vis,fade_out,box_op,blur_r,padding,pad_top,pad_bottom,pad_left,pad_right,max_w,shared_blur):
        self.text=text;self.x,self.y=pos;self.font=font;self.fg=fg;self.bg=bg
        self.fade_in, self.visible, self.fade_out=map(float,(fade_in,vis,fade_out));self.box_op=float(box_op);self.blur=int(blur_r)
        l,t,r,b=norm_pad(padding);
        if pad_top is not None: t=int(pad_top)
        if pad_bottom is not None: b=int(pad_bottom)
        if pad_left is not None: l=int(pad_left)
        if pad_right is not None: r=int(pad_right)
        self.pl,self.pt,self.pr,self.pb = l,t,r,b
        self.max_w=max_w;self.start=time.time();self.finished=False
        self._make_text();self.bg_fbo=FBO(max(1,self.w),max(1,self.h));self.tmp=FBO(self.bg_fbo.w,self.bg_fbo.h);self._render_bg();self._blur_shared(shared_blur);
    def _wrap(self,txt,max_w):
        words=txt.split();
        if not words: return ['']
        lines=[];cur=words[0]
        for w in words[1:]:
            test=cur+' '+w;
            if self.font.size(test)[0] <= max_w: cur=test
            else: lines.append(cur); cur=w
        lines.append(cur);return lines
    def _make_text(self):
        sw,sh=WINDOW_SIZE
        max_w=self.max_w if self.max_w is not None else max(20,sw-int(self.x)-(self.pl+self.pr)-10)
        lines=self._wrap(self.text,max_w)
        w=0;h=0
        for line in lines:
            lw,lh=self.font.size(line);w=max(w,lw);h+=lh
        self.w=w+self.pl+self.pr;self.h=h+self.pt+self.pb
        surf=pygame.Surface((self.w,self.h),SRCALPHA,32).convert_alpha();surf.fill((0,0,0,0))
        y=self.pt
        for line in lines:
            surf.blit(self.font.render(line,True,self.fg),(self.pl,y));y+=self.font.get_linesize()
        self.tex=GLTex(self.w,self.h,surf);surf=None
    def _render_bg(self):
        self.bg_fbo.bind();glClearColor(0,0,0,0);glClear(GL_COLOR_BUFFER_BIT)
        glMatrixMode(GL_PROJECTION);glPushMatrix();glLoadIdentity();glOrtho(0,self.bg_fbo.w,self.bg_fbo.h,0,-1,1)
        glMatrixMode(GL_MODELVIEW);glPushMatrix();glLoadIdentity()
        r,g,b=[c/255.0 for c in self.bg];glColor4f(r,g,b,1.0);glDisable(GL_TEXTURE_2D);glBegin(GL_QUADS);glVertex2f(0,0);glVertex2f(self.bg_fbo.w,0);glVertex2f(self.bg_fbo.w,self.bg_fbo.h);glVertex2f(0,self.bg_fbo.h);glEnd()
        glPopMatrix();glMatrixMode(GL_PROJECTION);glPopMatrix();glMatrixMode(GL_MODELVIEW);self.bg_fbo.unbind(*WINDOW_SIZE)
    def _blur_shared(self,prog):
        iters=max(1,self.blur//4)
        for i in range(iters):
            # horiz
            self._run_blur(prog,self.bg_fbo.tex,self.tmp,1,0)
            # vert
            self._run_blur(prog,self.tmp.tex,self.bg_fbo,0,1)
    def _run_blur(self,prog,src_tex,target_fbo,dx,dy):
        target_fbo.bind();glClearColor(0,0,0,0);glClear(GL_COLOR_BUFFER_BIT);glUseProgram(prog)
        u=glGetUniformLocation(prog,'u_texture');
        if u!=-1:glUniform1i(u,0)
        t=glGetUniformLocation(prog,'u_texelSize');
        if t!=-1:glUniform2f(t,1.0/float(src_tex.w),1.0/float(src_tex.h))
        d=glGetUniformLocation(prog,'u_dir');
        if d!=-1:glUniform2f(d,float(dx),float(dy))
        draw_rect(prog,src_tex,0,0,target_fbo.w,target_fbo.h,target_fbo.w,target_fbo.h,alpha=1.0)
        glUseProgram(0);target_fbo.unbind(*WINDOW_SIZE)
    def update_and_draw(self,now,draw_prog,sw,sh):
        e=now-self.start;total=self.fade_in+self.visible+self.fade_out
        if e>=total:self.finished=True;return
        if e<self.fade_in:fa=e/self.fade_in
        elif e<self.fade_in+self.visible:fa=1.0
        else:fa=1.0-((e-self.fade_in-self.visible)/self.fade_out)
        fa=max(0.0,min(1.0,fa))
        draw_rect(draw_prog,self.bg_fbo.tex,int(self.x),int(self.y),int(self.w),int(self.h),sw,sh,alpha=self.box_op*fa)
        draw_rect(draw_prog,self.tex,int(self.x),int(self.y),int(self.w),int(self.h),sw,sh,alpha=fa)
    def delete(self):
        self.tex.delete();self.bg_fbo.delete();self.tmp.delete()

def ortho(w, h):
    """Set up an orthographic projection where origin is top-left (pixel coordinates)."""
    glViewport(0, 0, w, h); glMatrixMode(GL_PROJECTION); glLoadIdentity(); glOrtho(0, w, h, 0, -1, 1); glMatrixMode(GL_MODELVIEW); glLoadIdentity();glOrtho(0,w,h,0,-1,1);glMatrixMode(GL_MODELVIEW);glLoadIdentity()

def setup():
    """Initialize pygame/OpenGL, compile shared shaders and return (screen, clock, base_font, blur_prog, draw_prog)."""
    pygame.init()
    pygame.display.set_caption('Message Rect - lean')
    screen = pygame.display.set_mode(WINDOW_SIZE, DOUBLEBUF | OPENGL)
    clock = pygame.time.Clock()
    ortho(*WINDOW_SIZE)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    base_font = None
    blur_prog = link(VS, FS_BLUR)
    draw_prog = link(VS, FS_DRAW)
    return screen, clock, base_font, blur_prog, draw_prog


def make_spawner(msgs, base_font, blur_prog):
    """Return a spawn_message function that appends Message instances to msgs.

    The returned function signature matches previous spawn(...) but is named spawn_message.
    """
    def spawn_message(text, pos=(20,20), fg=(255,255,255), bg=(30,30,30),
                      font=None, font_path=None, font_size=None,
                      fade_in=4.0, visible=3.0, fade_out=4.0,
                      box_opacity=1.0, blur_radius=6, padding=PADDING,
                      pad_top=None, pad_bottom=None, pad_left=None, pad_right=None, max_width=None):
        """Spawn a new message with the provided parameters and add it to msgs."""
        size = int(font_size) if font_size is not None else DEFAULT_FONT_SIZE
        if font is None:
            if font_path is not None:
                try:
                    font = pygame.font.Font(font_path, size)
                except Exception as e:
                    print('Font load failed, using system font', e)
                    font = pygame.font.SysFont(None, size)
            else:
                font = pygame.font.SysFont(None, size)
        m = Message(text, pos, font, fg, bg, fade_in, visible, fade_out,
                    box_opacity, blur_radius, padding, pad_top, pad_bottom, pad_left, pad_right, max_width, blur_prog)
        msgs.append(m)
        return m
    return spawn_message


def run_loop(clock, msgs, spawn_fn, draw_prog):
    """Main event & render loop. spawn_fn is the function used to create messages."""
    spawn_fn('Press E to spawn a message with a faint blurred box (box_opacity=0.9).', pos=(40,30), font_size=22, box_opacity=0.9, blur_radius=8, fade_in=1.0, visible=5.0, fade_out=1.0)
    running = True
    while running:
        now = time.time()
        for ev in pygame.event.get():
            if ev.type == QUIT:
                running = False
            elif ev.type == KEYDOWN:
                if ev.key == K_ESCAPE:
                    running = False
                elif ev.key == K_e:
                    y = 100 + (len(msgs) % 6) * 64
                    spawn_fn('Example: blurred background ONLY, text remains crisp. box_opacity=0.9, blur_radius=10.', pos=(60,y), font_size=20, box_opacity=0.9, blur_radius=10, fade_in=4.0, visible=3.0, fade_out=4.0, max_width=540)
        # cleanup finished
        keep = []
        for m in msgs:
            if m.finished:
                m.delete()
            else:
                keep.append(m)
        msgs[:] = keep
        glClearColor(0.08,0.08,0.08,1.0); glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        for m in msgs:
            m.update_and_draw(now, draw_prog, WINDOW_SIZE[0], WINDOW_SIZE[1])
        pygame.display.flip(); clock.tick(60)


def cleanup(msgs, blur_prog, draw_prog):
    """Free GPU resources, delete messages and terminate pygame."""
    for m in msgs:
        m.delete()
    try:
        glDeleteProgram(blur_prog); glDeleteProgram(draw_prog)
    except:
        pass
    pygame.quit()


def main():
    """Program entry: setup, create spawner, run main loop, cleanup on exit."""
    screen, clock, base_font, blur_prog, draw_prog = setup()
    msgs = []
    spawn_message = make_spawner(msgs, base_font, blur_prog)
    try:
        run_loop(clock, msgs, spawn_message, draw_prog)
    finally:
        cleanup(msgs, blur_prog, draw_prog)


if __name__=='__main__':
    main()
