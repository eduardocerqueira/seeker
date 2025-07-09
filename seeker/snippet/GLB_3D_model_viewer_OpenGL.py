#date: 2025-07-09T16:46:59Z
#url: https://api.github.com/gists/11c6751b2318823358950f2f11fcbae5
#owner: https://api.github.com/users/EncodeTheCode

"""
Dependencies:
py -m pip install pygltflib
pip install pygltflib
"""

import pygame
import numpy as np
from pygame.locals import QUIT, KEYDOWN, KEYUP
from OpenGL.GL import *
from OpenGL.GLU import *
import sys
import io
from pygltflib import GLTF2, Image as GLTFImage

class Engine:
    def __init__(self, w, h,
                 fov=np.pi/4, near=0.1, far=100.0,
                 init_pos=None, init_yaw=-90.0, init_pitch=0.0):
        pygame.init()
        pygame.display.set_mode((w, h), pygame.DOUBLEBUF | pygame.OPENGL)
        pygame.display.set_caption("3D Renderer - GLB Embedded Textures")
        self.clock = pygame.time.Clock()
        self.w, self.h = w, h
        self.aspect = w / h

        # Camera
        self.pos = np.array(init_pos if init_pos is not None else [0.0, 5.0, 10.0], dtype=np.float32)
        self.yaw, self.pitch = init_yaw, init_pitch
        self._update_dir(); self.speed = 0.1; self.sensitivity = 0.1
        self.keys = {pygame.K_w: False, pygame.K_s: False,
                     pygame.K_a: False, pygame.K_d: False,
                     pygame.K_SPACE: False, pygame.K_LSHIFT: False}
        self.left_down = False
        pygame.mouse.set_visible(False); pygame.event.set_grab(True)

        # Projection
        glMatrixMode(GL_PROJECTION); glLoadIdentity()
        gluPerspective(np.degrees(fov), self.aspect, near, far)
        glMatrixMode(GL_MODELVIEW)

        # Enable
        glEnable(GL_DEPTH_TEST); glEnable(GL_TEXTURE_2D)
        glEnable(GL_LIGHTING); glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, (1,1,1,0))
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0.2,0.2,0.2,1))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.8,0.8,0.8,1))
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

        self.model_list = None

    def _update_dir(self):
        ry, rp = np.radians(self.yaw), np.radians(self.pitch)
        vec = np.array([np.cos(ry)*np.cos(rp), np.sin(rp), np.sin(ry)*np.cos(rp)], dtype=np.float32)
        self.dir = vec / np.linalg.norm(vec)

    def handle_input(self):
        for e in pygame.event.get():
            if e.type == QUIT: pygame.quit(); sys.exit()
            if e.type in (KEYDOWN, KEYUP):
                if e.key == pygame.K_ESCAPE: pygame.quit(); sys.exit()
                if e.key in self.keys: self.keys[e.key] = (e.type==KEYDOWN)
            if e.type == pygame.MOUSEBUTTONDOWN and e.button==1: self.left_down=True
            if e.type == pygame.MOUSEBUTTONUP and e.button==1: self.left_down=False
            if e.type == pygame.MOUSEMOTION and self.left_down:
                dx,dy = e.rel; self.yaw+=dx*self.sensitivity; self.pitch=np.clip(self.pitch-dy*self.sensitivity,-89,89)
                self._update_dir()

    def update_camera(self):
        fwd = self.dir*self.speed
        right = np.cross(self.dir, [0,1,0]); right=right/np.linalg.norm(right)*self.speed
        upv = np.array([0,1,0],dtype=np.float32)*self.speed
        if self.keys[pygame.K_w]: self.pos+=fwd
        if self.keys[pygame.K_s]: self.pos-=fwd
        if self.keys[pygame.K_a]: self.pos-=right
        if self.keys[pygame.K_d]: self.pos+=right
        if self.keys[pygame.K_SPACE]: self.pos+=upv
        if self.keys[pygame.K_LSHIFT]: self.pos-=upv

    def set_camera(self):
        tgt = self.pos + self.dir; glLoadIdentity(); gluLookAt(*self.pos, *tgt, 0,1,0)

    def load_texture_from_bytes(self, img_bytes):
        # Load image from memory bytes
        surf = pygame.image.load(io.BytesIO(img_bytes))
        data = pygame.image.tostring(surf, 'RGBA', True)
        w,h = surf.get_size()
        tex = glGenTextures(1); glBindTexture(GL_TEXTURE_2D, tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
        return tex

    def load_glb(self, file_path):
        gltf = GLTF2().load(file_path)
        bin_blob = gltf.binary_blob()
        mesh = gltf.meshes[0]; prim = mesh.primitives[0]
        # Read accessors
        def read(acc):
            bv = gltf.bufferViews[acc.bufferView]; off=bv.byteOffset or 0; raw=bin_blob[off:off+bv.byteLength]
            dt = {5126:np.float32,5123:np.uint16}.get(acc.componentType,np.uint32)
            arr = np.frombuffer(raw, dtype=dt)
            comps={'VEC3':3,'VEC2':2,'SCALAR':1}[acc.type]
            return arr.reshape(acc.count, comps)
        pos = read(gltf.accessors[prim.attributes.POSITION])
        norm = read(gltf.accessors[prim.attributes.NORMAL]) if hasattr(prim.attributes,'NORMAL') else None
        uv = read(gltf.accessors[prim.attributes.TEXCOORD_0]) if hasattr(prim.attributes,'TEXCOORD_0') else None
        idx = read(gltf.accessors[prim.indices]).flatten() if prim.indices is not None else np.arange(len(pos),dtype=np.uint32)
        # Embedded texture
        tex_id=None
        mat=gltf.materials[prim.material].pbrMetallicRoughness
        if mat and mat.baseColorTexture:
            tex_info = gltf.textures[mat.baseColorTexture.index]
            img:GLTFImage = gltf.images[tex_info.source]
            if img.bufferView is not None:
                bv = gltf.bufferViews[img.bufferView]; off=bv.byteOffset or 0
                img_bytes = bin_blob[off:off+bv.byteLength]
                tex_id=self.load_texture_from_bytes(img_bytes)
        # Build list
        self.model_list=glGenLists(1); glNewList(self.model_list,GL_COMPILE)
        if tex_id: glEnable(GL_TEXTURE_2D); glBindTexture(GL_TEXTURE_2D,tex_id)
        glBegin(GL_TRIANGLES)
        for i in idx:
            if norm is not None: glNormal3f(*norm[i])
            if uv is not None: u,v=uv[i]; glTexCoord2f(u,1-v)
            glVertex3f(*pos[i])
        glEnd();
        if tex_id: glDisable(GL_TEXTURE_2D)
        glEndList()

    def render(self):
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        self.set_camera(); glCallList(self.model_list)

    def run(self, path):
        if not path.lower().endswith('.glb'): raise ValueError(f"Unsupported: {path}")
        self.load_glb(path)
        while True: self.handle_input(); self.update_camera(); self.render(); pygame.display.flip(); self.clock.tick(60)

if __name__=='__main__':
    Engine(800,600,init_pos=[0,15,7.5]).run('3d_model_object.glb')
