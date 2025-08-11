#date: 2025-08-11T17:01:35Z
#url: https://api.github.com/gists/5f4fe3fcd88f9170204a595a712f79fe
#owner: https://api.github.com/users/n1clud3

#!/usr/bin/env python
# SCP - Containment Breach RoomMesh (.rmesh) to OBJ converter
# Vibecoded by n1clude and GPT-5
# Usage: `./rmesh_convert.py <path_to_rmesh>`
# The output (.obj and .mtl) is stored on the same path as .rmesh file.

import struct
from pathlib import Path
import sys

def read_byte(f):
    return struct.unpack('<B', f.read(1))[0]

def read_int(f):
    return struct.unpack('<i', f.read(4))[0]

def read_float(f):
    return struct.unpack('<f', f.read(4))[0]

def read_string(f):
    length = read_int(f)
    if length <= 0:
        return ""
    return f.read(length).decode('utf-8')

def load_rmesh_to_obj(rmesh_path, obj_path):
    rmesh_path = Path(rmesh_path)
    obj_path = Path(obj_path)
    mtl_path = obj_path.with_suffix(".mtl")

    vertices = []
    uvs_layer0 = []
    uvs_layer1 = []
    faces = []
    materials = {}  # mat_name -> (tex0, tex1)
    face_materials = []

    with open(rmesh_path, "rb") as f:
        sig = read_string(f)
        if sig not in ("RoomMesh", "RoomMesh.HasTriggerBox"):
            raise ValueError(f"{rmesh_path} is not RMESH ({sig})")

        mesh_count = read_int(f)

        for mesh_index in range(mesh_count):
            tex_files = [None, None]
            is_alpha = 0
            for j in range(2):
                temp1i = read_byte(f)
                if temp1i != 0:
                    tex_files[j] = read_string(f)
                    if temp1i == 3:
                        is_alpha = 1
                    else:
                        is_alpha = 2

            mat_name = f"mesh{mesh_index}"
            materials[mat_name] = (tex_files[1], tex_files[0])

            vert_count = read_int(f)
            vert_indices = []
            for _ in range(vert_count):
                x = read_float(f)
                y = read_float(f)
                z = read_float(f)
                vertices.append((x, y, z))
                vert_indices.append(len(vertices))  # OBJ is 1-based

                u0 = read_float(f)
                v0 = read_float(f)
                uvs_layer0.append((u0, 1.0 - v0))

                u1 = read_float(f)
                v1 = read_float(f)
                uvs_layer1.append((u1, 1.0 - v1))

                # Skip color bytes
                read_byte(f)
                read_byte(f)
                read_byte(f)

            tri_count = read_int(f)
            for _ in range(tri_count):
                i1 = read_int(f)
                i2 = read_int(f)
                i3 = read_int(f)
                faces.append((vert_indices[i1], vert_indices[i2], vert_indices[i3]))
                face_materials.append(mat_name)

    # Write MTL
    with open(mtl_path, "w") as mtl:
        mtl.write("# Exported from RMESH with dual textures\n")
        for name, (tex0, tex1) in materials.items():
            mtl.write(f"newmtl {name}\n")
            mtl.write("Kd 1.000 1.000 1.000\n")
            mtl.write("Ka 1.000 1.000 1.000\n")
            mtl.write("Ks 0.000 0.000 0.000\n")
            if tex0:
                mtl.write(f"map_Kd {tex0}\n")
            if tex1:
                mtl.write(f"map_Ka {tex1}\n")  # Ambient map for lightmap
            mtl.write("\n")

    # Write OBJ
    with open(obj_path, "w") as out:
        out.write("# Converted from RMESH\n")
        out.write(f"mtllib {mtl_path.name}\n")
        for v in vertices:
            out.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        # Layer 0 UVs
        for uv in uvs_layer0:
            out.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")

        current_mat = None
        for face, mat in zip(faces, face_materials):
            if mat != current_mat:
                out.write(f"usemtl {mat}\n")
                current_mat = mat
            out.write(f"f {face[0]}/{face[0]} {face[2]}/{face[2]} {face[1]}/{face[1]}\n")

    print(f"Saved OBJ: {obj_path}")
    print(f"Saved MTL: {mtl_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("SCP - Containment Breach RoomMesh to OBJ converter by n1clude")
        print(f"Usage: {sys.argv[0]} <path_to_rmesh1> [path_to_rmesh2 ..]")
        print("The output (.obj and .mtl) is stored on the same path as .rmesh file")
        sys.exit(1)
    
    files = sys.argv
    files.pop(0)

    for rmesh in files:
        obj = rmesh.split(".")
        obj.pop()
        obj.append("obj")
        obj = '.'.join(obj)
        load_rmesh_to_obj(rmesh, obj)
