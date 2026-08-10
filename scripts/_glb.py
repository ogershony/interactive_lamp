"""
Minimal binary-glTF writer.

Only what the lamp needs: indexed triangle meshes with POSITION (+ optional
NORMAL), one flat-colour PBR material each, one node per part, no transforms
(the exporter bakes rest-pose world coordinates into the vertices and the
browser reparents nodes by name into the joint pivots).

Hand-rolled rather than pulling in trimesh/pygltflib for ~150 lines -- the
project has neither, and the write path is fully specified by the spec's
table of component types.
"""

import json
import struct

FLOAT, USHORT, UINT = 5126, 5123, 5125
ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER = 34962, 34963


def _pad4(b, fill=b"\0"):
    return b + fill * ((4 - len(b) % 4) % 4)


class GlbBuilder:
    def __init__(self):
        self.bin = bytearray()
        self.views = []
        self.accessors = []
        self.meshes = []
        self.nodes = []
        self.materials = []

    def _view(self, data, target):
        # accessor byteOffset must be a multiple of the component size; keeping
        # every view 4-aligned satisfies that for all types used here
        while len(self.bin) % 4:
            self.bin.append(0)
        off = len(self.bin)
        self.bin.extend(data)
        self.views.append(
            {"buffer": 0, "byteOffset": off, "byteLength": len(data),
             "target": target})
        return len(self.views) - 1

    def _accessor(self, view, ctype, count, atype, mn=None, mx=None):
        acc = {"bufferView": view, "componentType": ctype, "count": count,
               "type": atype}
        if mn is not None:
            acc["min"], acc["max"] = mn, mx
        self.accessors.append(acc)
        return len(self.accessors) - 1

    def material(self, rgba, name=None):
        r, g, b, a = (float(x) for x in rgba)
        mat = {
            "pbrMetallicRoughness": {
                "baseColorFactor": [r, g, b, a],
                "metallicFactor": 0.15,
                "roughnessFactor": 0.75,
            },
            "doubleSided": False,
        }
        if a < 1.0:
            mat["alphaMode"] = "BLEND"
        if name:
            mat["name"] = name
        self.materials.append(mat)
        return len(self.materials) - 1

    def add_mesh(self, name, verts, faces, material, normals=None):
        """verts (V,3) float32, faces (F,3) int, normals (V,3) or None."""
        import numpy as np

        verts = np.ascontiguousarray(verts, dtype="<f4")
        faces = np.asarray(faces)
        v_view = self._view(verts.tobytes(), ARRAY_BUFFER)
        v_acc = self._accessor(
            v_view, FLOAT, len(verts), "VEC3",
            verts.min(axis=0).tolist(), verts.max(axis=0).tolist())
        attrs = {"POSITION": v_acc}

        if normals is not None:
            nrm = np.ascontiguousarray(normals, dtype="<f4")
            n_view = self._view(nrm.tobytes(), ARRAY_BUFFER)
            attrs["NORMAL"] = self._accessor(n_view, FLOAT, len(nrm), "VEC3")

        small = len(verts) < 65536
        idx = np.ascontiguousarray(faces.reshape(-1),
                                   dtype="<u2" if small else "<u4")
        i_view = self._view(idx.tobytes(), ELEMENT_ARRAY_BUFFER)
        i_acc = self._accessor(i_view, USHORT if small else UINT,
                               len(idx), "SCALAR")

        self.meshes.append({
            "name": name,
            "primitives": [{"attributes": attrs, "indices": i_acc,
                            "material": material}],
        })
        self.nodes.append({"name": name, "mesh": len(self.meshes) - 1})
        return len(self.nodes) - 1

    def serialize(self):
        gltf = {
            "asset": {"version": "2.0", "generator": "interactive_lamp/export_web_assets"},
            "scene": 0,
            "scenes": [{"nodes": list(range(len(self.nodes)))}],
            "nodes": self.nodes,
            "meshes": self.meshes,
            "materials": self.materials,
            "accessors": self.accessors,
            "bufferViews": self.views,
            "buffers": [{"byteLength": len(self.bin)}],
        }
        js = _pad4(json.dumps(gltf, separators=(",", ":")).encode(), b" ")
        bn = _pad4(bytes(self.bin))
        total = 12 + 8 + len(js) + 8 + len(bn)
        out = bytearray()
        out += struct.pack("<III", 0x46546C67, 2, total)
        out += struct.pack("<II", len(js), 0x4E4F534A) + js
        out += struct.pack("<II", len(bn), 0x004E4942) + bn
        return bytes(out)

    def write(self, path):
        data = self.serialize()
        path.write_bytes(data)
        return len(data)
