import bpy
from mathutils import Vector
import sys

meshpath = sys.argv[5]
model_name = sys.argv[6]
if model_name[-4:] == '.obj':
	model_name = model_name[:-4]
meshpath = meshpath + '/%s.obj'%(model_name)

# import mesh
bpy.ops.import_scene.obj(filepath=meshpath)

# alias the old and new mesh
omesh = bpy.data.objects['Default_Mesh']
#omesh.rotation_euler[2]=0.0
nmesh = bpy.data.objects[model_name]

# transfer properties
# nmesh.data.materials.clear()
nmesh.scale = omesh.scale
nmesh.location = omesh.location
nmesh.rotation_euler = omesh.rotation_euler
# nmesh.data.materials.append(omesh.data.materials[0])
omesh.data = nmesh.data

# delete new mesh
bpy.ops.object.select_all(action='DESELECT')
#nmesh.select = True
#bpy.ops.object.delete()

# move the ground
verts = [vert.co for vert in bpy.data.meshes[model_name].vertices]
z_min = 3
for v in verts:
	if z_min > v[1]:
		z_min = v[1]

bpy.data.objects['Default'].location = Vector((0.0,0.0,z_min))

# render animation
bpy.context.scene.render.image_settings.file_format='PNG'
bpy.context.scene.render.filepath = sys.argv[7]
bpy.ops.render.render(use_viewport = True, write_still=True, animation=False)
