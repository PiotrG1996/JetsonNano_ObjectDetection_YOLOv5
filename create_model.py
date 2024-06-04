import math
import random

import bpy

# Clear existing mesh objects
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.object.select_by_type(type='MESH')
bpy.ops.object.delete()

# Set render output directory
output_directory = "/Users/admin/Blender/Tutorial/"

# Create a camera
bpy.ops.object.camera_add(location=(0, 0, 10))  # Set Z-axis as up
camera = bpy.context.object
bpy.context.scene.camera = camera

# Set up lighting
bpy.ops.object.light_add(type='SUN', location=(
    10, 10, 10))  # Adjust light location

# Load the .stl file
stl_path = "/Users/admin/Blender/Tutorial/Cola.stl"
bpy.ops.import_mesh.stl(filepath=stl_path)

# Find the mesh object in the scene
cola_obj = None
for obj in bpy.context.scene.objects:
    if obj.type == 'MESH' and obj.data:
        cola_obj = obj
        break

if cola_obj:
    # Create a new material
    material = bpy.data.materials.new(name="ObjectMaterial")
    cola_obj.data.materials.append(material)

    # Assign the material to all faces of the mesh
    for face in cola_obj.data.polygons:
        face.material_index = 0

    # Create a shader node tree for the material
    material.use_nodes = True
    material.node_tree.links.clear()

    # Create a texture node and load the image texture
    texture_node = material.node_tree.nodes.new(type='ShaderNodeTexImage')
    # Replace with the path to your texture image
    image_path = "/Users/admin/Blender/Tutorial/texture.png"
    image_texture = bpy.data.images.load(image_path)
    texture_node.image = image_texture

    # Create a principled shader node and connect it to the material output
    principled_shader = material.node_tree.nodes.new(
        type='ShaderNodeBsdfPrincipled')
    material.node_tree.links.new(
        principled_shader.outputs['BSDF'], material.node_tree.nodes['Material Output'].inputs['Surface'])

    # Create a UV map layer if not already present
    if not cola_obj.data.uv_layers.active:
        cola_obj.data.uv_layers.new()

    # Create a UV map node and connect it to the texture node
    uv_map_node = material.node_tree.nodes.new(type='ShaderNodeUVMap')
    uv_map_node.uv_map = cola_obj.data.uv_layers.active.name
    material.node_tree.links.new(
        uv_map_node.outputs['UV'], texture_node.inputs['Vector'])

    # Connect the texture color output to the principled shader base color input
    material.node_tree.links.new(
        texture_node.outputs['Color'], principled_shader.inputs['Base Color'])

    # Adjust material settings for better color representation
    material.blend_method = 'OPAQUE'
    material.shadow_method = 'NONE'
    material.use_backface_culling = True

    # Set the number of images and camera angles
    num_images = 20  # Limit to 20 images per scene
    angle_increment = 360.0 / num_images

    # Set the camera zoom factor
    zoom_factor = 0.1  # Adjust the zoom factor as needed

    # Create random scenes with variations
    for scene_index in range(5):  # Generate 5 random scenes
        # Randomly transform the mesh object
        cola_obj.location.x = random.uniform(-2, 2)
        cola_obj.location.y = random.uniform(-2, 2)
        cola_obj.rotation_euler = (random.uniform(
            0, 2*math.pi), random.uniform(0, 2*math.pi), random.uniform(0, 2*math.pi))
        cola_obj.scale.x = random.uniform(0.8, 1.2)
        cola_obj.scale.y = random.uniform(0.8, 1.2)
        cola_obj.scale.z = random.uniform(0.8, 1.2)

        # Render images from different perspectives
        for i in range(num_images):
            # Position camera to center on the object and rotate around it
            camera.location.x = cola_obj.location.x
            camera.location.y = cola_obj.location.y
            camera.location.z = 10  # Set a fixed distance from the object
            camera.location.z *= zoom_factor  # Apply the zoom factor
            camera.rotation_euler = (0, 0, math.radians(i * angle_increment))

            # Set up rendering
            bpy.context.scene.render.resolution_x = 1920
            bpy.context.scene.render.resolution_y = 1080
            bpy.context.scene.render.image_settings.file_format = 'PNG'
            bpy.context.scene.render.image_settings.color_mode = 'RGBA'

            # Render the image
            bpy.context.scene.render.filepath = f"{output_directory}scene_{scene_index}_image_{i:03d}.png"
            bpy.ops.render.render(write_still=True)

    # Remove camera after rendering
    bpy.data.objects.remove(camera)
else:
    print("No valid mesh object found in the scene.")
