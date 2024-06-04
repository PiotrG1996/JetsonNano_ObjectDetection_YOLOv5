import os
import sys
import bpy
import math
import random
import time
from mathutils import Euler, Color
from pathlib import Path


def randomly_rotate_object(obj_to_change):
    """Applies a random rotation to an object 
    """
    random_rot = (random.random() * 2 * math.pi, random.random() * 2 * math.pi, random.random() * 2 * math.pi)
    obj_to_change.rotation_euler = Euler(random_rot, 'XYZ')

def randomly_change_color(material_to_change):
    
    color = Color()
    hue = random.random() # random between 0 and 1
    color.hsv = (hue, 1, 1)
    rgba = [color.r, color.g, color.b, 1]
    material_to_change.node_tree.nodes['Principled BSDF'].inputs[0].default_value = rgba

# Create automatically 3D letters
"""Create it manually for now
"""

#bpy.data.curves.new(type="FONT", name="Font Curve").body = "A"
#font_obj = bpy.data.objects.new(name="Font Object", object_data=bpy.data.curves["Font Curve"])
#bpy.context.scene.collection.objects.link(font_obj)
#bpy.context.object.data.extrude = 0.12
#bpy.context.object.rotation_euler[0] = 1.5708

# Obj names to render
obj_names = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','R','S','T','U','W','X','Y','Z']
obj_count = len(obj_names)
# Number of images to generate
obj_renders_per_split = [('train', 70), ('val', 20), ('test', 10)]
output_path = Path('/Users/admin/Blender/Tutorial/ABC')
# Total number of renders
total_render_count = sum([obj_count * r[1] for r in obj_renders_per_split])

# Set all objects to be hidden during render process
for name in obj_names:
    bpy.context.scene.objects[name].hide_render = True

# Tracks the starting image index for each object loop
start_index = 0
start_time = time.time()
for split_name, renders_per_object in obj_renders_per_split:
    print(f'Starting split: {split_name} | Total renders: {renders_per_object * obj_count}')
    print("-------------------------------")
    
    for obj_name in obj_names:
        print(f'Starting object: {split_name}/{obj_name}')
        print("===================================")
        
        # Get next object and make it visable
        obj_to_render = bpy.context.scene.objects[obj_name]
        obj_to_render.hide_render = False
        
        for i in range(start_index, start_index + renders_per_object):
            
            # Change the object
            randomly_rotate_object(obj_to_render)
            randomly_change_color(obj_to_render.material_slots[0].material)
            
            # Log status
            print(f'Rendering image {i+1} of {total_render_count}')
            seconds_per_render = (time.time() - start_time) / (i + 1)
            seconds_remaining = seconds_per_render * (total_render_count - i - 1)
            print(f'Estimated time remaining: {time.strftime("%H:%M:%S", time.gmtime(seconds_remaining))}')
            
            # Update file path and render
            bpy.context.scene.render.filepath = str(output_path / split_name / obj_name / f'{str(i).zfill(6)}.png')
            bpy.ops.render.render(write_still=True)
            
        # Hide object once it's done
        obj_to_render.hide_render = True
             
        # Update the starting image index
        start_index += renders_per_object
    
# Set all objects to be visable in rendering
for name in obj_names:
    bpy.context.scene.objects[name].hide_render = False        

        
