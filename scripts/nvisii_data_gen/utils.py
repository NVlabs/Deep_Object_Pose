import colorsys
import glob
import math
import os
import random
import subprocess
import warnings

import cv2
import numpy as np
import nvisii as visii
import pybullet as p
import randomcolor
import simplejson as json
from PIL import Image
from pyquaternion import Quaternion


def add_random_obj(name = "name",
    x_lim = [-1,1],
    y_lim = [-1,1],
    z_lim = [-1,1],
    scale_lim = [0.01,1],
    obj_id = None,
    ):
    
    # obj = visii.entity.get(name)
    # if obj is None:
    obj= visii.entity.create(
        name = name,
        transform = visii.transform.create(name),
        material = visii.material.create(name)
    )
    if obj_id is None:
        obj_id = random.randint(0,15)

    mesh = None
    if obj_id == 0:
        if add_random_obj.create_sphere is None:
            add_random_obj.create_sphere = visii.mesh.create_sphere(name) 
        mesh = add_random_obj.create_sphere
    if obj_id == 1:
        if add_random_obj.create_torus_knot is None:
            add_random_obj.create_torus_knot = visii.mesh.create_torus_knot(name, 
                random.randint(2,6),
                random.randint(4,10))
        mesh = add_random_obj.create_torus_knot
    if obj_id == 2:
        if add_random_obj.create_teapotahedron is None:
            add_random_obj.create_teapotahedron = visii.mesh.create_teapotahedron(name) 
        mesh = add_random_obj.create_teapotahedron
    if obj_id == 3:
        if add_random_obj.create_box is None:
            add_random_obj.create_box = visii.mesh.create_box(name)
             
        mesh = add_random_obj.create_box
    if obj_id == 4:
        if add_random_obj.create_capped_cone is None:
            add_random_obj.create_capped_cone = visii.mesh.create_capped_cone(name) 
        mesh = add_random_obj.create_capped_cone
    if obj_id == 5:
        if add_random_obj.create_capped_cylinder is None:
            add_random_obj.create_capped_cylinder = visii.mesh.create_capped_cylinder(name) 
        mesh = add_random_obj.create_capped_cylinder
    if obj_id == 6:
        if add_random_obj.create_capsule is None:
            add_random_obj.create_capsule = visii.mesh.create_capsule(name) 
        mesh = add_random_obj.create_capsule
    if obj_id == 7:
        if add_random_obj.create_cylinder is None:
            add_random_obj.create_cylinder = visii.mesh.create_cylinder(name) 
        mesh = add_random_obj.create_cylinder
    if obj_id == 8:
        if add_random_obj.create_disk is None:
            add_random_obj.create_disk = visii.mesh.create_disk(name) 
        mesh = add_random_obj.create_disk
    if obj_id == 9:
        if add_random_obj.create_dodecahedron is None:
            add_random_obj.create_dodecahedron = visii.mesh.create_dodecahedron(name) 
        mesh = add_random_obj.create_dodecahedron
    if obj_id == 10:
        if add_random_obj.create_icosahedron is None:
            add_random_obj.create_icosahedron = visii.mesh.create_icosahedron(name) 
        mesh = add_random_obj.create_icosahedron
    if obj_id == 11:
        if add_random_obj.create_icosphere is None:
            add_random_obj.create_icosphere = visii.mesh.create_icosphere(name) 
        mesh = add_random_obj.create_icosphere
    if obj_id == 12:
        if add_random_obj.create_rounded_box is None:
            add_random_obj.create_rounded_box = visii.mesh.create_rounded_box(name) 
        mesh = add_random_obj.create_rounded_box
    if obj_id == 13:
        if add_random_obj.create_spring is None:
            add_random_obj.create_spring = visii.mesh.create_spring(name) 
        mesh = add_random_obj.create_spring
    if obj_id == 14:
        if add_random_obj.create_torus is None:
            add_random_obj.create_torus = visii.mesh.create_torus(name) 
        mesh = add_random_obj.create_torus
    if obj_id == 15:
        if add_random_obj.create_tube is None:
            add_random_obj.create_tube = visii.mesh.create_tube(name) 
        mesh = add_random_obj.create_tube
    if obj_id == 16:
        if add_random_obj.create_surface is None:
            add_random_obj.create_surface = visii.mesh.create_plane(name) 
        mesh = add_random_obj.create_tube

    obj.set_mesh(mesh)

    obj.get_transform().set_position(
        visii.vec3(
            random.uniform(x_lim[0],x_lim[1]),
            random.uniform(y_lim[0],y_lim[1]),
            random.uniform(z_lim[0],z_lim[1])        
        )
    )
    
    obj.get_transform().set_rotation(
        visii.quat(1.0 ,random.random(), random.random(), random.random()) 
    )    

    obj.get_transform().set_scale(
        visii.vec3(
            random.uniform(scale_lim[0],scale_lim[1])
        )
    )
    return obj
add_random_obj.rcolor = randomcolor.RandomColor()
add_random_obj.create_sphere = None
add_random_obj.create_torus_knot = None
add_random_obj.create_teapotahedron = None
add_random_obj.create_box = None
add_random_obj.create_capped_cone = None
add_random_obj.create_capped_cylinder = None
add_random_obj.create_capsule = None
add_random_obj.create_cylinder = None
add_random_obj.create_disk = None
add_random_obj.create_dodecahedron = None
add_random_obj.create_icosahedron = None
add_random_obj.create_icosphere = None
add_random_obj.create_rounded_box = None
add_random_obj.create_spring = None
add_random_obj.create_torus = None
add_random_obj.create_tube = None



def random_material(
    obj_id,
    color = None, # list of 3 numbers between [0..1]
    just_simple = False,
    ):

    if random_material.textures is None:
        textures = []

        path = 'content/materials_omniverse/'
        for folder in glob.glob(path + "/*/"):
            for folder_in in glob.glob(folder+"/*/"):
                name = folder_in.replace(folder,'').replace('/','').replace('\\', '')
                # print (folder_in)
                # print (name)
                # print (folder_in + "/" + name + "_BaseColor.png")
                if os.path.exists(folder_in + "/" + name + "_BaseColor.png"):
                    if os.path.exists(folder_in + "/" + name + "_Normal.png"):
                        normal = folder_in + "/" + name + "_Normal.png"
                    else:
                        normal = folder_in + "/" + name + "_N.png"

                    textures.append({
                        'base':folder_in + "/" + name + "_BaseColor.png",
                        'normal':folder_in + "/" + name + "_Normal.png",
                        'orm':folder_in + "/" + name + "_ORM.png",
                        })

            if 'glass' in folder.lower():
                #read the mtl 
                print("---")
                for mdl in glob.glob(folder + "/*.mdl"):    
                    
                    with open(mdl,'r') as f:
                        data = {'glass':1}
                        for line in f.readlines():
                            if "transmission_color" in line and not 'transmission_color_texture' in line:
                                text = line.replace("transmission_color","")\
                                    .replace(" ","").replace('color','').replace(":","")\
                                    .replace("f",'')
                                text = eval(text)[0]
                                # print(text)
                                data['color'] = text
                            if 'roughness' in line and not 'roughness_texture_influence' in line\
                                and not "frosting_roughness" in line and not 'roughness_texture' in line\
                                and not "reflection_roughness_constant" in line:
                                # print(line)
                                text = line.replace("roughness","")\
                                    .replace(" ","").replace('color','').replace(":","")\
                                    .replace("f",'').replace(',','')
                                # print(text)
                                text = eval(text)
                                data['roughness'] = text
                            if 'frosting_roughness' in line and not 'roughness_texture_influence' in line\
                                and not 'roughness_texture' in line\
                                and not "reflection_roughness_constant" in line:
                                # print(line)
                                text = line.replace("frosting_roughness","")\
                                    .replace(" ","").replace('color','').replace(":","")\
                                    .replace("f",'').replace(',','')
                                # print(text)
                                text = eval(text)

                                data['transmisive_roughness'] = text
                            if 'ior' in line and not 'stop' in line\
                                and not 'roughness_texture' in line\
                                and not "reflection_roughness_constant" in line:
                                text = line.replace("ior","")\
                                    .replace(" ","").replace('glass_ior','').replace(":","")\
                                    .replace("f",'').replace(',','').replace('glass','')\
                                    .replace("_",'')
                                text = eval(text)

                                data['ior'] = text
                            if 'specular_level' in line and not 'stop' in line\
                                and not 'roughness_texture' in line\
                                and not "reflection_roughness_constant" in line:
                                # print(text)
                                text = line.replace("specular_level","")\
                                    .replace(" ","").replace('glass_ior','').replace(":","")\
                                    .replace("f",'').replace(',','').replace('glass','')\
                                    .replace("_",'')
                                text = eval(text)
                                # print(text)
                                data['specular'] = text

                            if 'metallic_constant' in line and not 'stop' in line\
                                and not 'roughness_texture' in line\
                                and not "reflection_roughness_constant" in line:
                                text = line.replace("metallic_constant","")\
                                    .replace(" ","").replace('glass_ior','').replace(":","")\
                                    .replace("f",'').replace(',','').replace('glass','')\
                                    .replace("_",'')
                                text = eval(text)
                                data['metallic'] = text
                                
                        textures.append(data)
        random_material.textures = textures

    obj_mat = visii.material.get(str(obj_id))
    
    if color is None: 
        rgb = colorsys.hsv_to_rgb(
            random.uniform(0,1),
            random.uniform(0.1,1),
            random.uniform(0.1,1)
        )

        obj_mat.set_base_color(
            visii.vec3(
                rgb[0],
                rgb[1],
                rgb[2],
            )
        )          
    else:
        obj_mat.set_base_color(
            visii.vec3(
                color[0],color[1],color[2]
            )  
        )

    
    texture = random_material.textures[random.randint(0,len(random_material.textures)-1)]

    if random.uniform(0,1) < 0.25 or just_simple is True:
        r = random.randint(0,2)

        if r == 0:  
            # Plastic / mat
            obj_mat.set_metallic(0)  # should 0 or 1      
            obj_mat.set_transmission(0)  # should 0 or 1      
            obj_mat.set_roughness(random.uniform(0,1)) # default is 1  
        if r == 1:  
            # metallic
            obj_mat.set_metallic(random.uniform(0.9,1))  # should 0 or 1      
            obj_mat.set_transmission(0)  # should 0 or 1      
        if r == 2:  
            # glass
            obj_mat.set_metallic(0)  # should 0 or 1      
            obj_mat.set_transmission(random.uniform(0.9,1))  # should 0 or 1      

        if r > 0: # for metallic and glass
            r2 = random.randint(0,1)
            if r2 == 1: 
                obj_mat.set_roughness(random.uniform(0,.1)) # default is 1  
            else:
                obj_mat.set_roughness(random.uniform(0.9,1)) # default is 1  

        obj_mat.set_sheen(random.uniform(0,1))  # degault is 0     
        obj_mat.set_clearcoat(random.uniform(0,1))  # degault is 0     
        obj_mat.set_specular(random.uniform(0,1))  # degault is 0     

        r = random.randint(0,1)
        if r == 0:
            obj_mat.set_anisotropic(random.uniform(0,0.1))  # degault is 0     
        else:
            obj_mat.set_anisotropic(random.uniform(0.9,1))  # degault is 0     


    elif 'glass' in texture:
        obj_mat.set_transmission(1)
        if 'metallic' in texture.keys(): 
            obj_mat.set_transmission(0)
            obj_mat.set_metallic(texture['metallic'])
        if 'specular' in texture.keys():
            obj_mat.set_specular(texture['specular'])
        if 'transmisive_roughness' in texture.keys():
            obj_mat.set_transmission_roughness(texture['transmisive_roughness'])
        if 'roughness' in texture.keys(): 
            obj_mat.set_roughness(texture['roughness'])
        if 'color' in texture.keys():
            obj_mat.set_base_color(visii.vec3(
                texture['color'][0],
                texture['color'][1],
                texture['color'][2],
            ))

    else:
        # print(texture['base'])
        tex = visii.texture.get(texture['base'])
        tex_r = visii.texture.get(texture['base']+"_r")
        tex_m = visii.texture.get(texture['base']+"_m")
        tex_n = visii.texture.get(texture['base']+"_n")
        # tex_n = None
        
        if tex is None:
            tex = visii.texture.create_from_image(texture['base'],texture['base'])
            
            # load metallic and roughness 
            im = np.array(Image.open(texture['orm']))            
            
            im_r = np.concatenate(
                [
                    im[:,:,1].reshape(im.shape[0],im.shape[1],1),
                    im[:,:,1].reshape(im.shape[0],im.shape[1],1),
                    im[:,:,1].reshape(im.shape[0],im.shape[1],1),
                    im[:,:,1].reshape(im.shape[0],im.shape[1],1)
                ],
                2
            ) 
            # im_r = Image.fromarray(im_r)
            im_r = (im_r/255.0)

            tex_r = visii.texture.create_from_data(texture['base']+'_r',
                im_r.shape[0],
                im_r.shape[1],
                im_r.reshape(im_r.shape[0]*im_r.shape[1],4).astype(np.float32).flatten().tolist()
            )

            # im_r.save('tmp.png')
            # tex_r = visii.texture.create_from_image(texture['base']+'_r',"tmp.png")

            im_m = np.concatenate(
                [
                    im[:,:,2].reshape(im.shape[0],im.shape[1],1),
                    im[:,:,2].reshape(im.shape[0],im.shape[1],1),
                    im[:,:,2].reshape(im.shape[0],im.shape[1],1)
                ],
                2
            ) 
            im_m = Image.fromarray(im_m)
            im_m.save('tmp.png')

            tex_m = visii.texture.create_from_image(texture['base']+'_m',"tmp.png")

            if os.path.exists(texture['normal']):
                tex_n = visii.texture.create_from_image(
                    texture['base']+"_n",
                    texture['normal'],
                    linear=True
                )
            else:
                tex_n = None
        obj_mat.set_base_color_texture(tex)
        obj_mat.set_metallic_texture(tex_m)
        obj_mat.set_roughness_texture(tex_r)
        if not tex_n is None:
            obj_mat.set_normal_map_texture(tex_n)


####


random_material.textures = None

########################################
# ANIMATION RANDOMIZATION 
########################################



def distance(v0,v1=[0,0,0]):
    l2 = 0
    try:
        for i in range(len(v0)):
            l2 += (v0[i]-v1[i])**2
    except:
        for i in range(3):
            l2 += (v0[i]-v1[i])**2
    return math.sqrt(l2)

def normalize(v):
    l2 = distance(v)
    return [v[0]/l2,v[1]/l2,v[2]/l2]

def random_translation(obj_id,
    x_lim = [-1,1],
    y_lim = [-1,1],
    z_lim = [-1,1],
    speed_lim = [0.01,0.05],
    sample_method = None,
    motion_blur = False,
    strenght = 1,
    ):
    # return
    trans = visii.transform.get(str(obj_id))

    # Position    
    if not str(obj_id) in random_translation.destinations.keys() :
        if sample_method is None:
            random_translation.destinations[str(obj_id)] = [
                random.uniform(x_lim[0],x_lim[1]),
                random.uniform(y_lim[0],y_lim[1]),
                random.uniform(z_lim[0],z_lim[1])
            ]
        else:
            random_translation.destinations[str(obj_id)] = sample_method()
        random_translation.speeds[str(obj_id)] = random.uniform(speed_lim[0],speed_lim[1])
    else:
        goal = random_translation.destinations[str(obj_id)]
        pos = trans.get_position()

        if distance(goal,pos) < min(speed_lim)*2:
            if sample_method is None:
                random_translation.destinations[str(obj_id)] = [
                    random.uniform(x_lim[0],x_lim[1]),
                    random.uniform(y_lim[0],y_lim[1]),
                    random.uniform(z_lim[0],z_lim[1])
                ]
            else:
                random_translation.destinations[str(obj_id)] = sample_method()            

            random_translation.speeds[str(obj_id)] = random.uniform(speed_lim[0],speed_lim[1])    
            goal = random_translation.destinations[str(obj_id)]

        dir_vec = normalize(
            [
                goal[0] - pos[0],
                goal[1] - pos[1],
                goal[2] - pos[2]
            ]   
        )
        
        trans.add_position(
            visii.vec3(
                dir_vec[0] * random_translation.speeds[str(obj_id)],
                dir_vec[1] * random_translation.speeds[str(obj_id)],
                dir_vec[2] * random_translation.speeds[str(obj_id)]
            )
        )
        if motion_blur:
            trans.set_linear_velocity(
                visii.vec3(

                    dir_vec[0] * random_translation.speeds[str(obj_id)] * strenght,
                    dir_vec[1] * random_translation.speeds[str(obj_id)] * strenght,
                    dir_vec[2] * random_translation.speeds[str(obj_id)] * strenght
                )
            )

random_translation.destinations = {}
random_translation.speeds = {}


def circle_path(camera_distance = 0.5, camera_height=0.3):
    global opt 

    if circle_path.locations is None:
        circle_path.locations = []
        for i in range(0,210,10):
            circle_path.locations.append([ 
                math.cos(i*0.01745) * camera_distance,
                math.sin(i*0.01745) * camera_distance,
                camera_height
            ])
        circle_path.last = circle_path.locations[-1]
    if len(circle_path.locations) == 0:
        return circle_path.last 

    pos = circle_path.locations[0]
    circle_path.locations = circle_path.locations[1:]
    return pos

circle_path.locations = None

def random_sample_hemispher(
    max_radius = 2.5, 
    min_radius = 0.2
    ):
    if min_radius > max_radius:
        print('min bigger than max')
    outside = True
    while outside:

        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        z = random.uniform( 0, 0.95)
        if  (x**2 + y**2 + z**2) * max_radius < max_radius \
        and (x**2 + y**2 + z**2) * max_radius > min_radius:
            outside = False

    return [x*max_radius,y*max_radius,z*max_radius]



def random_rotation(obj_id,
    speed_lim = [0.01,0.05],
    motion_blur = False,
    strenght = 1,
    ):
    # return

    trans = visii.transform.get(str(obj_id))    

    # Rotation
    if not str(obj_id) in random_rotation.destinations.keys() :
        random_rotation.destinations[str(obj_id)] = Quaternion.random()
        random_rotation.speeds[str(obj_id)] = random.uniform(speed_lim[0],speed_lim[1])

    else:
        goal = random_rotation.destinations[str(obj_id)]
        rot = trans.get_rotation()
        rot = Quaternion(w=rot.w,x=rot.x,y=rot.y,z=rot.z)
        if Quaternion.sym_distance(goal, rot) < 0.1:
            random_rotation.destinations[str(obj_id)] = Quaternion.random()    
            random_rotation.speeds[str(obj_id)] = random.uniform(speed_lim[0],speed_lim[1])
            goal = random_rotation.destinations[str(obj_id)]
        dir_vec = Quaternion.slerp(rot,goal,random_rotation.speeds[str(obj_id)])
        q = visii.quat()
        q.w,q.x,q.y,q.z = dir_vec.w,dir_vec.x,dir_vec.y,dir_vec.z
        trans.set_rotation(q)
        # if motion_blur: 
        #     dir_vec = Quaternion.slerp(rot,goal,random_rotation.speeds[str(obj_id)] * strenght)

        #     q.w,q.x,q.y,q.z = dir_vec.w,dir_vec.x,dir_vec.y,dir_vec.z
        #     trans.set_angular_velocity(q)

random_rotation.destinations = {}
random_rotation.speeds = {}

def random_scale(obj_id,
    scale_lim = [0.01,0.2],
    speed_lim = [0.01,0.02],
    x_lim = None,
    y_lim = None,
    z_lim = None,
    motion_blur = False,
    strenght = 1,
    ):
    # return
    # This assumes only one dimensions gets scale

    trans = visii.transform.get(str(obj_id))    

    limit = min(speed_lim)*2
    
    if not x_lim is None:
        limit = [min(y_lim)*2,min(x_lim)*2,min(z_lim)*2]
    
    if not str(obj_id) in random_scale.destinations.keys() :
        if not x_lim is None:
            random_scale.destinations[str(obj_id)] = [
                                        random.uniform(x_lim[0],x_lim[1]),
                                        random.uniform(y_lim[0],y_lim[1]),
                                        random.uniform(z_lim[0],z_lim[1])
                                        ]
        else:    
            random_scale.destinations[str(obj_id)] = random.uniform(scale_lim[0],scale_lim[1])

        random_scale.speeds[str(obj_id)] = random.uniform(speed_lim[0],speed_lim[1])

    else:
        goal = random_scale.destinations[str(obj_id)]

        if x_lim is None:
            current = trans.get_scale()[0]
        else:
            current = trans.get_scale()
        
        if x_lim is None:

            if abs(goal-current) < limit:
                random_scale.destinations[str(obj_id)] = random.uniform(scale_lim[0],scale_lim[1])
                random_scale.speeds[str(obj_id)] = random.uniform(speed_lim[0],speed_lim[1])
                goal = random_scale.destinations[str(obj_id)]
            if goal>current:
                q = random_scale.speeds[str(obj_id)]
            else:
                q = -random_scale.speeds[str(obj_id)]
            trans.set_scale(current + q)

        else:
            limits  = [x_lim,y_lim,z_lim]
            q = [0,0,0]
            for i in range(3):
                if abs(goal[i]-current[i]) < limit[i]:
                    random_scale.destinations[str(obj_id)][i] = random.uniform(limits[i][0],limits[i][1])
                    random_scale.speeds[str(obj_id)] = random.uniform(speed_lim[0],speed_lim[1])                    
                    goal = random_scale.destinations[str(obj_id)]
                if goal[i]>current[i]:
                    q[i] = random_scale.speeds[str(obj_id)]
                else:
                    q[i] = -random_scale.speeds[str(obj_id)]
            trans.set_scale(current + visii.vec3(q[0],q[1],q[2]))
        # if motion_blur:
        #     trans.set_scalar_velocity(visii.vec3(q[0]*strenght,q[1]*strenght,q[2]*strenght))

random_scale.destinations = {}
random_scale.speeds = {}


def random_color(obj_id,
    speed_lim = [0.01,0.1]
    ):

    # color
    if not str(obj_id) in random_color.destinations.keys() :
        c = eval(str(random_color.rcolor.generate(luminosity='bright',format_='rgb')[0])[3:])
        random_color.destinations[str(obj_id)] = visii.vec3(c[0]/255.0, c[1]/255.0, c[2]/255.0)
        random_color.speeds[str(obj_id)] = random.uniform(speed_lim[0],speed_lim[1])

    else:
        goal = random_color.destinations[str(obj_id)]
        current = visii.material.get(str(obj_id)).get_base_color()

        if distance(goal,current) < 0.1:
            random_color.speeds[str(obj_id)] = random.uniform(speed_lim[0],speed_lim[1])
            c = eval(str(random_color.rcolor.generate(luminosity='bright',format_='rgb')[0])[3:])
            random_color.destinations[str(obj_id)] = visii.vec3(c[0]/255.0, c[1]/255.0, c[2]/255.0)
            goal = random_color.destinations[str(obj_id)]

        target = visii.mix(current,goal,
            visii.vec3( random_color.speeds[str(obj_id)],
                        random_color.speeds[str(obj_id)],
                        random_color.speeds[str(obj_id)]
                        )
        ) 

        visii.material.get(str(obj_id)).set_base_color(target)

random_color.destinations = {}
random_color.speeds = {}
random_color.rcolor = randomcolor.RandomColor()

######## RANDOM LIGHTS ############

def random_light(obj_id,
    intensity_lim = [5000,10000],
    color = None,
    temperature_lim  = [10,10000],
    exposure_lim = [0,0],
    ):

    obj = visii.entity.get(str(obj_id))
    obj.set_light(visii.light.create(str(obj_id)))

    obj.get_light().set_intensity(random.uniform(intensity_lim[0],intensity_lim[1]))
    obj.get_light().set_exposure(random.uniform(exposure_lim[0],exposure_lim[1]))
    # obj.get_light().set_temperature(np.random.randint(100,9000))


    if not color is None:
        obj.get_material().set_base_color(color[0],color[1],color[2])  
        # c = eval(str(rcolor.generate(luminosity='bright',format_='rgb')[0])[3:])
        # obj.get_light().set_color(
        #     c[0]/255.0,
        #     c[1]/255.0,
        #     c[2]/255.0)  
       
    else:
        obj.get_light().set_temperature(random.uniform(temperature_lim[0],temperature_lim[1]))

def random_intensity(obj_id,
    intensity_lim = [5000,10000],
    speed_lim = [0.1,1]
    ):

    obj = visii.entity.get(str(obj_id)).get_light()

    if not str(obj_id) in random_intensity.destinations.keys() :
        random_intensity.destinations[str(obj_id)] = random.uniform(intensity_lim[0],intensity_lim[1])
        random_intensity.speeds[str(obj_id)] = random.uniform(speed_lim[0],speed_lim[1])
        random_intensity.current[str(obj_id)] = random_intensity.destinations[str(obj_id)]
        obj.set_intensity(random_intensity.current[str(obj_id)]) 
    else:
        goal = random_intensity.destinations[str(obj_id)]
        current = random_intensity.current[str(obj_id)]
        
        if abs(goal-current) < min(speed_lim)*2:
            random_intensity.destinations[str(obj_id)] = random.uniform(intensity_lim[0],intensity_lim[1])
            random_intensity.speeds[str(obj_id)] = random.uniform(speed_lim[0],speed_lim[1])
            goal = random_intensity.destinations[str(obj_id)]
        if goal>current:
            q = random_intensity.speeds[str(obj_id)]
        else:
            q = -random_intensity.speeds[str(obj_id)]
        obj.set_intensity(random_intensity.current[str(obj_id)] + q)
        random_intensity.current[str(obj_id)] = random_intensity.current[str(obj_id)] + q

random_intensity.destinations = {}
random_intensity.current = {}
random_intensity.speeds = {}


def random_texture_material(entity):
    # select a random texture
    textures = random_texture_material.textures
    if len(textures) == 0: 
        "load the textures"
        path = 'content/materials_omniverse/'
        for folder in glob.glob(path + "/*/"):
            for folder_in in glob.glob(folder+"/*/"):
                name = folder_in.replace(folder,'').replace('/','').replace('\\', '')
                # print (folder_in)
                # print (name)
                # print (folder_in + "/" + name + "_BaseColor.png")
                if os.path.exists(folder_in + "/" + name + "_BaseColor.png"):
                    if os.path.exists(folder_in + "/" + name + "_Normal.png"):
                        normal = folder_in + "/" + name + "_Normal.png"
                    else:
                        normal = folder_in + "/" + name + "_N.png"

                    textures.append({
                        'base':folder_in + "/" + name + "_BaseColor.png",
                        'normal':folder_in + "/" + name + "_Normal.png",
                        'orm':folder_in + "/" + name + "_ORM.png",
                        })

            if 'glass' in folder.lower():
                #read the mtl 
                # print("---")
                for mdl in glob.glob(folder + "/*.mdl"):    
                    
                    with open(mdl,'r') as f:
                        data = {'glass':1}
                        for line in f.readlines():
                            if "transmission_color" in line and not 'transmission_color_texture' in line:
                                text = line.replace("transmission_color","")\
                                    .replace(" ","").replace('color','').replace(":","")\
                                    .replace("f",'')
                                text = eval(text)[0]
                                # print(text)
                                data['color'] = text
                            if 'roughness' in line and not 'roughness_texture_influence' in line\
                                and not "frosting_roughness" in line and not 'roughness_texture' in line\
                                and not "reflection_roughness_constant" in line:
                                # print(line)
                                text = line.replace("roughness","")\
                                    .replace(" ","").replace('color','').replace(":","")\
                                    .replace("f",'').replace(',','')
                                # print(text)
                                text = eval(text)
                                data['roughness'] = text
                            if 'frosting_roughness' in line and not 'roughness_texture_influence' in line\
                                and not 'roughness_texture' in line\
                                and not "reflection_roughness_constant" in line:
                                # print(line)
                                text = line.replace("frosting_roughness","")\
                                    .replace(" ","").replace('color','').replace(":","")\
                                    .replace("f",'').replace(',','')
                                # print(text)
                                text = eval(text)

                                data['transmisive_roughness'] = text
                            if 'ior' in line and not 'stop' in line\
                                and not 'roughness_texture' in line\
                                and not "reflection_roughness_constant" in line:
                                text = line.replace("ior","")\
                                    .replace(" ","").replace('glass_ior','').replace(":","")\
                                    .replace("f",'').replace(',','').replace('glass','')\
                                    .replace("_",'')
                                text = eval(text)

                                data['ior'] = text
                            if 'specular_level' in line and not 'stop' in line\
                                and not 'roughness_texture' in line\
                                and not "reflection_roughness_constant" in line:
                                # print(text)
                                text = line.replace("specular_level","")\
                                    .replace(" ","").replace('glass_ior','').replace(":","")\
                                    .replace("f",'').replace(',','').replace('glass','')\
                                    .replace("_",'')
                                text = eval(text)
                                # print(text)
                                data['specular'] = text

                            if 'metallic_constant' in line and not 'stop' in line\
                                and not 'roughness_texture' in line\
                                and not "reflection_roughness_constant" in line:
                                # print(text)
                                text = line.replace("metallic_constant","")\
                                    .replace(" ","").replace('glass_ior','').replace(":","")\
                                    .replace("f",'').replace(',','').replace('glass','')\
                                    .replace("_",'')
                                text = eval(text)
                                
                                data['metallic'] = text
                                
                        textures.append(data)



    texture = random_texture_material.textures[random.randint(0,len(textures)-1)]
    if random.uniform(0,1) < 0.25:
        random_material(entity)

    elif 'glass' in texture:

        mat = visii.material.get(entity)

        mat.set_transmission(1)
        if 'metallic' in texture.keys(): 
            mat.set_transmission(0)
            mat.set_metallic(texture['metallic'])
        if 'specular' in texture.keys():
            mat.set_specular(texture['specular'])
        if 'transmisive_roughness' in texture.keys():
            mat.set_transmission_roughness(texture['transmisive_roughness'])
        if 'roughness' in texture.keys(): 
            mat.set_roughness(texture['roughness'])
        if 'color' in texture.keys():
            mat.set_base_color(visii.vec3(
                texture['color'][0],
                texture['color'][1],
                texture['color'][2],
            ))

    else:
        print(texture['base'])
        tex = visii.texture.get(texture['base'])
        tex_r = visii.texture.get(texture['base']+"_r")
        tex_m = visii.texture.get(texture['base']+"_m")

        if tex is None:
            tex = visii.texture.create_from_image(texture['base'],texture['base'])
            
            # load metallic and roughness 
            im = np.array(Image.open(texture['orm']))            
            
            im_r = np.concatenate(
                [
                    im[:,:,1].reshape(im.shape[0],im.shape[1],1),
                    im[:,:,1].reshape(im.shape[0],im.shape[1],1),
                    im[:,:,1].reshape(im.shape[0],im.shape[1],1),
                    im[:,:,1].reshape(im.shape[0],im.shape[1],1)
                ],
                2
            ) 
            # im_r = Image.fromarray(im_r)
            im_r = (im_r/255.0)

            tex_r = visii.texture.create_from_data(texture['base']+'_r',
                im_r.shape[0],
                im_r.shape[1],
                im_r.reshape(im_r.shape[0]*im_r.shape[1],4).astype(np.float32).flatten().tolist()
            )

            # im_r.save('tmp.png')
            # tex_r = visii.texture.create_from_image(texture['base']+'_r',"tmp.png")

            im_m = np.concatenate(
                [
                    im[:,:,2].reshape(im.shape[0],im.shape[1],1),
                    im[:,:,2].reshape(im.shape[0],im.shape[1],1),
                    im[:,:,2].reshape(im.shape[0],im.shape[1],1)
                ],
                2
            ) 
            im_m = Image.fromarray(im_m)
            im_m.save('tmp.png')
            tex_m = visii.texture.create_from_image(texture['base']+'_m',"tmp.png")
            if os.path.exists(texture['normal']):
                tex_n = visii.texture.create_from_image(
                    texture['normal'],
                    texture['normal'],
                    linear=True
                )
            else:
                tex_n = None
        visii.material.get(entity).set_base_color_texture(tex)
        visii.material.get(entity).set_metallic_texture(tex_m)
        visii.material.get(entity).set_roughness_texture(tex_r)
        if not tex_n is None:
            visii.material.get(entity).set_normal_map_texture(tex_n)


random_texture_material.textures = []



######## NDDS ##########

def add_cuboid(name, scale=1, debug=False):
    """
    Computes the 3D bounding cuboid for object `name` and the centroid, adds the
    corresponding child transforms to the object (which will later be used in
    `get_cuboid_image_space`), and optionally adds small spheres as debug markers
    to the scene.

    The order of the indexes is the same as NVidia Deep learning Dataset Synthesizer (NDDS),
    nvdu_viz from NVidia Dataset Utilities and DOPE.

    The indexes of the 3D bounding cuboid are in the order shown in the sketch
    below (0..7), with the object being in its neutral orientation (X axis pointing
    forward, Y left, Z up).

       (m) 3 +-----------------+ 0 (b)
            /                 /|
           /                 / |
    (m) 2 +-----------------+ 1| (b)
          |                 |  |
          |       ^ z       |  |
          |       |         |  |
          |  y <--x         |  |
      (y) |                 |  + 4 (g)
          |                 | /
          |                 |/
    (y) 6 +-----------------+ 5 (g)

    Debug markers for the cuboid corners can be rendered using the `--debug` option,
    with (b) = blue, (m) = magenta, (g) = green, (y) = yellow and the centroid being
    white.
    """
    obj = visii.entity.get(name)

    min_obj = obj.get_mesh().get_min_aabb_corner()
    max_obj = obj.get_mesh().get_max_aabb_corner()
    centroid_obj = obj.get_mesh().get_aabb_center()

    cuboid = [
        visii.vec3(max_obj[0], min_obj[1], max_obj[2]),
        visii.vec3(min_obj[0], min_obj[1], max_obj[2]),
        visii.vec3(min_obj[0], max_obj[1], max_obj[2]),
        visii.vec3(max_obj[0], max_obj[1], max_obj[2]),
        visii.vec3(max_obj[0], min_obj[1], min_obj[2]),
        visii.vec3(min_obj[0], min_obj[1], min_obj[2]),
        visii.vec3(min_obj[0], max_obj[1], min_obj[2]),
        visii.vec3(max_obj[0], max_obj[1], min_obj[2]),
        visii.vec3(centroid_obj[0], centroid_obj[1], centroid_obj[2]),
    ]

    # same colors as nvdu_viz
    cuboid_colors = [
        visii.vec3(0.0, 0.0, 1.0),  # blue
        visii.vec3(0.0, 0.0, 1.0),  # blue
        visii.vec3(1.0, 0.0, 1.0),  # magenta
        visii.vec3(1.0, 0.0, 1.0),  # magenta
        visii.vec3(0.0, 1.0, 0.0),  # green
        visii.vec3(0.0, 1.0, 0.0),  # green
        visii.vec3(1.0, 1.0, 0.0),  # yellow
        visii.vec3(1.0, 1.0, 0.0),  # yellow
        visii.vec3(1.0, 1.0, 1.0),  # centroid: white
    ]

    for i_p, p in enumerate(cuboid):
        child_transform = visii.transform.create(f"{name}_cuboid_{i_p}")
        child_transform.set_position(p)

        # Attach the cuboids not directly to the object's transform, but to the symmetry_corrected
        # transform instead. That way, when the symmetry_corrected transform gets updated to the
        # "canonical" orientation (i.e., the symmetry transformation with the best alignment),
        # the cuboid corners get rotated along with it.
        child_transform.set_parent(visii.transform.get(f"{name}_symmetry_corrected"))

        if debug: 
            visii.entity.create(
                name = f"{name}_cuboid_{i_p}",
                mesh = visii.mesh.create_sphere(f"{name}_cuboid_{i_p}", radius=(0.006 / scale)),
                transform = child_transform, 
                material = visii.material.create(f"{name}_cuboid_{i_p}", base_color=cuboid_colors[i_p])
            )
    
    for i_v, v in enumerate(cuboid):
        cuboid[i_v]=[v[0], v[1], v[2]]

    return cuboid

def get_cuboid_image_space(obj_id, camera_name = 'my_camera'):
    # return cubdoid + centroid projected to the image, values [0..1]

    cam_matrix = visii.entity.get(camera_name).get_transform().get_world_to_local_matrix()
    cam_proj_matrix = visii.entity.get(camera_name).get_camera().get_projection()

    points = []
    points_cam = []
    for i_t in range(9):
        trans = visii.transform.get(f"{obj_id}_cuboid_{i_t}")
        if trans is None: 
            return None,None
        mat_trans = trans.get_local_to_world_matrix()
        pos_m = visii.vec4(
            mat_trans[3][0],
            mat_trans[3][1],
            mat_trans[3][2],
            1)
        
        p_cam = cam_matrix * pos_m

        # The nvisii pixel coordinates have the origin in the middle of the image,
        # with X going right, Y going up and range [-1..1].
        # We want to transform it so that the origin is at the top left of the image,
        # with X going right, Y going down and range [0..1].
        p_image = cam_proj_matrix * (cam_matrix * pos_m)
        p_image = visii.vec2(p_image) / p_image.w  # normalize
        p_image = p_image * visii.vec2(1, -1)  # flip Y
        p_image = (p_image + visii.vec2(1, 1)) * 0.5  # shift origin to top left corner and divide by 2

        points.append([p_image[0],p_image[1]])
        points_cam.append([p_cam[0],p_cam[1],p_cam[2]])
    return points, points_cam


def object_to_class_name(obj_name):
    try:
        # strip off part before first '_' ('hope_' or 'google_') and after last '_' ('_1234')
        return obj_name.split('_', 1)[1].rsplit('_', 1)[0]
    except:
        return obj_name


def export_to_ndds_folder_settings_files(
        output_folder=".",
        obj_names=None,
        height=500,
        width=500,
        camera_name='my_camera',
):
    if obj_names is None:
        obj_names = []

    # write _camera_settings.json
    cam_intrinsics = visii.entity.get(camera_name).get_camera().get_intrinsic_matrix(width, height)
    dict_out = {
        "camera_settings": [
            {
                "name": camera_name,
                "intrinsic_settings":
                    {
                        "resX": width,
                        "resY": height,
                        "fx": cam_intrinsics[0][0],
                        "fy": cam_intrinsics[1][1],
                        "cx": cam_intrinsics[2][0],
                        "cy": cam_intrinsics[2][1],
                        "s": cam_intrinsics[1][0]
                    },
                "captured_image_size":
                    {
                        "width": width,
                        "height": height
                    }
            }
        ]
    }
    camera_settings_filename = output_folder + "/_camera_settings.json"
    with open(camera_settings_filename, 'w') as fp:
        json.dump(dict_out, fp, indent=4)

    # write _object_settings.json
    exported_object_classes = []
    exported_objects = {}
    for obj_name in obj_names:
        class_name = object_to_class_name(obj_name)
        if class_name in exported_object_classes:
            continue
        exported_object_classes.append(class_name)
        mesh = visii.mesh.get(obj_name)
        if mesh is None:
            continue
        bbox_size = mesh.get_max_aabb_corner() - mesh.get_min_aabb_corner()
        exported_objects[class_name] = {"class": class_name,
                                        "cuboid_dimensions": [bbox_size[0], bbox_size[1], bbox_size[2]]}
    exported_object_classes.sort()
    dict_out = {
        "exported_object_classes": exported_object_classes,
        "exported_objects": [exported_objects[class_name] for class_name in exported_object_classes]
    }
    object_settings_filename = output_folder + "/_object_settings.json"
    with open(object_settings_filename, 'w') as fp:
        json.dump(dict_out, fp, indent=4)


def export_to_ndds_file(
    filename = "tmp.json", #this has to include path as well
    obj_names = [], # this is a list of ids to load and export
    height = 500, 
    width = 500,
    camera_name = 'my_camera',
    cuboids = None,
    camera_struct = None,
    segmentation_mask = None,
    compute_visibility_fraction = True,
    ):

    if segmentation_mask is None:
        segmentation_mask = visii.render_data(
            width=int(width),
            height=int(height),
            start_frame=0,
            frame_count=1,
            bounce=int(0),
            options="entity_id",
        )
        segmentation_mask = np.array(segmentation_mask).reshape((height, width, 4))[:, :, 0]

    visible_object_ids = np.unique(segmentation_mask.astype(int))

    # assume we only use the view camera
    cam_matrix = visii.entity.get(camera_name).get_transform().get_world_to_local_matrix()

    # rotate camera by 180° around x
    # from: X right, Y up and Z out of the image towards the viewer
    # to: X right, Y down and Z into the image away from the viewer (= OpenCV / NDDS / ROS "optical" frame)
    cam_matrix = visii.mat4(1, 0, 0, 0,
                            0, -1, 0, 0,
                            0, 0, -1, 0,
                            0, 0, 0, 1) * cam_matrix

    cam_matrix_export = []
    for row in cam_matrix:
        cam_matrix_export.append([row[0],row[1],row[2],row[3]])

    inverse_cam_matrix = visii.inverse(cam_matrix)
    cam_world_location = visii.vec3(inverse_cam_matrix[3][0], inverse_cam_matrix[3][1], inverse_cam_matrix[3][2])
    cam_world_quaternion = visii.quat(inverse_cam_matrix)

    cam_intrinsics = visii.entity.get(camera_name).get_camera().get_intrinsic_matrix(width, height)

    if camera_struct is None:
        camera_struct = {
            'at': [0,0,0,],
            'eye': [0,0,0,],
            'up': [0,0,0,]
        }

    dict_out = {
                "camera_data" : {
                    "width" : width,
                    'height' : height,
                    'camera_look_at':
                    {
                        'at': [
                            camera_struct['at'][0],
                            camera_struct['at'][1],
                            camera_struct['at'][2],
                        ],
                        'eye': [
                            camera_struct['eye'][0],
                            camera_struct['eye'][1],
                            camera_struct['eye'][2],
                        ],
                        'up': [
                            camera_struct['up'][0],
                            camera_struct['up'][1],
                            camera_struct['up'][2],
                        ]
                    },
                    'camera_view_matrix':cam_matrix_export,
                    'location_worldframe':
                    [
                        cam_world_location[0],
                        cam_world_location[1],
                        cam_world_location[2],
                    ],
                    'quaternion_xyzw_worldframe':[
                        cam_world_quaternion[0],
                        cam_world_quaternion[1],
                        cam_world_quaternion[2],
                        cam_world_quaternion[3],
                    ],
                    'intrinsics':{
                        'fx':cam_intrinsics[0][0],
                        'fy':cam_intrinsics[1][1],
                        'cx':cam_intrinsics[2][0],
                        'cy':cam_intrinsics[2][1]
                    }
                }, 
                "objects" : []
            }

    # Segmentation id to export
    id_keys_map = visii.entity.get_name_to_id_map()
    for obj_name in obj_names:
        # don't write objects without a single visible pixel in the output image to json file
        if int(id_keys_map[obj_name]) not in visible_object_ids:
            continue

        projected_keypoints, _ = get_cuboid_image_space(obj_name, camera_name=camera_name)

        if projected_keypoints is not None:                
            # put them in the image space. 
            for i_p, p in enumerate(projected_keypoints):
                projected_keypoints[i_p] = [p[0]*width, p[1]*height]

        # Get the location and rotation of the object in the camera frame
        trans = visii.transform.get(f"{obj_name}_symmetry_corrected")
        if trans is None:
            trans = visii.entity.get(f"{obj_name}_symmetry_corrected").get_transform()
            if trans is None: 
                continue

        quaternion_xyzw = visii.inverse(cam_world_quaternion) * visii.quat(trans.get_local_to_world_matrix())

        object_world = visii.vec4(
            trans.get_world_position()[0],
            trans.get_world_position()[1],
            trans.get_world_position()[2],
            1
        ) 
        pos_camera_frame = cam_matrix * object_world


        if not cuboids is None:
            cuboid = cuboids[obj_name]
        else:
            cuboid = None

        # compute visibility fraction
        visibility = 1
        px_count_visib = 0
        px_count_all = 0
        if compute_visibility_fraction:
            transforms_to_keep = {}

            for name in id_keys_map.keys():
                if 'camera' in name.lower() or obj_name in name:
                    continue
                trans_to_keep = visii.entity.get(name).get_transform()
                transforms_to_keep[name]=trans_to_keep
                visii.entity.get(name).clear_transform()

            # render segmentation mask of object in isolation to determine px_count_all (number of
            # pixels in the object silhouette without occlusions)
            segmentation_unique_mask = visii.render_data(
                width=int(width),
                height=int(height),
                start_frame=0,
                frame_count=1,
                bounce=int(0),
                options="entity_id",
            )
            segmentation_unique_mask = np.array(segmentation_unique_mask).reshape((height, width, 4))[:, :, 0]

            px_count_visib = len(np.where(segmentation_mask == int(id_keys_map[obj_name]))[0])
            px_count_all = len(np.where(segmentation_unique_mask == int(id_keys_map[obj_name]))[0])
            visibility = px_count_visib / px_count_all

            # set back the objects from remove
            for entity_name in transforms_to_keep.keys():
                visii.entity.get(entity_name).set_transform(transforms_to_keep[entity_name])

        # bounding box calculation
        y, x = np.where(segmentation_mask == int(id_keys_map[obj_name]))
        bounding_box = [int(min(x)), int(max(x)), height - int(max(y)), height - int(min(y))]

        tran_matrix = trans.get_local_to_world_matrix()
    
        trans_matrix_export = []
        for row in tran_matrix:
            trans_matrix_export.append([row[0],row[1],row[2],row[3]])

        # Final export
        class_name = object_to_class_name(obj_name)
        try:
            seg_id = id_keys_map[obj_name]
        except :
            seg_id = -1

        dict_out['objects'].append({
            'class':class_name,
            'name':obj_name,
            'provenance':'nvisii',
            # TODO check the location
            'location': [
                pos_camera_frame[0],
                pos_camera_frame[1],
                pos_camera_frame[2]
            ],
            'location_worldframe': [
                trans.get_world_position()[0],
                trans.get_world_position()[1],
                trans.get_world_position()[2]
            ],
            'quaternion_xyzw':[
                quaternion_xyzw[0],
                quaternion_xyzw[1],
                quaternion_xyzw[2],
                quaternion_xyzw[3],
            ],
            'quaternion_xyzw_worldframe':[
                visii.quat(trans.get_local_to_world_matrix())[0],
                visii.quat(trans.get_local_to_world_matrix())[1],
                visii.quat(trans.get_local_to_world_matrix())[2],
                visii.quat(trans.get_local_to_world_matrix())[3],
            ],
            'local_to_world_matrix':trans_matrix_export,
            'projected_cuboid':projected_keypoints,
            'segmentation_id':seg_id,
            'local_cuboid': cuboid,
            'px_count_visib': px_count_visib,
            'px_count_all': px_count_all,
            'visibility':visibility,
            'bounding_box_minx_maxx_miny_maxy':bounding_box
        })
        
    with open(filename, 'w+') as fp:
        json.dump(dict_out, fp, indent=4, sort_keys=True)
    # return bounding_box


def change_image_extension(path,extension="jpg"):
    im = cv2.imread(path)
    cv2.imwrite(path.replace("png",extension),im)
    subprocess.call(['rm',path])
    del im


######## SYMMETRIES ##########

def update_symmetry_corrected_transform(obj_id, model_info, symmetry_transforms, camera_name='camera'):
    """
    Updates the "object transform" -> "symmetry_corrected transform" to the "canonical" symmetry
    transform. The "canonical" symmetry transform is the one with the smallest angle
    between the object and camera axes specified in "align_axes" in the model_info.

    The cuboid corners are child transforms of the "symmetry_corrected" transform, so that when that
    transform is updated here, the cuboid corners will be rotated along with it such that they are
    identical for two poses which are symmetrical.
    """

    # convert align_axes vectors to visii.vec3
    align_axes = []
    try:
        # e.g., "align_axes": [{"object": [0, 1, 0], "camera": [0, 0, 1]}, {"object": [0, 0, 1], "camera": [0, 1, 0]}]
        for axis_pair in model_info["align_axes"]:
            object_axis = axis_pair["object"]
            object_axis = visii.vec3(object_axis[0], object_axis[1], object_axis[2])
            camera_axis = axis_pair["camera"]
            camera_axis = visii.vec3(camera_axis[0], camera_axis[1], camera_axis[2])
            align_axes.append({"object": object_axis, "camera": camera_axis})
    except (KeyError, IndexError):
        align_axes.append({"object": visii.vec3(1, 0, 0), "camera": visii.vec3(0, 0, 1)})

    # find the symmetry transform with the best alignment between the first object_axis and the first camera_axis,
    # using the second object_axis/camera_axis as a tie breaker

    cam_matrix = visii.entity.get(camera_name).get_transform().get_world_to_local_matrix()

    # calculate rotation error for each symmetry transform
    def calc_alignment_score(object_axis, camera_axis, symmetry_trans):
        """
        Transforms the `object_axis` vector into the camera frame using the transformation `trans` and calculates
        the "alignment score" between the rotated vector and the `camera_axis` vector. The "alignment score" is
        just the dot product between the vectors, so `math.acos(alignment_score)` is the angle between the vectors.
        `alignment_score` is between -1 (180 degree angle, worst alignment) and 1 (0 degree angle, best alignment).
        """
        mat_trans = visii.transform.get(f"{obj_id}").get_local_to_world_matrix() * symmetry_trans
        sym_object_to_camera_rotation = visii.quat(cam_matrix * mat_trans)
        sym_object_axis = sym_object_to_camera_rotation * object_axis
        return visii.dot(sym_object_axis, camera_axis)

    # score according to the first element of align_axes
    scored_symmetry_transforms = []
    for i, trans in enumerate(symmetry_transforms):
        alignment_score = calc_alignment_score(align_axes[0]["object"], align_axes[0]["camera"], trans)
        scored_symmetry_transforms.append((alignment_score, trans))

    # Sort from best to worst. Using only first element (score) for sorting to
    # avoid undefined "<" operator between second element (transforms).
    scored_symmetry_transforms.sort(key=lambda x: x[0], reverse=True)

    best_symmetry_transform = scored_symmetry_transforms[0][1]

    if len(align_axes) >= 2:
        # collect best transforms with identical score
        if len(scored_symmetry_transforms) == 0:
            return
        reference_score = scored_symmetry_transforms[0][0]
        top_symmetry_transforms = []
        EPSILON = 1e-3
        for (score, symmetry_transform) in scored_symmetry_transforms:
            if math.fabs(score - reference_score) > EPSILON:
                break
            top_symmetry_transforms.append(symmetry_transform)

        # use second element of align_axes as a tie breaker
        max_alignment_score = float("-inf")
        for trans in top_symmetry_transforms:
            alignment_score = calc_alignment_score(align_axes[1]["object"], align_axes[1]["camera"], trans)
            if alignment_score > max_alignment_score:
                max_alignment_score = alignment_score
                best_symmetry_transform = trans

    # update the symmetry_corrected transform
    symmetry_corrected_trans = visii.transform.get(f"{obj_id}_symmetry_corrected")
    symmetry_corrected_trans.set_transform(best_symmetry_transform)


# Modified from https://github.com/thodan/bop_toolkit/
def rotation_matrix(angle, direction, point=None):
    """Return matrix to rotate about axis defined by point and direction.

    >>> R = rotation_matrix(math.pi/2, [0, 0, 1], [1, 0, 0])
    >>> np.allclose(np.dot(R, [0, 0, 0, 1]), [1, -1, 0, 1])
    True
    >>> angle = (random.random() - 0.5) * (2*math.pi)
    >>> direc = np.random.random(3) - 0.5
    >>> point = np.random.random(3) - 0.5
    >>> R0 = rotation_matrix(angle, direc, point)
    >>> R1 = rotation_matrix(angle-2*math.pi, direc, point)
    >>> is_same_transform(R0, R1)
    True
    >>> R0 = rotation_matrix(angle, direc, point)
    >>> R1 = rotation_matrix(-angle, -direc, point)
    >>> is_same_transform(R0, R1)
    True
    >>> I = np.identity(4, np.float64)
    >>> np.allclose(I, rotation_matrix(math.pi*2, direc))
    True
    >>> np.allclose(2, np.trace(rotation_matrix(math.pi/2,
    ...                                               direc, point)))
    True

    """
    sina = math.sin(angle)
    cosa = math.cos(angle)
    direction = direction[:3] / np.linalg.norm(direction[:3])
    # rotation matrix around unit vector
    R = np.diag([cosa, cosa, cosa])
    R += np.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += np.array([[0.0, -direction[2], direction[1]],
                   [direction[2], 0.0, -direction[0]],
                   [-direction[1], direction[0], 0.0]])
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:
        # rotation not around origin
        point = np.array(point[:3], dtype=np.float64, copy=False)
        M[:3, 3] = point - np.dot(R, point)
    return M


def get_symmetry_transformations(model_info_path, discrete_steps_count=64):
    if model_info_path is not None:
        try:
            with open(model_info_path) as json_file:
                model_info = json.load(json_file)
        except FileNotFoundError:
            model_info = {}
    else:
        model_info = {}

    trans = _get_symmetry_transformations(model_info, discrete_steps_count)

    # convert to visii format
    trans_visii = []
    for tran in trans:
        R = tran['R']
        t = tran['t']
        mat_4x4 = visii.mat4(R[0][0], R[1][0], R[2][0], t[0],
                             R[0][1], R[1][1], R[2][1], t[1],
                             R[0][2], R[1][2], R[2][2], t[2],
                             0, 0, 0, 1)
        trans_visii.append(mat_4x4)

    return trans_visii


# Modified from https://github.com/thodan/bop_toolkit/
def _get_symmetry_transformations(model_info, discrete_steps_count=64):
    """Returns a set of symmetry transformations for an object model.

    :param model_info: See files models_info.json provided with the datasets.
    :param discrete_steps_count: The number of discretization steps
      for continuous rotational symmetries.
    :return: The set of symmetry transformations.
    """
    # Discrete symmetries.
    trans_disc = [{'R': np.eye(3), 't': np.array([[0.0, 0.0, 0.0]]).T}]  # Identity.
    if 'symmetries_discrete' in model_info:
        for sym in model_info['symmetries_discrete']:
            sym_4x4 = np.array(sym, dtype=np.float64).reshape((4, 4))
            R = sym_4x4[:3, :3]
            t = sym_4x4[:3, 3].reshape((3, 1))
            trans_disc.append({'R': R, 't': t})

    # Discretized continuous symmetries.
    trans_cont = []
    discrete_steps_count = int(discrete_steps_count)
    if 'symmetries_continuous' in model_info:
        for sym in model_info['symmetries_continuous']:
            axis = np.array(sym['axis'], dtype=np.float64)
            offset = np.array(sym['offset'], dtype=np.float64).reshape((3, 1))

            # Discrete step in radians.
            discrete_step = 2.0 * np.pi / discrete_steps_count

            for i in range(0, discrete_steps_count):
                R = rotation_matrix(i * discrete_step, axis)[:3, :3]
                t = -R.dot(offset) + offset
                trans_cont.append({'R': R, 't': t})

    # Combine the discrete and the discretized continuous symmetries.
    trans = []
    for tran_disc in trans_disc:
        if len(trans_cont):
            for tran_cont in trans_cont:
                R = tran_cont['R'].dot(tran_disc['R'])
                t = tran_cont['R'].dot(tran_disc['t']) + tran_cont['t']
                trans.append({'R': R, 't': t})
        else:
            trans.append(tran_disc)

    return trans


#################### BULLET THINGS ##############################

def load_obj_scene(path):
    """
        This loads a single entity low poly from turbosquid. 

        return: a list of visii entity names
    """
    
    obj_to_load = path    
    name_model = path

    # print("loading:",name_model)
    name = path.split('/')[-2]

    toys = visii.import_scene(obj_to_load,
        visii.vec3(0,0,0),
        visii.vec3(1,1,1), # the scale
        visii.angleAxis(1.57, visii.vec3(1,0,0))
        )
    for material in toys.materials:
        # print(material.get_name())
        if 'Glass' in material.get_name():
            # print('changing')
            # material.set_transmission(0.7)
            material.set_metallic(0.7)
            material.set_roughness(0)
    entity_names = []
    for entity in toys.entities:
        entity_names.append(entity.get_name())

    return entity_names


def create_obj(
    name = 'name',
    path_obj = "",
    path_tex = None,
    scale = 1, 
    rot_base = None
    ):

    
    # This is for YCB like dataset
    if path_obj in create_obj.meshes:
        obj_mesh = create_obj.meshes[path_obj]
    else:
        # obj_mesh = visii.mesh.create_from_obj(name, path_obj)
        # create_obj.meshes[path_obj] = obj_mesh

        obj_mesh = visii.mesh.create_from_file(name, path_obj)
        create_obj.meshes[path_obj] = obj_mesh


    
    obj_entity = visii.entity.create(
        name = name,
        # mesh = visii.mesh.create_sphere("mesh1", 1, 128, 128),
        mesh = obj_mesh,
        transform = visii.transform.create(name),
        material = visii.material.create(name)
    )

    # should randomize
    obj_entity.get_material().set_metallic(0)  # should 0 or 1      
    obj_entity.get_material().set_transmission(0)  # should 0 or 1      
    obj_entity.get_material().set_roughness(random.uniform(0,1)) # default is 1  

    if not path_tex is None:

        if path_tex in create_obj.textures:
            obj_texture = create_obj.textures[path_tex]
        else:
            obj_texture = visii.texture.create_from_file(name,path_tex)
            create_obj.textures[path_tex] = obj_texture


        obj_entity.get_material().set_base_color_texture(obj_texture)

    obj_entity.get_transform().set_scale(visii.vec3(scale))

    return obj_entity
create_obj.meshes = {}
create_obj.textures = {}

def create_physics(
    name="",
    mass = 1,         # mass in kg
    concave = False,
    ):

    # Set the collision with the floor mesh
    # first lets get the vertices 
    obj = visii.entity.get(name)
    vertices = []
    # print(name)
    # print(obj.get_mesh())
    # print(obj.to_string())
    # print(visii.mesh.get("mesh_floor"))

    for v in obj.get_mesh().get_vertices():
        vertices.append([float(v[0]),float(v[1]),float(v[2])])

    # get the position of the object
    trans = obj.get_transform().get_parent()
    if trans is None:
        trans = obj.get_transform()
    pos = trans.get_position()
    pos = [pos[0],pos[1],pos[2]]
    scale = trans.get_scale()
    scale = [scale[0],scale[1],scale[2]]
    rot = trans.get_rotation()
    rot = [rot[0],rot[1],rot[2],rot[3]]

    # create a collision shape that is a convez hull

    if concave:
        indices = obj.get_mesh().get_triangle_indices()
        obj_col_id = p.createCollisionShape(
            p.GEOM_MESH,
            vertices = vertices,
            meshScale = scale,
            indices = indices,
        )


    else:
        try:
            obj_col_id = p.createCollisionShape(
                p.GEOM_MESH,
                vertices = vertices,
                meshScale = scale,
            )
        except:
            # TODO use the size of the object to 
            # create the cuboid.
            print(f'cannot load the collision from the mesh for {name}')
            obj_col_id = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents = [0.1,0.1,0.1]
                )
    # create a body without mass so it is static
    if not mass is None : 
        obj_id = p.createMultiBody(  
            baseMass = mass, 
            baseCollisionShapeIndex = obj_col_id,
            basePosition = pos,
            baseOrientation= rot,
        )        
    else:
        obj_id = p.createMultiBody(
            baseCollisionShapeIndex = obj_col_id,
            basePosition = pos,
            baseOrientation= rot,
        )    

    return obj_id


def update_pose(obj_dict,parent=False):
    pos, rot = p.getBasePositionAndOrientation(obj_dict['bullet_id'])
    # print(pos)

    obj_entity = visii.entity.get(obj_dict['visii_id'])
    if parent:
        obj_entity.get_transform().get_parent().set_position(visii.vec3(
                                            pos[0],
                                            pos[1],
                                            pos[2]
                                            )
                                        )
    else:
        obj_entity.get_transform().set_position(visii.vec3(
                                            pos[0],
                                            pos[1],
                                            pos[2]
                                            )
                                        )

    if not obj_dict['base_rot'] is None: 
        obj_entity.get_transform().set_rotation(visii.quat(
                                                rot[3],
                                                rot[0],
                                                rot[1],
                                                rot[2]
                                                ) * obj_dict['base_rot']   
                                            )
    else:
        if parent:
            obj_entity.get_transform().get_parent().set_rotation(visii.quat(
                                                rot[3],
                                                rot[0],
                                                rot[1],
                                                rot[2]
                                                )   
                                            )
        else:
            obj_entity.get_transform().set_rotation(visii.quat(
                                    rot[3],
                                    rot[0],
                                    rot[1],
                                    rot[2]
                                    )   
                                )
    update_symmetry_corrected_transform(obj_dict['visii_id'], obj_dict['model_info'], obj_dict['symmetry_transforms'])

def material_from_json(path_json, randomize = False):
    """Load a json file visii material definition. 

    Parameters:
        path_json (str): The path to the json file to load
        randomize (bool): Randomize the material using the file definition (default:False)
    Return: 
        visii.material (visii.material): Returns a material object or None if there is a problem

    """

    if not os.path.isfile(path_json):
        warnings.warn(f"{path_json} does not exist")
        return None

    with open(path_json) as json_file:
        data = json.load(json_file)

    mat = visii.material.create(data['name'])


    # Load rgb image for base color
    if visii.texture.get(data['color']):
        mat.set_base_color_texture(visii.texture.get(data['color']))
    else:
        mat.set_base_color_texture(visii.texture.create_from_file(data['color'],data['color']))
    
    if 'gloss' in data:
        if visii.texture.get(data['gloss']):
            mat.set_roughness_texture(visii.texture.get(data['gloss']))
        else:
            im = cv2.imread(data['gloss'])
            # print(im.shape)
            im = np.power(1-(im/255.0),2)
            im_r = np.concatenate(
                [
                    im[:,:,1].reshape(im.shape[0],im.shape[1],1),
                    im[:,:,1].reshape(im.shape[0],im.shape[1],1),
                    im[:,:,1].reshape(im.shape[0],im.shape[1],1),
                    im[:,:,1].reshape(im.shape[0],im.shape[1],1)
                ],
                2
            ) 
            roughness_tex = visii.texture.create_from_data(
                data['gloss'],
                im_r.shape[0],
                im_r.shape[1],
                im_r.reshape(im_r.shape[0]*im_r.shape[1],4).astype(np.float32).flatten()
            )
            mat.set_roughness_texture(visii.texture.create_from_file(data['gloss'],roughness_tex))        
    
    if "normal" in data:
        if visii.texture.get(data['normal']):
            mat.set_normal_texture(visii.texture.get(data['normal']))
        else:
            mat.set_normal_texture(visii.texture.create_from_file(data['normal'],data['normal'],linear = True))

    return mat


def material_from_cco(path_folder,scale=1,name = None):
    """Load a json file visii material definition. 

    Parameters:
        path_json (str): The path to the textures location
        scale (float): by how much to scale the texture - default 1.0
    Return: 
        visii.material (visii.material): Returns a material object or None if there is a problem
    """
    print(path_folder.split("/")[-2])
    if not name is None:
        name = name + path_folder.split("/")[-2]
    else:
        name = path_folder.split("/")[-2]
            
    mat = visii.material.create(name)

    files = glob.glob(path_folder+'/*.jpg')+glob.glob(path_folder+'/*.png')
    print(files)

    for file in files:
        name_file_visii = name + file + str(scale)
        if 'color' in file.lower():
            if visii.texture.get(name_file_visii):
                mat.set_base_color_texture(visii.texture.get(name_file_visii))
            else:
                mat.set_base_color_texture(visii.texture.create_from_file(name_file_visii,file))
        if 'normal' in file.lower():
            if visii.texture.get(name_file_visii):
                mat.set_normal_map_texture(visii.texture.get(name_file_visii))
            else:
                mat.set_normal_map_texture(visii.texture.create_from_file(name_file_visii,file,linear=True))
        if 'rough' in file.lower():
            if visii.texture.get(name_file_visii):
                mat.set_roughness_texture(visii.texture.get(name_file_visii))
            else:
                mat.set_roughness_texture(visii.texture.create_from_file(name_file_visii,file,linear=True))
        if 'metal' in file.lower():
            if visii.texture.get(name_file_visii):
                mat.set_metallic_texture(visii.texture.get(name_file_visii))
            else:
                mat.set_metallic_texture(visii.texture.create_from_file(name_file_visii,file,linear=True))

        if visii.texture.get(name_file_visii):
            visii.texture.get(name_file_visii).set_scale((scale,scale))

    return mat
