import xml.etree.ElementTree as ET
# import tempfile
import os 
from .utils_mjcf import sort_elements


def urdf_to_xml(urdf_path: str, saved_xml_path: str):
    """
        - The mesh files must be in the same folder as the urdf file (no sub-folders).
        - The input urdf file may be modified.
    """
    import mujoco

    # file_name, _ = os.path.splitext(urdf_path)
    # urdf_name = urdf_path.split("/")[-1]
    # temp_dir = tempfile.mkdtemp(prefix="utils_urdf-")
    # temp_path = f"{temp_dir}/{urdf_name}"

    # add mujoco compiler to the urdf file and save it
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    if root.find("mujoco"):
        root.remove(root.find("mujoco"))
    mujoco_element = ET.Element("mujoco")
    compile_element = ET.Element("compiler", 
                                 balanceinertia="true", 
                                 discardvisual="false", 
                                 fusestatic="false",
                                 strippath="false")  # ref: https://mujoco.readthedocs.io/en/latest/XMLreference.html#compiler
    mujoco_element.append(compile_element)
    root.insert(0, mujoco_element)

    # tree.write(temp_path)
    tree.write(urdf_path)

    # urdf file to mujoco xml file
    """
        Issue: mujoco can only import meshes for collision from the urdf, while ignoring the meshes for visual?
    """
    model = mujoco.MjModel.from_xml_path(urdf_path)
    xml_path = saved_xml_path
    dir_path, _ = os.path.split(xml_path)
    os.makedirs(dir_path, exist_ok=True)
    mujoco.mj_saveLastXML(xml_path, model)

    # add a 'body' element named 'base' below the 'worldbody'
    tree = ET.parse(xml_path)
    root = tree.getroot()
    worldbody_element = root.find("worldbody")
    child_elements = worldbody_element.findall('*')

    base_element = ET.Element("body", name="base")
    base_element.extend(child_elements)

    for element in child_elements:
        worldbody_element.remove(element)
    worldbody_element.append(base_element)

    # save the xml file
    tree.write(xml_path)

    

def create_rigid_object_mjcf(object_name: str, meshfile: str, savepath: str, density=1000):
    root = ET.Element("mujoco", model=object_name)
    root.append(ET.Element("compiler", angle="radian"))
    worldbody = ET.Element("worldbody")
    
    asset = ET.Element("asset")
    mesh_name = f"{object_name}_mesh"
    asset.append(ET.Element("mesh", name=mesh_name, file=meshfile))
    
    body = ET.Element("body", name="object", pos="0 0 0")
    geom = ET.Element("geom", name="collision", type="mesh", mesh=mesh_name, density=str(density))
    body.append(geom)
    
    worldbody.append(body)
    
    root.append(asset)
    root.append(worldbody)
    
    tree = ET.ElementTree(root)
    tree.write(savepath)
    

def create_rigid_object_urdf(object_name: str, meshfile: str, savepath: str):
    root = ET.Element('robot', name=object_name)
    
    link = ET.Element('link', name=object_name)

    visual = ET.Element('visual')
    visual.append(ET.Element('origin', xyz="0 0 0"))
    visual_geom = ET.Element('geometry')
    visual_geom.append(ET.Element('mesh', filename=meshfile))
    visual.append(visual_geom)

    collision = ET.Element('collision')
    collision_geom = ET.Element('geometry')
    collision_geom.append(ET.Element('mesh', filename=meshfile))
    collision.append(collision_geom)

    link.append(visual)
    link.append(collision)

    root.append(link)

    xml_string = ET.tostring(root, encoding='UTF-8', method='xml')
    xml_string_with_declaration = b'<?xml version="1.0" encoding="UTF-8"?>\n' + xml_string

    with open(savepath, 'wb') as f:
        f.write(xml_string_with_declaration)


# ---------------------------------------------------
def test_urdf_to_mujoco_xml():
    robot_urdf_path = "/home/mingrui/Mingrui/Research/project_hand_teleoperation/our_retargeting/ws_catkin/src/hand_retargeting/robot_hand_retargeting/assets/robots/hands/allegro_hand/mujoco/allegro_hand_right.urdf"
    urdf_to_xml(robot_urdf_path)
    

def test_create_rigid_object_mjcf():
    object_name = "abc"
    meshfile = "/home/mingrui/Mingrui/Research/project_grasping/DexYCB/models/021_bleach_cleanser/textured_simple.obj"
    savepath = "/home/mingrui/Mingrui/Research/project_grasping/DexYCB/models/021_bleach_cleanser/mjcf.xml"
    create_rigid_object_mjcf(object_name, meshfile, savepath)


if __name__ == '__main__':
    test_create_rigid_object_mjcf()