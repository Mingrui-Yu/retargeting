

import mujoco
import numpy
import os
from xml.etree import ElementTree as ET
from xml.dom import minidom
from scipy.spatial.transform import Rotation
import numpy as np
from stl import mesh

def array2str(arr):
    return " ".join([str(x) for x in arr])

def create_body(xml_root, name, inertial_pos=None, inertial_rpy=None, mass=None, ixx=None, iyy=None, izz=None):
    """
    create a body with given mass and inertia
    """
    # create XML element for this body
    body = ET.SubElement(xml_root, 'link', {'name': name})

    # add inertial element
    if inertial_pos is not None:
        inertial = ET.SubElement(body, 'inertial')
        ET.SubElement(inertial, 'origin', {'xyz': array2str(inertial_pos), 'rpy': array2str(inertial_rpy)})
        ET.SubElement(inertial, 'mass', {'value': str(mass)})
        ET.SubElement(inertial, 'inertia', {'ixx': str(ixx), 'iyy': str(iyy), 'izz': str(izz),
                                            'ixy': "0", 'ixz': "0", 'iyz': "0"})
    return body

def create_dummy_body(xml_root, name):
    """
    create a dummy body with negligible mass and inertia
    """
    # mass = 0.001
    # mass_moi = mass * (0.001 ** 2)  # mass moment of inertia
    # return create_body(xml_root, name, np.zeros(3), np.zeros(3), mass, mass_moi, mass_moi, mass_moi)
    return create_body(xml_root, name)
    

def create_joint(xml_root, name, type, parent, child, pos, rpy, axis=None, jnt_range=None,
                 mimic_jnt_name=None, mimic_coef=None):
    """
    if axis and jnt_range is None, create a fixed joint. otherwise, create a revolute joint
    """
    if axis is None:
        assert jnt_range is None
        joint_type = 'fixed'
    elif type == mujoco.mjtJoint.mjJNT_HINGE:
        joint_type = 'revolute'
    elif type == mujoco.mjtJoint.mjJNT_SLIDE:
        joint_type = 'prismatic'
    # create joint element connecting this to parent
    jnt_element = ET.SubElement(xml_root, 'joint', {'type': joint_type, 'name': name})
    ET.SubElement(jnt_element, 'parent', {'link': parent})
    ET.SubElement(jnt_element, 'child', {'link': child})
    ET.SubElement(jnt_element, 'origin', {'xyz': array2str(pos), 'rpy': array2str(rpy)})
    if axis is not None:
        ET.SubElement(jnt_element, 'axis', {'xyz': array2str(axis)})
        ET.SubElement(jnt_element, 'limit', {'lower': str(jnt_range[0]), 'upper': str(jnt_range[1]), 'effort': "100", 'velocity': "100"})
    
    assert (mimic_jnt_name is None and mimic_coef is None) or (mimic_jnt_name is not None and mimic_coef is not None)
    if mimic_jnt_name is not None:
        ET.SubElement(jnt_element, 'mimic', {'joint': mimic_jnt_name, 'multiplier': str(mimic_coef), 'offset':"0"})

    return jnt_element


def get_coupled_joints_from_mjcf(mjcf_file, type='tendon'):
    model = mujoco.MjModel.from_xml_path(mjcf_file)
    if type == 'tendon':
        return get_coupled_joints_from_tendon(model)
    else:
        raise ValueError(f"Do not support type {type}.")


def get_coupled_joints_from_tendon(model):
    """
    Assume the first joint in a fixed tendon is the active joint, and the second joint is the mimic joint.
    The setting of the tendon must be: coef_a * jpos_a + coef_b * jpos_b = 0.
    """
    tendons_type = model.wrap_type
    tendons_jnt_id = model.wrap_objid
    tendons_coef = model.wrap_prm

    assert len(tendons_type) % 2 == 0
    n_tendon_group = len(tendons_type) // 2
    coupled_joint_groups = []
    for i in range(n_tendon_group):
        if tendons_type[2 * i] == 1 and tendons_type[2 * i + 1] == 1:  # object type is joint
            tendon = {}
            active_jnt_id = tendons_jnt_id[2 * i]
            passive_jnt_id = tendons_jnt_id[2 * i + 1]
            tendon['active_jnt_name'] = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, active_jnt_id)
            tendon['passive_jnt_name'] = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, passive_jnt_id)
            tendon['mimic_coef'] = - tendons_coef[2 * i] / tendons_coef[2 * i + 1]
            coupled_joint_groups.append(tendon)

    return coupled_joint_groups
    

def convert(mjcf_file, urdf_file, asset_file_prefix=""):
    """
    load MJCF file, parse it in mujoco and save it as URDF
    replicate just the kinematic structure, ignore most dynamics, actuators, etc.
    only works with mesh geoms
    https://mujoco.readthedocs.io/en/stable/APIreference.html#mjmodel
    http://wiki.ros.org/urdf/XML

    Currently support:
        fixed, hinge, and slide joints.
        (partially) tendon (mimic joints).
        converting 'site' to 'link'.
    
    :param mjcf_file: path to existing MJCF file which will be loaded
    :param urdf_file: path to URDF file which will be saved
    :param asset_file_prefix: prefix to add to the stl file names (e.g. package://my_package/meshes/)
    """
    model = mujoco.MjModel.from_xml_path(mjcf_file)
    root = ET.Element('robot', {'name': "converted_robot"})
    urdf_dir = os.path.split(urdf_file)[0]

    coupled_joint_groups = get_coupled_joints_from_tendon(model)
    passive_joints_name = [group['passive_jnt_name'] for group in coupled_joint_groups]

    # processing bodies
    for id in range(model.nbody):
        child_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, id)
        parent_id = model.body_parentid[id]
        parent_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent_id)

        # URDFs assume that the link origin is at the joint position, while in MJCF they can have user-defined values
        # this requires some conversion for the visual, inertial, and joint elements...
        # in this script, this is done by creating a dummy body with negligible mass and inertia at the joint position.
        # the parent and joint body (dummy body) are connected with a revolute joint,
        # and the joint body and child body are connected with a fixed joint.
        parentbody2childbody_pos = model.body_pos[id]
        parentbody2childbody_quat = model.body_quat[id]  # [w, x, y, z]
        # change to [x, y, z, w]
        parentbody2childbody_quat = [parentbody2childbody_quat[1], parentbody2childbody_quat[2], parentbody2childbody_quat[3], parentbody2childbody_quat[0]]
        parentbody2childbody_Rot = Rotation.from_quat(parentbody2childbody_quat).as_matrix()
        parentbody2childbody_rpy = Rotation.from_matrix(parentbody2childbody_Rot).as_euler('xyz')

        # read inertial info
        mass = model.body_mass[id]
        inertia = model.body_inertia[id]
        childbody2childinertia_pos = model.body_ipos[id]
        childbody2childinertia_quat = model.body_iquat[id]  # [w, x, y, z]
        # change to [x, y, z, w]
        childbody2childinertia_quat = [childbody2childinertia_quat[1], childbody2childinertia_quat[2], childbody2childinertia_quat[3], childbody2childinertia_quat[0]]
        childbody2childinertia_Rot = Rotation.from_quat(childbody2childinertia_quat).as_matrix()
        childbody2childinertia_rpy = Rotation.from_matrix(childbody2childinertia_Rot).as_euler('xyz')

        jntnum = model.body_jntnum[id]
        assert jntnum <= 1, "only one joint per body supported"

        if jntnum == 1:
            # load joint info
            jntid = model.body_jntadr[id]
            # assert model.jnt_type[jntid] == mujoco.mjtJoint.mjJNT_HINGE, "only hinge joints supported"
            jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jntid)
            jnt_type = model.jnt_type[jntid]
            jnt_range = model.jnt_range[jntid]  # [min, max]
            jnt_axis_childbody = model.jnt_axis[jntid]  # [x, y, z]
            childbody2jnt_pos = model.jnt_pos[jntid]  # [x, y, z]
            parentbody2jnt_axis = jnt_axis_childbody
        else:
            # create a fixed joint instead
            jnt_name = f"{parent_name}2{child_name}_fixed"
            jnt_range = None
            jnt_type = "Fixed"
            childbody2jnt_pos = np.zeros(3)
            parentbody2jnt_axis = None

        # create child body
        body_element = create_body(root, child_name, childbody2childinertia_pos, childbody2childinertia_rpy, mass, inertia[0], inertia[1], inertia[2])

        # read geom info and add it child body
        geomnum = model.body_geomnum[id]
        for geomnum_i in range(geomnum):
            geomid = model.body_geomadr[id] + geomnum_i
            if model.geom_type[geomid] != mujoco.mjtGeom.mjGEOM_MESH:
                # only support mesh geoms
                continue
            geom_dataid = model.geom_dataid[geomid]  # id of geom's mesh
            geom_pos = model.geom_pos[geomid]
            geom_quat = model.geom_quat[geomid]  # [w, x, y, z]
            # change to [x, y, z, w]
            geom_quat = [geom_quat[1], geom_quat[2], geom_quat[3], geom_quat[0]]
            geom_rpy = Rotation.from_quat(geom_quat).as_euler('xyz')
            mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, geom_dataid)

            # create visual element within body element
            visual_element = ET.SubElement(body_element, 'visual', {'name': mesh_name})
            origin_element = ET.SubElement(visual_element, 'origin', {'xyz': array2str(geom_pos), 'rpy': array2str(geom_rpy)})
            geometry_element = ET.SubElement(visual_element, 'geometry')
            mesh_element = ET.SubElement(geometry_element, 'mesh', {'filename': f"{asset_file_prefix}converted_{mesh_name}.stl"})
            material_element = ET.SubElement(visual_element, 'material', {'name': 'white'})

            # create STL
            # the meshes in the MjModel seem to be different (have different pose) from the original STLs
            # so rather than using the original STLs, write them out from the MjModel
            # https://stackoverflow.com/questions/60066405/create-a-stl-file-from-a-collection-of-points
            vertadr = model.mesh_vertadr[geom_dataid]  # first vertex address
            vertnum = model.mesh_vertnum[geom_dataid]
            vert = model.mesh_vert[vertadr:vertadr+vertnum]
            normal = model.mesh_normal[vertadr:vertadr+vertnum]
            faceadr = model.mesh_faceadr[geom_dataid]  # first face address
            facenum = model.mesh_facenum[geom_dataid]
            face = model.mesh_face[faceadr:faceadr+facenum]
            data = np.zeros(facenum, dtype=mesh.Mesh.dtype)
            for i in range(facenum):
                data['vectors'][i] = vert[face[i]]
            m = mesh.Mesh(data, remove_empty_areas=False)
            m.save(os.path.join(urdf_dir, f"converted_{mesh_name}.stl"))


        if child_name == "world":
            # there is no joint connecting the world to anything, since it is the root
            assert parent_name == "world"
            assert jntnum == 0
            continue  # skip adding joint element or parent body

        # create dummy body for joint (position at joint, orientation same as child oody)
        jnt_body_name = f"{jnt_name}_jointbody"  # to avoid cases where the joint name is the same as the body name, add "_jointbody"
        create_dummy_body(root, jnt_body_name)
        # connect parent to joint body with revolute/prismatic joint
        parentbody2jnt_pos = parentbody2childbody_pos + parentbody2childbody_Rot @ childbody2jnt_pos
        parentbody2jnt_rpy = parentbody2childbody_rpy
        if jnt_name in passive_joints_name:
            tendon = coupled_joint_groups[passive_joints_name.index(jnt_name)]
            mimic_jnt_name, mimic_coef = tendon["active_jnt_name"], tendon["mimic_coef"]
        else:
            mimic_jnt_name, mimic_coef = None, None
        create_joint(root, jnt_name, jnt_type, parent_name, jnt_body_name, parentbody2jnt_pos, parentbody2jnt_rpy, parentbody2jnt_axis, jnt_range,
                     mimic_jnt_name, mimic_coef)
        # connect joint body to child body with fixed joint
        jnt2childbody_pos = - childbody2jnt_pos
        jnt2childbody_rpy = np.zeros(3)
        create_joint(root, f"{jnt_name}_offset", "Fixed", jnt_body_name, child_name, jnt2childbody_pos, jnt2childbody_rpy)


    # processing sites
    for id in range(model.nsite):
        child_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, id)
        parent_id = model.site_bodyid[id]
        parent_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent_id)
        parentbody2childbody_pos = model.site_pos[id]
        parentbody2childbody_quat = model.site_quat[id]  # [w, x, y, z]
        # change to [x, y, z, w]
        parentbody2childbody_quat = [parentbody2childbody_quat[1], parentbody2childbody_quat[2], parentbody2childbody_quat[3], parentbody2childbody_quat[0]]
        parentbody2childbody_rpy = Rotation.from_quat(parentbody2childbody_quat).as_euler('xyz')
        body_element = create_dummy_body(root, child_name)
        create_joint(root, f"{child_name}_joint", "Fixed", parent_name, child_name, parentbody2childbody_pos, parentbody2childbody_rpy)
    

    # define white material
    material_element = ET.SubElement(root, 'material', {'name': 'white'})
    color_element = ET.SubElement(material_element, 'color', {'rgba': '1 1 1 1'})

    # write to file with pretty printing
    xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="   ")
    os.makedirs(urdf_dir, exist_ok=True)
    with open(urdf_file, "w") as f:
        f.write(xmlstr)


def hierarchical_mjcf_to_standalone_mjcf(mjcf_file, converted_file):
    model = mujoco.MjModel.from_xml_path(mjcf_file)
    dir_path, _ = os.path.split(converted_file)
    os.makedirs(dir_path, exist_ok=True)
    mujoco.mj_saveLastXML(converted_file, model)


def get_mesh_dict_from_mjcf(mjcf_file):
    """
    Returns:
        mesh_dir: {name: {'file': file, 'scale': [x, y, z]}}
    """
    tree = ET.parse(mjcf_file)
    root = tree.getroot()
    compiler_elem = root.find('compiler')
    meshdir = compiler_elem.get('meshdir')
    asset_elem = root.find('asset')
    mesh_elems = asset_elem.findall('mesh')

    mesh_dict = {}
    for mesh in mesh_elems:
        mesh_name = mesh.get('name')
        mesh_path = mesh.get('file')
        mesh_scale = mesh.get('scale')  # if no scale, return None.
        if mesh_scale is not None:
            mesh_scale = [float(x) for x in mesh_scale.split()]
        mesh_dict[mesh_name] = {'file': os.path.join(meshdir, mesh_path),
                                'scale': mesh_scale}
    return mesh_dict
    


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mjcf_file', type=str)
    parser.add_argument('urdf_file', type=str)
    args = parser.parse_args()
    convert(args.mjcf_file, args.urdf_file)
