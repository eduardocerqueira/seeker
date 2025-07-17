#date: 2025-07-17T16:53:31Z
#url: https://api.github.com/gists/f8443b1e72ebd5c6c74869c3649300ce
#owner: https://api.github.com/users/lmoadeck-Lunity

# Made by Claude Sonnet 4 with Agent mode in VS Code
# SEKAI Character Batch Importer with Face-Body ID Linking
# This script allows batch importing of SEKAI characters, stages, and decorations
# with automatic face-body linking based on ID matching

import bpy
import os
import re
from typing import List, Tuple, Optional, Dict
from pathlib import Path
from blender.panels.importer import update_environment
from blender.operators.importer import SSSekaiBlenderImportHierarchyOperator
from blender.core.asset import import_scene_hierarchy, import_mesh_data
from blender.core.helpers import create_empty, ensure_sssekai_shader_blend
from blender.core.consts import *
from blender import sssekai_global
from UnityPy.classes import SkinnedMeshRenderer, MeshRenderer, MeshFilter

class SekaiCharacterLinker:
    """Handles linking of face and body parts for SEKAI characters based on ID matching"""
    
    def __init__(self):
        self.face_assets = {}  # ID -> (path, asset_type)
        self.body_assets = {}  # ID -> (path, asset_type)
        self.other_assets = []  # Non-character assets (stages, decorations)
    
    def extract_id_from_path(self, path: str, asset_type: str) -> Optional[str]:
        """
        Extract ID from file path based on asset type
        
        Args:
            path: Asset bundle path
            asset_type: Type of asset ("face", "body_*", etc.)
            
        Returns:
            Extracted ID or None
        """
        path_obj = Path(path)
        
        if asset_type == "face":
            # For face: use the last file/folder name in the path
            # Example: "K:\...\face\19\0815" -> ID = "0815"
            return path_obj.name
        elif asset_type.startswith("body"):
            # For body: use the last folder name in the path
            # Example: "K:\...\body\99\0816" -> ID = "0816"
            return path_obj.name
        
        return None
    
    def add_asset(self, path: str, asset_type: str):
        """Add an asset to the appropriate category"""
        if asset_type == "face":
            asset_id = self.extract_id_from_path(path, asset_type)
            if asset_id:
                self.face_assets[asset_id] = (path, asset_type)
                print(f"Added face asset with ID {asset_id}: {path}")
        elif asset_type.startswith("body"):
            asset_id = self.extract_id_from_path(path, asset_type)
            if asset_id:
                self.body_assets[asset_id] = (path, asset_type)
                print(f"Added body asset with ID {asset_id}: {path}")
        else:
            # Stage, stage_deco, and other non-character assets
            self.other_assets.append((path, asset_type))
            if asset_type in ["stage", "stage_deco"]:
                print(f"Added stage asset: {asset_type} from {path}")
            else:
                print(f"Added other asset: {asset_type} from {path}")
    
    def get_linked_characters(self) -> List[Tuple[str, str, str, str]]:
        """
        Get list of linked character pairs (face + body)
        
        Returns:
            List of tuples: (face_path, face_type, body_path, body_type)
        """
        linked_characters = []
        
        # Find matching IDs between face and body assets
        for face_id, (face_path, face_type) in self.face_assets.items():
            if face_id in self.body_assets:
                body_path, body_type = self.body_assets[face_id]
                linked_characters.append((face_path, face_type, body_path, body_type))
                print(f"Linked character ID {face_id}: face={face_path}, body={body_path}")
        
        return linked_characters
    
    def get_orphaned_assets(self) -> Dict[str, List]:
        """Get assets that don't have matching pairs"""
        orphaned = {"faces": [], "bodies": []}
        
        for face_id, (face_path, face_type) in self.face_assets.items():
            if face_id not in self.body_assets:
                orphaned["faces"].append((face_path, face_type, face_id))
        
        for body_id, (body_path, body_type) in self.body_assets.items():
            if body_id not in self.face_assets:
                orphaned["bodies"].append((body_path, body_type, body_id))
        
        return orphaned
    
    def find_matches(self) -> List[Tuple[str, str, str, str]]:
        """Find matching face-body pairs and return them as tuples"""
        return self.get_linked_characters()
    
    def import_linked_character(self, match: Tuple[str, str, str, str]) -> bool:
        """Import a linked character pair using character controller system"""
        face_path, face_type, body_path, body_type = match
        return import_linked_character(face_path, face_type, body_path, body_type)



def get_expected_body_size_from_asset_type(asset_type: str) -> str:
    """Extract expected body size from asset_type like 'body_ss', 'body_m', etc."""
    if asset_type.startswith("body_"):
        body_size = asset_type.split("_")[-1]  # Get the size part (ss, s, m, l)
        size_map = {"ss": "ladies_ss", "s": "ladies_s", "m": "ladies_m", "l": "ladies_l"}
        return size_map.get(body_size, "ladies_s")  # Default to ladies_s if unknown
    return "ladies_s"  # Default for non-body assets

def setup_material_options(import_type: str, material_mode: str = "GENERIC"):
    """Configure material import options for proper SEKAI material handling"""
    wm = bpy.context.window_manager
    
    if import_type in ["face", "body", "body_ss", "body_s", "body_m", "body_l"]:
        wm.sssekai_hierarchy_import_mode = "SEKAI_CHARACTER"
        wm.sssekai_character_type = "HEAD" if import_type == "face" else "BODY"
        # Use GENERIC mode with BASIC materials to avoid texture issues with SEKAI auto mode
        wm.sssekai_sekai_material_mode = "GENERIC"
        wm.sssekai_generic_material_import_mode = "BASIC"
    elif import_type in ["stage", "stage_deco"]:
        wm.sssekai_hierarchy_import_mode = "SEKAI_STAGE"
        # Use GENERIC mode with BASIC materials for stages too
        wm.sssekai_sekai_material_mode = "GENERIC"
        wm.sssekai_generic_material_import_mode = "BASIC"
    else:
        wm.sssekai_hierarchy_import_mode = "GENERIC"
        wm.sssekai_sekai_material_mode = "GENERIC"
        wm.sssekai_generic_material_import_mode = "BASIC"

def get_or_create_character_controller(character_id: str) -> bpy.types.Object:
    """Get existing character controller or create new one"""
    controller_name = f"SekaiCharacterRoot_{character_id}"
    
    # Check if controller already exists
    existing_controller = bpy.data.objects.get(controller_name)
    if existing_controller and KEY_SEKAI_CHARACTER_ROOT in existing_controller:
        print(f"Using existing character controller: {controller_name}")
        return existing_controller
    
    # Create new character controller
    print(f"Creating new character controller: {controller_name}")
    
    # Ensure shader blend is available
    ensure_sssekai_shader_blend()
    
    # Create character controller root
    root = create_empty(controller_name)
    root[KEY_SEKAI_CHARACTER_ROOT] = True
    root[KEY_SEKAI_CHARACTER_HEIGHT] = 1.0  # Default height
    root[KEY_SEKAI_CHARACTER_BODY_OBJ] = None
    root[KEY_SEKAI_CHARACTER_FACE_OBJ] = None
    
    # Create rim light controller if SekaiCharaRimLight exists
    rim_light_source = bpy.data.objects.get("SekaiCharaRimLight")
    if rim_light_source:
        rim_controller = rim_light_source.copy()
        rim_controller.name = f"SekaiCharaRimLight_{character_id}"
        rim_controller.parent = root
        root[KEY_SEKAI_CHARACTER_LIGHT_OBJ] = rim_controller
        bpy.context.collection.objects.link(rim_controller)
        print(f"Created rim light controller: {rim_controller.name}")
    else:
        print("Warning: SekaiCharaRimLight not found, rim lighting may not work properly")
        root[KEY_SEKAI_CHARACTER_LIGHT_OBJ] = None
    
    return root

def import_sekai_asset(asset_path: str, asset_type: str, character_controller: bpy.types.Object = None, container_key: str = "<default>") -> bool:
    """Import a SEKAI asset with proper configuration and character controller support"""
    try:
        # Check if path exists
        if not os.path.exists(asset_path):
            print(f"Warning: Path does not exist: {asset_path}")
            return False
        
        # Load the environment
        update_environment(asset_path)
        
        # Setup material options - use GENERIC mode with BASIC materials
        setup_material_options(asset_type, "GENERIC")
        
        # Debug: Print available containers
        print(f"Available containers: {list(sssekai_global.containers.keys())}")
        
        # For character imports, ensure we have a character controller
        active_obj = None
        if asset_type in ["face", "body", "body_ss", "body_s", "body_m", "body_l"]:
            if character_controller:
                active_obj = character_controller
                print(f"Using provided character controller: {active_obj.name}")
            else:
                # Extract character ID from path for controller naming
                from pathlib import Path
                character_id = Path(asset_path).name
                active_obj = get_or_create_character_controller(character_id)
                print(f"Created/found character controller: {active_obj.name}")
            
            # Set as active object for the import
            bpy.context.view_layer.objects.active = active_obj
        
        # Find the best container to use
        best_container = None
        best_hierarchy = None
        
        # Try to find a suitable container
        for container_name, container in sssekai_global.containers.items():
            if container.hierarchies:
                # Prefer .prefab containers over .fbx containers
                is_prefab = container_name.endswith('.prefab')
                is_fbx = container_name.endswith('.fbx')
                
                # For character assets, use the asset_type to determine expected size
                if asset_type.startswith("body") or asset_type == "face":
                    # Get the expected size from the asset_type (user-specified)
                    expected_size = get_expected_body_size_from_asset_type(asset_type)
                    print(f"Looking for container with size: {expected_size} for asset_type: {asset_type}")
                    
                    # Check if container contains the expected size
                    container_lower = container_name.lower()
                    if expected_size in container_lower:
                        # Priority: body.prefab > sit_body.prefab > .fbx files
                        if is_prefab and "/body.prefab" in container_name:
                            best_container = container_name
                            best_hierarchy = list(container.hierarchies.keys())[0]
                            print(f"Found matching body.prefab container for {asset_type} ({expected_size}): {container_name}")
                            break  # This is the best match, stop searching
                        elif is_prefab and "/sit_body.prefab" in container_name and (not best_container or "/body.prefab" not in best_container):
                            best_container = container_name
                            best_hierarchy = list(container.hierarchies.keys())[0]
                            print(f"Found matching sit_body.prefab container for {asset_type} ({expected_size}): {container_name}")
                        elif is_fbx and not best_container:
                            best_container = container_name
                            best_hierarchy = list(container.hierarchies.keys())[0]
                            print(f"Found matching fbx container for {asset_type} ({expected_size}): {container_name}")
                    elif asset_type == "face" and is_prefab:
                        # For faces, any prefab container is generally acceptable
                        if not best_container or "/body.prefab" in container_name:
                            best_container = container_name
                            best_hierarchy = list(container.hierarchies.keys())[0]
                            print(f"Found container for face: {container_name}")
                else:
                    # For non-character assets, prefer .prefab containers
                    if is_prefab and (not best_container or not best_container.endswith('.prefab')):
                        best_container = container_name
                        best_hierarchy = list(container.hierarchies.keys())[0]
                        print(f"Found prefab container for {asset_type}: {container_name}")
                    elif not best_container:  # Fallback to any container
                        best_container = container_name
                        best_hierarchy = list(container.hierarchies.keys())[0]
        
        # Fallback: use any available container
        if not best_container and sssekai_global.containers:
            for container_name, container in sssekai_global.containers.items():
                if container.hierarchies:
                    best_container = container_name
                    best_hierarchy = list(container.hierarchies.keys())[0]
                    print(f"Using fallback container: {container_name}")
                    break
        
        if best_container and best_hierarchy:
            print(f"Importing {asset_type} using container: {best_container}, hierarchy: {best_hierarchy}")
            
            try:
                # Get the hierarchy object directly from sssekai_global
                hierarchy = sssekai_global.containers[best_container].hierarchies[int(best_hierarchy)]
                print(f"Found hierarchy: {hierarchy.name}")
                
                # Import directly using the core function instead of the operator
                from blender.core.asset import import_scene_hierarchy, import_mesh_data
                from UnityPy.classes import SkinnedMeshRenderer, MeshRenderer, MeshFilter
                
                # Setup material options first
                setup_material_options(asset_type, "GENERIC")
                
                # Import the scene hierarchy directly (creates armatures)
                scene = import_scene_hierarchy(
                    hierarchy,
                    use_bindpose=False,  # bindpose correction
                    seperate_armatures=False  # separate armatures for skinned meshes
                )
                
                if not scene:
                    print(f"Failed to import scene hierarchy for {asset_type}")
                    return False
                
                print(f"✓ Successfully imported {len(scene)} armature(s)")
                
                # For character imports, link to character controller
                if asset_type in ["face", "body", "body_ss", "body_s", "body_m", "body_l"] and active_obj:
                    if asset_type == "face":
                        active_obj[KEY_SEKAI_CHARACTER_FACE_OBJ] = scene[0][0]
                        print(f"Linked face armature to character controller")
                    else:  # body
                        active_obj[KEY_SEKAI_CHARACTER_BODY_OBJ] = scene[0][0]
                        # Parent body armature to character controller
                        scene[0][0].parent = active_obj
                        print(f"Linked body armature to character controller")
                        # Update body position driver
                        try:
                            bpy.context.view_layer.objects.active = active_obj
                            bpy.ops.sssekai.update_character_controller_body_position_driver_op()
                            print("Updated character controller body position driver")
                        except Exception as e:
                            print(f"Warning: Could not update body position driver: {e}")
                
                # Import skinned meshes with materials
                imported_objects = []
                sm_mapping = {
                    sm_pathid: (armature_obj, bone_names)
                    for armature_obj, bone_names, sm_pathid in scene
                }
                
                mesh_count = 0
                for node in hierarchy.nodes.values():
                    game_object = node.game_object
                    if game_object.m_SkinnedMeshRenderer:
                        try:
                            sm = game_object.m_SkinnedMeshRenderer.read()
                            if not sm.m_Mesh:
                                continue
                            mesh = sm.m_Mesh.read()
                            bone_names = [
                                hierarchy.nodes[pptr.m_PathID].name for pptr in sm.m_Bones
                            ]
                            mesh_data, mesh_obj = import_mesh_data(
                                game_object.m_Name, mesh, bone_names
                            )
                            armature_obj, _mapping = sm_mapping.get(
                                sm.object_reader.path_id, (None, None)
                            )
                            if not armature_obj:
                                armature_obj, _mapping = sm_mapping.get(0, (None, None))
                                if not armature_obj:
                                    print(f"⚠ No armature found for skinned mesh {game_object.m_Name}")
                                    continue
                            # Parent mesh to armature and add armature modifier
                            mesh_obj.parent = armature_obj
                            mesh_obj.modifiers.new("Armature", "ARMATURE").object = armature_obj
                            imported_objects.append((mesh_obj, sm.m_Materials, mesh))
                            mesh_count += 1
                            print(f"  - Skinned Mesh: {mesh_obj.name} (parented to {armature_obj.name})")
                        except Exception as e:
                            print(f"⚠ Failed to import skinned mesh {game_object.m_Name}: {str(e)}")
                
                # Import static meshes with materials
                for armature_obj, nodes, sm_id in scene:
                    imported_objects.append((armature_obj, [], None))  # Add armature to objects list
                    print(f"  - Armature: {armature_obj.name} ({len(nodes)} bones)")
                    
                    for path_id, bone_name in nodes.items():
                        node = hierarchy.nodes[path_id]
                        game_object = node.game_object
                        if game_object.m_MeshFilter:
                            try:
                                m = game_object.m_MeshRenderer.read()
                                mf = game_object.m_MeshFilter.read()
                                if not mf.m_Mesh:
                                    continue
                                mesh = mf.m_Mesh.read()
                                mesh_data, mesh_obj = import_mesh_data(game_object.m_Name, mesh)
                                # Set bone parent
                                mesh_obj.parent = armature_obj
                                mesh_obj.parent_type = 'BONE'
                                mesh_obj.parent_bone = bone_name
                                imported_objects.append((mesh_obj, m.m_Materials, mesh))
                                mesh_count += 1
                                print(f"  - Static Mesh: {mesh_obj.name} (parented to bone {bone_name})")
                            except Exception as e:
                                print(f"⚠ Failed to import static mesh {game_object.m_Name}: {str(e)}")
                
                # Import materials using Generic mode with Basic materials
                print(f"Importing materials for {len(imported_objects)} objects...")
                
                try:
                    # Use the same generic material import logic as the main operator
                    from blender.core.asset import import_all_material_inputs
                    from blender.operators.material import set_generic_material_nodegroup
                    
                    texture_cache = dict()
                    material_cache = dict()
                    
                    wm = bpy.context.window_manager
                    
                    def import_material_generic(material, obj_name=""):
                        """Import material using generic mode with intelligent material type detection"""
                        try:
                            name = material.m_Name
                            envs = dict(material.m_SavedProperties.m_TexEnvs)
                            floats = dict(material.m_SavedProperties.m_Floats)
                            
                            # Import using generic material system
                            imported = import_all_material_inputs(name, material, texture_cache)
                            
                            # Apply material node group based on material type (matching sssekai logic)
                            material_mode = "BASIC"  # Default
                            
                            # Special handling for different material types
                            if "_ehl_" in name:  # Eyelight materials
                                material_mode = "COLORADD"
                                print(f"    Detected eyelight material: {name} -> COLORADD")
                            elif "_FaceShadowTex" in envs and floats.get("_UseFaceSDF", 0):  # Face SDF materials
                                material_mode = "EMISSIVE"
                                print(f"    Detected face SDF material: {name} -> EMISSIVE")
                            elif "_Color_Add" in name:  # Stage color add materials
                                material_mode = "COLORADD"
                                print(f"    Detected color add material: {name} -> COLORADD")
                            else:
                                print(f"    Standard material: {name} -> BASIC")
                            
                            # Apply the determined material node group
                            set_generic_material_nodegroup(imported, material_mode)
                            print(f"    Imported generic material: {name} ({material_mode})")
                            return imported
                        except Exception as e:
                            print(f"    Failed to import material {material.m_Name}: {str(e)}")
                            return None
                    
                    # Import materials for each object with proper material slot assignment
                    materials_imported = 0
                    for obj, materials, mesh in imported_objects:
                        if hasattr(obj, 'data') and hasattr(obj.data, 'materials'):  # Only for mesh objects
                            print(f"  Processing materials for object: {obj.name} ({len(materials)} materials)")
                            
                            # Clear existing materials first to prevent assignment issues
                            obj.data.materials.clear()
                            
                            for mat_index, ppmat in enumerate(materials):
                                if ppmat:
                                    try:
                                        material = ppmat.read()
                                        if material.object_reader.path_id in material_cache:
                                            imported_mat = material_cache[material.object_reader.path_id]
                                            obj.data.materials.append(imported_mat)
                                            print(f"    Slot {mat_index}: Reused cached material: {imported_mat.name}")
                                        else:
                                            imported_mat = import_material_generic(material, obj.name)
                                            if imported_mat:
                                                obj.data.materials.append(imported_mat)
                                                material_cache[material.object_reader.path_id] = imported_mat
                                                materials_imported += 1
                                                print(f"    Slot {mat_index}: Added new material {imported_mat.name} to {obj.name}")
                                            else:
                                                # Add empty slot to maintain material indices
                                                obj.data.materials.append(None)
                                                print(f"    Slot {mat_index}: Added empty material slot (import failed)")
                                    except Exception as e:
                                        print(f"    Failed to process material slot {mat_index} for {obj.name}: {str(e)}")
                                        # Add empty slot to maintain material indices
                                        obj.data.materials.append(None)
                                else:
                                    # Add empty slot for None materials
                                    obj.data.materials.append(None)
                                    print(f"    Slot {mat_index}: Added empty material slot (None material)")
                            
                            # Set material indices afterwards (like in main operator lines 438-449)
                            if len(materials) > 0 and hasattr(mesh, 'm_SubMeshes'):
                                print(f"    Setting material indices for {len(mesh.m_SubMeshes)} submeshes")
                                bpy.context.view_layer.objects.active = obj
                                bpy.ops.object.mode_set(mode="OBJECT")
                                
                                for index, sub in enumerate(mesh.m_SubMeshes):
                                    if index < len(obj.data.materials):
                                        start, count = sub.firstVertex, sub.vertexCount
                                        # Clear previous selection
                                        bpy.ops.object.mode_set(mode="EDIT")
                                        bpy.ops.mesh.select_all(action='DESELECT')
                                        bpy.ops.object.mode_set(mode="OBJECT")
                                        
                                        # Select vertices for this submesh
                                        for i in range(start, min(start + count, len(obj.data.vertices))):
                                            obj.data.vertices[i].select = True
                                        
                                        # Assign material
                                        bpy.ops.object.mode_set(mode="EDIT")
                                        bpy.context.object.active_material_index = index
                                        bpy.ops.object.material_slot_assign()
                                        bpy.ops.object.mode_set(mode="OBJECT")
                                        print(f"      Assigned material index {index} to submesh {index}")
                    
                    print(f"✓ Imported {materials_imported} materials with proper slot assignment")
                    
                except Exception as e:
                    print(f"⚠ Material import failed: {str(e)}")
                    import traceback
                    traceback.print_exc()
                
                print(f"✓ Complete import: {len(scene)} armature(s) and {mesh_count} mesh(es)")
                
                if scene:
                    print(f"Successfully imported {asset_type} from {asset_path}")
                    return True
                else:
                    print(f"Failed to import scene for {asset_type} from {asset_path}")
                    return False
                
            except Exception as import_error:
                print(f"Error importing {asset_type}: {str(import_error)}")
                import traceback
                traceback.print_exc()
                return False
        else:
            print(f"No suitable container/hierarchy found for {asset_type} in {asset_path}")
            print(f"Available containers: {list(sssekai_global.containers.keys())}")
            return False
        
    except Exception as e:
        print(f"Error importing {asset_type} from {asset_path}: {str(e)}")
        return False

def import_linked_character(face_path: str, face_type: str, body_path: str, body_type: str) -> bool:
    """Import a complete character (face + body) using character controller"""
    print(f"Importing linked character: face={face_path}, body={body_path}")
    
    # Extract character ID for controller naming
    from pathlib import Path
    character_id = Path(face_path).name  # Should be the same for both face and body
    
    # Create or get character controller
    character_controller = get_or_create_character_controller(character_id)
    print(f"Using character controller: {character_controller.name}")
    
    # Import face first using the character controller
    face_success = import_sekai_asset(face_path, face_type, character_controller)
    if not face_success:
        print(f"Failed to import face from {face_path}")
        return False
    
    # Import body using the same character controller
    body_success = import_sekai_asset(body_path, body_type, character_controller)
    if not body_success:
        print(f"Failed to import body from {body_path}")
        return False
    
    # Character controller should now have both face and body linked
    face_obj = character_controller.get(KEY_SEKAI_CHARACTER_FACE_OBJ)
    body_obj = character_controller.get(KEY_SEKAI_CHARACTER_BODY_OBJ)
    
    if face_obj and body_obj:
        print(f"✓ Successfully linked character controller:")
        print(f"  Face: {face_obj.name}")
        print(f"  Body: {body_obj.name}")
        print(f"  Controller: {character_controller.name}")
        
        # Optionally attempt to merge face and body armatures using SEKAI's merge operator
        try:
            # Select the character controller and attempt merge
            bpy.context.view_layer.objects.active = character_controller
            
            # Check if the merge operator is available
            if hasattr(bpy.ops.sssekai, 'util_chara_root_armature_merge_op'):
                print("Attempting to merge face and body armatures...")
                bpy.ops.sssekai.util_chara_root_armature_merge_op()
                print("✓ Successfully merged face and body armatures")
            else:
                print("Note: Armature merge operator not available, armatures remain separate")
                
        except Exception as e:
            print(f"Warning: Could not merge armatures: {str(e)}")
            # This is not a failure - characters can work with separate armatures
        
        return True
    else:
        print(f"❌ Character controller linking failed:")
        print(f"  Face object: {face_obj}")
        print(f"  Body object: {body_obj}")
        return False

def import_stages(stage_paths: List[str]) -> bool:
    """Import stage assets from a list of paths"""
    print(f"\n=== IMPORTING {len(stage_paths)} STAGES ===")
    
    success_count = 0
    for stage_path in stage_paths:
        print(f"\nImporting stage: {stage_path}")
        
        # Determine asset type based on path
        asset_type = "stage_deco" if "decoration" in stage_path.lower() else "stage"
        
        success = import_sekai_asset(stage_path, asset_type)
        if success:
            success_count += 1
            print(f"✓ Successfully imported stage")
        else:
            print(f"❌ Failed to import stage")
    
    print(f"\n=== STAGE IMPORT COMPLETE ===")
    print(f"Successfully imported: {success_count}/{len(stage_paths)}")
    return success_count == len(stage_paths)

def batch_import_sekai_assets(asset_list: List[Tuple[str, str]], auto_link_characters: bool = True) -> dict:
    """
    Batch import multiple SEKAI assets with automatic character linking
    
    Args:
        asset_list: List of tuples containing (path, asset_type)
        auto_link_characters: Whether to automatically link face and body by ID
    
    Returns:
        Dictionary with import results
    """
    results = {
        "linked_characters": [],
        "other_assets": [],
        "orphaned_faces": [],
        "orphaned_bodies": [],
        "failed": []
    }
    
    if auto_link_characters:
        # Use the linker to organize and link assets
        linker = SekaiCharacterLinker()
        
        # Add all assets to the linker
        for asset_path, asset_type in asset_list:
            linker.add_asset(asset_path, asset_type)
        
        # Import linked characters
        linked_characters = linker.get_linked_characters()
        for face_path, face_type, body_path, body_type in linked_characters:
            if import_linked_character(face_path, face_type, body_path, body_type):
                results["linked_characters"].append((face_path, body_path))
            else:
                results["failed"].append((f"{face_path} + {body_path}", "linked_character", "Import failed"))
        
        # Import other assets (stages, decorations)
        stage_assets = []
        other_assets = []
        
        # Separate stage assets from other assets for better handling
        for asset_path, asset_type in linker.other_assets:
            if asset_type in ["stage", "stage_deco"]:
                stage_assets.append(asset_path)
            else:
                other_assets.append((asset_path, asset_type))
        
        # Import stages using the dedicated stage import function
        if stage_assets:
            print(f"\nDetected {len(stage_assets)} stage assets, using stage import function")
            stage_success = import_stages(stage_assets)
            if stage_success:
                for stage_path in stage_assets:
                    asset_type = "stage_deco" if "decoration" in stage_path.lower() else "stage"
                    results["other_assets"].append((stage_path, asset_type))
            else:
                for stage_path in stage_assets:
                    asset_type = "stage_deco" if "decoration" in stage_path.lower() else "stage"
                    results["failed"].append((stage_path, asset_type, "Stage import failed"))
        
        # Import remaining non-stage assets
        for asset_path, asset_type in other_assets:
            if import_sekai_asset(asset_path, asset_type):
                results["other_assets"].append((asset_path, asset_type))
            else:
                results["failed"].append((asset_path, asset_type, "Import failed"))
        
        # Report orphaned assets
        orphaned = linker.get_orphaned_assets()
        results["orphaned_faces"] = orphaned["faces"]
        results["orphaned_bodies"] = orphaned["bodies"]
        
        # Import orphaned assets individually
        for face_path, face_type, face_id in orphaned["faces"]:
            if import_sekai_asset(face_path, face_type):
                print(f"Imported orphaned face (ID: {face_id}): {face_path}")
            else:
                results["failed"].append((face_path, face_type, f"Orphaned face import failed (ID: {face_id})"))
        
        for body_path, body_type, body_id in orphaned["bodies"]:
            if import_sekai_asset(body_path, body_type):
                print(f"Imported orphaned body (ID: {body_id}): {body_path}")
            else:
                results["failed"].append((body_path, body_type, f"Orphaned body import failed (ID: {body_id})"))
    
    else:
        # Import assets individually without linking
        stage_assets = []
        other_assets = []
        
        # Separate stage assets for dedicated handling
        for asset_path, asset_type in asset_list:
            if asset_type in ["stage", "stage_deco"]:
                stage_assets.append(asset_path)
            else:
                other_assets.append((asset_path, asset_type))
        
        # Import stages using the dedicated stage import function
        if stage_assets:
            print(f"\nDetected {len(stage_assets)} stage assets, using stage import function")
            stage_success = import_stages(stage_assets)
            if stage_success:
                for stage_path in stage_assets:
                    asset_type = "stage_deco" if "decoration" in stage_path.lower() else "stage"
                    results["other_assets"].append((stage_path, asset_type))
            else:
                for stage_path in stage_assets:
                    asset_type = "stage_deco" if "decoration" in stage_path.lower() else "stage"
                    results["failed"].append((stage_path, asset_type, "Stage import failed"))
        
        # Import other non-stage assets
        for asset_path, asset_type in other_assets:
            if import_sekai_asset(asset_path, asset_type):
                results["other_assets"].append((asset_path, asset_type))
            else:
                results["failed"].append((asset_path, asset_type, "Import failed"))
    
    return results

def print_import_results(results: dict):
    """Print detailed import results"""
    print(f"\n=== SEKAI Assets Import Results ===")
    
    print(f"\nLinked Characters: {len(results['linked_characters'])}")
    for face_path, body_path in results["linked_characters"]:
        face_id = Path(face_path).name
        body_id = Path(body_path).name
        print(f"  ✓ Character ID {face_id}: {face_path} + {body_path}")
    
    print(f"\nOther Assets: {len(results['other_assets'])}")
    for path, asset_type in results["other_assets"]:
        print(f"  ✓ {asset_type}: {path}")
    
    if results["orphaned_faces"]:
        print(f"\nOrphaned Faces: {len(results['orphaned_faces'])}")
        for path, asset_type, asset_id in results["orphaned_faces"]:
            print(f"  ⚠ Face ID {asset_id}: {path} (no matching body)")
    
    if results["orphaned_bodies"]:
        print(f"\nOrphaned Bodies: {len(results['orphaned_bodies'])}")
        for path, asset_type, asset_id in results["orphaned_bodies"]:
            print(f"  ⚠ Body ID {asset_id}: {path} (no matching face)")
    
    if results["failed"]:
        print(f"\nFailed Imports: {len(results['failed'])}")
        for path, asset_type, error in results["failed"]:
            print(f"  ✗ {asset_type}: {path} - {error}")



def analyze_asset_list(asset_list: List[Tuple[str, str]]) -> dict:
    """Analyze an asset list and categorize by type"""
    analysis = {
        "characters": {"faces": [], "bodies": []},
        "stages": [],
        "stage_decorations": [],
        "other": []
    }
    
    for path, asset_type in asset_list:
        if asset_type == "face":
            analysis["characters"]["faces"].append((path, asset_type))
        elif asset_type.startswith("body"):
            analysis["characters"]["bodies"].append((path, asset_type))
        elif asset_type == "stage":
            analysis["stages"].append((path, asset_type))
        elif asset_type == "stage_deco":
            analysis["stage_decorations"].append((path, asset_type))
        else:
            analysis["other"].append((path, asset_type))
    
    return analysis

def print_asset_analysis(asset_list: List[Tuple[str, str]]):
    """Print analysis of asset types in the list"""
    analysis = analyze_asset_list(asset_list)
    
    print("=== ASSET LIST ANALYSIS ===")
    print(f"👥 Character Assets:")
    print(f"   Faces: {len(analysis['characters']['faces'])}")
    print(f"   Bodies: {len(analysis['characters']['bodies'])}")
    
    print(f"🎭 Stage Assets:")
    print(f"   Stages: {len(analysis['stages'])}")
    print(f"   Stage Decorations: {len(analysis['stage_decorations'])}")
    
    print(f"❓ Other Assets: {len(analysis['other'])}")
    
    # Show potential character matches
    face_ids = [Path(path).name for path, _ in analysis['characters']['faces']]
    body_ids = [Path(path).name for path, _ in analysis['characters']['bodies']]
    matching_ids = set(face_ids) & set(body_ids)
    
    if matching_ids:
        print(f"🔗 Potential Character Matches: {len(matching_ids)}")
        for match_id in sorted(matching_ids):
            print(f"   Character ID: {match_id}")
    
    # Show stage assets that will use dedicated import
    stage_total = len(analysis['stages']) + len(analysis['stage_decorations'])
    if stage_total > 0:
        print(f"🎭 Stage assets will use dedicated stage import function: {stage_total} assets")

def example_batch_import():
    """Example using your exact asset list format with automatic stage detection"""
    
    asset_list = [
        (r"K:\sssekai\live_pv\live_pv\model\characterv2\body\99\0816", "body_ss"),
        (r"K:\sssekai\live_pv\live_pv\model\characterv2\body\99\0815", "body_m"),
        (r"K:\sssekai\live_pv\live_pv\model\characterv2\face\19\0815", "face"),
        (r"K:\sssekai\live_pv\live_pv\model\characterv2\face\20\0816", "face"),
        # Example stage assets - will be automatically detected and imported using stage import function
        (r"K:\sssekai\live_pv\live_pv\model\stage\0006", "stage"),
        (r"K:\sssekai\live_pv\live_pv\model\stage_decoration\0019", "stage_deco")
    ]
    
    print("=== ASSET TYPE DETECTION ===")
    print_asset_analysis(asset_list)
    
    # Run batch import with automatic linking and stage detection
    print("\n=== RUNNING BATCH IMPORT WITH AUTO-DETECTION ===")
    results = batch_import_sekai_assets(asset_list, auto_link_characters=True)
    
    # Print detailed results
    print_import_results(results)

def import_character_by_id(base_path: str, character_id: str, body_size: str = "ss"):
    """
    Import a character by ID, automatically finding face and body paths
    
    Args:
        base_path: Base directory containing character assets
        character_id: Character ID (e.g., "0815", "0816")
        body_size: Body size ("ss", "s", "m", "l")
    """
    # Construct expected paths
    face_path = Path(base_path) / "face" / "19" / character_id  # Adjust face subfolder as needed
    body_path = Path(base_path) / "body" / "99" / character_id  # Adjust body subfolder as needed
    
    asset_list = [
        (str(face_path), "face"),
        (str(body_path), f"body_{body_size}")
    ]
    
    results = batch_import_sekai_assets(asset_list, auto_link_characters=True)
    print_import_results(results)
    
    return results

# Main execution function
def main():
    """Main function to run the batch importer"""
    print("=== RUNNING BATCH IMPORT ===")
    
    # You can modify this to use your actual asset list
    example_batch_import()

if __name__ == "__main__":
    import os
    os.system('cls' if os.name == 'nt' else 'clear')  # Clear console for better readability
    main()
