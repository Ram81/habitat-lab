import json
import random
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import habitat.sims.habitat_simulator.sim_utilities as sutils
import habitat_sim
import magnum as mn
from habitat.core.logging import logger
from habitat_sim.physics import ManagedArticulatedObject, ManagedRigidObject

warnings.filterwarnings("ignore")
import os

import habitat.datasets.rearrange.samplers.receptacle as hab_receptacle
import numpy as np
import pandas as pd
from habitat.datasets.rearrange.samplers.receptacle import Receptacle

default_metadata_dict = {
    "metadata_folder": "data/scene_datasets/hssd-hab/metadata/",
    "obj_metadata": "object_categories_filtered.csv",
    "room_objects_json": "room_objects.json",
    "staticobj_metadata": "fpmodels-with-decomposed.csv",
}


class MetadataInterface:
    """
    Lightweight interface for managing the object semantic metadata from csv files for HSSD.
    TODO: should be promoted and used throughout codebase
    """

    def __init__(
        self, metadata_source_dict: Dict[str, str] = default_metadata_dict
    ) -> None:
        # digest the HSSD metadata
        self.metadata_source_dict = metadata_source_dict
        # common sense mappings from region categories to object categories typically found in the room
        self.commonsense_room_objects: Dict[str, List[str]] = {}
        # cache the source of each hash
        self.hash_to_source: Dict[str, str] = {}
        # cache the lexicon of available non-static (ovmm) objects
        self.dynamic_lexicon: List[str] = []
        self.metadata = self.load_metadata(self.metadata_source_dict)

        self.receptacles: List[hab_receptacle.Receptacle] = None

        # generate a lexicon from the metadata
        self.hash_to_cat: Dict[str, str] = {}
        self.lexicon: List[str] = []  # all object classes annotated
        for index in range(self.metadata.shape[0]):
            cat = self.metadata.at[index, "type"]
            self.hash_to_cat[self.metadata.at[index, "handle"]] = cat
            self.lexicon.append(cat)
        # deduplicate lexicon
        self.lexicon = list(set(self.lexicon))
        # remove non strings (e.g. nan)
        self.lexicon = [
            entry for entry in self.lexicon if isinstance(entry, str)
        ]

        # semantic naming for ReceptacleObjects (e.g. table_1)
        self.recobj_semname_to_handle: Dict[str, str] = {}
        self.recobj_handle_to_semname: Dict[str, str] = {}
        # consistent semantic naming for SemanticRegions (e.g. "living room" becomes "living_room_0")
        self.region_ix_to_semname: Dict[int, str] = {}
        self.region_semname_to_id: Dict[str, int] = {}
        # maps region index to key in commonsense_room_objects if a match was found
        self.region_ix_to_room_key: Dict[int, str] = {}

    def load_metadata(self, metadata_dict):
        """
        This method loads the metadata about objects and receptacles.
        This data will typically include tags associated with these objects such as type, color etc.
        """

        # Make sure that metadata_dict is not None
        if not metadata_dict:
            raise ValueError("Cannot load metadata from None")

        # Fetch relevant paths
        metadata_folder = metadata_dict["metadata_folder"]
        object_metadata_path = os.path.join(
            metadata_folder, metadata_dict["obj_metadata"]
        )
        static_object_metadata_path = os.path.join(
            metadata_folder, metadata_dict["staticobj_metadata"]
        )
        room_objects_json_path = os.path.join(
            metadata_folder, metadata_dict["room_objects_json"]
        )

        # Make sure that the paths are valid
        if not os.path.exists(object_metadata_path):
            raise Exception(
                f"Object metadata file not found, {object_metadata_path}"
            )
        if not os.path.exists(static_object_metadata_path):
            raise Exception(
                f"Receptacle metadata file not found, {static_object_metadata_path}"
            )
        if not os.path.exists(room_objects_json_path):
            raise Exception(
                f"Common sense region to object class json mapping file not found, {room_objects_json_path}"
            )

        # first load the json room->object categories map
        self.commonsense_room_objects = {}
        with open(room_objects_json_path, "r") as f:
            commonsense_room_objects = json.load(f)
            for room_name in commonsense_room_objects:
                # NOTE: removing upper case letters here
                self.commonsense_room_objects[room_name.lower()] = (
                    commonsense_room_objects[room_name]
                )

        # Read the metadata files
        df_static_objects = pd.read_csv(static_object_metadata_path)
        df_objects = pd.read_csv(object_metadata_path)

        # Rename some columns
        df1 = df_static_objects.rename(
            columns={"id": "handle", "main_category": "type"}
        )
        df2 = df_objects.rename(
            columns={"id": "handle", "clean_category": "type"}
        )

        # Drop the rest of the columns in both DataFrames
        df1 = df1[["handle", "type"]]
        df2 = df2[["handle", "type"]]

        # setup the hash to source mapping
        for index in range(df1.shape[0]):
            self.hash_to_source[df1.at[index, "handle"]] = "hssd"
        for index in range(df2.shape[0]):
            cat = df2.at[index, "type"]
            self.hash_to_source[df2.at[index, "handle"]] = "dynamic"
            self.dynamic_lexicon.append(cat)
        self.dynamic_lexicon = list(set(self.dynamic_lexicon))

        # Merge the two data frames
        union_df = pd.concat([df1, df2], ignore_index=True)

        return union_df

    def match_common_sense_region_objects_for_scene(
        self, sim: habitat_sim.Simulator
    ) -> None:
        """
        Attempts to match Simulator SemanticRegions to room name keys in self.commonsense_room_objects.
        Uses SemanticCategory.name() and attempts string regularization to match the keys in the map.
        """
        self.region_ix_to_room_key = {}
        for rix, region in enumerate(sim.semantic_scene.regions):
            cat_names = [region.category.name()]
            cat_names.extend(
                cat_names[0].split("/")
            )  # sometimes annotations contain multiple synonyms split by "/"
            matching_room_name_keys = [
                cat_name
                for cat_name in cat_names
                # NOTE: by construction, only lower case in self.commonsense_room_objects
                if cat_name.lower() in self.commonsense_room_objects
            ]
            matching_room_name_keys = list(set(matching_room_name_keys))
            if len(matching_room_name_keys) == 0:
                logger.info(
                    f"Cannot find common sense object mapping for region '{region.id}' with category name set: {cat_names}."
                )
            else:
                self.region_ix_to_room_key[rix] = matching_room_name_keys[0]
                if len(matching_room_name_keys) > 1:
                    logger.info(
                        f"Multiple common sense object mappings found for region {region.id} with category name set: {cat_names}. Matches = {matching_room_name_keys}, only using {matching_room_name_keys[0]}."
                    )

    def get_object_category(self, obj_hash: str) -> Optional[str]:
        """
        Get the semantic class lexicon entry corresponding to the object hash.
        Returns None if the object does not have a class or cannot be found.
        """

        if obj_hash in self.hash_to_cat:
            obj_class = self.hash_to_cat[obj_hash]
            if isinstance(obj_class, str) and len(obj_class) > 0:
                return obj_class
        return None

    def get_object_instance_category(
        self, obj: Union[ManagedRigidObject, ManagedArticulatedObject]
    ) -> Optional[str]:
        """
        Get the semantic class lexicon entry corresponding to the object isntance's template hash.
        Returns None if the object does not have a class or cannot be found.
        """

        obj_hash = sutils.object_shortname_from_handle(obj.handle)
        obj_cat = self.get_object_category(obj_hash)
        return obj_cat

    def get_scene_lexicon(self, sim: habitat_sim.Simulator):
        """
        Get the lexicon of the current scene contents by scraping the contents.
        """

        scene_lexicon = []
        rom = sim.get_rigid_object_manager()
        aom = sim.get_articulated_object_manager()
        # get all rigid and articulated object instances
        all_objs = list(rom.get_objects_by_handle_substring().values()) + list(
            aom.get_objects_by_handle_substring().values()
        )
        for obj in all_objs:
            obj_hash = sutils.object_shortname_from_handle(obj.handle)
            obj_cat = self.get_object_category(obj_hash)
            scene_lexicon.append(obj_cat)
        # de-dup
        scene_lexicon = list(set(scene_lexicon))
        # remove non-strings (e.g. None)
        scene_lexicon = [
            entry for entry in scene_lexicon if isinstance(entry, str)
        ]
        return scene_lexicon

    def get_scene_objs_of_class(
        self, sim: habitat_sim.Simulator, sem_class: str
    ) -> List[Union[ManagedRigidObject, ManagedArticulatedObject]]:
        """
        Get all objects of a given class in the currently instanced scene.
        """

        objs_of_class: List[
            Union[ManagedRigidObject, ManagedArticulatedObject]
        ] = []
        if sem_class not in self.lexicon:
            print(
                f"get_scene_objs_of_class failed, sem_class '{sem_class}' not in lexicon."
            )
            return objs_of_class

        rom = sim.get_rigid_object_manager()
        aom = sim.get_articulated_object_manager()
        # get all rigid and articulated object instances
        all_objs = list(rom.get_objects_by_handle_substring().values()) + list(
            aom.get_objects_by_handle_substring().values()
        )
        for obj in all_objs:
            obj_hash = sutils.object_shortname_from_handle(obj.handle)
            obj_cat = self.get_object_category(obj_hash)
            if obj_cat == sem_class:
                objs_of_class.append(obj)

        return objs_of_class

    def get_scene_recs_of_class(
        self,
        sem_class: str,
    ):
        """
        Get all Receptacles of a given class in the currently instanced scene.

        :param receptacles: Provide a pre-computed list of Receptacles to search. If not provided, receptacles will be pre-acquired from the Simulator.
        """

        assert (
            self.receptacles is not None
        ), "No receptacles have been scraped from the scene, call  refresh_scene_caches() first."

        class_recs: List[Receptacle] = []

        for rec in self.receptacles:
            parent_obj_hash = sutils.object_shortname_from_handle(
                rec.parent_object_handle
            )
            rec_cat = self.get_object_category(parent_obj_hash)
            if rec_cat == sem_class:
                class_recs.append(rec)

        return class_recs

    def get_template_handles_of_class(
        self,
        mm: habitat_sim.metadata.MetadataMediator,
        sem_class: str,
        dynamic_source: bool = True,
    ) -> List[str]:
        """
        Search the MetadataMediator for all objects of a given class and return a list of template handles.

        :param mm: The MetadataMediator from which to query the handles.
        :param sem_class: The semantic category of the objects which should be returned.
        :param dynamic_source: Whether or not to limit the resulting handles to those enumerated in the external dynamic objects csv file.

        :return: The list of relevant object template handles.
        """

        otm = mm.object_template_manager
        class_handles: List[str] = []
        if (
            dynamic_source and sem_class not in self.dynamic_lexicon
        ) or sem_class not in self.lexicon:
            print(
                f"'{sem_class}' not in {('dynamic' if dynamic_source else '')} lexicon."
            )
            return class_handles
        for handle in otm.get_file_template_handles():
            obj_short_name = sutils.object_shortname_from_handle(handle)
            if dynamic_source and (
                obj_short_name not in self.hash_to_source
                or self.hash_to_source[obj_short_name] != "dynamic"
            ):
                continue
            obj_cat = self.get_object_category(obj_short_name)
            if obj_cat == sem_class:
                class_handles.append(handle)
        return class_handles

    def get_region_rec_contents(
        self, sim: habitat_sim.Simulator
    ) -> Dict[str, List[str]]:
        """
        Get the current region set membership for the loaded scene.
        Returns a map of region names to a list of furniture semantic names for all objects which have Receptacles annotated.
        """
        assert (
            self.receptacles is not None
        ), "No receptacles have been scraped from the scene, call  refresh_scene_caches() first."

        region_recs: Dict[str, List[str]] = {}

        ao_link_map = sutils.get_ao_link_id_map(sim)
        ao_aabbs = sutils.get_ao_root_bbs(sim)

        # search directly in the semantic names map constructed during scene refresh
        for (
            rec_obj_handle,
            obj_sem_name,
        ) in self.recobj_handle_to_semname.items():
            parent_obj = sutils.get_obj_from_handle(sim, rec_obj_handle)
            obj_regions = sutils.get_object_regions(
                sim, parent_obj, ao_link_map=ao_link_map, ao_aabbs=ao_aabbs
            )
            for rix, _ratio in obj_regions:
                reg_name = self.region_ix_to_semname[rix]
                if reg_name not in region_recs:
                    region_recs[reg_name] = []
                region_recs[reg_name].append(obj_sem_name)

        return region_recs
