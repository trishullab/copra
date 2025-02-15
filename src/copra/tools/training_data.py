
#!/usr/bin/env python3

import sys

root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import os
import copy
import ray
import typing
import logging
import time
import psutil
from src.tools.ray_utils import RayUtils
from src.tools.training_data_format import LemmaRefWithScore, LemmaReferencesCollection, MergableCollection, TrainingDataCollection, TrainingDataFormat, TrainingDataMetadataFormat


class TrainingData(MergableCollection):
    def __init__(
            self, 
            folder: str, 
            training_meta_filename: str,
            training_meta: TrainingDataMetadataFormat = None,
            max_parallelism: int = 4,
            remove_from_store_after_loading: bool = True,
            logger: logging.Logger = None):
        assert os.path.exists(folder), f"Folder {folder} does not exist"
        assert os.path.isdir(folder), f"Folder {folder} is not a directory"
        assert training_meta_filename is not None, "Training meta filename cannot be None"
        self.folder = folder
        self._is_loaded = False
        self.training_meta_filename = training_meta_filename
        self.meta : TrainingDataMetadataFormat = training_meta
        self.lemma_ref_collection : LemmaReferencesCollection = LemmaReferencesCollection()
        self._lemma_ref_filename : str = None
        self._training_data_filenames : typing.List[str] = []
        self.training_data_collections : typing.List[TrainingDataCollection] = []
        self._max_parallelism: int = max_parallelism
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        self.remove_from_store_after_loading = remove_from_store_after_loading
        self._meta_loaded = False
        self._object_id_map : typing.List[ray.ObjectRef] = []
        super().__init__()

    def __len__(self) -> int:
        assert self.meta is not None, "Training meta is not set"
        return self.meta.total_proof_step_cnt

    @property
    def is_readonly(self) -> bool:
        return os.path.exists(os.path.join(self.folder, self.training_meta_filename))

    def load_meta(self):
        assert self.is_readonly, "Training data is not loadable"
        self.meta : TrainingDataMetadataFormat = TrainingDataMetadataFormat.load_from_file(os.path.join(self.folder, self.training_meta_filename))
        self.training_data_collections.clear()
        self._training_data_filenames.clear()
        lemma_file_cnt = 0
        for filename in os.listdir(self.folder):
            if filename.startswith(self.meta.lemma_ref_filename_prefix) and filename.endswith(self.meta.lemma_ref_filename_suffix):
                self._lemma_ref_filename = filename
                lemma_file_cnt += 1
            elif filename.startswith(self.meta.data_filename_prefix) and filename.endswith(self.meta.data_filename_suffix):
                self._training_data_filenames.append(filename)
        assert lemma_file_cnt == 1, "There must be exactly one lemma reference file"
        self._training_data_filenames.sort()
        self.logger.info(f"Loading lemma reference from {self.folder}: {self._lemma_ref_filename}")
        self.logger.info(f"Loading training data from {self.folder}: {self._training_data_filenames}")
        assert self._lemma_ref_filename is not None, "Lemma reference filename is not set"
        self._object_id_map = [None] * (len(self._training_data_filenames) + 2)
        self.training_data_collections = [None] * len(self._training_data_filenames)
        self.lemma_ref_collection = None
        meta_id = ray.put(self.meta)
        self._object_id_map[0] = meta_id
        self._meta_loaded = True
        pass

    def load(self):
        assert self.is_readonly, "Training data is not loadable"
        if not self._meta_loaded:
            self.load_meta()
        files_to_load = [self.training_meta_filename, self._lemma_ref_filename] + self._training_data_filenames
        self.logger.info(f"Loading {len(files_to_load)} files...")
        last_loaded_idx = 1

        def _create_remote(filenames):
            remotes = []
            base_idx = last_loaded_idx - len(filenames)
            for i, filename in enumerate(filenames):
                self.logger.info(f"[TrainingData] Starting the loading of [{base_idx + i}] {filename}...")
                collection_fn = TrainingData._get_lemma_ref_collection if filename == self._lemma_ref_filename else TrainingData._get_training_data_collection
                remotes.append(collection_fn.remote(base_idx + i, self.folder, filename))
            return remotes

        def _transform_remote(results):
            for res in results:
                if isinstance(res, tuple):
                    assert len(res) == 2, "Invalid tuple length"
                    assert isinstance(res[0], int), "Invalid type"
                    assert res[0] < len(self._object_id_map), f"Invalid index {res[0]} in {len(self._object_id_map)}"
                    obj = res[1]
                    is_lemma_ref = res[0] == 1
                    if not is_lemma_ref:
                        assert isinstance(obj, TrainingDataCollection), "Invalid type"
                        self.training_data_collections[res[0] - 2] = obj
                        self.logger.info(f"[TrainingData] Finished the loading of {res[0]}")
                    else:
                        assert isinstance(obj, LemmaReferencesCollection), "Invalid type"
                        self.lemma_ref_collection = obj
                        self.logger.info(f"[TrainingData] Finished the loading of {self._lemma_ref_filename}")
                else:
                    raise Exception(f"Invalid type {type(res)}")
                process = psutil.Process()
                self.logger.info(f"[TrainingData] Memory usage: {process.memory_info().rss / 2**30} GiB, Process: {process.pid}")

        def _prepare_next_batch(num:int):
            nonlocal last_loaded_idx
            filenames = files_to_load[last_loaded_idx: last_loaded_idx + num]
            last_loaded_idx += len(filenames)
            return filenames

        RayUtils.ray_run_within_parallel_limits(self._max_parallelism, len(files_to_load) - 1, _transform_remote, _prepare_next_batch, _create_remote, logger=self.logger)
        self.logger.info(f"Finished loading {len(files_to_load)} files")
        self._is_loaded = True
    
    def unload(self):
        assert self.is_readonly, "Training data is not loadable"
        if self._is_loaded:
            self._is_loaded = False
            # Reload the metadata
            self.load_meta()

    def merge(self, __o: object, new_lemma_ref_idx: typing.List[int] = None):
        assert isinstance(__o, TrainingDataFormat) or \
        isinstance(__o, TrainingData), "other must be a TrainingDataFormat or TrainingDataMetadata"
        assert not self.is_readonly, "Training data is read only"
        assert self.lemma_ref_collection is not None, "Lemma reference collection is not set"
        if isinstance(__o, TrainingData):
            assert new_lemma_ref_idx is None, "new_lemma_ref_idx must be None"
            new_lemma_ref_idx = self.lemma_ref_collection.merge(__o.lemma_ref_collection) # merge lemma references
            for idx in range(len(__o)):
                self._merge_training_data_format(__o[idx], new_lemma_ref_idx) # merge training data
            assert (len(__o) > 0 and len(self.training_data_collections[-1]) <= self.meta.training_data_buffer_size) or len(__o) == 0, "Training data buffer size is too large"
        else:
            self._merge_training_data_format(__o, new_lemma_ref_idx)
            assert len(self.training_data_collections[-1]) <= self.meta.training_data_buffer_size, "Training data buffer size is too large"

    def clone_skeleton(self, training_data, lemma_ref_collection: LemmaReferencesCollection = None):
        assert self.meta is not None, "Metadata is not set"
        assert isinstance(training_data, TrainingData), "Invalid type"
        assert not self._meta_loaded, "Training metadata is already loaded"
        self.meta.training_data_buffer_size = training_data.meta.training_data_buffer_size
        self.meta.total_proof_step_cnt = training_data.meta.total_proof_step_cnt
        self.meta.external_theorems_used_cnt = training_data.meta.external_theorems_used_cnt
        self.meta.local_theorems_used_cnt = training_data.meta.local_theorems_used_cnt
        self.meta.last_proof_id = training_data.meta.last_proof_id
        self.meta.last_training_data = training_data.meta.last_training_data
        # Add all the training data to the new training data
        if lemma_ref_collection is None:
            lemma_ref_id = training_data._object_id_map[1]
        else:
            lemma_ref_id = ray.put(lemma_ref_collection)
        meta_id = ray.put(self.meta)
        self._object_id_map = [None] * (len(training_data.training_data_collections) + 2)
        self._object_id_map[0] = meta_id
        self._object_id_map[1] = lemma_ref_id
        self._training_data_filenames.clear()
        lemma_len = int(training_data._lemma_ref_filename[len(training_data.meta.lemma_ref_filename_prefix): -1*len(training_data.meta.lemma_ref_filename_suffix)])
        self._lemma_ref_filename = self.meta.lemma_ref_filename_prefix + f"{lemma_len:010d}" + self.meta.lemma_ref_filename_suffix
        self.lemma_ref_collection = training_data.lemma_ref_collection
        for filename in training_data._training_data_filenames:
            idx_len = int(filename[len(training_data.meta.data_filename_prefix): -1*len(training_data.meta.data_filename_suffix)])
            self._training_data_filenames.append(self.meta.data_filename_prefix + f"{idx_len:010d}" + self.meta.data_filename_suffix)
            self.training_data_collections.append(None)
        assert len(self._training_data_filenames) == len(self.training_data_collections), "Invalid length"
        assert len(self._training_data_filenames) == len(training_data.training_data_collections), "Invalid length"

    def __getitem__(self, idx: int) -> TrainingDataFormat:
        tdc_idx = idx // self.meta.training_data_buffer_size
        idx_in_tdc = idx % self.meta.training_data_buffer_size
        if tdc_idx >= len(self.training_data_collections):
            raise IndexError(f"Index out of (len(self.training_data_collections)={len(self.training_data_collections)}, buffer={self.meta.training_data_buffer_size}, range idx={idx}, tdc_idx={tdc_idx}, idx_in_tdc={idx_in_tdc}, len(self.training_data_collections)={len(self.training_data_collections)})")
        tdc = self.training_data_collections[tdc_idx]
        if idx_in_tdc >= len(tdc):
            raise IndexError(f"Index out of range (len(self.training_data_collections)={len(self.training_data_collections)},buffer={self.meta.training_data_buffer_size}, range idx={idx}, tdc_idx={tdc_idx}, idx_in_tdc={idx_in_tdc}, len(tdc)={len(tdc)})")
        training_data = copy.deepcopy(tdc.training_data[idx_in_tdc])
        lemma_refs : typing.Set[int] = set()
        for goal in training_data.start_goals:
            lemma_refs.update([ref.lemma_idx for ref in goal.relevant_defns])
            lemma_refs.update([ref.lemma_idx for ref in goal.used_theorems_local])
            lemma_refs.update([ref.lemma_idx for ref in goal.used_theorems_external])
            lemma_refs.update([ref.lemma_idx for ref in goal.possible_useful_theorems_local])
            lemma_refs.update([ref.lemma_idx for ref in goal.possible_useful_theorems_external])
        ordered_lemma_refs = sorted(list(lemma_refs))
        lemma_ref_map = {lemma_ref: idx for idx, lemma_ref in enumerate(ordered_lemma_refs)}
        training_data.all_useful_defns_theorems = [self.lemma_ref_collection.lemma_references[lemma_idx].clone(idx) for idx, lemma_idx in enumerate(ordered_lemma_refs)]
        # Change the lemma references
        for goal in training_data.start_goals:
            goal.relevant_defns = [LemmaRefWithScore(lemma_ref_map[lemma_ref.lemma_idx], lemma_ref.score) for lemma_ref in goal.relevant_defns]
            goal.used_theorems_local = [LemmaRefWithScore(lemma_ref_map[lemma_ref.lemma_idx], lemma_ref.score) for lemma_ref in goal.used_theorems_local]
            goal.used_theorems_external = [LemmaRefWithScore(lemma_ref_map[lemma_ref.lemma_idx], lemma_ref.score) for lemma_ref in goal.used_theorems_external]
            goal.possible_useful_theorems_local = [LemmaRefWithScore(lemma_ref_map[lemma_ref.lemma_idx], lemma_ref.score) for lemma_ref in goal.possible_useful_theorems_local]
            goal.possible_useful_theorems_external = [LemmaRefWithScore(lemma_ref_map[lemma_ref.lemma_idx], lemma_ref.score) for lemma_ref in goal.possible_useful_theorems_external]
        return training_data

    def save(self) -> str:
        assert not self.is_readonly, "Training data is read only"
        self.logger.info(f"[TrainingData] Saving training data {self.folder} ...")

        use_named_reference = len(self._object_id_map) == len(self._training_data_filenames) + 2

        if not use_named_reference:
            # Generate lemma ref file name
            if self._lemma_ref_filename is None:
                self._lemma_ref_filename = self.meta.lemma_ref_filename_prefix + f"{len(self.lemma_ref_collection):010d}" + self.meta.lemma_ref_filename_suffix
            
            if len(self._training_data_filenames) == 0:
                # Generate training data file names
                cum_cnt = 0
                for tdc in self.training_data_collections:
                    cum_cnt += len(tdc)
                    training_data_filename = self.meta.data_filename_prefix + f"{cum_cnt:010d}" + self.meta.data_filename_suffix
                    self._training_data_filenames.append(training_data_filename)
            assert len(self._training_data_filenames) == len(self.training_data_collections), "Invalid length"
            self._object_id_map = [None] * (len(self._training_data_filenames) + 2)
        else:
            assert len(self._object_id_map) == len(self._training_data_filenames) + 2, "Invalid length"
        files_to_save = [self.training_meta_filename, self._lemma_ref_filename] + self._training_data_filenames
        last_idx = 0

        self.logger.info(f"[TrainingData] Files to save: {files_to_save}")

        if not use_named_reference:
            tdcs = [self.meta, self.lemma_ref_collection] + self.training_data_collections
            self.logger.info(f"[TrainingData] Putting tdc to ray...")
            for idx, tdc in enumerate(tdcs):
                self._object_id_map[idx] = ray.put(tdc)
                self.logger.info(f"[TrainingData] Put [{idx}] to ray")
            self.logger.info(f"[TrainingData] Finished putting tdc to ray")
        else:
            self.logger.info(f"[TrainingData] Using named reference")

        assert len(self._object_id_map) == len(files_to_save), "Invalid length"
        assert all([obj_ref is not None for obj_ref in self._object_id_map]), "Invalid object id map"

        def _create_remote(filenames):
            remotes = []
            base_idx = last_idx - len(filenames)
            for i, filename in enumerate(filenames):
                self.logger.info(f"[TrainingData] Starting the saving of [{base_idx + i}] {filename}...")
                obj_ref = self._object_id_map[base_idx + i]
                remotes.append(TrainingData._save_object.remote(base_idx + i, obj_ref, os.path.join(self.folder, filename)))
            return remotes
        
        def _transform_remote(results):
            for res in results:
                if isinstance(res, tuple):
                    assert len(res) == 2, "Invalid return value"
                    assert isinstance(res[0], int), "Invalid return value"
                    assert isinstance(res[1], str), "Invalid return value"
                    self.logger.info(f"[TrainingData] Saved [{res[0]}] in file {res[1]}")
                else:
                    raise Exception(f"Unable to save {res}")
            process = psutil.Process()
            self.logger.info(f"[TrainingData] Memory usage: {process.memory_info().rss / 2**30} GiB, Process: {process.pid}")

        def _prepare_next_batch(num:int):
            nonlocal last_idx, files_to_save
            filenames = files_to_save[last_idx:last_idx + num]
            last_idx += len(filenames)
            return filenames

        RayUtils.ray_run_within_parallel_limits(self._max_parallelism, len(files_to_save), _transform_remote, _prepare_next_batch, _create_remote, self.logger)
        return self.folder

    def _merge_training_data_format(self, other: TrainingDataFormat, new_lemma_ref_idx: typing.List[int] = None):
        assert isinstance(other, TrainingDataFormat), "other must be a TrainingDataFormat"
        assert self.lemma_ref_collection is not None, "Lemma ref collection is None"
        if new_lemma_ref_idx is None:
            new_lemma_ref_idx : typing.List[int] = self.lemma_ref_collection.merge(other.all_useful_defns_theorems)
            assert len(new_lemma_ref_idx) == len(other.all_useful_defns_theorems), "Invalid lemma ref idx"
        if len(self.training_data_collections) == 0:
            self.training_data_collections.append(TrainingDataCollection())
        last_training_data_collection = self.training_data_collections[-1]
        if len(last_training_data_collection) + 1 > self.meta.training_data_buffer_size:
            self.training_data_collections.append(TrainingDataCollection())
            last_training_data_collection = self.training_data_collections[-1]
        TrainingData._merge_training_data_collection(last_training_data_collection, [other], new_lemma_ref_idx)
        # Update the metadata
        self.meta.last_proof_id = other.proof_id
        self.meta.last_training_data += 1
        self.meta.external_theorems_used_cnt += sum([len(goal.used_theorems_external) for goal in other.start_goals])
        self.meta.local_theorems_used_cnt += sum([len(goal.used_theorems_local) for goal in other.start_goals])
        self.meta.total_proof_step_cnt += len(other.proof_steps)

    @ray.remote(max_retries=-1)
    def _get_training_data_collection(idx : int, folder: str, filename: str) -> typing.Tuple[int, ray.ObjectID]:
        file_path = os.path.join(folder, filename)
        start_time = time.time()
        ray.logger.info(f"[TrainingData] Trying to load {file_path}")
        tdc = TrainingDataCollection.load_from_file(file_path)
        end_time = time.time()
        ray.logger.info(f"[TrainingData] Loaded {file_path} in {end_time - start_time} seconds")
        return idx, tdc
    
    @ray.remote(max_retries=-1)
    def _get_lemma_ref_collection(idx : int, folder: str, filename: str) -> typing.Tuple[int, ray.ObjectID]:
        file_path = os.path.join(folder, filename)
        start_time = time.time()
        ray.logger.info(f"[TrainingData] Trying to load {file_path}")
        res = LemmaReferencesCollection.load_from_file(file_path)
        end_time = time.time()
        ray.logger.info(f"[TrainingData] Loaded {file_path} in {end_time - start_time} seconds")
        return idx, res

    @ray.remote(max_retries=-1)
    def _save_object(i : int, obj: typing.Union[TrainingDataCollection, TrainingDataMetadataFormat, LemmaReferencesCollection], filepath: str):
        save_start_time = time.time()
        ray.logger.info(f"[TrainingData] Saving {filepath}")
        with open(filepath, 'w') as f:
            # serialize the current metadata
            json_str = obj.to_json()
            # update the metadata in the file
            f.write(json_str)
        save_end_time = time.time()
        ray.logger.info(f"[TrainingData] Saved {filepath} in {save_end_time - save_start_time}s")
        return i, filepath

    def _merge_training_data_collection(other: TrainingDataCollection, training_data_points: typing.List[TrainingDataFormat], new_lemma_ref_idx: typing.List[int]):
        assert isinstance(other, TrainingDataCollection), "other must be a TrainingDataFormat or TrainingDataCollection"
        assert isinstance(training_data_points, list), "training_data_points must be a list"
        assert isinstance(new_lemma_ref_idx, list), "new_lemma_ref_idx must be a list"
        new_tdps : typing.List[TrainingDataFormat] = []
        for tdp in training_data_points:
            assert isinstance(tdp, TrainingDataFormat), "training_data_points must contain TrainingDataFormat objects"
            new_tdp = TrainingDataFormat(
                tdp.proof_id,
                all_useful_defns_theorems=[],
                start_goals=tdp.start_goals,
                end_goals=tdp.end_goals,
                proof_steps=tdp.proof_steps,
                simplified_goals=tdp.simplified_goals,
                addition_state_info=tdp.addition_state_info)
            # Reset the lemma references
            for goal in new_tdp.start_goals:
                goal.relevant_defns = [LemmaRefWithScore(new_lemma_ref_idx[lemma_ref.lemma_idx], lemma_ref.score) for lemma_ref in goal.relevant_defns]
                goal.used_theorems_local = [LemmaRefWithScore(new_lemma_ref_idx[lemma_ref.lemma_idx] , lemma_ref.score) for lemma_ref in goal.used_theorems_local]
                goal.used_theorems_external = [LemmaRefWithScore(new_lemma_ref_idx[lemma_ref.lemma_idx], lemma_ref.score) for lemma_ref in goal.used_theorems_external]
                goal.possible_useful_theorems_local = [LemmaRefWithScore(new_lemma_ref_idx[lemma_ref.lemma_idx], lemma_ref.score) for lemma_ref in goal.possible_useful_theorems_local]
                goal.possible_useful_theorems_external = [LemmaRefWithScore(new_lemma_ref_idx[lemma_ref.lemma_idx], lemma_ref.score) for lemma_ref in goal.possible_useful_theorems_external]
            new_tdps.append(new_tdp)
        other.training_data.extend(new_tdps)