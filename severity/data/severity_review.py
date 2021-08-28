from typing import List, Dict, Optional, Any
from collections import defaultdict
import copy

from prodigy.core import recipe
from prodigy.components.db import connect
from prodigy.util import log, split_string, set_hashes, msg, get_labels
from prodigy.util import INPUT_HASH_ATTR, TASK_HASH_ATTR, SESSION_ID_ATTR, VIEW_ID_ATTR

#from prodigy_util_functions import *
import sys 
import os
sys.path.append(os.path.abspath("/home/joe/research/transgression_ambiguity/annotation_code/notebooks/"))
from prodigy_util_functions import *
#watchout()




class ReviewStream(object):
    def __init__(self, data: Dict[int, dict], by_input: bool = False):
        """Initialize a review stream. This class mostly exists so we can
        expose a __len__ (to show the total progress) and to separate out some
        of the task-type specific abstractions like by_input.

        data (dict): The merged data: {INPUT_HASH: { (TASK_HASH, answer): task }}.
        by_input (bool): Whether to consider everything with the same input hash
            to be the same task the review. This makes sense for datasets with
            ner_manual annotations on the same text. Different task hashes on
            the same input would then be considered conflicts. If False,
            examples with different task hashes are considered different tasks
            to review and only the answers (accept / reject) are what could be
            considered a conflict. This makes sense for binary annotations
            where a reviewer would only be judging the accept/reject decisions.
        """
        if by_input:
            self.data = self.get_data_by_input(data)
        else:
            self.data = self.get_data_by_task(data)

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for eg in self.data:
            yield eg

    def make_eg(self, versions):
        default_version = max(versions, key=lambda v: len(v["sessions"]))
        default_version_idx = versions.index(default_version)
        eg = copy.deepcopy(default_version)
        for i, version in enumerate(versions):
            version["default"] = i == default_version_idx
        eg["versions"] = versions
        eg["view_id"] = eg[VIEW_ID_ATTR]
        return eg

    def get_data_by_input(self, data):
        # We're considering everything with the same input hash to
        # be the same task to review (e.g. different spans on same
        # text when highlighted manually). Different task hashes on the same
        # input are treated as conflicts to resolve. Rejected answers are
        # automatically excluded.
        examples = []
        for input_versions in data.values():
            versions = []
            for (task_hash, answer), task_versions in input_versions.items():
                task_versions = [v for v in task_versions if v["answer"] != "reject"]
                if task_versions:
                    version = copy.deepcopy(task_versions[0])
                    sessions = sorted([eg[SESSION_ID_ATTR] for eg in task_versions])
                    version["sessions"] = sessions
                    versions.append(version)
            if versions:
                examples.append(self.make_eg(versions))
        return examples

    def get_data_by_task(self, data):
        # We're only considering everything with the same task hash to be the
        # same task to review and provide only two versions: accept and reject.
        examples = []
        by_task = defaultdict(list)
        for input_versions in data.values():
            for (task_hash, answer), task_versions in input_versions.items():
                if task_versions:
                    version = copy.deepcopy(task_versions[0])
                    sessions = sorted([eg[SESSION_ID_ATTR] for eg in task_versions])
                    version["sessions"] = sessions
                    by_task[task_hash].append(version)
        for versions in by_task.values():
            examples.append(self.make_eg(versions))
        return examples


def get_stream(datasets: Dict[str, List[dict]], default_view_id: Optional[str] = None):
    merged = defaultdict(dict)
    global_view_id = False
    n_merged = 0
    for set_id, examples in datasets.items():
        examples = (eg for eg in examples if eg["answer"] != "ignore")
        for eg in examples:
            # Rehash example to make sure we're comparing correctly. In this
            # case, we want to consider "options" an input key and "accept" a
            # task key, so we can treat choice examples as by_input. We also
            # want to ignore the answer and key by it separately.
            eg = set_hashes(
                eg,
                overwrite=True,
                input_keys=("text", "spans"),
                task_keys=("accept"),
                ignore=("score", "rank", "answer"),
            )
            
            # Make sure example has session ID (backwards compatibility)
            session_id = eg.get(SESSION_ID_ATTR, set_id)
            eg[SESSION_ID_ATTR] = session_id if session_id is not None else set_id
            # Make sure example has view ID (backwards compatibility)
            eg_view_id = eg.get(VIEW_ID_ATTR, default_view_id)
            if eg_view_id is None:
                msg.fail(
                    f"No '{eg_view_id}' found in the example",
                    "This is likely because it was created with Prodigy <1.8). "
                    "Please specify a --view-id on the command line. For "
                    "example, 'ner_manual' (if the annotations were created with "
                    "the manual interface), 'classification', 'choice' etc.",
                    exits=1,
                )
            if eg_view_id in ("image_manual", "compare", "diff"):
                msg.fail(
                    f"Reviewing '{eg_view_id}' annotations isn't supported yet",
                    "You can vote for this feature on the forum: https://support.prodi.gy",
                    exits=1,
                )
            eg[VIEW_ID_ATTR] = eg_view_id
            if global_view_id is False:
                global_view_id = eg_view_id
            if global_view_id != eg_view_id:
                msg.fail(
                    "Conflicting view_id values in datasets",
                    f"Can't review annotations of '{eg_view_id}' (in dataset "
                    f"'{set_id}') and '{global_view_id}' (in previous examples)",
                    exits=1,
                )
            input_hash = eg[INPUT_HASH_ATTR]
            key = (eg[TASK_HASH_ATTR], eg["answer"])
            merged[input_hash].setdefault(key, []).append(eg)
            n_merged += 1
            
    log(f"RECIPE: Merged {n_merged} examples from {len(datasets)} datasets")
    is_manual = global_view_id and global_view_id.endswith(("_manual", "choice"))
    stream = ReviewStream(merged, by_input=is_manual)

    return stream


@recipe(
    "review",
    # fmt: off
    dataset=("Dataset to save annotations to", "positional", None, str),
    input_sets=("Comma-separated names of datasets to review", "positional", None, split_string),
    view_id=("Default view_id (e.g. 'ner' or 'ner_manual') to use if none present in the task", "option", "v", str),
    label=("Comma-separated label(s) to annotate or text file with one label per line", "option", "l", get_labels),
    str_only=("Comma-separated label(s) to annotate or text file with one label per line", "option", "s", bool),
    # fmt: on
)
def review(
    dataset: str,
    input_sets: List[str],
    view_id: Optional[str] = None,
    label: Optional[List[str]] = None,
    str_only:  Optional[str] = False, 
) -> Dict[str, Any]:
    """Review existing annotations created by multiple annotators and
    resolve potential conflicts by creating one final "master annotation". Can
    be used for both binary and manual annotations. If the annotations were
    created with a manual interface, the "most popular" version will be
    pre-selected automatically.

    NOTE: If you're using this recipe with annotations created in Prodigy v1.7.1
    or lower, you'll need to define a --view-id argument with the annotation
    interface ID to use. For example, 'ner_manual' or 'classification'.
    """
    log("RECIPE: Starting recipe review", locals())
    DB = connect()
    for set_id in input_sets:
        if set_id not in DB:
            msg.fail(f"Can't find input dataset '{set_id}' in database", exits=1)
            
    all_examples = {set_id: DB.get_dataset(set_id) for set_id in input_sets}
    
    options_map = extract_options(all_examples)
    merged_annotations, annotations_by_hash = extract_severity_annotations(all_examples)

    # Given the input data, count and select number of annotations that have not been annotated
    print(f'Processing raw input data for cross-check analysis...')
    #input_data = load_input_datasets(input_data_paths)

    print(f'\n----Extracting unlabeled data----')
    #unlabeled_data = select_unlabeled_data(input_data, all_dat_dic)
    annotated_df = merged_annotations_to_df(merged_annotations, min_annotator_n = len(input_sets), colnames = input_sets)
    annotated_df = add_diffs(annotated_df)
    hash_ids_to_reannotate = annotated_df[annotated_df.diff_thresh == True].index
    
    
    print(f'{len(hash_ids_to_reannotate)} selected for reannotation' )
    
    all_examples_beyond_thresh = {}
    
    for set_id in all_examples.keys():
#         processed_egs = []
        all_examples_beyond_thresh[set_id] = []

        for eg in all_examples[set_id]:
            if eg['_input_hash'] in hash_ids_to_reannotate:
                if str_only:
                    eg['accept'] = [i for i in eg['accept'] if type(i) is str]
                all_examples_beyond_thresh[set_id].append(eg)
            
#                 processed_egs.append(eg)
                
        #all_examples_beyond_thresh[set_id] = [i for i in all_examples[set_id] if i['_input_hash'] in hash_ids_to_reannotate]


    stream = get_stream(all_examples_beyond_thresh, view_id)

    return {
        "view_id": "review",
        "dataset": dataset,
        "stream": stream,
        "config": {"labels": label} if label else {},
    }
