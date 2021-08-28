import prodigy
from prodigy.components.loaders import JSONL
from prodigy.models.textcat import TextClassifier
from prodigy.models.matcher import PatternMatcher
from prodigy.components.sorters import prefer_uncertain
from prodigy.util import combine_models, split_string, set_hashes
import spacy
from typing import List, Optional
from prodigy.components.preprocess import add_labels_to_stream
import ujson
from pathlib import Path
import copy
from prodigy.components.preprocess import add_tokens

# Recipe decorator with argument annotations: (description, argument type,
# shortcut, type / converter function called on value before it's passed to
# the function). Descriptions are also shown when typing --help.
@prodigy.recipe(
    "textcat.severity",
    dataset=("The dataset to use", "positional", None, str),
    #spacy_model=("The base model", "positional", None, str),
    source=("The source data as a JSONL file", "positional", None, str),
    options=("Path to options jsonl specifying options", "positional", None, str),
    label=("One or more comma-separated labels", "option", "l", split_string),
    patterns=("Optional match patterns", "option", "p", str),
    exclude=("Names of datasets to exclude", "option", "e", split_string),
    keymap=("Key mapping for labels", "option", 'km', str),
    keywords=("Keywords for positive filtering", 'option','kw', split_string)
)
def textcat_teach(
    dataset: str,
    #spacy_model: str,
    source: str,
    options: str,
    label: Optional[List[str]] = None,
    patterns: Optional[str] = None,
    exclude: Optional[List[str]] = None,
    keymap: Optional[List[str]] = None,
    keywords: Optional[List[str]] = None,
):
    """
    Collect the best possible training data for a text classification model
    with the model in the loop. Based on your annotations, Prodigy will decide
    which questions to ask next.
    """
    # Load the stream from a JSONL file and return a generator that yields a
    # dictionary for each example in the data.
    #stream = JSONL(source)
    
    #nlp = spacy.load(spacy_model)
    
    def get_options():
        task_options = []
        
        with Path(options).open('r', encoding='utf8') as f:
            for l in f:
                task_options.append(ujson.loads(l.strip()))
                
        return(task_options)


    
    def get_stream(): 
            db = prodigy.core.connect()
            hashes_in_dataset = db.get_input_hashes(dataset)
            
            eg_list = []
            
            with Path(source).open('r', encoding='utf8') as f:
                for line in f:
                    try:  # hack to handle broken jsonl
                        
                        eg = ujson.loads(line.strip())

                        eg['options'] = task_options
                        
                        eg = set_hashes(eg,
                        overwrite=True,
                        input_keys=("text", "spans"),
                        task_keys=("accept"),
                        ignore=("score", "rank", "answer"),
                        )
                        del eg['accept']
                        
                        if eg['_input_hash'] not in hashes_in_dataset:
                              
                            eg_list.append(eg)

                    except ValueError:
                        continue
            n_unique = len(list(set([i['_input_hash'] for i in eg_list])))
            print(f'N in stream: {len(eg_list)}')
            print(f"N unique: {n_unique}")
            return(eg_list) 
    
    
    
#     def get_stream_loop():
        
        
         
#         cur_stream = get_new_annotations()

#         for eg in cur_stream:
#             yield eg
            
#         db = prodigy.core.connect()
#         hashes_in_dataset = db.get_input_hashes(dataset)
        
#         n_stored_annotations = len(hashes_in_dataset)
        
#         if n_stored_annotations + len(cur_stream) >= n:
#             return(cur_stream)
           
        
#         else:
#             get_

#             eg in cur_stream:
#             yield eg
         
        
         
        
  
    def get_stream_loop():
        db = prodigy.core.connect()
       # n = get_n()
        #print(n)
        #counter = 0

        while not STOP:
            
            #cur_stream = add_tokens(nlp, cur_stream, skip=True)
            hashes_in_dataset = db.get_input_hashes(dataset)
            total_annotated = len(hashes_in_dataset)
            counter = total_annotated
            
          #  print(f'total annotated before stream refresh {total_annotated}')
            cur_stream = get_stream()

            total_annotated_plus_batch = total_annotated + batch_size
            
            if STOP:
                #print('here')
                break
            
#             if not STOP:
#                     cur_stream = get_stream()
#                     #cur_stream = [eg for eg in cur_stream if eg['_input_hash'] not in hashes_in_dataset]
#                     print(f'Total in stream: {len(cur_stream)}')

            yielded = False
            
            for eg in cur_stream:
               # print(f'Stop is set to: {STOP}')
#                 eg = set_hashes(eg,
#                 overwrite=False,
#                 input_keys=("text", "spans"),
#                 task_keys=("accept"),
#                 ignore=("score", "rank", "answer"),
#                 )
                # Only send out task if its hash isn't in the dataset yet
                if eg["_input_hash"] not in hashes_in_dataset:
                    if STOP is True:
                        yielded = False
                        break
                        
                    yielded = True
                    counter +=1
                    print(eg['_input_hash'])
                    yield eg
                    
            if not yielded:
                break
                        
    def get_n():
        n = 0
        with Path(source).open('r', encoding='utf8') as f:
                for line in f:
                    n+=1
        return(n)
        
#   
#     def progress(*args, **kwargs):
#         db = prodigy.core.connect()
#         n_cur = len(db.get_input_hashes(dataset))
#         #print(f'n_cur = {n_cur}')
#         return (n_cur)/(n)       
    
    def update(answers):
        db = prodigy.core.connect()
        hashes_in_dataset = db.get_input_hashes(dataset)
        answers = [eg for eg in answers if eg['_input_hash'] not in hashes_in_dataset]

        
        n_annotated = len(hashes_in_dataset) + len(answers)
       # print(f'n_annotated: {n_annotated}')

       # db.add_examples(answers, [dataset])
        
        
#         n_cur = len(db.get_input_hashes(dataset))
        
#         n_examined = len([a for a in answers])
        
        
        
        if n_annotated >= n:
            
            STOP = True
           
        pass
                        
    n = get_n()
    
    batch_size = 5
    STOP = False
    
    task_options = get_options()
    
#     stream = get_stream()
    stream = get_stream_loop()
    
    #FINISHED = False
 
   # stream = get_stream()
    
   # stream = add_tokens(nlp, stream, skip=True)
    
    return {
            "view_id": "choice",  # Annotation interface to use
            "dataset": dataset,  # Name of dataset to save annotations
            "stream": stream,  # Incoming stream of examples
            #"db": False,
          #  "progress": progress,
            "config": {#"batch_size": batch_size, 
                       "exclude_by": 'input',
                      # "force_stream_order": False, 
                       # "history_size": 5},
            },
            "update": update,
         #   "update": update,
            #"update": update,  # Update callback, called with batch of answers
            #"exclude": exclude,  # List of dataset names to exclude
#             "config": {"lang": 'en',
#                        "blocks": [#{"view_id": "ner_manual"},
#                                   {"view_id": "choice", "text": None}],
#                        "labels": label,
#                       },  # Additional config settings, mostly for app UI
        }
    
    #stream = add_labels_to_stream(stream, label)

    # Load the spaCy model
    #nlp = spacy.load(spacy_model)

    # Initialize Prodigy's text classifier model, which outputs
    # (score, example) tuples
    #model = TextClassifier(nlp, label)

#     if patterns is None:
#         # No patterns are used, so just use the model to suggest examples
#         # and only use the model's update method as the update callback
#         predict = model
#         update = model.update
#     else:
#         # Initialize the pattern matcher and load in the JSONL patterns.
#         # Set the matcher to not label the highlighted spans, only the text.
#         matcher = PatternMatcher(
#             nlp,
#             prior_correct=5.0,
#             prior_incorrect=5.0,
#             label_span=False,
#             label_task=True,
#         )
#         matcher = matcher.from_disk(patterns)
#         # Combine the NER model and the matcher and interleave their
#         # suggestions and update both at the same time
#         predict, update = combine_models(model, matcher)

    # Use the prefer_uncertain sorter to focus on suggestions that the model
    # is most uncertain about (i.e. with a score closest to 0.5). The model
    # yields (score, example) tuples and the sorter yields just the example
    #stream = prefer_uncertain(predict(stream))

#     if keymap is not None:
#         keymap_out = get_keymap()
        
#         return {
#             "view_id": "blocks",  # Annotation interface to use
#             "dataset": dataset,  # Name of dataset to save annotations
#             "stream": stream,  # Incoming stream of examples
#             #"update": update,  # Update callback, called with batch of answers
#             #"exclude": exclude,  # List of dataset names to exclude
#             "config": {"lang": 'en',
#                        "blocks": [{"view_id": "ner_manual"},
#                                   {"view_id": "choice", "text": None}],
#                        "labels": label,
#                        "keymap_by_label": {"All": "1", "Majority": "2", "Minority": "3", "Particular": "4", "Subgroup": "q", "Terrorist": "w", "Islam": "e"},
#                        }  # Additional config settings, mostly for app UI
#         }
    
#     else:
#         return {
#             "view_id": "blocks",  # Annotation interface to use
#             "dataset": dataset,  # Name of dataset to save annotations
#             "stream": stream,  # Incoming stream of examples
#             #"update": update,  # Update callback, called with batch of answers
#             #"exclude": exclude,  # List of dataset names to exclude
#             "config": {"lang": 'en',
#                        "blocks": [{"view_id": "ner_manual"},
#                                   {"view_id": "choice", "text": None}],
#                        "labels": label,
#                         "keymap_by_label": {"All": "1", "Majority": "2", "Minority": "3", "Particular": "4", "Subgroup": "q", "Terrorist": "w", "Islam": "e"},
#                        }  # Additional config settings, mostly for app UI
#         }
    
