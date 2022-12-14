import json
import pandas as pd
import pyarrow as pa
import random
import os

from tqdm import tqdm
from glob import glob
from collections import defaultdict, Counter
from .glossary import normalize_word


def path2rest(path, split, annotations):
    iid = int(path.split("/")[-1].split("_")[-1][:-4])

    with open(path, "rb") as fp:
        binary = fp.read()

    _annot = annotations[split][iid]
    _annot = list(_annot.items())
    qids, qas = [a[0] for a in _annot], [a[1] for a in _annot]
    questions = [qa[0] for qa in qas]
    answers = [qa[1] for qa in qas] if "test" not in split else list(list())
    labels = (
        [a["multiple_choice_answer"] for a in answers] if "test" not in split else list(list())
    )
    answer_type = (
        [a["answer_type"] for a in answers] if "test" not in split else list(list())
    )
    question_type = (
        [a["question_type"] for a in answers] if "test" not in split else list(list())
    )
    answers = (
        [a["answers"] for a in answers] if "test" not in split else list(list())
    )
    return [binary, questions, labels, answers, question_type, answer_type, iid, qids, split]


def make_arrow(root, dataset_root):
    with open(f"{root}/v2_OpenEnded_mscoco_train2014_questions.json", "r") as fp:
        questions_train2014 = json.load(fp)["questions"]
    with open(f"{root}/v2_OpenEnded_mscoco_val2014_questions.json", "r") as fp:
        questions_val2014 = json.load(fp)["questions"]
    with open(f"{root}/v2_OpenEnded_mscoco_test2015_questions.json", "r") as fp:
        questions_test2015 = json.load(fp)["questions"]
    with open(f"{root}/v2_OpenEnded_mscoco_test-dev2015_questions.json", "r") as fp:
        questions_test_dev2015 = json.load(fp)["questions"]

    with open(f"{root}/v2_mscoco_train2014_annotations.json", "r") as fp:
        annotations_train2014 = json.load(fp)["annotations"]
    with open(f"{root}/v2_mscoco_val2014_annotations.json", "r") as fp:
        annotations_val2014 = json.load(fp)["annotations"]

    annotations = dict()

    for split, questions in zip(
        ["train", "val", "test", "test-dev"],
        [
            questions_train2014,
            questions_val2014,
            questions_test2015,
            questions_test_dev2015,
        ],
    ):
        _annot = defaultdict(dict)
        for q in tqdm(questions):
            _annot[q["image_id"]][q["question_id"]] = [q["question"]]

        annotations[split] = _annot

    for split, annots in zip(
        ["train", "val"], [annotations_train2014, annotations_val2014],
    ):
        _annot = annotations[split]
        for q in tqdm(annots):
            question_type = q["question_type"]
            multiple_choice_answer = normalize_word(q["multiple_choice_answer"])
            answer_type = q["answer_type"]
            answers = [{'answer': normalize_word(ans['answer']), 'answer_id': ans['answer_id']}
                       for ans in q["answers"]]   # ans = {'answer': 'yes', 'answer_confidence': 'yes', 'answer_id': 1}

            _annot[q["image_id"]][q["question_id"]].append(
                {"answers": answers,
                 "multiple_choice_answer": multiple_choice_answer,
                 "answer_type": answer_type,
                 "question_type": question_type}
            )

    for split in ["train", "val"]:
        filtered_annot = dict()
        for ik, iv in annotations[split].items():
            new_q = dict()
            for qk, qv in iv.items():
                if len(qv[1]["multiple_choice_answer"]) != 0:
                    new_q[qk] = qv
            if len(new_q) != 0:
                filtered_annot[ik] = new_q
        annotations[split] = filtered_annot

    for split in [
        "train",
        "val",
        "test",
        "test-dev",
    ]:
        annot = annotations[split]
        split_name = {
            "train": "train2014",
            "val": "val2014",
            "test": "test2015",
            "test-dev": "test2015",
        }[split]
        paths = list(glob(f"{root}/{split_name}/*.jpg"))
        random.shuffle(paths)
        annot_paths = [
            path
            for path in paths
            if int(path.split("/")[-1].split("_")[-1][:-4]) in annot
        ]

        if len(paths) == len(annot_paths):
            print("all images have caption annotations")
        else:
            print("not all images have caption annotations")
        print(
            len(paths), len(annot_paths), len(annot),
        )

        bs = [
            path2rest(path, split, annotations) for path in tqdm(annot_paths)
        ]

        dataframe = pd.DataFrame(
            bs,
            columns=[
                "image",
                "questions",
                "labels",
                "answers",
                "question_type",
                "answer_type",
                "image_id",
                "question_id",
                "split",
            ],
        )

        table = pa.Table.from_pandas(dataframe)

        os.makedirs(dataset_root, exist_ok=True)
        with pa.OSFile(f"{dataset_root}/vqav2_{split}.arrow", "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)
