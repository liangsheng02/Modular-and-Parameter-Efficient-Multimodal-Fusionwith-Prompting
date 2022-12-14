from .base_dataset import BaseDataset
from .glossary import normalize_word
import re, tqdm, json
from typing import Sequence
import torch


class VQACollator:
    def __init__(self, feature_extractor, tokenizer, blacked=False):
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.blacked = blacked

    def __call__(self, batch):
        keys = set([key for b in batch for key in b.keys()])
        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}
        new_dict_batch = {}

		new_img = self.feature_extractor(dict_batch[img_key], return_tensors="pt")
		new_dict_batch["vis_feas"] = new_img["pixel_values"]

        txt_key = "text"
        new_text = self.tokenizer(dict_batch[txt_key], padding=True, return_tensors="pt")
        new_dict_batch["input_ids"] = new_text.input_ids
        new_dict_batch["attention_mask"] = new_text.attention_mask

        label_key = "labels"
        if label_key in keys:
            # [t.capitalize()+"." for t in dict_batch[label_key]]
            new_label = self.tokenizer(dict_batch[label_key], padding=True, return_tensors="pt").input_ids
            new_dict_batch["labels"] = new_label
        return new_dict_batch


class VQAv2Dataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split

        if split == "train":
            names = ["vqav2_train"]
        elif split == "val":
            names = ["vqav2_val"]
        elif split == "test":
            names = ["vqav2_test"]  # vqav2_test-dev for test-dev

        super().__init__(
            *args,
            **kwargs,
            names=names,
            text_column_name="questions",
            remove_duplicate=False,
        )

        # id2datum
        if self.split != "test":
            self.id2datum = {}
            for index in range(self.__len__()):
                index, question_index = self.index_mapper[index]
                qid = self.table["question_id"][index][question_index].as_py()
                answers = self.table["answers"][index][question_index].as_py()
                question_type = self.table["question_type"][index][question_index].as_py()
                answer_type = self.table["answer_type"][index][question_index].as_py()
                self.id2datum[qid] = {'answers': answers,
                                      'question_type': question_type,
                                      'answer_type': answer_type}

    def __getitem__(self, index):
        image_tensor = self.get_image(index)["image"]
        text = self.get_text(index)["text"]

        index, question_index = self.index_mapper[index]
        qid = self.table["question_id"][index][question_index].as_py()

        if self.split != "test":
            answers = self.table["answers"][index][question_index].as_py()
            labels = self.table["labels"][index][question_index].as_py()
            question_type = self.table["question_type"][index][question_index].as_py()
            answer_type = self.table["answer_type"][index][question_index].as_py()
        else:
            answers = list()
            labels = list()
            question_type = list()
            answer_type = list()

        return {
            "image": image_tensor,
            "text": text,
            "answers": answers,
            "labels": labels,
            "question_type": question_type,
            "answer_type": answer_type,
            "qid": qid,
        }


class VQAEvaluator:
    def __init__(self, dataset: VQAv2Dataset = None, predictions=[]):
        """https://github.com/GT-Vision-Lab/VQA/blob/master/PythonEvaluationTools/vqaEvaluation/vqaEval.py"""
        self.dataset = dataset
        assert len(dataset) == len(predictions)
        self.n = 2
        # id2pred dict
        self.quesid2pred = {}
        for i in range(len(dataset)):
            self.quesid2pred[dataset[i]["qid"]] = normalize_word(predictions[i])

    def get_id2datum(self):
        # id2datum dict
        # {qid: answers}
        # answers = [{'answer': 'yes', 'answer_id': 1}, ..., {'answer': 'yes', 'answer_id': 10}]
        id2datum = {}
        for i in range(len(self.dataset)):
            id2datum[self.dataset[i]['qid']] = {'answers': self.dataset[i]['answers'],
                                                'question_type': self.dataset[i]['question_type'],
                                                'answer_type': self.dataset[i]['answer_type']}
        return id2datum

    def dump_result(self, path):
        quesid2ans = self.quesid2pred
        with open(path, 'w') as f:
            result = []
            for ques_id, ans in quesid2ans.items():
                result.append({
                    'question_id': ques_id,
                    'answer': ans
                })
            json.dump(result, f, indent=4, sort_keys=True)

    def evaluate_raw(self):
        """https://github.com/GT-Vision-Lab/VQA/blob/master/PythonEvaluationTools/vqaEvaluation/vqaEval.py"""
        quesid2ans = self.quesid2pred
        gts = self.dataset.id2datum if hasattr(self.dataset, "id2datum") else self.get_id2datum()

        self.accuracy     = {}
        self.evalQA       = {}
        self.evalQuesType = {}
        self.evalAnsType  = {}

        accQA = []
        accQuesType = {}
        accAnsType = {}

        for quesId, resAns in quesid2ans.items():
            quesId = int(quesId)
            gtAcc  = []
            for gtAnsDatum in gts[quesId]['answers']:
                otherGTAns = [item for item in gts[quesId]['answers'] if item!=gtAnsDatum]
                matchingAns= [item for item in otherGTAns if item['answer']==resAns]
                acc = min(1, float(len(matchingAns))/3)
                gtAcc.append(acc)
            quesType    = gts[quesId]['question_type']
            ansType     = gts[quesId]['answer_type']
            avgGTAcc    = float(sum(gtAcc))/len(gtAcc)
            accQA.append(avgGTAcc)
            if quesType not in accQuesType:
                accQuesType[quesType] = []
            accQuesType[quesType].append(avgGTAcc)
            if ansType not in accAnsType:
                accAnsType[ansType] = []
            accAnsType[ansType].append(avgGTAcc)

            self.setEvalQA(quesId, avgGTAcc)
            self.setEvalQuesType(quesId, quesType, avgGTAcc)
            self.setEvalAnsType(quesId, ansType, avgGTAcc)

        if len(accQA) == 0:
            return {
                'overall': 0,
                'perQuestionType': {},
                'perAnswerType': {}
            }
        else:
            self.setAccuracy(accQA, accQuesType, accAnsType)
        return self.accuracy

    def setEvalQA(self, quesId, acc):
        self.evalQA[quesId] = round(100*acc, self.n)

    def setEvalQuesType(self, quesId, quesType, acc):
        if quesType not in self.evalQuesType:
            self.evalQuesType[quesType] = {}
        self.evalQuesType[quesType][quesId] = round(100*acc, self.n)

    def setEvalAnsType(self, quesId, ansType, acc):
        if ansType not in self.evalAnsType:
            self.evalAnsType[ansType] = {}
        self.evalAnsType[ansType][quesId] = round(100*acc, self.n)

    def setAccuracy(self, accQA, accQuesType, accAnsType):
        self.accuracy['overall']         = round(100*float(sum(accQA))/len(accQA), self.n)
        self.accuracy['perQuestionType'] = {quesType: round(100*float(sum(accQuesType[quesType]))/len(accQuesType[quesType]), self.n) for quesType in accQuesType}
        self.accuracy['perAnswerType']   = {ansType:  round(100*float(sum(accAnsType[ansType]))/len(accAnsType[ansType]), self.n) for ansType in accAnsType}
