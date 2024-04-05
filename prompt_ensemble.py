import os
from typing import Union, List
from pkg_resources import packaging
import torch
import numpy as np


def tokenizing_sentences(prompted_sentence, tokenizer, model, device):
    prompted_sentence = tokenizer(prompted_sentence).to(device)
    class_embeddings = model.encode_text(prompted_sentence)
    class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)  # [245, 768]

    class_embedding = class_embeddings.mean(dim=0)
    class_embedding /= class_embedding.norm()  # [768]
    return class_embedding


def encode_text_with_prompt_ensemble(model, objs, tokenizer, device):
    prompt_normal = [
        "{}",
        "flawless {}",
        "perfect {}",
        "unblemished {}",
        "{} without flaw",
        "{} without defect",
        "{} without damage",
    ]
    prompt_abnormal = [
        "damaged {}",
        "broken {}",
        "{} with flaw",
        "{} with defect",
        "{} with damage",
    ]
    prompt_state = [prompt_normal, prompt_abnormal]
    prompt_templates = [
        "a bad photo of a {}.",
        "a low resolution photo of the {}.",
        "a bad photo of the {}.",
        "a cropped photo of the {}.",
        "a bright photo of a {}.",
        "a dark photo of the {}.",
        "a photo of my {}.",
        "a photo of the cool {}.",
        "a close-up photo of a {}.",
        "a black and white photo of the {}.",
        "a bright photo of the {}.",
        "a cropped photo of a {}.",
        "a jpeg corrupted photo of a {}.",
        "a blurry photo of the {}.",
        "a photo of the {}.",
        "a good photo of the {}.",
        "a photo of one {}.",
        "a close-up photo of the {}.",
        "a photo of a {}.",
        "a low resolution photo of a {}.",
        "a photo of a large {}.",
        "a blurry photo of a {}.",
        "a jpeg corrupted photo of the {}.",
        "a good photo of a {}.",
        "a photo of the small {}.",
        "a photo of the large {}.",
        "a black and white photo of a {}.",
        "a dark photo of a {}.",
        "a photo of a cool {}.",
        "a photo of a small {}.",
        "there is a {} in the scene.",
        "there is the {} in the scene.",
        "this is a {} in the scene.",
        "this is the {} in the scene.",
        "this is one {} in the scene.",
    ]

    text_prompts = {}
    for obj in objs:  # for each category
        text_features = []
        for i in range(len(prompt_state)):  # normal, abnormal
            # FIXME: replace _ with blank
            prompted_state = [
                state.format(obj.replace("_", " ")) for state in prompt_state[i]
            ]
            prompted_sentence = []  # complete sentences
            for s in prompted_state:
                for template in prompt_templates:
                    prompted_sentence.append(template.format(s))

            class_embedding = tokenizing_sentences(
                prompted_sentence, tokenizer, model, device
            )

            text_features.append(class_embedding)

        text_features = torch.stack(text_features, dim=1).to(device)  # [768, 2]
        text_prompts[obj] = text_features

    return text_prompts


def encode_text_with_LOCO(model, objs, tokenizer, loco_template, device):
    #### templates for STRUCTURAL information ####
    items_in_loco_image = {
        "breakfast_box": [
            "orange",
            "nectarine",
            "peach",
            "cereal",
            "banana chips and almond mix",
            "white food box",
        ],
        "juice_bottle": [
            "white banana juice",
            "yellow orange juice",
            "red cherry juice",
            "banana label",
            "orange label",
            "cherry label",
            "text saying 100% juice",
            "bottle",
        ],
        "pushpins": ["pushpin", "box with dividers"],
        "screw_bag": [
            "long screw",
            "short screw",
            "washer",
            "nut hardware",
            "zip lock plastic bag",
        ],
        "splicing_connectors": [
            "red cable",
            "yellow cable",
            "blue cable",
            "cable clamps",
        ],
    }

    prompt_structural_normal = [
        "flawless {}",
        "perfect {}",
        "unblemished {}",
        "{} without flaw",
        "{} without defect",
        "{} without damage",
    ]
    prompt_structural_abnormal = [
        "damaged {}",
        "broken {}",
        "{} with flaw",
        "{} with defect",
        "{} with damage",
    ]
    prompt_structural_state = [prompt_structural_normal, prompt_structural_abnormal]
    prompt_structural_templates = [
        "there is a {} in the scene.",
        "there is the {} in the scene.",
        "there is some {} in the scene.",
        "there is a {} in the photo.",
        "there is the {} in the photo.",
        "there is some {} in the photo.",
    ]

    #### templates for LOGICAL information ####
    rules_in_loco = {
        "breakfast_box": [
            "the photo shows a food box",
            "there are two oranges on the left",
            "there is one nectarine on the left",
            "there is cereal on the upper right",
            "there is banana chips and almond mix on the lower right",
        ],
        "juice_bottle": [
            "the photo shows a juice bottle",
            "the fruit label is at the center of the bottle",
            "the text saying 100% juice is at the bottom of the bottle",
            "the red juice has cherry label",
            "the white juice has banana label",
            "the yellow juice has orange label",
            "the juice is filled to the bottle's neck",
        ],
        "pushpins": [
            "the photo shows a box of fifteen grids",
            "each grid has exactly one pushpin",
        ],
        "screw_bag": [
            "the photo shows a zip lock plastic bag",
            "there is one long screw in the bag",
            "there is one short screw in the bag",
            "there are two washers in the bag",
            "there are two nut hardware in the bag",
        ],
        "splicing_connectors": [
            "the photo shows two splicing connectors linked by one continuous cable",
            "the red cable connects two splicing connectors with five cable clamps",
            "the blue cable connects two splicing connectors with three cable clamps",
            "the yellow cable connects two splicing connectors with two cable clamps",
            "the left and right splicing connectors have the same number of cable clamps",
            "the cable terminates in the same relative position on the two splicing connectors",
        ],
    }

    text_prompts = {}
    for obj in objs:
        text_features = []

        ### STRUCTURAL ###
        for i in range(len(prompt_structural_state)):  # normal, abnormal [STRUCTURAL]
            prompted_sentence = []  # complete sentences for STRUCTURAL
            for fine_grained_item in items_in_loco_image[obj]:  # orange, nectarine, ...
                prompted_state = [
                    state.format(fine_grained_item)
                    for state in prompt_structural_state[i]
                ]
                for s in prompted_state:
                    for template in prompt_structural_templates:
                        prompted_sentence.append(template.format(s))

            class_embedding = tokenizing_sentences(
                prompted_sentence, tokenizer, model, device
            )

            text_features.append(
                class_embedding
            )  # text embeddings of: [normal, abnormal], STRUCTURAL

        ### LOGICAL ###
        if loco_template == "v1":
            logical_rules = rules_in_loco[obj]
            class_embedding_normal_logical = tokenizing_sentences(
                logical_rules, tokenizer, model, device
            )
            # merge logical embeddings into structural embeddings
            text_features[0] = torch.mean(
                torch.stack([class_embedding_normal_logical, text_features[0]], dim=0),
                dim=0,
            )

        if loco_template == "v2":
            # TODO:
            pass

        ### FINALIZE ###
        text_features = torch.stack(text_features, dim=1).to(device)  # [768, 2]
        text_prompts[obj] = text_features

    return text_prompts
