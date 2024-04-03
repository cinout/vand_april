import os
from typing import Union, List
from pkg_resources import packaging
import torch
import numpy as np


# TODO: update for LOCO
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
    for obj in objs:
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
            prompted_sentence = tokenizer(prompted_sentence).to(device)
            class_embeddings = model.encode_text(prompted_sentence)
            class_embeddings /= class_embeddings.norm(
                dim=-1, keepdim=True
            )  # [245, 768]

            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()  # [768]

            text_features.append(class_embedding)

        text_features = torch.stack(text_features, dim=1).to(device)  # [768, 2]
        text_prompts[obj] = text_features

    return text_prompts
