import torch
from dataset import VisaDataset, MVTecDataset, LOCODataset


def memory(
    model_name,
    model,
    obj_list,
    dataset_dir,
    save_path,
    preprocess,
    transform,
    k_shot,
    few_shot_features,
    dataset_name,
    device,
):
    mem_features = {}
    for obj in obj_list:
        if dataset_name == "mvtec":
            data = MVTecDataset(
                root=dataset_dir,
                transform=preprocess,
                target_transform=transform,
                aug_rate=-1,
                mode="train",
                k_shot=k_shot,
                save_dir=save_path,
                obj_name=obj,
            )
        elif dataset_name == "loco":
            data = LOCODataset(
                root=dataset_dir,
                transform=preprocess,
                target_transform=transform,
                mode="train",
                k_shot=k_shot,
                save_dir=save_path,
                obj_name=obj,
            )
        else:
            data = VisaDataset(
                root=dataset_dir,
                transform=preprocess,
                target_transform=transform,
                mode="train",
                k_shot=k_shot,
                save_dir=save_path,
                obj_name=obj,
            )
        dataloader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False)
        features = []
        for items in dataloader:  # for each shot
            image = items["img"].to(device)
            with torch.no_grad():
                image_features, patch_tokens = model.encode_image(
                    image, few_shot_features
                )  # patch_tokens.len: 4 (4 stages), each shape [1, 1370, 1024]

                if "ViT" in model_name:
                    patch_tokens = [p[0, 1:, :] for p in patch_tokens]
                    # patch_tokens.len: 4 (4 stages), each shape [1369, 1024]
                else:
                    patch_tokens = [
                        p[0].view(p.shape[1], -1).permute(1, 0).contiguous()
                        for p in patch_tokens
                    ]
                features.append(patch_tokens)

        mem_features[obj] = [
            torch.cat([features[j][i] for j in range(len(features))], dim=0)
            for i in range(len(features[0]))
        ]  # i: each stage; j: each shot; purpose: at each stage i, concatenate the patch_tokens of all shots j together
        # mem_features[obj].len is #stages, each with shape: [5476, 1024]

    return mem_features
