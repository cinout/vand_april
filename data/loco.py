import os
import json


class LOCOSolver(object):
    CLSNAMES = [
        "breakfast_box",
        "juice_bottle",
        "pushpins",
        "screw_bag",
        "splicing_connectors",
    ]

    def __init__(self, root="data/loco"):
        self.root = root
        self.meta_path = f"{root}/meta.json"

    def run(self):
        info = dict(train={}, test={})
        for cls_name in self.CLSNAMES:
            cls_dir = f"{self.root}/{cls_name}"
            for phase in ["train", "test"]:
                cls_info = []
                species = os.listdir(f"{cls_dir}/{phase}")
                species = list(filter(lambda x: not x.startswith("."), species))

                for specie in species:  # good, logical_anomalies, struc....
                    is_abnormal = True if specie not in ["good"] else False
                    img_names = os.listdir(f"{cls_dir}/{phase}/{specie}")
                    img_names.sort()

                    mask_names_dir = (
                        os.listdir(f"{cls_dir}/ground_truth/{specie}")
                        if is_abnormal
                        else None
                    )  # directories
                    mask_names_dir = (
                        list(filter(lambda x: not x.startswith("."), mask_names_dir))
                        if is_abnormal
                        else None
                    )

                    mask_names_dir.sort() if mask_names_dir is not None else None

                    for idx, img_name in enumerate(img_names):
                        mask_names = []
                        if is_abnormal:
                            mask_folder = f"{cls_name}/ground_truth/{specie}/{mask_names_dir[idx]}"
                            mask_names = os.listdir(
                                os.path.join(self.root, mask_folder)
                            )
                            mask_names = list(
                                filter(lambda x: not x.startswith("."), mask_names)
                            )
                            mask_names.sort()
                            mask_names = [
                                os.path.join(mask_folder, item) for item in mask_names
                            ]

                        info_img = dict(
                            img_path=f"{cls_name}/{phase}/{specie}/{img_name}",
                            mask_path=mask_names,
                            cls_name=cls_name,
                            specie_name=specie,
                            anomaly=1 if is_abnormal else 0,
                        )
                        cls_info.append(info_img)
                info[phase][cls_name] = cls_info
        with open(self.meta_path, "w") as f:
            f.write(json.dumps(info, indent=4) + "\n")


if __name__ == "__main__":
    runner = LOCOSolver(root="data/loco")
    runner.run()
