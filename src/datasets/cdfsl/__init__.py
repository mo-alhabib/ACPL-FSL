from .episodic_cdfsl import EpisodicCDFSL

def get_cdfsl_dataset(dataset_name, data_root, split, transform,
                      classes, support_size, query_size, n_iters, unlabeled_size):
    # infer domain from suffix
    name = dataset_name.lower()
    if "chestx" in name:
        domain = "chestx"
    elif "crop" in name or "cropdiseases" in name:
        domain = "cropdiseases"
    elif "eurosat" in name:
        domain = "eurosat"
    elif "isic" in name:
        domain = "isic"
    else:
        domain = "generic"

    return EpisodicCDFSL(
        data_root=data_root, split=split, transform=transform,
        classes=classes, support_size=support_size, query_size=query_size,
        n_iters=n_iters, unlabeled_size=unlabeled_size, domain=domain
    )
