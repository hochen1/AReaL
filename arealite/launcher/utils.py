from omegaconf import DictConfig, OmegaConf

def find_config(config: DictConfig, name: str) -> DictConfig | None:
    # iterate through the nested DictConfig and find the first matching config with name
    for key, value in config.items():
        if key == name:
            return value
        if isinstance(value, DictConfig):
            found = find_config(value, name)
            if found:
                return found
    return None

def amend_config(config: DictConfig, config_cls):
    default_config = OmegaConf.structured(config_cls)
    config = OmegaConf.merge(default_config, config)
    config = OmegaConf.to_object(config)
    assert isinstance(config, config_cls)
    return config

def find_and_amend_config(config: DictConfig, name: str, config_cls):
    # Find the config with the given name and amend it with the given config_cls
    found = find_config(config, name)
    if found is not None:
        return amend_config(found, config_cls)
    return None