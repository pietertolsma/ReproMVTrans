from repromvtrans.model.multiview_net import MVNet

def factory(cfg):
    if cfg.model.selected == "MVNet":
        return MVNet(cfg)
    else:
        raise NotImplementedError(f"Model {cfg.model.name}")