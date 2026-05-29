from net.comparison_models import build_comparison_model
from net.sia_prompt_net_ablation import build_model as build_ablation_model
from net.sia_prompt_net_bdf import build_model as build_sia_stpnet


def build_unet(**kwargs):
    return build_comparison_model("unet", input_mode="center", **kwargs)


def build_unetpp(**kwargs):
    return build_comparison_model("unetpp", input_mode="center", **kwargs)


def build_deeplabv3plus(**kwargs):
    return build_comparison_model("deeplabv3plus", input_mode="center", **kwargs)


def build_unet3plus(**kwargs):
    return build_comparison_model("unet3plus", input_mode="center", **kwargs)


def build_attention_unet(**kwargs):
    return build_comparison_model("attention_unet", input_mode="center", **kwargs)


def build_transunet(**kwargs):
    return build_comparison_model("transunet", input_mode="center", **kwargs)


def build_swinunet(**kwargs):
    return build_comparison_model("swin_unet", input_mode="center", **kwargs)


def build_2_5d_unet(**kwargs):
    return build_comparison_model("2_5d_unet", input_mode="stack", **kwargs)


def build_siamese_encoder_decoder(**kwargs):
    kwargs.pop("base_ch", None)
    return build_ablation_model(ablation_name="baseline", deep_supervision=False, **kwargs)


def build_siamese_biconvlstm(**kwargs):
    kwargs.pop("base_ch", None)
    return build_ablation_model(ablation_name="biconvlstm", deep_supervision=True, **kwargs)


def build_siamese_stpnet(**kwargs):
    kwargs.pop("base_ch", None)
    return build_sia_stpnet(**kwargs)


MODEL_REGISTRY = {
    "unet": build_unet,
    "unetpp": build_unetpp,
    "deeplabv3plus": build_deeplabv3plus,
    "unet3plus": build_unet3plus,
    "attention_unet": build_attention_unet,
    "transunet": build_transunet,
    "swinunet": build_swinunet,
    "2_5d_unet": build_2_5d_unet,
    "siamese_encoder_decoder": build_siamese_encoder_decoder,
    "siamese_biconvlstm": build_siamese_biconvlstm,
    "siamese_stpnet": build_siamese_stpnet,
}


MODEL_REGISTRY.update({
    "unet_3plus": build_unet3plus,
    "attention_unet": build_attention_unet,
    "swin_unet": build_swinunet,
    "trans_unet": build_transunet,
})


def build_registered_model(model_name, cfg):
    if model_name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{model_name}'. Available: {sorted(MODEL_REGISTRY)}")
    model_cfg = dict(cfg.get("models", {}).get(model_name, {}))
    base_ch = int(model_cfg.get("base_ch", cfg.get("model", {}).get("base_ch", 32)))
    return MODEL_REGISTRY[model_name](
        window_size=int(cfg["data"]["window_size"]),
        image_size=(int(cfg["data"]["img_size"]), int(cfg["data"]["img_size"])),
        num_classes=int(cfg["model"].get("num_classes", 1)),
        input_channels=int(cfg["data"].get("input_channels", 1)),
        base_ch=base_ch,
    )
