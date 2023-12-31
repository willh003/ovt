import logging
from typing import Tuple
import os

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import clip

from detectron2.config import configurable, get_cfg
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList
from detectron2.utils.logger import log_first_n
from .modeling.clip_adapter import (
    ClipAdapter,
    MaskFormerClipAdapter,
    build_text_prompt,
)
from .mask_former_model import MaskFormer
from .utils.misc import get_gt_binary_masks


from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.engine.defaults import DefaultPredictor

if os.getcwd().split('/')[-1] == 'ovseg':
    # Playground
    from open_vocab_seg import add_ovseg_config
    from open_vocab_seg.utils import VisualizationDemo
    from open_vocab_seg.ovseg_model import OVSeg
    from open_vocab_seg.modeling.clip_adapter.utils import build_clip_model
else:
    # ROS
    from modules.ovseg.open_vocab_seg.modeling.clip_adapter.utils import build_clip_model
    from modules.ovseg.open_vocab_seg import add_ovseg_config
    from modules.ovseg.open_vocab_seg.utils import VisualizationDemo
    from modules.ovseg.open_vocab_seg.ovseg_model import OVSeg


class WSImageEncoder(DefaultPredictor):
    def __init__(self, root_dir=None, ovseg_dir='modules/ovseg', config='configs/ovt.yaml', use_large=True):
        """
        Inputs:
            root_dir: the directory from which python is being called (PYTHON_PATH)
            ovseg_dir: the relative installation path of ovseg relative to root_dir
        """

        # handle case where ovseg is called from outside the root
        # main use case is for ros
        self.use_large = use_large
        if use_large:
            weights_file= 'models/ovseg_swinbase_vitL14_ft_mpt.pth'
        else:
            weights_file = 'models/ovseg_R101c_vitB16_ft_mpt.pth.pt'
        if root_dir:
            config_path = os.path.join(root_dir, ovseg_dir, config) 
            weights_path = os.path.join(root_dir, ovseg_dir, weights_file)
        else:
            weights_path = weights_file
            config_path = config
        
        opts = ['MODEL.WEIGHTS', weights_path]
        cfg = self.setup_cfg(config_path, opts)

        super().__init__(cfg)
        self.text_model = WSTextFeatureModel(cfg.MODEL.CLIP_ADAPTER.CLIP_MODEL_NAME, cfg.MODEL.CLIP_ADAPTER.MASK_PROMPT_DEPTH)

    def setup_cfg(self, config_file, opts):
        cfg = get_cfg()
        # for poly lr schedule
        add_deeplab_config(cfg)
        add_ovseg_config(cfg)
        cfg.merge_from_file(config_file)
        cfg.merge_from_list(opts)
        cfg.freeze()
        return cfg

    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            
            predictions = self.model(image[None])
            
            
            #predictions = self.model([inputs])
            return predictions
    
    def batch_call(self, images):
        """
        Runs the model on a batch of images
            images: torch.tensor, (batch, channels, height, width)
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            predictions = self.model(images)
            return predictions
        
    def call_with_classes(self, images, classes, use_adapter=False) -> torch.tensor:
        """
        Runs the model on a batch of images, and uses the clip adaptor for classes
            images: torch.tensor, (batch, channels, height, width)
            classes: list of strings
        Returns:
            class_probs: torch.tensor, (batch, class_num, height, width)
        """
        with torch.no_grad():
            if self.use_large:
                class_probs = self.model(images, classes, self.text_model,768, use_adapter)
            else:
                class_probs = self.model(images, classes, self.text_model,512, use_adapter)
            return class_probs

    def image_list_to_tensor(self, images):
        """
        images: Arraylike, containing ndarray images in "BGR" format
        """
        altered_images = []

        for i in range(len(images)):
            original_image = images[i]

            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]

            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            altered_images.append(image)

        torch_images = torch.stack(altered_images)
        return torch_images




class WSTextFeatureModel:

    def __init__(self, model_name, mask_prompt_depth):
        '''
        model_name from cfg.MODEL.CLIP_ADAPTER.CLIP_MODEL_NAME
        mask_prompt_depth from cfg.MODEL.CLIP_ADAPTER.MASK_PROMPT_DEPTH,
        '''
        self.clip_model = build_clip_model(model_name, mask_prompt_depth)
    
        self.templates = [
            "a photo of a {}.",
            "This is a photo of a {}",
            "There is a {} in the scene",
            "There is the {} in the scene",
            "a photo of a {} in the scene",
            "a photo of a small {}.",
            "a photo of a medium {}.",
            "a photo of a large {}.",
            "This is a photo of a small {}.",
            "This is a photo of a medium {}.",
            "This is a photo of a large {}.",
            "There is a small {} in the scene.",
            "There is a medium {} in the scene.",
            "There is a large {} in the scene.",
        ]

    def get_text_features(self, text):
        """
        If text is a dict, returns features for each key in the dict (assumes the values are lists of prompts)
        Else if it is a list, returns features for each class in the list
        """
        if type(text) == dict:
            return self.get_prompt_features(text)
        elif type(text) == list:
            return self.get_class_features(text)
        else:
            raise Exception('Wrong text type input')

    def get_logits(self, embeddings, classes, temperature=100, manual_prompts=False):
        """
        embeddings: torch.tensor, (N, embed_dim) 
        classes: list of nouns that represent classes of interest, or dictionary of classes to corresponding prompts
        manual_prompts: True to specify prompts instead of just class names

        """
        if manual_prompts:
            # WARNING: cheeky reassignment of classes 
            class_features = self.get_prompt_features(classes)
        else:
            class_features = self.get_class_features(classes)
        normalized_class_features = self.normalize_feature(class_features)
        normalized_embeddings = self.normalize_feature(embeddings)

        # compute cosine similarities
        # NOTE: it seems like they don't normalize the text features (only the embeddings), but this feels wrong because it should be cos similarity
        similarities = temperature * normalized_embeddings @ normalized_class_features.T 
        return similarities


    def get_nearest_classes(self, embeddings,classes,temperature=100,manual_prompts=False):
        """
        embeddings: torch.tensor, (N, embed_dim) 
        classes: list of nouns that represent classes of interest
        """
        logits = self.get_logits(embeddings, classes, temperature, manual_prompts)
        bests = torch.argmax(logits, dim=1) 
        
        return bests

    def normalize_feature(self, feat):
            
        return feat / feat.norm(dim=-1, keepdim=True)

    def get_prompt_features(self, class_prompts, device='cuda'):
        """
        Inputs: 
            class_prompts: dictionary of {class id: [prompts corresponding to class]}
        
        Returns:
            classes, features (average pooling over prompts)
        """

        if len(class_prompts) == 1:
            class_prompts['other'] = ['other', 'miscellaneous']
        
        text_features_bucket = []
        for class_name in class_prompts.keys():
            tokens = [clip.tokenize(prompt for prompt in class_prompts[class_name])]
            text_inputs =torch.cat(tokens).to(
                self.clip_model.text_projection.data.device
            )
            text_features = self.clip_model.encode_text(text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            text_features_bucket.append(text_features)

        del text_inputs
        # ensemble by averaging
        text_features = torch.stack([t.mean(dim=0) for t in text_features_bucket])
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features.to(device)

    def get_class_features(self, noun_list, device='cuda'):
        """
        Inputs:
            noun_list: list of class names
        """

        if len(noun_list) == 1:
            noun_list.append('other') # ensure contrastive feedback

        text_features_bucket = []
        for template in self.templates:
            noun_tokens = [clip.tokenize(template.format(noun)) for noun in noun_list]
            
            text_inputs = torch.cat(noun_tokens).to(
                self.clip_model.text_projection.data.device
            )
            text_features = self.clip_model.encode_text(text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            text_features_bucket.append(text_features)
        del text_inputs
        # ensemble by averaging
        text_features = torch.stack(text_features_bucket).mean(dim=0)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.to(device)

@META_ARCH_REGISTRY.register()
class OVTArch(MaskFormer):
    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        clip_adapter: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        panoptic_on: bool,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        clip_ensemble: bool,
        clip_ensemble_weight: float,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            clip_adapter: adapter for clip-based mask classification
            num_queries: int, number of queries
            panoptic_on: bool, whether to output panoptic segmentation prediction
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
        """
        super().__init__(
            backbone=backbone,
            sem_seg_head=sem_seg_head,
            criterion=criterion,
            num_queries=num_queries,
            panoptic_on=panoptic_on,
            object_mask_threshold=object_mask_threshold,
            overlap_threshold=overlap_threshold,
            metadata=metadata,
            size_divisibility=size_divisibility,
            sem_seg_postprocess_before_inference=sem_seg_postprocess_before_inference,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
        )
        self.clip_adapter: ClipAdapter = clip_adapter

        self.clip_ensemble: bool = clip_ensemble
        self.clip_ensemble_weight: float = clip_ensemble_weight

    @classmethod
    def from_config(cls, cfg):
        init_kwargs = MaskFormer.from_config(cfg)
        text_templates = build_text_prompt(cfg.MODEL.CLIP_ADAPTER)

        clip_adapter = MaskFormerClipAdapter(
            cfg.MODEL.CLIP_ADAPTER.CLIP_MODEL_NAME,
            text_templates,
            mask_fill=cfg.MODEL.CLIP_ADAPTER.MASK_FILL,
            mask_expand_ratio=cfg.MODEL.CLIP_ADAPTER.MASK_EXPAND_RATIO,
            mask_thr=cfg.MODEL.CLIP_ADAPTER.MASK_THR,
            mask_matting=cfg.MODEL.CLIP_ADAPTER.MASK_MATTING,
            region_resized=cfg.MODEL.CLIP_ADAPTER.REGION_RESIZED,
            mask_prompt_depth=cfg.MODEL.CLIP_ADAPTER.MASK_PROMPT_DEPTH,
            mask_prompt_fwd=cfg.MODEL.CLIP_ADAPTER.MASK_PROMPT_FWD,
        )
        init_kwargs["clip_adapter"] = clip_adapter
        init_kwargs["clip_ensemble"] = cfg.MODEL.CLIP_ADAPTER.CLIP_ENSEMBLE
        init_kwargs[
            "clip_ensemble_weight"
        ] = cfg.MODEL.CLIP_ADAPTER.CLIP_ENSEMBLE_WEIGHT

        return init_kwargs
    

    def forward(self, images, class_names, text_model, embed_size=768, use_adapter=False):
        """
        images: b, c, h, w, in BGR format
        class_names: dict of class to prompts or list of class names
        text_model: model with a function called get_text_features, which returns n, 768 text features for classes
        embed_size: 768 for large, 512 for small
        use_adapter: true to use the clip adapter (overrides CLIP_ENSEMBLE in the yaml)
        """
        if not hasattr(self,"non_object_embedding"):
            self.non_object_embedding = torch.normal(mean=0, std=embed_size ** (-.5), size=(1,embed_size)).cuda()

        height = images.shape[-2]
        width = images.shape[-1]

        images = (images.to(self.device) - self.pixel_mean) / self.pixel_std 

        features = self.backbone(images)
        outputs = self.sem_seg_head(features)

        # Get the outputs using their method, which takes into account the class names (and gets the logits immediately)
        # We can't do this in voxseg, because we want to build a representation of the embeddings in the env (so we have to argmax)
        text_features_ws = text_model.get_text_features(class_names)
        
        # sorcery
        non_object_text_features = (
            self.non_object_embedding
            / self.non_object_embedding.norm(dim=-1, keepdim=True)
        )
        text_features= torch.cat([text_features_ws, non_object_text_features], dim=0)
        
        outputs["pred_logits"] = self.clip_adapter.get_sim_logits(
            text_features, self.clip_adapter.normalize_feature(outputs["pred_logits"])
        )

        logits = outputs['pred_logits'].to(self.device)
        segs = outputs['pred_masks'].to(self.device)

        batch_class_probs = torch.zeros(len(images), len(class_names), height, width)

        for i, (mask_cls, mask_pred, image) in enumerate(zip(logits, segs, images)):
            
            mask_pred_interpolated = sem_seg_postprocess(
                mask_pred, (height, width), height, width
            )

            if use_adapter:
                # Class scores: (num_classes, h, w)
                class_scores, region = self.semantic_inference(mask_cls, mask_pred_interpolated, image, class_names)
                class_probs = class_scores / class_scores.sum(dim=0)

            else:
                
                class_scores = torch.einsum("qc,qhw->chw", mask_cls, mask_pred_interpolated)

                # Remove the non object embedding (maybe keep it for the future)
                class_scores = class_scores[:-1]

                # normalize, but since they seem to be log probs, also need to flip the negative
                class_probs = 1 - (class_scores / class_scores.sum(dim=0))

            
            
            batch_class_probs[i] = class_probs

        return batch_class_probs
    
    def semantic_inference(self, mask_cls, mask_pred, image, class_names):
            mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
            mask_pred = mask_pred.sigmoid()

            regions = None
            if self.clip_ensemble:
                clip_cls, regions, valid_flag = self.clip_adapter(
                    image, class_names, mask_pred, normalize=True
                )
                if clip_cls is None:
                    clip_cls = torch.empty(0, mask_cls.shape[-1] + 1, device=self.device)
                # softmax before index or after?
                clip_cls = F.softmax(clip_cls[:, :-1], dim=-1)
                if self.clip_ensemble_weight > 0:
                    map_back_clip_cls = mask_cls.new_ones(mask_cls.shape)
                    map_back_clip_cls[valid_flag] = clip_cls
                    mask_cls = torch.pow(mask_cls, 1 - self.clip_ensemble_weight) * \
                            torch.pow(map_back_clip_cls, self.clip_ensemble_weight)


                else:
                    # only clip model predictions are used
                    mask_cls = clip_cls
                    mask_pred = mask_pred[valid_flag]
            
            semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
            return semseg, regions

    def interpolate_features(self, feature_map, new_height, new_width):
        """
        Inputs:
            feature_map: torch.tensor (batch, network img height, network img width, feature size)
        
        Returns:
            feature_map down or upsampled to have size (batch, new_height, new_width, feature_size)
        """

        input = feature_map.permute(0, 3, 1, 2)

        # large-ish batch sizes (>7, for img dim 540x720) result in
        # RuntimeError: upsample_nearest_nhwc only supports output tensors with less than INT_MAX elements
        # this is a hacky workaround
        if len(feature_map) > 7:  
            batch, _, _, features = feature_map.size()
            output = torch.zeros((batch, features, new_height, new_width), device=feature_map.device)
            for i in range(len(input)): 
                output[i] = F.interpolate(input[i][None], size=(new_height, new_width), mode='nearest').squeeze()
            return output.permute(0,2, 3, 1)
        else:
            output = F.interpolate(input, size=(new_height, new_width), mode='nearest')
            return output.permute(0,2, 3, 1)



@META_ARCH_REGISTRY.register()
class WSDemo(MaskFormer):
    """
    This class is loaded in as the model for forward pass classification
    It is referenced in the config as "META_ARCHITECTURE: WSDemo"
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        clip_adapter: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        panoptic_on: bool,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        clip_ensemble: bool,
        clip_ensemble_weight: float,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            clip_adapter: adapter for clip-based mask classification
            num_queries: int, number of queries
            panoptic_on: bool, whether to output panoptic segmentation prediction
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
        """
        super().__init__(
            backbone=backbone,
            sem_seg_head=sem_seg_head,
            criterion=criterion,
            num_queries=num_queries,
            panoptic_on=panoptic_on,
            object_mask_threshold=object_mask_threshold,
            overlap_threshold=overlap_threshold,
            metadata=metadata,
            size_divisibility=size_divisibility,
            sem_seg_postprocess_before_inference=sem_seg_postprocess_before_inference,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
        )
        self.clip_adapter: ClipAdapter = clip_adapter

        self.clip_ensemble: bool = clip_ensemble
        self.clip_ensemble_weight: float = clip_ensemble_weight

    @classmethod
    def from_config(cls, cfg):
        init_kwargs = MaskFormer.from_config(cfg)
        text_templates = build_text_prompt(cfg.MODEL.CLIP_ADAPTER)

        clip_adapter = MaskFormerClipAdapter(
            cfg.MODEL.CLIP_ADAPTER.CLIP_MODEL_NAME,
            text_templates,
            mask_fill=cfg.MODEL.CLIP_ADAPTER.MASK_FILL,
            mask_expand_ratio=cfg.MODEL.CLIP_ADAPTER.MASK_EXPAND_RATIO,
            mask_thr=cfg.MODEL.CLIP_ADAPTER.MASK_THR,
            mask_matting=cfg.MODEL.CLIP_ADAPTER.MASK_MATTING,
            region_resized=cfg.MODEL.CLIP_ADAPTER.REGION_RESIZED,
            mask_prompt_depth=cfg.MODEL.CLIP_ADAPTER.MASK_PROMPT_DEPTH,
            mask_prompt_fwd=cfg.MODEL.CLIP_ADAPTER.MASK_PROMPT_FWD,
        )
        init_kwargs["clip_adapter"] = clip_adapter
        init_kwargs["clip_ensemble"] = cfg.MODEL.CLIP_ADAPTER.CLIP_ENSEMBLE
        init_kwargs[
            "clip_ensemble_weight"
        ] = cfg.MODEL.CLIP_ADAPTER.CLIP_ENSEMBLE_WEIGHT

        return init_kwargs

    def forward(self, batched_images):
        """
        Args:
            batched_images: torch.tensor containing the images (B, 3, H, W)
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """

        images = (batched_images.to(self.device) - self.pixel_mean) / self.pixel_std
        features = self.backbone(images)
        outputs = self.sem_seg_head(features)


        return outputs


    def demo_inference(self, mask_cls, mask_pred, image, class_names):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()

        regions = None
        if self.clip_ensemble:
            clip_cls, regions, valid_flag = self.clip_adapter(
                image, class_names, mask_pred, normalize=True
            )
            if clip_cls is None:
                clip_cls = torch.empty(0, mask_cls.shape[-1] + 1, device=self.device)
            # softmax before index or after?
            clip_cls = F.softmax(clip_cls[:, :-1], dim=-1)
            if self.clip_ensemble_weight > 0:
                map_back_clip_cls = mask_cls.new_ones(mask_cls.shape)
                map_back_clip_cls[valid_flag] = clip_cls
                mask_cls = torch.pow(mask_cls, 1 - self.clip_ensemble_weight) * \
                           torch.pow(map_back_clip_cls, self.clip_ensemble_weight)

            else:
                # only clip model predictions are used
                mask_cls = clip_cls
                mask_pred = mask_pred[valid_flag]
        bin_mask = mask_pred > self.clip_adapter.mask_thr
        select_cls = torch.zeros(sum(valid_flag), mask_cls.shape[-1], device=self.device)
        select_mask = torch.argmax(mask_cls, dim=0)
        if len(class_names) == 2 and class_names[-1] == 'others':
            select_mask = select_mask[:-1]
        for idx in select_mask:
            select_cls[idx] = mask_cls[idx]
        semseg = torch.einsum("qc,qhw->chw", select_cls, bin_mask.float())
        return semseg, regions