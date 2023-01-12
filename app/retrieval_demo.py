import itertools
import math
import os.path as osp

import debugpy
import numpy as np
import requests
import streamlit as st
from mmengine.dataset import Compose, default_collate
from mmengine.fileio import list_from_file
from mmengine.registry import init_default_scope
from PIL import Image

from mmcls import list_models as list_models_
from mmcls.apis.model import ModelHub, init_model


@st.cache()
def debug():
    debugpy.listen(('localhost', 5678))
    debugpy.wait_for_client()


@st.cache()
def load_demo_image():
    response = requests.get(
        'https://github.com/open-mmlab/mmclassification/blob/master/demo/bird.JPEG?raw=true',  # noqa
        stream=True).raw
    img = Image.open(response).convert('RGB')
    return img


@st.cache()
def list_models(*args, **kwargs):
    return sorted(list_models_(*args, **kwargs))


DATA_ROOT = 'data/data/imagenet-tiny'
ANNO_FILE = 'meta/val.txt'


def get_model(model_name, pretrained=True):

    metainfo = ModelHub.get(model_name)

    if pretrained:
        if metainfo.weights is None:
            raise ValueError(
                f"The model {model_name} doesn't have pretrained weights.")
        ckpt = metainfo.weights
    else:
        ckpt = None

    cfg = metainfo.config
    cfg.model.backbone.init_cfg = dict(
        type='Pretrained', checkpoint=ckpt, prefix='backbone')
    new_model_cfg = dict()
    new_model_cfg['type'] = 'ImageToImageRetriever'
    if hasattr(cfg.model, 'neck') and cfg.model.neck is not None:
        new_model_cfg['image_encoder'] = [cfg.model.backbone, cfg.model.neck]
    else:
        new_model_cfg['image_encoder'] = cfg.model.backbone
    cfg.model = new_model_cfg

    # prepare prototype
    cached_path = f'/home/PJLAB/huyingfan/.cache/demo/{model_name}_prototype.pt'  # noqa
    cached_exists = osp.exists(cached_path)
    if cached_exists:
        # use saved prototype
        cfg.model.prototype = cached_path
    else:
        # generate dataloader for prototype
        cfg.model.prototype = cfg.val_dataloader
        cfg.model.prototype.dataset.data_root = DATA_ROOT
        cfg.model.prototype.dataset.ann_file = ANNO_FILE

    model = init_model(metainfo.config, None, device='cuda')
    with st.spinner(f'Downloading model {model_name} on the server...This is '
                    'slow at the first time.'):
        model.init_weights()
    st.success('Model loaded!')

    with st.spinner('Preparing prototype for all image...This is '
                    'slow at the first time.'):
        model.prepare_prototype()
        if not cached_exists:
            print(f'save prototype to {cached_path}')
            model.dump_prototype(cached_path)
            st.success('Prototype cached!')

    return model


def get_pred(name, img):

    init_default_scope('mmcls')

    model = get_model(name)

    cfg = model.cfg
    # build the data pipeline
    test_pipeline_cfg = cfg.test_dataloader.dataset.pipeline
    if isinstance(img, str):
        if test_pipeline_cfg[0]['type'] != 'LoadImageFromFile':
            test_pipeline_cfg.insert(0, dict(type='LoadImageFromFile'))
        data = dict(img_path=img)
    elif isinstance(img, np.ndarray):
        if test_pipeline_cfg[0]['type'] == 'LoadImageFromFile':
            test_pipeline_cfg.pop(0)
        data = dict(img=img)
    elif isinstance(img, Image.Image):
        if test_pipeline_cfg[0]['type'] == 'LoadImageFromFile':
            test_pipeline_cfg[0] = dict(type='ToNumpy', keys=['img'])
        data = dict(img=img)

    test_pipeline = Compose(test_pipeline_cfg)
    data = test_pipeline(data)
    data = default_collate([data])

    labels = model.val_step(data)[0].pred_label.label
    scores = model.val_step(data)[0].pred_label.score[labels]

    # result_list = [(model.prototype.dataset.get_data_info(idx)['img_path'],
    #                 score) for idx, score in zip(labels, scores)]
    image_list = list_from_file(
        osp.join('/home/PJLAB/huyingfan/Desktop/openmmlab/mmclassification',
                 DATA_ROOT, ANNO_FILE))
    data_root = osp.join(
        '/home/PJLAB/huyingfan/Desktop/openmmlab/mmclassification', DATA_ROOT,
        'val')
    result_list = [(osp.join(data_root, image_list[idx].rsplit()[0]), score)
                   for idx, score in zip(labels, scores)]
    return result_list


def app():
    # debug()
    model_name = st.sidebar.selectbox('Model:', list_models('[!clip]*'))
    # model_name = st.sidebar.selectbox("Model:", ['resnet18_8xb32_in1k'])

    st.markdown(
        "<h1 style='text-align: center;'>Image To Image Retrieval</h1>",
        unsafe_allow_html=True,
    )

    file = st.file_uploader(
        'Please upload your own image or use the provided:')

    container1 = st.container()
    if file:
        raw_img = Image.open(file).convert('RGB')
    else:
        raw_img = load_demo_image()

    container1.header('Image')

    w, h = raw_img.size
    scaling_factor = 360 / w
    resized_image = raw_img.resize(
        (int(w * scaling_factor), int(h * scaling_factor)))

    container1.image(resized_image, use_column_width='auto')
    button = container1.button('Search')

    st.header('Results')

    topk = st.sidebar.number_input('Topk(1-50)', min_value=1, max_value=50)

    # search on both selection of topk and button
    if button or topk > 1:

        result_list = get_pred(model_name, raw_img)
        # auto adjust number of images in a row but 5 at most.
        col = min(int(math.sqrt(topk)), 5)
        row = math.ceil(topk / col)

        grid = []
        for i in range(row):
            with st.container():
                grid.append(st.columns(col))

        grid = list(itertools.chain.from_iterable(grid))[:topk]

        for cell, (image_path, score) in zip(grid, result_list[:topk]):
            image = Image.open(image_path).convert('RGB')

            w, h = raw_img.size
            scaling_factor = 360 / w
            resized_image = raw_img.resize(
                (int(w * scaling_factor), int(h * scaling_factor)))

            cell.caption('Score: {:.4f}'.format(float(score)))
            cell.image(image)


if __name__ == '__main__':
    app()
