# -*- coding: utf-8 -*-
# server.py  ——  替换 Gradio 的 FastAPI 后端
# 启动方式：python server.py
# 依赖：pip install fastapi uvicorn python-multipart

import os
import uuid
import sys
import re
import json
import numpy as np
import torch
import faiss
from dataclasses import dataclass
from typing import Optional
from PIL import Image
from torchvision import transforms, models
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import pandas as pd
import tempfile
import shutil
import traceback

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI

# ==========================================================
# 0) 让 KD_CMN 工程的"根目录导入"在 Windows 下可用
# ==========================================================
KDCMN_ROOT = r"C:\Users\18368\Desktop\learn_pytorch\test\eyeagent\tool\KD_CMN"
sys.path.insert(0, KDCMN_ROOT)

from models.models import BaseCMNModel
from modules.tokenizer import Tokenizer

# ==========================================================
# OCT 项目路径导入
# ==========================================================
OCT_ROOT = r"C:\Users\18368\Desktop\learn_pytorch\test\基于KAN增强视觉Transformer的OCT疾病分类方法 （录用）刘雪晴\代码第一部分\代码第一部分"
sys.path.insert(0, OCT_ROOT)

from classic_models import find_model_using_name

# ==========================================================
# 无赤光眼底 7类模型导入
# ==========================================================
NORED_ROOT = r"E:\崔心赞\densenet"
sys.path.insert(0, NORED_ROOT)
from model import densenet201

# ==========================================================
# 1) API KEY
# ==========================================================
##os.environ["OPENAI_API_KEY"] = "sk-SvznmM6oVVbUcW4SkxO88xFWdDEnacDQoqhpe4FmWWovvHXI"

# ==========================================================
# 2) 图片路径工具函数
# ==========================================================
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def looks_like_path(s: str) -> bool:
    s = (s or "").strip().strip('"').strip("'")
    if re.match(r"^[a-zA-Z]:\\", s):
        return True
    if any(ch in s for ch in ["\\", "/"]) and s.lower().endswith(IMAGE_EXTS):
        return True
    return False


def normalize_path(s: str) -> str:
    return (s or "").strip().strip('"').strip("'")


def is_existing_image_path(s: str) -> bool:
    p = normalize_path(s)
    return p.lower().endswith(IMAGE_EXTS) and os.path.exists(p)


# ==========================================================
# RAG 配置
# ==========================================================
OUT_DIR = r"C:\Users\18368\Desktop\learn_pytorch\test\eyeagent\tool\rag_store"
INDEX_PATH = os.path.join(OUT_DIR, "faiss.index")
META_PATH = os.path.join(OUT_DIR, "meta.json")
EMB_MODEL_DIR = r"C:\Users\18368\Desktop\learn_pytorch\test\eyeagent\models\beg-small-zh"
RAG_TOP_K = 5
FORCE_CPU = False


class HFEmbedder:
    def __init__(self, model_dir, device=None):
        if device is None:
            device = "cpu" if FORCE_CPU else ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModel.from_pretrained(model_dir).to(device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts):
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(self.device)
        outputs = self.model(**inputs)
        emb = outputs.last_hidden_state[:, 0]
        emb = torch.nn.functional.normalize(emb, dim=1)
        return emb.cpu().numpy().astype("float32")


# ==========================================================
# 3) SYSTEM PROMPT
# ==========================================================
SYSTEM_PROMPT = """
你是一位智能医学图像助手。用户输入可能是普通聊天文本，也可能包含本地图片路径（或上传图片后得到的临时路径）。

你有九个工具：
1) classify_medical_image(image_path) -> 返回图像类型（octa毛细血管图像 / OCT图像 / 彩色眼底图像 / 无赤光眼底图像 / 超声图像）
2) generate_kdcmn_report(image_path) -> 生成诊断报告（注意：只支持超声图像；OCTA不支持）
3) retrieve_from_rag(query) -> 从眼科医学知识库检索相关信息（用于回答医学问题、减少幻觉）
4) classify_vitreous_image(image_path) -> 对玻璃体相关超声返图像进行5类分类
5) predict_rfmid_labels(image_path) -> 对眼底图像进行多标签疾病分类
6) predict_dr_grade(image_path) -> 对彩色眼底图像进行糖尿病视网膜病变（DR）五级分级
7) predict_glaucoma_grade(image_path) -> 对彩色眼底图像进行青光眼三类分级
8) predict_oct_disease(image_path) -> 对OCT图像进行7类疾病分类
9) predict_nored_fundus_disease(image_path) -> 对无赤光眼底图像进行7类疾病分类

你必须遵循以下规则：
- 只有当用户提供的是"真实存在的本地图片路径"时，才允许调用图像相关工具。
- 如果用户只是聊天/提问但没有给路径，不要调用图像工具，先引导用户上传图片或提供图片路径。
- 工具调用要按用户意图来：
  - 如果用户没有提出需要生成报告：只调用 classify_medical_image，并在回答里询问是否需要生成医学报告。
  - 如果用户的需求是"判断图像类型/是什么图/分类/是不是OCTA或超声"，只调用 classify_medical_image。
  - 如果用户的需求是"生成诊断报告/输出报告/写结论/给我诊断"，你可以调用 generate_kdcmn_report，但必须先知道图像类型：
    - 若你尚未确定图像类型，先调用 classify_medical_image。
    - 只有当分类结果为"超声图像"时，才调用 generate_kdcmn_report。
    - 如果分类结果为"octa毛细血管图像"或"彩色眼底图像"，不要调用报告工具，解释：目前报告生成仅支持超声图像，并询问用户是否只需要分类结果/或提供超声图像。
  - 如果用户的问题涉及眼科医学知识、症状解释、治疗建议或其他非图像特定问题，先调用 retrieve_from_rag(query) 检索知识库，然后基于检索结果生成回答，以确保准确性并减少幻觉。
  - 对于复杂问题，可以先检索RAG，然后结合图像工具（如果有图像）。
  - 如果用户的需求涉及玻璃体相关分类，且图像类型为超声图像，先调用 classify_vitreous_image，然后基于结果生成回答或结合其他工具。
  - 如果用户的需求涉及眼底疾病多标签分类，且图像类型适合（为"彩色眼底图像"），调用 predict_rfmid_labels。
  - 如果用户需求涉及"糖尿病视网膜病变分级/糖网分级/DR分级"等，且图像为彩色眼底图像，调用 predict_dr_grade。
  - 如果用户需求涉及"青光眼/青光眼分级"等，且图像为彩色眼底图像，调用 predict_glaucoma_grade。
  - 当 RFMiD 预测结果中出现"ODC"或"AION"时，必须再调用 predict_glaucoma_grade。
  - 当 RFMiD 预测结果中出现"DR"时，必须再调用 predict_dr_grade。
  - predict_dr_grade 输出必须用中文等级名称+通俗解释+置信度，绝不输出英文缩写。
  - predict_glaucoma_grade 输出必须用中文等级名称（正常 / 早期青光眼 / 晚期青光眼）+通俗解释+置信度，绝不输出英文缩写。
- 当用户问题中包含"青光眼"相关词汇时，无论 RFMiD 结果如何，都必须调用 predict_glaucoma_grade。
- 如果用户的需求涉及"OCT图像"相关关键词，或 classify_medical_image 返回"octa毛细血管图像"，必须调用 predict_oct_disease。
- 如果用户的需求涉及"无赤光眼底图像"，或 classify_medical_image 返回"无赤光眼底图像"，必须调用 predict_nored_fundus_disease。

structured_response 输出要求：
- punny_response：必须输出给用户看的"单段完整结果"，不要出现"分类结果：""诊断报告："这种分栏标题。
  如果生成了超声报告，必须严格按下面模板输出（保持换行）：
  您好，已为您生成超声图像诊断报告：
  诊断报告
  <把报告正文按条目换行输出>
  请注意：本报告仅为AI辅助分析结果，具体诊断请以临床医生判断为准。
  如果使用了RAG检索，必须在响应中自然融入检索到的信息，并注明"基于知识库检索"。
  如果使用了玻璃体分类，必须自然融入结果。
  如果使用了RFMiD预测，必须自然融入结果。
  如果使用了DR分级，必须自然融入结果。
  如果使用了青光眼分级，必须自然融入结果。
  如果使用了OCT分类，必须自然融入结果。
  如果使用了无赤光眼底分类，必须自然融入结果。

- classification_result：可以填写（内部字段）
- report_result：可以填写（内部字段）
""".strip()

RFMID_CN_MAP = {
    "Disease_Risk": ("眼病风险（综合异常）", "眼底整体有明显异常，需要重视"),
    "DR": ("糖尿病视网膜病变", "糖尿病引起的眼底出血、渗出等"),
    "ARMD": ("年龄相关性黄斑变性", "老年人黄斑区退行性病变，易影响中心视力"),
    "MH": ("黄斑裂孔", "黄斑中心出现破洞，影响看东西清晰度"),
    "DN": ("糖尿病相关视网膜病变", "与糖尿病相关的视网膜异常"),
    "MYA": ("近视", "高度近视引起的眼底改变"),
    "BRVO": ("视网膜分支静脉阻塞", "眼底静脉分支堵塞，导致局部出血水肿"),
    "TSLN": ("豹纹状眼底", "高度近视常见的眼底表现"),
    "ERM": ("视网膜前膜", "黄斑表面长出一层膜，影响视力"),
    "LS": ("激光瘢痕", "以前做过激光治疗留下的瘢痕"),
    "MS": ("黄斑瘢痕", "黄斑区瘢痕形成，影响中心视力"),
    "CSR": ("中心性浆液性脉络膜视网膜病变", "黄斑下积液，看东西会变形或有暗点"),
    "ODC": ("视盘凹陷", "可能与青光眼相关，视盘杯盘比增大"),
    "CRVO": ("视网膜中央静脉阻塞", "大范围眼底出血、水肿"),
    "TV": ("血管迂曲", "眼底血管走行异常弯曲"),
    "AH": ("星状玻璃体变性", "玻璃体内出现星状钙化物"),
    "ODP": ("视盘小凹", "先天性视盘发育异常"),
    "ODE": ("视神经乳头水肿", "视盘水肿，可能颅内压增高或其他原因"),
    "ST": ("颞侧视盘变浅", "视盘颞侧变薄"),
    "AION": ("前部缺血性视神经病变", "视神经缺血导致视力急剧下降"),
    "PT": ("视乳头水肿/视盘周围异常", "视盘及其周围异常"),
    "RT": ("视网膜裂孔", "视网膜有裂口，易发生脱离"),
    "RS": ("视网膜劈裂", "视网膜层间分离"),
    "CRS": ("脉络膜视网膜瘢痕", "陈旧性脉络膜视网膜病变瘢痕"),
    "EDN": ("渗出性病变/水肿", "眼底有渗出或水肿"),
    "RPEC": ("视网膜色素上皮改变", "色素上皮层异常"),
}


# ==========================================================
# 4) Context + ResponseFormat
# ==========================================================
@dataclass
class Context:
    image_path: Optional[str] = None


@dataclass
class ResponseFormat:
    punny_response: Optional[str] = None
    classification_result: Optional[str] = None
    report_result: Optional[str] = None


# ==========================================================
# 图像类型分类（ResNet18）
# ==========================================================
_CLASSIFIER_READY = False
_CLASSIFIER_DEVICE = None
_CLASSIFIER_MODEL = None
_CLASSIFIER_TFM = None

CLASSIFIER_CFG = {
    "ckpt_path": r"C:\Users\18368\Desktop\learn_pytorch\test\eyeagent\tool\图像种类分类\ckpts\resnet18_3class_best.pth",
    "n_gpu": 1,
}


def _classifier_init_once():
    global _CLASSIFIER_READY, _CLASSIFIER_DEVICE, _CLASSIFIER_MODEL, _CLASSIFIER_TFM
    if _CLASSIFIER_READY:
        return
    _CLASSIFIER_DEVICE = torch.device(
        "cuda:0" if (CLASSIFIER_CFG["n_gpu"] > 0 and torch.cuda.is_available()) else "cpu"
    )
    m = models.resnet18(weights=None)
    m.fc = torch.nn.Linear(m.fc.in_features, 5)
    sd = torch.load(CLASSIFIER_CFG["ckpt_path"], map_location=_CLASSIFIER_DEVICE)
    m.load_state_dict(sd["model"])
    m.to(_CLASSIFIER_DEVICE).eval()
    _CLASSIFIER_MODEL = m
    _CLASSIFIER_TFM = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    _CLASSIFIER_READY = True


@tool
def classify_medical_image(image_path: str) -> str:
    """分类医学图像：octa毛细血管图像 / OCT图像 / 彩色眼底图像 / 无赤光眼底图像 / 超声图像"""
    image_path = normalize_path(image_path)
    if not is_existing_image_path(image_path):
        return "❌ 输入不是有效的本地图片路径（或文件不存在）。请提供 .jpg/.png 等图片的完整路径。"
    _classifier_init_once()
    img = Image.open(image_path).convert("RGB")
    x = _CLASSIFIER_TFM(img).unsqueeze(0).to(_CLASSIFIER_DEVICE)
    with torch.no_grad():
        logits = _CLASSIFIER_MODEL(x)
        pred = torch.argmax(logits, dim=1).item()
    class_names = ["octa毛细血管图像", "oct图像", "彩色眼底图像", "无赤光眼底图像", "超声图像"]
    return f"{class_names[pred]}"


# ==========================================================
# KD_CMN 报告生成
# ==========================================================
_KDCMN_READY = False
_KDCMN_DEVICE = None
_KDCMN_MODEL = None
_KDCMN_TOKENIZER = None
_KDCMN_NODE_FEATURES = None
_KDCMN_ADJ = None
_KDCMN_IMG_TFM = None

KDCMN_CFG = {
    "ann_path": r"C:\Users\18368\Desktop\learn_pytorch\test\eyeagent\tool\KD_CMN\data\eye_22173\eye_annotation.json",
    "node_features_path": r"C:\Users\18368\Desktop\learn_pytorch\test\eyeagent\tool\KD_CMN\models\node_features.npy",
    "adj_matrix_path": r"C:\Users\18368\Desktop\learn_pytorch\test\eyeagent\tool\KD_CMN\models\adj_matrix.npy",
    "ckpt_path": r"C:\Users\18368\Desktop\learn_pytorch\test\eyeagent\tool\KD_CMN\results\LC\model_best.pth",
    "n_gpu": 1,
}


class KDCMNArgs:
    dataset_name = "mimic_cxr"
    max_seq_length = 80
    threshold = 3
    ann_path = None
    bert_size = 768
    hidden_size = 2048
    num_attention_heads = 8
    visual_extractor = "resnet101"
    visual_extractor_pretrained = True
    d_model = 512
    d_ff = 512
    d_vf = 2048
    num_heads = 8
    num_layers = 3
    dropout = 0.1
    logit_layers = 1
    bos_idx = 0
    eos_idx = 0
    pad_idx = 0
    use_bn = 0
    drop_prob_lm = 0.5
    topk = 32
    cmm_size = 2048
    cmm_dim = 512
    sample_method = "beam_search"
    beam_size = 3
    temperature = 1.0
    sample_n = 1
    group_size = 1
    output_logsoftmax = 1
    decoding_constraint = 0
    block_trigrams = 1
    seed = 9233
    n_gpu = 1


def _kdcmn_init_once():
    global _KDCMN_READY, _KDCMN_DEVICE, _KDCMN_MODEL, _KDCMN_TOKENIZER
    global _KDCMN_NODE_FEATURES, _KDCMN_ADJ, _KDCMN_IMG_TFM
    if _KDCMN_READY:
        return
    _KDCMN_DEVICE = torch.device(
        "cuda:0" if (KDCMN_CFG["n_gpu"] > 0 and torch.cuda.is_available()) else "cpu"
    )
    args = KDCMNArgs()
    args.ann_path = KDCMN_CFG["ann_path"]
    args.n_gpu = KDCMN_CFG["n_gpu"]
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    _KDCMN_TOKENIZER = Tokenizer(args)
    num_features = 768
    _KDCMN_MODEL = BaseCMNModel(args, _KDCMN_TOKENIZER, num_features).to(_KDCMN_DEVICE)
    ckpt = torch.load(KDCMN_CFG["ckpt_path"], map_location=_KDCMN_DEVICE)
    state_dict = ckpt.get("state_dict", ckpt)
    _KDCMN_MODEL.load_state_dict(state_dict, strict=True)
    _KDCMN_MODEL.eval()
    node_features = np.load(KDCMN_CFG["node_features_path"])
    adj = np.load(KDCMN_CFG["adj_matrix_path"])
    _KDCMN_NODE_FEATURES = torch.from_numpy(node_features).float().to(_KDCMN_DEVICE)
    _KDCMN_ADJ = torch.from_numpy(adj).float().to(_KDCMN_DEVICE)
    if _KDCMN_NODE_FEATURES.dim() == 3 and _KDCMN_NODE_FEATURES.size(0) == 1:
        _KDCMN_NODE_FEATURES = _KDCMN_NODE_FEATURES.squeeze(0)
    if _KDCMN_ADJ.dim() == 3 and _KDCMN_ADJ.size(0) == 1:
        _KDCMN_ADJ = _KDCMN_ADJ.squeeze(0)
    _KDCMN_IMG_TFM = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    _KDCMN_READY = True


def _format_report_text(raw: str) -> str:
    if not raw:
        return ""
    text = raw.strip()
    text = text.replace("；", "；\n").replace("。", "。\n")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines)


@tool
def generate_kdcmn_report(image_path: str) -> str:
    """使用KD_CMN对单张图像生成报告文本（输入必须是存在的本地图像路径）"""
    image_path = normalize_path(image_path)
    if not is_existing_image_path(image_path):
        return "❌ 输入不是有效的本地图片路径（或文件不存在）。请提供 .jpg/.png 等图片的完整路径。"
    _kdcmn_init_once()
    img = Image.open(image_path).convert("RGB")
    img_t = _KDCMN_IMG_TFM(img).unsqueeze(0).to(_KDCMN_DEVICE)
    with torch.no_grad():
        output, _ = _KDCMN_MODEL(img_t, _KDCMN_NODE_FEATURES, _KDCMN_ADJ, mode="sample")
        report = _KDCMN_MODEL.tokenizer.decode_batch(output.cpu().numpy())[0]
    return _format_report_text(report)


# ==========================================================
# RAG 检索
# ==========================================================
_RAG_READY = False
_RAG_DEVICE = None
_RAG_EMBEDDER = None
_RAG_INDEX = None
_RAG_META = None


def _rag_init_once():
    global _RAG_READY, _RAG_DEVICE, _RAG_EMBEDDER, _RAG_INDEX, _RAG_META
    if _RAG_READY:
        return
    _RAG_DEVICE = torch.device(
        "cuda:0" if torch.cuda.is_available() and not FORCE_CPU else "cpu"
    )
    _RAG_EMBEDDER = HFEmbedder(EMB_MODEL_DIR, device=_RAG_DEVICE)
    if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
        raise RuntimeError(f"RAG 数据库未构建：{INDEX_PATH} 或 {META_PATH} 不存在。请先运行 build_rag.py")
    _RAG_INDEX = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        _RAG_META = json.load(f)
    _RAG_READY = True


@tool
def retrieve_from_rag(query: str) -> str:
    """从眼科医学知识库中检索相关信息（输入查询字符串，返回top-5相关文本片段）"""
    if not query.strip():
        return "❌ 查询字符串为空。"
    _rag_init_once()
    q_emb = _RAG_EMBEDDER.encode([query])
    D, I = _RAG_INDEX.search(q_emb, RAG_TOP_K)
    results = []
    for rank, (dist, idx) in enumerate(zip(D[0], I[0])):
        if idx < 0:
            continue
        meta = _RAG_META[idx]
        text = meta.get("text", "")
        source = meta.get("source", "")
        results.append(f"[{rank+1}] 相关度: {1/(1+dist):.3f} | 来源: {source}\n{text}")
    return "\n\n".join(results) if results else "未找到相关内容。"


# ==========================================================
# 玻璃体超声分类
# ==========================================================
_ALEXNET_READY = False
_ALEXNET_DEVICE = None
_ALEXNET_MODEL = None
_ALEXNET_TFM = None
_ALEXNET_CLASS_INDICES = None

ALEXNET_CFG = {
    "ckpt_path": r"C:\Users\18368\Desktop\learn_pytorch\test\田泽翰\alexnet\AlexNet.pth",
    "json_path": r"C:\Users\18368\Desktop\learn_pytorch\test\田泽翰\alexnet\class_indices.json",
}


class AlexNet(torch.nn.Module):
    def __init__(self, num_classes=5, init_weights=False):
        super(AlexNet, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            torch.nn.Conv2d(48, 128, kernel_size=5, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            torch.nn.Conv2d(128, 192, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(192, 192, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(192, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(128 * 6 * 6, 2048),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(2048, 2048),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(2048, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.constant_(m.bias, 0)


def _alexnet_init_once():
    global _ALEXNET_READY, _ALEXNET_DEVICE, _ALEXNET_MODEL, _ALEXNET_TFM, _ALEXNET_CLASS_INDICES
    if _ALEXNET_READY:
        return

    _ALEXNET_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _ALEXNET_MODEL = AlexNet(num_classes=5).to(_ALEXNET_DEVICE)
    sd = torch.load(ALEXNET_CFG["ckpt_path"], map_location=_ALEXNET_DEVICE, weights_only=True)
    _ALEXNET_MODEL.load_state_dict(sd)
    _ALEXNET_MODEL.eval()

    with open(ALEXNET_CFG["json_path"], "r", encoding="utf-8") as f:
        _ALEXNET_CLASS_INDICES = json.load(f)

    _ALEXNET_TFM = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    _ALEXNET_READY = True


@tool
def classify_vitreous_image(image_path: str) -> str:
    """对玻璃体相关超声图像进行5类分类"""
    image_path = normalize_path(image_path)
    if not is_existing_image_path(image_path):
        return "❌ 输入不是有效的本地图片路径（或文件不存在）。"

    _alexnet_init_once()

    img = Image.open(image_path).convert("RGB")
    x = _ALEXNET_TFM(img).unsqueeze(0).to(_ALEXNET_DEVICE)

    with torch.no_grad():
        output = torch.squeeze(_ALEXNET_MODEL(x)).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).item()
        prob = predict[predict_cla].item()

    class_name = _ALEXNET_CLASS_INDICES.get(str(predict_cla), f"未知类别({predict_cla})")

    return f"玻璃体图像分类为：{class_name}（概率：{prob:.3f}）"

# ==========================================================
# RFMiD 多标签眼底疾病分类
# ==========================================================
# ==========================================================
# RFMiD 多标签预测（移植自老代码）
# ==========================================================
_RFMID_READY = False
_RFMID_DEVICE = None
_RFMID_MODEL = None
_RFMID_TFM = None
_RFMID_LABEL_COLS = None

RFMID_CFG = {
    "ckpt_path": r"C:\Users\18368\Desktop\learn_pytorch\test\eyeagent\dataset\RFMiD\ckpts\effb0_rfmID_best_f1.pth",
    "label_csv_path": r"C:\Users\18368\Desktop\learn_pytorch\test\eyeagent\dataset\RFMiD\Test_Set\Test_Set\RFMiD_Testing_Labels.csv",
    "threshold": 0.5,
}


def _rfmid_init_once():
    global _RFMID_READY, _RFMID_DEVICE, _RFMID_MODEL, _RFMID_TFM, _RFMID_LABEL_COLS
    if _RFMID_READY:
        return

    _RFMID_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 读取标签列（推荐方式）
    df = pd.read_csv(RFMID_CFG["label_csv_path"])
    _RFMID_LABEL_COLS = [c for c in df.columns if c not in ["ID"]]

    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    _RFMID_MODEL = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = _RFMID_MODEL.classifier[1].in_features
    _RFMID_MODEL.classifier[1] = nn.Linear(in_features, len(_RFMID_LABEL_COLS))

    ckpt = torch.load(RFMID_CFG["ckpt_path"], map_location=_RFMID_DEVICE, weights_only=True)
    _RFMID_MODEL.load_state_dict(ckpt["model"])
    _RFMID_MODEL.to(_RFMID_DEVICE)
    _RFMID_MODEL.eval()

    _RFMID_TFM = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])

    _RFMID_READY = True


@tool
def predict_rfmid_labels(image_path: str) -> str:
    """对眼底图像进行多标签疾病分类（RFMiD模型），返回医生式中文解释"""
    image_path = normalize_path(image_path)
    if not is_existing_image_path(image_path):
        return "❌ 输入不是有效的本地图片路径（或文件不存在）。请提供 .jpg/.png 等图片的完整路径。"

    _rfmid_init_once()

    img = Image.open(image_path).convert("RGB")
    x = _RFMID_TFM(img).unsqueeze(0).to(_RFMID_DEVICE)

    with torch.no_grad():
        logits = _RFMID_MODEL(x)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()

    findings = []
    threshold = RFMID_CFG["threshold"]
    for label, prob in zip(_RFMID_LABEL_COLS, probs):
        if prob >= threshold:
            cn_name, desc = RFMID_CN_MAP.get(label, (label, "该项目含义待确认"))
            findings.append((cn_name, prob, desc))

    if not findings:
        return "目前未发现明显异常，建议定期检查眼底。"

    findings.sort(key=lambda x: x[1], reverse=True)

    high_risk = [f for f in findings if f[1] >= 0.70]
    other = [f for f in findings if f[1] < 0.70]

    response = "您好，我仔细看了您的眼底照片，整体情况有几点需要特别注意。\n\n"

    if high_risk:
        response += "目前风险最高的是："
        for name, prob, desc in high_risk[:3]:
            response += f"{name}（{prob:.1%}），{desc}；"
        response += "这些概率比较高，建议您尽快去眼科做进一步检查。\n\n"

    if other:
        response += "另外还有一些相对轻一些的变化，比如"
        for name, prob, desc in other[:3]:
            response += f"{name}（{prob:.1%}），{desc}；"
        response += "这些也值得关注，但优先处理高风险的部分。\n\n"

    response += "眼底情况比较复杂，强烈建议您带上照片尽快找眼科医生复查，尤其是如果有糖尿病、高血压等基础病，更要重视。早发现早处理，效果会好很多。\n"
    response += "请注意：本结果仅为AI辅助分析，具体诊断和治疗方案请以临床医生判断为准。"

    return response

# ==========================================================
# DR 糖尿病视网膜病变分级
# ==========================================================
_DR_GRADE_READY = False
_DR_GRADE_DEVICE = None
_DR_GRADE_MODEL = None
_DR_GRADE_TFM = None

DR_GRADE_CFG = {
    "ckpt_path": r"C:\Users\18368\Desktop\learn_pytorch\test\eyeagent\dataset\sugar\ckpts\effb0_dr5_best_by_val_f1.pth",  # 改回正确路径
}


def _dr_grade_init_once():
    global _DR_GRADE_READY, _DR_GRADE_DEVICE, _DR_GRADE_MODEL, _DR_GRADE_TFM
    if _DR_GRADE_READY:
        return

    _DR_GRADE_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    _DR_GRADE_MODEL = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = _DR_GRADE_MODEL.classifier[1].in_features
    _DR_GRADE_MODEL.classifier[1] = nn.Linear(in_features, 5)

    ckpt = torch.load(DR_GRADE_CFG["ckpt_path"], map_location=_DR_GRADE_DEVICE)
    _DR_GRADE_MODEL.load_state_dict(ckpt["model"])
    _DR_GRADE_MODEL.to(_DR_GRADE_DEVICE).eval()

    _DR_GRADE_TFM = transforms.Compose([
        transforms.Resize((224, 224)),   # effb0 用 224 就够
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    _DR_GRADE_READY = True


@tool
def predict_dr_grade(image_path: str) -> str:
    """对彩色眼底图像进行糖尿病视网膜病变（DR）五级分级，返回中文等级名称+解释+置信度（仅适用于眼底图像）"""
    image_path = normalize_path(image_path)
    if not is_existing_image_path(image_path):
        return "❌ 输入不是有效的本地图片路径（或文件不存在）。请提供 .jpg/.png 等图片的完整路径。"
    _dr_grade_init_once()
    img = Image.open(image_path).convert("RGB")
    x = _DR_GRADE_TFM(img).unsqueeze(0).to(_DR_GRADE_DEVICE)
    with torch.no_grad():
        logits = _DR_GRADE_MODEL(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()
        pred_idx = int(probs.argmax())
        confidence = probs[pred_idx]
    grades = [
        ("轻度非增殖性糖尿病视网膜病变", "仅有少量微动脉瘤，属于早期变化"),
        ("中度非增殖性糖尿病视网膜病变", "出现较多出血、渗出等病变，需要关注"),
        ("无糖尿病视网膜病变", "眼底未见糖尿病相关病变，目前眼底正常"),
        ("增殖性糖尿病视网膜病变", "新生血管形成，出血风险高，需及时治疗"),
        ("重度非增殖性糖尿病视网膜病变", "严重缺血表现，接近增殖期，需密切随访"),
    ]
    cn_name, desc = grades[pred_idx]
    return (f"糖尿病视网膜病变分级：{cn_name}\n"
            f"解释：{desc}\n"
            f"模型置信度：{confidence:.1%}")


# ==========================================================
# 青光眼三类分级
# ==========================================================
_GLAUCOMA_GRADE_READY = False
_GLAUCOMA_GRADE_DEVICE = None
_GLAUCOMA_GRADE_MODEL = None
_GLAUCOMA_GRADE_TFM = None

GLAUCOMA_GRADE_CFG = {
    "ckpt_path": r"C:\Users\18368\Desktop\learn_pytorch\test\eyeagent\dataset\qgy\ckpts_glaucoma_qgy\effb0_qgy_best_by_val_macroF1.pth",
}


def _glaucoma_grade_init_once():
    global _GLAUCOMA_GRADE_READY, _GLAUCOMA_GRADE_DEVICE, _GLAUCOMA_GRADE_MODEL, _GLAUCOMA_GRADE_TFM
    if _GLAUCOMA_GRADE_READY:
        return
    _GLAUCOMA_GRADE_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    _GLAUCOMA_GRADE_MODEL = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = _GLAUCOMA_GRADE_MODEL.classifier[1].in_features
    _GLAUCOMA_GRADE_MODEL.classifier[1] = nn.Linear(in_features, 3)
    ckpt = torch.load(GLAUCOMA_GRADE_CFG["ckpt_path"], map_location=_GLAUCOMA_GRADE_DEVICE)
    _GLAUCOMA_GRADE_MODEL.load_state_dict(ckpt["model"])
    _GLAUCOMA_GRADE_MODEL.to(_GLAUCOMA_GRADE_DEVICE)
    _GLAUCOMA_GRADE_MODEL.eval()
    _GLAUCOMA_GRADE_TFM = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    _GLAUCOMA_GRADE_READY = True


@tool
def predict_glaucoma_grade(image_path: str) -> str:
    """对彩色眼底图像进行青光眼三类分级（正常 / 早期青光眼 / 晚期青光眼），返回中文等级名称+解释+置信度（仅适用于眼底图像）"""
    image_path = normalize_path(image_path)
    if not is_existing_image_path(image_path):
        return "❌ 输入不是有效的本地图片路径（或文件不存在）。请提供 .jpg/.png 等图片的完整路径。"
    _glaucoma_grade_init_once()
    img = Image.open(image_path).convert("RGB")
    x = _GLAUCOMA_GRADE_TFM(img).unsqueeze(0).to(_GLAUCOMA_GRADE_DEVICE)
    with torch.no_grad():
        logits = _GLAUCOMA_GRADE_MODEL(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()
        pred_idx = int(probs.argmax())
        confidence = probs[pred_idx]
    grades = [
        ("晚期青光眼", "视盘杯盘比显著增大，视神经严重损伤，视野缺损可能明显"),
        ("早期青光眼", "视盘杯盘比轻度增大，视神经损伤初期"),
        ("正常", "眼底视盘未见明显异常，目前无青光眼迹象"),
    ]
    cn_name, desc = grades[pred_idx]
    return (f"青光眼分级：{cn_name}\n"
            f"解释：{desc}\n"
            f"模型置信度：{confidence:.1%}")


# ==========================================================
# OCT 7类疾病分类
# ==========================================================
_OCT_READY = False
_OCT_DEVICE = None
_OCT_MODEL = None
_OCT_TFM = None

OCT_CFG = {
    "ckpt_path": r"C:\Users\18368\Desktop\learn_pytorch\test\基于KAN增强视觉Transformer的OCT疾病分类方法 （录用）刘雪晴\代码第一部分\代码第一部分\results\weights\vision_transformer1\kansformer_att.pth",
    "model_name": "vision_transformer1",
    "num_classes": 7,
}

OCT_CLASS_NAMES = ["AMD", "CNV", "CSR", "DME", "DR", "DRUSEN", "NORMAL"]
OCT_CN_MAP = {
    "AMD": ("年龄相关性黄斑变性", "老年人黄斑区退行性病变，易影响中心视力"),
    "CNV": ("脉络膜新生血管", "新生血管异常增生，导致渗漏和出血"),
    "CSR": ("中心性浆液性脉络膜视网膜病变", "黄斑下积液，看东西变形或有暗点"),
    "DME": ("糖尿病黄斑水肿", "糖尿病导致的黄斑中心水肿，视力下降明显"),
    "DR": ("糖尿病视网膜病变", "糖尿病引起的眼底出血、渗出等"),
    "DRUSEN": ("玻璃膜疣", "黄斑区沉积物，早期年龄相关性黄斑变性常见表现"),
    "NORMAL": ("正常", "OCT图像未见明显异常"),
}


def _oct_init_once():
    global _OCT_READY, _OCT_DEVICE, _OCT_MODEL, _OCT_TFM
    if _OCT_READY:
        return
    _OCT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _OCT_MODEL = find_model_using_name(OCT_CFG["model_name"], num_classes=OCT_CFG["num_classes"]).to(_OCT_DEVICE)
    if os.path.exists(OCT_CFG["ckpt_path"]):
        _OCT_MODEL.load_state_dict(torch.load(OCT_CFG["ckpt_path"], map_location=_OCT_DEVICE))
    else:
        raise FileNotFoundError(f"OCT模型权重未找到：{OCT_CFG['ckpt_path']}")
    _OCT_MODEL.eval()
    _OCT_TFM = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    _OCT_READY = True


@tool
def predict_oct_disease(image_path: str) -> str:
    """对OCT图像进行7类疾病分类（AMD / CNV / CSR / DME / DR / DRUSEN / NORMAL），返回中文解释+置信度（仅适用于OCT/OCTA图像）"""
    image_path = normalize_path(image_path)
    if not is_existing_image_path(image_path):
        return "❌ 输入不是有效的本地图片路径（或文件不存在）。请提供 .jpg/.png 等图片的完整路径。"
    _oct_init_once()
    img = Image.open(image_path).convert("RGB")
    x = _OCT_TFM(img).unsqueeze(0).to(_OCT_DEVICE)
    with torch.no_grad():
        outputs = _OCT_MODEL(x)
        probs = torch.softmax(outputs, dim=1).cpu().numpy().flatten()
        pred_idx = int(probs.argmax())
        confidence = probs[pred_idx]
        pred_class = OCT_CLASS_NAMES[pred_idx]
    cn_name, desc = OCT_CN_MAP.get(pred_class, (pred_class, "该疾病含义待确认"))
    return (f"OCT图像疾病诊断结果：{cn_name}（{pred_class}）\n"
            f"解释：{desc}\n"
            f"模型置信度：{confidence:.1%}\n\n"
            f"请注意：本结果仅为AI辅助分析，具体诊断请以临床医生判断为准。")


# ==========================================================
# 无赤光眼底 7类疾病分类
# ==========================================================
_NORED_READY = False
_NORED_DEVICE = None
_NORED_MODEL = None
_NORED_TFM = None
_NORED_CLASS_INDICES = None

NORED_CFG = {
    "ckpt_path": r"E:\崔心赞\densenet\weights\best_model.pth",
    "json_path": r'E:\崔心赞\densenet\data\class_indices.json',
}


def _nored_init_once():
    global _NORED_READY, _NORED_DEVICE, _NORED_MODEL, _NORED_TFM, _NORED_CLASS_INDICES
    if _NORED_READY:
        return
    _NORED_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 7
    _NORED_MODEL = densenet201(num_classes=num_classes).to(_NORED_DEVICE)
    sd = torch.load(NORED_CFG["ckpt_path"], map_location=_NORED_DEVICE)
    _NORED_MODEL.load_state_dict(sd)
    _NORED_MODEL.eval()
    with open(NORED_CFG["json_path"], "r", encoding="utf-8") as f:
        _NORED_CLASS_INDICES = json.load(f)
    _NORED_TFM = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    _NORED_READY = True


@tool
def predict_nored_fundus_disease(image_path: str) -> str:
    """对无赤光眼底图像进行7类疾病分类，返回预测疾病名称+置信度（仅适用于无赤光眼底图像）"""
    image_path = normalize_path(image_path)
    if not is_existing_image_path(image_path):
        return "❌ 输入不是有效的本地图片路径（或文件不存在）。请提供 .jpg/.png 等图片的完整路径。"
    _nored_init_once()
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        return f"❌ 打开图片失败：{e}"
    x = _NORED_TFM(img).unsqueeze(0).to(_NORED_DEVICE)
    with torch.no_grad():
        output = torch.squeeze(_NORED_MODEL(x)).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).item()
        prob = predict[predict_cla].item()
    class_name = _NORED_CLASS_INDICES.get(str(predict_cla), f"未知类别 ({predict_cla})")
    return (f"无赤光眼底图像疾病诊断结果：{class_name}\n"
            f"模型置信度：{prob:.3f}\n\n"
            f"请注意：本结果仅为AI辅助分析，具体诊断请以临床医生判断为准。")


# ==========================================================
# 7) LLM + Agent
# ==========================================================
os.environ["OPENAI_API_KEY"] = "your-key"

model = ChatOpenAI(
    model="claude-haiku-4-5-20251001",
    temperature=0.3,
    max_tokens=4096,
    timeout=180,
    max_retries=5,
    openai_api_key=os.environ["OPENAI_API_KEY"],
    openai_api_base="https://lanyiapi.com/v1",
)


def build_agent():
    return create_agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        tools=[
            classify_medical_image,
            generate_kdcmn_report,
            retrieve_from_rag,
            classify_vitreous_image,
            predict_rfmid_labels,
            predict_dr_grade,
            predict_glaucoma_grade,
            predict_oct_disease,
            predict_nored_fundus_disease,
        ],
        context_schema=Context,

        checkpointer=InMemorySaver(),
    )


agent = build_agent()
current_thread_id = str(uuid.uuid4())

# 临时文件目录（保存上传图片）
TMP_DIR = tempfile.mkdtemp(prefix="eyeagent_")





# ==========================================================
# 8) FastAPI 应用
# ==========================================================
app = FastAPI(title="眼科医学影像智能分析系统 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境请改为具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/new-chat")
async def new_chat():
    """开启新对话，重置 thread_id 和 agent"""
    global current_thread_id, agent
    current_thread_id = str(uuid.uuid4())
    agent = build_agent()
    return {"thread_id": current_thread_id, "status": "ok"}


@app.post("/api/chat")
async def chat(
    message: str = Form(""),
    file: Optional[UploadFile] = File(None),
):
    """
    发送消息（支持可选图片上传）。
    前端使用 FormData：
      form.append("message", "用户文本")
      form.append("file", imageFile)  // 可选
    """
    global current_thread_id, agent

    image_path = None

    # 保存上传图片到临时目录
    if file and file.filename:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in IMAGE_EXTS:
            raise HTTPException(status_code=400, detail=f"不支持的文件类型：{ext}")
        tmp_path = os.path.join(TMP_DIR, f"{uuid.uuid4()}{ext}")
        with open(tmp_path, "wb") as f_out:
            shutil.copyfileobj(file.file, f_out)
        image_path = tmp_path

    content = (message.strip() or "你好")
    if image_path and is_existing_image_path(image_path):
        content = f"{content}\n\n（已上传图像路径：{image_path}）"

    try:
        resp = agent.invoke(
            {"messages": [{"role": "user", "content": content}]},
            config={"configurable": {"thread_id": current_thread_id}},
            context=Context(
                image_path=image_path if (image_path and is_existing_image_path(image_path)) else None
            ),
        )
        reply = resp["messages"][-1].content
        return {"reply": reply, "status": "ok"}


    except Exception as e:

        traceback.print_exc()

        return JSONResponse(

            status_code=500,

            content={

                "reply": f"❌ 错误：{str(e)}",

                "status": "error"

            }

        )


# ==========================================================
# 启动
# ==========================================================
app.mount("/", StaticFiles(directory=".", html=True), name="static")
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
