亮眼 AI · 眼科影像智能分析系统
面向基层医疗机构的眼科多模态影像智能分析平台，集成 LLM Agent 推理引擎与九类专病分析模型，实现从图像上传到辅助诊断的全流程闭环。

Python FastAPI LangChain PyTorch License

📌 项目背景
基层医疗机构普遍面临眼科诊断资源匮乏、优质医疗资源分布不均的现实困境。本系统旨在通过 AI 辅助手段，帮助医生完成多模态眼科影像的快速筛查与辅助诊断，降低专科诊断门槛。

✨ 核心功能
功能模块	支持图像类型	说明
🔍 图像模态分类	全部	自动识别 OCT / OCTA / 彩色眼底 / 无赤光眼底 / 超声，共 5 类
📄 超声诊断报告生成	超声图像	基于 KD-CMN 模型自动生成结构化诊断报告
🧬 OCT 疾病分类	OCT 图像	7 类疾病分类（AMD / CNV / CSR / DME / DR / DRUSEN / NORMAL）
🩸 糖尿病视网膜病变分级	彩色眼底	DR 五级分级（0–4 级），输出中文等级名称 + 置信度
👁 青光眼分级	彩色眼底	三类分级（正常 / 早期 / 晚期青光眼）
🔬 眼底多标签疾病识别	彩色眼底	基于 RFMiD 数据集，支持 26 种眼底疾病标签
🌑 无赤光眼底分类	无赤光眼底	DenseNet201，7 类疾病分类
🫧 玻璃体超声分类	超声图像	AlexNet，5 类玻璃体病变分类
💬 眼科知识问答（RAG）	文本	FAISS 向量检索 + 眼科医学知识库，减少幻觉
🏗 系统架构
用户前端 (index.html)
      │  HTTP / FormData
      ▼
FastAPI 后端 (server.py)
      │
      ▼
LangChain / LangGraph Agent
      │  工具路由（准确率 100%）
      ├── classify_medical_image      → ResNet18 图像模态分类器
      ├── generate_kdcmn_report       → KD-CMN 超声报告生成
      ├── retrieve_from_rag           → FAISS + HuggingFace Embedder
      ├── classify_vitreous_image     → AlexNet 玻璃体分类
      ├── predict_rfmid_labels        → 眼底多标签分类
      ├── predict_dr_grade            → DR 五级分级
      ├── predict_glaucoma_grade      → 青光眼三类分级
      ├── predict_oct_disease         → KAN-ViT OCT 分类
      └── predict_nored_fundus_disease→ DenseNet201 无赤光眼底分类
🖥 界面预览
系统采用深色科技风 UI，具备：

粒子动效背景
侧边栏工具导航（可折叠）
图像上传预览 + 拖放支持
对话流式展示，支持 Markdown 渲染
快捷提示词（一键触发常用分析）
🛠 技术栈
后端

FastAPI + uvicorn — 异步 HTTP 服务
LangChain / LangGraph — Agent 框架与多工具调用
PyTorch — 模型推理（ResNet18 / ResNet101 / DenseNet201 / KAN-ViT）
FAISS + HuggingFace Transformers — 向量检索 RAG 知识库
前端

原生 HTML / CSS / JavaScript（无框架依赖）
Canvas 粒子动画背景
FormData 多模态上传
🚀 快速启动
1. 安装依赖
pip install fastapi uvicorn python-multipart torch torchvision \
            transformers faiss-cpu langchain langgraph langchain-openai \
            Pillow numpy pandas
2. 配置模型路径
在 server.py 中修改各模型的 ckpt_path 为本地实际路径：

CLASSIFIER_CFG = {
    "ckpt_path": "your/path/to/resnet18_3class_best.pth",
    ...
}
3. 配置 LLM API Key
os.environ["OPENAI_API_KEY"] = "your-api-key"
4. 启动服务
python server.py
访问 http://127.0.0.1:8000 即可使用。

📊 项目成果
✅ 集成 9 类眼科影像分析功能，覆盖 OCT、眼底、超声等多种模态
✅ Agent 工具路由准确率达 100%，可根据用户意图与图像模态自动匹配分析模型
✅ RAG 知识库支持 26 类眼底疾病的中文语义检索与问答
✅ 系统已取得软件著作权登记
⚠️ 免责声明
本系统输出结果仅为 AI 辅助分析，不构成临床诊断依据。具体诊断请以临床医生判断为准。

👤 作者
如有问题或合作意向，欢迎通过 GitHub Issues 联系。
