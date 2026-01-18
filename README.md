# Milvus 向量数据库进阶使用指南 (Milvus RAG Retrieval Guide)

本仓库包含了一套完整的 Milvus 向量数据库使用指南与代码示例，专注于构建高性能的 RAG（检索增强生成）检索流程。内容涵盖环境部署、数据管理、混合检索（Hybrid Search）以及重排序（Rerank）等核心环节。

## 目录 (Table of Contents)

* [项目介绍](https://www.google.com/search?q=%23-%E9%A1%B9%E7%9B%AE%E4%BB%8B%E7%BB%8D)
* [环境准备](https://www.google.com/search?q=%23-%E7%8E%AF%E5%A2%83%E5%87%86%E5%A4%87)
* [快速开始](https://www.google.com/search?q=%23-%E5%BF%AB%E9%80%9F%E5%BC%80%E5%A7%8B)
* [核心功能模块](https://www.google.com/search?q=%23-%E6%A0%B8%E5%BF%83%E5%8A%9F%E8%83%BD%E6%A8%A1%E5%9D%97)
  * [1. 数据与权限管理](https://www.google.com/search?q=%231-%E6%95%B0%E6%8D%AE%E4%B8%8E%E6%9D%83%E9%99%90%E7%AE%A1%E7%90%86)
  * [2. 粗召回 (Coarse Retrieval)](https://www.google.com/search?q=%232-%E7%B2%97%E5%8F%AC%E5%9B%9E-coarse-retrieval)
  * [3. 精排 (Fine Reranking)](https://www.google.com/search?q=%233-%E7%B2%BE%E6%8E%92-fine-reranking)
  * [4. 便捷模型调用](https://www.google.com/search?q=%234-%E4%BE%BF%E6%8D%B7%E6%A8%A1%E5%9E%8B%E8%B0%83%E7%94%A8)
* [可视化工具](https://www.google.com/search?q=%23-%E5%8F%AF%E8%A7%86%E5%8C%96%E5%B7%A5%E5%85%B7)
* [参考文档](https://www.google.com/search?q=%23-%E5%8F%82%E8%80%83%E6%96%87%E6%A1%A3)

---

## 项目介绍

本项目旨在提供 Milvus 在实际生产环境中的最佳实践，特别关注 **NLP** 和 **RAG** 场景下的高级检索技术。项目代码演示了如何从零开始构建一个支持  **BM25 关键词检索** 、**语义向量检索** 以及 **RRF 混合检索** 的系统，并集成了 **BGE-Reranker** 进行精细化排序。

---

## 环境准备

### 1. 启动 Milvus 服务

本项目推荐使用 Docker 容器化部署 Milvus Standalone 版本。

* **下载安装脚本** :
  **PowerShell**

```
  # 下载官方 standalone.bat 脚本
  Invoke-WebRequest https://raw.githubusercontent.com/milvus-io/milvus/refs/heads/master/scripts/standalone_embed.bat -OutFile standalone.bat
```

* **启动服务** :
  **PowerShell**

```
  standalone.bat start
```

### 2. Python 依赖安装

使用 `pip` 安装 Milvus 客户端及模型支持库：

**Bash**

```
pip install pymilvus
# 如果需要使用内置模型接口
pip install "pymilvus[model]"
pip install peft FlagEmbedding
```

### 3. 模型服务

部分高级检索（如 BGE-M3, BGE-Reranker）示例代码依赖 HuggingFace TEI (Text Embeddings Inference) 服务接口。基于Docker部署后可在Milvus中调用。

---

## 快速开始

克隆本仓库并按以下顺序阅读或运行 Notebook：

1. **基础入门** : 运行 `milvus_数据管理.ipynb` 了解如何创建 Collection 和 Schema。
2. **构建检索** : 运行 `milvus_粗召回.ipynb` 实现 关键词匹配、语义匹配、混合检索。
3. **优化结果** : 运行 `milvus_精召回.ipynb` 对召回结果进行语义重排序。

---

## 核心功能模块

### 1. 数据与权限管理

* **Schema 设计** : 涵盖 `FLOAT_VECTOR` (Dense), `SPARSE_FLOAT_VECTOR` (Sparse), `VARCHAR`, `JSON` 等多种数据类型定义。
* **索引构建** :
* Dense: `AUTOINDEX`, `HNSW`, `IVF_FLAT` (支持 L2, IP, COSINE)。
* Sparse: `SPARSE_INVERTED_INDEX` (配合 BM25 算法参数)。
* **CRUD 操作** : 数据的 Insert / Upsert、Search / Query (带 Filter)、Delete 操作示例。
* **权限控制** : 用户创建、密码修改及鉴权配置。

### 2. 粗召回 (Coarse Retrieval)

本模块实现了关键词匹配、语义匹配、多路召回策略，以提高检索覆盖率：

* **BM25 关键词检索** : 通过 Schema 中的 `Function` 自动将文本转换为稀疏向量，实现高效的倒排索引检索。
* **BGE-M3 语义检索** : 集成 HuggingFace TEI 接口，处理长文本（支持 8194 tokens）的稠密向量检索。
* **Hybrid Search (混合检索)** : 使用 `AnnSearchRequest` 结合 `RRF` (Reciprocal Rank Fusion) 算法，融合稀疏向量和稠密向量的检索结果，平衡关键词匹配与语义理解。

### 3. 精排 (Fine Reranking)

在粗召回的基础上，引入 Cross-Encoder 模型进行二次排序：

* **BGE-Reranker-Base** : 利用 Attention 内部权重进行 Token 级别的深度语义匹配。
* **集成方式** :
* 作为 `search()` 或 `hybrid_search()` 的 `ranker` 参数直接调用。
* 通过 `pymilvus.model` 自行调用模型库。
* 支持通过 HTTP 调用 TEI Rerank 接口，或本地加载 PyTorch 模型。

### 4. 便捷模型调用

演示了如何使用 `pymilvus[model]` 快速调用 SentenceTransformer 和 CrossEncoder，简化本地开发流程。

---

## 可视化工具

推荐使用Dock Desktop作为Docker管理工具。  

推荐使用 **Attu** 作为 Milvus 的 GUI 管理工具：

* **功能** : 集合管理、向量搜索测试、系统监控。

---

## 参考文档

* [Milvus 官方文档](https://milvus.io/docs)
* [HuggingFace TEI 部署指南](https://huggingface.co/docs/text-embeddings-inference)
