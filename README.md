# Knowledge Q&A Agent

基于 RAG（检索增强生成）架构的公司内部知识库智能问答系统。支持 Web 界面和命令行两种交互方式，可对 PDF、Word、Markdown、TXT 等格式的文档进行索引和智能问答。

## ✨ 特性

- **🔍 RAG 问答** — 基于检索增强生成，回答严格依据已索引文档，杜绝幻觉
- **🌐 Web 界面** — 简洁美观的聊天 UI，开箱即用
- **💻 命令行** — 完整的 CLI 工具，适合脚本和自动化场景
- **🤖 多模型支持** — DeepSeek / OpenAI / Anthropic 三种 LLM 可切换
- **📄 多格式文档** — PDF、Word (.docx)、Markdown、TXT 一键索引
- **🧠 本地嵌入** — 使用 BGE-M3 模型在本地生成向量，无需外部 API 调用
- **💾 持久化存储** — 基于 ChromaDB 的向量数据库，索引一次永久使用
- **💬 多会话记忆** — 支持多个独立对话会话，历史记录自动持久化

## 🏗️ 架构

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Web UI     │     │   CLI (Typer) │     │   REST API   │
│  (HTML/CSS)  │     │   + Rich      │     │  (FastAPI)   │
└──────┬───────┘     └──────┬───────┘     └──────┬───────┘
       │                    │                    │
       └────────────────────┼────────────────────┘
                            │
                   ┌────────▼────────┐
                   │   RAG Chain     │
                   │  (LangChain)    │
                   └────────┬────────┘
                            │
          ┌─────────────────┼─────────────────┐
          │                 │                 │
   ┌──────▼──────┐  ┌──────▼──────┐  ┌───────▼──────┐
   │  Retriever  │  │    LLM      │  │   Memory     │
   │ (ChromaDB)  │  │ (DeepSeek/  │  │  (JSON File) │
   │             │  │  OpenAI/    │  │              │
   │             │  │  Anthropic) │  │              │
   └──────┬──────┘  └─────────────┘  └──────────────┘
          │
   ┌──────▼──────┐  ┌──────────────┐
   │ Embeddings  │  │  Ingestion   │
   │ (BGE-M3)    │  │  Pipeline    │
   └─────────────┘  └──────┬───────┘
                           │
                    ┌──────▼──────┐
                    │  Documents  │
                    │ PDF/DOCX/   │
                    │  MD/TXT     │
                    └─────────────┘
```

**处理流程**：文档 → 分块 → 向量嵌入 → 存入 ChromaDB → 用户提问 → 检索相关块 → LLM 生成回答

## 📋 环境要求

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/) 包管理器
- 至少一种 LLM 的 API Key（DeepSeek / OpenAI / Anthropic）

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone <repo-url>
cd knowledge-qa
```

### 2. 安装依赖

```bash
uv sync
```

### 3. 配置 API Key

```bash
cp .env.example .env
```

编辑 `.env` 文件，填入你的 LLM API Key。默认使用 DeepSeek：

```ini
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=sk-your-deepseek-key
```

### 4. 添加文档

将公司文档放入 `docs/` 目录。支持的格式：

| 格式 | 扩展名 | 说明 |
|------|--------|------|
| PDF | `.pdf` | 支持中文，自动提取文本 |
| Word | `.docx` | 2007 版及以后格式 |
| Markdown | `.md` | 保留标题结构 |
| 纯文本 | `.txt` | UTF-8 编码 |

```bash
# 示例：放入一些公司制度文档
cp ~/Documents/员工手册.pdf docs/
cp ~/Documents/考勤制度.docx docs/
cp ~/Documents/福利政策.md docs/
```

### 5. 索引文档

```bash
uv run knowledge-qa ingest
```

首次运行会自动下载 BGE-M3 嵌入模型（约 2GB），后续无需重复下载。

```
+----------------------------- Ingestion Summary -----------------------------+
| Files processed: 3                                                          |
| Chunks ingested: 47                                                         |
| Status: OK Complete                                                         |
+-----------------------------------------------------------------------------+
```

### 6. 启动 Web 界面

```bash
uv run knowledge-qa web
```

浏览器打开 `http://127.0.0.1:8000`，开始提问。

> 自定义端口：`uv run knowledge-qa web --port 8080`

### 7. 或使用命令行提问

```bash
uv run knowledge-qa ask "公司年假政策是什么？"
```

## 🖥️ Web 界面

启动 Web 服务后，访问 `http://127.0.0.1:8000` 即可使用聊天界面。

**功能：**
- 💬 对话式问答，支持多轮对话
- 📊 点击 📊 按钮查看知识库状态
- 🗑 点击 🗑 按钮清除当前会话
- 📄 每条回答自动附带来源文档引用

**Web API 端点：**

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/` | Web 聊天界面 |
| `POST` | `/api/ask` | 提问 `{"question": "...", "session_id": "..."}` |
| `GET` | `/api/status` | 知识库统计信息 |
| `POST` | `/api/clear` | 清除会话 `{"session_id": "..."}` |
| `POST` | `/api/ingest` | 重新索引文档 |

## 💻 命令行

```bash
# 索引全部文档
uv run knowledge-qa ingest

# 索引指定目录
uv run knowledge-qa ingest --docs-dir ./my_docs

# 索引单个文件
uv run knowledge-qa ingest --file ./docs/policy.pdf

# 提问
uv run knowledge-qa ask "远程办公的规定是什么？"

# 多会话（不同 session 独立记忆）
uv run knowledge-qa ask "年假天数？" --session-id hr
uv run knowledge-qa ask "加班费怎么算？" --session-id hr

# 查看知识库状态
uv run knowledge-qa status

# 清除会话历史
uv run knowledge-qa clear --session-id hr

# 启动 Web 服务
uv run knowledge-qa web
uv run knowledge-qa web --host 0.0.0.0 --port 8080
```

## ⚙️ 配置详解

所有配置通过 `.env` 文件设置：

### LLM 模型

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `LLM_PROVIDER` | 模型提供商：`deepseek`、`openai`、`anthropic` | `deepseek` |
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥 | — |
| `DEEPSEEK_MODEL` | DeepSeek 模型名 | `deepseek-v4-pro` |
| `DEEPSEEK_BASE_URL` | DeepSeek API 地址 | `https://api.deepseek.com` |
| `OPENAI_API_KEY` | OpenAI API 密钥 | — |
| `OPENAI_MODEL` | OpenAI 模型名 | `gpt-4o-mini` |
| `ANTHROPIC_API_KEY` | Anthropic API 密钥 | — |
| `ANTHROPIC_MODEL` | Anthropic 模型名 | `claude-sonnet-4-20250514` |

**切换模型示例：**

```ini
# 使用 OpenAI
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-openai-key
OPENAI_MODEL=gpt-4o

# 使用 Anthropic
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-your-key
ANTHROPIC_MODEL=claude-sonnet-4-20250514
```

### 嵌入模型

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `EMBEDDING_PROVIDER` | 嵌入提供商：`local`、`openai` | `local` |
| `EMBEDDING_MODEL` | 嵌入模型名 | `BAAI/bge-m3` |
| `LOCAL_MODEL_DIR` | 本地模型存储目录 | `models` |

默认使用 BGE-M3 本地模型（支持中英文，1024 维向量），首次运行自动从 ModelScope 下载。

### 文档分块与检索

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `CHUNK_SIZE` | 分块大小（tokens） | `500` |
| `CHUNK_OVERLAP` | 块间重叠（tokens） | `100` |
| `RETRIEVAL_K` | 检索返回文档数 | `4` |

### 路径

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `DOCS_DIR` | 文档存放目录 | `docs` |
| `CHROMA_PERSIST_DIR` | 向量数据库目录 | `chroma_db` |
| `MEMORY_FILE` | 对话历史文件 | `conversation_history.json` |

## 📁 项目结构

```
knowledge-qa/
├── docs/                       # 文档目录（PDF/Word/MD/TXT 放这里）
├── src/
│   ├── cli/                    # 命令行入口
│   │   ├── app.py              # Typer 应用 + 命令注册
│   │   └── commands.py         # 命令实现（ingest/ask/status/clear/web）
│   ├── config/
│   │   └── settings.py         # 配置管理（pydantic-settings）
│   ├── embeddings/
│   │   ├── embedder.py         # 嵌入模型工厂（local/openai）
│   │   └── model_downloader.py # BGE-M3 模型下载（ModelScope）
│   ├── ingestion/
│   │   ├── loader.py           # 文档加载器（PDF/DOCX/MD/TXT）
│   │   ├── pipeline.py         # 索引流水线（加载→分块→嵌入→存储）
│   │   └── splitter.py         # 文档分块策略
│   ├── llm/
│   │   ├── base.py             # LLM Provider 接口协议
│   │   ├── deepseek_provider.py
│   │   ├── openai_provider.py
│   │   └── anthropic_provider.py
│   ├── memory/
│   │   └── history.py          # 对话历史管理（JSON 持久化）
│   ├── rag/
│   │   ├── chain.py            # RAG 链组装（检索增强生成）
│   │   └── retriever.py        # 检索器配置
│   ├── vectordb/
│   │   └── store.py            # ChromaDB 向量库封装
│   └── web/
│       ├── app.py              # FastAPI 应用 + REST API
│       └── templates/
│           └── index.html      # Web 聊天界面
├── tests/                      # 单元测试
├── models/                     # 本地模型（自动下载，不入 git）
├── chroma_db/                  # 向量数据库（自动生成，不入 git）
├── .env.example                # 环境变量模板
├── pyproject.toml              # 项目配置与依赖
└── README.md
```

## 🔄 RAG 工作流程

```
1. 文档索引（Ingestion）
   文档 → 加载器(Loader) → 分块器(Splitter) → 嵌入(Embeddings) → 存入 ChromaDB

2. 问答检索（Query）
   用户提问 → 历史感知重述(History-Aware) → 向量检索(Retrieval) → LLM 生成(Generation)
                                         ↓
                                  对话历史记忆(Memory)
```

### 索引阶段

1. **文档加载** — 根据文件扩展名自动选择加载器（PyMuPDF / Docx2txt / TextLoader）
2. **文本分块** — 使用 `RecursiveCharacterTextSplitter`，按段落 > 句子 > 字符的优先级切分，块大小 500 tokens，重叠 100 tokens
3. **向量嵌入** — BGE-M3 模型将每个文本块转换为 1024 维向量
4. **持久化存储** — 向量存入 ChromaDB，数据保存在 `chroma_db/` 目录

### 问答阶段

1. **问题重述** — 结合对话历史，将多轮对话中的上下文依赖转为独立问题
2. **向量检索** — 在 ChromaDB 中检索 Top-K（默认 4）个最相关的文档块
3. **上下文增强** — 将检索到的文档块拼入 System Prompt
4. **LLM 生成** — 模型基于提供的上下文生成回答，无法回答时明确告知
5. **来源引用** — 自动标注回答所引用的源文档

## 🧪 运行测试

```bash
uv run pytest tests/ -v
```

## 📄 License

MIT
