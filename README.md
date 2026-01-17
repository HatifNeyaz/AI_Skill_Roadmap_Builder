# FlowLearn AI: Multimodal Learning Roadmap Generator 🗺️ (Multi Agent Orchestration)

**FlowLearn AI** is an intelligent, multi-agent orchestration platform that transforms any learning goal, whether typed, spoken, or handwritten, into a structured, interactive roadmap.

It uses a **"Committee of Experts"** architecture where specialized AI agents (an Academic Professor and a Practical Engineer) brainstorm distinct learning paths. A "Chief Architect" agent then synthesizes these perspectives into a visual flowchart and curates verified learning resources (YouTube tutorials, documentation, courses) for each step.

---

## 🚀 Key Features

* **Multimodal Input:**
    * **📝 Text:** Type any topic (e.g., "Learn Rust for Systems Programming").
    * **🎙️ Audio:** Speak your goals; the system transcribes and analyzes your voice.
    * **🖼️ Vision:** Upload handwritten notes or whiteboard diagrams; the system extracts the concepts.
* **Multi-Agent Orchestration:**
    * **Agent A (The Theorist):** Focuses on foundational concepts and academic rigor.
    * **Agent B (The Practitioner):** Focuses on hands-on projects and real-world skills.
    * **The Big Boss (The Architect):** Merges both views into a coherent strategy.
* **Interactive Flowcharts:** Generates renderable Mermaid.js diagrams for visual learning.
* **Smart Resource Curation:** Scours the web for *real* links (YouTube, Medium, Blogs) specific to your roadmap's nodes.

---

## 🛠️ Tech Stack & Tools

| Component | Tool / Library | Role |
| :--- | :--- | :--- |
| **Frontend** | `Streamlit` | Interactive web UI for inputs and flowchart rendering. |
| **LLM Engine** | `Groq` | Ultra-fast inference API. |
| **Orchestration** | `LangChain` | Managing prompts, chains, and agent workflows. |
| **Audio Model** | `whisper-large-v3` | Transcribing user voice notes with high accuracy. |
| **Vision Model** | `llama-3.2-90b-vision` | Analyzing handwritten notes and images. |
| **Reasoning Models** | `llama-3.1-8b` / `llama-3.3-70b` | Powering the specialist agents and the architect. |
| **Search Tool** | `duckduckgo-search` | Fetching live URLs for tutorials and courses. |
| **Visualization** | `streamlit-mermaid` | Rendering the generated flowchart code. |

---

## 🧩 System Architecture

```mermaid
graph TD
    User([User Inputs])
    
    subgraph Inputs ["Input Processing"]
        direction TB
        Text[Text Input]
        Audio[Audio Input] -->|Whisper-v3| Trans[Transcript]
        Image[Image Input] -->|Llama Vision| Desc[Description]
    end
    
    User --> Inputs
    
    subgraph Committee ["The Committee of Experts"]
        direction TB
        AgentA[Agent A: Academic Professor]
        AgentB[Agent B: Practical Engineer]
        Boss[The Big Boss: Architect]
    end
    
    Inputs --> AgentA
    Inputs --> AgentB
    
    AgentA -->|Theoretical Path| Boss
    AgentB -->|Project Path| Boss
    
    Boss -->|Mermaid Code| Chart[Flowchart Generation]
    
    subgraph Resources ["Resource Discovery"]
        Chart --> Ext[Topic Extractor]
        Ext --> Search[DuckDuckGo Search]
        Search --> Curator[Resource Curator Agent]
    end
    
    Chart --> UI[Streamlit Display]
    Curator --> UI
    
    classDef light fill:#c3e6cb,stroke:#155724,stroke-width:2px,color:#000;
    class Inputs,Committee,Resources box;