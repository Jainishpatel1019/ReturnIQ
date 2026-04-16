# Architecture
ReturnIQ uses an **Offline-to-Online (O2O)** architecture to maintain high-fidelity causal inference with zero-latency reporting.

```mermaid
graph TD
    subgraph "Offline Pipeline (3.7M Reviews)"
        A[Amazon Review Corpus] --> B[DuckDB Feature Warehouse]
        B --> C[DistilBERT Sentiment Scorer]
        C --> D[EconML CausalForestDML]
        D --> E[Serialized Artifacts: .pkl / .parquet]
    end
    
    subgraph "Online Dashboad (Streamlit)"
        E --> F[Inference Logic]
        F --> G[Policy Simulator]
        F --> H[Seller Deep-Dive]
        G --> I[User Interaction]
    end
```

### Data Flow
1. **Ingestion**: Raw review JSONs are normalized into a relational DuckDB instance.
2. **Identification**: We use a Double ML strategy to isolate the Treatment (Seller Quality) from Confounders.
3. **Serialization**: The final Heterogeneous Treatment Effects (CATE) are saved alongside the full estimator.
4. **Telemetry**: The Streamlit frontend uses the serialized model for live 'What-if' forecasting.
