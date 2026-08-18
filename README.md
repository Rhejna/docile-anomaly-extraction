# Document Extraction & Anomaly Detection

A more complete and modern reproduction of a project I worked on in a company, focused on information extraction from banking documents and anomaly detection.

The goal is to build a pipeline that extracts structured information from documents (invoices, receipts, forms) and detects anomalies on the extracted data.

## Technical backbone (planned)
1. Ingestion pipeline (deskew, denoise, quality checks)
2. Baseline: Tesseract + rule-based extraction (comparison point only)
3. Modern approach: LayoutLMv3 / Donut / small VLM (Qwen2-VL-2B or Florence-2) fine-tuned for key information extraction
4. Evaluation: official DocILE / SROIE metrics + latency & cost per document
5. Anomaly layer: Isolation Forest + stronger model, evaluated with precision@k (not only AUC)
6. Delivery: FastAPI, Docker, tests, GitHub Actions, Gradio demo
7. Honest results table (baseline vs modern) + error analysis

## License
MIT
