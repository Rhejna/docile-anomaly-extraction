# Document Extraction & Anomaly Detection

An end-to-end pipeline for key information extraction from business documents (invoices, receipts, forms) with a downstream anomaly detection layer, benchmarked on DocILE and SROIE.

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
