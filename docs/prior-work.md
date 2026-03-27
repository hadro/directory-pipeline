# Prior work and inspirations

Annotated citations of foundational papers and related projects. See the [main README](../README.md) for the overview.

---

**Greif, Griesshaber & Greif (2025) — "Multimodal LLMs for OCR, OCR Post-Correction, and Named Entity Recognition in Historical Documents"** ([arXiv:2504.00414](https://arxiv.org/abs/2504.00414)) / [pipeline](https://github.com/niclasgriesshaber/gemini_historical_dataset_pipeline) / [benchmarking code](https://github.com/niclasgriesshaber/llm_historical_dataset_benchmarking)

The foundational paper for this pipeline's architecture. Benchmarks multimodal LLMs against Tesseract and Transkribus on 18th–19th century German city directories, finding that feeding both the source image *and* noisy conventional OCR into a multimodal LLM produces far lower error rates than either alone (0.84% CER with Gemini 2.0 Flash). The paper directly motivates the two-stage design (Tesseract for coordinates → Gemini for accuracy), temperature 0.0, and the separation of OCR, post-correction, and NER into distinct pipeline stages.

**Bell, Marlow, Wombacher et al. (2020) — *directoreadr*** ([PLOS ONE 15(8): e0220219](https://doi.org/10.1371/journal.pone.0220219)) / [code](https://github.com/brown-ccv/directoreadr)

The closest prior work: an end-to-end pipeline for extracting geocoded business data from scanned Polk city directory yellow pages (Providence, RI, 1936–1990) using classical computer vision, Tesseract, fuzzy street matching, and ArcGIS geocoding. Achieves 94.4% automated page processing. Documents the brittle year-specific heuristics required for header detection and the historical street change problem that dominates geocoding failures — both of which motivate the mLLM-based approach here.

**Fleischhacker, Kern & Göderle (2025) — "Enhancing OCR in historical documents with complex layouts through machine learning"** ([Int. J. Digital Libraries 26:3](https://doi.org/10.1007/s00799-025-00413-z))

Demonstrates that layout detection as a preprocessing step improves OCR accuracy by over 15 percentage points on multi-column historical documents (Habsburg civil service directories). The key mechanism: without layout detection, Tesseract reads across columns rather than down them, scrambling the text. This directly motivates the column reading-order correction in `align_ocr.py` and the `detect_columns.py` / `surya_detect.py` stages.

**Cook, Jones, Rosé & Logan (2020) — "The Green Books and the Geography of Segregation in Public Accommodations"** ([NBER Working Paper 26819](https://www.nber.org/papers/w26819))

The canonical prior digitization of the Green Books. Establishes that the pre-mLLM state of the art was entirely manual data entry (OCR was explicitly rejected due to irregular formatting and ad placement), uses the US Census Geocoder as a national baseline (~50% exact match), and produces the canonical six-category establishment taxonomy used in the NER schema here. Also documents the cross-year identity matching problem and calls for city directory cross-referencing as a research next step.

**Smith & Cordell (2018) — "A Research Agenda for Historical and Multilingual Optical Character Recognition"** ([Northeastern University / NEH](https://repository.library.northeastern.edu/files/neu:f1881m035))

A practitioner-consensus research agenda identifying layout analysis as the top barrier to historical OCR progress and OCR post-correction as high-leverage and underinvested. Validates line-level sequence alignment for ground truth creation (the same approach as Needleman-Wunsch alignment used here) and argues that "how dirty is too dirty" is a task-specific empirical question — informing the pipeline's decision to expose confidence metrics rather than hard-filter at a fixed threshold.

**Berenbaum, Deighan, Marlow et al. (2016) — *georeg*** ([arXiv:1612.00992](https://arxiv.org/abs/1612.00992)) / [code](https://bitbucket.org/brown-data-science/georeg)

Predecessor to *directoreadr*, applying a morphological contour merging + k-means column clustering approach to Rhode Island manufacturing registries. Documents the cross-page context inheritance pattern (nearest heading above, including the last heading from the prior page) and the sobering finding that geocoding success compounds OCR errors, parse failures, and historical street changes — even with 99% record identification, geocoding reached only 61%.

**Carlson, Bryan & Dell (2023) — *EffOCR*: "Efficient OCR for Building a Diverse Digital History"** ([arXiv:2304.02737](https://arxiv.org/abs/2304.02737)) / [code](https://github.com/dell-research-harvard/effocr)

Provides the key OCR benchmarks on historical US newspapers: off-the-shelf Tesseract at ~10.6% CER, fine-tuned TrOCR at 1.3% CER. These establish both the baseline noisy-input quality for the alignment stage and the comparison target for evaluating whether mLLM post-correction is competitive with conventional fine-tuned OCR. Also documents Google Cloud Vision's failure on full-page historical newspaper scans — reinforcing the decision to use Tesseract as the noisy-input stage rather than a cloud OCR API.

**HuggingFace (2025) — "Supercharge your OCR Pipelines with Open Models"** ([huggingface.co/blog/ocr-open-models](https://huggingface.co/blog/ocr-open-models))

A practitioner survey of the current open-weight VLM-based OCR landscape (Nanonets-OCR2, PaddleOCR-VL, dots.ocr, OlmOCR-2, Granite-Docling, DeepSeek-OCR, Chandra, Qwen3-VL) that introduces "locality awareness" — the ability to produce corrected text paired with bounding boxes — as a first-class capability distinction. Models with grounding support (Chandra, OlmOCR-2, dots.ocr, Granite-Docling) could in principle replace the Tesseract → Needleman-Wunsch alignment pipeline with a single-pass architecture. None has been tested on degraded historical scans; Granite-Docling (258M parameters, CPU-runnable, DocTags structured output) is the most tractable starting point for empirical evaluation on Green Books pages.

**Wolf, Chioh, Balogh & Spaan (2020) — "New York City Directories Extracted Persons Entries, 1850–1890"** ([NYU Faculty Digital Archive, hdl.handle.net/2451/61521](https://archive.nyu.edu/handle/2451/61521))

A dataset of machine-readable entries extracted from NYPL-digitized New York City directories (Doggett's 1850–51; Trow/Wilson 1852–1890), covering names, occupations, and work and home addresses across forty annual editions. Released as 40 NDJSON files under CC-BY-SA-NC 4.0. A direct precedent for applying this pipeline to city directories: the same NYPL collections, the same structured entry types (name, occupation, address), and a concrete existence proof that machine-readable extraction at scale is achievable for this document type.
