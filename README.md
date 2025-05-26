# File Sorter - Intelligent Content-Based File Organization

An advanced file organization system that automatically analyzes and sorts files based on their actual content rather than just file extensions. Uses local AI models for semantic analysis and computer vision for image classification.

## 🚀 Features

### Content-First Clustering
- **Semantic Analysis**: Uses sentence transformers to understand document content
- **Multi-Modal Feature Extraction**: Combines text embeddings with structural and linguistic features
- **Domain Detection**: Automatically categorizes files into academic, business, technical, programming, and other domains
- **Adaptive Clustering**: Dynamically determines optimal cluster counts using multiple validation metrics

### Image Classification
- **ResNet50 CNN**: Uses pre-trained ResNet50 with ImageNet weights for image classification
- **Broad Categorization**: Groups images into categories like vehicles, animals, people, sports, food, landscapes, buildings, and objects
- **Confidence-Based Clustering**: Sub-clusters images based on classification confidence levels
- **Visual Content Analysis**: Analyzes image dimensions, aspect ratios, and visual features

### Intelligent Organization
- **Content-Aware Naming**: Generates meaningful folder names based on actual content themes
- **Hierarchical Structure**: Creates organized folder structures with descriptive names
- **Duplicate Detection**: Identifies and handles duplicate files using content hashing
- **Policy Generation**: Creates detailed organization policies explaining the sorting logic

## 📁 Supported File Types

### Text Files
- **Documents**: `.md`, `.txt`, `.doc`, `.docx`, `.pdf`, `.rtf`, `.odt`, `.tex`
- **Code Files**: `.py`, `.js`, `.java`, `.cpp`, `.c`, `.h`, `.cs`, `.php`, `.go`, `.rb`, `.swift`, `.kt`, `.rs`, `.scala`, `.r`, `.m`, `.sh`, `.bat`
- **Data Files**: `.csv`, `.json`, `.xml`, `.yaml`, `.yml`, `.tsv`, `.parquet`, `.xlsx`, `.xls`, `.sqlite`, `.db`
- **Configuration**: `.ini`, `.conf`, `.cfg`, `.config`, `.properties`, `.env`

### Image Files
- **Common Formats**: `.jpg`, `.jpeg`, `.png`, `.gif`, `.bmp`, `.tiff`, `.webp`
- **Advanced Analysis**: Extracts visual features, classifies content, and groups by semantic similarity

## 🛠️ Technical Architecture

### Core Components
- **Enhanced Cluster Engine**: Content-first clustering with domain detection
- **Image Analyzer**: ResNet50-based computer vision analysis
- **Text Analyzer**: Semantic embedding and keyword extraction
- **Category Namer**: Intelligent naming based on content themes
- **File Organizer**: Physical file organization and policy generation

### AI Models Used
- **Sentence Transformers**: `all-MiniLM-L6-v2` for text embeddings
- **ResNet50**: Pre-trained CNN with ImageNet weights for image classification
- **HDBSCAN**: Density-based clustering for variable cluster sizes
- **Gaussian Mixture Models**: For code file clustering by programming language

## 🎯 Use Cases

### Academic Research
- Organizes research papers by field and methodology
- Groups related studies and literature reviews
- Separates theoretical from empirical work

### Business Documents
- Sorts contracts, invoices, and proposals
- Groups meeting notes and project documentation
- Organizes financial and strategic planning documents

### Software Development
- Clusters code files by functionality and language
- Groups documentation, tutorials, and examples
- Organizes configuration and setup files

### Media Collections
- Classifies images by content (people, animals, vehicles, etc.)
- Groups photos by scene type (landscapes, buildings, objects)
- Organizes visual content by confidence levels

## 🚫 Limitations

### File Type Limitations
- **Binary Files**: Cannot analyze proprietary binary formats without specific parsers
- **Encrypted Files**: Cannot process password-protected or encrypted content
- **Large Files**: Performance may degrade with very large files (>100MB)
- **Corrupted Files**: Cannot process damaged or corrupted files

### Content Analysis Limitations
- **Language Support**: Optimized for English text (other languages may have reduced accuracy)
- **Domain Specificity**: Best performance on academic, business, and technical content
- **Context Understanding**: May struggle with highly specialized or niche terminology
- **Handwritten Content**: Cannot process handwritten text in images

### Technical Limitations
- **Local Processing**: Requires sufficient RAM and CPU for AI model inference
- **GPU Optional**: Faster with CUDA-compatible GPU but not required
- **Internet Independent**: Works entirely offline (no cloud dependencies)
- **Model Size**: Requires ~500MB disk space for AI models

## 📋 Requirements

### System Requirements
- **Python**: 3.8 or higher
- **RAM**: Minimum 4GB, recommended 8GB+
- **Storage**: 2GB free space for models and processing
- **OS**: Windows, macOS, or Linux
