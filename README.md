# AutoGluon 

This repository contains a comprehensive collection of Jupyter notebooks demonstrating various capabilities of AutoGluon for automated machine learning across different domains.

## Tabular
- [`tabular_quick_start.ipynb`](Tabular/tabular_quick_start.ipynb): Quick start guide for tabular prediction
- [`tabular_essentials.ipynb`](Tabular/tabular_essentials.ipynb): Essential functionality for tabular prediction
- [`tabular_feature_engineering.ipynb`](Tabular/tabular_feature_engineering.ipynb): Guide to feature engineering
- [`tabular_foundational_models.ipynb`](Tabular/tabular_foundational_models.ipynb): Working with foundation models for tabular data
- [`tabular_indepth.ipynb`](Tabular/tabular_indepth.ipynb): In-depth exploration of tabular prediction capabilities
- [`tabular_multimodal.ipynb`](Tabular/tabular_multimodal.ipynb): Working with multimodal data

### Advanced Tabular
- [`tabular_custom_metric.ipynb`](Tabular/Advanced/tabular_custom_metric.ipynb): Adding custom evaluation metrics
- [`tabular_custom_model.ipynb`](Tabular/Advanced/tabular_custom_model.ipynb): Creating custom models
- [`tabular_custom_model_advanced.ipynb`](Tabular/Advanced/tabular_custom_model_advanced.ipynb): Advanced custom model features
- [`tabular_deployment.ipynb`](Tabular/Advanced/tabular_deployment.ipynb): Deployment optimization
- [`tabular_gpu.ipynb`](Tabular/Advanced/tabular_gpu.ipynb): GPU training
- [`tabular_kaggle.ipynb`](Tabular/Advanced/tabular_kaggle.ipynb): Using AutoGluon for Kaggle competitions
- [`tabular_multilabel.ipynb`](Tabular/Advanced/tabular_multilabel.ipynb): Multi-label prediction

## Multimodal

### Image Prediction
- [`beginner_image_cls.ipynb`](Multimodal/Image%20Prediction/beginner_image_cls.ipynb): Getting started with image classification
- [`clip_zeroshot.ipynb`](Multimodal/Image%20Prediction/clip_zeroshot.ipynb): Zero-shot image classification using CLIP

### Image Segmentation
- [`beginner_semantic_seg.ipynb`](Multimodal/Image%20Segmentation/beginner_semantic_seg.ipynb): Introduction to semantic segmentation

### Object Detection
#### Quick Start
- [`quick_start_coco.ipynb`](Multimodal/Object%20Detection/Object%20Detection%20Quick%20Start/quick_start_coco.ipynb): Quick start guide for object detection using COCO dataset

#### Advanced
- [`finetune_coco.ipynb`](Multimodal/Object%20Detection/Object%20Detection%20Advanced/finetune_coco.ipynb): Fine-tuning models on COCO dataset

#### Data Preparation
- [`convert_data_to_coco_format.ipynb`](Multimodal/Object%20Detection/Object%20Detection%20Data%20Preparation/convert_data_to_coco_format.ipynb): Converting data to COCO format
- [`prepare_coco17.ipynb`](Multimodal/Object%20Detection/Object%20Detection%20Data%20Preparation/prepare_coco17.ipynb): COCO 2017 dataset preparation
- [`prepare_voc.ipynb`](Multimodal/Object%20Detection/Object%20Detection%20Data%20Preparation/prepare_voc.ipynb): Pascal VOC dataset preparation
- [`prepare_pothole.ipynb`](Multimodal/Object%20Detection/Object%20Detection%20Data%20Preparation/prepare_pothole.ipynb): Pothole dataset preparation
- [`prepare_watercolor.ipynb`](Multimodal/Object%20Detection/Object%20Detection%20Data%20Preparation/prepare_watercolor.ipynb): Watercolor dataset preparation
- [`voc_to_coco.ipynb`](Multimodal/Object%20Detection/Object%20Detection%20Data%20Preparation/voc_to_coco.ipynb): Converting VOC to COCO format

### Document Prediction
- [`document_classification.ipynb`](Multimodal/Document%20Prediction/document_classification.ipynb): Document classification tasks
- [`pdf_classification.ipynb`](Multimodal/Document%20Prediction/pdf_classification.ipynb): PDF document classification

### Text Segmentation
- [`beginner_text.ipynb`](Multimodal/Text%20Segmentation/beginner_text.ipynb): Introduction to text segmentation
- [`ner.ipynb`](Multimodal/Text%20Segmentation/ner.ipynb): Named Entity Recognition
- [`chinese_ner.ipynb`](Multimodal/Text%20Segmentation/chinese_ner.ipynb): Chinese Named Entity Recognition
- [`multilingual_text.ipynb`](Multimodal/Text%20Segmentation/multilingual_text.ipynb): Multilingual text processing

### Semantic Matching
- [`text2text_matching.ipynb`](Multimodal/Semantic%20Matching/text2text_matching.ipynb): Text-to-text semantic matching
- [`image2image_matching.ipynb`](Multimodal/Semantic%20Matching/image2image_matching.ipynb): Image-to-image matching
- [`image_text_matching.ipynb`](Multimodal/Semantic%20Matching/image_text_matching.ipynb): Image-to-text matching
- [`text_semantic_search.ipynb`](Multimodal/Semantic%20Matching/text_semantic_search.ipynb): Semantic search for text
- [`zero_shot_img_txt_matching.ipynb`](Multimodal/Semantic%20Matching/zero_shot_img_txt_matching.ipynb): Zero-shot image-text matching

### Multimodal Prediction
- [`beginner_multimodal.ipynb`](Multimodal/Multimodal%20Prediction/beginner_multimodal.ipynb): Getting started with multimodal prediction
- [`multimodal_ner.ipynb`](Multimodal/Multimodal%20Prediction/multimodal_ner.ipynb): Multimodal Named Entity Recognition
- [`multimodal_text_tabular.ipynb`](Multimodal/Multimodal%20Prediction/multimodal_text_tabular.ipynb): Combined text and tabular prediction

### Advanced Topics
- [`continuous_training.ipynb`](Multimodal/Advance%20Topic/continuous_training.ipynb): Continuous model training strategies
- [`customization.ipynb`](Multimodal/Advance%20Topic/customization.ipynb): Model customization techniques
- [`efficient_finetuning_basic.ipynb`](Multimodal/Advance%20Topic/efficient_finetuning_basic.ipynb): Basic efficient fine-tuning methods
- [`few_shot_learning.ipynb`](Multimodal/Advance%20Topic/few_shot_learning.ipynb): Few-shot learning approaches
- [`focal_loss.ipynb`](Multimodal/Advance%20Topic/focal_loss.ipynb): Working with focal loss
- [`hyperparameter_optimization.ipynb`](Multimodal/Advance%20Topic/hyperparameter_optimization.ipynb): Hyperparameter optimization techniques
- [`model_distillation.ipynb`](Multimodal/Advance%20Topic/model_distillation.ipynb): Model distillation methods
- [`multiple_label_columns.ipynb`](Multimodal/Advance%20Topic/multiple_label_columns.ipynb): Handling multiple label columns
- [`presets.ipynb`](Multimodal/Advance%20Topic/presets.ipynb): Using model presets
- [`problem_types_and_metrics.ipynb`](Multimodal/Advance%20Topic/problem_types_and_metrics.ipynb): Understanding problem types and metrics
- [`tensorrt.ipynb`](Multimodal/Advance%20Topic/tensorrt.ipynb): TensorRT optimization