# Atividade de programação: Análise do serviço de NLP do Google Cloud - Vertex AI

&emsp;&emsp;A Vertex AI capacita desenvolvedores de machine learning, cientistas de dados e engenheiros de dados a tirarem os projetos do papel e implantarem de maneira rápida e econômica. A Vertex é, em suma, um LLM(Large language model) que pode ser treinado de acordo com o contexto ao qual o desenvolvedor quiser utilizar. Para isso, são utlizados alguns modelos como gemini para entradas multimodais, englobando idioma, áudio e visão(resolução de problemas diversos), PaLM 2 para texto e chat, Codey para gerar, entender e realizar chat de código, Embeddings para texto e tambem multimodal, Imagen para geração de imagens e legendas/descrição e o Chirp, para tradução e entendimento por voz com isso, possibilitando algumas ferramentas:
- Model Garden: Permite personalizar modelos e adaptálos para que funcionem da melhor forma possível. Além de também ajudar desenvolvedores a terem um desenvolvimento mais rapido e facilitado. Além disso ele é integravel com os seguintes modelos:
| Modelo                      | Tarefa                                                    |
| ---------------------------- | ----------------------------------------------------------- |
| Llama 3                      | Análise e criação de texto                                  |
| Gemma                        | Geração e preenchimento de texto                           |
| CodeGemma                    | Geração e preenchimento de código                           |
| Vicuna v1.5                  | Geração de texto                                            |
| NLLB                         | Tradução em vários idiomas                                  |
| Mistral-7B                   | Geração de texto                                            |
| BioGPT                       | Geração de texto (domínio biomédico)                        |
| BiomedCLIP                   | Multimodal (domínio biomédico)                              |
| ImageBind                    | Incorporação multimodal                                     |
| DITO                         | Multimodal (detecção de objetos)                            |
| OWL-ViT v2                   | Multimodal (detecção de objetos)                            |
| FaceStylizer (Mediapipe)     | Transformação de imagens de rostos                         |
| Llama 2                      | Geração de texto                                            |
| Code Llama                   | Geração de código                                           |
| Falcon-instruct              | Geração de texto                                            |
| OpenLLaMA                    | Geração de texto                                            |
| T5-FLAN                      | Geração de texto                                            |
| BERT                         | Processamento de linguagem natural (PNL)                    |
| BART-large-cnn               | Processamento de linguagem natural (PNL)                    |
| RoBERTa-large                | Processamento de linguagem natural (PNL)                    |
| XLM-RoBERTa-large            | Processamento de linguagem natural (PNL)                    |
| Dolly-v2-7b                  | Geração de texto                                            |
| Stable Diffusion XL v1.0     | Geração de texto para imagem                                |
| Stable Diffusion v2.1        | Geração de texto para imagem                                |
| Stable Diffusion 4x Upscaler | Super-resolução de imagens                                 |
| InstructPix2Pix              | Edição de imagens                                           |
| Stable Diffusion Inpainting  | Retoque de imagens                                          |
| SAM                          | Segmentação de imagens                                      |
| Texto para vídeo (ModelScope) | Geração de texto para vídeo                                |
| Texto para vídeo zero-shot   | Geração de texto para vídeo                                |
| Pic2Word                     | Recuperação multimodal de imagens                           |
| BLIP2                        | Legendagem de imagens e resposta a perguntas visuais         |
| Open-CLIP                    | Classificação zero-shot                                    |
| F-VLM                        | Detecção de objetos                                        |
| tfhub/EfficientNetV2         | Classificação de imagens                                   |
| EfficientNetV2 (TIMM)        | Classificação de imagens                                   |
| EfficientNetV2/Reservado     | Classificação de imagens                                   |
| EfficientNetLite (MediaPipe)  | Classificação de imagens                                   |
| tfvision/vit                 | Classificação de imagens                                   |
| ViT (TIMM)                   | Classificação de imagens                                   |
| ViT/Reservado                | Classificação de imagens                                   |
| MaxViT/Reservado             | Classificação de imagens                                   |
| ViT (JAX)                    | Classificação de imagens                                   |
| tfvision/SpineNet            | Detecção de objetos                                        |
| SpineNet/Reservado           | Detecção de objetos                                        |
| tfvision/YOLO                | Detecção de objetos                                        |
| YOLO/Reservado               | Detecção de objetos                                        |
| YOLOv8 (Keras)               | Detecção de objetos                                        |
| tfvision/YOLOv7              | Detecção de objetos                                        |
| ByteTrack                    | Rastreamento de objetos de vídeo                            |
| ResNeSt (TIMM)               | Classificação de imagens                                   |
| ConvNeXt (TIMM)              | Classificação de imagens                                   |
| CspNet (TIMM)                | Classificação de imagens                                   |
| Inception (TIMM)             | Classificação de imagens                                   |
| DeepLabv3+ (com checkpoint)  | Segmentação de imagens                                      |
| Faster R-CNN (Detectron2)    | Detecção de objetos                                        |
| RetinaNet (Detectron2)       | Detecção de objetos                                        |
| Mask R-CNN (Detectron2)      | Detecção e segmentação de objetos                          |
| ControlNet                   | Geração de texto para imagem                                |
| MobileNet (TIMM)             | Classificação de imagens                                   |
| MobileNetV2 (MediaPipe)      | Classificação de imagens                                   |
| MobileNetV2 (MediaPipe)      | Detecção de objetos                                        |
| MobileNet-MultiHW-AVG        | Detecção de objetos                                        |
| DeiT                         | Classificação de imagens                                   |
| BEiT                         | Classificação de imagens                                   |
| MoViNet                      | Classificação de videoclipes e reconhecimento de ações      |
| LCM Stable Diffusion XL      | Geração de texto para imagem                                |
| LLaVA 1.5                    | Multimodal                                                  |
| PyTorch-ZipNeRF              | Reconstrução 3D                                             |
| WizardLM                     | Geração de texto                                            |
| WizardCoder                  | Geração de código                                           |
| Mixtral 8x7B                 | Geração de texto                                            |
| Llama 2 (quantizada)         | Geração de texto                                            |
| LaMa (retoque de máscaras grandes) | Retoque de imagens                                          |
| AutoGluonTabular             | Machine learning para dados tabulares                       |
