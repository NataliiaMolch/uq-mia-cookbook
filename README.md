# Cookbook on Uncertainty Quantification in Medical Image Analysis

A collection of useful references to help you start your journey with uncertainty quantification (UQ) in medical image analysis (MIA).

We are continuing the work on updating the cookbook and your pull requests will be highly appreciated !

Other cookbooks
---

- Awesome uncertainty in Deep Learning [[GitHub]](https://github.com/ENSTA-U2IS-AI/awesome-uncertainty-deeplearning)
- UQ in DL [[GitHub]](https://github.com/AlaaLab/deep-learning-uncertainty)
- ~~Mathy~~ UQ in DL [[ArXiv 2024]](https://arxiv.org/abs/2405.20550)
- Awesome Conformal Prediction [[GitHub]](https://github.com/valeman/awesome-conformal-prediction.git)

Tutorials
---

- UQ in MIA at MICCAI 2023 and 2024 [[GitHub]](https://github.com/agaldran/uqinmia-miccai-2023)
- UQ in ML for Engineering design and medical prognostics: A Tutorial [[GitHuB]](https://github.com/VNemani14/UQ_ML_Tutorial)
- Calibrated uncertainty for regression [[GutHub]](https://github.com/mlaves/well-calibrated-regression-uncertainty)

Software
---

- Uncertainty Toolbox [[GitHub]](https://github.com/uncertainty-toolbox/uncertainty-toolbox)

Reviews and surveys
---

**General scope** 

  _(Source: [Awesome uncertainty in Deep Learning](https://github.com/ENSTA-U2IS-AI/awesome-uncertainty-deeplearning))_
  
  - Gawlikowski et al. A survey of uncertainty in deep neural networks [[Artificial Intelligence Review 2023]](https://link.springer.com/article/10.1007/s10462-023-10562-9) - [[GitHub]](<https://github.com/JakobCode/UncertaintyInNeuralNetworks_Resources>)
  - Zhou et al. A survey on epistemic (model) uncertainty in supervised learning: Recent advances and applications. [[Neurocomputing 2022]](https://www.sciencedirect.com/science/article/abs/pii/S0925231221019068)
  - Abdar et al. A review of uncertainty quantification in deep learning: Techniques, applications and challenges [[Information Fusion 2021]](<https://www.sciencedirect.com/science/article/pii/S1566253521001081>)
  - Hüllermeier & Willem Waegeman. Aleatoric and epistemic uncertainty in machine learning: an introduction to concepts and methods [[Machine Learning 2021]](<https://link.springer.com/article/10.1007/s10994-021-05946-3>)

**Medical image analysis** 

  _(Source: Google Scholar)_
  
  - Huang et al. A review of uncertainty quantification in medical image analysis: Probabilistic and non-probabilistic methods. [[Medical Image Analysis 2024]](https://www.sciencedirect.com/science/article/abs/pii/S1361841524001488?via%3Dihub)
  - Lambert et al. Trustworthy clinical AI solutions: A unified review of uncertainty quantification in Deep Learning models for medical image analysis. [[Artificial Intelligence in Medicine 2024]](https://www.sciencedirect.com/science/article/pii/S0933365724000721)
  - Zou et al. A review of uncertainty estimation and its application in medical imaging. [[Meta-Radiology 2023]](https://www.sciencedirect.com/science/article/pii/S2950162823000036)
  - Kurz et al. Uncertainty Estimation in Medical Image Classification: Systematic Review. [[JMIR Medical Informatics 2022]](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9382553/)

**Healthcare** 

  _(Source: Other review papers, Google Scholar)_
  
  - Seoni et al. Application of uncertainty quantification to artificial intelligence in healthcare: A review of last decade (2013–2023). [[Computers in Biology and Medicine 2023]](https://www.sciencedirect.com/science/article/pii/S001048252300906X#bib20)
  - Loftus et al. Uncertainty-aware deep learning in healthcare: A scoping review. [[PLOS Digit Health 2022]](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9802673/)
  - Broekhuizen et al. A Review and Classification of Approaches for Dealing with Uncertainty in Multi-Criteria Decision Analysis for Healthcare Decisions. [[PharmacoEconomics 2015]](https://link.springer.com/article/10.1007/s40273-014-0251-x)

**Out-of-distribution detection**

  _(Source: Google Scholar)_

  - Hong et al. Out-of-distribution Detection in Medical Image Analysis: A survey. [[ArXiv]](https://arxiv.org/abs/2404.18279)

**Active learning**

  _(Source: Google Scholar)_
  
  - Wang et al. A comprehensive survey on deep active learning in medical image analysis. [[Medical Image Analysis 2024]](https://doi.org/10.1016/j.media.2024.103201)
  - Budd et al. A survey on active learning and human-in-the-loop deep learning for medical image analysis. [[Medical Image Analysis 2021]](https://doi.org/10.1016/j.media.2021.102062)

Applications
---

Separated by tasks (I. Classification, II. Segmentation, III. Regression) and by topics (I. Active learning, II. Domain adapration, III. ).

### I. Classification

  **General**
  
  _(Source: Google Scholar)_
  
  - Kurz et al. Uncertainty Estimation in Medical Image Classification: Systematic Review. [[JMIR Medical Informatics 2022]](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9382553/)
  
  **Out-of-distribution detection**
  
  _(Source: Google Scholar)_
  
  - Linmans et al. Predictive uncertainty estimation for out-of-distribution detection in digital pathology. [[Medical Image Analysis 2023]](https://doi.org/10.1016/j.media.2022.102655)

### II. Segmentation

  **General**
  
  _(Source: Google Scholar)_
  
  - Zepf. Aleatoric and Epistemic Uncertainty in Image Segmentation. [[PhD Thesis]](https://backend.orbit.dtu.dk/ws/portalfiles/portal/376365025/phd_thesis_kilian_zepf.pdf)
  
  **Quality Control**
  
  _(Source: Lambert et al., 2024)_
  
  - Gonzalez et al. Distance-based detection of out-of-distribution silent failures for covid-19 lung lesion segmentation. [[Medical Image Analysis 2022]](https://www.sciencedirect.com/science/article/pii/S1361841522002298)
  - Jungo et al. Analyzing the quality and challenges of uncertainty estimations for brain tumor segmentation. [[Frontiers Neuroscience 2020]](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2020.00282/full)
  - Roy et al. Bayesian quicknat: Model uncertainty in deep whole-brain segmentation for structure-wise quality control. [[NeuroImage 2019]](https://www.sciencedirect.com/science/article/pii/S1053811919302319)
  - Graham et al. Mild-net: Minimal information loss dilated network for gland instance segmentation in colon histology images. [[Medical Image Analysis 2019]](https://www.sciencedirect.com/science/article/abs/pii/S1361841518306030)
  - Wang et al. Aleatoric uncertainty estimation with test-time augmentation for medical image segmentation with convolutional neural networks. [[Neurocomputing 2019]](https://www.sciencedirect.com/science/article/pii/S0925231219301961)
  - McClure et al. Knowing What You Know in Brain Segmentation Using Bayesian Deep Neural Networks [[Frontiers Neuroinform 2019]](https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2019.00067/full)
  
  **Inter-Rater Variability Modelling** 
  
  _(Source: Lambert et al., 2024)_
  
  - Baumgartner et al. PHiSeg: Capturing Uncertainty in Medical Image Segmentation. [[MICCAI 2019]](https://link.springer.com/chapter/10.1007/978-3-030-32245-8_14)
  - Hu et al. Supervised Uncertainty Quantification for Segmentation with Multiple Annotations. [[MICCAI 2019]](https://link.springer.com/chapter/10.1007/978-3-030-32245-8_16)
  - Jungo et al. On the Effect of Inter-observer Variability for a Reliable Estimation of Uncertainty of Medical Image Segmentation. [[MICCAI 2018]](https://link.springer.com/chapter/10.1007/978-3-030-00928-1_77)
  - Kohl et al. A Probabilistic U-Net for Segmentation of Ambiguous Images [[NeurIPS 2018]](https://proceedings.neurips.cc/paper_files/paper/2018/file/473447ac58e1cd7e96172575f48dca3b-Paper.pdf)
  
  **Human evaluation**

  - Huet-Dastarac et al. Quantifying and visualising uncertainty in deep learning-based segmentation for radiation therapy treatment planning: What do radiation oncologists and therapists want? [[Radiother. Oncol.]](https://doi.org/10.1016/j.radonc.2024.110545)
  - Evans et al. The explainability paradox: Challenges for xAI in digital pathology. [[Future Genertion Computer Systems 2022]](https://doi.org/10.1016/j.future.2022.03.009)
  
  **Domain Adaptation**
  
  - Xia et al. Uncertainty-aware multi-view co-training for semi-supervised medical image segmentation and domain adaptation. [[Medical Image Analysis 2020]](https://doi.org/10.1016/j.media.2020.101766)

### III. Regression

  **Super-Resolution Reconstuction** 
  
  _(Source: Google Scholar)_
  
  - Angelopoulos et al. Image-to-Image Regression with Distribution-Free Uncertainty Quantification and Applications in Imaging. [[ICML 2022]](https://proceedings.mlr.press/v162/angelopoulos22a/angelopoulos22a.pdf)
  - Qin et al. Super-Resolved q-Space deep learning with uncertainty quantification. [[Medical Image Analysis 2021]](https://www.sciencedirect.com/science/article/abs/pii/S1361841520302498?via%3Dihub)
  - Tanno et al. Bayesian Image Quality Transfer. [[MICCAI 2016]](https://link.springer.com/chapter/10.1007/978-3-319-46723-8_31)
  


