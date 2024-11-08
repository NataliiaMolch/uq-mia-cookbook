# MICCAI 2024 Papers on Uncertainty (Main conference)

This page provides a list of papers presented at MICCAI 2024 in Marrakesh, Morocco along with their shot summary. Contributions summary are AI-generated from the papers abstract and validated by a human. Key words used for search include "uncertainty", "uncertain", "conformal", "confidence", "calibration".

## AI generated summary

(Given the list of papers with their short description here is a summary of current trends)

These MICCAI 2024 papers present various methods for uncertainty quantification in medical imaging applications, a crucial element in improving the reliability and safety of AI-driven diagnostic tools. Here’s a breakdown of the main approaches:

1. Uncertainty-aware Learning: Approaches such as multi-view learning for prostate cancer grading, domain-generalized segmentation, and risk-controlled dose estimation in radiotherapy showcase how integrating uncertainty into model training enhances reliability, particularly in OOD (out-of-distribution) cases and subgroup-specific scenarios.

2. Conformal Prediction: Robust conformal techniques, like those applied in 3D medical images and multi-view learning for echocardiography, introduce weighted or adjusted prediction intervals to address data variability and ensure reliable prediction bounds, particularly for complex 3D structures visualized in 2D.

3. Confidence and Calibration Enhancements: Innovations in confidence estimation and network calibration tackle overconfidence in neural networks. Methods such as informed label smoothing (LS+) and region-adaptive constraints (CRaC) enhance calibration, ensuring confidence scores better reflect prediction certainty, which is critical in medical applications.

4. New methods for UQ: Sparse Bayesian networks and Laplacian Segmentation Network provide robust uncertainty estimates. Sparse Bayesian networks combine deterministic and Bayesian parameters to optimize computation without compromising predictive reliability, making them feasible for real-time applications. Laplace approximations for epistemic uncertainty quantification has been proposed for image segmentation tasks.

5. Datasets and Frameworks for Uncertainty Modeling: Contributions like the SkinCON dataset and DRAPS framework address variability in expert classifications by offering adaptive prediction sets, especially valuable for conditions like skin cancer diagnosis where labeling discrepancies are common.

## The list of papers

#uncertainty, #uncertain

**Uncertainty-Aware Multi-View Learning for Prostate Cancer Grading with DWI**
- Authors: Dong, Zhicheng; Yue, Xiaodong; Chen, Yufei; Zhou, Xujing; Liang, Jiye
- Contributions: The paper proposes a novel uncertainty-aware multi-view classification method for prostate cancer grading using mp-MRI. The method treats DWIs with different b-values as distinct views and integrates them based on their uncertainty measurements, leading to improved performance compared to existing multi-view learning methods.
- UQ: Evidential DL
- [GitHub](https://github.com/dzc2000/UMC-DWI.git)

**Uncertainty-aware meta-weighted optimization framework for domain-generalized medical image segmentation**
- Authors: Oh, Seok-Hwan; Jung, Guil; Kim, Sang-Yun; Kim, Myeong-Gee; Kim, Young-Min; Lee, Hyeon-Jik; Kwon, Hyuk-Sool; Bae, Hyeon-Min
- Contributions: The paper proposes a domain generalization method for echocardiography image segmentation using a generative model for data augmentation. The model synthesizes images of diverse cardiac anatomy and measurement conditions, and a meta-learning-based spatial weighting scheme prevents the network from training on unreliable pixels.
- UQ: Meta-learner
- [GitHub](https://github.com/Seokhwan-Oh/MLSW.git)

**Towards Integrating Epistemic Uncertainty Estimation into the Radiotherapy Workflow**
- Authors: Teichmann, Marvin Tom; Datar, Manasi; Kratzke, Lisa; Vega, Fernando; Ghesu, Florin C.
- Contributions: The paper proposes a method to improve the reliability of deep learning models for OAR contouring. It uses epistemic uncertainty estimation to identify unreliable predictions, achieving high performance in OOD detection. This work addresses the lack of ground truth for uncertainty estimation and provides a clinically relevant application.
- UQ: deep ensemples and Monte Carlo dropout
- No public code

**Subgroup-Specific Risk-Controlled Dose Estimation in Radiotherapy**
- Authors: Fischer, Paul; Willms, Hannah; Schneider, Moritz; Thorwarth, Daniela; Muehlebach, Michael; Baumgartner, Christian F.
- Contributions: The paper proposes a novel method, subgroup Risk-controlling prediction sets (SG-RCPS), for improving the accuracy and reliability of deep learning models for dose prediction in MR-Linac Radiotherapy (RT). By extending RCPS to control risk for multiple subgroups, SGRCPS provides prediction intervals with coverage guarantees, especially for critical voxels along the radiation beam.
- UQ: Risk-controlling prediction sets
- [GitHub](https://github.com/paulkogni/SG-RCPS)


**Sparse Bayesian Networks: Efficient Uncertainty Quantification in Medical Image Analysis**
- Authors: Abboud, Zeinab; Lombaert, Herve; Kadoury, Samuel
- Contributions: The paper proposes a novel approach to efficiently quantify predictive uncertainty in medical images by combining deterministic and Bayesian parameters within a neural network. By selectively assigning Bayesian status to important parameters, the method significantly reduces computational costs while maintaining high performance and reliable uncertainty estimation. This approach is demonstrated on various medical image datasets, showing significant improvements over existing methods.
- [GitHub](https://github.com/zabboud/SparseBayesianNetwork)

**SkinCON: Towards consensus for the uncertainty of skin cancer sub-typing through distribution regularized adaptive predictive sets (DRAPS)**
- Authors: Ren, Zhihang; Li, Yunqi; Li, Xinyu; Xie, Xinrong; Duhaime, Erik P.; Fang, Kathy; Chakraborty, Tapabrata; Guo, Yunhui; Yu, Stella X.; Whitney, David
- Contributions: The authors introduce a large dataset (SkinCON) containing instance-level class distributions for skin cancer images. This dataset is unique because it captures the variability in how different experts would classify a given image. By leveraging SkinCON, the authors propose a novel method (DRAPS) for estimating uncertainty in skin cancer diagnosis. 
- UQ: distribution regularized adaptive prediction sets (DRAPS)
- [Website](https://skincon.github.io/) with code, data, poster, and paper

  **Laplacian Segmentation Networks Improve Epistemic Uncertainty Quantification**
  - Authors: Kilian Zepf, Selma Wanna, Marco Miani, Juston Moore, Jes Frellsen, Søren Hauberg, Frederik Warburg, Aasa Feragen
- Contributions: This work proposes Laplacian Segmentation Networks (LSN): methods which jointly model epistemic (model) and aleatoric (data) uncertainty for OOD detection. Laplace approximation of the weight posterior is proposed for large neural networks with skip connections that have high-dimensional outputs.
- UQ: Laplace approximation
- [Github](https://github.com/kilianzepf/laplacian_segmentation.git)

#conformal

**Robust Conformal Volume Estimation in 3D Medical Images**
- Authors: Lambert, Benjamin; Forbes, Florence; Doyle, Senan; Dojat, Michel
- Contributions: The paper proposes a novel method for quantifying uncertainty in 3D medical image volumetry using conformal prediction. To address the issue of covariate shift between calibration and test data, the authors introduce a weighted conformal prediction approach that relies on density ratio estimation. To efficiently estimate the density ratio, they leverage the compressed latent representations generated by the segmentation model. The proposed method is shown to be effective in reducing coverage error in both synthetic and real-world medical image segmentation tasks.
- UQ: conformal prediction
- [GitHub](https://github.com/benolmbrt/wcp_miccai)

**Reliable Multi-View Learning with Conformal Prediction for Aortic Stenosis Classification in Echocardiography**
- Authors: Gu, Ang Nan; Tsang, Michael; Vaseli, Hooman; Tsang, Teresa; Abolmaesumi, Purang
- Contributions: The paper addresses the inherent uncertainty in ultrasound diagnosis due to limitations in visualizing 3D structures with 2D images. Existing machine learning models often lack this nuance as they rely on one-hot labels. This work proposes RT4U, a method to introduce uncertainty into training data for weakly informative ultrasound images. RT4U improves the accuracy of existing aortic stenosis classification methods and, when combined with conformal prediction, generates prediction sets with guaranteed accuracy. 
- UQ: conformal prediction
- [GitHub](https://github.com/an-michaelg/RT4U)

#confidence

**Deep Model Reference: Simple yet Effective Confidence Estimation for Image Classification**
- Authors: Zheng, Yuanhang; Qiu, Yiqiao; Che, Haoxuan; Chen, Hao; Zheng, Wei-Shi; Wang, Ruixuan
- Contributions: The paper addresses the issue of overconfidence in deep neural networks, particularly in medical image classification. It proposes a novel method called Deep Model Reference (DMR) to improve confidence estimation. DMR utilizes a group of individual models to refine the confidence of a primary decision-making model. This approach is inspired by the way doctors consult with colleagues to strengthen their diagnoses.
- [Code](https://openi.pcl.ac.cn/OpenMedIA/MICCAI2024_DMR)

**Confidence-guided Semi-supervised Learning for Generalized Lesion Localization in X-ray Images**
- Authors: Das, Abhijit; Gorade, Vandan; Kumar, Komal; Chakraborty, Snehashis; Mahapatra, Dwarikanath; Roy, Sudipta
- Contributions: The paper introduces AnoMed, a novel semi-supervised learning framework for chest X-ray disease localization. AnoMed tackles the challenges of biased attention to minor classes and pseudo-label inconsistency by incorporating two key components: a scale-invariant bottleneck (SIB) and a confidence-guided pseudo-label optimizer (PLO). SIB effectively captures multi-granular anatomical structures and underlying representations from the base features extracted by any encoder. PLO, on the other hand, refines uncertain pseudo-labels and guides them separately for the unsupervised loss function, thereby mitigating inconsistency issues. 
- [GitHub](https://github.com/aj-das-research/AnoMed)

**Confidence Matters: Enhancing Medical Image Classification Through Uncertainty-Driven Contrastive Self-Distillation**
- Authors: Sharma, Saurabh; Kumar, Atul; Chandra, Joydeep
- Contributions: The paper proposes UDCD, a novel self-distillation framework specifically designed for medical image classification with limited data. UDCD tackles the issue of knowledge transfer bias in medical images, caused by high intra-class variance and class imbalance. It achieves this by employing uncertainty-driven contrastive learning to regulate the transfer of knowledge from a teacher model to a student model. This approach ensures that only relevant and reliable knowledge gets transferred, leading to improved classification performance, especially in datasets with limited data and class imbalance.
- [GitHub](https://github.com/philsaurabh/UDCD_MICCAI)

#calibration

**LS+: Informed Label Smoothing for Improving Calibration in Medical Image Classification**
- Authors: Sambyal, Abhishek Singh; Niyaz, Usma; Shrivastava, Saksham; Krishnan, Narayanan C.; Bathula, Deepti R.
- Contributions: The paper addresses the issue of miscalibration in Deep Neural Networks (DNNs) used for medical image classification. Miscalibration means the model's confidence scores don't accurately reflect the true probability of a prediction. This is a major concern in healthcare where reliability is crucial. Existing methods like label smoothing improve calibration but rely on generic assumptions. The paper proposes Label Smoothing Plus (LS+), a novel method that uses class-specific information from the validation set to create more accurate soft targets. Experiments show LS+ significantly reduces calibration errors while maintaining performance, making DNN predictions more trustworthy for medical applications.
- [GitHub](https://github.com/abhisheksambyal/lsplus)

**Class and Region-Adaptive Constraints for Network Calibration**
- Authors: Murugesan, Balamurali; Silva-Rodriguez, Julio; Ben Ayed, Ismail; Dolz, Jose
- Contributions: The paper proposes a novel method called CRaC (Class and Region-Adaptive Constraints) for calibrating segmentation networks. CRaC addresses the challenge of segmenting objects with different categories and regions by incorporating class-wise and region-wise constraints into the training process. It also introduces a mechanism to learn the optimal weights for these constraints during training, eliminating the need for manual tuning and improving the overall segmentation performance compared to existing approaches. 
- [GitHub](https://github.com/Bala93/CRac/)

**Average Calibration Error: A Differentiable Loss for Improved Reliability in Image Segmentation**
- Authors: Barfoot, Theodore; Garcia Peraza Herrera, Luis C.; Glocker, Ben; Vercauteren, Tom
- Contributions: The paper proposes a novel method to improve calibration in medical image segmentation. Existing methods often require complex techniques, but this work introduces a new loss function (mL1-ACE) that is efficient and achieves significant calibration improvements without impacting segmentation performance. Additionally, a new visualization tool helps assess model calibration across the entire dataset. 
- [GitHub](https://github.com/cai4cai/ACE-DLIRIS)

