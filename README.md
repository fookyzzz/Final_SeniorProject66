# Classification of normal and abnormal symptoms of the gastrointestinal tract from wireless capsule endoscopy with convolutional neural networks and transformer.

**Abstract**

This research originates from the importance of developing models to differentiate images of the digestive system from wireless capsule endoscopy cameras. Motivated by the challenges in diagnosing gastrointestinal diseases using limited camera capabilities, this research aims to develop efficient models capable of distinguishing between normal and abnormal images of the digestive system. Both VGG16 and Vision Transformer models were trained using original and augmented datasets, with Adam and Adamax optimizers. Parameters were kept consistent for comparison. The highest accuracy was achieved by the Vision Transformer model trained with Adamax optimizer on augmented data, reaching 98.15%. Another notable metric, Sensitivity (Recall), peaked with the Vision Transformer model trained with Adam optimizer on the original dataset, scoring 99.82%. In summary, the Vision Transformer model outperformed VGG16 in image separation efficiency.

## Library Requirement

1. torch (Ver. 2.0.1)
2. torchvision (Ver. 0.15.2)
3. tqdm (Ver. 4.66.1)
4. numpy (Ver. 1.24.3)
5. matplotlib.pyplot (Ver. 3.7.1)
6. time
7. os
8. copy
9. random
10. seaborn (Ver. 0.12.2)
11. scikit-learn (Ver. 1.3.0)
12. datetime
13. pytorch_grad_cam (Ver. 1.5.0)
14. pillow (Ver. 9.4.0)
15. sys
16. torchinfo (Ver. 1.8.0)
17. prettytable (Ver. 3.5.0)

## Dataset

Gastrointestinal Tract Image From Wireless Endoscopy (WCE)

1. Download from the link below (Login via G.SWU Account)
https://drive.google.com/file/d/1-bICBpc-2Kig7HvtJJqf0iVGTmzf_5Nx/view?usp=sharing

2. Extract the dataset files and place them in the "Dataset" folder.

****
**&copy; 2024 Srinakharinwirot University**