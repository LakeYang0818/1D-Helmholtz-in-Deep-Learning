# 1D-Helmholtz-in-Deep-Learning
This project focuses on numerically solving the 1D Helmholtz equation by implementing the deep learning neural network method (a neural network in physics).

The code is written in Python and for the deep learning part Pytorch is used.

The methods for solving the one-dimensional Helmholtz equation involved in this project.
1. Analytical Method (Analytical)
2. FDM (Finite Difference Method)
3. FEM (Finite Element Method)
4. PINN (Deep learning method).

The explanatory text for each method is included in the project report.


Example Plots:
(Denote the number of segmentation by N and the value wave numbers in the Helmholtz equation by k.)

FDM:
Finite Differenc Method with N = 50, k = 32:
![FDM_N = 50_k = 32](https://user-images.githubusercontent.com/91699109/219965701-59a4a10f-9ed9-4eeb-abd4-7b22380c95c0.png)
Time Elapsed for k = 32:
![Time Elapsed k = 32](https://user-images.githubusercontent.com/91699109/219965710-3240c091-1f53-447f-ab4c-38a93776d7e2.png)
Errors of FEM with k = 32:
![FDM_k = 32](https://user-images.githubusercontent.com/91699109/219965722-1124c25a-1bc8-4cc6-b880-dfe3446e213b.png)

FEM:
Finite Element Method with N = 50, k = 32:
![FEM_N = 50_k = 32](https://user-images.githubusercontent.com/91699109/219965564-4456ae8a-691c-4ab2-b24e-43c5910579a6.png)
Time Elapsed for k = 32:
![Time Elapsed k = 32](https://user-images.githubusercontent.com/91699109/219965583-79a570bf-c8eb-41a5-9135-1c60ece4c59f.png)
Errors of FEM with k = 32:
![Errors of k = 32](https://user-images.githubusercontent.com/91699109/219965671-c9541dd0-3d5d-4bcc-b408-26ea9918d829.png)

PINN:
Deep Learning Method with k = 16, test_functions = chebyshev polynomial:
![sol - epoch20000](https://user-images.githubusercontent.com/91699109/219965776-9d48c2ac-8c91-45d2-9c09-c3f277eff85a.png)
Time Elapsed for k = 16:
![history](https://user-images.githubusercontent.com/91699109/219965800-db74ebe0-db56-42bb-bf43-74f9a4e529d9.png)
Errors of FEM with k = 16:
![errors](https://user-images.githubusercontent.com/91699109/219965781-90147efb-adb8-4dd9-abd2-3e9cc9c7a672.png)
