# ICH_phenogrouping
Phenogrouping patients with spontaneous intracerebral hemorrhage
1. It is recommended to store files in the d:/ICH_phenogrouping directory. If stored in other directories, please change the paths in the code accordingly.

2. The function of the ICH_phenogrouping.py file is to perform Principal Component Analysis (PCA) and k-means clustering analysis based on the data in the model_building.csv file, and to save the trained model as phenogrouping.joblib. (The trained model has been uploaded).

3. The sample_data.csv file contains new patients who need phenogrouping, with each file containing only one new patient by default.

4. The data structure of model_building.csv and sample_data.csv is the same, with the first row being variable names and the first column being patient IDs.

5. Users can directly use the phenogrouping.joblib model for grouping new patients, or they can add, delete, or replace training set data, but the data structure must remain unchanged.

6. We recommend that users use the free SeeIt software (www.medaifan.net) for ROI drawing and radiomics analysis. Considering that some users may not be familiar with radiomics analysis, here we provide code for drawing ROI, extracting radiomics features, and calculating hematoma volume (radiomics.py) for reference. The specific requirement is to convert the patient's non-contrast head CT DICOM files into .nii files (for example, named ID.nii) and save them in the d:/ICH_phenogrouping directory. If stored in other directories, please change the paths in the code accordingly. After running radiomics.py, an ID.csv file will be obtained. Copy the fields and data from this file to sample_data.csv, ensuring that the data structure is consistent with that of model_building.csv. Due to different versions of software packages, the names of the radiomics features obtained may vary. Users can adjust the field names in ID.csv or radiomics.py according to the field names in model_building.csv.
