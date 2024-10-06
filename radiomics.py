import SimpleITK as sitk
import os
from radiomics import featureextractor
from radiomics.imageoperations import resampleImage
import pandas as pd

# Define file paths
nii_file_path = 'd:/ICH_phenogrouping/ID.nii'
output_dir = 'd:/ICH_phenogrouping'
output_file_path = os.path.join(output_dir, 'ID.csv')

# Read the NII file
image = sitk.ReadImage(nii_file_path)

# Define the CT value range for the hematoma
lower_bound = 40  # Hounsfield Units
upper_bound = 80

# Create a mask to segment the hematoma region
mask = sitk.BinaryThreshold(image, lowerBound=lower_bound, upperBound=upper_bound, insideValue=1, outsideValue=0)

# Save the mask file
mask_filename = os.path.join(output_dir, 'ID_MASK.nii')
sitk.WriteImage(mask, mask_filename)

# Set up the feature extractor
feature_extractor = featureextractor.RadiomicsFeatureExtractor()
params = feature_extractor.getDefaultParams()
params['binWidth'] = 25  # Bin width for histogram computation

# Extract features from the original image
image_features = feature_extractor.execute(image, mask)

# Resample the image to 3mm isotropic voxel size
resampled_image = resampleImage(image, newSpacing=[3, 3, 3], interpolator=sitk.sitkNearestNeighbor)

# Extract features from the resampled image
resampled_image_features = feature_extractor.execute(resampled_image, mask)

# Calculate LoG features
log_image = sitk.SmoothingRecursiveGaussian(resampled_image, sigma=1)
log_image_features = feature_extractor.execute(log_image, mask)

# Calculate Wavelet features
wavelet_image = sitk.WaveletTransform(resampled_image, "sym8", 8)
wavelet_image_features = feature_extractor.execute(wavelet_image, mask)

# Combine all features
all_features = {**image_features, **resampled_image_features, **log_image_features, **wavelet_image_features}

# Select specific features
selected_features = {
    'original_shape_LeastAxisLength': all_features.get('original_shape_LeastAxisLength'),
    'original_shape_MajorAxisLength': all_features.get('original_shape_MajorAxisLength'),
    'original_shape_Maximum2DDiameterRow': all_features.get('original_shape_Maximum2DDiameterRow'),
    'original_shape_Maximum2DDiameterSlice': all_features.get('original_shape_Maximum2DDiameterSlice'),
    'original_shape_MinorAxisLength': all_features.get('original_shape_MinorAxisLength'),
    'original_firstorder_10Percentile': all_features.get('original_firstorder_10Percentile'),
    'original_firstorder_InterquartileRange': all_features.get('original_firstorder_InterquartileRange'),
    'original_firstorder_Kurtosis': all_features.get('original_firstorder_Kurtosis'),
    'original_firstorder_Maximum': all_features.get('original_firstorder_Maximum'),
    'original_firstorder_Range': all_features.get('original_firstorder_Range'),
    'original_firstorder_Skewness': all_features.get('original_firstorder_Skewness'),
    'original_firstorder_Variance': all_features.get('original_firstorder_Variance'),
    'original_glrlm_RunLengthNonUniformity': all_features.get('original_glrlm_RunLengthNonUniformity'),
    'original_glszm_GrayLevelNonUniformity': all_features.get('original_glszm_GrayLevelNonUniformity'),
    'original_glszm_HighGrayLevelZoneEmphasis': all_features.get('original_glszm_HighGrayLevelZoneEmphasis'),
    'original_glszm_SizeZoneNonUniformity': all_features.get('original_glszm_SizeZoneNonUniformity'),
    'original_glszm_SmallAreaHighGrayLevelEmphasis': all_features.get('original_glszm_SmallAreaHighGrayLevelEmphasis'),
    'original_glszm_ZoneEntropy': all_features.get('original_glszm_ZoneEntropy'),
    'original_glszm_ZoneVariance': all_features.get('original_glszm_ZoneVariance'),
    'logarithm_firstorder_90Percentile': all_features.get('logarithm_firstorder_90Percentile'),
    'logarithm_firstorder_Range': all_features.get('logarithm_firstorder_Range'),
    'logarithm_glrlm_RunLengthNonUniformity': all_features.get('logarithm_glrlm_RunLengthNonUniformity'),
    'logarithm_glszm_GrayLevelNonUniformity': all_features.get('logarithm_glszm_GrayLevelNonUniformity'),
    'logarithm_glszm_ZoneVariance': all_features.get('logarithm_glszm_ZoneVariance'),
    # Add other wavelet features...
    'wavelet.LLH_firstorder_10Percentile': all_features.get('wavelet.LLH_firstorder_10Percentile'),
    'wavelet.LLH_firstorder_90Percentile': all_features.get('wavelet.LLH_firstorder_90Percentile'),
    'wavelet.LLH_firstorder_InterquartileRange': all_features.get('wavelet.LLH_firstorder_InterquartileRange'),
    'wavelet.LLH_firstorder_Kurtosis': all_features.get('wavelet.LLH_firstorder_Kurtosis'),
    'wavelet.LLH_firstorder_Maximum': all_features.get('wavelet.LLH_firstorder_Maximum'),
    'wavelet.LLH_firstorder_Mean': all_features.get('wavelet.LLH_firstorder_Mean'),
    'wavelet.LLH_firstorder_Median': all_features.get('wavelet.LLH_firstorder_Median'),
    'wavelet.LLH_firstorder_Skewness': all_features.get('wavelet.LLH_firstorder_Skewness'),
    'wavelet.LLH_firstorder_TotalEnergy': all_features.get('wavelet.LLH_firstorder_TotalEnergy'),
    'wavelet.LLH_glcm_Autocorrelation': all_features.get('wavelet.LLH_glcm_Autocorrelation'),
    'wavelet.LLH_glcm_ClusterProminence': all_features.get('wavelet.LLH_glcm_ClusterProminence'),
    'wavelet.LLH_glcm_ClusterShade': all_features.get('wavelet.LLH_glcm_ClusterShade'),
    'wavelet.LLH_glcm_ClusterTendency': all_features.get('wavelet.LLH_glcm_ClusterTendency'),
    'wavelet.LLH_glcm_DifferenceVariance': all_features.get('wavelet.LLH_glcm_DifferenceVariance'),
    'wavelet.LLH_glrlm_HighGrayLevelRunEmphasis': all_features.get('wavelet.LLH_glrlm_HighGrayLevelRunEmphasis'),
    'wavelet.LLH_glrlm_LongRunHighGrayLevelEmphasis': all_features.get('wavelet.LLH_glrlm_LongRunHighGrayLevelEmphasis'),
    'wavelet.LLH_glszm_GrayLevelNonUniformity': all_features.get('wavelet.LLH_glszm_GrayLevelNonUniformity'),
    'wavelet.LLH_glszm_GrayLevelVariance': all_features.get('wavelet.LLH_glszm_GrayLevelVariance'),
    'wavelet.LLH_glszm_HighGrayLevelZoneEmphasis': all_features.get('wavelet.LLH_glszm_HighGrayLevelZoneEmphasis'),
    'wavelet.LLH_glszm_SmallAreaHighGrayLevelEmphasis': all_features.get('wavelet.LLH_glszm_SmallAreaHighGrayLevelEmphasis'),
    'wavelet.LLH_glszm_ZoneEntropy': all_features.get('wavelet.LLH_glszm_ZoneEntropy'),
    # ... continue adding other required features
}

# Save features to a CSV file
df = pd.DataFrame([selected_features])
df.to_csv(output_file_path, index=False)

print(f"Features have been saved to {output_file_path}")