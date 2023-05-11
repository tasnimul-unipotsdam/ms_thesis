# Benchmarking Bayesian deep learning models for Uncertainty quantification under the distributional shift of 3D medical images in segmentation tasks

# Motivation: 
A safe system for medical diagnosis should withhold diagnosis on cases outside
its validated expertise. On the other hand, medical image analysis must often deal
with orientation invariance (e.g., in cell images), high variance in feature scale
(in X-ray images), and locale-specific features (e.g., CT). A systematic evaluation
of Out of Distribution (OOD) methods for applications specific to medical image
domains remains absent, leaving practitioners blind as to which OOD methods
perform well and under which circumstances.
In this master's thesis, we want to fill this gap. Our goal is a comprehensive
evaluation of uncertainty estimations from different methods under dataset shift.
Effective evaluation of predictive uncertainty is most meaningful under
conditions of distributional shift. Understanding questions of risk, uncertainty,
and trust in a model’s output becomes increasingly critical as the shift from the
original training data grows larger.
