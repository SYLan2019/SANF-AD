# SANF-AD
ICME2023--A Semantic-awareness Normalizing Flow Model for Anomaly Detection. 
(Accepted).



Anomaly detection in computer vision aims to detect outliers from input image data. Examples include texture defect detection and semantic discrepancy detection. However, existing methods are limited in detecting both types of anomalies, especially for the latter. In this work, we propose a novel semantics-aware normalizing flow model to address the above challenges. First, we employ the semantic features extracted from a backbone network as the initial input of the normalizing flow model, which learns the mapping from the normal data to a normal distribution according to semantic attributes, thus enhances the discrimination of semantic anomaly detection. Second, we design a new feature fusion module in the normalizing flow model to integrate texture features and semantic features, which can substantially improve the fitting of the distribution function with input data, thus achieving improved performance for the detection of both types of anomalies. Extensive experiments on five well-known datasets for semantic anomaly detection show that the proposed method outperforms the state-of-the-art baselines.
