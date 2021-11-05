## Adversary-aware Multimodal Neural Networks for Cancer Diagnosis based on Multi-omics Data
Code and supplementary materials for our paper titled "Adversary-aware Multimodal Neural Networks for Cancer Diagnosis based on Multi-omics Data", submitted to IEEE Access journal. Codes and data will be added soon. 

#### Methods
Artificial intelligence (AI)-based systems that are increasingly deployed in numerous mission-critical situations are already outperforming medical experts (i.e., radiologists at spotting malignant tumors or diagnosis, prognosis, patient engagement). Yet, the adoption of data-driven approaches in many clinical settings has been hampered by their inability to perform reliably and safely to leverage accurate and trustworthy clinical diagnoses. More critical use of AI is aiding the treatment of cancerous conditions characterized as heterogeneous disease and many types and subtypes. Although approaches based on machine learning (ML) and deep neural networks (DNNs) were found useful in cancer diagnosis and subsequent treatment recommendations, ML models are maybe vulnerable to adversarial attacks. This vulnerability to adversarial attacks is more critical in healthcare as such an adversarially weak model can make wrong clinical recommendations, especially when AI-guided systems are used to help doctors. Therefore, a model should be not only capable of detecting adversarial or anomalous inputs but also robust to adversaries. In this paper, an attempt is taken to improve the adversarial robustness of the multimodal convolutional autoencoder (MCAE) model in the case of cancer diagnosis based on multi-omics data. To tackle the curse of dimensionality of high dimensional omics data, we employ different representational learning techniques to learn the representations. The learned representations are used to train the MCAE classifier to classify the patients into different cancer types. Both proactive and reactive measures (e.g., adversarial retraining and identification of adversarial inputs) ensure the MCAE model is robust to adversaries and behaves as intended. For this, we formulate the robustness as a property to make sure that the predictions remain stable to small variations in the input so that a small invisible noise to the supplied input would not flip the diagnosis to an entirely different cancer type. Experiment results show that our approach exhibits high confidence at predicting the cancer types correctly, giving an average precision of 96.25%. Our study suggests that a well-fitted and adversarially robust model can provide consistent and reliable cancer diagnosis. 

#### Datasets
Copy number variations, miRNA expression, and GE profiles of 9,074 samples from the Pan-Cancer Atlas project is used, covering 33 tumour types are considered. This dataset is widely used as prior knowledge to generate tumour-specific biomarkers. Data for each modality is hybridized with the Affymetrix Genome-Wide Human SNP Array 6.0, which allows us to examine most significant number of cases along with the highest probe density. 

### Availability of data
Due to size, we cannot share the data here, instead contact the author if you're interested to use it for your research. 

### Citation request
If you use the code of this repository in your research, please consider citing the folowing papers:

    @inproceedings{karim2020DeepKneeXAI,
        title={Adversary-aware Multimodal Neural Networks for Cancer Diagnosis based on Multi-omics Data},
        author={Karima, Md. Rezaul and Islam, Tanhim and Rebholz-Schuhmannd, Dietrich and Deckera, Stefan},
        journal={IEEE Access},
        publisher={IEEE (under review)},
        year={2021}
    }

### Contributing
For any questions, feel free to open an issue or contact at rezaul.karim@rwth-aachen.de
