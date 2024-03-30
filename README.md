# Fair Transition Loss: From label noise robustness to bias mitigation

https://doi.org/10.1016/j.knosys.2024.111711

The Machine learning widespread adoption has inadvertently led to the amplification of societal biases and discrimination, with many consequential decisions now influenced by data-driven systems. In this scenario, fair machine learning techniques has become a frontier for AI researchers and practitioners. Addressing fairness is intricate; one cannot solely rely on the data used to train models or the metrics that assess them, as this data is often the primary source of bias — akin to noisy data. This paper delves into the convergence of these two research domains, highlighting the similarities and differences between fairness and noise in machine learning. We introduce the Fair Transition Loss, a novel method for fair classification inspired by label noise robustness techniques. Traditional loss functions tend to ignore distributions of sensitive features and their impact on outcomes. Our approach uses transition matrices to adjust predicted label probabilities based on this ignored data. The empirical evaluation indicates that this method outperforms many benchmarked approaches in a variety of scenarios and remains competitive when compared with prominent fair classification strategies.


## Highlights
- Novel approach to fair classification with insights from label noise robustness.
- Multi-Objective Hyperparameter Optimization to tackle performance-fairness trade-off.
- Outperforms relevant methods in common fair classification tasks.
- Effective on unbalanced datasets.
