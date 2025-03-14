# FL-Bioinformatics: Code Examples for Federated Learning  

This repository contains code examples related to the review article [**"Technical Insights and Legal Considerations for Advancing Federated Learning in Bioinformatics,"**](https://arxiv.org/abs/2503.09649) currently under evaluation for publication in *OUP Bioinformatics*.  

We provide three different examples:  
- **WeaklyExpressed**: Detecting weakly expressed genes in a federated dataset.  
- **MinorAllele**: Computing minor allele frequencies in a federated dataset.  
- **SecureSum**: Implementing a custom strategy in Flower for secure summation.  

We recommend exploring the examples in the given order, as this will simplify the learning process. All examples refer the paper’s section *"Practical insights on federation"*.

Please note that Flower does not provide a built-in strategy for federated secure summation (which is frequently mentioned in the article). Instead, it provides **FedAvg** as a built-in strategy, as it is far more commonly used (e.g., for deep learning). In the first two examples, while the article discusses them in the context of summation, we use **FedAvg** to keep the implementation as simple as possible, since the same operations can be performed using FedAvg as well. In the third example (**SecureSum**), we show how to implement a **custom strategy** specifically for federated summation.  

The examples are implemented using the simulation mode of the Federated Learning framework Flower. For more information on Flower, we refer the reader to [Flower's webpage](https://flower.ai/docs/framework/tutorial-series-get-started-with-flower-pytorch.html).



## Create the conda environment

To run the examples in this Project install all the dependencies in a conda environment by running the following commands.

```bash
conda env create -f FL_Bio.yaml
```
Next, activate the environment by running:

```bash
conda activate FL_Bio

```

