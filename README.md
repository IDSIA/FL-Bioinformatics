# FL-Bioinformatics: Code Examples for Federated Learning  

This repository contains code examples related to the review article **"Technical Insights and Legal Considerations for Advancing Federated Learning in Bioinformatics"**, currently under evaluation for publication in *OUP Bioinformatics*.  

We provide three different examples:  
- **WeaklyExpressed**: Detecting weakly expressed genes in a federated dataset.  
- **MinorAllele**: Computing minor allele frequencies in a federated dataset.  
- **SecureSum**: Implementing a custom strategy in Flower for secure summation.  

We recommend exploring the examples in the given order, as this will simplify the learning process. All examples refer the paper’s section *"How to Conduct Federation of Specific Operations in Bioinformatics."*

Flower does not provide a built-in strategy for federated secure summation (which is frequently mentioned in the article). Instead, it includes a built-in **FedAvg** strategy, as it is more commonly used. In the first two examples, although the article discusses them with a focus on summation, we use **FedAvg** to keep the implementation as simple as possible, since the operations can be performed using FedAvg too. In the third example (**SecureSum**), we show how to implement a **custom strategy** specifically for federated summation.  

The examples are implemented using the simulation mode of the Federated Learning framework Flower. For more information on Flower, we refer the reader to [Flower's webpage](https://flower.ai/docs/framework/tutorial-series-get-started-with-flower-pytorch.html).

