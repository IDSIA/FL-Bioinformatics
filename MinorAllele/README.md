# WeaklyExpressed: A Flower for weakly expressed genes detection.

## Introduction

This example shows how Flower can be used to detect minor allele frequency (MAF) (i.e., given a single nucleotide polymorphism (SNP), the frequence of the less represented allele in a sample) in a federated dataset. The good practice when quality control over SNPs is performed is to exclude SNPs with a MAF $< 0.05$, we show that on a client, that was included in the federated analysis, would have been excluded if only the local dataset was analyzed.

We recommend that users first run the simulation using the federated process. Once the federated process is complete, the results can be analyzed using the Jupyter notebook provided in this folder. The notebook compares the federated analysis with a centralized analysis, demonstrating that the results are identical. Additionally, it compares the federated results with local analyses, showing what each client would obtain if they were only able to analyze their dataset independently, without authorization to share data in a centralized or federated process.

This analysis simulates federation using Flower's simulation engine. It is based on Flower’s **FedAvg** aggregation strategy, with secure aggregation performed using **SecAgg+**. For more information on Flower, we refer the reader to [Flower's webpage](https://flower.ai/docs/framework/tutorial-series-get-started-with-flower-pytorch.html).

## If not installed, install Flower as well as dependencies

To install Flower, run the following command:

```bash
pip install flwr
```
Next, install dependencies: 

```bash
pip install -e .
```

## Run the federated simulation with the Simulation Engine

In the `MinorAllele` directory, use `flwr run` to run a local simulation of the federated process:

```bash
flwr run .
```

