=============
Overview
=============

:Date: October 25, 2021

This page is an overview of the intended analysis of 'DiagnosticTool.' The goal of the analysis is to evaluate the
value of different testing tools with regards to post-maket surveillance (PMS) in a supply-chain context.
Testing tools have different values for sensitivity and specificity, as well as different costs.
The question we investigate is "Under what environments does it make sense to choose one testing tool over another?"
Environments have characteristics including budgetary limits, supply-chain structures, and the prevalence of SFPs.

'DiagnosticTool' takes arguments for system size and character to generate random systems. PMS data are generated
under each testing tool and the PMS data are used to infer the SFP rates at each node in the system. Metrics are
calculated using the inferences.

The efficacy of different testing tools is measured through four metrics, detailed as follows:

* Interval scoring: What proportion of true node SFP rates fall within the credible intervals formed under different testing tools?
* Gneiting loss: What are the interval scores, penalized for distance if the true rate is missed, for the credible intervals formed under different testing tools?
* Suspect accuracy: What are the Type I and Type II error rates for each testing tool with respect to suspecting nodes with SFP rates above the suspect threshold, t?
* Exoneration accuracy: What are the Type I and Type II error rates for each testing tool with respect to exonerating nodes with SFP rates below the exoneration threshold, u?




Input
-----
Elements pertaining to the following areas are entered into 'DiagnosticTool':

* Testing tool
   * Name
   * Sensitivity
   * Specificity
* Environment
   * Tracked or untracked data collection
   * Number of test nodes, |A|
   * Ratio of supply nodes to test nodes, \rho
   * Desired alpha level for credible intervals
   * Exoneration and suspect thesholds, u and t
   * True SFP rates at test nodes and supply nodes, \eta and \theta
   * Pareto scale parameter characterizing the sourcing probability matrix, \lambda
   * Proportion of non-zero elements of the sourcing probability matrix, \zeta
   * Prior mean and variance, \gamma and \nu
   * Number of samples per iteration, n
   * Number of inter-calculation samples, m
   * Number of system iterations, N


Output
---------

* [hold]
* [hold]

