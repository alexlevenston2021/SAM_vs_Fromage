# SAM_vs_Fromage
Benchmark code (TensorFlow) and results for SAM vs Fromage optimizers

Sharpness Aware Minimization (SAM) is a universal optimizer which converges to smooth-loss neighborhoods of weight space. When benchmarked by its authors, SAM beat baseline models across all tested datasets. SAM works by ascending to the point of maximum loss in a neighborhood of weight space, and then descending to the point of minimum loss from this 'vantage point'. The intuition behind this is that by descending from the neighborhood maximum, the SAM optimizer will only converge to weights which lie in a neighborhood of smoothly decreasing loss. SAM is parameterized by a gradient scale coefficient, rho, which defines neighborhood size.

Fromage is a universal optimizer which computes stability-optimized weight updates. It is base-optimizer agnostic, since Fromage takes as input only the weights being optimized, the gradient, and a base learning rate.
