# Multiple Model Comparison Test.

This repository contains resources for finding the best model with multiple model
comparison. The algorithm is described in our [paper](https://arxiv.org/abs/1910.12252),

    Kernel Stein Tests for Multiple Model Comparison
    Jen Ning Lim, Makoto Yamada, Bernhard Schölkopf, Wittawat Jitkrittum
    NeurIPS 2019

## How to install?

Requires ```numpy```, ```autograd```, ```matplotlib``` and ```SciPy```.
The package can be installed
with `pip` command. 

    pip install -e /path/to/the/folder/of/this/repo/after/clone

Or alternatively,

    pip install git+https://github.com/jenninglim/model-comparison-test.git

Once installed, you should be able to do `import reltest` without any error.

## Demo

See ```notebooks/demo_reltest.ipynb```.

## Reproducing results

See ```reproduce-results```.

## Disclaimer

The current implementation relies on an accurate approximation of the inverse CDF
in the tail regions of the truncated normal 
(see [6](https://www.iro.umontreal.ca/~lecuyer/myftp/papers/truncated-normal-book-chapter.pdf)).
The implementation uses the inverse CDF to calculate the rejection threshold. If
the test statistic is greater than the rejection threshold, we reject the null
hypothesis.

The p-values produced from the modules may not be accurate. At the time of
writting this code there were several problems with
the  ```scipy.stats.truncnorm``` module. See 
[1](https://github.com/scipy/scipy/issues/2477), 
[2](https://github.com/scipy/scipy/issues/2477),
[3](https://github.com/scipy/scipy/issues/9902) and 
[4](https://github.com/scipy/scipy/issues/10092). This may have been fixed in
the pull request [5](https://github.com/scipy/scipy/pull/10104) in SciPy 1.4.0.


## See also

* Kernel Relative Test of Fit: [Code](https://github.com/wittawatj/kernel-mod) [Paper](https://arxiv.org/abs/1810.11630)
* Kernel Goodness of Fit: [Code](https://github.com/wittawatj/kernel-gof), [Paper](https://arxiv.org/abs/1705.07673)

