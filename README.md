# Wasserstein Smoothing: Certified Robustness against Wasserstein Adversarial Attacks

*Code for the paper ["Wasserstein Smoothing: Certified Robustness against Wasserstein Adversarial Attacks"][paper] by Alexander Levine and Soheil Feizi. Provides a smoothing-based defense against the Wasserstein Adversarial attack proposed by [Wong et al. (2019)][paperwong] with a robustness certificate  which is nonvacuous for the Wasserstein metric compared to smoothing-based L1 certified defenses.*

[paper]: https://arxiv.org/abs/1910.10783
[paperwong]: http://arxiv.org/abs/1902.07906


## Usage Examples
+ To train a model with smoothing standard deviation of 0.01: `python3 wass_smooth_training_mnist.py --stdev 0.01`
+ To compute the accuracy of a trained smoothed model, run `python3 mnist_wass_predict.py --stdev 0.01 --model mnist_smooth_base_lr_0.001_stddev_0.01_epoch_199.pth`. This will save accuracy information in a `.pth` file in the `accuaracies` directory.
+ To compute robustness certificates for a trained smoothed model, run `python3  wass_mnist_certify.py --stdev 0.01 --model mnist_smooth_base_lr_0.001_stddev_0.01_epoch_199.pth`. This will save the certified robustness of each image in the test set in a `.pth` file in the `radii` directory.
+ To attack a smoothed classifier, run `python3 attack_mnist_smoothed.py --stdev 0.01 --checkpoint mnist_smooth_base_lr_0.001_stddev_0.01_epoch_199.pth`. This will save the empirical attack radius of each image in the test set in a `.pth` file in the `epsilons` directory.
+ Files with 'laplace' in their names use Laplace smoothing instead of the proposed  Wasserstein smoothing, but still compute certificates relative to the Wasserstein matric.
+ Files with 'cifar' in their names use CIFAR-10 rather than MNIST.



