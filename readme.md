# Unexpected Benefits of Self-Modeling in Neural Systems

This project is an implementation of the concepts presented in the paper "Unexpected Benefits of Self-Modeling in Neural Systems" by Premakumar et al. (2024). The original paper explores how neural networks that learn to predict their internal states as an auxiliary task undergo fundamental changes, becoming simpler, more regularized, and more parameter-efficient.

## About the Paper

The paper "Unexpected Benefits of Self-Modeling in Neural Systems" introduces a novel concept in machine learning: when artificial networks learn to predict their internal states as an auxiliary task, they change in a fundamental way. To better perform the self-model task, the network learns to make itself simpler, more regularized, more parameter-efficient, and therefore more amenable to being predictively modeled.

The authors tested this hypothesis using a range of network architectures performing three classification tasks across two modalities: MNIST, CIFAR-10, and IMDB sentiment analysis. In all cases, adding self-modeling caused a significant reduction in network complexity, measured by the distribution of weights and the real log canonical threshold (RLCT).

## This Implementation

In this project, I've implemented the key concepts from the paper, allowing for experimentation with self-modeling neural networks across the same three tasks: MNIST, CIFAR-10, and IMDB sentiment analysis. The implementation includes:

1. Custom neural network architectures for each task, incorporating self-modeling capabilities.
2. Training and evaluation functions that account for the self-modeling auxiliary task.
3. Complexity measures as described in the paper: weight distribution analysis and RLCT estimation.
4. Hyperparameter tuning using Optuna to find optimal self-modeling configurations.
5. Visualization tools to analyze the results, including weight distribution plots and RLCT vs. self-model weight graphs.

## How to Use

1. Ensure you have all required dependencies installed. You can do this by running:
   ```
   pip install torch torchvision numpy matplotlib tensorboard optuna tqdm pandas scikit-learn
   ```

2. Run the main script:
   ```
   python self_modeling_experiment.py
   ```

3. The script will perform hyperparameter tuning, run experiments for each task, generate visualizations, and save results to JSON files.

4. Analyze the results in the generated JSON files and PNG visualizations.

## Results

After running the experiments, you'll find:
- JSON files containing detailed results for each task.
- PNG files visualizing weight distributions and RLCT vs. self-model weight relationships.
- TensorBoard logs in the `runs` directory, which you can view using TensorBoard.

## Acknowledgements

I want to express my gratitude to the authors of the original paper:

Vickram N. Premakumar, Michael Vaiana, Florin Pop, Judd Rosenblatt, Diogo Schwerz de Lucena, Kirsten Ziman, and Michael S. A. Graziano.

Their innovative work on self-modeling in neural systems has provided the foundation for this implementation.

## License

This project is released under the MIT License. See the LICENSE file for details.

## Contact

If you have any questions or feedback about this implementation, please feel free to contact me at san.hashimhama@outlook.com.