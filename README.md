# 🧠 Neural Networks from Scratch: Devanagari Character Recognition

> **Learn how neural networks actually work by building one completely from scratch using only NumPy!**

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://python.org)
[![NumPy](https://img.shields.io/badge/NumPy-Only-green)](https://numpy.org)
[![Pandas](https://img.shields.io/badge/Pandas-Required-orange)](https://pandas.pydata.org)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-red)](https://matplotlib.org)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Dataset: CC BY 4.0](https://img.shields.io/badge/Dataset-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Educational](https://img.shields.io/badge/Purpose-Educational-brightgreen)](https://github.com)

## 🎯 What is This Project?

This project teaches you **exactly** how neural networks work by building one from scratch to recognize handwritten Devanagari characters (the script used for Hindi, Nepali, and Sanskrit). 

**No fancy libraries** - just pure Python and NumPy mathematics!

### 🤔 What You'll Learn
- How neurons actually process information
- What forward propagation really means
- How backpropagation updates weights (the "learning" part)
- Why we use different activation functions
- How gradient descent finds the best solution
- What overfitting looks like and how to spot it

---

## 🚀 Quick Start (5 Minutes)

### Step 1: Get the Code
```bash
git clone https://github.com/yourusername/devanagariMNIST.git
cd devanagariMNIST
```

### Step 2: Install Dependencies
```bash
pip install numpy pandas matplotlib jupyter
```

### Step 3: Get the Data
Download the preprocessed dataset from Kaggle: [Devanagari MNIST Dataset](https://www.kaggle.com/datasets/prabeshsagarbaral/mnistdevanagari)

Place the CSV files in the `data/` folder:
```
data/
├── trainDataMNIST.csv
└── testDataMNIST.csv
```

### Step 4: Run the Notebook
```bash
jupyter notebook devanagarimnist.ipynb
```

**That's it!** Start from the first cell and run everything step by step.

---

## 📁 Project Structure

```
devanagariMNIST/
├── 📊 data/                          # Dataset files
│   ├── trainDataMNIST.csv           # Training data (73,600 images)
│   └── testDataMNIST.csv            # Test data (18,400 images)
├── 🔬 preProcessing/                 # Data cleaning scripts
│   └── preProcess.ipynb             # Data preparation notebook
├── 🧠 devanagarimnist.ipynb         # MAIN NOTEBOOK (start here!)
├── 📖 README.md                     # This file
```

---

## 🎓 What's Inside the Main Notebook?

### 1. **Data Loading & Exploration** 📥
- Load 92,000 images of Devanagari characters
- Understand the train/test split (why it matters)
- See what the data actually looks like

### 2. **Data Preprocessing** 🧹
- Convert text labels to numbers
- Create one-hot encoding (and learn why we need it)
- Shuffle data to prevent bias
- Separate features from targets

### 3. **Neural Network Architecture** 🏗️
- Build a 4-layer network from scratch
- Understand what each layer does
- Learn about activation functions (tanh, softmax)
- See how information flows through the network

### 4. **The Math Behind It** 🔢
- **Forward Propagation**: How the network makes predictions
- **Loss Function**: How we measure "wrongness"
- **Backpropagation**: How the network learns from mistakes
- **Gradient Descent**: How we find the best weights

### 5. **Training Process** 🏋️
- Train the network epoch by epoch
- Watch loss decrease and accuracy improve
- Understand mini-batch processing
- Monitor for overfitting

### 6. **Results Visualization** 📈
- Plot training curves
- Interpret what the graphs mean
- Understand when training is successful
- Learn warning signs of problems

---

## 🔍 Understanding the Dataset

### **What are Devanagari Characters?**
Devanagari is the script used to write Hindi, Nepali, Sanskrit, and other languages. Think of it like the "alphabet" for these languages.

Examples: क (ka), ख (kha), ग (ga), घ (gha)

### **Dataset Details**
- **46 different characters** (like having 46 letters in an alphabet)
- **92,000 total images** (32×32 pixels each, grayscale)
- **Training set**: 73,600 images (80%) - for teaching the network
- **Test set**: 18,400 images (20%) - for testing how well it learned
- **Balanced**: Each character has exactly the same number of examples

### **Why This Dataset?**
- **Perfect size**: Not too big, not too small for learning
- **Clear images**: 32×32 pixels are easy to visualize
- **Real challenge**: 46 classes is complex enough to be interesting
- **Cultural significance**: Learn about non-Latin scripts

---

## 🧠 Neural Network Explained (Really Simply)

### **What is a Neural Network?**
Imagine you're trying to recognize handwritten letters. A neural network is like having thousands of tiny decision-makers (neurons) that each look at different parts of the image and vote on what letter they think it is.

### **How Does It Work?**

#### 1. **Input Layer** (1,024 neurons)
- Each neuron looks at one pixel of the 32×32 image
- "Is this pixel dark or light?"

#### 2. **Hidden Layers** (512 → 256 → 128 neurons)
- **First hidden layer**: Detects basic shapes (lines, curves)
- **Second hidden layer**: Combines shapes into parts (tops of letters, bottoms)
- **Third hidden layer**: Recognizes whole character patterns

#### 3. **Output Layer** (46 neurons)
- Each neuron represents one Devanagari character
- The neuron with the highest value is the network's guess

### **Learning Process**
1. **Show the network an image**: "Here's a क (ka)"
2. **Network makes a guess**: "I think it's ग (ga)" (wrong!)
3. **Calculate the error**: "You were wrong by this much"
4. **Adjust the weights**: "Next time, pay more attention to these features"
5. **Repeat 57,600 times**: Network gets better and better

---

## 🎯 Key Features & Learning Objectives

### ✅ **What You'll Master**
- **Pure Implementation**: No TensorFlow, PyTorch, or Keras black boxes
- **Mathematical Understanding**: See every equation in action
- **Debugging Skills**: Understand when and why things go wrong
- **Performance Analysis**: Interpret training curves and metrics
- **Real Problem Solving**: Build something that actually works

### 🌟 **Unique Aspects**
- **Complete Transparency**: Every line of code is explained
- **Educational Focus**: Built for learning, not just results
- **Step-by-Step**: Never assumes prior knowledge
- **Real Data**: Work with actual handwriting recognition
- **Practical Skills**: Applicable to any neural network problem

---

## 📊 Expected Results

### **What Should Happen**
- **Training Accuracy**: Should reach 85-95% after enough epochs
- **Validation Accuracy**: Should be close to training accuracy (within 5-10%)
- **Loss**: Should decrease steadily during training
- **Time**: Training takes a few minutes on a modern laptop

### **What Success Looks Like**
```
Epoch 1/100 -> Train Loss: 3.8234, Val Loss: 3.7891 | Train Acc: 12.45%, Val Acc: 13.20%
Epoch 10/100 -> Train Loss: 2.1543, Val Loss: 2.2156 | Train Acc: 45.67%, Val Acc: 44.32%
Epoch 50/100 -> Train Loss: 0.8932, Val Loss: 0.9445 | Train Acc: 78.90%, Val Acc: 76.54%
Epoch 100/100 -> Train Loss: 0.4521, Val Loss: 0.5234 | Train Acc: 89.12%, Val Acc: 87.65%
```

---

## 🛠️ Troubleshooting

### **Common Issues & Solutions**

#### 🔴 "No module named 'numpy'"
```bash
pip install numpy pandas matplotlib
```

#### 🔴 "FileNotFoundError: data/trainDataMNIST.csv"
- Download the dataset from the Kaggle link above
- Make sure CSV files are in the `data/` folder

#### 🔴 "Loss is not decreasing"
- Lower the learning rate: `learningRate=0.001` instead of `0.01`
- Train for more epochs: `epochs=100` instead of `10`
- Check your data is shuffled properly

#### 🔴 "Training accuracy much higher than validation"
- This is overfitting - the network memorized training data
- Try a smaller network: `layers=[256, 128]` instead of `[512, 256, 128]`
- Add more training data or use regularization techniques

#### 🔴 "RuntimeWarning: invalid value encountered"
- This usually means gradients exploded
- Use a much smaller learning rate: `learningRate=0.0001`
- Check that your data is normalized (pixel values 0-1)

---

## 🎯 Next Steps & Extensions

### **Beginner Extensions**
1. **Try Different Architectures**: Change layer sizes `[128, 64]` vs `[512, 256, 128]`
2. **Experiment with Learning Rates**: Try `0.001`, `0.01`, `0.1` and see what happens
3. **More Epochs**: Train for 50, 100, or 200 epochs
4. **Visualize Predictions**: Show which characters the network gets wrong

### **Intermediate Extensions**
1. **Add Dropout**: Prevent overfitting by randomly turning off neurons
2. **Learning Rate Scheduling**: Start high, decrease over time
3. **Different Optimizers**: Implement momentum or Adam optimizer
4. **Data Augmentation**: Rotate, scale, or shift images for more training data

### **Advanced Extensions**
1. **Convolutional Neural Network**: Preserve spatial structure of images
2. **Attention Mechanisms**: Let the network focus on important parts
3. **Transfer Learning**: Use pretrained features
4. **Deploy as Web App**: Make it usable by others online

---

## 🤝 Contributing

Found a bug? Have a suggestion? Want to add features?

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b amazing-feature`
3. **Make your changes**: Add code, fix bugs, improve documentation
4. **Test thoroughly**: Make sure everything still works
5. **Submit a pull request**: Explain what you changed and why

### **Ideas for Contributions**
- Add more comments to complex functions
- Create additional visualizations
- Write tutorials for specific concepts
- Add support for other languages/scripts
- Optimize training speed
- Add more activation functions

---

## 📖 Dataset Citation

This project uses the **Devanagari Handwritten Character Dataset** from the UCI Machine Learning Repository:

```bibtex
@misc{prabesh_sagar_baral_shailesh_acharya_prashnna_gyawali_2025,
	title={mnistDevanagari},
	url={https://www.kaggle.com/ds/8256842},
	DOI={10.34740/KAGGLE/DS/8256842},
	publisher={Kaggle},
	author={Prabesh Sagar Baral and Shailesh Acharya and Prashnna Gyawali},
	year={2025}
}
```

**Original Dataset**: [UCI ML Repository](https://archive.ics.uci.edu/dataset/389/devanagari+handwritten+character+dataset)  
**Preprocessed Version**: [Kaggle Dataset](https://www.kaggle.com/datasets/prabeshsagarbaral/mnistdevanagari)

---

## 🙏 Acknowledgments

- **UCI Machine Learning Repository** for the original dataset
- **NumPy Community** for the amazing mathematical library
- **Jupyter Project** for the interactive notebook environment
- **Open Source Community** for making education accessible to everyone

---

## 💬 Questions & Support

### **Got Stuck?**
1. **Read the notebook carefully** - every step is explained
2. **Check the troubleshooting section** above
3. **Search existing issues** on GitHub
4. **Create a new issue** with:
   - What you were trying to do
   - What error you got
   - Your Python version and operating system

### **Want to Connect?**
- **GitHub Issues**: For bugs and feature requests

---

## 🌟 Show Your Support

If this project helped you understand neural networks, please:
- ⭐ **Star this repository**
- 🍴 **Fork it** and try your own experiments
- 📢 **Share it** with friends learning AI/ML
- 💝 **Contribute** improvements back to the community

---

**Happy Learning! 🎉**

*Remember: The goal isn't just to get good accuracy, but to understand exactly HOW and WHY neural networks work. Take your time, experiment, and don't be afraid to break things!*