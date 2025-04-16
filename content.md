# Speaker Notes for Slides 11–19 (Expanded with All Technical Terms Explained)

---

# Slide 9: Approach Overview

- **Training Pipeline Diagram (Figure 2):**  
  - *Comprehensive Explanation*: The diagram shows the complete SelfHAR pipeline, a novel four-stage framework that synergistically combines supervised learning, self-supervised learning, and knowledge distillation (knowledge distillation: a process where a simpler model—the student—learns to mimic a more complex model—the teacher—in a transfer learning scenario).
  
  - *Pipeline Components in Detail*:
    1. **Left Section**: Shows the foundation of supervised learning with the limited labeled dataset (e.g., HHAR with accelerometer data) divided into training and test partitions.
    2. **Top-Center**: Depicts the large unlabeled dataset containing raw sensor measurements from real-world environments.
    3. **Center Flow**: Illustrates the knowledge transfer process where information flows from labeled to unlabeled data.
    4. **Lower-Right**: Shows the transformation process that increases data diversity through systematic augmentation.
    
  - *Why This Architecture?*: This specific pipeline architecture addresses the fundamental challenge of HAR: limited labeled data versus abundant unlabeled data. The architecture was designed after observing that previous methods like those in Saeed et al. (2019) yielded limited improvement when using external unlabeled datasets.

- **Key Components:**  
  - **Teacher Model:** 
    - *Definition*: A neural network trained exclusively on the labeled dataset D using knowledge distillation principles (as detailed in section 3.2.2 of the paper).
    - *Implementation*: Uses temporal convolutional layers to process time-series data, followed by a HAR classification head with softmax activation.
    - *Purpose*: To generate high-quality pseudo-labels for unlabeled data, effectively transferring knowledge from labeled to unlabeled datasets.
    - *Example*: If we have 500 labeled walking samples, the teacher learns the core patterns of walking from these samples to later identify similar patterns in unlabeled data.

  - **Self-Labeling:** 
    - *Definition*: The process where labeled dataset D is mixed with unlabeled data U to form dataset W, after which the teacher model assigns pseudo-labels that are filtered via confidence thresholding.
    - *Technical Implementation*: Uses softmax confidence scores to rank samples and keeps only those exceeding threshold τ (typically 0.5), selecting the top K samples per class.
    - *Purpose*: To expand the effective training set while minimizing noise from incorrect labels.
    - *Example*: If the teacher predicts "walking" with 95% confidence for an unlabeled sample, this sample is retained; if predicted with only 40% confidence, it's discarded.

  - **Signal Transformation:** 
    - *Definition*: A data augmentation strategy where each sample is systematically modified with eight distinct transformations to mimic real-world signal variations.
    - *Transformations Used*: Random noise injection, scalar scaling, 3D rotation, signal inversion, time reversal, segment scrambling, time-series stretching/warping, and channel shuffling (described in section 3.2.4 of the paper).
    - *Purpose*: To increase internal data diversity and teach the model invariance to common signal distortions encountered in mobile/wearable sensing.
    - *Example*: Rotating the signal simulates different device orientations (e.g., phone in pocket vs. hand), teaching the model position invariance.

  - **Student Model:** 
    - *Definition*: A multi-task neural network that learns from both the augmented self-labeled dataset D′ (with transformation and activity labels) and original dataset D.
    - *Training Process*: First pre-trained on D′ using a combined loss function (HAR classification + transformation discrimination + regularization), then fine-tuned on D with early convolutional layers frozen.
    - *Purpose*: To leverage both self-supervised learning and supervised fine-tuning for optimal performance.
    - *Example*: The student learns to recognize both a "walking" activity and whether a signal has been time-reversed, developing more robust internal representations.

- **Information Flow and Integration:**
  - *System Synergy*: The four components form an integrated pipeline where knowledge flows from labeled to unlabeled data and from single-task to multi-task learning.
  - *Data Amplification*: The pipeline effectively expands the training data from the original limited labeled set to a diverse, augmented set with both internal diversity (transformations) and external diversity (different devices/users/environments).


- **Image Content - SelfHAR Pipeline Diagram**
  - *Overview*: The diagram illustrates the complete SelfHAR pipeline with its four main components: (1) Teacher Model Training, (2) Self-Labeling, (3) Signal Transformation, and (4) Student Model Training.
  
  - *Component 1 - Teacher Model Training*:
    - The left section shows a labeled dataset (e.g., HHAR) with raw sensor measurements (such as accelerometer data) and activity labels (HAR).
    - The dataset is split into training and test sets (visualized by blue and yellow blocks).
    - The teacher model (shown as a neural network structure) is trained using supervised learning on this data.
    - *Technical detail*: The model architecture uses a CNN core (Convolutional Neural Network, which is a deep learning architecture specialized in extracting spatial features from input data) with a HAR classification head (a fully connected layer with softmax activation that outputs probability distributions over activity classes).
    - *Why CNNs?*: CNNs are used because of their ability to extract hierarchical patterns from time series data through convolution operations, effectively capturing temporal dependencies in accelerometer signals.

  - *Component 2 - Self-Labeling*:
    - The center-top shows a large unlabeled dataset containing raw sensor measurements without activity labels.
    - The trained teacher model processes this unlabeled data to generate activity predictions.
    - A critical filtering process occurs where predictions are ranked by confidence (the probability assigned to the predicted class by the softmax function).
    - *Technical detail*: The filtering operation uses a threshold $\tau$ to select only high-confidence predictions, creating what's called a "self-labeled HAR dataset" (represented by the purple blocks).
    - *Why filtering?*: This confidence-based selection is essential because it removes potentially noisy or incorrect pseudo-labels, which could otherwise introduce error propagation in the student model training.

  - *Component 3 - Signal Transformation*:
    - The lower-right section shows how the self-labeled data undergoes various signal transformations (such as noise injection, negation, and flipping).
    - *Technical detail*: These transformations generate multi-task training samples, where each sample is associated with both activity labels and transformation labels.
    - The output is a multi-task self-labeled dataset with three components: augmented data, transformation labels, and HAR labels.
    - *Why transformations?*: Signal transformations increase the diversity of training data, teaching the model invariance to common signal variations that occur in real-world settings, such as device orientation changes and sensor noise.

  - *Component 4 - Student Model Training*:
    - The bottom-left shows the pre-trained student model with a HAR head.
    - The student model is first trained using the multi-task dataset to learn both activity recognition and transformation detection.
    - *Technical detail*: After pre-training, the CNN core is partially frozen (meaning early convolutional layers' weights are not updated) and only the HAR classification head is fine-tuned using the original labeled dataset.
    - The final evaluation occurs on the test portion of the labeled dataset.
    - *Why partial freezing?*: This transfer learning technique preserves general feature extraction capabilities learned from the diverse augmented data while allowing the final layers to specialize for the specific HAR task.

  - *Data Flow*:
    - The arrows illustrate how information flows through the pipeline, starting with separate labeled and unlabeled datasets.
    - The process culminates in a sophisticated student model that has learned from both sources of data in complementary ways.
    - *Evaluation Flow*: The final arrow leading to "Evaluation Score on Labeled Dataset" shows how the model's performance is assessed on the held-out test data.
    - *Why this entire pipeline?*: This comprehensive approach allows the model to gain the benefits of both supervised learning (accurate ground truth labels) and self-supervised learning (data diversity and transformation invariance), resulting in superior performance without increasing model complexity at inference time.


---

# Slide 10: Teacher Model Training

## Knowledge Distillation in SelfHAR: The Inverted Paradigm

The knowledge distillation paradigm mentioned in your notes represents a key technique in machine learning where a simpler model (student) learns to replicate the behavior of a more complex model (teacher). Let me explain how SelfHAR uses this concept differently from traditional approaches:

### Traditional vs. SelfHAR Knowledge Distillation

#### Traditional Knowledge Distillation
In traditional knowledge distillation (introduced by Hinton et al. in 2015):
- A complex, large **teacher** model is trained first on a dataset
- The **student** model (smaller/simpler) then learns from the teacher's outputs
- The goal is model compression while maintaining performance
- Both models typically work with the same quality of data

#### SelfHAR's Inverted Approach
SelfHAR inverts this typical usage pattern:
1. The **teacher** model is initially trained on a **small but high-quality labeled dataset**
2. This teacher then generates pseudo-labels for a **much larger unlabeled dataset** 
3. The **student** model learns from both:
   - The original high-quality labeled data
   - The larger teacher-labeled dataset (which may be noisier)

#### SelfHAR's Differentiating Features

From the paper, SelfHAR distinguishes itself through:

1. **Combined Learning Paradigms**: SelfHAR integrates teacher-student self-training with multi-task self-supervision (signal transformation discrimination).

2. **Data Efficiency**: The paper demonstrates that SelfHAR can achieve similar performance using up to 10 times less labeled data compared to fully supervised approaches.

3. **Unlabeled Data Leverage**: Unlike previous approaches like the one by Saeed et al. that showed limited benefit when using additional unlabeled datasets for pre-training, SelfHAR is specifically designed to effectively incorporate unlabeled datasets.

4. **Implementation Details**:
   - A teacher model distills knowledge from labeled accelerometer data
   - Only high-confidence predictions are selected from the unlabeled dataset
   - The student model is trained on two tasks simultaneously: discriminating signal transformations and recognizing activities
   - Final fine-tuning uses ground truth labels from the training set

This approach allows SelfHAR to outperform both supervised and previous semi-supervised methods by up to 12% increase in F1 score, while maintaining the same number of model parameters at inference time.  


  - *Implementation in SelfHAR*: 
    - Phase 1: Train a robust teacher model on the manually labeled dataset D.
    - Phase 2: Use this teacher to generate pseudo-labels for unlabeled data.
  
  - *Technical Rationale*: This approach allows us to transfer the discriminative knowledge learned from clean, labeled data to the vast unlabeled dataset. The teacher acts as a filter, identifying patterns similar to those in the labeled data.
  
  - *Example*: For instance, if the labeled data contains clear examples of "walking" with precise vertical acceleration patterns, the teacher learns to recognize these patterns and can identify similar motion signatures in unlabeled data, even if they contain slight variations or noise.

- **Loss Function – Cross-Entropy:**  
  - *Formula Breakdown*:  
    $$L_{classification} = -\frac{1}{|D|}\sum_{d\in D}\sum_{a\in A}y_{d,a}\log\left({M_\theta(d)}_a\right)$$  

    Where:
    - $D$ represents the labeled dataset (all training samples)
    - $|D|$ is the number of samples in the dataset (normalizing factor)
    - $d$ represents an individual data sample (window of sensor readings)
    - $A$ is the set of all possible activity classes (e.g., "walking," "running")
    - $y_{d,a}$ is a binary indicator: 1 if sample $d$ has true class $a$, 0 otherwise (one-hot encoding)
    - $M_\theta(d)$ represents the model's output (probability distribution over classes)
    - ${M_\theta(d)}_a$ is the probability assigned to class $a$ for sample $d$
    - $\log$ is the natural logarithm function
  
  - *Why Cross-Entropy?*: 
    - Information Theory Basis: Cross-entropy quantifies the difference between the predicted probability distribution and the true distribution, effectively measuring how "surprised" the model is by the true labels.
    - Optimization Properties: Its gradient is well-behaved for gradient descent optimization, especially with softmax outputs (smooth and convex).
    - Multi-class Capability: Naturally extends to multi-class classification problems common in HAR.
    - Signal Fidelity: Particularly sensitive to misclassifications with high confidence, which is critical for the teacher model's reliability.

  - *Mathematical Intuition*:
    - For correct classifications, ${M_\theta(d)}_a$ should approach 1, making $\log({M_\theta(d)}_a)$ approach 0 and minimizing the loss.
    - For incorrect classifications, ${M_\theta(d)}_a$ should be 0, but $\log(0)$ is undefined, which severely penalizes wrong predictions (in practice, small epsilon values prevent numerical issues).

- **Objective:**  
  - *Primary Goal*: To train a teacher model that generates reliable and high-confidence activity labels for the self-labeling phase, effectively transferring knowledge from labeled to unlabeled data.
  
  - *Success Criteria*:
    - High accuracy on labeled validation data
    - Generalizable feature extraction (avoids overfitting to specific users or sensors)
    - Well-calibrated confidence scores (critical for the subsequent filtering step)
  
  - *Theoretical Foundation*: The teacher model creates a form of "weak supervision" for the unlabeled data, converting an unsupervised problem into a supervised one—a principle proven effective in both computer vision (e.g., Yalniz et al., 2019) and now adapted for time-series HAR data.
  
  - *Implementation Details*:
    - Architecture: TPN (Transformation Prediction Network) with three temporal convolutional layers
    - Regularization: L2 weight decay (0.0001) and dropout (0.1) to improve generalization
    - Training Protocol: Adam optimizer with learning rate 0.0003, early stopping on validation loss

## Slide 11: Self-Labeling and Sample Selection

- **Mixed Dataset Generation**
  - *Explanation*: We combine the labeled dataset $D$ (a collection of sensor samples manually annotated with ground-truth activity labels—as detailed in our paper "SelfHAR: Improving Human Activity Recognition through Self-training with Unlabeled Data") with the unlabeled dataset $U$ (sensor data collected from wearable/mobile devices without annotations, representing real-world, unconstrained inputs) to form a new dataset $W$. In $W$, we remove explicit labels so that the teacher model can later assign pseudo-labels.
  - *Why?*: Leveraging a large pool of unlabeled data helps overcome the cost and difficulty of manual annotation, thereby increasing data diversity. This improves the model’s generalization—the ability to perform well on unseen data—as explained in our paper. (Function Note: Merging $D$ and $U$ amplifies the effective dataset size, a critical need in mobile sensing where labeled data is scarce.)
  - *Example*: Suppose you have 1,000 high-quality lab samples but 100,000 unlabeled recordings from daily usage; combining them allows the teacher model to generate reliable pseudo-labels from the abundant data.

- **Teacher Model Predictions**
  - *Definition*: The “teacher model” is a neural network trained on the high-quality labeled dataset $D$ using supervised learning (optimizing with cross-entropy loss, which quantifies the difference between predicted and true probability distributions). This aligns with the knowledge distillation approach described in our paper.
  - *Process*: The teacher processes each sample in $W$, applying the softmax function (which converts raw output scores, or logits, into a probability distribution that sums to one—vital for multi-class classification) to assign a probability for each activity class along with a confidence score.
  - *Why?*: Using softmax enables clear probabilistic predictions, so the teacher model’s high-confidence outputs can be trusted as pseudo-labels. This step is critical because it extends the small labeled set to the larger dataset, providing a basis for training the student model.

- **Confidence Filtering**
  - *Definition*: “Softmax confidence” refers to the maximum probability output by the softmax function for a given sample—indicating how sure the model is about its prediction.
  - *Process*: For each activity class:
    - Sort samples by their softmax confidence.
    - Apply a threshold $\tau$ (e.g., 0.5, a typical cut-off to ensure only predictions with over 50% confidence are retained) to discard uncertain predictions.
    - Select the top $K$ samples per class that meet or exceed this threshold.
  - *Why?*: This filtering reduces the risk of incorporating noisy or incorrect pseudo-labels into the training set, ensuring that only reliable data influences the subsequent learning phase. (Function Note: Thresholding using $\tau$ is a simple yet effective technique to minimize label noise.)
  - *Example*: A sample predicted as "walking" with 95% confidence is selected, whereas one predicted with only 40% confidence is discarded.

- **Purpose and Model Roles**
  - *Summary*: 
    - The **Teacher Model** is trained solely on the original labeled dataset $D$ (as described in section 3.2.2 of our paper) to learn accurate activity recognition using functions like softmax and cross-entropy loss.
    - The **Student Model** is later trained on the self-labeled (and augmented) dataset derived from $W$, then fine-tuned on $D$. This teacher-student framework (a core semi-supervised technique) allows us to bridge the gap between limited labeled data and abundant sensor data.
  - *Explanation*: By generating pseudo-labels with high confidence (using functions such as softmax for normalization and thresholding for noise reduction), our approach, as detailed in the paper, improves model performance without increasing model complexity. Each function is selected to ensure robustness—softmax for probability distribution, cross-entropy for accurate supervision, and thresholding to control noise propagation.

---

## Slide 12: Data Augmentation & Transformation Labeling

- **Signal Transformations Applied**  
  - **Data augmentation** (*data augmentation: the process of artificially increasing the diversity of the training data by applying various transformations to existing samples*) is used to make the model robust to real-world variations.
  - Each selected sample is augmented into 9 versions (the original plus one per transformation) using:
    1. **Random Noise Injection** (*adds small random values to the signal, simulating sensor noise that occurs in real devices*)
    2. **Random Scalar Scaling** (*multiplies the signal by a random factor, simulating changes in sensor sensitivity or calibration*)
    3. **Random 3D Rotation** (*rotates the signal in three-dimensional space, simulating different device orientations, e.g., phone in pocket vs. hand*)
    4. **Signal Inversion** (*flips the signal, simulating the device being upside down or reversed*)
    5. **Time Reversal** (*reverses the order of the signal in time, testing the model's temporal invariance*)
    6. **Random Scrambling of Signal Sections** (*shuffles parts of the signal, simulating interruptions or artifacts in sensor data*)
    7. **Time-Series Stretching and Warping** (*alters the speed or timing of the signal, simulating faster or slower movements*)
    8. **Channel Shuffling** (*changes the order of sensor axes, simulating sensor misalignment or hardware differences*)
  - **Why?** These augmentations mimic real-world conditions and device variability, helping the model learn features that are invariant (*invariant: unaffected by certain transformations or noise*) to such changes.

- **Transformation Labels**  
  - Each augmented sample is paired with 8 binary labels (*binary label: a label that is either 0 or 1, indicating the presence or absence of a transformation*) indicating which transformation was applied.
  - **Why?** This enables the model to learn not only to recognize activities but also to detect transformations, supporting **multi-task learning** (*multi-task learning: training a model to solve multiple related tasks simultaneously, which can improve feature learning and generalization*).

- **Purpose**  
  - The self-supervised dataset $D'$ (*self-supervised dataset: a dataset where labels are generated from the data itself, not requiring manual annotation*) enhances internal diversity (*internal diversity: variation within the dataset due to transformations*) and helps the student model learn robust, invariant representations.

---

## Slide 13: Student Model Training & Fine-tuning

- **Multi-task Learning Setup**  
  - The student model predicts 9 tasks: 8 binary transformation tasks + 1 multi-class activity recognition task (*multi-class classification: predicting one label out of many possible classes*).
  - **Why?** Multi-task learning encourages the model to learn shared representations that are useful for both activity recognition and transformation discrimination, improving generalization and robustness.

- **Supervised Pre-training**  
  - The student model is trained on $D'$ using a **combined loss** (*combined loss: a sum of loss functions for each task, e.g., activity classification loss + transformation discrimination loss + regularization*).
    - **HAR classification loss** (*cross-entropy loss: measures the difference between the predicted probability distribution and the true distribution for activity recognition*)
    - **Transformation discrimination loss** (*binary cross-entropy loss: measures the error in predicting whether a transformation was applied*)
    - **L2 regularization** (*L2 regularization: adds a penalty proportional to the square of the model weights, discouraging large weights and reducing overfitting*)
  - **Formula**:  
    $L_{total} = L^{HAR}_{classification} + L^{TD}_{classification} + \beta \|\theta'\|_2$  
    (*$\beta$: regularization coefficient, $\theta'$: model parameters*)

- **Fine-tuning Phase**  
  - After pre-training, the early convolutional layers (*convolutional layers: neural network layers that apply filters to extract local patterns from time-series data*) are **frozen** (*frozen: weights are not updated during training*), and only the activity recognition branch is fine-tuned on the original labeled dataset $D$.
  - **Why?** Freezing preserves the robust, general features learned during pre-training, while fine-tuning allows the model to specialize for the target HAR task.

- **Goal**  
  - To align the learned representations with the specific activity recognition task, while retaining robustness to signal variations and noise.

---

## Slide 14: Pipeline Configurations Comparison

- **Component Breakdown**  
  - The SelfHAR pipeline consists of 4 main components:
    1. **Teacher Model Training** (*training a model on labeled data to serve as a source of pseudo-labels*)
    2. **Self-Labeling** (*using the teacher model to generate labels for unlabeled data*)
    3. **Signal Transformation** (*applying data augmentation to increase diversity and robustness*)
    4. **Student Model Training** (*training a new model on the expanded, augmented dataset*)
  - **Why?** Each component addresses a specific challenge: limited labeled data, leveraging unlabeled data, handling real-world variability, and learning robust representations.

- **Training Configurations Explored**  
  - **Fully Supervised**: Only Teacher Model Training (Component 1); serves as a baseline.
  - **Transformation Discrimination Training**: Components 0 (pre-training teacher with transformation discrimination) + 1; tests the effect of self-supervised pre-training.
  - **Self-training**: Components 1, 2, and 4; teacher-student setup without augmentation.
  - **Transformation Knowledge Distillation**: Components 0, 1, 2, and 4; combines self-supervised pre-training and self-training.
  - **SelfHAR (Proposed)**: Components 1, 2, 3, and 4; integrates all steps for maximum benefit.
  - **Why?** Comparing these configurations helps isolate the contribution of each component to overall performance.

- **Note**  
  - All configurations use the same neural network architecture (*architecture: the structure and connectivity of layers in a neural network*) and number of parameters; only the training strategy differs.

---

## Slide 15: Datasets Overview

- **Dataset Table Explanation**  
  - **Users** (*number of participants: more users means more diversity and better generalization*)
  - **Activity Classes** (*number of distinct activities: more classes make the recognition task harder*)
  - **Samples** (*number of data windows: more samples help model training and reduce overfitting*)
  - **Device Placement** (*where the sensor was worn: affects signal characteristics and model robustness*)
  - **Notes** (*special characteristics, e.g., small dataset, complex activities*)
  - **Why so many datasets?** To demonstrate that SelfHAR generalizes across a wide range of real-world scenarios, devices, and activity types.

---

## Slide 16: Data Preparation

- **Pre-processing Steps**  
  - **Z-normalization**: 
    - *Technical Explanation*: A mathematical standardization technique that transforms each sensor channel's data to have zero mean and unit variance by subtracting the mean and dividing by the standard deviation of the training set.
    - *Formula*: $z = \frac{x - \mu}{\sigma}$ where $\mu$ is the channel's mean and $\sigma$ is its standard deviation.
    - *Why Used*: Essential for HAR because different sensors and devices produce signals with varying scales and baselines. As described in Section 4.2 of our paper, normalization ensures fair comparison between data from heterogeneous devices (particularly important for datasets like HHAR which were collected from various smartphones with different sensor characteristics). Without normalization, models would learn magnitude differences between devices rather than activity patterns.
    - *Benefit*: Stabilizes gradient descent during training and prevents features with larger scales from dominating the learning process.

  - **Segmentation**: 
    - *Technical Explanation*: The process of dividing continuous sensor streams into fixed-length windows of 400 timestamps × 3 channels (x, y, z accelerometer axes).
    - *Why Used*: HAR requires analyzing patterns over time, but neural networks need fixed-size inputs. As noted in our experimental protocol, this window size (approximately 3-8 seconds depending on sampling frequency) captures complete motion cycles for most activities while maintaining computational efficiency. The 3 channels represent the triaxial nature of accelerometer data, preserving the spatial relationships between movement axes.
    - *Implementation Detail*: Each window becomes a single training example (sample) for the neural network, transforming a continuous time-series problem into a classification task.

  - **Overlap**: 
    - *Technical Explanation*: Creating windows where each shares 50% of its timestamps with the previous window, rather than using discrete, non-overlapping segments.
    - *Why Used*: Since activities transition smoothly in real-world scenarios, overlapping windows ensure these transitions are captured in the training data. As mentioned in Section 4.2, this technique effectively doubles the number of training samples without collecting additional data, addressing the scarcity of labeled examples in HAR—a key challenge identified in our introduction.
    - *Benefit*: Improves model robustness to phase shifts (same activity starting at different points in the window), which is critical for deployment in unconstrained environments.

  - **Test Partition**: 
    - *Technical Explanation*: Reserving data from 20–25% of participants as a completely unseen test set, rather than randomly splitting data points.
    - *Why Used*: This user-based partitioning strategy, described in Section 4.3.1, provides a more realistic evaluation of how models will perform on new users. As emphasized in our paper, HAR models must generalize across different individuals with unique movement patterns, body types, and sensor placements.
    - *Scientific Validity*: This approach prevents data leakage between training and testing, which would occur if windows from the same user appeared in both sets, artificially inflating performance metrics.

  - **No Resampling**: 
    - *Technical Explanation*: Maintaining the original sampling frequencies of each dataset (ranging from 20Hz to 200Hz depending on device) rather than standardizing to a common frequency.
    - *Why Used*: As noted in Section 4.2 of our paper, resampling can introduce artifacts and distortions in the signal. By keeping original frequencies, we ensure the neural network learns to handle the actual data characteristics it will encounter in real-world deployment.
    - *Real-world Relevance*: This decision makes our evaluation more authentic to deployment scenarios, where models must work with data as it comes from various devices without preprocessing that may not be feasible in resource-constrained environments.

---

## Slide 17: Experimental Setup – Standard and Linear Evaluation

- **TPN Architecture Details**  
  - **TPN (Transformation Prediction Network)**: 
    - *Detailed Definition*: A specialized neural network architecture designed specifically for HAR with self-supervised learning that uses temporal convolutions to extract meaningful patterns from time-series data.
    - *Why Used*: As noted in Section 4.3.1 of our paper, we adopted this architecture to enable direct comparisons with prior work by Saeed et al. It's specifically optimized for capturing temporal dependencies in accelerometer signals without requiring manual feature engineering.
    - *Benefit to Pipeline*: Provides a consistent architectural backbone across all evaluated configurations, ensuring fair comparison of training strategies rather than model capacity differences.

  - **Convolutional Layers**: 
    - *Detailed Definition*: Specialized neural network layers that apply sliding filters across the input signal to detect local patterns regardless of their position in the time series.
    - *Configuration Details*: Three layers with 32, 64, and 96 filters respectively, and decreasing kernel sizes (24, 16, 8). This progressive structure creates a feature hierarchy where:
      - First layer (32 filters): Captures basic motion primitives (e.g., peaks, valleys)
      - Second layer (64 filters): Detects combinations of primitives (e.g., repetitive patterns)
      - Third layer (96 filters): Identifies higher-level activity signatures
    - *Why These Values*: These specific parameters were carefully selected through extensive experimentation as reported in Section 4.3.1. The increasing filter count (32→64→96) allows progressive abstraction of features, while decreasing kernel sizes (24→16→8) focus on increasingly refined temporal patterns.
    - *Stride Setting*: Fixed at 1 for all layers, meaning the filter moves one time step at a time, ensuring no temporal information is missed—critical for detecting subtle differences between similar activities (e.g., walking vs. walking upstairs).

  - **ReLU Activation**: 
    - *Detailed Definition*: Rectified Linear Unit, a non-linear function defined as f(x) = max(0, x), which outputs the input directly if positive and zero otherwise.
    - *Why Used*: ReLU introduces crucial non-linearity without the vanishing gradient problems of earlier activation functions (like sigmoid or tanh). For sensor data with both positive and negative acceleration values, ReLU helps the network focus on the magnitude and direction of movements separately.
    - *Pipeline Benefit*: Enables deeper networks to learn complex activity patterns while maintaining efficient training through simple gradient computation.

  - **L2 Regularization (0.0001)**: 
    - *Detailed Definition*: A penalty term added to the loss function that is proportional to the square of the model's weight values, forcing the model to prefer smaller weights.
    - *Why This Value*: The coefficient 0.0001 was selected through validation experiments—strong enough to prevent overfitting to user-specific patterns in small datasets but not so strong as to prevent learning meaningful features. As mentioned in Section 4.3.1, this specific value balances generalization across different users with model expressiveness.
    - *Pipeline Benefit*: Critical for our teacher-student framework where the teacher must generalize well to provide reliable pseudo-labels for the unlabeled dataset.

  - **Dropout (0.1)**: 
    - *Detailed Definition*: A regularization technique that randomly sets 10% of activations to zero during each training step, forcing the network to learn redundant representations.
    - *Why This Value*: The relatively low rate of 0.1 (10%) was chosen because time-series data for HAR contains important sequential information that shouldn't be disrupted too aggressively. This value was determined experimentally to provide regularization without destroying crucial temporal patterns.
    - *Pipeline Benefit*: Particularly important in our semi-supervised approach since both the teacher and student models need to be robust against variations in sensor data that weren't seen in the limited labeled dataset.

  - **Global 1D Max Pooling**: 
    - *Detailed Definition*: An operation that reduces each feature map to its single maximum value across the entire time dimension, identifying the strongest activation of each filter regardless of when it occurred.
    - *Why Used*: For activity recognition, the presence of specific motion patterns is often more important than their exact timing. Global max pooling creates time-invariant features, which is crucial for handling activities performed at different speeds by different users.
    - *Pipeline Benefit*: Creates a fixed-length feature vector regardless of the original signal length, enabling efficient transfer of knowledge between the teacher and student models.

- **Task-Specific Heads**  
  - **Transformation Discrimination Head**: 
    - *Detailed Architecture*: A fully connected layer with 256 units and ReLU activation, followed by a single-unit layer with sigmoid activation.
    - *Loss Function*: Binary cross-entropy, which measures the distance between two probability distributions (predicted vs. actual transformation application) and is mathematically optimal for binary classification problems.
    - *Why This Design*: The 256-unit layer provides sufficient capacity to identify subtle signal transformations while remaining computationally efficient. Sigmoid activation is used because transformation detection is a binary classification task (was a specific transformation applied or not?).
    - *Pipeline Benefit*: Enables the model to learn invariances to common signal distortions, which improves generalization to real-world noise patterns not seen in the training data.

  - **Activity Recognition Head**: 
    - *Detailed Architecture*: A fully connected layer with 1024 units and ReLU activation, followed by a multi-unit softmax output layer (one unit per activity class).
    - *Loss Function*: Categorical cross-entropy, which is the multi-class extension of binary cross-entropy, optimally suited for mutually exclusive class assignments.
    - *Why More Units (1024)*: The activity recognition task is more complex than transformation detection, requiring greater model capacity to distinguish between similar activities (e.g., walking vs. jogging vs. running) that may have subtle motion signature differences.
    - *Pipeline Benefit*: The larger capacity enables fine-grained activity distinction, while the softmax provides properly normalized confidence scores essential for the self-labeling process.

- **Training Protocol**  
  - **Adam Optimizer**: 
    - *Detailed Definition*: An adaptive gradient-based optimization algorithm that combines the benefits of AdaGrad (which works well with sparse gradients) and RMSProp (which adapts to changing gradient magnitudes).
    - *Why Used*: As explained in Section 4.3.1, Adam was selected for its robustness to different signal scales and noise levels in sensor data, making it particularly well-suited for the heterogeneous datasets in our study. Its adaptive learning rate handling prevents oscillation during training.
    - *Pipeline Benefit*: Enables stable convergence across both the teacher training and student training phases, despite their different loss landscapes.

  - **Learning Rate (0.0003)**: 
    - *Detailed Definition*: Controls the step size during gradient descent—how much to adjust model weights in response to the estimated error.
    - *Why This Value*: This relatively conservative value (0.0003) was determined through extensive experimentation across multiple datasets. Larger values caused training instability with noisy sensor data, while smaller values converged too slowly.
    - *Pipeline Benefit*: Strikes the optimal balance between convergence speed and stability, particularly important when training with self-labeled data that may contain some noise.

  - **Early Stopping**: 
    - *Detailed Definition*: A regularization technique that halts training when performance on a validation set stops improving, preventing overfitting to the training data.
    - *Implementation Details*: As mentioned in Section 4.3.1, we monitored validation loss with a patience of 5 epochs (training continues for 5 more epochs after the last improvement before stopping).
    - *Pipeline Benefit*: Particularly critical in our teacher-student framework to prevent the teacher from overfitting to the limited labeled data, which would propagate errors to the student through incorrect pseudo-labels.

- **Linear Evaluation Protocol**  
  - **Linear Evaluation**: 
    - *Detailed Definition*: A standard assessment method for representation learning where the convolutional feature extractor is frozen, and only a simple linear classifier (no additional non-linearities) is trained on top.
    - *Why Used*: As described in Section 4.3.2, this protocol specifically evaluates the quality of learned representations independent of classifier complexity. It reveals whether the network has discovered linearly separable features for the activities.
    - *Pipeline Benefit*: Provides a more stringent evaluation of feature quality than full fine-tuning, ensuring that our semi-supervised approach truly improves representation learning rather than just classification performance.

  - **Random Weight Initialization**: 
    - *Detailed Definition*: Initializing the linear classifier weights from a normal distribution with mean 0 and standard deviation 0.01.
    - *Why These Parameters*: This initialization scheme ensures that the classifier starts from an unbiased position with appropriately scaled weights. The small standard deviation (0.01) prevents initial saturation of softmax outputs.
    - *Pipeline Benefit*: Ensures fair comparison between different pre-training strategies by standardizing the classifier initialization, isolating the effect of the learned representations.

---

## Slide 18: Alternative Baseline Algorithms

- **En-Co-Training [10]**  
  - **Ensemble Co-Training** (*combines predictions from multiple classifiers—decision tree, Naïve Bayes, k-NN—trained on different features; ensemble: combining multiple models to improve performance; co-training: using multiple views/features to iteratively label unlabeled data*)
  - **Statistical Features** (*mean, correlation, interquartile range, mean absolute deviation, root mean square, standard deviation, variance, spectral energy; simple summary statistics extracted from each window*)
  - **Majority Voting** (*final prediction is the class predicted by the majority of classifiers*)
  - **Why?** Designed to leverage unlabeled data by combining diverse classifiers, but less effective for high-dimensional deep learning tasks.

- **Sparse-Encoding [3]**  
  - **Sparse Coding** (*learns a dictionary of basis vectors from unlabeled data; each sample is represented as a sparse linear combination of these vectors*)
  - **Empirical Entropy** (*measures the information content of basis vectors; low-entropy vectors are discarded as uninformative*)
  - **Support Vector Machine (SVM)** (*a supervised learning model for classification; RBF kernel: a popular nonlinear kernel function*)
  - **Grid-Search Cross-Validation** (*systematically searches for the best hyperparameters by evaluating performance on a validation set*)
  - **Why?** Attempts to extract useful features from unlabeled data, but computationally expensive and less scalable than deep learning.

---

## Slide 19: Comparison Against Baselines

- **Performance Gains**  
  - **Weighted F1 Score** (*harmonic mean of precision and recall, weighted by class frequency; standard metric for imbalanced classification tasks*)
  - SelfHAR achieves significantly higher weighted F1 scores than fully supervised models, especially on challenging datasets like UniMiB SHAR.

- **Summary of Results**  
  - **Statistical Significance** (*improvements are statistically significant if confidence intervals do not overlap; ensures results are not due to random chance*)
  - **Five Independent Runs** (*multiple runs with different random seeds to ensure robustness of results*)

- **Diagram Reference**  
  - Encourage the audience to refer to Table 2 for a visual summary of F1 scores across all methods and datasets.

---
