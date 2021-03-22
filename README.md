[TensorFlow] Anomaly Detection with Adversarial Dual Autoencoders
=====

TensorFlow implementation of Anomaly Detection with Adversarial Dual Autoencoders (ADAE) with MNIST dataset.  

## Architecture

### Objective Functions
<div align="center">
  <img src="./figures/losses.png" width="500">  
  <p>The objective functions (losses) for training ADAE [1].</p>
</div>

### ADAE architecture
<div align="center">
  <img src="./figures/adae.png" width="500">  
  <p>The architecture of ADAE.</p>
</div>

### Graph in TensorBoard
<div align="center">
  <img src="./figures/graph.png" width="650">  
  <p>Graph of ADAE.</p>
</div>

### Problem Definition
<div align="center">
  <img src="./figures/definition.png" width="450">  
  <p>'Class-1' is defined as normal and the others are defined as abnormal.</p>
</div>

## Results

### Training Procedure
<div align="center">
  <p>
    <img src="./figures/ADAE_G_loss_g.svg" width="250">
    <img src="./figures/ADAE_G_loss_g_term1.svg" width="250">
    <img src="./figures/ADAE_G_loss_g_term2.svg" width="250">
  </p>
  <p>Loss graphs in the training procedure.</br>Each graph shows the generative loss, and the two terms that make loss-G.</p>
</div>

<div align="center">
  <p>
    <img src="./figures/ADAE_G_loss_g.svg" width="250">
    <img src="./figures/ADAE_G_loss_g_term1.svg" width="250">
    <img src="./figures/ADAE_G_loss_g_term2.svg" width="250">
  </p>
  <p>Loss graphs in the training procedure.</br>Each graph shows the discriminative loss, and the two terms that make loss-G.</p>
</div>

<div align="center">
  <img src="./figures/restoring.png" width="800">  
  <p>Restoration result by ADAE.</p>
</div>

### Test Procedure
<div align="center">
  <img src="./figures/test-box.png" width="400">
  <p>Box plot with encoding loss of test procedure.</p>
</div>

<div align="center">
  <p>
    <img src="./figures/in_in01.png" width="130">
    <img src="./figures/in_in02.png" width="130">
    <img src="./figures/in_in03.png" width="130">
  </p>
  <p>Normal samples classified as normal.</p>

  <p>
    <img src="./figures/in_out01.png" width="130">
    <img src="./figures/in_out02.png" width="130">
    <img src="./figures/in_out03.png" width="130">
  </p>
  <p>Abnormal samples classified as normal.</p>

  <p>
    <img src="./figures/out_in01.png" width="130">
    <img src="./figures/out_in02.png" width="130">
    <img src="./figures/out_in03.png" width="130">
  </p>
  <p>Normal samples classified as abnormal.</p>

  <p>
    <img src="./figures/out_out01.png" width="130">
    <img src="./figures/out_out02.png" width="130">
    <img src="./figures/out_out03.png" width="130">
  </p>
  <p>Abnormal samples classified as abnormal.</p>
</div>


## Environment
* Python 3.7.4  
* Tensorflow 1.14.0  
* Numpy 1.17.1  
* Matplotlib 3.1.1  
* Scikit Learn (sklearn) 0.21.3  


## Reference
[1] Ha Son Vu et al. (2019). <a href="https://arxiv.org/abs/1902.06924">Anomaly Detection with Adversarial Dual Autoencoders</a>. arXiv preprint arXiv:1902.06924.
