"""Verify saved Question 4 checkpoints and write an English results record (no PDF)."""
from pathlib import Path
import csv
import hashlib
import json
import zipfile
from q4_cnn import SmallCNN, CIFAR10, make_tensors, evaluate, DataLoader, OUT, ROOT
import numpy as np
import torch

def main():
    config=json.loads((OUT/'experiment_config.json').read_text())
    results=json.loads((OUT/'results.json').read_text())
    torch.set_num_threads(config['threads'])
    raw=CIFAR10(str(ROOT/'data'),train=False,download=False)
    loader=DataLoader(make_tensors(raw.data,raw.targets),batch_size=256,shuffle=False)
    validations=[]
    for row in results:
        name=row['model']
        saved=torch.load(OUT/f'{name}.pt',map_location='cpu',weights_only=True)
        model=SmallCNN(saved['second_channels'])
        model.load_state_dict(saved['state_dict'])
        actual,pred,truth=evaluate(model,loader,torch.device('cpu'))
        assert actual['correct']==row['test']['correct']
        assert len(pred)==10000
        with (OUT/f'{name}_test_predictions.csv').open(encoding='utf-8') as f:
            stored=list(csv.DictReader(f))
        assert len(stored)==10000
        assert all(raw.classes[int(p)]==s['predicted_class'] and raw.classes[int(t)]==s['true_class']
                   for p,t,s in zip(pred,truth,stored))
        validations.append({'model':name,'checkpoint_reload_verified':True,'predictions_match_all_10000_rows':True,
                            'correct_predictions':actual['correct']})
    indices=np.load(OUT/'training_indices.npy')
    assert len(indices)==len(np.unique(indices))==10000
    assert indices.min()>=0 and indices.max()<50000
    with (OUT/'training_history.csv').open() as f:history=list(csv.DictReader(f))
    assert len(history)==2*config['epochs']
    for row in results:
        h=[r for r in history if r['model']==row['model']]
        assert [int(r['epoch']) for r in h]==list(range(1,config['epochs']+1))
        assert abs(float(h[-1]['test_accuracy'])-row['test']['accuracy'])<1e-12
    b,w=results
    increase=w['parameters']-b['parameters']
    delta=100*(w['test']['accuracy']-b['test']['accuracy'])
    train_delta=100*(w['train']['accuracy']-b['train']['accuracy'])
    metric_table='\n'.join(f"{r['model']:14} {r['parameters']:>10,}  {100*r['train']['accuracy']:>8.2f}%  {100*r['test']['accuracy']:>7.2f}%  {r['test']['correct']:>5}/10000" for r in results)
    report=f'''HOMEWORK 4 - QUESTION 4: CNN ON CIFAR-10
English supporting results record for Question 4.

4(a) DATA AND EXPERIMENTAL PROCEDURE
The dataset was loaded with torchvision.datasets.CIFAR10. A NumPy generator
with seed 42 sampled 10,000 unique indices without replacement from the
standard 50,000-image training split. The same saved subset was used for both
models. Evaluation used all 10,000 images in the standard test split.
The mirrored archive was verified against the official MD5:
c58f30108f718f92721af3b95e74349a.

RGB pixels were converted to float tensors with shape (3, 32, 32), divided
by 255, then normalized with mean (0.5,0.5,0.5) and std (0.5,0.5,0.5).
No data augmentation was used. Class counts in the sampled training subset:
{dict(zip(config['class_names'],config['training_class_counts']))}

4(b) ARCHITECTURE, TRAINING, AND ACCURACY
Baseline architecture (batch dimension omitted):
Input                          3 x 32 x 32
Conv2d(3,16,3,stride=1,pad=1)  16 x 32 x 32       448 parameters
ReLU + MaxPool2d(2,2)          16 x 16 x 16         0 parameters
Conv2d(16,32,3,stride=1,pad=1) 32 x 16 x 16     4,640 parameters
ReLU + MaxPool2d(2,2)          32 x 8 x 8           0 parameters
Flatten                       2,048                 0 parameters
Linear(2048,64) + ReLU         64              131,136 parameters
Linear(64,10)                 10 logits           650 parameters
TOTAL                                         136,874 parameters

The model outputs logits directly to CrossEntropyLoss (no separate Softmax
is applied during training). Adam used learning rate 0.001 and weight decay
0.0001, batch size 128, for exactly {config['epochs']} epochs. Both models used
initialization seed 42 and a separate shuffle generator seeded 2026, so their
training-batch orders were identical. Training ran on CPU with four threads.
The final epoch was selected in advance; the test set was not used for
gradient updates, hyperparameter selection, or checkpoint selection.
Accuracy is measured after each epoch in eval mode, including training-set
accuracy. Thus training and test metrics use the same evaluation convention.

Model          Parameters   Train acc.  Test acc.   Test correct
{metric_table}

4(c) FIRST-CONVOLUTION HAND CALCULATION AND PYTORCH CHECK
Input: (N,3,32,32); filters=16; kernel=3x3; stride=1; padding=1; dilation=1.
H_out = floor((H + 2p - d(k-1) - 1)/s + 1)
      = floor((32 + 2 - 2 - 1)/1 + 1) = 32.
W_out = 32 by the same calculation.
Therefore the output shape is (N,16,32,32), BEFORE pooling.
Parameters = C_out * (C_in * k_h * k_w + 1)
           = 16 * (3 * 3 * 3 + 1) = 448,
including 432 kernel weights and 16 bias terms.
PyTorch verified this using a dummy (1,3,32,32) tensor and the sum of
parameter numel() values; the observed shape was (1,16,32,32), count 448.

Each filter's weights and bias are shared across all spatial locations.
Increasing input height or width changes the number of applications of the
same filter and the output size, but does not add new convolution weights.
This statement concerns the convolution layer; a flatten-to-linear layer
can acquire more parameters if its spatial input size is increased.

4(d) CONTROLLED ARCHITECTURE EXPERIMENT
Only the output-channel count of conv2 changed, from 32 to 64. The first
convolution, activation functions, two pooling layers, 64-unit hidden layer,
training subset, normalization, optimizer, learning rate, batch order, and
epoch count were kept fixed. The flattened input size of fc1 necessarily
changed from 2048 to 4096 as a consequence of doubling conv2 channels.

Wider conv2: 64 * (16 * 3 * 3 + 1) = 9,280 parameters.
Wider fc1: 4096 * 64 + 64 = 262,208 parameters.
Wider total: 448 + 9,280 + 262,208 + 650 = 272,586 parameters.
Change: +{increase:,} parameters ({100*increase/b['parameters']:.2f}%).
Change in training accuracy: {train_delta:+.2f} percentage points.
Change in test accuracy: {delta:+.2f} percentage points.

Before training, the expectation was that additional filters would increase
feature capacity and could improve fitting, while a test-accuracy gain was
not guaranteed with only 10,000 training examples. In this run, the wider
model {'improved' if train_delta>0 else 'did not improve'} training accuracy and
{'improved' if delta>0 else 'did not improve'} test accuracy.
The train-test gaps were {100*(b['train']['accuracy']-b['test']['accuracy']):.2f} percentage points for the baseline and
{100*(w['train']['accuracy']-w['test']['accuracy']):.2f} percentage points for the wider model. These gaps indicate a difference
between fitting the sampled training data and generalizing to unseen data.
The capacity increase therefore {'provided a test-set benefit in this experiment' if delta>0 else 'did not translate into a test-set benefit in this experiment'}.
This is one controlled comparison with one initialization/subset seed, not
evidence that a wider network will always be better. Random variation and
other training choices could change the outcome.

PREDICTION EXAMPLES AND WHY CNNs SUIT IMAGES
baseline_predictions.png and wider_conv2_predictions.png each show six
correct and six incorrect predictions, with test index, true label, and
predicted label. Examples were chosen deterministically as the first six
correct and first six incorrect test cases, rather than by visual appeal.
All 10,000 predictions per model and the confusion matrices are also saved.

For example, the baseline correctly recognizes the cat in test image #0,
but predicts deer for the dog in #24 and frog for the bird in #25. These
examples show that recognizing some visual patterns does not ensure robust
class separation in every image.

Convolutions exploit spatial locality: small filters detect patterns such
as edges, colors, and textures; subsequent layers combine these into more
complex features. Shared filters recognize a pattern at different positions
with relatively few parameters. Pooling reduces spatial size and provides
some tolerance to small shifts, while also discarding detail. These
properties suit image grids. Mistakes can arise from low 32x32 resolution,
similar shapes/textures, background variation, and the limited training
subset; the example images illustrate errors but do not establish a unique
cause for any particular prediction.

REPRODUCTION AND SAVED FILES
Install: python -m pip install -r q4_requirements.txt
Train:   python q4_cnn.py --epochs 12 --threads 4
Verify:  python q4_analyze.py
Run from the extracted assignment/bundle folder. CIFAR-10 is stored in data/.
Actual runtime: Python {platform_version()}, torch {config['torch']}, torchvision {config['torchvision']}.
The script can download from the original source. If that server is slow,
download the unchanged archive from the mirror noted in data/download_provenance.json
into data/cifar-10-python.tar.gz; torchvision verifies the official checksum.

output/q4/experiment_config.json: settings, class counts, and subset hash.
output/q4/training_indices.npy: exact 10,000 training indices.
output/q4/training_history.csv: per-epoch losses, accuracies, and durations.
output/q4/results.json: final metrics and parameter counts.
output/q4/*_architecture.json: checked layer shapes and parameter counts.
output/q4/*.pt: final model weights and configuration.
output/q4/*_test_predictions.csv: every test prediction.
output/q4/*_confusion_matrix.csv: true classes in rows, predicted classes in columns.
output/q4/*_examples.json: indices and labels in the example plots.
output/q4/training_curves.png: train/test accuracy and loss for both models.
output/q4/verification.json: checkpoint and saved-prediction validation.

References:
https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html
https://www.cs.toronto.edu/~kriz/cifar.html
'''
    (OUT/'q4_results_and_analysis.txt').write_text(report,encoding='utf-8')
    verified={'checks':validations,'training_subset_unique_count':10000,
              'history_epochs_per_model':config['epochs'],
              'training_code_sha256':hashlib.sha256((ROOT/'q4_cnn.py').read_bytes()).hexdigest()}
    (OUT/'verification.json').write_text(json.dumps(verified,indent=2),encoding='utf-8')
    with zipfile.ZipFile(ROOT/'output/Q4_code_and_results.zip','w',zipfile.ZIP_DEFLATED) as z:
        for path in [ROOT/'q4_cnn.py',ROOT/'q4_analyze.py',ROOT/'q4_requirements.txt',ROOT/'data/download_provenance.json']:
            z.write(path,path.relative_to(ROOT))
        for path in sorted(OUT.iterdir()):
            if path.is_file():z.write(path,path.relative_to(ROOT))
    print(json.dumps(verified,indent=2))
    print(metric_table)
    print('Wrote English results record and portable code/results archive. No PDF changed.')

def platform_version():
    import platform
    return platform.python_version()

if __name__=='__main__':main()
