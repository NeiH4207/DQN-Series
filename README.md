# DQN-Series

### Installation

```python
# conda create -n dqn python=3.10
# conda activate dqn
pip install --e .
```

#### How to training CartPole

**Deep Q Learning**

```python
python test_gym/DQN.py \
    --num-episodes 1500 \
    --memory-size 8192 \
    --batch-size 128 \
    --lr 1e-3 \
    --optimizer adamw \
    --tau 0.01 \
    --gamma 0.99 \
    --model-path trained_models/model.pt 
```

**Double Deep Q Learning**

```python
python test_gym/DDQN.py \
    --num-episodes 1500 \
    --memory-size 8192 \
    --batch-size 128 \
    --lr 1e-3 \
    --optimizer adamw \
    --tau 0.01 \
    --gamma 0.99 \
    --model-path trained_models/model.pt 
```

** Prioritized Experience Replay Deep Q Learning**

```python
python test_gym/PER.py \
    --num-episodes 1500 \
    --memory-size 8192 \
    --batch-size 128 \
    --lr 1e-3 \
    --optimizer adamw \
    --tau 0.01 \
    --gamma 0.99 \
    --alpha 0.2 \
    --beta 0.6 \
    --prior_eps 1e-6 \
    --model-path trained_models/model.pt 
```

** Multi-step Deep Q Learning**

```
python test_gym/MultiStep.py \
    --num-episodes 1500 \
    --memory-size 8192 \
    --batch-size 128 \
    --lr 1e-3 \
    --optimizer adamw \
    --tau 0.01 \
    --gamma 0.99 \
    --alpha 0.2 \
    --beta 0.6 \
    --prior_eps 1e-6 \
    --n-step 3 \
    --model-path trained_models/model.pt 
```