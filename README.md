# Stochastic Lower Bound Optimization

This is the TensorFlow implementation for the paper [
Algorithmic Framework for Model-based Deep Reinforcement Learning with Theoretical Guarantees](https://arxiv.org/abs/1807.03858).
A PyTorch version will be released later.  


## Requirements
1. OpenAI Baselines
2. rllab (commit number `b3a2899`)
3. MuJoCo (1.5)
4. TensorFlow (>= 1.9)
5. NumPy (>= 1.14.5)
6. Python 3.6

## Run

Before running, please make sure that `rllab` and `baselines` are available 

```bash
python main.py -c configs/algos/slbo.yml configs/envs/half_cheetah.yml -s log_dir=/tmp
```

If you want to change hyper-parameters, you can either modify a corresponding `yml` file or 
change it temporarily by appending `model.hidden_sizes='[1000,1000]'` in the command line.

## License

See [LICENSE](LICENSE) for additional details.
