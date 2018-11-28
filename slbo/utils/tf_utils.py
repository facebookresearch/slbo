import tensorflow as tf


def get_tf_config():
    gpu_frac = 1

    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=gpu_frac,
        allow_growth=True,
    )
    config = tf.ConfigProto(
        gpu_options=gpu_options,
        log_device_placement=False,
        allow_soft_placement=True,
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1,
    )

    return config
