"""GPU configuration setup for TensorFlow/LSTM training."""
import tensorflow as tf
from loguru import logger
from config.config import Config

def configure_gpu():
    """Configure GPU settings for optimal performance."""
    try:
        # Check GPU availability
        gpus = tf.config.experimental.list_physical_devices('GPU')
        
        if gpus and Config.USE_GPU:
            logger.info(f"Found {len(gpus)} GPU(s): {[gpu.name for gpu in gpus]}")
            
            # Configure memory growth to avoid allocating all GPU memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Set memory limit if specified
            if Config.GPU_MEMORY_LIMIT:
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[0],
                    [tf.config.experimental.VirtualDeviceConfiguration(
                        memory_limit=Config.GPU_MEMORY_LIMIT
                    )]
                )
                logger.info(f"GPU memory limit set to {Config.GPU_MEMORY_LIMIT}MB")
            
            logger.info("GPU configuration completed successfully")
            return True
            
        else:
            logger.warning("No GPU found or GPU disabled. Using CPU for training.")
            return False
            
    except Exception as e:
        logger.error(f"GPU configuration failed: {e}")
        logger.info("Falling back to CPU training")
        return False

def test_gpu_setup():
    """Test GPU setup with a simple operation."""
    try:
        with tf.device('/GPU:0' if configure_gpu() else '/CPU:0'):
            # Simple test operation
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
            c = tf.matmul(a, b)
            
        logger.info("GPU/CPU test operation completed successfully")
        logger.info(f"Test result shape: {c.shape}")
        return True
        
    except Exception as e:
        logger.error(f"GPU/CPU test failed: {e}")
        return False

if __name__ == "__main__":
    configure_gpu()
    test_gpu_setup()