import os
import numpy as np
import h5py
import tensorflow as tf
import lgn_model
import stim_dataset

def generate_mnist_spikes(output_h5, n_samples=100, data_usage=0, intensity=1.0, 
                          im_slice=100, pre_delay=50, post_delay=150, seed=3000):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    all_ds = tf.keras.datasets.mnist.load_data()
    train, test = all_ds
    
    def unison_shuffled_copies(a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]
    
    train = unison_shuffled_copies(train[0], train[1])
    test = unison_shuffled_copies(test[0], test[1])
    all_ds = (tuple(train), tuple(test))
    
    if data_usage == 0:
        images, labels = all_ds[data_usage]
    else:
        images, labels = all_ds[data_usage]
    
    images = images[:n_samples]
    labels = labels[:n_samples]
    
    if len(images.shape) > 3:
        images = tf.image.rgb_to_grayscale(images) / 255
    else:
        images = images[..., None] / 255
    
    lgn = lgn_model.LGN()
    seq_len = pre_delay + im_slice + post_delay
    
    all_spike_trains = []
    all_labels = []
    
    print(f'Generating spike trains for {n_samples} images')
    for ind in range(n_samples):
        if (ind + 1) % 10 == 0:
            print(f'  Processing {ind + 1}/{n_samples}')
        
        img = tf.image.resize_with_pad(images[ind], 120, 240, method='lanczos5')
        tiled_img = tf.tile(img[None, ...], (im_slice, 1, 1, 1))
        tiled_img = (tiled_img - .5) * intensity / .5
        
        z1 = tf.tile(tf.zeros_like(img)[None, ...], (pre_delay, 1, 1, 1))
        z2 = tf.tile(tf.zeros_like(img)[None, ...], (post_delay, 1, 1, 1))
        videos = tf.concat((z1, tiled_img, z2), 0)
        
        spatial = lgn.spatial_response(videos)
        firing_rates = lgn.firing_rates_from_spatial(*spatial)
        
        _p = 1 - tf.exp(-firing_rates / 1000.)
        _z = _p * 1.3
        
        all_spike_trains.append(_z.numpy())
        all_labels.append(labels[ind])
    
    all_spike_trains = np.array(all_spike_trains, dtype=np.float32)
    all_labels = np.array(all_labels, dtype=np.int32)
    
    print(f'Saving to {output_h5}')
    with h5py.File(output_h5, 'w') as f:
        f.attrs['n_samples'] = n_samples
        f.attrs['n_lgn'] = 17400
        f.attrs['seq_len'] = seq_len
        f.attrs['pre_delay'] = pre_delay
        f.attrs['im_slice'] = im_slice
        f.attrs['post_delay'] = post_delay
        
        f.create_dataset('spike_trains', data=all_spike_trains, compression='gzip')
        f.create_dataset('labels', data=all_labels)
        
        chunk_size = 50
        n_chunks = seq_len // chunk_size
        time_windows = np.zeros((n_chunks, 2), dtype=np.int32)
        for i in range(n_chunks):
            time_windows[i] = [i * chunk_size, (i + 1) * chunk_size]
        f.create_dataset('time_windows', data=time_windows)
        
        response_window = np.array([pre_delay, pre_delay + chunk_size], dtype=np.int32)
        f.create_dataset('response_window', data=response_window)
    
    print(f'Generated {n_samples} spike trains with shape {all_spike_trains.shape}')
    print(f'  Sequence length: {seq_len} ms')
    print(f'  LGN neurons: 17400')
    print(f'  Response window: [{pre_delay}, {pre_delay + chunk_size}] ms')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='mnist.h5')
    parser.add_argument('--n_samples', type=int, default=100)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--intensity', type=float, default=2.0)
    parser.add_argument('--im_slice', type=int, default=100)
    parser.add_argument('--pre_delay', type=int, default=50)
    parser.add_argument('--post_delay', type=int, default=450)
    parser.add_argument('--seed', type=int, default=3000)
    args = parser.parse_args()
    
    data_usage = 1 if args.test else 0
    generate_mnist_spikes(args.output, args.n_samples, data_usage, args.intensity,
                         args.im_slice, args.pre_delay, args.post_delay, args.seed)