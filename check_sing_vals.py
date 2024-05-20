import torch
import numpy as np
import matplotlib.pyplot as plt
from resnet import resnet12
from config import args
from skew_ortho_conv import l2_normalize
from tqdm.notebook import tqdm
import argparse


parser = argparse.ArgumentParser(description='Script to parse command line arguments')
parser.add_argument('--model_path', type=str, help='Path to the model')
parser.add_argument('--save_title', type=str, help='Title for saving')
parser.add_argument('--layer_idx', type=int, help='Index of the layer')
parser.add_argument('--bn', action='store_true', help='Include Batch Normalization')

INPUT_SIZES = [84, 84, 84, 84, 42, 42, 42, 42, 21, 21, 21, 21, 10, 10, 10, 10]


def get_sing_vals_of_soc(layer, input_shape, iterations=15):
    return layer.norm(input_shape)


def get_sing_vals_of_kernel(kernel, input_shape):
    kernel = kernel.detach().cpu().permute([2, 3, 0, 1])
    transforms = np.fft.fft2(kernel, input_shape, axes=[0, 1])
    return np.linalg.svd(transforms, compute_uv=False)


def get_sing_vals_of_model(model, input_sizes, with_bn=False, layer_idx=-1):
    sing_vals = []
    idx = 0
    all_modules = list(model.named_modules())
    cur_conv_idx = 0
    with torch.no_grad():
        for cur_module_idx, (key, layer) in enumerate(all_modules):
            if 'Conv' in layer.__class__.__name__:
                if layer_idx == -1 or cur_conv_idx == layer_idx:
                    input_shape = [input_sizes[cur_conv_idx], input_sizes[cur_conv_idx]]
                    weight = layer.weight
                    if with_bn:
                        _, batch_norm = all_modules[cur_module_idx + 1]
                        weight = (weight.T / ((batch_norm.eps + batch_norm.running_var) ** 0.5)).T
                    cur_vals = get_sing_vals_of_kernel(weight, input_shape)
                    sing_vals.append(np.max(cur_vals))
                cur_conv_idx += 1
            elif 'SOC' in layer.__class__.__name__:
                if layer_idx == -1 or cur_conv_idx == layer_idx:
                    largest_sv = get_sing_vals_of_soc(layer, input_sizes[cur_conv_idx])
                    sing_vals.append(largest_sv)
                cur_conv_idx += 1
    if layer_idx == -1:
        return sing_vals
    else:
        return sing_vals[0]


def main():
    cmd_args = parser.parse_args()
    model_path = cmd_args.model_path
    save_title = cmd_args.save_title
    layer_idx = cmd_args.layer_idx
    bn = cmd_args.bn

    all_sing_vals = []
    for ep in tqdm(range(10, 101, 10)):
        model_path_full = model_path.format(ep=ep)
        checkpoint = torch.load(model_path_full, map_location='cpu')

        conv_type = 'soc' if args.use_soc else 'standart'
        block_type = 'full_relu' if args.jac_reg_type == 'tiny_block' and args.use_jac_reg else 'standart'
        model = resnet12(avg_pool=True, block_type=block_type, conv_type=conv_type, drop_rate=0.1, dropblock_size=5,
                         num_classes=args.n_cls)
        model.load_state_dict(checkpoint['model_state_dict'])

        sing_vals = get_sing_vals_of_model(model, INPUT_SIZES, bn, layer_idx)
        all_sing_vals.append(sing_vals)

        if layer_idx == -1:
            plt.figure()
            plt.plot(range(len(sing_vals)), sing_vals)
            plt.title(f'max singular values of each conv in {save_title}_{ep}ep')
            plt.xlabel('idx of conv layer')
            plt.ylabel('max singular value')
            plt.savefig(f'{save_title}_{ep}ep_sing_vals.png')

            np.save(f'{save_title}_{ep}ep_sing_vals.npy', sing_vals)

    np.save(f'{save_title}.npy', all_sing_vals)


if __name__ == '__main__':
    main()
