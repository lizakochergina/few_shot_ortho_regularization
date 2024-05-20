import torch


# ===== for jacobian =====
def orthogonal_loss(model, vectors, points, reg_type, cnt_convs=3):
    if reg_type == 'full':
        loss = apply_to_vectors(model, vectors, points)
        return loss / vectors.numel()
    elif reg_type == 'block':
        loss = 0.0
        for i, (name, layer) in enumerate(model.named_children()):
            if name.startswith('layer'):
                loss += apply_to_vectors(layer, vectors[i], points) / vectors[i].numel()
                with torch.no_grad():
                    points = layer(points)
        return loss
    elif reg_type == 'tiny_block':
        loss = 0.0
        inp_idx = 0
        ort_vec_idx = 0
        cur_points = None
        for name, layer in model.named_modules():
            k = ort_vec_idx % cnt_convs + 1
            if name.endswith(f'tiny_block_{k}'):
                if ort_vec_idx % cnt_convs == 0:
                    cur_points = points[inp_idx]
                    inp_idx += 1

                loss += apply_to_vectors(layer, vectors[ort_vec_idx], cur_points) / vectors[ort_vec_idx].numel()
                with torch.no_grad():
                    cur_points = layer(cur_points)
                ort_vec_idx += 1

        return loss / len(vectors)


def apply_to_vectors(child, vectors, points, device=torch.device('cuda:0'), no_transpose=False):
    if no_transpose:
        x = jac_matvec_t(child, points, vectors, device=device)
        mean_norm = (torch.linalg.norm(x.reshape(len(vectors), -1), dim=1) ** 2).sum()
    else:
        x_p = jac_matvec_t(child, points, vectors, device=device)
        x = jac_matvec(child, points, x_p, device=device)
        mean_norm = (torch.linalg.norm((x - vectors).reshape(len(vectors), -1), dim=1) ** 2).sum()
    return mean_norm


def jac_matvec_t(layer, x_, w, device=torch.device('cuda:0')):
    x = x_.clone().detach().requires_grad_(True).to(device)
    out = layer(x)
    t = torch.sum(out * w)
    dx = torch.autograd.grad(t, x, create_graph=True)[0]
    return dx


def jac_matvec(layer, x, v_, device=torch.device('cuda:0')):
    with torch.no_grad():
        out_size = layer(x).size()
    w = torch.zeros(out_size, requires_grad=True).to(device)
    jm_t = jac_matvec_t(layer, x, w, device=device)
    v = v_.clone().detach().requires_grad_(True).to(device)

    g = torch.sum(jm_t * v)
    dw = torch.autograd.grad(g, w, create_graph=True)[0]
    return dw


def generate_random_vectors(num_of_vectors, dim_vectors, dist,
                            dist_mean, dist_std, reg_type, device):
    if reg_type == 'full':
        vectors = generate_batch_random_vectors(
            num_of_vectors, dim_vectors, dist, dist_mean, dist_std, device
        )
        return vectors.to(device)
    elif reg_type == 'block' or reg_type == 'tiny_block':
        vectors = []
        for dim in dim_vectors:
            batch = generate_batch_random_vectors(num_of_vectors, dim, dist, dist_mean, dist_std, device)
            batch = batch.to(device)
            vectors.append(batch)
        return vectors
    else:
        raise Exception("no such type of jacobian regularization, check args.type")


def generate_batch_random_vectors(num_of_vectors, dim_vectors, dist,
                                  dist_mean, dist_std, device):
    if dist == 'normal':
        standard_normal_dist = torch.randn([num_of_vectors] + dim_vectors)
        normal_dist = standard_normal_dist * dist_std + dist_mean
        random_vectors = normal_dist.to(device)
    elif dist == 'rademacher':
        uniform_dist = torch.rand([num_of_vectors] + dim_vectors)
        rademacher_dist = torch.where(uniform_dist < 0.5, -1., 1.)
        random_vectors = rademacher_dist.to(device)
    else:
        raise Exception("Specify correct vector distribution: --dist <>")
    return random_vectors
