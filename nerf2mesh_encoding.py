import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math

class MLP(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        dim_hidden,
        num_layers,
        bias=True,
        geom_init=False,
        weight_norm=False,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers
        self.geom_init = geom_init

        net = []
        for l in range(num_layers):
            in_dim = self.dim_in if l == 0 else self.dim_hidden
            out_dim = self.dim_out if l == num_layers - 1 else self.dim_hidden

            net.append(nn.Linear(in_dim, out_dim, bias=bias))

            if geom_init:
                if l == num_layers - 1:
                    torch.nn.init.normal_(
                        net[l].weight,
                        mean=math.sqrt(math.pi) / math.sqrt(in_dim),
                        std=1e-4,
                    )
                    if bias:
                        torch.nn.init.constant_(
                            net[l].bias, -0.5
                        )  # sphere init (very important for hashgrid encoding!)

                elif l == 0:
                    torch.nn.init.normal_(
                        net[l].weight[:, :3], 0.0, math.sqrt(2) / math.sqrt(out_dim)
                    )
                    torch.nn.init.constant_(net[l].weight[:, 3:], 0.0)
                    if bias:
                        torch.nn.init.constant_(net[l].bias, 0.0)

                else:
                    torch.nn.init.normal_(
                        net[l].weight, 0.0, math.sqrt(2) / math.sqrt(out_dim)
                    )
                    if bias:
                        torch.nn.init.constant_(net[l].bias, 0.0)

            if weight_norm:
                net[l] = nn.utils.weight_norm(net[l])

        self.net = nn.ModuleList(net)

    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                if self.geom_init:
                    x = F.softplus(x, beta=100)
                else:
                    x = F.relu(x, inplace=True)
        return x


class FreqEncoder_torch(nn.Module):
    def __init__(
        self,
        input_dim,
        max_freq_log2,
        N_freqs,
        log_sampling=True,
        include_input=True,
        periodic_fns=(torch.sin, torch.cos),
    ):
        super().__init__()

        self.input_dim = input_dim
        self.include_input = include_input
        self.periodic_fns = periodic_fns

        self.output_dim = 0
        if self.include_input:
            self.output_dim += self.input_dim

        self.output_dim += self.input_dim * N_freqs * len(self.periodic_fns)

        if log_sampling:
            self.freq_bands = 2.0 ** torch.linspace(0.0, max_freq_log2, N_freqs)
        else:
            self.freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq_log2, N_freqs)

        self.freq_bands = self.freq_bands.numpy().tolist()

    def forward(self, input, **kwargs):
        out = []
        if self.include_input:
            out.append(input)

        for i in range(len(self.freq_bands)):
            freq = self.freq_bands[i]
            for p_fn in self.periodic_fns:
                out.append(p_fn(input * freq))

        out = torch.cat(out, dim=-1)

        return out


class TCNN_hashgrid(nn.Module):
    def __init__(
        self,
        num_levels,
        level_dim,
        log2_hashmap_size,
        base_resolution,
        desired_resolution,
        interpolation,
        **kwargs
    ):
        super().__init__()
        import tinycudann as tcnn

        self.encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": num_levels,
                "n_features_per_level": level_dim,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_resolution,
                "per_level_scale": np.exp2(
                    np.log2(desired_resolution / num_levels) / (num_levels - 1)
                ),
                "interpolation": "Smoothstep"
                if interpolation == "smoothstep"
                else "Linear",
            },
            dtype=torch.float32,
        )
        self.output_dim = self.encoder.n_output_dims  # patch

    def forward(self, x, bound=1, **kwargs):
        return self.encoder((x + bound) / (2 * bound))


# ref: IDE in Ref-NeRF
class IntegratedDirectionalEncoder(nn.Module):
    def __init__(self, degree):
        super().__init__()

        from shencoder import SHEncoder

        self.sh_encoder = SHEncoder(input_dim=3, degree=degree)

        sigma = []
        for l in range(degree):
            sigma.extend([0.5 * l * (l + 1)] * (2 * l + 1))
        self.sigma = torch.tensor(
            sigma, dtype=torch.float32, device="cuda"
        )  # [output_dim,]

        self.degree = degree
        self.output_dim = self.sh_encoder.output_dim

    def forward(self, x, r=None):
        # x: [..., 3], normalized directions
        # r: [..., 1] or float , roughness

        att = torch.exp(-self.sigma * r)
        enc = self.sh_encoder(x).float()  # [..., C]

        return att * enc


# ref: IDE in NeRO
def generalized_binomial_coeff(a, k):
    """Compute generalized binomial coefficients."""
    return np.prod(a - np.arange(k)) / np.math.factorial(k)


def assoc_legendre_coeff(l, m, k):
    """Compute associated Legendre polynomial coefficients.

    Returns the coefficient of the cos^k(theta)*sin^m(theta) term in the
    (l, m)th associated Legendre polynomial, P_l^m(cos(theta)).

    Args:
      l: associated Legendre polynomial degree.
      m: associated Legendre polynomial order.
      k: power of cos(theta).

    Returns:
      A float, the coefficient of the term corresponding to the inputs.
    """
    return (
        (-1) ** m
        * 2**l
        * np.math.factorial(l)
        / np.math.factorial(k)
        / np.math.factorial(l - k - m)
        * generalized_binomial_coeff(0.5 * (l + k + m - 1.0), l)
    )


def sph_harm_coeff(l, m, k):
    """Compute spherical harmonic coefficients."""
    return np.sqrt(
        (2.0 * l + 1.0)
        * np.math.factorial(l - m)
        / (4.0 * np.pi * np.math.factorial(l + m))
    ) * assoc_legendre_coeff(l, m, k)


def get_ml_array(degree):
    """Create a list with all pairs of (l, m) values to use in the encoding."""
    ml_list = []
    for i in range(degree):
        l = 2**i
        # Only use nonnegative m values, later splitting real and imaginary parts.
        for m in range(l + 1):
            ml_list.append((m, l))

    # Convert list into a numpy array.
    ml_array = np.array(ml_list).T
    return ml_array


def generate_ide_fn(degree):
    """Generate integrated directional encoding (IDE) function.

    This function returns a function that computes the integrated directional
    encoding from Equations 6-8 of arxiv.org/abs/2112.03907.

    Args:
      degree: number of spherical harmonics degrees to use.

    Returns:
      A function for evaluating integrated directional encoding.

    Raises:
      ValueError: if degree is larger than 5.
    """
    if degree > 5:
        raise ValueError("Only degree of at most 5 is numerically stable.")

    ml_array = get_ml_array(degree)
    l_max = 2 ** (degree - 1)

    # Create a matrix corresponding to ml_array holding all coefficients, which,
    # when multiplied (from the right) by the z coordinate Vandermonde matrix,
    # results in the z component of the encoding.
    mat = np.zeros((l_max + 1, ml_array.shape[1]))
    for i, (m, l) in enumerate(ml_array.T):
        for k in range(l - m + 1):
            mat[k, i] = sph_harm_coeff(l, m, k)

    mat = torch.from_numpy(mat.astype(np.float32)).cuda()
    ml_array = torch.from_numpy(ml_array.astype(np.float32)).cuda()

    def integrated_dir_enc_fn(xyz, kappa_inv):
        """Function returning integrated directional encoding (IDE).

        Args:
          xyz: [..., 3] array of Cartesian coordinates of directions to evaluate at.
          kappa_inv: [..., 1] reciprocal of the concentration parameter of the von
            Mises-Fisher distribution.

        Returns:
          An array with the resulting IDE.
        """
        x = xyz[..., 0:1]
        y = xyz[..., 1:2]
        z = xyz[..., 2:3]

        # Compute z Vandermonde matrix.
        vmz = torch.concat([z**i for i in range(mat.shape[0])], dim=-1)

        # Compute x+iy Vandermonde matrix.
        vmxy = torch.concat([(x + 1j * y) ** m for m in ml_array[0, :]], dim=-1)

        # Get spherical harmonics.
        sph_harms = vmxy * torch.matmul(vmz, mat)

        # Apply attenuation function using the von Mises-Fisher distribution
        # concentration parameter, kappa.
        sigma = 0.5 * ml_array[1, :] * (ml_array[1, :] + 1)
        ide = sph_harms * torch.exp(-sigma * kappa_inv)

        # Split into real and imaginary parts and return
        return torch.concat([torch.real(ide), torch.imag(ide)], dim=-1)

    return integrated_dir_enc_fn


def get_encoder(
    encoding,
    input_dim=3,
    output_dim=1,
    resolution=300,
    mode="bilinear",  # dense grid
    multires=6,  # freq
    degree=4,  # SH
    num_levels=16,
    level_dim=2,
    base_resolution=16,
    log2_hashmap_size=19,
    desired_resolution=2048,  # hash/tiled grid
    align_corners=False,
    interpolation="linear",  # grid
    **kwargs
):
    if encoding == "None":
        return lambda x, **kwargs: x, input_dim

    elif encoding == "frequency_torch":
        encoder = FreqEncoder_torch(
            input_dim=input_dim,
            max_freq_log2=multires - 1,
            N_freqs=multires,
            log_sampling=True,
        )

    elif encoding == "frequency":
        from freqencoder import FreqEncoder

        encoder = FreqEncoder(input_dim=input_dim, degree=multires)

    elif encoding == "sh":
        from shencoder import SHEncoder

        encoder = SHEncoder(input_dim=input_dim, degree=degree)

    elif encoding == "ide":
        encoder = IntegratedDirectionalEncoder(degree=degree)

    elif encoding == "nero_ide":
        encoder = generate_ide_fn(degree=degree)
        output_dim = 2 * (degree + 2**5 - 1)
        return encoder, output_dim

    elif encoding == "hashgrid":
        from gridencoder import GridEncoder

        encoder = GridEncoder(
            input_dim=input_dim,
            num_levels=num_levels,
            level_dim=level_dim,
            base_resolution=base_resolution,
            log2_hashmap_size=log2_hashmap_size,
            desired_resolution=desired_resolution,
            gridtype="hash",
            align_corners=align_corners,
            interpolation=interpolation,
        )

    elif encoding == "hashgrid_tcnn":
        encoder = TCNN_hashgrid(
            num_levels=num_levels,
            level_dim=level_dim,
            log2_hashmap_size=log2_hashmap_size,
            base_resolution=base_resolution,
            desired_resolution=desired_resolution,
            interpolation=interpolation,
        )

    elif encoding == "tiledgrid":
        from gridencoder import GridEncoder

        encoder = GridEncoder(
            input_dim=input_dim,
            num_levels=num_levels,
            level_dim=level_dim,
            base_resolution=base_resolution,
            log2_hashmap_size=log2_hashmap_size,
            desired_resolution=desired_resolution,
            gridtype="tiled",
            align_corners=align_corners,
            interpolation=interpolation,
        )

    else:
        raise NotImplementedError(
            "Unknown encoding mode, choose from [None, frequency, sh, hashgrid, tiledgrid]"
        )

    return encoder, encoder.output_dim
