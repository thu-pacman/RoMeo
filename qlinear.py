import torch
from functools import partial

from qmatmul import matmul_w4a4, matmul_w8a8
from quant import pertoken_quantize, pertoken_quantize_search, pertoken_threshold, get_complementary, perchannel_threshold
from rotate import Rotation

@torch._dynamo.disable
class RLinear(torch.nn.Module):
    def __init__(self, args, linear, name, layer_id, norm_scale):
        super().__init__()
        self.linear = linear
        self.layer_id = layer_id
        self.name = name
        self.norm_scale = norm_scale

        self.rotate = Rotation.get_rotation(args.rotate, linear.in_features, linear.weight.dtype, linear.weight.device)
        self.rotate_t = Rotation.get_rotation(args.rotate, linear.out_features, linear.weight.dtype, linear.weight.device)
        rotate_type = 'down' if 'o_proj' in name or 'down_proj' in name else 'up'
        self.rotate_type = rotate_type

        if args.smooth_quant:
            self.pretrain_scale = linear.pretrain_scale
            self.linear.weight.data *= linear.pretrain_scale
        else:
            self.pretrain_scale = torch.tensor(1.0, dtype=linear.weight.dtype, device=linear.weight.device)
        
        if args.rotate_opt:
            if self.rotate_type == 'up':
                self.linear.weight.data *= self.norm_scale.unsqueeze(0)
                self.linear.weight.data = self.rotate.apply(self.linear.weight.data)
            elif self.rotate_type == 'down':
                self.linear.weight.data = self.rotate.apply(self.linear.weight.data)
                self.linear.weight.data = self.rotate_t.apply(self.linear.weight.data.transpose(-1, -2)).transpose(-1, -2)
            else:
                raise RuntimeError(f"Unsupported rotate type {self.rotate_type}")
        else:
            self.rotate_type = 'down' # online hadamard transform
            self.linear.weight.data = self.rotate.apply(self.linear.weight.data)

        if args.capture_layer_ids:
            capture_layer_ids = list(map(int, args.capture_layer_ids.split(',')))
            if layer_id in capture_layer_ids:
                self.register_buffer('unquantized_weight', self.linear.weight.data.detach().clone())

    def dump_mm_input(self, inp):
        return self.rotate.apply(inp)

    def dump_mm_weight(self):
        return self.unquantized_weight
    
    def forward(self, inp):
        scaled_input = inp / self.pretrain_scale
        if self.rotate_type == 'up':
            return self.linear_method(scaled_input / self.norm_scale.unsqueeze(0))
        if self.rotate_type == 'down':
            return self.linear_method(self.rotate.apply(scaled_input))

class RQErrLinear(RLinear):
    def __init__(self, args, linear, name, layer_id):
        args.capture_layer_ids = None
        super().__init__(args, linear, name, layer_id)
        self.linear_method = self.linear_with_errs
        self.linear.requires_grad_(False)

    def init_errs(self, inp):
        act, act_scale = pertoken_quantize(inp, 4, self.linear.in_features)
        w, w_scale = pertoken_quantize(self.linear.weight, 4, self.linear.in_features)
        self.register_parameter('input_err', torch.nn.Parameter(act * act_scale - inp))
        self.register_parameter('weight_err', torch.nn.Parameter(w * w_scale - self.linear.weight))
    
    def linear_with_errs(self, inp):
        num_batch = inp.shape[0]
        inp = inp.view(-1, self.linear.in_features)
        if getattr(self, 'input_err', None) is None:
            self.init_errs(inp)
        out = (inp + self.input_err) @ (self.linear.weight + self.weight_err).T
        out = out.view(num_batch, -1, self.linear.out_features)
        return out
    
    def dump_layer_err(self, input_err_dir, weight_err_dir):
        if not hasattr(self, 'input_err'):
            raise RuntimeError("Input error is not initialized.")
        input_err_grad_dir = input_err_dir + "_grad"
        weight_err_grad_dir = weight_err_dir + "_grad"
        torch.save(self.input_err, f"{input_err_dir}/layer{self.layer_id}.{self.name}.pt")
        torch.save(self.weight_err, f"{weight_err_dir}/layer{self.layer_id}.{self.name}.pt")
        torch.save(self.input_err.grad, f"{input_err_grad_dir}/layer{self.layer_id}.{self.name}.pt")
        torch.save(self.weight_err.grad, f"{weight_err_grad_dir}/layer{self.layer_id}.{self.name}.pt")

class RQLinear(RLinear):

    def __init__(self, args, linear, name, layer_id, norm_scale):
        super().__init__(args, linear, name, layer_id, norm_scale)
        act_group_size = args.a_group if args.a_group is not None else self.linear.in_features
        weight_group_size = args.w_group if args.w_group is not None else self.linear.in_features
        if args.w_clip:
            self.weight_quantize_method = partial(pertoken_quantize_search, bits=args.w_bits, group_size=weight_group_size)
        else:
            self.weight_quantize_method = partial(pertoken_quantize, bits=args.w_bits, group_size=weight_group_size)
        self.act_quantize_method = partial(pertoken_quantize, bits=args.a_bits, group_size=act_group_size)

        if args.w_bits < 16:
            w, w_scale = self.weight_quantize_method(self.linear.weight) # linear.weight nt format
            self.register_buffer('w', w)
            self.register_buffer('w_scale', w_scale)
            del self.linear.weight
        
        if max(args.a_bits, args.w_bits) == 16:
            self.linear_method = self.linear
        elif max(args.a_bits, args.w_bits) == 8:
            self.linear_method = partial(self.linear_quantized, act_quant_fn=self.act_quantize_method, matmul_fn=partial(matmul_w8a8, group_size_a=act_group_size, group_size_b=weight_group_size))
        elif max(args.a_bits, args.w_bits) == 4:
            self.linear_method = partial(self.linear_quantized, act_quant_fn=self.act_quantize_method, matmul_fn=partial(matmul_w4a4, group_size_a=act_group_size, group_size_b=weight_group_size))
        else:
            raise RuntimeError(f"Unsupported QLinear a_bits={args.a_bits}, w_bits={args.w_bits}")
    
    def linear_quantized(self, inp, act_quant_fn, matmul_fn):
        assert len(inp.shape) == 3
        num_batch = inp.shape[0]
        inp = inp.view(-1, self.linear.in_features)
        act, act_scale = act_quant_fn(inp)
        out = matmul_fn(act, act_scale, self.w.T, self.w_scale.T)

        out = out.view(num_batch, -1, self.linear.out_features)
        if torch.isnan(out).any():
            raise RuntimeError("NaN detected in output of linear layer")
        if torch.isinf(out).any():
            raise RuntimeError("Inf detected in output of linear layer")
        return out

@torch._dynamo.disable
class MixedRQLinear(RLinear):

    low_precision_counter = 0
    total_counter = 0

    @staticmethod
    def inc_low_precision_counter(cnt):
        MixedRQLinear.low_precision_counter += cnt
    
    @staticmethod
    def inc_total_counter(cnt):
        MixedRQLinear.total_counter += cnt

    @staticmethod
    def dump_counter():
        print(f"Low precision counter: {MixedRQLinear.low_precision_counter}")
        print(f"Total counter: {MixedRQLinear.total_counter}")
        print(f"Low precision ratio: {MixedRQLinear.low_precision_counter / MixedRQLinear.total_counter:.2%}")

    def __init__(self, args, linear, name, layer_id, norm_scale):
        super().__init__(args, linear, name, layer_id, norm_scale)
        assert args.mixed_precision != 'none', "MixedRQLinear requires --mixed-precision argument"

        self.act_group_size = args.a_group if args.a_group is not None else self.linear.in_features
        self.weight_group_size = args.w_group if args.w_group is not None else self.linear.in_features
        if args.mixed_precision == 'bitweaver':
            if args.qfactory_kernel:
                from qfactory.linears import OptMixedQLinear as MixedQLinearImpl
                self.linear_impl = MixedQLinearImpl(self.linear.weight, args.activation_threshold, args.weight_threshold, multistream=args.multistream, unifiedkernel=args.unifiedkernel)
                def reshaped_forward(inp):
                    if isinstance(inp, tuple):  # skip quant
                        return self.linear_impl.forward(inp)
                    return self.linear_impl.forward(inp.view(-1, self.linear.in_features)).view(inp.shape[0], -1, self.linear.out_features)
                self.linear_method = reshaped_forward
            else:
                weight_outlier_idx = pertoken_threshold(self.linear.weight, args.weight_threshold, args.threshold_policy)
                self.register_buffer('weight_outlier_idx', weight_outlier_idx)
                if args.w_clip:
                    wo, wo_scale = pertoken_quantize_search(self.linear.weight[weight_outlier_idx], 8, self.weight_group_size)
                    wn, wn_scale = pertoken_quantize_search(self.linear.weight, 4, self.weight_group_size)
                else:
                    wo, wo_scale = pertoken_quantize(self.linear.weight[weight_outlier_idx], 8, self.weight_group_size)
                    wn, wn_scale = pertoken_quantize(self.linear.weight, 4, self.weight_group_size)
                self.register_buffer('wo', wo)
                self.register_buffer('wo_scale', wo_scale)
                self.register_buffer('wn', wn)
                self.register_buffer('wn_scale', wn_scale)
                self.linear_method = self.linear_bitweaver

            del self.linear.weight
        elif args.mixed_precision == 'mixq':
            assert args.w_clip == False, "mixq MixedRQLinear does not support weight clipping"
            if args.a_group is not None or args.w_group is not None:
                raise RuntimeError("MixedRQLinear does not support a_group or w_group arguments for mixq")
            if args.weight_threshold > 0:
                print(f"Warning: weight threshold {args.weight_threshold} is ignored for mixed precision {args.mixed_precision}")
            self.linear_method = partial(
                self.linear_mixq,
                act_quant_fn=partial(pertoken_quantize, bits=args.a_bits),
                weight_quant_fn=partial(pertoken_quantize, bits=args.w_bits),
                matmul_fn=matmul_w4a4 if min(args.a_bits, args.w_bits) == 4 else matmul_w8a8
            )
        else:
            raise RuntimeError(f"Unsupported mixed precision {args.mixed_precision}")

        self.activation_threshold = args.activation_threshold
        self.threshold_policy = args.threshold_policy
    
    def linear_bitweaver(self, inp):
        assert len(inp.shape) == 3
        out = torch.empty(inp.shape[0], inp.shape[1], self.linear.out_features, device=inp.device, dtype=inp.dtype)
        inp_view = inp.view(-1, self.linear.in_features)
        out_view = out.view(-1, self.linear.out_features)

        act_outlier_idx = pertoken_threshold(inp_view, self.activation_threshold, self.threshold_policy)
        ao, ao_scale = pertoken_quantize(inp_view[act_outlier_idx], 8, self.act_group_size)
        an, an_scale = pertoken_quantize(inp_view, 4, self.act_group_size)

        out_view.copy_(matmul_w4a4(an, an_scale, self.wn.T, self.wn_scale.T, self.act_group_size, self.weight_group_size))
        if self.weight_outlier_idx.numel() > 0:
            out_view[:, self.weight_outlier_idx] = matmul_w8a8(an, an_scale, self.wo.T, self.wo_scale.T, self.act_group_size, self.weight_group_size)
        if act_outlier_idx.numel() > 0:
            out_view[act_outlier_idx, :] = matmul_w8a8(ao, ao_scale, self.wn.T, self.wn_scale.T, self.act_group_size, self.weight_group_size)
            if self.weight_outlier_idx.numel() > 0:
                out_view[act_outlier_idx[:, None], self.weight_outlier_idx[None, :]] = matmul_w8a8(ao, ao_scale, self.wo.T, self.wo_scale.T, self.act_group_size, self.weight_group_size)

        self.inc_low_precision_counter(out_view.numel())
        self.inc_total_counter(out_view.numel() + an.shape[0] * self.wo.shape[0] + ao.shape[0] * self.wn.shape[0] + ao.shape[0] * self.wo.shape[0])
        return out
    
    def linear_mixq(self, inp, act_quant_fn, weight_quant_fn, matmul_fn):
        assert len(inp.shape) == 3
        inp_view = inp.view(-1, self.linear.in_features)

        act_outlier_idx = perchannel_threshold(inp_view, self.activation_threshold, self.threshold_policy)
        act_normal_idx = get_complementary(inp_view.shape[1], act_outlier_idx)
        ao = inp_view[:, act_outlier_idx]
        an, an_scale = act_quant_fn(inp_view[:, act_normal_idx], group_size=act_normal_idx.shape[0])
        wo = self.linear.weight[:, act_outlier_idx]
        wn, wn_scale = weight_quant_fn(self.linear.weight[:, act_normal_idx], group_size=act_normal_idx.shape[0])

        out = matmul_fn(an, an_scale, wn.T, wn_scale.T, group_size_a=an.shape[1], group_size_b=wn.shape[1]) + ao @ wo.T
        out = out.view(inp.shape[0], -1, self.linear.out_features)

        self.inc_low_precision_counter(act_normal_idx.numel())
        self.inc_total_counter(inp_view.shape[1])
        return out
