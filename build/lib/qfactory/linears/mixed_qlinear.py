import math
import torch
from ..kernels import *

@torch.no_grad()
def pertoken_threshold(inp, threshold, policy):
    assert len(inp.shape) == 2
    pertoken_max = torch.max(torch.abs(inp), dim=1, keepdim=True).values
    if policy == 'value':
        high_idx = (pertoken_max >= threshold).squeeze(1).nonzero().squeeze(1)
    else:
        number_outliers = int(threshold)
        _, high_idx = torch.topk(pertoken_max.squeeze(1), number_outliers, largest=True, sorted=False)
    return high_idx

@torch.no_grad()
def pertoken_quantize(inp, bits):
    assert len(inp.shape) == 2
    int_range = 2 ** (bits - 1) - 1
    pertoken_max = torch.max(torch.abs(inp), dim=1, keepdim=True).values # (token, 1)
    pertoken_scale = pertoken_max / int_range
    quantized_inp = torch.round(inp / pertoken_scale).to(torch.int32)
    return quantized_inp, pertoken_scale

def pack_int4_to_uint8(inp):
    assert inp.dtype == torch.int32
    assert inp.shape[1] % 2 == 0
    unpacked = (inp + 16 * (inp < 0))
    # assert 0 <= unpacked.min() and unpacked.max() < 16
    packed = unpacked[:, 0::2] | (unpacked[:, 1::2] << 4)
    return packed

class MixedQLinear(torch.nn.Module):
    def __init__(self, weight, p_a_outlier, p_w_outlier): # weight is in transposed format
        super().__init__()

        w_outliers = math.ceil(weight.shape[0] * p_w_outlier / 256) * 256

        assert weight.shape[1] % 2 == 0
        qweight = torch.empty(weight.shape[0] + 2 * w_outliers, weight.shape[1] // 2, dtype=torch.uint8, device=weight.device)
        qweight_scale = torch.empty(weight.shape[0] + w_outliers, dtype=torch.bfloat16, device=weight.device)

        weight_outlier_idx = pertoken_threshold(weight, w_outliers, "amount")
        wo, wo_scale = pertoken_quantize(weight[weight_outlier_idx], 8)
        wn, wn_scale = pertoken_quantize(weight, 4)

        qweight[:-2 * w_outliers].copy_(pack_int4_to_uint8(wn))
        qweight_scale[:-w_outliers].copy_(wn_scale.squeeze(1))
        qweight[-2 * w_outliers:].view(w_outliers, weight.shape[1]).copy_(wo.to(torch.uint8))
        qweight_scale[-w_outliers:].copy_(wo_scale.squeeze(1))

        self.register_buffer('w', qweight)
        self.register_buffer('w_scale', qweight_scale)
        self.register_buffer('w_idx', weight_outlier_idx)
        self.w_outliers = w_outliers
        self.p_a_outlier = p_a_outlier

    def act_quantize(self, inp):
        assert inp.shape[1] % 2 == 0
        qinput = torch.empty(inp.shape[0] + 2 * self.outliers, inp.shape[1] // 2, dtype=torch.uint8, device=inp.device)
        qinput_scale = torch.empty(inp.shape[0] + self.outliers, dtype=torch.bfloat16, device=inp.device)

        inp_outlier_idx = pertoken_threshold(inp, self.outliers, "amount")
        ao, ao_scale = pertoken_quantize(inp[inp_outlier_idx], 8)
        an, an_scale = pertoken_quantize(inp, 4)

        qinput[:-2 * self.outliers].copy_(pack_int4_to_uint8(an))
        qinput_scale[:-self.outliers].copy_(an_scale.squeeze(1))
        qinput[-2 * self.outliers:].view(self.outliers, inp.shape[1]).copy_(ao.to(torch.uint8))
        qinput_scale[-self.outliers:].copy_(ao_scale.squeeze(1))

        return qinput, qinput_scale, inp_outlier_idx

    def forward(self, inp):
        output = torch.empty(inp.shape[0] + self.outliers, self.w.shape[0] - self.outliers, dtype=torch.bfloat16, device=inp.device)

        qinput, qinput_scale, inp_outlier_idx = self.act_quantize(inp)

        gemm_int4_int4_nt_mixed_precision_naive(qinput, qinput_scale, self.w, self.w_scale, output, self.outliers)

        output[inp_outlier_idx] = output[-self.outliers:].clone()
        output[:, self.w_idx] = output[:, -self.outliers:]
        ret = output[:-self.outliers, :-self.outliers]

        return ret

class OptMixedQLinear(MixedQLinear):
    def __init__(self, *args, **kwargs):
        multistream = kwargs.pop("multistream", False)
        unifiedkernel = kwargs.pop("unifiedkernel", False)
        super().__init__(*args, **kwargs)
        self.streams = [torch.cuda.Stream() for _ in range(6)]
        self.quant_method = self.act_quantize
        self.kernel = gemm_int4_int4_nt_mixed_precision_separate
        if multistream:
            self.forward = self.forward_multistream
            self.quant_method = self.act_quantize_multistream
        if unifiedkernel:
            assert not multistream, "Unified kernel does not support multistream"
            self.kernel = lambda *args: gemm_int4_int4_nt_mixed_precision(*args[:-1])

    def act_quantize_multistream(self, inp, outliers):
        assert inp.shape[1] % 2 == 0

        with torch.cuda.nvtx.range("act-quant"):
            pertoken_max, inp_outlier_idx = mat_topk(inp, outliers)
        
        e0 = torch.cuda.current_stream().record_event()
        self.streams[0].wait_event(e0)
        self.streams[1].wait_event(e0)
        
        with torch.cuda.stream(self.streams[0]):
            with torch.cuda.nvtx.range("act-quant"):
                packed_an = quantize_pack(inp, pertoken_max)
                an_scale = (pertoken_max / 7)
        e_finish_normal = self.streams[0].record_event()

        with torch.cuda.stream(self.streams[1]):
            with torch.cuda.nvtx.range("act-quant-outliers"):
                ao, ao_scale = pertoken_quantize(inp[inp_outlier_idx], 8)
                ao, ao_scale = ao.to(torch.uint8), ao_scale.squeeze(1)
        e_finish_outliers = self.streams[1].record_event()

        return packed_an, an_scale, ao, ao_scale, inp_outlier_idx, e_finish_normal, e_finish_outliers

    def forward_multistream(self, inp):
        torch.cuda.set_device(inp.device)
        a_outliers = math.ceil(inp.shape[0] * self.p_a_outlier / 256) * 256
        output = torch.empty(inp.shape[0] + a_outliers, self.w.shape[0] - self.w_outliers, dtype=torch.bfloat16, device=inp.device)

        act, act_scale, act_outlier, act_scale_outlier, inp_outlier_idx, e_finish_normal, e_finish_outliers = self.act_quantize_multistream(inp, a_outliers)

        self.streams[2].wait_event(e_finish_normal)
        self.streams[3].wait_event(e_finish_outliers)
        self.streams[4].wait_event(e_finish_normal)
        self.streams[5].wait_event(e_finish_outliers)

        with torch.cuda.nvtx.range("gemm"):
            es_finish = gemm_int4_int4_nt_mixed_precision_separate(
                act, act_scale,
                act_outlier, act_scale_outlier,
                self.w[:-2 * self.w_outliers], self.w_scale[:-self.w_outliers],
                self.w[-2 * self.w_outliers:].view(self.w_outliers, -1), self.w_scale[-self.w_outliers:],
                output, a_outliers, self.w_outliers,
                self.streams[2:]
            )
        
        for e in es_finish:
            torch.cuda.current_stream().wait_event(e)

        with torch.cuda.nvtx.range("post-reorder"):
            output[inp_outlier_idx] = output[-a_outliers:].clone()
            output[:, self.w_idx] = output[:, -self.w_outliers:]
            ret = output[:-a_outliers, :-self.w_outliers]
        return ret
    
    def act_quantize(self, inp, outliers):
        assert inp.shape[1] % 2 == 0
        with torch.cuda.nvtx.range("act-quant"):
            pertoken_max, inp_outlier_idx = mat_topk(inp, outliers)
        with torch.cuda.nvtx.range("act-quant"):
            packed_an = quantize_pack(inp, pertoken_max)
            an_scale = (pertoken_max / 7)
        with torch.cuda.nvtx.range("act-quant-outliers"):
            ao, ao_scale = pertoken_quantize(inp[inp_outlier_idx], 8)
            ao, ao_scale = ao.to(torch.uint8), ao_scale.squeeze(1)
        return packed_an, an_scale, ao, ao_scale, inp_outlier_idx

    def forward(self, inp):
        torch.cuda.set_device(inp.device)
        a_outliers = math.ceil(inp.shape[0] * self.p_a_outlier / 256) * 256
        output = torch.empty(inp.shape[0] + a_outliers, self.w.shape[0] - self.w_outliers, dtype=torch.bfloat16, device=inp.device)
        act, act_scale, act_outlier, act_scale_outlier, inp_outlier_idx = self.act_quantize(inp, a_outliers)
        with torch.cuda.nvtx.range("gemm"):
            self.kernel(
                act, act_scale,
                act_outlier, act_scale_outlier,
                self.w[:-2 * self.w_outliers], self.w_scale[:-self.w_outliers],
                self.w[-2 * self.w_outliers:].view(self.w_outliers, -1), self.w_scale[-self.w_outliers:],
                output, a_outliers, self.w_outliers,
                [torch.cuda.current_stream() for _ in range(4)]
            )
        with torch.cuda.nvtx.range("post-reorder"):
            output[inp_outlier_idx] = output[-a_outliers:].clone()
            output[:, self.w_idx] = output[:, -self.w_outliers:]
            ret = output[:-a_outliers, :-self.w_outliers]
        return ret
    
    from ..profile import profile_latency as default_profile_latency
    
    def benchmark(self, x, profile_latency=default_profile_latency):
        assert self.forward != self.forward_multistream
        output = torch.empty(x.shape[0] + self.outliers, self.w.shape[0] - self.outliers, dtype=torch.bfloat16, device=x.device)
        time_act_quant = profile_latency(lambda: self.act_quantize(x))
        act, act_scale, act_outlier, act_scale_outlier, inp_outlier_idx = self.act_quantize(x)
        time_gemm = profile_latency(lambda: gemm_int4_int4_nt_mixed_precision_separate(
            act, act_scale, act_outlier, act_scale_outlier,
            self.w[:-2 * self.outliers], self.w_scale[:-self.outliers],
            self.w[-2 * self.outliers:].view(self.outliers, -1), self.w_scale[-self.outliers:],
            output, self.outliers, [torch.cuda.current_stream() for _ in range(4)])
        )
        def post_reorder():
            output[inp_outlier_idx] = output[-self.outliers:].clone()
            output[:, self.w_idx] = output[:, -self.outliers:]
            ret = output[:-self.outliers, :-self.outliers]
            return ret
        time_post_reorder = profile_latency(post_reorder)
        print(f"Act-quant: {time_act_quant:.2f} us | Gemm: {time_gemm:.2f} us | Post: {time_post_reorder:.2f} us", end=" | ")
        return time_act_quant + time_gemm + time_post_reorder

class QLinear(torch.nn.Module):
    def __init__(self, weight, bits): # weight is in transposed format
        super().__init__()

        w, w_scale = pertoken_quantize(weight, bits)

        self.register_buffer('w', w)
        self.register_buffer('w_scale', w_scale)
        self.bits = bits

    def forward(self, inp):
        a, a_scale = pertoken_quantize(inp, self.bits)
        return ((a.to(torch.float32) * a_scale.view(-1, 1)) @ (self.w.to(torch.float32) * self.w_scale.view(-1, 1)).t()).to(torch.float16)
