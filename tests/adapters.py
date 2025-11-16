from __future__ import annotations

from typing import Type

import torch

from torch.autograd import Function

# ----------------------------------------------------------------------------
# FlashAttention autograd.Function (PyTorch 实现)
# ----------------------------------------------------------------------------
class FlashAttentionPytorch(Function):
    """
    使用分块、重计算和在线 Softmax 实现 FlashAttention (纯 PyTorch)。
    这遵循了 FlashAttention 论文 (Algorithm 1 和 2)。
    """

    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        """
        正向传播 (Algorithm 1)
        
        我们按块计算 O，并为每个 Q 块 (Qi) 维护“在线”统计量 (m_i, l_i)。
        """
        batch_size, n_queries, d = q.shape
        _, n_keys, _ = k.shape
        scale = 1 / (d ** 0.5)

        # O 和 L (logsumexp)
        o = torch.zeros_like(q)
        # L 是测试代码所期望的 logsumexp(S)
        # 我们将在计算过程中跟踪 m_i (max) 和 l_i (sum)，
        # 并在最后计算 L = m_i + log(l_i)
        lse = torch.full((batch_size, n_queries), -float('inf'), device=q.device)

        # 块大小 (Tiling)
        # 在实际的 CUDA 内核中，这些块大小会根据 SRAM 仔细调整
        # 这里我们选择一个大小来演示分块逻辑 (N=128, M=128)
        BLOCK_N = 64  # Q 的块大小 (行)
        BLOCK_M = 64  # K/V 的块大小 (列)

        # 遍历 Q 的块 (外部循环)
        for i in range(0, n_queries, BLOCK_N):
            n_start = i
            n_end = min(i + BLOCK_N, n_queries)
            
            # 当前 Q 块
            q_i = q[..., n_start:n_end, :] # (B, BLOCK_N, D)

            # 在线统计量 (Online stats)
            m_i = torch.full((batch_size, n_end - n_start), -float('inf'), device=q.device)
            l_i = torch.zeros((batch_size, n_end - n_start), device=q.device)
            o_i = torch.zeros_like(q_i)

            # 遍历 K/V 的块 (内部循环)
            for j in range(0, n_keys, BLOCK_M):
                m_start = j
                m_end = min(j + BLOCK_M, n_keys)

                # 当前 K, V 块
                k_j = k[..., m_start:m_end, :] # (B, BLOCK_M, D)
                v_j = v[..., m_start:m_end, :] # (B, BLOCK_M, D)

                # 1. 计算 S_ij = Q_i @ K_j^T * scale
                s_ij = torch.einsum('...nd, ...md -> ...nm', q_i, k_j) * scale

                # 2. 应用因果掩码 (如果需要)
                if is_causal:
                    # 我们需要全局索引来进行掩码
                    global_rows = torch.arange(n_start, n_end, device=q.device)[None, :, None]
                    global_cols = torch.arange(m_start, m_end, device=q.device)[None, None, :]
                    mask = global_rows >= global_cols
                    s_ij = torch.where(mask, s_ij, -1e6)

                # 3. 计算在线 softmax 统计量 (m_ij, p_ij, l_ij)
                m_ij = torch.max(s_ij, dim=-1)[0] # (B, BLOCK_N)
                p_ij = torch.exp(s_ij - m_ij[..., None]) # (B, BLOCK_N, BLOCK_M)
                l_ij = torch.sum(p_ij, dim=-1) # (B, BLOCK_N)
                
                # 4. 更新旧的统计量 (m_i, l_i, o_i)
                m_i_old = m_i
                l_i_old = l_i

                # m_i_new = max(m_i_old, m_ij)
                m_i = torch.maximum(m_i_old, m_ij)

                # 计算缩放因子
                # scale_old = exp(m_i_old - m_i_new)
                # scale_new = exp(m_ij - m_i_new)
                scale_old = torch.exp(m_i_old - m_i)
                scale_new = torch.exp(m_ij - m_i)

                # l_i_new = scale_old * l_i_old + scale_new * l_ij
                l_i = scale_old * l_i_old + scale_new * l_ij

                # O_i_new = ( (scale_old * l_i_old) / l_i_new ) * O_i_old + 
                #           ( (scale_new * l_ij) / l_i_new ) * (P_ij @ V_j)
                #
                # 为避免除法，我们先计算分子 (un-normalized O_i)，
                # 最后再除以 l_i
                o_i_scaled_old = o_i * scale_old[..., None]
                o_i_scaled_new = torch.einsum('...nm, ...md -> ...nd', p_ij, v_j) * scale_new[..., None]
                
                o_i = o_i_scaled_old + o_i_scaled_new

            # 内部循环结束 (j)
            # 5. 存储最终的 o_i 和 lse_i
            # O_i = O_i (un-normalized) / l_i
            o[..., n_start:n_end, :] = o_i / l_i[..., None]
            
            # LSE_i = m_i + log(l_i)
            lse[..., n_start:n_end] = m_i + torch.log(l_i)

        # 外部循环结束 (i)
        
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.scale = scale
        ctx.is_causal = is_causal
        ctx.BLOCK_N = BLOCK_N
        ctx.BLOCK_M = BLOCK_M
        
        return o

    @staticmethod
    def backward(ctx, do):
        """
        反向传播 (Algorithm 2)

        我们从 Q 块开始循环，并在内部循环 K/V 块。
        在内部循环中，我们 *重计算* P_ij 和 S_ij。
        """
        q, k, v, o, lse = ctx.saved_tensors
        scale = ctx.scale
        is_causal = ctx.is_causal
        BLOCK_N = ctx.BLOCK_N
        BLOCK_M = ctx.BLOCK_M

        batch_size, n_queries, d = q.shape
        _, n_keys, _ = k.shape

        # 初始化梯度
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)

        # 遍历 Q 的块 (外部循环)
        for i in range(0, n_queries, BLOCK_N):
            n_start = i
            n_end = min(i + BLOCK_N, n_queries)

            # 获取当前块
            q_i = q[..., n_start:n_end, :]   # (B, BLOCK_N, D)
            o_i = o[..., n_start:n_end, :]   # (B, BLOCK_N, D)
            do_i = do[..., n_start:n_end, :] # (B, BLOCK_N, D)
            lse_i = lse[..., n_start:n_end] # (B, BLOCK_N)
            
            # 计算 D_i = sum(dO_i * O_i)
            d_i = torch.einsum('...nd, ...nd -> ...n', do_i, o_i) # (B, BLOCK_N)

            # dQ_i 在内部循环中累加
            dq_i = torch.zeros_like(q_i)

            # 遍历 K/V 的块 (内部循环)
            for j in range(0, n_keys, BLOCK_M):
                m_start = j
                m_end = min(j + BLOCK_M, n_keys)

                # 获取当前块
                k_j = k[..., m_start:m_end, :] # (B, BLOCK_M, D)
                v_j = v[..., m_start:m_end, :] # (B, BLOCK_M, D)

                # --- 重计算 (Recomputation) ---
                # 1. 重计算 S_ij
                s_ij = torch.einsum('...nd, ...md -> ...nm', q_i, k_j) * scale

                # 2. 应用因果掩码 (必须与前向传播完全一致)
                if is_causal:
                    global_rows = torch.arange(n_start, n_end, device=q.device)[None, :, None]
                    global_cols = torch.arange(m_start, m_end, device=q.device)[None, None, :]
                    mask = global_rows >= global_cols
                    s_ij = torch.where(mask, s_ij, -1e6)

                # 3. 重计算 P_ij = exp(S_ij - LSE_i)
                p_ij = torch.exp(s_ij - lse_i[..., None]) # (B, BLOCK_N, BLOCK_M)
                # --- 重计算结束 ---

                # --- 计算梯度 ---
                
                # 4. 计算 dV_j = P_ij^T @ dO_i
                dv_j = torch.einsum('...nm, ...nd -> ...md', p_ij, do_i)
                # 累加到 dv (dK 和 dV 在外部循环中累加)
                dv[..., m_start:m_end, :] += dv_j

                # 5. 计算 dP_ij = dO_i @ V_j^T
                dp_ij = torch.einsum('...nd, ...md -> ...nm', do_i, v_j) # (B, BLOCK_N, BLOCK_M)

                # 6. 计算 dS_ij = P_ij * (dP_ij - D_i)
                ds_ij = p_ij * (dp_ij - d_i[..., None]) # (B, BLOCK_N, BLOCK_M)
                
                # 7. 计算 dQ_i = dS_ij @ K_j
                # 累加到 dq_i (dQ 在内部循环中累加)
                dq_i += torch.einsum('...nm, ...md -> ...nd', ds_ij, k_j) * scale

                # 8. 计算 dK_j = dS_ij^T @ Q_i
                dk_j = torch.einsum('...nm, ...nd -> ...md', ds_ij, q_i) * scale
                # 累加到 dk (dK 和 dV 在外部循环中累加)
                dk[..., m_start:m_end, :] += dk_j
            
            # 内部循环结束 (j)
            # 存储 dQ_i
            dq[..., n_start:n_end, :] = dq_i

        # 外部循环结束 (i)

        return dq, dk, dv, None

class NormalAttentionPytorch(Function):
    """
    NormalAttention 的 PyTorch autograd.Function 实现。
    """

    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        """
        正向传播实现。
        计算 O = softmax( (QK^T) / sqrt(d) ) @ V
        并保存 L = logsumexp( (QK^T) / sqrt(d) ) 以供测试。
        """
        n_queries = q.shape[-2]
        n_keys = k.shape[-2]
        d = q.shape[-1]
        scale = 1 / (d ** 0.5)

        # 1. 计算 S = QK^T * scale
        S = torch.einsum('...qd, ...kd -> ...qk', q, k) * scale

        # 2. 应用因果掩码 (causal mask)
        if is_causal:
            # 完全按照 _attention_and_lse 中的参考逻辑
            S = torch.where(
                torch.arange(n_queries, device=S.device)[None, :, None] >= torch.arange(n_keys, device=S.device)[None, None, :],
                S,
                -1e6  # 使用一个足够大的负数
            )

        # 3. 计算 P = softmax(S)
        P = torch.softmax(S, dim=-1)

        # 4. 计算 L = logsumexp(S) (用于测试)
        # 测试代码 _test_flash_forward_pass 会检查这个 L
        L = torch.logsumexp(S, dim=-1) # 形状: (batch, n_queries)

        # 5. 计算 O = PV
        o = torch.einsum('...qk, ...kd -> ...qd', P, v)

        # 6. 保存反向传播所需的张量
        # 测试要求 L 是唯一一个形状为 (B, Q) 的张量
        ctx.save_for_backward(q, k, v, o, P, L)
        ctx.scale = scale
        
        return o

    @staticmethod
    def backward(ctx, do):
        """
        反向传播实现。
        根据 dO (do) 计算 dQ, dK, dV。
        """
        # 1. 检索保存的张量
        # L 被保存了以通过前向测试，但反向传播不需要它
        q, k, v, o, P, L = ctx.saved_tensors
        scale = ctx.scale

        # 2. 计算 dV = P^T * dO
        # P: (...q, k), dO: (...q, d) -> dV: (...k, d)
        dv = torch.einsum('...qk, ...qd -> ...kd', P, do)

        # 3. 计算 dP = dO * V^T
        # dO: (...q, d), V: (...k, d) -> dP: (...q, k)
        dP = torch.einsum('...qd, ...kd -> ...qk', do, v)

        # 4. 计算 dS = P * (dP - sum(dP * P))
        # 这是 softmax 的标准梯度
        D_row = torch.sum(dP * P, dim=-1, keepdim=True) # 形状: (...q, 1)
        dS = P * (dP - D_row)
        dS1 = dP * P - torch.sum(dP * P, dim=-1, keepdim=True) * P
        assert torch.allclose(dS, dS1), "两种计算方式不一致"

        # 5. 反向传播 dS 到 dQ 和 dK
        # S = (Q @ K^T) * scale
        
        # dQ = (dS @ K) * scale
        # dS: (...q, k), K: (...k, d) -> dQ: (...q, d)
        dq = torch.einsum('...qk, ...kd -> ...qd', dS, k) * scale

        # dK = (dS^T @ Q) * scale
        # dS: (...q, k), Q: (...q, d) -> dK: (...k, d)
        # (dS^T @ Q) 等价于在 torch.einsum 中交换 qk
        dk = torch.einsum('...qk, ...qd -> ...kd', dS, q) * scale

        # 对应 forward 的输入 (q, k, v, is_causal)
        return dq, dk, dv, None

def get_flashattention_autograd_function_pytorch() -> Type:
    """
    Returns a torch.autograd.Function subclass that implements FlashAttention2.
    The expectation is that this class will implement FlashAttention2
    using only standard PyTorch operations (no Triton!).

    Returns:
        A class object (not an instance of the class)
    """
    # For example: return MyFlashAttnAutogradFunctionClass

    # return NormalAttentionPytorch
    return FlashAttentionPytorch

    # raise NotImplementedError


import torch
import triton
import triton.language as tl
from torch.autograd import Function
# ----------------------------------------------------------------------------
# Triton Kernels
# ----------------------------------------------------------------------------
@triton.jit
def _flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    LSE_ptr,
    stride_q_b, stride_q_h, stride_q_n, stride_q_d,
    stride_k_b, stride_k_h, stride_k_n, stride_k_d,
    stride_v_b, stride_v_h, stride_v_n, stride_v_d,
    stride_o_b, stride_o_h, stride_o_n, stride_o_d,
    stride_lse_b, stride_lse_h, stride_lse_n,
    B, H, N_q, N_k,
    scale,
    IS_CAUSAL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    D_HEAD: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N))
    offs_d = tl.arange(0, D_HEAD)
    
    # Q Pointer & Load
    q_ptrs = Q_ptr + (pid_b * stride_q_b + pid_h * stride_q_h + 
                      offs_n[:, None] * stride_q_n + 
                      offs_d[None, :] * stride_q_d)
    mask_n = (offs_n < N_q)[:, None]
    q_i = tl.load(q_ptrs, mask=mask_n, other=0.0)

    # Init accumulators
    m_i = tl.full([BLOCK_N], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_N], dtype=tl.float32)
    o_i = tl.zeros([BLOCK_N, D_HEAD], dtype=tl.float32)

    for j in range(0, N_k, BLOCK_M):
        start_m = j
        offs_m = (start_m + tl.arange(0, BLOCK_M))
        
        # K/V Pointers (Transposed layout for K handled via strides)
        # We load K as (BLOCK_M, D_HEAD)
        k_ptrs = K_ptr + (pid_b * stride_k_b + pid_h * stride_k_h + 
                          offs_m[:, None] * stride_k_n + 
                          offs_d[None, :] * stride_k_d)
        v_ptrs = V_ptr + (pid_b * stride_v_b + pid_h * stride_v_h + 
                          offs_m[:, None] * stride_v_n + 
                          offs_d[None, :] * stride_v_d)

        mask_m = (offs_m < N_k) # (BLOCK_M,)
        
        k_j = tl.load(k_ptrs, mask=mask_m[:, None], other=0.0)
        v_j = tl.load(v_ptrs, mask=mask_m[:, None], other=0.0)
        
        # 1. Score Calculation
        # q_i (N, D) @ k_j.T (D, M) -> (N, M)
        s_ij = (tl.dot(q_i, k_j.T) * scale)
        
        # 2. Masking (Causal + Padding)
        # Crucial Fix: Mask out padding columns explicitly
        s_ij = tl.where(offs_m[None, :] < N_k, s_ij, -1e6)
        
        if IS_CAUSAL:
            causal_mask = (offs_n[:, None] >= offs_m[None, :])
            s_ij = tl.where(causal_mask, s_ij, -1e6)

        # 3. Online Softmax
        m_ij = tl.max(s_ij, axis=1)
        p_ij = tl.exp(s_ij - m_ij[:, None])
        l_ij = tl.sum(p_ij, axis=1)
        
        m_i_new = tl.maximum(m_i, m_ij)
        scale_old = tl.exp(m_i - m_i_new)
        scale_new = tl.exp(m_ij - m_i_new)

        l_i_new = scale_old * l_i + scale_new * l_ij
        
        o_i_scaled_old = o_i * scale_old[:, None]
        o_i_scaled_new = tl.dot(p_ij, v_j) * scale_new[:, None]
        o_i = o_i_scaled_old + o_i_scaled_new
        
        m_i = m_i_new
        l_i = l_i_new

    # Finalize
    o_i = o_i / l_i[:, None]
    lse_i = m_i + tl.log(l_i)

    # Write Output
    o_ptrs = O_ptr + (pid_b * stride_o_b + pid_h * stride_o_h + 
                      offs_n[:, None] * stride_o_n + 
                      offs_d[None, :] * stride_o_d)
    lse_ptrs = LSE_ptr + (pid_b * stride_lse_b + pid_h * stride_lse_h + offs_n)
    
    tl.store(o_ptrs, o_i, mask=mask_n)
    tl.store(lse_ptrs, lse_i, mask=offs_n < N_q)

@triton.jit
def _flash_bwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr, DO_ptr,
    DQ_ptr, DK_ptr, DV_ptr,
    stride_q_b, stride_q_h, stride_q_n, stride_q_d,
    stride_k_b, stride_k_h, stride_k_n, stride_k_d,
    stride_v_b, stride_v_h, stride_v_n, stride_v_d,
    stride_o_b, stride_o_h, stride_o_n, stride_o_d,
    stride_lse_b, stride_lse_h, stride_lse_n,
    stride_do_b, stride_do_h, stride_do_n, stride_do_d,
    stride_dq_b, stride_dq_h, stride_dq_n, stride_dq_d,
    stride_dk_b, stride_dk_h, stride_dk_n, stride_dk_d,
    stride_dv_b, stride_dv_h, stride_dv_n, stride_dv_d,
    B, H, N_q, N_k,
    scale,
    IS_CAUSAL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    D_HEAD: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N))
    offs_d = tl.arange(0, D_HEAD)
    mask_n = (offs_n < N_q)[:, None]

    # Pointers for Q, O, DO, LSE
    q_ptrs = Q_ptr + (pid_b * stride_q_b + pid_h * stride_q_h + 
                      offs_n[:, None] * stride_q_n + 
                      offs_d[None, :] * stride_q_d)
    o_ptrs = O_ptr + (pid_b * stride_o_b + pid_h * stride_o_h +
                      offs_n[:, None] * stride_o_n +
                      offs_d[None, :] * stride_o_d)
    do_ptrs = DO_ptr + (pid_b * stride_do_b + pid_h * stride_do_h +
                       offs_n[:, None] * stride_do_n +
                       offs_d[None, :] * stride_do_d)
    lse_ptrs = LSE_ptr + (pid_b * stride_lse_b + pid_h * stride_lse_h + offs_n)
    
    q_i = tl.load(q_ptrs, mask=mask_n, other=0.0)
    o_i = tl.load(o_ptrs, mask=mask_n, other=0.0)
    do_i = tl.load(do_ptrs, mask=mask_n, other=0.0)
    lse_i = tl.load(lse_ptrs, mask=(offs_n < N_q), other=0.0)

    # D_i = sum(dO_i * O_i)
    d_i = tl.sum(do_i * o_i, axis=1)
    
    dq_i = tl.zeros([BLOCK_N, D_HEAD], dtype=tl.float32)

    for j in range(0, N_k, BLOCK_M):
        start_m = j
        offs_m = (start_m + tl.arange(0, BLOCK_M))
        mask_m = (offs_m < N_k) # (BLOCK_M,)

        k_ptrs = K_ptr + (pid_b * stride_k_b + pid_h * stride_k_h + 
                          offs_m[:, None] * stride_k_n + 
                          offs_d[None, :] * stride_k_d)
        v_ptrs = V_ptr + (pid_b * stride_v_b + pid_h * stride_v_h + 
                          offs_m[:, None] * stride_v_n + 
                          offs_d[None, :] * stride_v_d)
        
        k_j = tl.load(k_ptrs, mask=mask_m[:, None], other=0.0)
        v_j = tl.load(v_ptrs, mask=mask_m[:, None], other=0.0)

        # Recompute S_ij
        s_ij = (tl.dot(q_i, k_j.T) * scale)
        
        # Masking (Must match forward exactly)
        s_ij = tl.where(offs_m[None, :] < N_k, s_ij, -1e6)
        
        if IS_CAUSAL:
            causal_mask = (offs_n[:, None] >= offs_m[None, :])
            s_ij = tl.where(causal_mask, s_ij, -1e6)

        p_ij = tl.exp(s_ij - lse_i[:, None])

        # Gradients
        # dv_j = p_ij.T @ do_i
        dv_j = tl.dot(p_ij.T, do_i)
        
        # dp_ij = do_i @ v_j.T
        dp_ij = tl.dot(do_i, v_j.T)
        
        # ds_ij = p_ij * (dp_ij - D_i)
        ds_ij = p_ij * (dp_ij - d_i[:, None])

        # dq_i += ds_ij @ k_j
        dq_i += (tl.dot(ds_ij, k_j) * scale)
        
        # dk_j = ds_ij.T @ q_i
        dk_j = (tl.dot(ds_ij.T, q_i) * scale)

        # Atomic add for dK and dV
        dv_ptrs = DV_ptr + (pid_b * stride_dv_b + pid_h * stride_dv_h + 
                            offs_m[:, None] * stride_dv_n + 
                            offs_d[None, :] * stride_dv_d)
        dk_ptrs = DK_ptr + (pid_b * stride_dk_b + pid_h * stride_dk_h + 
                            offs_m[:, None] * stride_dk_n + 
                            offs_d[None, :] * stride_dk_d)
        
        tl.atomic_add(dv_ptrs, dv_j, mask=mask_m[:, None])
        tl.atomic_add(dk_ptrs, dk_j, mask=mask_m[:, None])

    # Write dQ
    dq_ptrs = DQ_ptr + (pid_b * stride_dq_b + pid_h * stride_dq_h + 
                      offs_n[:, None] * stride_dq_n + 
                      offs_d[None, :] * stride_dq_d)
    tl.store(dq_ptrs, dq_i, mask=mask_n)

class FlashAttentionTriton(Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal):
        # q, k, v are (B, N, D) -> View as (B, H=1, N, D)
        B, N_q, D_HEAD = q.shape
        _, N_k, _ = k.shape
        H = 1 
        
        scale = D_HEAD ** -0.5
        
        # View as 4D for kernel
        q_4d = q.view(B, H, N_q, D_HEAD)
        k_4d = k.view(B, H, N_k, D_HEAD)
        v_4d = v.view(B, H, N_k, D_HEAD)
        
        o = torch.empty_like(q_4d)
        # LSE shape (B, H, N_q)
        lse = torch.empty((B, H, N_q), device='cuda', dtype=torch.float32)

        BLOCK_N = 16
        BLOCK_M = 16
        grid = (triton.cdiv(N_q, BLOCK_N), B * H)
        
        _flash_fwd_kernel[grid](
            q_4d, k_4d, v_4d, o, lse,
            q_4d.stride(0), q_4d.stride(1), q_4d.stride(2), q_4d.stride(3),
            k_4d.stride(0), k_4d.stride(1), k_4d.stride(2), k_4d.stride(3),
            v_4d.stride(0), v_4d.stride(1), v_4d.stride(2), v_4d.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            B, H, N_q, N_k,
            scale,
            IS_CAUSAL=is_causal,
            BLOCK_N=BLOCK_N,
            BLOCK_M=BLOCK_M,
            D_HEAD=D_HEAD
        )
        
        # Squeeze LSE to (B, N_q) for the test assertion
        lse_saved = lse.squeeze(1)
        
        ctx.save_for_backward(q_4d, k_4d, v_4d, o, lse_saved)
        ctx.scale = scale
        ctx.is_causal = is_causal
        ctx.BLOCK_N = BLOCK_N
        ctx.BLOCK_M = BLOCK_M
        ctx.D_HEAD = D_HEAD
        
        return o.view(B, N_q, D_HEAD)

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        
        # Restore LSE to 3D: (B, N) -> (B, 1, N)
        lse = lse.unsqueeze(1)
        
        scale = ctx.scale
        is_causal = ctx.is_causal
        BLOCK_N = ctx.BLOCK_N
        BLOCK_M = ctx.BLOCK_M
        D_HEAD = ctx.D_HEAD

        B, H, N_q, D_HEAD = q.shape
        _, _, N_k, _ = k.shape
        
        do = do.view(B, H, N_q, D_HEAD)
        
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        
        grid = (triton.cdiv(N_q, BLOCK_N), B * H)
        
        _flash_bwd_kernel[grid](
            q, k, v, o, lse, do,
            dq, dk, dv,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
            dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
            dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
            B, H, N_q, N_k,
            scale,
            IS_CAUSAL=is_causal,
            BLOCK_N=BLOCK_N,
            BLOCK_M=BLOCK_M,
            D_HEAD=D_HEAD
        )
        
        return dq.view(B, N_q, D_HEAD), dk.view(B, N_k, D_HEAD), dv.view(B, N_k, D_HEAD), None

def get_flashattention_autograd_function_triton() -> Type:
    """
    Returns a torch.autograd.Function subclass that implements FlashAttention2
    using Triton kernels.
    The expectation is that this class will implement the same operations
    as the class you return in get_flashattention_autograd_function_pytorch(),
    but it should do so by invoking custom Triton kernels in the forward
    and backward passes.

    Returns:
        A class object (not an instance of the class)
    """
    # For example: return MyTritonFlashAttentionAutogradFunctionClass

    return FlashAttentionTriton
    # raise NotImplementedError

import torch
import torch.distributed as dist
import torch.nn as nn

class DDPIndividualParameters(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        # 存储结构改为列表，保存 (handle, reduced_grad_tensor, param_reference)
        self.handles = []
        
        # 1. 初始化广播
        self._broadcast_parameters_and_buffers()
        
        # 2. 注册 Hook
        self._register_hooks()

    def _broadcast_parameters_and_buffers(self):
        src_rank = 0
        for tensor in self.module.state_dict().values():
            dist.broadcast(tensor, src=src_rank)

    def _register_hooks(self):
        world_size = dist.get_world_size()
        
        # 我们需要将 param 传给 hook 生成器，以便知道梯度属于哪个参数
        def get_hook(param):
            def hook(grad):
                # 1. Clone 梯度。
                # 我们必须使用副本进行通信，因为 PyTorch 会在 hook 返回后立即读取原始 grad 
                # 进行累加。如果我们在原始 grad 上做异步 AllReduce，会产生竞态条件。
                grad_clone = grad.clone().detach()
                
                # 2. 预处理：除以 world_size (求平均)
                grad_clone.div_(world_size)
                
                # 3. 发起异步 AllReduce (Sum)
                # 结果会直接写回 grad_clone
                handle = dist.all_reduce(grad_clone, op=dist.ReduceOp.SUM, async_op=True)
                
                # 4. 保存句柄、缓冲区副本和参数引用
                self.handles.append((handle, grad_clone, param))
                
                # 返回原始 grad 让 PyTorch 继续它本地的逻辑（虽然我们稍后会覆盖它）
                return grad
            return hook

        for param in self.module.parameters():
            if param.requires_grad:
                param.register_hook(get_hook(param))

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        """
        等待通信完成，并将全局梯度应用到 param.grad
        """
        # 用于处理 Tied Weights（共享权重）：一个参数可能有多个梯度贡献
        # Map: param -> list of gradients
        param_grads = {}
        
        # 1. 等待所有异步操作完成
        for handle, reduced_grad, param in self.handles:
            handle.wait()
            
            if param not in param_grads:
                param_grads[param] = []
            param_grads[param].append(reduced_grad)
            
        # 2. 聚合梯度并覆盖 param.grad
        for param, grads in param_grads.items():
            # 如果有共享权重，将所有 reduce 后的梯度加起来
            final_grad = grads[0]
            for g in grads[1:]:
                final_grad.add_(g)
            
            # 关键步骤：直接覆盖 param.grad
            # 这丢弃了 PyTorch 本地累加的中间值，使用了我们计算出的精确全局平均值
            param.grad = final_grad
            
        # 3. 清理
        self.handles = []

def get_ddp_individual_parameters(module: torch.nn.Module) -> torch.nn.Module:
    """
    Returns a torch.nn.Module container that handles
    parameter broadcasting and gradient synchronization for
    distributed data parallel training.

    This container should overlaps communication with backprop computation
    by asynchronously communicating gradients as they are ready
    in the backward pass. The gradient for each parameter tensor
    is individually communicated.

    Args:
        module: torch.nn.Module
            Underlying model to wrap with DDP.
    Returns:
        Instance of a DDP class.
    """
    # For example: return DDPIndividualParameters(module)

    return DDPIndividualParameters(module)
    raise NotImplementedError

def ddp_individual_parameters_on_after_backward(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """
    Code to run after the backward pass is completed, but before we take
    an optimizer step.

    Args:
        ddp_model: torch.nn.Module
            DDP-wrapped model.
        optimizer: torch.optim.Optimizer
            Optimizer being used with the DDP-wrapped model.
    """
    # For example: ddp_model.finish_gradient_synchronization()
    
    if isinstance(ddp_model, DDPIndividualParameters):
        ddp_model.finish_gradient_synchronization()
    return
    raise NotImplementedError


import torch
import torch.distributed as dist
import torch.nn as nn
from typing import List, Optional

class DDPBucketed(nn.Module):
    def __init__(self, module: nn.Module, bucket_size_mb: float):
        super().__init__()
        self.module = module
        self.bucket_size_mb = bucket_size_mb
        self.buckets = []
        self.param_to_bucket_info = {}
        
        self._broadcast_parameters_and_buffers()
        self._init_buckets()
        self._register_hooks()

    def _broadcast_parameters_and_buffers(self):
        # 确保所有 rank 的初始权重一致，否则训练必定发散
        src_rank = 0
        for tensor in self.module.state_dict().values():
            dist.broadcast(tensor, src=src_rank)

    def _init_buckets(self):
        params = [p for p in self.module.parameters() if p.requires_grad]
        # 按照 backward 执行顺序（逆序）排列，有助于重叠通信与计算
        params.reverse()
        
        current_bucket_params = []
        current_bucket_size = 0
        target_bucket_bytes = self.bucket_size_mb * 1024 * 1024

        for param in params:
            param_bytes = param.numel() * param.element_size()
            # 简单的分桶逻辑
            if current_bucket_size + param_bytes > target_bucket_bytes and current_bucket_params:
                self._create_bucket(current_bucket_params)
                current_bucket_params = []
                current_bucket_size = 0
            current_bucket_params.append(param)
            current_bucket_size += param_bytes

        if current_bucket_params:
            self._create_bucket(current_bucket_params)

    def _create_bucket(self, params):
        total_numel = sum(p.numel() for p in params)
        dtype = params[0].dtype
        device = params[0].device
        
        # 创建扁平 Bucket
        bucket_tensor = torch.zeros(total_numel, dtype=dtype, device=device)
        
        bucket_idx = len(self.buckets)
        current_offset = 0
        
        for param in params:
            numel = param.numel()
            self.param_to_bucket_info[param] = (bucket_idx, current_offset, numel)
            current_offset += numel
            
        self.buckets.append({
            "tensor": bucket_tensor,
            "params": params,
            "handle": None,
            "ready_params": 0,
            "total_params": len(params)
        })

    def _prepare_for_backward(self):
        # 每个 Batch 开始前调用，重置计数器和 Tensor
        for bucket in self.buckets:
            bucket["ready_params"] = 0
            bucket["handle"] = None
            bucket["tensor"].zero_() 

    def _register_hooks(self):
        world_size = dist.get_world_size()
        
        def get_hook(param):
            # 使用闭包捕获 param
            def hook(grad):
                # 1. 查找位置
                if param not in self.param_to_bucket_info:
                    # 防御性编程：如果 param 虽然 requires_grad 但未被归入 bucket（极少见）
                    return grad
                    
                bucket_idx, offset, numel = self.param_to_bucket_info[param]
                bucket = self.buckets[bucket_idx]
                
                # 2. Copy-in: 将梯度拷入 Bucket
                # grad 可能是非连续的，view(-1) 可能会失败，使用 reshape 或 flatten 更安全，
                # 但通常 autograd 传出的 grad 是连续的。
                bucket["tensor"][offset : offset + numel].copy_(grad.detach().view(-1))
                
                bucket["ready_params"] += 1
                
                # 3. 触发通信
                if bucket["ready_params"] == bucket["total_params"]:
                    # 除以 world_size 做平均
                    bucket["tensor"].div_(world_size)
                    # 异步 AllReduce
                    bucket["handle"] = dist.all_reduce(
                        bucket["tensor"], op=dist.ReduceOp.SUM, async_op=True
                    )
                
                # 返回 grad 以保持 PyTorch 默认行为（虽然会被我们在 step 前覆盖）
                return grad
            return hook

        for param in self.module.parameters():
            if param.requires_grad:
                # 注册 hook
                param.register_hook(get_hook(param))

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        # 同步所有通信，并 Copy-out
        for bucket in self.buckets:
            if bucket["handle"]:
                bucket["handle"].wait()
            
            current_offset = 0
            for param in bucket["params"]:
                numel = param.numel()
                grad_data = bucket["tensor"][current_offset : current_offset + numel]
                
                # 此时 bucket 中的数据已经是 全局平均梯度
                # 将其写回 param.grad，供 optimizer 使用
                if param.grad is None:
                    # 显式 detach 避免建立计算图
                    param.grad = grad_data.view(param.shape).clone().detach()
                else:
                    param.grad.copy_(grad_data.view(param.shape))
                
                current_offset += numel

def get_ddp_bucketed(module: torch.nn.Module, bucket_size_mb: float) -> torch.nn.Module:
    """
    Returns a torch.nn.Module container that handles
    parameter broadcasting and gradient synchronization for
    distributed data parallel training.

    This container should overlaps communication with backprop computation
    by asynchronously communicating buckets of gradients as they are ready
    in the backward pass.

    Args:
        module: torch.nn.Module
            Underlying model to wrap with DDP.
        bucket_size_mb: The bucket size, in megabytes. If None, use a single
            bucket of unbounded size.
    Returns:
        Instance of a DDP class.
    """

    return DDPBucketed(module, bucket_size_mb)
    raise NotImplementedError


def ddp_bucketed_on_after_backward(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """
    Code to run after the backward pass is completed, but before we take
    an optimizer step.

    Args:
        ddp_model: torch.nn.Module
            DDP-wrapped model.
        optimizer: torch.optim.Optimizer
            Optimizer being used with the DDP-wrapped model.
    """
    # For example: ddp_model.finish_gradient_synchronization()
    
    """
    Backward 结束后，Optimizer Step 之前调用。
    必须在这里等待通信完成，并把梯度写回 param.grad。
    """
    if isinstance(ddp_model, DDPBucketed):
        # 修正：这里应该调用 finish，确保梯度就位
        ddp_model.finish_gradient_synchronization()
    return
    raise NotImplementedError


def ddp_bucketed_on_train_batch_start(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """
    Code to run at the very start of the training step.

    Args:
        ddp_model: torch.nn.Module
            DDP-wrapped model.
        optimizer: torch.optim.Optimizer
            Optimizer being used with the DDP-wrapped model.
    """

    """
    训练 Step 开始时调用。
    清理上一轮的状态，重置 Bucket。
    """
    if isinstance(ddp_model, DDPBucketed):
        # 修正：这里应该调用 prepare，清零 Bucket
        ddp_model._prepare_for_backward()
    return
    raise NotImplementedError

import torch
import torch.distributed as dist
from torch.distributed.optim import ZeroRedundancyOptimizer
from typing import Type, Iterable, Dict, Union

class _ShardedOptimizer(torch.optim.Optimizer):
    def __init__(self, params: Iterable, optimizer_cls: Type[torch.optim.Optimizer], **kwargs):
        # 1. 将输入参数统一转换为列表，以便进行索引和切分
        self.all_params = list(params)
        
        # 检查分布式环境是否初始化
        if not dist.is_initialized():
            raise RuntimeError("ShardedOptimizer requires torch.distributed to be initialized.")
        
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        
        # 2. 参数所有权划分 (Partitioning)
        # 计算分片大小，将参数列表均匀分配给各个 Rank
        total_params = len(self.all_params)
        chunk_size = (total_params + self.world_size - 1) // self.world_size
        
        self.my_params = []  # 当前 Rank 负责更新的参数
        self.param_owners = [] # 记录每个参数归哪个 Rank 管
        
        for i, param in enumerate(self.all_params):
            # 计算当前参数的 Owner Rank
            owner = i // chunk_size
            if owner >= self.world_size:
                owner = self.world_size - 1
            
            self.param_owners.append(owner)
            
            # 如果是我的参数，加入本地列表
            if owner == self.rank:
                self.my_params.append(param)
        
        # 3. 初始化本地优化器
        # 关键点：本地优化器只管理属于自己的那部分参数 (my_params)
        # 这样就实现了 Optimizer State 的内存分片
        self.local_optimizer = optimizer_cls(self.my_params, **kwargs)
        
        # 初始化父类 (主要为了满足类型检查和基本的 API 兼容)
        super().__init__(self.my_params, kwargs)

    def zero_grad(self, set_to_none: bool = False):
        # 必须对所有参数清零梯度，不仅仅是本地参数。
        # 因为 Backward 过程是利用全量参数计算的，所有参数都会产生梯度。
        for p in self.all_params:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    if p.grad.grad_fn is not None:
                        p.grad.detach_()
                    p.grad.zero_()

    def step(self, closure=None):
        # 1. 本地更新
        # 调用实际的优化器，只更新当前 Rank 负责的参数权重
        loss = self.local_optimizer.step(closure)
        
        # 2. 权重同步 (Synchronization)
        # 每个 Rank 的参数现在只有一部分是新的，其他部分是旧的。
        # 我们需要遍历所有参数，让 Owner 把最新的权重广播给所有人。
        # (注：在生产环境中通常使用 Bucket AllGather 来优化性能，这里为了逻辑清晰使用逐个 Broadcast)
        for i, param in enumerate(self.all_params):
            owner = self.param_owners[i]
            # 将 param.data 从 owner 广播到其他所有节点
            dist.broadcast(param.data, src=owner)
            
        return loss

def get_sharded_optimizer(params, optimizer_cls: Type[torch.optim.Optimizer], **kwargs) -> torch.optim.Optimizer:
    """
    Returns a torch.optim.Optimizer that handles optimizer state sharding
    of the given optimizer_cls on the provided parameters.

    Arguments:
        params (``Iterable``): an ``Iterable`` of :class:`torch.Tensor` s
            or :class:`dict` s giving all parameters, which will be sharded
            across ranks.
        optimizer_class (:class:`torch.nn.Optimizer`): the class of the local
            optimizer.
    Keyword arguments:
        kwargs: keyword arguments to be forwarded to the optimizer constructor.
    Returns:
        Instance of sharded optimizer.
    """

    # 使用 PyTorch 原生的 ZeroRedundancyOptimizer
    # 它会将参数组分配给不同的 rank，每个 rank 只负责更新一部分参数，
    # 并在 step() 后广播更新后的权重，从而保证所有 rank 的模型权重一致。
    # return ZeroRedundancyOptimizer(
    #     params,
    #     optimizer_class=optimizer_cls,
    #     **kwargs
    # )

    ## 自己手写ZeRO类
    return _ShardedOptimizer(params, optimizer_cls, **kwargs)

    raise NotImplementedError
