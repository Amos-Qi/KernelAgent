from typing import List
import torch
import triton
import triton.language as tl

USERS = 24
CAND = 1024
ROWS = USERS * CAND
TOTAL_W = 1322
ATTN_W = 32
DENSE_N = 130
DENSE_OUT = 3

LAYOUT = [
    ('installed_store_ids', 'attn', 0, 0, 32, 'cand', 0),
    ('ad_req_project_id', 'attn', 0, 0, 32, 'cand', 0),
    ('raw_uups_adrev_levelplay_v2', 'attn', 0, 0, 32, 'cand', 0),
    ('raw_uups_adrev_oecpm_interstitial_v2', 'attn', 0, 0, 32, 'cand', 0),
    ('raw_uups_adrev_oecpm_rewarded_v2', 'attn', 0, 0, 32, 'cand', 0),
    ('raw_uups_adrev_s2s_attributed_v2', 'attn', 0, 0, 32, 'cand', 0),
    ('raw_uups_adrev_s2s_unattributed_v2', 'attn', 0, 0, 32, 'cand', 0),
    ('raw_uups_purchase_attributed_v3', 'attn', 0, 0, 32, 'cand', 0),
    ('raw_uups_purchase_uasdk_v3', 'attn', 0, 0, 32, 'cand', 0),
    ('raw_uups_purchase_unattributed_v3', 'attn', 0, 0, 32, 'cand', 0),
    ('feature_store_gamer_levelplay_adrev', 'attn', 0, 0, 32, 'cand', 0),
    ('feature_store_gamer_mmp_s2s_adrev_attributed', 'attn', 0, 0, 32, 'cand', 0),
    ('feature_store_gamer_mmp_s2s_adrev_unattributed', 'attn', 0, 0, 32, 'cand', 0),
    ('ad_format', 'embed', 10, 0, 5, 'user', 0),
    ('audience_id', 'embed', 46000, 0, 29, 'cand', 0),
    ('gamer_id_scope', 'embed', 10, 0, 5, 'user', 0),
    ('geolocation_country', 'embed', 300, 0, 8, 'user', 0),
    ('platform', 'embed', 10, 0, 5, 'user', 0),
    ('publisher_developer_id', 'embed', 40000, 0, 28, 'user', 0),
    ('publisher_game_id', 'embed', 56000, 0, 30, 'user', 0),
    ('publisher_store_id', 'embed', 110000, 0, 40, 'user', 0),
    ('target_developer_id', 'embed', 40000, 0, 28, 'cand', 0),
    ('target_game_id', 'embed', 56000, 0, 30, 'cand', 0),
    ('target_store_id', 'embed', 110000, 0, 40, 'cand', 0),
    ('device_connection_type', 'embed', 10, 0, 5, 'user', 0),
    ('device_orientation', 'embed', 10, 0, 5, 'user', 0),
    ('device_type', 'embed', 67000, 0, 32, 'user', 0),
    ('apptopia_storeinfo_source_category_id', 'embed', 50, 0, 5, 'user', 0),
    ('apptopia_storeinfo_source_cross_store_app_id', 'embed', 250000, 0, 20, 'user', 0),
    ('apptopia_storeinfo_source_offers_in_app_purchases', 'embed', 10, 0, 5, 'user', 0),
    ('apptopia_storeinfo_source_subcategory_id', 'embed', 200, 0, 7, 'user', 0),
    ('apptopia_storeinfo_target_category_id', 'embed', 50, 0, 5, 'cand', 0),
    ('apptopia_storeinfo_target_cross_store_app_id', 'embed', 250000, 0, 20, 'cand', 0),
    ('apptopia_storeinfo_target_offers_in_app_purchases', 'embed', 10, 0, 5, 'cand', 0),
    ('apptopia_storeinfo_target_subcategory_id', 'embed', 200, 0, 7, 'cand', 0),
    ('device_atlas_device_vendor', 'embed', 5000, 0, 16, 'user', 0),
    ('device_atlas_hardware_classification', 'embed', 20, 0, 5, 'user', 0),
    ('device_atlas_highest_cellular_generation', 'embed', 20, 0, 5, 'user', 0),
    ('device_atlas_os_vendor', 'embed', 50, 0, 5, 'user', 0),
    ('sensor_tower_source_game_class', 'embed', 14, 0, 5, 'user', 0),
    ('sensor_tower_source_game_genre', 'embed', 27, 0, 5, 'user', 0),
    ('sensor_tower_source_game_subcategory', 'embed', 35, 0, 5, 'user', 0),
    ('sensor_tower_source_game_subgenre', 'embed', 150, 0, 6, 'user', 0),
    ('sensor_tower_source_game_theme', 'embed', 188, 0, 7, 'user', 0),
    ('sensor_tower_source_primary_category', 'embed', 85, 0, 6, 'user', 0),
    ('sensor_tower_target_game_class', 'embed', 14, 0, 5, 'cand', 0),
    ('sensor_tower_target_game_genre', 'embed', 27, 0, 5, 'cand', 0),
    ('sensor_tower_target_game_monetization_model', 'embed', 15, 0, 5, 'cand', 0),
    ('sensor_tower_target_game_subcategory', 'embed', 35, 0, 5, 'cand', 0),
    ('sensor_tower_target_game_subgenre', 'embed', 150, 0, 6, 'cand', 0),
    ('sensor_tower_target_game_theme', 'embed', 188, 0, 7, 'cand', 0),
    ('sensor_tower_target_primary_category', 'embed', 85, 0, 6, 'cand', 0),
    ('platform_os_version', 'embed', 400, 0, 8, 'user', 0),
    ('installed_store_ids_channel', 'bag', 15, 0, 5, 'user', 50),
    ('raw_uups_adrev_levelplay_v2_channel', 'bag', 15, 0, 5, 'user', 30),
    ('raw_uups_adrev_oecpm_interstitial_v2_channel', 'bag', 15, 0, 5, 'user', 10),
    ('raw_uups_adrev_oecpm_rewarded_v2_channel', 'bag', 15, 0, 5, 'user', 10),
    ('raw_uups_adrev_s2s_attributed_v2_channel', 'bag', 15, 0, 5, 'user', 10),
    ('raw_uups_adrev_s2s_unattributed_v2_channel', 'bag', 15, 0, 5, 'user', 30),
    ('raw_uups_purchase_attributed_v3_channel', 'bag', 15, 0, 5, 'user', 5),
    ('raw_uups_purchase_uasdk_v3_channel', 'bag', 15, 0, 5, 'user', 5),
    ('dense_features', 'dense', 130, 0, 390, 'cand', 0),
]

COL_OFF = []
_c = 0
for _e in LAYOUT:
    COL_OFF.append(_c)
    _c += _e[4]

N_ATTN = sum(1 for e in LAYOUT if e[1] == "attn")


@triton.jit
def _embed_gather_kernel(tbl_ptrs, idx_ptrs, widths, cols, is_user, expand_ptr, out_ptr,
                         M, W_TOT: tl.constexpr, BM: tl.constexpr, WMAX: tl.constexpr):
    f = tl.program_id(0)
    pid_m = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    m_mask = offs_m < M
    w = tl.load(widths + f)
    col = tl.load(cols + f)
    user = tl.load(is_user + f)
    tbl = tl.load(tbl_ptrs + f).to(tl.pointer_type(tl.bfloat16))
    idxp = tl.load(idx_ptrs + f).to(tl.pointer_type(tl.int64))
    rows = tl.where(user > 0, tl.load(expand_ptr + offs_m, mask=m_mask, other=0), offs_m.to(tl.int64))
    idx = tl.load(idxp + rows, mask=m_mask, other=0)
    ow = tl.arange(0, WMAX)
    w_mask = ow < w
    val = tl.load(tbl + idx[:, None] * w + ow[None, :], mask=m_mask[:, None] & w_mask[None, :], other=0.0)
    tl.store(out_ptr + offs_m[:, None] * W_TOT + col + ow[None, :], val,
             mask=m_mask[:, None] & w_mask[None, :])


@triton.jit
def _bag_mean_kernel(tbl_ptrs, val_ptrs, lens, widths, cols, expand_ptr, out_ptr,
                     M, W_TOT: tl.constexpr, BM: tl.constexpr, LMAX: tl.constexpr, WMAX: tl.constexpr):
    f = tl.program_id(0)
    pid_m = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    m_mask = offs_m < M
    L = tl.load(lens + f)
    w = tl.load(widths + f)
    col = tl.load(cols + f)
    tbl = tl.load(tbl_ptrs + f).to(tl.pointer_type(tl.bfloat16))
    vals = tl.load(val_ptrs + f).to(tl.pointer_type(tl.int64))
    u = tl.load(expand_ptr + offs_m, mask=m_mask, other=0)
    ow = tl.arange(0, WMAX)
    w_mask = ow < w
    acc = tl.zeros((BM, WMAX), dtype=tl.float32)
    cnt = tl.zeros((BM,), dtype=tl.float32)
    for l in range(0, LMAX):
        in_len = l < L
        ii = tl.load(vals + u * L + l, mask=m_mask & in_len, other=0)
        keep = (ii != 0) & in_len & m_mask
        row = tl.load(tbl + ii[:, None] * w + ow[None, :],
                      mask=keep[:, None] & w_mask[None, :], other=0.0).to(tl.float32)
        acc += row
        cnt += keep.to(tl.float32)
    denom = tl.where(cnt > 0, cnt, 1.0)
    mean = acc / denom[:, None]
    tl.store(out_ptr + offs_m[:, None] * W_TOT + col + ow[None, :],
             mean.to(out_ptr.dtype.element_ty), mask=m_mask[:, None] & w_mask[None, :])


@triton.jit
def _bf16_rtne(x):
    u = x.to(tl.uint32, bitcast=True)
    bias = 0x00007FFF + ((u >> 16) & 1)
    q = (u + bias) & 0xFFFF0000
    return q.to(tl.float32, bitcast=True)


@triton.jit
def _dense_proj_kernel(x_ptr, w_ptr, b_ptr, out_ptr, M,
                       COL: tl.constexpr, NF: tl.constexpr, NO: tl.constexpr,
                       W_TOT: tl.constexpr, BM: tl.constexpr, WPAD: tl.constexpr, NW: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_w = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    m_mask = offs_m < M
    ow = pid_w * WPAD + tl.arange(0, WPAD)
    w_mask = ow < NW
    fidx = ow // NO
    j = ow % NO
    x = tl.load(x_ptr + offs_m[:, None] * NF + fidx[None, :],
                mask=m_mask[:, None] & w_mask[None, :], other=0.0).to(tl.float32)
    wv = tl.load(w_ptr + fidx * NO + j, mask=w_mask, other=0.0).to(tl.float32)
    bv = tl.load(b_ptr + fidx * NO + j, mask=w_mask, other=0.0).to(tl.float32)
    y = _bf16_rtne(x * wv[None, :])
    y = (y + bv[None, :]).to(tl.bfloat16)
    tl.store(out_ptr + offs_m[:, None] * W_TOT + COL + ow[None, :], y,
             mask=m_mask[:, None] & w_mask[None, :])


@triton.jit
def _attn_copy_fused_kernel(src_ptrs, out_ptr, M,
                            W_TOT: tl.constexpr, AW: tl.constexpr, BM: tl.constexpr):
    f = tl.program_id(0)
    pid_m = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    m_mask = offs_m < M
    ow = tl.arange(0, AW)
    src = tl.load(src_ptrs + f).to(tl.pointer_type(tl.bfloat16))
    val = tl.load(src + offs_m[:, None] * AW + ow[None, :], mask=m_mask[:, None], other=0.0)
    col = f * AW
    tl.store(out_ptr + offs_m[:, None] * W_TOT + col + ow[None, :], val, mask=m_mask[:, None])


_meta_cache = {}


def _metas(tensors, device):
    key = tuple(t.data_ptr() for t in tensors)
    m = _meta_cache.get(key)
    if m is not None:
        return m
    it = iter(tensors)
    expand_idx = next(it)

    e_groups = {8: ([], [], [], [], []), 32: ([], [], [], [], []), 64: ([], [], [], [], [])}
    b_groups = {16: ([], [], [], [], []), 64: ([], [], [], [], [])}
    a_src = []
    dense = {}

    for (name, kind, t_rows, _, width, level, bag_len), col in zip(LAYOUT, COL_OFF):
        if kind == "attn":
            a_src.append(next(it).data_ptr())
        elif kind == "embed":
            idx, tbl = next(it), next(it)
            if width <= 8:
                g = e_groups[8]
            elif width <= 32:
                g = e_groups[32]
            else:
                g = e_groups[64]
            g[0].append(tbl.data_ptr())
            g[1].append(idx.data_ptr())
            g[2].append(width)
            g[3].append(col)
            g[4].append(1 if level == "user" else 0)
        elif kind == "bag":
            idx, tbl = next(it), next(it)
            if bag_len <= 16:
                g = b_groups[16]
            else:
                g = b_groups[64]
            g[0].append(tbl.data_ptr())
            g[1].append(idx.data_ptr())
            g[2].append(bag_len)
            g[3].append(width)
            g[4].append(col)
        elif kind == "dense":
            dense["x"], dense["w"], dense["b"] = next(it), next(it), next(it)
            dense["col"] = col

    i64 = lambda v: torch.tensor(v, dtype=torch.int64, device=device)

    e_group_metas = {}
    for wmax, (tbls, idxs, ws, cols, users) in e_groups.items():
        if tbls:
            e_group_metas[wmax] = (
                i64(tbls), i64(idxs), i64(ws), i64(cols), i64(users), len(tbls)
            )

    b_group_metas = {}
    for lmax, (tbls, vals, lens, ws, cols) in b_groups.items():
        if tbls:
            b_group_metas[lmax] = (
                i64(tbls), i64(vals), i64(lens), i64(ws), i64(cols), len(tbls)
            )

    m = {
        "expand": expand_idx,
        "e_groups": e_group_metas,
        "b_groups": b_group_metas,
        "a": (i64(a_src), len(a_src)),
        "d": dense,
    }
    _meta_cache[key] = m
    return m


def kernel_function(*tensors: torch.Tensor) -> torch.Tensor:
    device = tensors[1].device
    dtype = tensors[1].dtype
    m = _metas(tensors, device)
    out = torch.empty(ROWS, TOTAL_W, dtype=dtype, device=device)

    for wmax, (e_tbl, e_idx, e_w, e_col, e_user, n_e) in m["e_groups"].items():
        if n_e == 0:
            continue
        bm = 128 if wmax <= 8 else 64
        nw = 4 if wmax <= 8 else 2
        _embed_gather_kernel[(n_e, triton.cdiv(ROWS, bm))](
            e_tbl, e_idx, e_w, e_col, e_user, m["expand"], out, ROWS,
            W_TOT=TOTAL_W, BM=bm, WMAX=wmax, num_warps=nw)

    for lmax, (b_tbl, b_val, b_len, b_w, b_col, n_b) in m["b_groups"].items():
        if n_b == 0:
            continue
        _bag_mean_kernel[(n_b, triton.cdiv(ROWS, 128))](
            b_tbl, b_val, b_len, b_w, b_col, m["expand"], out, ROWS,
            W_TOT=TOTAL_W, BM=128, LMAX=lmax, WMAX=8, num_warps=4)

    d = m["d"]
    BM_D = 128
    WPAD_D = 32
    NW_D = DENSE_N * DENSE_OUT
    _dense_proj_kernel[(triton.cdiv(ROWS, BM_D), triton.cdiv(NW_D, WPAD_D))](
        d["x"], d["w"], d["b"], out, ROWS,
        COL=d["col"], NF=DENSE_N, NO=DENSE_OUT, W_TOT=TOTAL_W,
        BM=BM_D, WPAD=WPAD_D, NW=NW_D, num_warps=4)

    a_src, n_a = m["a"]
    _attn_copy_fused_kernel[(n_a, triton.cdiv(ROWS, 128))](
        a_src, out, ROWS, W_TOT=TOTAL_W, AW=ATTN_W, BM=128, num_warps=4)

    return out