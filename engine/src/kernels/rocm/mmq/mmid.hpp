// Parts from llama.cpp/ggml (MIT, The ggml authors) via flash-next-iq-gufo-extension 9ce08ce (MIT); see THIRD_PARTY.md.
#pragma once

void ggml_hip_launch_mm_ids_helper(
        const int32_t * ids, int32_t * ids_src1, int32_t * ids_dst, int32_t * expert_bounds,
        int n_experts, int n_tokens, int n_expert_used, int nchannels_y, int si1, int sis1, hipStream_t stream);
