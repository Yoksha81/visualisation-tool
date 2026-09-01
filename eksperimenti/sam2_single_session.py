import torch
import torch.nn as nn

from sam2.build_sam import build_sam2

from model_visualizer import (
    ComponentSpec,
    OverviewNodeSpec,
    OverviewSpec,
    visualize,
)


class MaskDecoderWrapper(nn.Module):
    """
    SAM2-specific adapter ostaje samo u demo skripti.

    MaskDecoder high-resolution putanja očekuje već projektovane
    high-res feature-e. Wrapper uključuje conv_s0/conv_s1 u graf.
    """

    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(
        self,
        image_embeddings,
        image_pe,
        sparse_prompt_embeddings,
        dense_prompt_embeddings,
        high_res_s0,
        high_res_s1,
    ):

        high_res_features = [
            self.decoder.conv_s0(
                high_res_s0
            ),
            self.decoder.conv_s1(
                high_res_s1
            ),
        ]

        return self.decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_features,
        )


model = build_sam2(
    config_file="configs/sam2/sam2_hiera_t.yaml",
    ckpt_path=None,
    device="cpu",
    apply_postprocessing=False,
)

model.eval()


# ============================================================
# Example input-i za trace/export pojedinačnih komponenti
# ============================================================

# ImageEncoder
image = torch.zeros(
    1,
    3,
    1024,
    1024,
)

# MemoryEncoder
pix_feat = torch.zeros(
    1,
    256,
    64,
    64,
)

memory_masks = torch.zeros(
    1,
    1,
    1024,
    1024,
)

# MemoryAttention
curr = torch.zeros(
    4096,
    1,
    256,
)

memory = torch.zeros(
    4096,
    1,
    64,
)

curr_pos = torch.zeros(
    4096,
    1,
    256,
)

memory_pos = torch.zeros(
    4096,
    1,
    64,
)

# PromptEncoder
point_coords = torch.tensor(
    [
        [
            [256.0, 256.0],
            [768.0, 768.0],
        ]
    ],
    dtype=torch.float32,
)

point_labels = torch.tensor(
    [
        [1, 0]
    ],
    dtype=torch.int64,
)

points = (
    point_coords,
    point_labels,
)

boxes = torch.tensor(
    [
        [
            [128.0, 128.0],
            [896.0, 896.0],
        ]
    ],
    dtype=torch.float32,
)

prompt_masks = torch.zeros(
    1,
    1,
    256,
    256,
)

# MaskDecoder
image_embeddings = torch.zeros(
    1,
    256,
    64,
    64,
)

image_pe = torch.zeros(
    1,
    256,
    64,
    64,
)

sparse_prompt_embeddings = torch.zeros(
    1,
    4,
    256,
)

dense_prompt_embeddings = torch.zeros(
    1,
    256,
    64,
    64,
)

high_res_s0 = torch.zeros(
    1,
    256,
    256,
    256,
)

high_res_s1 = torch.zeros(
    1,
    256,
    128,
    128,
)


component_specs = {

    "image_encoder": ComponentSpec(
        example_inputs=(
            image,
        ),
        validate_inputs=False,
    ),

    "memory_encoder": ComponentSpec(
        example_inputs=(
            pix_feat,
            memory_masks,
        ),
        validate_inputs=False,
    ),

    "memory_attention": ComponentSpec(
        example_inputs=(
            curr,
            memory,
            curr_pos,
            memory_pos,
        ),
        validate_inputs=False,
    ),

    "sam_prompt_encoder": ComponentSpec(
        example_inputs=(
            points,
            boxes,
            prompt_masks,
        ),
        validate_inputs=False,
    ),

    "sam_mask_decoder": ComponentSpec(
        example_inputs=(
            image_embeddings,
            image_pe,
            sparse_prompt_embeddings,
            dense_prompt_embeddings,
            high_res_s0,
            high_res_s1,
        ),
        wrapper=MaskDecoderWrapper,
        validate_inputs=False,
    ),
}


# ============================================================
# SAM2 HIGH-LEVEL DATAFLOW
# ============================================================
#
# Ovo više NIJE containment pregled.
#
# Prikaz predstavlja jedan "unrolled" korak obrade video frame-a.
# Memory Bank je zato prikazan dva puta:
#
#   Memory Bank (previous frames)
#                 ...
#   Memory Bank (updated)
#
# Time recurrent/streaming arhitekturu predstavljamo kao DAG,
# što odgovara jednom vremenskom koraku i kompatibilno je sa
# postojećim VizTool rendererom.
#
# Klikabilni su samo stvarni nn.Module blokovi.
# Input Frame, Prompt, Memory Bank i Mask Output su runtime/data
# koncepti i zato se ne otvaraju.
# ============================================================

sam2_overview = OverviewSpec(

    title=(
        "SAM2 — high-level dataflow za jedan video frame\n"
        "Memory Bank je vremenski razmotan: prethodna memorija → "
        "trenutni frame → ažurirana memorija"
    ),

    nodes={

        "input_frame": OverviewNodeSpec(
            label="Input Frame",
        ),

        "memory_bank_prev": OverviewNodeSpec(
            label="Memory Bank\n(previous frames)",
        ),

        "prompt": OverviewNodeSpec(
            label="Prompt",
        ),

        "image_encoder": OverviewNodeSpec(
            label="Image Encoder",
            component_name="image_encoder",
        ),

        "memory_attention": OverviewNodeSpec(
            label="Memory Attention",
            component_name="memory_attention",
        ),

        "prompt_encoder": OverviewNodeSpec(
            label="Prompt Encoder",
            component_name="sam_prompt_encoder",
        ),

        "mask_decoder": OverviewNodeSpec(
            label="Mask Decoder",
            component_name="sam_mask_decoder",
        ),

        "mask_output": OverviewNodeSpec(
            label="Mask Output",
        ),

        "memory_encoder": OverviewNodeSpec(
            label="Memory Encoder",
            component_name="memory_encoder",
        ),

        "memory_bank_updated": OverviewNodeSpec(
            label="Memory Bank\n(updated)",
        ),
    },

    edges=[

        # Current visual frame.
        (
            "input_frame",
            "image_encoder",
        ),

        # Current image features conditioned by stored memories.
        (
            "image_encoder",
            "memory_attention",
        ),
        (
            "memory_bank_prev",
            "memory_attention",
        ),

        # Prompt branch.
        (
            "prompt",
            "prompt_encoder",
        ),

        # Main mask prediction path.
        (
            "memory_attention",
            "mask_decoder",
        ),
        (
            "prompt_encoder",
            "mask_decoder",
        ),

        # SAM2 high-resolution image features are also provided
        # directly to the mask decoder.
        (
            "image_encoder",
            "mask_decoder",
        ),

        (
            "mask_decoder",
            "mask_output",
        ),

        # New spatial memory is encoded from current image features
        # together with the predicted mask.
        (
            "image_encoder",
            "memory_encoder",
        ),
        (
            "mask_decoder",
            "memory_encoder",
        ),

        (
            "memory_encoder",
            "memory_bank_updated",
        ),
    ],
)


# ============================================================
# JEDAN POZIV -> JEDNA SESIJA
# ============================================================

visualize(
    model,
    component_specs=component_specs,
    overview_spec=sam2_overview,
)
