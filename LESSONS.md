# Lessons Learned

## HeXO Game AI Optimization
*   **Move Pruning Issue:** The initial HeXO AI logic aggressively downsampled candidate moves (randomly sampling 50 out of 200+ legal moves) to combat high branching factors. This random sampling caused the AI to frequently miss critical blocking and winning moves since it couldn't reliably see its most important immediate options.
*   **Smart Selection:** Replacing random downsampling with localized pruning (restricting candidate moves to distance <= 2 from any existing stone) naturally reduced the branching factor down to ~20-30 highly relevant moves without sacrificing crucial local play analysis. This ensures the AI continues to build contiguous structures effectively.

## PyTorch Model Training & RTX Ada Optimization
*   **Architecture Choice:** For HeXO's hexagonal axial-coordinate grid, employing scalable 2-Dimensional Convolutional Networks over a standardized matrix (`21x21`) resolves the enormous CPU latency historically created by Graph node instantiations. Models infer lightning fast globally instead of dragging.
*   **Ada Lovelace Hardware Optimization:** To make the network properly ingest training data efficiently on an RTX 2000 Ada equivalent platform, two mechanisms were needed: `torch.autocast('cuda')` together with `torch.cuda.amp.GradScaler()` to shrink the footprint using Automatic Mixed Precision (BFloat16/Float16), alongside setting `torch.set_float32_matmul_precision('high')`. This forcefully engages the TF32 tensor cores which are heavily under-utilized on consumer/laptop 4000-series cards if left standard.
